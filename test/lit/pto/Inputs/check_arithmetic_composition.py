# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare arithmetic-child composition against authored concrete byte conflicts."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False)
    require(result.returncode == 0, result.stderr + result.stdout)
    return result.stdout


def expected_payloads(name, n):
    whole = set(range(64))
    result = [("bulk-in", [], whole, whole)]

    def scalar(label, coordinates, element, write):
        footprint = set(range(4 * element, 4 * element + 4))
        result.append((label, coordinates, set() if write else footprint, footprint if write else set()))

    for row in (range(1, 3) if name == "prefix_owned" else range(2)):
        if name == "recurrence":
            scalar("init", [row], 4 * row, True)
        lower = 1 if name == "recurrence" else 0
        upper = min(n, 2 if name in {"persistent", "overlapping_widths"} else 4)
        for column in range(lower, max(lower, upper)):
            read = 4 * row + column - 1 if name == "recurrence" else column
            write = column if name == "persistent" else 4 * row + column
            scalar("read", [row, column], read, False)
            if name == "overlapping_widths":
                result.append(("write", [row, column], set(), {4 * column + 2, 4 * column + 3}))
            else:
                scalar("write", [row, column], write, True)
    result.append(("bulk-out", [], whole, whole))
    return result


def validate(document, name, n):
    require(document["accepted"], (name, n, document))
    trace = document["trace"]
    require(not trace["error"], (name, n, trace["error"]))
    expected = expected_payloads(name, n)
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    require([(x["label"], x["coordinates"]) for x in payloads] == [x[:2] for x in expected],
            (name, n, "payloads or iteration identities changed", payloads, expected))
    pipes = [event["pipe"] for event in payloads]
    edges = native(pipes)
    for a, (_, _, reads, writes) in enumerate(expected):
        for b in range(a + 1, len(expected)):
            later_reads, later_writes = expected[b][2:]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    # The implementation's stated local policy places a barrier immediately
    # before each same-pipe cover's consumer, including nonadjacent covers.
    _, covers = closure(2 * len(payloads), edges)
    for source, target in covers:
        if source % 2 and target % 2 == 0 and pipes[source // 2] == pipes[target // 2]:
            edges.update((2 * previous + 1, target) for previous in range(target // 2)
                         if pipes[previous] == pipes[target // 2])
    required = closure(2 * len(payloads), edges)[0]
    commands = []
    gap = 0
    drains = 0
    for event in trace["events"]:
        if event["kind"] == "payload":
            gap += 1
            continue
        require(event["gap"] == gap, "command moved to the wrong original cut")
        if event["kind"] == "barrier" and event["pipe"] == 6:
            require(gap == len(payloads), "arithmetic child drained the enclosing invocation")
            drains += 1
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    require(drains == 1, "expected one invocation completion barrier")
    actual = closure_with_commands(pipes, commands)
    expected_closure = [row & ~(1 << i) for i, row in enumerate(required)]
    require(actual == expected_closure, (name, n, "emitted order mismatch", actual, expected_closure))


def main():
    tool = shutil.which(sys.argv[1])
    require(tool is not None, "test executable unavailable")
    source = Path(sys.argv[2]).read_text()
    prefix = source[:source.index("module attributes")]
    functions = re.findall(r"  func.func @(\w+).*?(?=\n  func.func|\n}\n)", source, re.S)
    blocks = re.findall(r"  func.func @\w+.*?(?=\n  func.func|\n}\n)", source, re.S)
    require(functions == ["recurrence", "prefix_owned", "persistent", "overlapping_widths"],
            "fixture functions changed")
    checked = 0
    # Only small text fixtures live in tmpfs. Generated IR stays in subprocess
    # memory and the billion-trip case is never dynamically executed.
    with tempfile.TemporaryDirectory(prefix="arithmetic-composition-") as scratch:
        path = Path(scratch) / "case.pto"
        for name, function in zip(functions, blocks):
            program = prefix + 'module attributes {pto.target_arch = "a3"} {\n' + function + '\n}\n'
            for n in (-1, 0, 1, 2, 5):
                path.write_text(program.replace("array<i64: 3>", f"array<i64: {n}>"))
                validate(json.loads(invoke(tool, "--structured-trace", path)), name, n)
                checked += 1
            path.write_text(program)
            analysis = json.loads(invoke(tool, "--sequence-analysis", path))
            require(not analysis["error"] and analysis["prepared"], (name, "regional preparation failed", analysis))
            require(analysis["arithmetic_regions"] > 0, (name, "arithmetic regional route was not used", analysis))
            require(analysis["boundary_bytes"] > 0, (name, "missing materialized boundary byte count", analysis))
            scalar_sites = function.count("pto.tgetval") + function.count("pto.tsetval")
            require(analysis["crossings"] <= 2 * scalar_sites + 1,
                    (name, "boundary handoffs grew with per-byte selector alternatives", analysis))
            print(name, {key: analysis[key] for key in
                         ("arithmetic_regions", "boundary_bytes", "cells", "ports", "expressions", "emitted",
                          "child_preparation", "crossing_preparation")},
                  flush=True)
            small = invoke(tool, "--insert-logical", path)
            require(small.count("scf.for") == function.count("scf.for"), "analysis unfolded a nested loop")
            for operation in ("pto.tadds", "pto.tgetval", "pto.tsetval"):
                require(small.count(operation) == function.count(operation), "payload was cloned or removed")
            path.write_text(program.replace("array<i64: 3>", "array<i64: 1000000000>"))
            large = invoke(tool, "--insert-logical", path)
            require(large.replace("1000000000", "3") == small, "insertion depends on dynamic trip count")
    print(f"arithmetic composition: {checked} independent byte-conflict command closures passed")


if __name__ == "__main__":
    main()
