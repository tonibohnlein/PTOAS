# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unfold only in the oracle; compare the compact boundary plan with byte conflicts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native
from check_finite_allocation import physical_check


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=60, check=True)
    return result.stdout


def validate(document, trips, inclusive):
    assert document["accepted"], document
    assert not document["trace"]["error"], document
    effects = [(set(), {0})]
    for i in range(max(0, trips)):
        if i + 1 < trips or inclusive:
            effects.append((set(), {(i + 1) % 2}))
        effects.append(({i % 2}, set()))
    events = document["trace"]["events"]
    payloads = [e for e in events if e["kind"] == "payload"]
    assert len(payloads) == len(effects), document
    pipes = [e["pipe"] for e in payloads]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(effects), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (trips, actual, required)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="boundary-rotation-") as directory:
        path = Path(directory) / "case.pto"
        for predicate in ("ult", "slt", "ule", "sle", "uge", "sge", "ugt", "sgt"):
            variant = source.replace("cmpi ult", "cmpi " + predicate)
            if predicate in ("uge", "sge", "ugt", "sgt"):
                variant = variant.replace("scf.if %more {", "scf.if %more {\n      } else {")
            inclusive = predicate in ("ule", "sle", "ugt", "sgt")
            for trips in (-2, 0, 1, 2, 3, 5, 8):
                path.write_text(variant.replace("array<i64: 5>", f"array<i64: {trips}>"))
                validate(json.loads(invoke(tool, "--structured-trace", path)), trips, inclusive)
                checked += 1
                allocated = path.read_text().replace("test.trace_arguments =",
                    "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
                path.write_text(allocated)
                physical = json.loads(invoke(tool, "--structured-trace", path))
                validate(physical, trips, inclusive)
                physical_check(physical, set(range(6)))
        # Runtime counts alter neither emitted payloads nor the analysis graph.
        sizes = []
        for trips in (5, 1000000000):
            path.write_text(source.replace("array<i64: 5>", f"array<i64: {trips}>"))
            text = invoke(tool, "--insert-logical", path)
            assert text.count("scf.for") == 1
            assert text.count("pto.texpands") == 2
            sizes.append(len(text.splitlines()))
        assert sizes[0] == sizes[1], sizes
        # Compact bank counts must not be expanded into billion-entry summaries.
        huge = source.replace("%two = arith.constant 2 : index",
                              "%two = arith.constant 1000000000 : index")
        path.write_text(huge)
        rejected = json.loads(invoke(tool, "--sequence-analysis", path))
        assert rejected["error"] and rejected["unchanged"], rejected
        # Iteration-varying participation cannot silently use a periodic slice.
        invalid = source.replace("%more = arith.cmpi ult, %next, %n : index",
            "%odd = arith.remui %i, %two : index\n      %more = arith.cmpi eq, %odd, %zero : index")
        path.write_text(invalid)
        rejected = json.loads(invoke(tool, "--sequence-analysis", path))
        assert rejected["error"], rejected
    print(f"boundary rotation: {checked} unfolded physical-order checks and compact emission passed")


if __name__ == "__main__":
    main()
