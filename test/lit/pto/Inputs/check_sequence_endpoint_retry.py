# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare scalar-copy command graphs with original-coordinate byte hazards."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def scalar_edges(pipes):
    previous = None
    edges = set()
    for position, pipe in enumerate(pipes):
        if pipe != 0:
            continue
        if previous is not None:
            edges.add((2 * previous + 1, 2 * position))
        previous = position
    return edges


def validate(document, upper, outer_step, inner_step):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    full = set(range(16))
    expected = [("before", [], full, full)]
    for outer in range(1, upper, outer_step):
        for inner in range(0, 4, inner_step):
            expected.append(("read", [outer, inner], {inner}, set()))
            expected.append(("write", [outer, inner], set(), {4 * outer + inner}))
    expected.append(("after", [], full, full))
    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
    assert [(e["label"], e["coordinates"]) for e in payloads] == [(e[0], e[1]) for e in expected]
    pipes = [e["pipe"] for e in payloads]
    assert pipes == [1] + [0] * (len(expected) - 2) + [1]
    edges = native(pipes) | scalar_edges(pipes)
    for first, (_, _, reads, writes) in enumerate(expected):
        for last in range(first + 1, len(expected)):
            later_reads, later_writes = expected[last][2:]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * first + 1, 2 * last))
    required, _ = closure(2 * len(expected), edges)
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload":
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    # The generic command interpreter models asynchronous pipes. Add the
    # target's synchronous scalar-pipe order to its projected payload graph.
    for first, last in scalar_edges(pipes):
        actual[first] |= 1 << last
    for middle in reversed(range(len(actual))):
        for first in range(middle):
            if actual[first] & (1 << middle):
                actual[first] |= actual[middle]
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], "scalar-copy order changed"


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    cases = [(1, 1, 1), (2, 1, 1), (4, 1, 1), (4, 2, 1), (4, 1, 2), (4, 2, 2)]
    with tempfile.TemporaryDirectory(prefix="sequence-endpoint-retry-") as directory:
        path = Path(directory) / "case.pto"
        for upper, outer_step, inner_step in cases:
            variant = source.replace("    %four =", f"    %upper = arith.constant {upper} : index\n"
                                     f"    %outer_step = arith.constant {outer_step} : index\n"
                                     f"    %inner_step = arith.constant {inner_step} : index\n    %four =")
            variant = variant.replace("%one to %four step %one", "%one to %upper step %outer_step")
            variant = variant.replace("%zero to %four step %one", "%zero to %four step %inner_step")
            path.write_text(variant)
            result = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                    text=True, check=False, timeout=60)
            assert result.returncode == 0, result.stderr
            validate(json.loads(result.stdout), upper, outer_step, inner_step)
    print("sequence endpoint retry: six original-coordinate command closures passed")


if __name__ == "__main__":
    main()
