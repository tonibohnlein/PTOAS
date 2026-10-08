# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare independent rotating-bank effects with emitted command order."""
import json
from pathlib import Path
import re
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native
from check_repeated_region import run


def check(document, trips, inner_trips, banks, stride=1, offset=0):
    """Construct the full occurrence conflict graph without regional summaries."""
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    events = trace["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    expected, effects = [], []
    for visit in range(trips):
        left = ("left", (stride * visit + offset) % banks[0])
        right = ("right", visit % banks[1])
        expected.extend([(0, [visit]), (1, [visit]), (2, [visit])])
        effects.extend([({("mat", 0)}, {left}), ({("mat", 0)}, {right}),
                        ({left, right}, {("acc", 0)})])
        for inner in range(inner_trips):
            expected.append((3, [visit, inner]))
            effects.append(({left, right}, {("acc", 0)}))
    assert [(p["type"], p["coordinates"]) for p in payloads] == expected, document
    pipes = [p["pipe"] for p in payloads]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            next_reads, next_writes = effects[b]
            if writes & (next_reads | next_writes) or reads & next_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(effects), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads), "unexpected interior ALL-BARRIER"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], document


def variant(source, left_banks, right_banks, stride=1, offset=0):
    return (source.replace("count=3", f"count={left_banks}")
            .replace("count=5", f"count={right_banks}")
            .replace("%three = arith.constant 3", f"%three = arith.constant {left_banks}")
            .replace("%five = arith.constant 5", f"%five = arith.constant {right_banks}")
            .replace("%stride = arith.constant 1", f"%stride = arith.constant {stride}")
            .replace("%offset = arith.constant 0", f"%offset = arith.constant {offset}"))


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="rotating-region-") as scratch:
        path = Path(scratch) / "case.pto"
        for banks, stride, offset, counts in (
                ((3, 5), 1, 0, ((0, 0), (0, 3), (1, 0), (2, 1), (4, 0), (6, 1), (16, 3))),
                ((6, 5), 2, 1, ((1, 0), (4, 1), (7, 3))),
                ((3, 5), 0, 2, ((0, 3), (2, 0), (6, 1)))):
            program = variant(source, *banks, stride, offset)
            for trips, inner in counts:
                evaluated = program.replace("array<i64: 2, 3>", f"array<i64: {trips}, {inner}>")
                if stride > 1:
                    # A finite bound proves that the machine integer multiply
                    # cannot wrap; no unproved affine interpretation of overflow.
                    evaluated = evaluated.replace(
                        "    scf.for %visit", f"    %limit = arith.constant {trips} : index\n    scf.for %visit")
                    evaluated = evaluated.replace("%visit = %zero to %n", "%visit = %zero to %limit")
                path.write_text(evaluated)
                print(f"rotating trace: banks={banks}, stride={stride}, offset={offset}, trips={trips}, inner={inner}",
                      flush=True)
                check(json.loads(run(tool, "--structured-trace", path)), trips, inner, banks, stride, offset)
                checked += 1
        # Larger coprime banks stay within the IR's per-buffer capacity. The
        # direct constructor checks additionally establish route/size behavior;
        # this test checks that trip values do not specialize emitted code.
        program = variant(source, 13, 11)
        path.write_text(program)
        report = json.loads(run(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"] and report["numeric_visits"] == 0, report
        small = run(tool, "--insert-logical", path)
        path.write_text(program.replace("array<i64: 2, 3>", "array<i64: 1000000000, 1000000001>"))
        large = run(tool, "--insert-logical", path)
        assert small == large.replace("1000000000, 1000000001", "2, 3")
        assert len(re.findall(r"(?m)^\s*scf\.for\b", small)) == 2
    print(f"rotating regions: {checked} independent command closures and compact trip counts passed")


if __name__ == "__main__":
    main()
