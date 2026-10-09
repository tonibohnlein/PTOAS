# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check mixed owned/arithmetic regional commands against original effects."""
import json
from pathlib import Path
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native
from check_repeated_region import run


def check(document, trips, length, hole):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    events = trace["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    expected, effects = [], []
    for visit in range(trips):
        for step in range(length):
            expected.extend([(0, [visit, step]), (1, [visit, step])])
            effects.extend([({("mat", 0)}, {("left", step % 2)}),
                            ({("left", step % 2), ("right", 0)}, {("acc", 0)})])
        expected.append((2, [visit]))
        effects.append((set(), {("gm", 16 * visit)}))
    for visit in range(trips):
        expected.append((3, [visit]))
        effects.append(({("gm", 16 * visit + hole + i) for i in range(8)}, {("ub", 0)}))
    for visit in range(trips):
        expected.append((4, [visit]))
        effects.append((set(), {("gm", 16 * visit)}))
    assert [(p["type"], p["coordinates"]) for p in payloads] == expected, document
    pipes = [p["pipe"] for p in payloads]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            if pipes[a] == 0 and pipes[b] == 0:
                continue  # The shared scalar hardware rule protects these pairs.
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
    assert closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)], document


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="mixed-symbolic-regions-") as scratch:
        path = Path(scratch) / "case.pto"
        for hole in (0, 8):
            program = source
            if hole:
                program = program.replace("%source = arith.muli", "%raw_source = arith.muli")
                program = program.replace("      %part =", "      %source = arith.addi %raw_source, %eight "
                                          "overflow<nsw> : index\n      %part =")
            for trips, length in ((0, 3), (1, 3), (2, 3), (3, 3)):
                path.write_text(program.replace("array<i64: 2, 3>", f"array<i64: {trips}, {length}>"))
                print(f"mixed symbolic trace: trips={trips}, inner={length}, hole={hole}", flush=True)
                check(json.loads(run(tool, "--structured-trace", path)), trips, length, hole)
                checked += 1
        path.write_text(source)
        report = json.loads(run(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"], report
        small = run(tool, "--insert-logical", path)
        path.write_text(source.replace("array<i64: 2, 3>", "array<i64: 1000000, 1000001>"))
        large = run(tool, "--insert-logical", path)
        assert small == large.replace("1000000, 1000001", "2, 3")
    print(f"mixed symbolic regions: {checked} independent command closures and compact parameters passed")


if __name__ == "__main__":
    main()
