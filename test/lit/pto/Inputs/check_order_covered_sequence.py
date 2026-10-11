# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check original root crossings around a retained periodic child independently."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_allocation import physical_check
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def overlap(first, second):
    return max(first[0], second[0]) < min(first[1], second[1])


def check(report, lower, upper, step):
    assert report["accepted"] and not report["trace"]["error"], report
    # Prefix writes A; the loop reads A and reuses B; the suffix reads B.
    a, b = (0, 32), (32, 64)
    body = [([], [b]), ([a, b], [b]), ([b], [])]
    events = report["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    visits = list(range(lower, upper, step))
    accesses = [([], [a])] + body * len(visits) + [([b], [])]
    assert [event["coordinates"] for event in payloads] == [[]] + [[iv] for iv in visits for _ in body] + [[]]
    pipes = [event["pipe"] for event in payloads]
    assert pipes == [4] + [pipe for _ in visits for pipe in (4, 1, 5)] + [5]
    edges = native(pipes)
    for source in range(len(payloads)):
        reads, writes = accesses[source]
        for target in range(source + 1, len(payloads)):
            next_reads, next_writes = accesses[target]
            hazard = any(overlap(x, y) for x in writes for y in next_reads + next_writes)
            hazard |= any(overlap(x, y) for x in reads for y in next_writes)
            if (0 < source < len(payloads)-1 and 0 < target < len(payloads)-1
                    and (source-1)%3 == 2 and (target-1)%3 == 2):
                hazard |= abs(max(visits[(source-1)//3], 0) - max(visits[(target-1)//3], 0)) < 8
            if hazard:
                edges.add((2 * source + 1, 2 * target))
    _, covers = closure(2 * len(payloads), edges)
    for source, target in covers:
        if source % 2 and not target % 2 and pipes[source // 2] == pipes[target // 2]:
            edges.update((2 * previous + 1, target) for previous in range(target // 2)
                         if pipes[previous] == pipes[target // 2])
    expected = closure(2 * len(payloads), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads)
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert [row & ~(1 << i) for i, row in enumerate(actual)] == \
        [row & ~(1 << i) for i, row in enumerate(expected)], (lower, upper, step)
    physical_check(report, set(range(6)))


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().split("  func.func @sibling_conflict(")[0] + "}\n"
    cases = [(-1, -1, 20), (-1, -2, 20), (-1, 0, 20), (-1, 40, 20), (3, 24, 7), (0, 5, 1)]
    with tempfile.TemporaryDirectory(prefix="order-covered-sequence-") as temporary:
        path = Path(temporary) / "input.pto"
        for lower, upper, step in cases:
            text = source.replace("array<i64: 40, -1>", f"array<i64: {upper}, {lower}>")
            text = text.replace("arith.constant 20 : index", f"arith.constant {step} : index")
            path.write_text(text)
            run = subprocess.run([tool, "--gm-alias=may-not-alias", "--structured-trace", str(path)],
                                 capture_output=True, text=True, timeout=30)
            if run.returncode:
                raise RuntimeError(run.stderr)
            check(json.loads(run.stdout), lower, upper, step)
    print("six root sequence prefix/suffix command closure and physical pairing domains passed")


if __name__ == "__main__":
    main()
