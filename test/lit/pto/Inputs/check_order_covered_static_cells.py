# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check partial physical reuse across distinct allocation SSA roots."""
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


def origin(visit):
    bits = (visit * 64) % (1 << 64)
    return max(bits - (1 << 64) if bits >= 1 << 63 else bits, 0)


def check(report, lower, upper, step):
    assert report["accepted"] and not report["trace"]["error"], report
    # A and C have distinct allocation SSA roots at the same physical address;
    # B overlaps half of each. The two tadds therefore have physical RMW modes.
    a, b = (0, 64), (32, 96)
    accesses = [([], [a]), ([a], [b]), ([b], [a]), ([a], [])]
    events = report["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    visits = list(range(lower, upper, step))
    assert [event["coordinates"] for event in payloads] == [[iv] for iv in visits for _ in accesses]
    pipes = [event["pipe"] for event in payloads]
    assert pipes == [pipe for _ in visits for pipe in (4, 1, 1, 5)]
    edges = native(pipes)
    for source in range(len(payloads)):
        reads, writes = accesses[source % 4]
        for target in range(source + 1, len(payloads)):
            next_reads, next_writes = accesses[target % 4]
            hazard = any(overlap(x, y) for x in writes for y in next_reads + next_writes)
            hazard |= any(overlap(x, y) for x in reads for y in next_writes)
            if source % 4 == 3 and target % 4 == 3:
                hazard |= abs(origin(visits[source // 4]) - origin(visits[target // 4])) < 16
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
    source = Path(fixture).read_text().split("  func.func @dynamic_local(")[0] + "}\n"
    cases = [(-41, -41, 20), (-41, -42, 20), (-41, -40, 20), (-41, 0, 20),
             (-1, 40, 20), (3, 69, 20), (0, 1, 20), (0, 120, 20),
             (-(1 << 31), (1 << 63) - (1 << 31), 1 << 62)]
    with tempfile.TemporaryDirectory(prefix="order-covered-static-cells-") as directory:
        path = Path(directory) / "input.pto"
        for lower, upper, step in cases:
            attributes = (f"test.trace_arguments = array<i64: {upper}, {lower}>, "
                          "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>")
            text = source.replace("test.order_covered_periodic, test.order_covered_static_cells", attributes)
            text = text.replace("arith.constant 20 : index", f"arith.constant {step} : index")
            path.write_text(text)
            run = subprocess.run([tool, "--gm-alias=may-not-alias", "--structured-trace", str(path)],
                                 capture_output=True, text=True, timeout=60)
            if run.returncode:
                raise RuntimeError(f"static cell domain {lower, upper, step}: {run.stderr}")
            check(json.loads(run.stdout), lower, upper, step)
    print(f"static physical cells: {len(cases)} independent partial-overlap/RMW command and physical traces passed")


if __name__ == "__main__":
    main()
