# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare signed-domain periodic endpoints with independent concrete conflicts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_allocation import physical_check
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def signed64(value):
    value %= 1 << 64
    return value - (1 << 64) if value >= 1 << 63 else value


def check(report, lower, upper, step):
    assert report["accepted"] and not report["trace"]["error"], report
    events = report["trace"]["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    visits = list(range(lower, upper, step))
    assert [event["coordinates"] for event in payloads] == [[iv] for iv in visits for _ in range(2)]
    pipes = [event["pipe"] for event in payloads]
    assert all(pipes[i] == (4 if i % 2 == 0 else 5) for i in range(len(pipes)))
    edges = native(pipes)
    origins = [max(signed64(iv * 256), 0) for iv in visits]
    for source in range(len(payloads)):
        for target in range(source + 1, len(payloads)):
            # Every load writes the same UB tile; every store reads that tile.
            conflict = source % 2 == 0 or target % 2 == 0
            if source % 2 and target % 2:
                # GM read and write bases are distinct under MayNotAlias.
                # Store rows can nevertheless overlap across arbitrary IV wrap.
                difference = origins[source // 2] - origins[target // 2]
                conflict = any(abs(difference + row * 5120) < 256 for row in range(-127, 128))
            if conflict:
                edges.add((2 * source + 1, 2 * target))
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
    source = Path(fixture).read_text().split("  func.func @strong(")[0] + "}\n"
    cases = [(-41, -41, 20), (-41, -42, 20), (-41, -40, 20), (-41, 0, 20),
             (-1, 40, 20), (3, 69, 20), (0, 1, 20), (0, 120, 20),
             (-(1 << 31), (1 << 63) - (1 << 31), 1 << 62)]
    with tempfile.TemporaryDirectory(prefix="order-covered-periodic-") as directory:
        path = Path(directory) / "input.pto"
        for lower, upper, step in cases:
            attributes = (f"test.trace_arguments = array<i64: {upper}, {lower}, 0>, "
                          "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>")
            text = source.replace("test.order_covered_periodic", attributes)
            text = text.replace("arith.constant 20 : index", f"arith.constant {step} : index")
            path.write_text(text)
            run = subprocess.run([tool, "--gm-alias=may-not-alias", "--structured-trace", str(path)],
                                 capture_output=True, text=True, timeout=60)
            if run.returncode:
                raise RuntimeError(f"order-covered domain {lower, upper, step}: {run.stderr}")
            check(json.loads(run.stdout), lower, upper, step)
    print(f"order-covered periodic: {len(cases)} exact command and physical pairing traces passed")


if __name__ == "__main__":
    main()
