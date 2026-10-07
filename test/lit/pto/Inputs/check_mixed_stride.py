# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check original mixed-stride payloads, endpoint phase/truncation, and global physical reuse."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_finite_allocation import physical_check
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def validate(report, slots, lower, step, upper, reverse):
    assert report["accepted"], report
    trace = report["trace"]
    assert not trace["error"], trace
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    visits = list(range(lower, upper, step))
    assert len(payloads) == 2 * len(visits), trace
    edges = native([event["pipe"] for event in payloads])
    effects = []
    for i in visits:
        writer, reader = (0, i % slots) if reverse else (i % slots, 0)
        effects.extend([({("mat", 0)}, {("left", writer)}),
                        ({("left", reader), ("right", 0)}, {("acc", 0)})])
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2*a + 1, 2*b))
    required, covers = closure(2 * len(payloads), edges)
    commands = []
    for event in trace["events"]:
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
    pipes = [event["pipe"] for event in payloads]
    # First check the original all-conflict reduction, before applying the
    # insertion policy for local covers. Cross-pipe endpoints remain exact.
    publications = {}
    pairs, local_targets = set(), set()
    for command in commands:
        pipe, gap = command["pipe"], command["gap"]
        if command["kind"] == "set":
            publications[command["identity"]] = max(i for i in range(gap) if pipes[i] == pipe)
        else:
            consumer = next(i for i in range(gap, len(pipes)) if pipes[i] == pipe)
            if command["kind"] == "wait":
                pairs.add((publications.pop(command["identity"]), consumer))
            else:
                local_targets.add(consumer)
    assert not publications
    assert pairs == {(a // 2, b // 2) for a, b in covers if pipes[a // 2] != pipes[b // 2]}
    assert local_targets == {b // 2 for a, b in covers if pipes[a // 2] == pipes[b // 2]}
    # A barrier immediately before a local cover's consumer also waits for
    # the preceding payload on that pipe. Apply this only AFTER reduction;
    # applying it to raw conflicts would invent barriers for removed edges.
    for source, target in covers:
        a, b = source // 2, target // 2
        if pipes[a] == pipes[b]:
            previous = max(i for i in range(b) if pipes[i] == pipes[b])
            edges.add((2 * previous + 1, target))
    expected = closure(2 * len(payloads), edges)[0]
    actual = closure_with_commands(pipes, commands)
    strict_required = [row & ~(1 << i) for i, row in enumerate(required)]
    assert all((want & got) == want for want, got in zip(strict_required, actual))
    assert actual == [row & ~(1 << i) for i, row in enumerate(expected)], (report, actual, expected)
    physical_check(report, set(range(6)))


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    checked = 0
    with tempfile.TemporaryDirectory(prefix="mixed-stride-") as temporary:
        path = Path(temporary) / "case.pto"
        for slots, lower, step in ((2, 0, 1), (3, 0, 1), (3, 1, 2)):
            for reverse in (False, True):
                for trips in (0, 1, 2, 4, 7):
                    upper = lower + trips * step
                    text = source.replace("count=2>", f"count={slots}>")
                    text = text.replace("%banks = arith.constant 2", f"%banks = arith.constant {slots}")
                    text = text.replace("%lower = arith.constant 0", f"%lower = arith.constant {lower}")
                    text = text.replace("%step = arith.constant 1", f"%step = arith.constant {step}")
                    text = text.replace("array<i64: 7>", f"array<i64: {upper}>")
                    if reverse:
                        text = text.replace("%ring[%slot]", "%ring[%temporary]")
                        text = text.replace("%ring[%zero]", "%ring[%slot]")
                        text = text.replace("%ring[%temporary]", "%ring[%zero]")
                    path.write_text(text)
                    result = subprocess.run([tool, "--structured-trace", str(path)], check=True,
                                            capture_output=True, text=True, timeout=90)
                    validate(json.loads(result.stdout), slots, lower, step, upper, reverse)
                    checked += 1
    print(f"{checked} mixed-stride original-cut logical and physical traces passed")


if __name__ == "__main__":
    main()
