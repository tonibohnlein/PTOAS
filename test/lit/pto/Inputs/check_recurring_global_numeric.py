# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare recurring original signed-machine GM bytes with emitted command closure."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, mode, path, policy):
    run = subprocess.run([tool, f"--gm-alias={policy}", mode, str(path)], capture_output=True,
                         text=True, timeout=90, check=False)
    assert run.returncode == 0, run.stderr + run.stdout
    return run


def signed64(value):
    value &= (1 << 64) - 1
    return value - (1 << 64) if value >= 1 << 63 else value


def check_trace(report, trips, seed):
    assert report["allocated"] and not report["logical"]["error"] and not report["physical"]["error"]
    trace = report["logical"]
    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
    assert len(payloads) == 4 * trips
    pipes = [e["pipe"] for e in payloads]
    accesses = []
    for i, event in enumerate(payloads):
        assert (event["type"], event["ordinal"]) == (i % 4, i // 4)
        # Evaluate the original two machine operations before the mathematical
        # row-stride map. The byte address itself is not silently wrapped.
        origin = signed64(seed * 8)
        row = signed64(origin + (i % 4 // 2) * 8)
        gm = {("GM", row * 32 + byte) for byte in range(32)}
        ub = {("UB", byte + (32 if i % 2 else 0)) for byte in range(32)}
        accesses.append((gm, ub) if i % 2 == 0 else (ub, gm))
    edges = native(pipes)
    for j, (read_j, write_j) in enumerate(accesses):
        for i in range(j):
            read_i, write_i = accesses[i]
            if write_i & (read_j | write_j) or read_i & write_j:
                edges.add((2 * i + 1, 2 * j))
    expected = closure(2 * len(payloads), edges)[0]
    commands, uses = [], []
    old_events, new_events = trace["events"], report["physical"]["events"]
    assert len(old_events) == len(new_events)
    for old, new in zip(old_events, new_events):
        if old["kind"] == "payload":
            assert old == new
            continue
        assert (old["kind"], old["gap"], old["pipe"]) == (new["kind"], new["gap"], new["pipe"])
        command = dict(kind=old["kind"], gap=old["gap"], pipe=old["pipe"])
        if old["kind"] != "barrier":
            identity = (old["plan"], old["record"], old["source_ordinal"], tuple(old.get("members", [])))
            command["identity"] = identity
            assert new["physical_id"] in range(6)
            assert (old["source_pipe"], old["target_pipe"]) == (new["source_pipe"], new["target_pipe"])
            event = (new["source_pipe"], new["target_pipe"], new["physical_id"])
            uses.append((len(commands), new["kind"], event, identity))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(expected)]
    complete = closure_with_commands(pipes, commands, include_commands=True)
    live, consumed = {}, {}
    for index, kind, event, identity in uses:
        vertex = 2 * len(pipes) + index
        if kind == "set":
            assert event not in live
            if event in consumed:
                assert complete[consumed[event]] >> vertex & 1
            live[event] = identity
        else:
            assert live.pop(event, None) == identity
            consumed[event] = vertex
    assert not live


def main():
    tool = shutil.which(sys.argv[1]) or sys.argv[1]
    source = Path(sys.argv[2]).read_text()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recurring.pto"
        path.write_text(source)
        for policy in ("may-not-alias", "may-alias"):
            path.write_text(source)
            proof = invoke(tool, "--numeric-analysis", path, policy)
            marker = "recurring-numeric: sealed membership fresh plans retained math unavailable selectors"
            assert marker in proof.stderr
            documents = [json.loads(line) for line in proof.stdout.splitlines() if line.startswith("{")]
            forms = [a for d in documents for n in d["nodes"] for a in n["attempts"]
                     if a["route"] == "numeric-template" and a["state"] == "applicable"]
            assert len(forms) == 1 and forms[0]["estimated_attempted_fragments"] > 0
            for trips in (0, 1, 2, 5):
                for seed in (0, -1, (1 << 60) - 1, 1 << 60, -(1 << 60), (1 << 63) - 1):
                    path.write_text(source.replace("array<i64: 3, 0>", f"array<i64: {trips}, {seed}>"))
                    report = json.loads(invoke(tool, "--physical-trace", path, policy).stdout)
                    check_trace(report, trips, seed)
    print("recurring numerical GM: sealed owner, 48 original-byte closures and six-ID pairing passed")


if __name__ == "__main__":
    main()
