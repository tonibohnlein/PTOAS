# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check Delta's fixed one-token prefix protocol against command causality.

The invocation binds block/subblock=0, cu_seqlens=[0,1], state_indices[0]=-1,
and a[0]=dt_bias[0]=0. This is a scoped capacity witness, not a device bound.
Replay retains original payloads and commands, binding only observed control.
The resulting specialized body has a stale allocation seal: it is trace-only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from check_fixed_protocol_capacity import scoped_capacity_report
from check_fixed_protocol_capacity import (capacity, check_protocol_comparison, feedback_case,
                                           same_fixed_protocol)
from check_logical_insertion import closure_with_commands

BINDINGS = {"block_idx": 0, "subblock_idx": 0, "sequence_start": 0,
            "sequence_end": 1, "state_index": -1, "softplus_gt_20": False}


def bind_delta(logical):
    assert "test.trace_existing" not in logical
    marker = re.compile(r"^(\s*func\.func @fused_gating_delta\([^\n]*? attributes \{)", re.M)
    logical, count = marker.subn(r"\1test.trace_existing, ", logical)
    assert count == 1, "expected exactly one retained Delta function"
    counts = defaultdict(int)
    pattern = re.compile(
        r"^(\s*)(%\w+) = (pto\.get_block_idx|pto\.get_subblock_idx|"
        r"pto\.load %arg([89])\[[^\n]+|arith\.cmpf ogt, [^\n]+ : f32)$", re.M)

    def bind(match):
        indent, result, operation, argument = match.groups()
        if operation == "pto.get_block_idx":
            key, value, ty = "block_idx", 0, "i64"
        elif operation == "pto.get_subblock_idx":
            key, value, ty = "subblock_idx", 0, "i64"
        elif argument == "9":
            key = "sequence_start" if counts["sequence_start"] == 0 else "sequence_end"
            value, ty = BINDINGS[key], "i32"
            assert operation.endswith("!pto.ptr<i32, gm> -> i32")
        elif argument == "8":
            key, value, ty = "state_index", -1, "i32"
            assert operation.endswith("!pto.ptr<i32, gm> -> i32")
        else:
            key, value, ty = "softplus_gt_20", 0, "i1"
        counts[key] += 1
        return (f"{indent}%delta_observation_{result[1:]} = {operation}\n"
                f"{indent}{result} = arith.constant {value} : {ty}")

    bound = pattern.sub(bind, logical)
    assert dict(counts) == {key: 1 for key in BINDINGS}, dict(counts)
    return bound


def check(report):
    result = capacity(report)
    events = report["trace"]["events"]
    pipes = [event["pipe"] for event in events if event["kind"] == "payload"]
    commands, sets, waits, batches, current = [], {}, {}, [], []
    for event_index, event in enumerate(events):
        is_selected = (event["kind"] == "wait" and event["source_pipe"] == 1
                       and event["target_pipe"] == 5)
        if not is_selected and current:
            batches.append(current)
            current = []
        if event["kind"] == "payload":
            continue
        command = dict(event)
        vertex = 2 * len(pipes) + len(commands)
        if event["kind"] != "barrier":
            identity = (event["source_pipe"], event["target_pipe"], event["plan"],
                        event["record"], event["source_ordinal"], tuple(event.get("members", [])))
            command["identity"] = identity
            (sets if event["kind"] == "set" else waits)[identity] = vertex
            if is_selected:
                current.append((identity, event_index, len(commands), event["gap"]))
        commands.append(command)
    if current:
        batches.append(current)
    assert batches, "no V -> MTE3 acquisition batch"
    reach = closure_with_commands(pipes, commands, include_commands=True)
    witness = max(batches, key=len)
    identities = [row[0] for row in witness]
    assert len({identity[:4] for identity in identities}) == 1
    assert len({row[3] for row in witness}) == 1, "batch crosses a payload"
    assert all(b[4] == a[4] + 1 for a, b in zip(identities, identities[1:]))
    assert all(not (reach[waits[a]] >> sets[b] & 1) for a in identities for b in identities), \
        "batch is not a reuse antichain in the complete command graph"
    assert len(witness) > 6, "this batch does not establish overcapacity for six directed IDs"
    selected = next(row for row in result["directions"] if row["direction"] == [1, 5])
    assert selected["minimum_ids"] >= len(witness)
    result["batch_lower_bound"] = len(witness)
    result["witness_direction"] = [1, 5]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--logical", type=Path)
    parser.add_argument("--tracer")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    root = Path(__file__).resolve().parents[4]
    source = root / fixture["input"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fixture["input_sha256"]
    assert fixture["bindings"] == BINDINGS
    result = check(fixture["report"])
    expected = fixture["expected"]
    for key in ("payloads", "commands", "batch_lower_bound", "witness_direction"):
        assert result[key] == expected[key], key
    selected = next(row for row in result["directions"] if row["direction"] == expected["witness_direction"])
    assert selected["minimum_ids"] == expected["minimum_ids"]
    assert selected["handoffs"] == 46 and selected["reuse_edges"] == 0
    feedback_case(False)
    feedback_case(True)
    check_protocol_comparison()
    if bool(args.logical) != bool(args.tracer):
        parser.error("--logical and --tracer must be supplied together")
    if args.logical:
        executable = shutil.which(args.tracer)
        if executable is None:
            parser.error("--tracer must name an available executable")
        executable = str(Path(executable).resolve(strict=True))
        # The logical IR is large. Keep replay scratch beside its disk-backed input.
        with tempfile.TemporaryDirectory(prefix="delta-capacity-", dir=args.logical.resolve().parent) as scratch:
            path = Path(scratch) / "input.pto"
            path.write_text(bind_delta(args.logical.read_text()))
            run = subprocess.run([executable, "--structured-trace", str(path)], capture_output=True,
                                 text=True, check=True, timeout=120)
        replay = json.loads(run.stdout)
        assert same_fixed_protocol(replay["trace"], fixture["report"]["trace"]), \
            "fixed protocol changed; investigate before refreshing"
        assert check(replay) == result
    print(json.dumps(scoped_capacity_report(fixture, result), sort_keys=True))


if __name__ == "__main__":
    main()
