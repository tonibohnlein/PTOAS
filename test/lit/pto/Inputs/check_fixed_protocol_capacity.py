# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Certify notification capacity for an explicit, unchanged command graph.

This is a test oracle, not a runtime allocator. Commands, matching and legal cuts
are fixed. Reusing a binary ID requires WAIT(a) to causally precede SET(b).
Textual SET/WAIT order alone supplies no such edge. Other protocols are outside
the result, as are successful kernel compilation and device execution.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from check_logical_insertion import closure_with_commands


def capacity(report):
    assert report["accepted"] and not report["trace"]["error"], "trace was not accepted"
    trace = report["trace"]
    pipes = [event["pipe"] for event in trace["events"] if event["kind"] == "payload"]
    assert len(pipes) == trace["payloads"]
    commands, sets, waits = [], {}, {}
    directions = defaultdict(list)
    gap = 0
    for event in trace["events"]:
        kind = event["kind"]
        assert kind in ("payload", "set", "wait", "barrier"), "unknown event kind"
        assert isinstance(event["pipe"], int) and 0 <= event["pipe"] <= 6
        if kind == "payload":
            gap += 1
            continue
        assert event["gap"] == gap, "command cut disagrees with payload order"
        command = dict(event)
        if kind != "barrier":
            source, target = event["source_pipe"], event["target_pipe"]
            assert 0 <= source < 6 and 0 <= target < 6 and source != target
            assert event["pipe"] == (source if kind == "set" else target)
            identity = (source, target, event["plan"], event["record"],
                        event["source_ordinal"], tuple(event.get("members", [])))
            command["identity"] = identity
            table = sets if kind == "set" else waits
            assert identity not in table, "duplicate matching endpoint"
            table[identity] = 2 * len(pipes) + len(commands)
            if kind == "set":
                directions[(source, target)].append(identity)
        commands.append(command)
    assert sets.keys() == waits.keys(), "unmatched endpoints"
    reach = closure_with_commands(pipes, commands, include_commands=True)
    results = []
    for direction, identities in sorted(directions.items()):
        edges = [[j for j, later in enumerate(identities) if reach[waits[earlier]] >> sets[later] & 1]
                 for earlier in identities]
        # Matching makes this a strict partial order: W_a -> S_b -> W_b -> S_c.
        assert all(i not in row for i, row in enumerate(edges))
        assert all(set(edges[j]) <= set(row) for row in edges for j in row)
        matched = {}

        def augment(source, seen):
            for target in edges[source]:
                if target in seen:
                    continue
                seen.add(target)
                if target not in matched or augment(matched[target], seen):
                    matched[target] = source
                    return True
            return False

        for source in range(len(identities)):
            augment(source, set())
        left = set(range(len(identities))) - set(matched.values())
        right, pending = set(), list(left)
        while pending:
            source = pending.pop()
            for target in edges[source]:
                if matched.get(target) == source or target in right:
                    continue
                right.add(target)
                if target in matched and matched[target] not in left:
                    left.add(matched[target])
                    pending.append(matched[target])
        antichain = sorted(left - right)
        bound = len(identities) - len(matched)
        assert len(antichain) == bound
        assert all(j not in edges[i] for i in antichain for j in antichain)
        results.append({"direction": list(direction), "handoffs": len(identities),
                        "minimum_ids": bound, "reuse_edges": sum(map(len, edges)),
                        "antichain": [identities[i] for i in antichain]})
    return {"payloads": len(pipes), "commands": len(commands), "directions": results}


def feedback_case(feedback):
    # Both P->Q handoffs occur in textual order. A Q->P return handshake,
    # when present, makes the first consumption precede the second publication.
    events = []

    def command(kind, source, target, record):
        events.append({"kind": kind, "gap": 0, "pipe": source if kind == "set" else target,
                       "source_pipe": source, "target_pipe": target,
                       "plan": 0, "record": record, "source_ordinal": 0})

    command("set", 0, 1, 0)
    command("wait", 0, 1, 0)
    if feedback:
        command("set", 1, 0, 2)
        command("wait", 1, 0, 2)
    command("set", 0, 1, 1)
    command("wait", 0, 1, 1)
    report = {"accepted": True, "trace": {"error": "", "payloads": 0, "events": events}}
    result = capacity(report)
    forward = next(row for row in result["directions"] if row["direction"] == [0, 1])
    assert forward["minimum_ids"] == (1 if feedback else 2)
    return report


def bind_decode(logical, bindings):
    # Preserve every original payload. Only replace uses of scalar inputs with
    # the selected valuation; renaming the original result retains its load.
    marker = "attributes {pto.endpoint_families"
    assert logical.count(marker) == 1
    logical = logical.replace(marker, "attributes {test.trace_existing, pto.endpoint_families", 1)
    counts = defaultdict(int)
    pattern = re.compile(r"^(\s*)(%[\w]+) = (pto\.get_block_idx|pto\.load %arg([345])\[[^\n]+)$", re.M)

    def replace(match):
        indent, name, operation, argument = match.groups()
        key = "arg" + argument + "_load" if argument else "block_idx"
        counts[key] += 1
        kind = "i32" if argument else "i64"
        return (f"{indent}%capacity_original_{name[1:]} = {operation}\n"
                f"{indent}{name} = arith.constant {bindings[key]} : {kind}")

    result = pattern.sub(replace, logical)
    assert dict(counts) == {key: 1 for key in bindings}, counts
    return result


def same_fixed_protocol(left, right):
    """Compare protocol evidence while allowing measured scalar work to change."""
    counters = {"visits", "arithmetic_visits", "select_visits"}
    return ({key: value for key, value in left.items() if key not in counters}
            == {key: value for key, value in right.items() if key not in counters})


def check_protocol_comparison():
    original = feedback_case(False)["trace"]
    measured = dict(original, visits=100, arithmetic_visits=80, select_visits=20)
    assert same_fixed_protocol(original, measured)
    for changed in (dict(measured, error="invalid"), dict(measured, payloads=1),
                    dict(measured, events=list(reversed(original["events"]))),
                    dict(measured, unknown_evidence=1)):
        assert not same_fixed_protocol(original, changed)
    for field in ("gap", "record", "source_ordinal", "source_pipe", "target_pipe"):
        events = [dict(event) for event in original["events"]]
        events[0][field] += 1
        assert not same_fixed_protocol(original, dict(measured, events=events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--logical", type=Path)
    parser.add_argument("--tracer")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    root = Path(__file__).resolve().parents[4]
    assert hashlib.sha256((root / fixture["input"]).read_bytes()).hexdigest() == fixture["input_sha256"]
    assert fixture["bindings"] == {"block_idx": 0, "arg3_load": 0, "arg4_load": 0, "arg5_load": 1}
    result = capacity(fixture["report"])
    expected = fixture["expected"]
    assert result["payloads"] == expected["payloads"] and result["commands"] == expected["commands"]
    selected = next(row for row in result["directions"] if row["direction"] == expected["direction"])
    assert selected["minimum_ids"] == expected["minimum_ids"]
    assert selected["handoffs"] == 9 and selected["reuse_edges"] == 0
    feedback_case(False)
    feedback_case(True)
    check_protocol_comparison()
    # Reject malformed evidence rather than silently dropping commands.
    for damage in ("gap", "kind", "pipe", "unmatched", "duplicate", "cycle"):
        bad = feedback_case(False)
        events = bad["trace"]["events"]
        if damage == "gap":
            events[0]["gap"] = 1
        elif damage == "kind":
            events[0]["kind"] = "unknown"
        elif damage == "pipe":
            events[0]["pipe"] = 1
        elif damage == "unmatched":
            events.pop()
        elif damage == "duplicate":
            events.append(dict(events[0]))
        else:
            # Crossed WAIT->SET blockers and matching form a command cycle.
            bad = feedback_case(True)
            events = bad["trace"]["events"]
            events[:] = [events[3], events[0], events[1], events[2], events[4], events[5]]
        try:
            capacity(bad)
        except AssertionError:
            continue
        raise AssertionError("accepted malformed " + damage)
    if bool(args.logical) != bool(args.tracer):
        parser.error("--logical and --tracer must be supplied together")
    if args.logical:
        executable = shutil.which(args.tracer)
        if executable is None:
            parser.error("--tracer must name an available executable")
        executable = str(Path(executable).resolve(strict=True))
        # Only a small, short-lived textual trace input is placed in tmpfs.
        with tempfile.TemporaryDirectory(prefix="decode-capacity-") as scratch:
            path = Path(scratch) / "input.pto"
            path.write_text(bind_decode(args.logical.read_text(), fixture["bindings"]))
            run = subprocess.run([executable, "--structured-trace", str(path)], capture_output=True,
                                 text=True, check=True, timeout=30)
        replay = json.loads(run.stdout)
        assert same_fixed_protocol(replay["trace"], fixture["report"]["trace"]), \
            "fixed protocol changed; investigate before refreshing"
        assert capacity(replay) == result
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
