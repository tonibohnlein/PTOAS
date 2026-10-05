# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check emitted IR commands, preserving their observed instruction order."""
from collections import defaultdict, deque


def closure_with_commands(pipes, commands):
    # commands: kind(set/barrier/wait), gap in reference payload order, pipe,
    # and logical identity for SET/WAIT. Gap 0 is before first payload.
    n = len(pipes)
    out = [set() for _ in range(2*n + len(commands))]
    rows = defaultdict(list)
    for i, pipe in enumerate(pipes):
        out[2*i].add(2*i+1)
        rows[pipe].append((2*i+1, 0, i, 'payload'))
    publications = {}
    acquisitions = {}
    for i, cmd in enumerate(commands):
        vertex = 2*n+i
        rows[cmd['pipe']].append((2*cmd['gap'], i, vertex, cmd['kind']))
        if cmd['kind'] != 'barrier':
            table = publications if cmd['kind'] == 'set' else acquisitions
            identity = tuple(cmd['identity'])
            if identity in table:
                raise AssertionError(('duplicate endpoint', cmd))
            table[identity] = vertex
    if publications.keys() != acquisitions.keys():
        raise AssertionError('unmatched endpoints')
    for identity, source in publications.items():
        out[source].add(acquisitions[identity])
    for row in rows.values():
        previous = None
        prior_commands = []
        prior_blockers = []
        for _, _, ident, kind in sorted(row):
            if kind == 'payload':
                if previous is not None:
                    out[2*previous].add(2*ident)
                    out[2*previous+1].add(2*ident+1)
                for blocker in prior_blockers:
                    out[blocker].add(2*ident)
                previous = ident
            else:
                if previous is not None:
                    out[2*previous].add(ident)
                    if kind in ('set', 'barrier'):
                        out[2*previous+1].add(ident)
                for blocker in prior_blockers:
                    out[blocker].add(ident)
                if kind in ('set', 'barrier'):
                    for command in prior_commands:
                        out[command].add(ident)
                prior_commands.append(ident)
                if kind in ('wait', 'barrier'):
                    prior_blockers.append(ident)
    indegree = [0]*len(out)
    for edges in out:
        for b in edges:
            indegree[b] += 1
    ready = deque(i for i,d in enumerate(indegree) if not d)
    schedule = []
    while ready:
        a = ready.popleft()
        schedule.append(a)
        for b in out[a]:
            indegree[b] -= 1
            if not indegree[b]:
                ready.append(b)
    if len(schedule) != len(out):
        raise AssertionError('logical command cycle')
    reach = [0]*len(out)
    for a in reversed(schedule):
        for b in out[a]:
            reach[a] |= (1<<b) | reach[b]
    mask = (1<<(2*n))-1
    return [r & mask for r in reach[:2*n]]


import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from check_periodic_demands import unfolded


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def recognized(tool, path):
    docs = [json.loads(line) for line in invoke(tool, "--recognize", path).splitlines() if line.startswith("{")]
    assert len(docs) == 1
    candidates = [a for n in docs[0]["nodes"] for a in n["attempts"]
                  if a["route"] == "numeric-template" and a["state"] == "applicable"]
    assert len(candidates) == 1
    return candidates[0]


def validate(template, trace):
    assert not trace["error"], trace["error"]
    word = template["payloads"]
    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
    assert len(payloads) == trace["payloads"]
    assert not word or len(payloads) % len(word) == 0
    for i, payload in enumerate(payloads):
        assert (payload["type"], payload["ordinal"], payload["pipe"]) == (
            i % len(word), i // len(word), word[i % len(word)]["pipe"])
    commands = []
    gap = 0
    for event in trace["events"]:
        if event["kind"] == "payload":
            gap += 1
            continue
        assert event["gap"] == gap
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"])
            assert event["pipe"] == event["source_pipe" if event["kind"] == "set" else "target_pipe"]
        commands.append(command)
    effects = [[{"atom": atom, "read": effect["mode"] == "read", "write": effect["mode"] != "read"}
                for effect in item["effects"] for atom in effect["atoms"]] for item in word]
    case = {"pipes": [x["pipe"] for x in word], "word": effects, "records": []}
    required, covers = unfolded(case, len(payloads))
    actual = closure_with_commands([x["pipe"] for x in payloads], commands)
    # The independent DAG helper is reflexive; the command oracle is strict.
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], "inserted commands changed payload order"
    return {"payloads": len(payloads), "commands": len(commands), "covers": len(covers)}


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    original = Path(sys.argv[2]).read_text()
    summaries = []
    # Small, short-lived text inputs only; no compiler artifacts in tmpfs.
    with tempfile.TemporaryDirectory(prefix="logical-insertion-") as directory:
        path = Path(directory) / "case.pto"
        for upper in [-2, 1, 2, 5, 8]:
            path.write_text(original.replace("array<i64: 8>", f"array<i64: {upper}>"))
            template = recognized(tool, path)
            emitted = invoke(tool, "--insert-logical", path)
            assert "pto.logical_set" in emitted and "pto.logical_wait" in emitted
            assert "pto.barrier" in emitted and "PIPE_ALL" not in emitted
            assert "pto.set_flag" not in emitted and "pto.wait_flag" not in emitted
            # Structured payloads and loops remain single original operations.
            for name in ["scf.for", "pto.tload", "pto.textract"]:
                assert emitted.count(name) == original.count(name)
            trace = json.loads(invoke(tool, "--insertion-trace", path))
            assert trace["outer_trips"] == max(0, (upper - 1 + 1) // 2)
            summaries.append(validate(template, trace))
            lowered = Path(directory) / "inserted.pto"
            lowered.write_text(emitted)
            assert invoke(tool, "--roundtrip", lowered) == emitted
            repeated = subprocess.run([tool, "--insert-logical", str(lowered)], capture_output=True, text=True,
                                      check=False, timeout=90)
            assert repeated.returncode != 0, "insertion must reject an already synchronized program"
        typed = original
        typed = typed.replace("    %five = arith.constant 5 : index", """    %five = arith.constant 5 : index
    %inner_one = arith.constant 1 : i32
    %inner_two = arith.constant 2 : i32
    %inner_five = arith.constant 5 : i32""")
        typed = typed.replace("%k = %one to %five step %two {",
                              "%k = %inner_one to %inner_five step %inner_two : i32 {")
        typed = typed.replace("eq, %k, %one : index", "eq, %k, %inner_one : i32")
        path.write_text(typed)
        typed_template = recognized(tool, path)
        typed_trace = json.loads(invoke(tool, "--insertion-trace", path))
        assert typed_trace["outer_trips"] == 4
        summaries.append(validate(typed_template, typed_trace))
        # A compile-time empty invocation needs no schema for unreachable visits.
        empty = original.replace("%t = %lower to %n", "%t = %lower to %one")
        empty = empty.replace("%k = %one to %five", "%k = %one to %n")
        path.write_text(empty)
        empty_template = recognized(tool, path)
        assert empty_template["empty_invocation"] and not empty_template["payloads"]
        assert not empty_template["counted_visits"]
        assert invoke(tool, "--insert-logical", path) == invoke(tool, "--roundtrip", path)
        empty_trace = json.loads(invoke(tool, "--insertion-trace", path))
        assert empty_trace["outer_trips"] == 0 and not empty_trace["events"]
        validate(empty_template, empty_trace)
        # An unknown inner trip count cannot silently produce a partial plan.
        path.write_text(original.replace("%k = %one to %five", "%k = %one to %n"))
        rejected = subprocess.run([tool, "--insert-logical", str(path)], capture_output=True, text=True,
                                  check=False, timeout=90)
        assert rejected.returncode != 0 and "logical insertion requires" in rejected.stderr
    assert summaries[0]["commands"] == summaries[1]["commands"] == 0
    assert summaries[-1]["payloads"] > summaries[2]["payloads"]
    print(f"logical IR insertion: {len(summaries)} exact command-order checks, rejection and roundtrip passed")


if __name__ == "__main__":
    main()
