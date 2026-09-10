#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Portable production-helper tests and optional native delta-reconstruction tests.

Finite graphs test the update equation, not Ascend hardware semantics or
arbitrary symbolic loops. --driver exercises the actual RelationQueries binary.
Missing native dependencies are reported as not-run, never as successful tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def compose(a: set, b: set) -> set:
    by_source: dict = {}
    for x, y in b:
        by_source.setdefault(x, set()).add(y)
    return {(x, z) for x, y in a for z in by_source.get(y, ())}


def advance(reached: set, pending: set, transition: set, steps: int | None = None) -> tuple[set, set]:
    reached, pending = set(reached), set(pending)
    count = 0
    while pending and (steps is None or count < steps):
        pending = compose(pending, transition) - reached
        reached |= pending
        count += 1
    return reached, pending


def incremental(order: set, old: set, addition: set, reached: set, pending: set) -> tuple[set, set]:
    # Same exact equation as the native patch; the full closure reference below
    # is recomputed independently from the new primitive population.
    seeds = compose(order, addition) | compose(compose(reached, order), addition)
    return advance(reached | seeds, pending | seeds, compose(order, old | addition))


def fresh_reachability(order: set, handoffs: set) -> set:
    """Independent queue-based search, without the incremental compose/update."""
    issue, publish = {}, {}
    for a, b in order:
        issue.setdefault(a, set()).add(b)
    for a, b in handoffs:
        publish.setdefault(a, set()).add(b)
    answer = set()
    for source in issue:
        reached = set()
        work = [source]
        while work:
            start = work.pop()
            for producer in issue.get(start, ()):
                for consumer in publish.get(producer, ()):
                    if consumer not in reached:
                        reached.add(consumer)
                        work.append(consumer)
        answer.update((source, target) for target in reached)
    return answer


def finite_checks() -> dict:
    rng = random.Random(0x0A45D311)
    assertions = 0
    cases = 0
    for n in range(2, 11):
        for _ in range(120):
            lanes = [rng.randrange(3) for _ in range(n)]
            order = {(a, b) for a in range(n) for b in range(a, n) if lanes[a] == lanes[b]}
            possible = [(a, b) for a in range(n) for b in range(a + 1, n)]
            old = {p for p in possible if rng.randrange(5) == 0}
            addition = {p for p in possible if rng.randrange(5) == 0}
            expected = fresh_reachability(order, old | addition)
            for warmup in (0, 1, 2, None):
                reached, pending = advance(compose(order, old), compose(order, old), compose(order, old), warmup)
                got, tail = incremental(order, old, addition, reached, pending)
                assert got == expected and not tail, "delta update differs from fresh closure"
                assertions += 1
                # Addition of an empty family, and repeated additions, must not
                # change the completed fixed point.
                again, tail = incremental(order, old | addition, addition, got, set())
                assert again == expected and not tail
                assertions += 1
                empty, tail = incremental(order, old | addition, set(), got, set())
                assert empty == expected and not tail
                assertions += 1
                cases += 1
    # An example that fails if only new source-scoped direct paths are seeded.
    identity = {(i, i) for i in range(5)}
    old = {(0, 1), (1, 2)}
    reached, pending = advance(old, old, old)
    got, _ = incremental(identity, old, {(2, 3), (3, 4)}, reached, pending)
    assert (0, 4) in got
    assertions += 1
    return {"status": "passed", "graph_states": cases, "assertions": assertions,
            "scope": "finite relation-update equation, not native compiler or device execution"}


def relation(edges: set, dimensions: int = 1) -> dict:
    pieces = []
    for source, target in sorted(edges):
        source = (source,) if isinstance(source, int) else source
        target = (target,) if isinstance(target, int) else target
        values = (*source, *target)
        rows = []
        for i, value in enumerate(values):
            row = [0] * (2 * dimensions + 1)
            row[i], row[-1] = 1, -value
            rows.append(row)
        pieces.append({"locals": 0, "eq": rows, "ge": []})
    return {"d": dimensions, "r": dimensions, "s": 0, "pieces": pieces}


def native_cases() -> list[tuple[str, dict, list[str]]]:
    scenarios = [
        ("saturated-source-new-tail", {(0, 1), (1, 2)},
         [({(0, 7)}, None, 8), ({(0, 4)}, {(2, 3), (3, 4)}, 8), ({(0, 7)}, None, 8)]),
        ("retain-unfinished-work", {(0, 1), (1, 2), (2, 3)},
         [({(0, 1)}, None, 0), ({(0, 3)}, {(5, 6)}, 8), ({(0, 6)}, None, 8)]),
        ("new-direct-source", {(2, 3)},
         [({(0, 3)}, None, 8), ({(0, 3)}, {(0, 1), (1, 2)}, 8)]),
        ("repeat-and-empty-addition", {(0, 1), (1, 2)},
         [({(0, 2)}, None, 8), ({(0, 2)}, set(), 8), ({(0, 2)}, {(1, 2)}, 8)]),
    ]
    result = []
    for name, initial, steps in scenarios:
        identity = {(i, i) for i in range(8)}
        population = set(initial)
        needs, expected = [], []
        for need, addition, rounds in steps:
            item = {"relation": relation(need), "rounds": rounds}
            if addition is not None:
                item["add_handoffs"] = relation(addition)
                population |= addition
            closure = fresh_reachability(identity, population)
            expected.append("proved" if need <= closure else "not-established")
            needs.append(item)
        for lazy in (False, True):
            result.append((name + ("-lazy" if lazy else "-explicit"),
                {"op": "completion", "a": relation(initial), "issue_order": relation(identity),
                 "needs": needs, "record_steps": True, "lazy": lazy, "budget": 8000000}, expected))
    # One source phase has several distinct invocations. Only the first gains
    # a new chain; grouping by phase must not transfer it to the other one.
    identity = {((p, i), (p, i)) for p in range(4) for i in range(2)}
    old = {((0, i), (1, i)) for i in range(2)}
    addition = {((1, 0), (2, 0))}
    for lazy in (False, True):
        result.append(("invocation-correspondence" + ("-lazy" if lazy else "-explicit"), {
            "op": "completion", "a": relation(old, 2), "issue_order": relation(identity, 2),
            "lazy": lazy, "record_steps": True, "budget": 8000000,
            "needs": [
                {"relation": relation({((0, 0), (2, 0))}, 2), "rounds": 8},
                {"relation": relation({((0, 0), (2, 0))}, 2), "add_handoffs": relation(addition, 2), "rounds": 8},
                {"relation": relation({((0, 1), (2, 1))}, 2), "rounds": 8},
                {"relation": relation({((0, 1), (2, 0))}, 2), "rounds": 8}],
        }, ["not-established", "proved", "not-established", "not-established"]))
    result.append(("reject-incompatible-addition-without-losing-state", {
        "op": "completion", "a": relation({(0, 1), (1, 2)}),
        "issue_order": relation({(i, i) for i in range(4)}), "budget": 8000000,
        "needs": [
            {"relation": relation({(0, 2)}), "rounds": 8},
            {"relation": relation({(0, 2)}), "add_handoffs": relation({((0, 0), (1, 0))}, 2)},
            {"relation": relation({(0, 2)}), "rounds": 8}],
    }, ["proved", "unsupported", "proved"]))
    return result


def main() -> None:
    if not __debug__:
        raise RuntimeError("These acceptance tests require assertions enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"portable_cpp": {"status": "not-run"}, "finite_delta": {"status": "not-run"},
               "native_delta": {"status": "not-run", "reason": "--driver not supplied"},
               "device": {"status": "not-run"}}
    records = []

    def persist() -> None:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")

    def run(name: str, command: list[str], payload: str | None = None) -> subprocess.CompletedProcess:
        started = time.monotonic()
        try:
            completed = subprocess.run(command, input=payload, capture_output=True, text=True,
                                       check=False, timeout=40)
        except subprocess.TimeoutExpired as exc:
            records.append({"name": name, "command": command, "status": "timeout", "seconds": time.monotonic() - started})
            persist()
            raise RuntimeError(f"{name} timed out") from exc
        records.append({"name": name, "command": command, "returncode": completed.returncode,
                        "seconds": time.monotonic() - started})
        (args.output / (name + ".stdout")).write_text(completed.stdout)
        (args.output / (name + ".stderr")).write_text(completed.stderr)
        persist()
        if completed.returncode:
            raise RuntimeError(f"{name} failed:\n{completed.stderr[-4000:]}")
        return completed

    persist()
    executable = args.output.resolve() / "compact_forms_test"
    command = [*shlex.split(args.cxx), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
    if args.sanitize:
        command.append("-fsanitize=undefined,address")
    command += ["-I", str(ROOT / "include"), str(HERE / "compact_forms_test.cpp"), "-o", str(executable)]
    run("compile-portable", command)
    result = run("portable", [str(executable)])
    summary["portable_cpp"] = {"status": "passed", "result": result.stdout.strip(),
                               "sanitizers": args.sanitize, "compiler": args.cxx}
    summary["finite_delta"] = finite_checks()
    cases = native_cases()
    for name, request, _ in cases:
        (args.output / (name + ".request.json")).write_text(json.dumps(request, indent=2) + "\n")
    if args.driver:
        driver = args.driver.resolve(strict=True)
        checks = []
        summary["native_delta"] = {"status": "running", "driver": str(driver),
            "sha256": hashlib.sha256(driver.read_bytes()).hexdigest(), "cases": checks}
        persist()
        for name, request, expected in cases:
            actual = run(name, [str(driver)], json.dumps(request))
            response = json.loads(actual.stdout)
            statuses = [answer["status"] for answer in response["answers"]]
            if statuses != expected:
                summary["native_delta"]["status"] = "failed"
                persist()
                raise AssertionError(f"{name}: expected {expected}, received {statuses}")
            checks.append({"name": name, "status": "passed", "answers": statuses})
            persist()
        summary["native_delta"]["status"] = "passed"
    persist()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
