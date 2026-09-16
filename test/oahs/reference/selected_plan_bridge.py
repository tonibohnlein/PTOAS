#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check selected C++ constructions against the unchanged exact causal reference.

The reference collector is test-only. It has no dynamic trip-count bound and
checks the original control graph, not the constructor's discharge records.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

REFERENCE_SHA256 = "543a7c796acf602d068d3b374c287b224104ddbc5d1d71a785224cc8db5162d1"


def require(condition: bool, detail: object) -> None:
    if not condition:
        raise AssertionError(detail)


def reference_module():
    path = Path(__file__).resolve().parent / "vendor" / "v08" / "causal_interface.py"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256, "pinned reference changed")
    spec = importlib.util.spec_from_file_location("selected_causal_reference", path)
    require(spec is not None and spec.loader is not None, "reference loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The unchanged paired module imports its sibling by this original name.
    sys.modules["causal_interface"] = module
    return module


def paired_module():
    path = Path(__file__).resolve().parent / "vendor" / "v08" / "order_interface.py"
    expected = "49db607fa19919aec2a12c1a7e81fbee91501de59b26d984019219126520053f"
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "pinned order reference changed")
    spec = importlib.util.spec_from_file_location("selected_order_reference", path)
    require(spec is not None and spec.loader is not None, "order reference loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def original_graph(case: dict) -> tuple[list, int, int]:
    if case["observed"]:
        return case["sites"], case["entry"], case["exit"]
    count = len(case["operations"])
    nodes = [[i if i < count else None, []] for i in range(count + 1)]

    def fresh() -> int:
        nodes.append([None, []])
        return len(nodes) - 1

    def build(region: dict, after: int) -> int:
        kind, children = region["kind"], region["children"]
        if kind == 4:
            at = region["operation"]
            nodes[at][1] = [after]
            return at
        if kind == 0:
            current = after
            for child in reversed(children):
                current = build(child, current)
            return current
        if kind == 1:
            at = fresh()
            nodes[at][1] = [build(child, after) for child in children]
            return at
        if kind == 2:
            at = fresh()
            nodes[at][1] = [build(children[0], at), after]
            return at
        require(kind == 3 and len(children) == 2, "unsupported original region")
        decision = fresh()
        before = build(children[0], decision)
        body = build(children[1], before)
        nodes[decision][1] = [body, after]
        return before

    body = case["body"]
    if body["kind"] == 0 and not body["children"]:
        for i in range(count):
            nodes[i][1] = [i + 1]
        return nodes, 0, count
    return nodes, build(body, count), count


def mask(words: list[int]) -> int:
    return sum(word << (64 * i) for i, word in enumerate(words))


def compare(facts: dict, state, bindings: tuple, model, location: object) -> int:
    require(facts is not None, (location, "missing reachable frontier"))
    checks = 0
    for claimed, actual in zip(facts["reach"], state.reach):
        require(mask(claimed) & actual == mask(claimed), (location, "invented reachability"))
        checks += 1
    for claimed, actual in zip(facts["history"], state.history):
        if actual:
            require(claimed is not None, (location, "dropped access class"))
            for signature in actual:
                require(mask(claimed) & signature == mask(claimed), (location, "invented history"))
                checks += 1
    for index, event in enumerate(facts["events"]):
        full = bool(state.live & (1 << index))
        require(event["occupancy"] & (2 if full else 1), (location, "lost possible balance"))
        if full:
            require(list(bindings[index]) in event["publishers"], (location, "lost publication origin"))
        checks += 1
    require(len(facts["reach"]) == model.m and len(facts["history"]) == len(model.classes)
            and len(facts["events"]) == len(model.keys), (location, "incomplete exported dimensions"))
    return checks


def collect(case: dict, ref, paired=None) -> tuple[int, int]:
    keys = {":".join(map(str, key)): (f"p{key[0]}", f"p{key[1]}") for key in case["keys"]}
    base = ref.Interface([f"p{i}" for i in range(7)], [f"c{i}" for i in range(case["cells"])], keys)
    model = base if paired is None else paired.OrderedInterface(base)
    nodes, entry, exit_site = original_graph(case)
    initial = (model.initial(), tuple(None for _ in keys))
    reached = [set() for _ in nodes]
    reached[entry].add(initial)
    queue = deque([(entry, initial)])
    visits = checks = 0
    while queue:
        at, (state, origins) = queue.popleft()
        visits += 1
        bindings = list(origins)
        has_word = at < len(case["commands"])
        if has_word:
            checks += compare(case["cuts"][at][0], state if paired is None else state.base,
                              tuple(bindings), base, (case["name"], at, "incoming"))
            for index, (kind, source, observer, key) in enumerate(case["commands"][at]):
                if kind == 2:
                    command = ref.Command("fence", f"p{source}")
                else:
                    require(kind in (0, 1), "ordinary reference cannot certify ALL")
                    name = f"{source}:{observer}:{key}"
                    command = ref.Command("set" if kind == 0 else "wait",
                                          f"p{source if kind == 0 else observer}", key=name)
                    bindings[model.ki[name]] = (at, index) if kind == 0 else None
                state = model.transfer(state, command)
            checks += compare(case["cuts"][at][1], state if paired is None else state.base,
                              tuple(bindings), base, (case["name"], at, "before"))
        operation = nodes[at][0]
        if operation is not None:
            op = case["operations"][operation]
            reads = tuple(f"c{cell}" for cell, read, _ in op["effects"] if read)
            writes = tuple(f"c{cell}" for cell, _, write in op["effects"] if write)
            state = model.transfer(state, ref.Command("op", f"p{op['pipe']}", reads, writes))
        if at == exit_site:
            require(state.live == 0, (case["name"], "unconsumed exit"))
        if has_word:
            checks += compare(case["cuts"][at][2], state if paired is None else state.base,
                              tuple(bindings), base, (case["name"], at, "outgoing"))
        outgoing = (state, tuple(bindings))
        for successor in nodes[at][1]:
            if outgoing not in reached[successor]:
                reached[successor].add(outgoing)
                queue.append((successor, outgoing))
    return visits, checks


class Graph:
    """Explicit full-history graph independent of the live-port quotient."""

    def __init__(self) -> None:
        self.parents: list[set[int]] = []

    def node(self, parents=()) -> int:
        self.parents.append(set(parents))
        return len(self.parents) - 1

    def closure(self) -> list[set[int]]:
        result = []
        for node, parents in enumerate(self.parents):
            require(all(parent < node for parent in parents), "non-forward graph")
            known = set(parents)
            for parent in parents:
                known.update(result[parent])
            result.append(known)
        return result


def check_finite_order(case: dict) -> int:
    require(not case["observed"] and not case["body"]["children"], "finite order oracle requires flat original")
    actual, ideal = Graph(), Graph()
    issued = [None] * 7
    gated = [None] * 7
    finishes = [[] for _ in range(7)]
    reference_issue = [None] * 7
    publications: dict[tuple, int] = {}
    consumptions: dict[tuple, int] = {}
    observables, reference_observables, accesses = [], [], []
    rearm = []
    for cut, word in enumerate(case["commands"]):
        for kind, source, observer, key in word:
            pipe = observer if kind == 1 else source
            parents = [v for v in (issued[pipe], gated[pipe]) if v is not None]
            issue = actual.node(parents)
            predecessors = [issue]
            identity = (source, observer, key)
            if kind in (0, 2):
                predecessors += finishes[pipe]
            if kind == 1:
                require(identity in publications, "absent publication")
                predecessors.append(publications.pop(identity))
            finish = actual.node(predecessors)
            issued[pipe] = issue
            finishes[pipe].append(finish)
            if kind in (1, 2):
                gated[pipe] = finish
            if kind == 0:
                require(identity not in publications, "occupied publication")
                publications[identity] = finish
                if identity in consumptions:
                    rearm.append((consumptions[identity], finish))
            elif kind == 1:
                consumptions[identity] = finish
        if cut == len(case["operations"]):
            continue
        op = case["operations"][cut]
        pipe = op["pipe"]
        issue = actual.node([v for v in (issued[pipe], gated[pipe]) if v is not None])
        finish = actual.node([issue])
        issued[pipe] = issue
        finishes[pipe].append(finish)
        observables += [issue, finish]
        reads = {c for c, r, _ in op["effects"] if r}
        writes = {c for c, _, w in op["effects"] if w}
        parents = [] if reference_issue[pipe] is None else [reference_issue[pipe]]
        parents += [end for old_reads, old_writes, end in accesses
                    if writes & (old_reads | old_writes) or reads & old_writes]
        ri = ideal.node(parents)
        rc = ideal.node([ri])
        reference_issue[pipe] = ri
        accesses.append((reads, writes, rc))
        reference_observables += [ri, rc]
    require(not publications, "live finite exit")
    actual_closure, reference_closure = actual.closure(), ideal.closure()
    for consumption, publication in rearm:
        require(consumption in actual_closure[publication], "missing physical rearm")
    comparisons = 0
    for i, source in enumerate(observables):
        for j, target in enumerate(observables):
            require((source in actual_closure[target]) ==
                    (reference_observables[i] in reference_closure[reference_observables[j]]),
                    (case["name"], "different payload order", i, j))
            comparisons += 1
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    driver = args.driver.resolve(strict=True)
    run = subprocess.run([str(driver)], capture_output=True, text=True, timeout=180, check=False)
    require(run.returncode == 0, ("C++ driver failed", run.returncode, run.stderr))
    cases = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    require(len(cases) == 108, ("changed acceptance denominator", len(cases)))
    ref = reference_module()
    paired = paired_module()
    counts = {"constructed": 0, "exact_reference_states": 0, "fact_implications": 0,
              "finite_order_cases": 0, "finite_order_pairs": 0, "selected_updates": 0,
              "selected_replay_evaluations": 0, "forward_evaluations": 0,
              "paired_order_cases": 0, "paired_order_states": 0}
    for case in cases:
        require(case["success"], (case["name"], "construction refusal", case["reason"]))
        visits, checks = collect(case, ref)
        counts["constructed"] += 1
        counts["exact_reference_states"] += visits
        counts["fact_implications"] += checks
        counts["selected_updates"] += case["work"]["updates"]
        counts["selected_replay_evaluations"] += case["work"]["replay"]
        counts["forward_evaluations"] += case["work"]["forward"]
        if case["expected_exact"] or case["name"].startswith("cyclic-"):
            paired_visits, _ = collect(case, ref, paired)
            counts["paired_order_cases"] += 1
            counts["paired_order_states"] += paired_visits
        if case["expected_exact"]:
            counts["finite_order_pairs"] += check_finite_order(case)
            counts["finite_order_cases"] += 1
    text = json.dumps(counts, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
