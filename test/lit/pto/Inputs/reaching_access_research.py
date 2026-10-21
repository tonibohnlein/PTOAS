# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Static may-access experiment. No production dispatch, endpoint or token lowering."""

from collections import deque
from check_periodic_demands import closure, native


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Graph:
    def __init__(self, tree):
        self.nodes = [None]
        self.predecessors = [set()]
        self.successors = [set()]
        self.exit = self.sequence(tree, [0])

    def node(self, site, before):
        result = len(self.nodes)
        self.nodes.append(site)
        self.predecessors.append(set(before))
        self.successors.append(set())
        for source in before:
            self.successors[source].add(result)
        return result

    def sequence(self, tree, before):
        for item in tree:
            kind = item["kind"]
            if kind == "payload":
                before = [self.node(item["site"], before)]
            elif kind == "if":
                gate = self.node(None, before)
                left = self.sequence(item["then"], [gate])
                right = self.sequence(item["else"], [gate])
                before = [self.node(None, left + right)]
            elif kind == "loop":
                header = self.node(None, before)
                for backedge in self.sequence(item["body"], [header]):
                    self.predecessors[header].add(backedge)
                    self.successors[backedge].add(header)
                before = [header]  # Includes entry/zero-trip and final backedge.
            else:
                raise RuntimeError(f"unsupported research control: {kind}")
        return before


def generated(site, modes):
    return {
        (cell, mode, site)
        for cell, bits in enumerate(modes)
        for mode in (1, 2)
        if bits & mode
    }


def solve(graph, sites, refresh=None):
    """Optional refresh is supplied proven full-class GEN/KILL, not inferred."""
    refresh = refresh or {}
    incoming = [set() for _ in graph.nodes]
    outgoing = [set() for _ in graph.nodes]
    pending = deque(range(len(graph.nodes)))
    queued = set(pending)
    visits = 0
    while pending:
        node = pending.popleft()
        queued.remove(node)
        visits += 1
        facts = set().union(*(outgoing[p] for p in graph.predecessors[node]))
        incoming[node] = facts
        site = graph.nodes[node]
        after = facts
        if site is not None:
            after = {fact for fact in facts if fact[0] not in refresh.get(site, ())}
            after |= generated(site, sites[site]["facts"])
        if after != outgoing[node]:
            outgoing[node] = after
            for successor in graph.successors[node]:
                if successor not in queued:
                    pending.append(successor)
                    queued.add(successor)
    obligations = set()
    inspections = 0
    for node, site in enumerate(graph.nodes):
        if site is None:
            continue
        for cell, mode, origin in incoming[node]:
            inspections += 1
            target = sites[site]["facts"][cell]
            if target and (mode == 2 or target & 2):
                obligations.add((origin, site))
    return {
        "obligations": obligations,
        "incoming": incoming,
        "facts_retained": sum(map(len, outgoing)),
        "cfg_visits": visits,
        "fact_inspections": inspections,
        "cfg_nodes": len(graph.nodes),
    }


def unfold(tree, loops, branches, coordinates=()):
    """Bounded reference execution only; callbacks supply original control values."""
    result = []
    for item in tree:
        kind = item["kind"]
        if kind == "payload":
            result.append((item["site"], coordinates))
        elif kind == "if":
            arm = "then" if branches(item["line"], coordinates) else "else"
            result += unfold(item[arm], loops, branches, coordinates)
        elif kind == "loop":
            for visit in range(loops(item["line"], coordinates)):
                result += unfold(
                    item["body"],
                    loops,
                    branches,
                    coordinates + ((item["line"], visit),),
                )
        else:
            raise RuntimeError(f"unsupported control {kind}")
    return result


def overlap(a, b):
    # This independent oracle uses reservation effects, never class IDs or
    # generated obligations. GM is one may-alias space: no relative-root compare.
    if not a["space"] or not b["space"]:
        return True
    if a["space"] != b["space"]:
        return False
    if a["space"] == 1 or not a["known"] or not b["known"]:
        return True
    if not a["bytes"] or not b["bytes"] or not a["bases"] or not b["bases"]:
        return True
    return any(
        x < y + b["bytes"] and y < x + a["bytes"]
        for x in a["bases"]
        for y in b["bases"]
    )


def conflict(a, b):
    return any(
        (x["write"] or y["write"]) and overlap(x, y)
        for x in a["effects"]
        for y in b["effects"]
    )


def check_execution(sites, word, solution):
    edges = native([sites[site]["pipe"] for site, _ in word])
    actual, proposed = set(edges), set(edges)
    class_coverage = 0
    for b, (target, _) in enumerate(word):
        for a in range(b):
            source = word[a][0]
            if conflict(sites[source], sites[target]):
                actual.add((2 * a + 1, 2 * b))
                class_coverage += 1
                require(
                    any(
                        x and y and ((x | y) & 2)
                        for x, y in zip(sites[source]["facts"], sites[target]["facts"])
                    ),
                    "access overlap omitted by class summaries",
                )
            if (source, target) in solution["obligations"]:
                proposed.add((2 * a + 1, 2 * b))
    required, _ = closure(2 * len(word), actual)
    selected, _ = closure(2 * len(word), proposed)
    require(
        all((a & b) == a for a, b in zip(required, selected)),
        "static may-analysis lost a required payload prerequisite",
    )
    excess = sum(
        bool(selected[2 * a + 1] >> (2 * b) & 1)
        and not bool(required[2 * a + 1] >> (2 * b) & 1)
        for a in range(len(word))
        for b in range(a + 1, len(word))
    )
    return {
        "payloads": len(word),
        "checked_conflicts": class_coverage,
        "excess_completion_start_pairs": excess,
        "expanded_obligations": len(proposed - edges),
    }
