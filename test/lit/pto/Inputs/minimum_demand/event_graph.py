# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent finite-execution oracle for native and consuming-command contracts.

This test-only oracle builds full event graphs and computes generic reachability.
It does not use production rank reduction, allocation or storage extraction.
Cells are supplied concrete physical identities; callers unfold control first.
"""

from __future__ import annotations

from dataclasses import dataclass


class InvalidPlan(ValueError):
    """A command plan violates matching, order or physical reuse."""


@dataclass(frozen=True)
class Payload:
    """One concrete instruction occurrence and its complete cell effects."""

    pipe: str
    reads: frozenset[str] = frozenset()
    writes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Command:
    """One flag generation or local barrier at an explicit pipe-stream cut."""

    kind: str
    handoff: str = ""
    src: str = ""
    dst: str = ""
    event_id: int = 0


class Graph:
    """Finite event graph with cycle detection and unrestricted reachability."""

    def __init__(self):
        self.edges = {}

    def node(self, name):
        self.edges.setdefault(name, set())

    def add(self, source, target):
        self.node(source)
        self.node(target)
        self.edges[source].add(target)

    def closure(self):
        result = {}
        for source in self.edges:
            reached = set()
            pending = list(self.edges[source])
            while pending:
                target = pending.pop()
                if target not in reached:
                    reached.add(target)
                    pending.extend(self.edges[target])
            if source in reached:
                raise InvalidPlan("cyclic command graph")
            result[source] = reached
        return result


def start(site):
    """Return a payload start identity."""
    return ("payload", site, "start")


def finish(site):
    """Return a payload completion identity."""
    return ("payload", site, "finish")


def add_native(graph, payloads):
    """Starts and physical completions are independently ordered per pipe."""
    previous = {}
    for site, payload in enumerate(payloads):
        graph.add(start(site), finish(site))
        if payload.pipe in previous:
            prior = previous[payload.pipe]
            graph.add(start(prior), start(site))
            graph.add(finish(prior), finish(site))
        previous[payload.pipe] = site


def required_graph(payloads, prerequisites):
    """Derive every concrete RAW, WAR and WAW pair, without lifetime kills."""
    graph = Graph()
    add_native(graph, payloads)
    for consumer, later in enumerate(payloads):
        for source, earlier in enumerate(payloads[:consumer]):
            if (earlier.writes & (later.reads | later.writes)
                    or earlier.reads & later.writes):
                graph.add(finish(source), start(consumer))
    for source, consumer in prerequisites:
        if not (0 <= source < consumer < len(payloads)):
            raise InvalidPlan("invalid supplied prerequisite")
        graph.add(finish(source), start(consumer))
    return graph


def add_stream(graph, pipe, stream, payloads, endpoints):
    """Apply launch order, gating and publication-prefix contracts verbatim."""
    launches = []
    completions = []
    gate = None
    for position, item in enumerate(stream):
        if isinstance(item, int):
            if item < 0 or item >= len(payloads) or payloads[item].pipe != pipe:
                raise InvalidPlan("payload in wrong pipe stream")
            launch, complete = start(item), finish(item)
        else:
            launch = ("command", pipe, position, "start")
            complete = ("command", pipe, position, "finish")
            add_command(graph, pipe, item, complete, completions, endpoints)
        graph.add(launch, complete)
        if launches:
            graph.add(launches[-1], launch)
        if gate is not None:
            graph.add(gate, launch)
        if isinstance(item, Command) and item.kind in {"wait", "barrier"}:
            gate = complete
        launches.append(launch)
        completions.append(complete)


def add_command(graph, pipe, command, complete, prefix, endpoints):
    """SET/barrier observe all prior completions; WAIT observes its SET."""
    if command.kind not in {"set", "wait", "barrier"}:
        raise InvalidPlan("unknown command kind")
    if command.kind in {"set", "barrier"}:
        for earlier in prefix:
            graph.add(earlier, complete)
    if command.kind == "barrier":
        return
    owner = command.src if command.kind == "set" else command.dst
    if owner != pipe or command.src == command.dst:
        raise InvalidPlan("flag in wrong directed pipe pool")
    identity = (command.kind, command.handoff)
    if identity in endpoints:
        raise InvalidPlan("duplicate logical endpoint")
    endpoints[identity] = (complete, command)


def match(graph, endpoints, pools):
    """Match logical endpoints and retain separate physical pool identities."""
    generations = {}
    handoffs = {handoff for _, handoff in endpoints}
    for handoff in sorted(handoffs):
        if ("set", handoff) not in endpoints or ("wait", handoff) not in endpoints:
            raise InvalidPlan("unmatched logical handoff")
        publication, set_command = endpoints[("set", handoff)]
        consumption, wait_command = endpoints[("wait", handoff)]
        key = (set_command.src, set_command.dst, set_command.event_id)
        if key != (wait_command.src, wait_command.dst, wait_command.event_id):
            raise InvalidPlan("incorrect physical matching")
        if set_command.event_id not in pools.get(key[:2], set()):
            raise InvalidPlan("ineligible or reserved physical ID")
        graph.add(publication, consumption)
        generations.setdefault(key, []).append((publication, consumption))
    return generations


def check_reuse(closure, generations):
    """Require causal consumption before republication, without adding edges."""
    for group in generations.values():
        for index, (publication, consumption) in enumerate(group):
            for other_publication, other_consumption in group[index + 1:]:
                forward = other_publication in closure[consumption]
                backward = publication in closure[other_consumption]
                if not forward and not backward:
                    raise InvalidPlan("reuse lacks consumption-before-republication")


def check(payloads, streams, pools, prerequisites=()):
    """Validate total matching, closed keys, reuse and exact prerequisite profile."""
    graph = Graph()
    add_native(graph, payloads)
    endpoints = {}
    seen = []
    for pipe, stream in streams.items():
        seen.extend(item for item in stream if isinstance(item, int))
        add_stream(graph, pipe, stream, payloads, endpoints)
    if sorted(seen) != list(range(len(payloads))):
        raise InvalidPlan("payload missing or repeated")
    generations = match(graph, endpoints, pools)
    actual = graph.closure()
    check_reuse(actual, generations)
    required = required_graph(payloads, prerequisites).closure()
    for source in range(len(payloads)):
        for consumer in range(len(payloads)):
            wanted = start(consumer) in required[finish(source)]
            supplied = start(consumer) in actual[finish(source)]
            if wanted and not supplied:
                raise InvalidPlan("missing storage prerequisite")
            if supplied and not wanted:
                raise InvalidPlan("added payload prerequisite")
    return actual


def minimum_generators(payloads, generators=None):
    """Compute nonnative covers by generic closure and intermediate-node search.

    This deliberately expensive oracle is independent of completion-rank scans.
    An optional generator set lets tests compare equivalent generating graphs.
    """
    if generators is None:
        graph = required_graph(payloads, ())
        candidates = {(source, consumer) for source in range(len(payloads))
                      for consumer in range(len(payloads))
                      if start(consumer) in graph.edges[finish(source)]}
    else:
        candidates = set(generators)
        graph = Graph()
        add_native(graph, payloads)
        for source, consumer in candidates:
            if not (0 <= source < consumer < len(payloads)):
                raise InvalidPlan("invalid generator endpoints")
            graph.add(finish(source), start(consumer))
    closure = graph.closure()
    retained = set()
    for source, consumer in candidates:
        target = start(consumer)
        intermediate = any(target in closure[node]
                           for node in closure[finish(source)] if node != target)
        if not intermediate:
            retained.add((source, consumer))
    return retained
