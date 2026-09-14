# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded control-flow predecessors of a publication-refusal consumer.

These are intrainvocation MAY-reaching occurrences, not last-writer or causality proofs.
Unknown branch predicates, partial writes and publication kills are not inferred.
"""
from collections import defaultdict, deque


class IncompleteFlow(ValueError):
    pass


class WriterFlow:
    def __init__(self, report, limit=500000):
        self.work, self.limit = 0, limit
        self.graph, self.points, self.cache = defaultdict(list), {}, {}
        self.predecessors, self.backedges = {}, {}
        self.failure = None
        try:
            self.charge(len(report['nodes']) + len(report['accesses']))
            nodes = {n['id']: n for n in report['nodes']}
            if len(nodes) != len(report['nodes']) or report.get('report_exhausted'):
                raise IncompleteFlow('incomplete-report')
            phases = defaultdict(set)
            for access in report['accesses']:
                node = access.get('node')
                if node not in nodes:
                    raise IncompleteFlow('unmapped-operation')
                if nodes[node]['kind'] == 1:
                    phase = access.get('macro_phase')
                    if phase is None:
                        raise IncompleteFlow('unmapped-macro-phase')
                    phases[node].add(phase)
            entries, exits = {}, {}
            parents = set()
            for node in report['nodes']:
                ident, kind, children = node['id'], node['kind'], node['children']
                self.charge(4 + 5 * len(children) + len(phases[ident]))
                if kind not in range(6) or any(c not in entries or c >= ident for c in children):
                    raise IncompleteFlow('invalid-structured-tree')
                if any(c in parents for c in children) or len(children) != len(set(children)):
                    raise IncompleteFlow('shared-structural-child')
                parents.update(children)
                entry, exit = (ident, 'entry'), (ident, 'exit')
                entries[ident], exits[ident] = entry, exit

                def edge(a, b, back=False):
                    self.graph[a].append((b, back))
                    if back:
                        self.backedges[a, b] = ident
                if kind in (0, 1):
                    if children:
                        raise IncompleteFlow('physical-node-has-children')
                    previous = entry
                    for phase in sorted(phases[ident]) if kind == 1 else [None]:
                        point = (ident, phase)
                        self.points[point] = point
                        edge(previous, point)
                        previous = point
                    edge(previous, exit)
                elif kind == 2:
                    previous = entry
                    for child in children:
                        edge(previous, entries[child])
                        previous = exits[child]
                    edge(previous, exit)
                elif kind == 3 and len(children) == 2:
                    for child in children:
                        edge(entry, entries[child])
                        edge(exits[child], exit)
                elif kind == 4 and len(children) == 1:
                    body = children[0]
                    edge(entry, exit)  # zero-trip exit stays possible
                    edge(entry, entries[body])
                    edge(exits[body], exit)
                    edge(exits[body], entries[body], True)
                elif kind == 5 and len(children) == 2:
                    before, after = children
                    edge(entry, entries[before])
                    edge(exits[before], exit)
                    edge(exits[before], entries[after])
                    edge(exits[after], entries[before], True)
                else:
                    raise IncompleteFlow('unsupported-structured-shape')
            self.charge(2 * len(nodes))
            if len(set(nodes) - parents) != 1:
                raise IncompleteFlow('disconnected-structured-tree')
        except IncompleteFlow as error:
            self.failure = str(error)

    def charge(self, count):
        if count > self.limit - self.work:
            raise IncompleteFlow('analysis-exhausted')
        self.work += count

    def relation(self, writer, reader):
        if self.failure:
            return self.failure
        source = (writer.get('node'), writer.get('macro_phase'))
        target = (reader.get('node'), reader.get('macro_phase'))
        if source not in self.points or target not in self.points:
            return 'unmapped-operation-or-phase'
        try:
            self.charge(1)
            if source not in self.cache:
                # Start after the write. Same-operation reads precede writes;
                # only a real backedge can make this writer reach itself.
                self.charge(1 + len(self.graph[source]))
                queue = deque((point, back, None) for point, back in self.graph[source])
                seen, predecessors = set(), {}
                while queue:
                    self.charge(1)
                    point, back, previous = queue.popleft()
                    if (point, back) in seen:
                        continue
                    self.charge(2 + len(self.graph[point]))
                    seen.add((point, back))
                    predecessors[point, back] = previous
                    for following, loop_edge in self.graph[point]:
                        queue.append((following, back or loop_edge, (point, back)))
                self.cache[source] = seen
                self.predecessors[source] = predecessors
            found = self.cache[source]
            if (target, False) in found:
                return 'may-reach-without-backedge'
            if (target, True) in found:
                return 'may-reach-via-backedge'
            return 'cannot-precede-within-invocation'
        except IncompleteFlow as error:
            self.failure = str(error)
            return self.failure

    def witness(self, writer, reader):
        """One bounded CFG path, not predicate feasibility or reaching generation."""
        relation = self.relation(writer, reader)
        result = dict(relation=relation, scope='Within one invocation; CFG path only, not a causal dependency.')
        if not relation.startswith('may-reach-'):
            return result
        source = (writer.get('node'), writer.get('macro_phase'))
        target = (reader.get('node'), reader.get('macro_phase'))
        step = (target, relation == 'may-reach-via-backedge')
        try:
            path = []
            while step is not None:
                self.charge(2)
                path.append(step[0])
                step = self.predecessors[source][step]
            self.charge(2 * len(path) + 1)
            path.append(source)
            path.reverse()
            result['path'] = path
            result['backedge_owners'] = [self.backedges[a, b] for a, b in zip(path, path[1:])
                                         if (a, b) in self.backedges]
        except IncompleteFlow as error:
            self.failure = str(error)
            result = dict(relation=self.failure, scope=result['scope'])
        return result
