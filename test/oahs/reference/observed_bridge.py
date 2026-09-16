#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""M4 original-observation bridge to the unchanged pinned v0.8 algebra.

The C++ driver exports actual constructed words and the frontend's finite original
control model. Arithmetic/path tests separately check that model's normalized-loop
qualification. This collector has no dynamic iteration bound. Its explicit test
resource limit is an inconclusive error, not an accepted compiler result.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import bridge


def collect(document: dict, ordered: bool = False, state_limit: int = 50000) -> dict:
    case, words = document['case'], document['commands']
    q = case['observed']
    keys = sorted({(a, b, k) for word in words for kind, a, b, k in word if kind in (0, 1)})
    names = {bridge.key_name(key): (f'p{key[0]}', f'p{key[1]}') for key in keys}
    base = bridge.Interface([f'p{i}' for i in case['lanes']],
                            [f'c{i}' for i in range(case['cells'])], names)
    model = bridge.OrderedInterface(base) if ordered else base
    initial = model.initial()
    seen = [set() for _ in q['sites']]
    seen[q['entry']].add(initial)
    work = deque([(q['entry'], initial)])
    states = 1
    processed = 0
    # Availability/uniformity checks are independent of the compact verdict.
    by_observation = {}
    for at, node in enumerate(q['sites']):
        observation = node['observation']
        if observation is None:
            assert not words[at], ('commands at an unavailable control site', at)
        else:
            assert q['observations'][observation]['available']
            prior = by_observation.setdefault(observation, words[at])
            assert prior == words[at], ('nonuniform original observation', at)
    while work:
        at, state = work.popleft()
        processed += 1
        node = q['sites'][at]
        try:
            after = state
            for kind, a, b, key in words[at]:
                if kind == 0:
                    command = bridge.Command('set', f'p{a}', key=bridge.key_name((a, b, key)))
                elif kind == 1:
                    command = bridge.Command('wait', f'p{b}', key=bridge.key_name((a, b, key)))
                elif kind == 2:
                    command = bridge.Command('fence', f'p{a}')
                else:
                    raise ValueError('ALL is outside this pinned issue-only adapter')
                after = model.transfer(after, command)
            phase = node['operation']
            if phase is not None:
                op = case['ops'][phase]
                reads = tuple(sorted({f'c{c}' for c, r, w in op['accesses'] if r}))
                writes = tuple(sorted({f'c{c}' for c, r, w in op['accesses'] if w}))
                after = model.transfer(after, bridge.Command('op', f"p{op['pipe']}",
                                        reads, writes, label=f'original phase {phase}'))
            if at == q['exit'] and after.live:
                raise bridge.Rejected('exit', 'event remains live at designated invocation exit')
        except bridge.Rejected as error:
            return dict(accepted=False, kind=error.kind, detail=error.detail,
                        site=at, states=states, processed=processed)
        for successor in node['successors']:
            if after not in seen[successor]:
                seen[successor].add(after)
                work.append((successor, after))
                states += 1
                if states > state_limit:
                    raise RuntimeError('inconclusive reference state-resource limit')
    return dict(accepted=True, states=states, processed=processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', required=True)
    args = parser.parse_args()
    bridge.verify_vendor()
    rows = []
    comparisons = safe_count = exact_count = state_pairs = 0
    for name, size in [('ring', 1), ('ring', 2), ('ring', 3), ('stride', 2),
                       ('refined', 1), ('refined', 2),
                       ('nested', 0), ('mixed', 0), ('independent', 0)]:
        def run(mutation):
            raw = subprocess.run([args.driver, name, str(size), str(mutation)],
                                 text=True, capture_output=True, check=True, timeout=180)
            return json.loads(raw.stdout)
        base = run(0)
        assert base['verified'], (name, 'base compact refusal')
        safe = collect(base)
        assert safe['accepted'], (name, size, 'compact acceptance contradicted', safe)
        order = collect(base, ordered=True)
        if not order['accepted']:
            assert order['kind'] == 'extra_order', (name, order)
        safe_count += 1
        exact_count += order['accepted']
        state_pairs += safe['states']
        comparisons += 1
        if name in ('ring', 'stride', 'refined'):
            assert order['accepted'], (name, size, 'lost issue-only cyclic exactness', order)
        mutations = 0
        for mutation in range(1, min(base['slots'], 8) + 1):
            trial = run(mutation)
            verdict = collect(trial)
            assert not trial['verified'] or verdict['accepted'], (name, size, mutation, verdict)
            mutations += 1
            comparisons += 1
        rows.append(dict(name=name, period=size, states=safe['states'],
                         order_exact=order['accepted'], selected_slots=base['slots'],
                         mutation_comparisons=mutations))
    print(json.dumps(dict(cases=rows, comparisons=comparisons,
                          safe_constructions=safe_count, exact_constructions=exact_count,
                          reached_state_sites=state_pairs), indent=2))


if __name__ == '__main__':
    main()
