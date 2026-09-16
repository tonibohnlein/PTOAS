#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Production must-causal frontier versus the unchanged exact live-port model.

The v0.17 draft's checks/compact/causal_interface.py is byte-identical to the
already pinned v0.8 module. No new model or archive is vendored. Local joins
are checked against every member, and accepted fixed plans against a collecting
fixed point of original control (not bounded loop unrolling).
"""
from __future__ import annotations
import argparse
from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess

from bridge import (Command, Interface, Rejected, VENDOR, KINDS, control,
                    key_name, leaf, seq, loop, choice, op, verify_vendor)

REFERENCE_SHA256 = "543a7c796acf602d068d3b374c287b224104ddbc5d1d71a785224cc8db5162d1"
COUNTS = dict(primitive_trials=0, joins=0, accepted_transfers=0,
              rejected_transfers=0, fact_implications=0, fixed_plans=0,
              compact_accepted=0, reference_accepted=0, reference_states=0)


def require(value, detail):
    if not value:
        raise AssertionError(detail)


def encode(case, actions=None):
    values = [1, int(actions is not None), len(case['operations']), case['cells'], len(case['keys'])]
    for key in case['keys']:
        values.extend(key)
    for operation in case['operations']:
        effects = operation['accesses']
        values.extend([operation['pipe'], len(effects)])
        for effect in effects:
            values.extend(map(int, effect))

    def region(r):
        children = r.get('children', [])
        values.extend([KINDS[r['kind']], r.get('operation', 0), int(r.get('zero_trip_possible', False)), len(children)])
        for child in children:
            region(child)
    region(case.get('body', seq()))
    if actions is not None:
        values.append(len(actions))
        for action in actions:
            values.extend(action)
    else:
        for word in case['commands']:
            values.append(len(word))
            for command in word:
                values.extend(command)
    return ' '.join(map(str, values)) + '\n'


def run(driver, case, actions=None):
    result = subprocess.run([str(driver)], input=encode(case, actions), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    require(result.returncode == 0, (result.returncode, result.stderr))
    return json.loads(result.stdout)


def model_for(case):
    return Interface([f'p{i}' for i in range(7)], [f'c{i}' for i in range(case['cells'])],
                     {key_name(k): (f'p{k[0]}', f'p{k[1]}') for k in sorted(case['keys'])})


def payload(operation):
    return Command('op', f"p{operation['pipe']}",
                   tuple(f'c{c}' for c, r, _ in operation['accesses'] if r),
                   tuple(f'c{c}' for c, _, w in operation['accesses'] if w))


def endpoint(command):
    kind, a, b, key = command
    if kind == 2:
        return Command('fence', f'p{a}')
    return Command('set' if kind == 0 else 'wait', f'p{a if kind == 0 else b}', key=key_name((a, b, key)))


def mask(words):
    return sum(value << (64 * i) for i, value in enumerate(words))


def compare(facts, concrete, model, location):
    require(facts is not None, ('reachable reference state omitted', location))
    require(len(facts['reach']) == model.m and len(facts['history']) == len(model.classes)
            and len(facts['events']) == len(model.keys), ('truncated frontier report', location))

    def fact(ok, description):
        COUNTS['fact_implications'] += 1
        require(ok, (location, description))
    for i, row in enumerate(facts['reach']):
        claimed = mask(row)
        fact(claimed & concrete.reach[i] == claimed, ('invented causal edge', i))
    for i, (h, family) in enumerate(zip(facts['history'], concrete.history)):
        if family:
            fact(h is not None, ('lost access class', i))
            claim = mask(h)
            for signature in family:
                fact(claim & signature == claim, ('invented completion successor', i))
    for i, event in enumerate(facts['events']):
        full = bool(concrete.live & (1 << i))
        fact(event['occupancy'] & (2 if full else 1), ('incorrect may balance', i))
        publication = model.pi[f'S:{model.key_names[i]}']
        if event['occupancy'] != 2:
            fact(all(not mask(row) & (1 << publication) for row in facts['reach']), ('optional S in relation', i))
            fact(all(h is None or not mask(h) & (1 << publication) for h in facts['history']), ('optional S in history', i))


def primitive_trials(driver):
    rng = random.Random(170217)
    operations = [op(p, c, bool(mode & 1), bool(mode & 2))
                  for p in range(3) for c in range(2) for mode in (1, 2, 3)]
    keys = [(a, b, k) for a in range(3) for b in range(3) if a != b for k in range(2)]
    # Cross a 64-bit word boundary without changing the exercised payload core.
    keys += [(a, b, 0) for a in range(7) for b in range(7)
             if a != b and (a >= 3 or b >= 3)][:18]
    case = dict(cells=2, operations=operations, keys=sorted(keys))
    commands = [[2, p, p, 0] for p in range(3)]
    for a, b, k in keys[:12]:
        commands.extend([[0, a, b, k], [1, a, b, k]])
    actions = []

    def random_action(source):
        if rng.randrange(2):
            return [0, source, rng.randrange(len(operations))]
        return [1, source, *rng.choice(commands), rng.randrange(len(operations) + 1), rng.randrange(3)]

    for i in range(1400):
        actions.append(random_action(0 if i % 31 == 0 else i))
    for _ in range(160):
        actions.append([2, rng.randrange(1401), rng.randrange(1401)])
        actions.append([2, len(actions), rng.randrange(1401)])
        for _ in range(8):
            actions.append(random_action(len(actions)))
        actions.append([3, len(actions)])
    reports = run(driver, case, actions)
    model = model_for(case)
    exact = [{model.initial()}]
    # Current-publication identities are separate from reference causal facts.
    bindings = [{tuple(None for _ in model.keys)}]
    compare(reports[0], model.initial(), model, 'initial')
    for index, (action, report) in enumerate(zip(actions, reports[1:]), 1):
        kind, source, *args = action
        incoming = exact[source]
        before = reports[source]['state'] if source else reports[0]
        if kind == 2:
            COUNTS['joins'] += 1
            require(report['applied'], 'valid immutable states must join')
            after = incoming | exact[args[0]]
            bound = bindings[source] | bindings[args[0]]
        else:
            COUNTS['primitive_trials'] += 1
            command = payload(operations[args[0]]) if kind == 0 else endpoint(args[:4]) if kind == 1 else None
            after = set()
            rejected = False
            for state in incoming:
                try:
                    if kind == 3:
                        if state.live:
                            raise Rejected('exit', 'live publication')
                        result = state
                    else:
                        result = model.transfer(state, command)
                    after.add(result)
                except Rejected:
                    rejected = True
            if report['applied']:
                COUNTS['accepted_transfers'] += 1
                require(not rejected, ('accepted primitive contradicted by reference', index, action))
                bound = set()
                for entry in bindings[source]:
                    entry = list(entry)
                    if kind == 1 and command.kind in ('set', 'wait'):
                        entry[model.ki[command.key]] = tuple(args[4:6]) if command.kind == 'set' else None
                    bound.add(tuple(entry))
            else:
                COUNTS['rejected_transfers'] += 1
                require(report['state'] == before, ('rejected primitive changed state', index))
                after = incoming
                bound = bindings[source]
            if index <= 1400:
                require(bool(report['applied']) == (not rejected), ('singleton decision mismatch', index))
            if kind == 0:
                uncovered = {component for state in incoming for component in model.payload_residual(state, command)}
                reported = {(f'c{cell}', f'p{engine}', 'W' if write else 'R')
                            for cell, engine, write, *_ in report['residuals']}
                require(uncovered <= reported, ('missing residual access class', index))
                if index <= 1400:
                    require(uncovered == reported, ('singleton residual mismatch', index))
        exact.append(after)
        bindings.append(bound)
        for state in after:
            compare(report['state'], state, model, ('primitive', index))
        for e, event in enumerate(report['state']['events']):
            expected = {entry[e] for entry in bound if entry[e] is not None}
            require(set(map(tuple, event['publishers'])) == expected, ('publication alternatives', index, e))


def collect(case, report):
    model = model_for(case)
    edges, entry = control(case)  # independently built from the original tree
    seen = [set() for _ in edges]
    seen[entry].add(model.initial())
    queue = deque([(entry, model.initial())])
    n = len(case['operations'])
    while queue:
        at, state = queue.popleft()
        COUNTS['reference_states'] += 1
        try:
            if at <= n:
                if report['accepted']:
                    compare(report['cuts'][at][0], state, model, (case['name'], at, 'incoming'))
                for c in case['commands'][at]:
                    state = model.transfer(state, endpoint(c))
                if report['accepted']:
                    compare(report['cuts'][at][1], state, model, (case['name'], at, 'before'))
                if at < n:
                    state = model.transfer(state, payload(case['operations'][at]))
                if report['accepted']:
                    compare(report['cuts'][at][2], state, model, (case['name'], at, 'outgoing'))
                if at == n and state.live:
                    raise Rejected('exit', 'live publication')
        except Rejected:
            return False
        for successor in edges[at]:
            if state not in seen[successor]:
                seen[successor].add(state)
                queue.append((successor, state))
    return True


def fixed_cases():
    keys = [[a, b, 0] for a in range(3) for b in range(3) if a != b]

    def case(name, ops, body, words):
        return dict(name=name, cells=2, keys=keys, operations=ops, body=body, commands=words)

    yield case('alternative-publishers', [op(0, 0, write=True), op(0, 1, read=True),
                                         op(0, 0, write=True), op(0, 1, read=True), op(1, 0, read=True)],
               seq(choice(seq(leaf(0), leaf(1)), seq(leaf(2), leaf(3))), leaf(4)),
               [[], [[0, 0, 1, 0]], [], [[0, 0, 1, 0]], [[1, 0, 1, 0]], []])
    yield case('recurring-release', [op(0, 0, write=True), op(1, 0, read=True), op(0, 1, read=True)],
               loop(seq(leaf(0), leaf(1), leaf(2))),
               [[], [[0, 0, 1, 0], [1, 0, 1, 0]], [[0, 1, 0, 0], [1, 1, 0, 0]], []])
    yield case('while-before', [op(0, 0, write=True), op(1, 0, read=True)],
               seq({'kind': 'while', 'children': [leaf(0), seq()]}, leaf(1)),
               [[[2, 0, 0, 0]], [[0, 0, 1, 0], [1, 0, 1, 0]], []])
    yield case('zero-trip', [op(0, 0, write=True), op(1, 0, read=True)],
               seq(loop(leaf(0)), leaf(1)), [[[0, 0, 1, 0]], [[1, 0, 1, 0]], []])
    rng = random.Random(170218)
    for index in range(80):
        ops = []
        words = []
        for _ in range(6):
            engine, cell, role = rng.randrange(3), rng.randrange(2), rng.randrange(1, 4)
            ops.append(op(engine, cell, bool(role & 1), bool(role & 2)))
            word = [[2, engine, engine, 0]]
            # Two actual circulation rounds make every prior prefix available
            # at every payload engine. The fixed return paths also acknowledge
            # each previous consumption; reversing roles on the same keys in
            # arbitrary closed pairs would not establish that property.
            for _ in range(2):
                for source in range(3):
                    observer = (source + 1) % 3
                    word.extend([[0, source, observer, 0], [1, source, observer, 0]])
            words.append(word)
        words.append([])
        trees = [seq(*(leaf(i) for i in range(6))),
                 loop(seq(*(leaf(i) for i in range(6)))),
                 loop(seq(choice(seq(leaf(0), leaf(1)), seq(leaf(2), leaf(3))), leaf(4), leaf(5))),
                 {'kind': 'while', 'children': [seq(leaf(0), leaf(1)), loop(seq(*(leaf(i) for i in range(2, 6))))]}]
        yield case(f'generated-{index}', ops, trees[index % 4], words)


def fixed_trials(driver):
    for original in fixed_cases():
        variants = [original]
        # Endpoint and fence deletion: include every deletion in named cases,
        # and a reproducible subset in the generated population.
        sites = [(at, i) for at, word in enumerate(original['commands']) for i in range(len(word))]
        for at, i in sites if not original['name'].startswith('generated') else sites[::11]:
            mutation = copy.deepcopy(original)
            mutation['name'] += f'-delete-{at}-{i}'
            del mutation['commands'][at][i]
            variants.append(mutation)
        for case in variants:
            report = run(driver, case)
            require(report['complete'], (case['name'], report))
            reference = collect(case, report)
            require(not report['accepted'] or reference, ('unsafe fixed plan', case['name']))
            if not report['accepted']:
                require(not report['cuts'], 'rejected plan exported partial invariants')
            if case is original and case['name'] != 'zero-trip':
                require(report['accepted'], ('lost positive coverage', case['name'], report))
            COUNTS['fixed_plans'] += 1
            COUNTS['compact_accepted'] += bool(report['accepted'])
            COUNTS['reference_accepted'] += reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    args = parser.parse_args()
    verify_vendor()
    require(hashlib.sha256((VENDOR / 'causal_interface.py').read_bytes()).hexdigest() == REFERENCE_SHA256,
            'reference no longer matches v0.17 ordinary core')
    primitive_trials(args.driver)
    fixed_trials(args.driver)
    print(json.dumps(COUNTS, sort_keys=True))


if __name__ == '__main__':
    main()
