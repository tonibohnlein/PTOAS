#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Test-only bridge from the actual C++ compact checker to pinned draft v0.8.

Uses the original causal_interface.py/order_interface.py without editing them.
Collects complete reference states at static original control sites (no trip
bound). Unsupported contracts and resource interruption never count as passes.
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
import sys
import tempfile
import shutil

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor' / 'v08'
sys.path.insert(0, str(VENDOR))
from causal_interface import Command, Interface, Rejected  # noqa: E402
from order_interface import OrderedInterface  # noqa: E402
from check_emitted import check as check_emitted  # noqa: E402
from full_history_previous import FullHistory  # noqa: E402

LANES = 7
KINDS = {'sequence': 0, 'choice': 1, 'for': 2, 'while': 3, 'operation': 4}

def verify_vendor(root: Path = VENDOR) -> dict:
    manifest = json.loads((root / 'PROVENANCE.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'pinned reference source/artifact changed: {name}')
    return manifest

def encode(case: dict) -> str:
    """Versioned, length-prefixed numeric interchange; the driver is test-only."""
    if case.get('schema') != 'oahs.reference-case.v1':
        raise ValueError('unknown reference-case interchange version')
    ops = case['operations']
    values = [1, len(ops), case['cells']]
    for name in ('supported', 'barriers', 'synchronous'):
        values += [int(q in case.get(name, [])) for q in range(LANES)]
    values += [int(case.get('barrier_all', False)), int(case.get('retire', False))]
    for name in ('keys', 'reservations'):
        entries = case.get(name, [])
        values += [len(entries)]
        for entry in entries:
            if len(entry) != 3:
                raise ValueError('event identity must be [source, observer, number]')
            values += entry
    for op in ops:
        effects = op.get('accesses', [])
        values += [op['pipe'], len(effects)]
        for effect in effects:
            values += effect
    def region(r):
        kids = r.get('children', [])
        values.extend([KINDS[r['kind']], r.get('operation', 0),
                       int(r.get('zero_trip_possible', False)), len(kids)])
        for child in kids:
            region(child)
    region(case.get('body', {'kind': 'sequence'}))
    values += [int(case.get('construct', False))]
    if not case.get('construct', False):
        words = case.get('commands', [[] for _ in range(len(ops) + 1)])
        if len(words) != len(ops) + 1:
            raise ValueError('wrong number of command cuts')
        for word in words:
            values += [len(word)]
            for command in word:
                if len(command) != 4:
                    raise ValueError('command must be [kind, source, observer, key]')
                values += command
    if any(type(x) is not int or x < 0 or x > (1 << 63)-1 for x in values):
        raise ValueError('interchange requires nonnegative integral fields')
    return ' '.join(map(str, values)) + '\n'

def compact(driver: Path, case: dict) -> dict:
    run = subprocess.run([str(driver)], input=encode(case), text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if run.returncode:
        raise RuntimeError(f'compact driver failed ({run.returncode}): {run.stderr}')
    return json.loads(run.stdout)

def key_name(key):
    return ':'.join(map(str, key))

def commands_at(case, report, cut, tagged=False):
    out = []
    for kind, source, observer, key in report['commands'][cut]:
        if kind == 0:
            out.append(Command('set', f'p{source}', key=key_name((source, observer, key))))
        elif kind == 1:
            out.append(Command('wait', f'p{observer}', key=key_name((source, observer, key))))
        elif kind == 2:
            out.append(Command('fence', f'p{source}'))
        else:
            raise ValueError('ALL is outside the pinned issue-only reference adapter')
    payload = None
    if cut < len(case['operations']):
        op = case['operations'][cut]
        reads = sorted({f'c{c}' for c, r, w in op.get('accesses', []) if r})
        writes = sorted({f'c{c}' for c, r, w in op.get('accesses', []) if w})
        if tagged:
            # Private read-only ghost cell: observes this phase's completion;
            # never conflicts and adds no edge to an actual payload endpoint.
            reads.append(f'@phase{cut}')
        payload = Command('op', f"p{op['pipe']}", tuple(reads), tuple(writes), label=f'phase{cut}')
    return out, payload

def control(case):
    """Independent construction from Region, not C++ Control.h's edge table."""
    n = len(case['operations'])
    edges = [[] for _ in range(n + 1)]
    def node():
        edges.append([])
        return len(edges)-1
    def build(r, continuation):
        kind = r['kind']
        children = r.get('children', [])
        if kind == 'operation':
            at = r['operation']
            edges[at] = [continuation]
            return at
        if kind == 'sequence':
            result = continuation
            for c in reversed(children):
                result = build(c, result)
            return result
        if kind == 'choice':
            at = node()
            edges[at] = [build(c, continuation) for c in children]
            return at
        if kind == 'for':
            header = node()
            edges[header] = [build(children[0], header), continuation]
            return header
        if kind == 'while':
            decision = node()
            before = build(children[0], decision)
            after = build(children[1], before)
            edges[decision] = [after, continuation]
            return before
        raise ValueError(kind)
    body = case.get('body', {'kind': 'sequence'})
    if body['kind'] == 'sequence' and not body.get('children'):
        for at in range(n):
            edges[at] = [at+1]
        entry = 0
    else:
        entry = build(body, n)
    return edges, entry

def unsupported(case, report):
    if case.get('synchronous'):
        return 'synchronous-engine contract has no pinned reference adapter'
    if case.get('retire'):
        return 'DrainAllAtReturn is outside this issue-only/fresh-entry adapter'
    if any(c[0] == 3 for word in report['commands'] for c in word):
        return 'actual ALL command has no pinned reference adapter'
    if case.get('exclusive_cells') or case.get('resources') or case.get('visibility'):
        return 'typed/exclusive/visibility effects have no pinned reference adapter'
    for op in case['operations']:
        if op.get('resources') or op.get('visibility') or op.get('internal_transfers'):
            return 'compound operation contract has no pinned reference adapter'
    return None

def compare_facts(model, state, facts, case, report, location, counts):
    if facts is None:
        return
    def demand(ok, what):
        counts['fact_implications'] += 1
        if not ok:
            raise AssertionError(f'compact fact contradicted by exact reference at {location}: {what}')
    for observer in case['supported']:
        a = model.pi[f'A:p{observer}']
        for phase, op in enumerate(case['operations']):
            family = state.history[model.ci[f'@phase{phase}', f"p{op['pipe']}", 'R']]
            if not facts['pending'][observer][phase]:
                demand(all(s & (1 << a) for s in family), f'completion phase={phase} observer={observer}')
    for index, identity in enumerate(report['keys']):
        key = key_name(identity)
        e = facts['events'][index]
        live = bool(state.live & (1 << model.ki[key]))
        demand(e['o'] & (2 if live else 1), f'occupancy {key}')
        d = model.pi[f'D:{key}']
        for observer in case['supported']:
            if e['k'] & (1 << observer):
                demand(state.reach[d] & (1 << model.pi[f'A:p{observer}']), f'consumption {key} at {observer}')
        if not e['v']:
            continue
        demand(live, f'live receipt {key}')
        s = model.pi[f'S:{key}']
        for phase, op in enumerate(case['operations']):
            family = state.history[model.ci[f'@phase{phase}', f"p{op['pipe']}", 'R']]
            if not e['u'][phase]:
                demand(all(sig & (1 << s) for sig in family), f'receipt {key} phase={phase}')
        for j, carried in enumerate(e['j']):
            if carried:
                other = key_name(report['keys'][j])
                demand(state.reach[model.pi[f'D:{other}']] & (1 << s), f'carried consumption {other} by {key}')

def collect(case, report, *, ordered=False, compare=True, state_limit=None):
    engines = [f'p{i}' for i in case['supported']]
    n = len(case['operations'])
    cells = [f'c{i}' for i in range(case['cells'])]
    if not ordered:
        cells += [f'@phase{i}' for i in range(n)]
    keys = {key_name(k): (f'p{k[0]}', f'p{k[1]}') for k in report['keys']}
    base = Interface(engines, cells, keys)
    model = OrderedInterface(base) if ordered else base
    edges, entry = control(case)
    start = model.initial()
    if state_limit is not None and state_limit < 1:
        return dict(accepted=None, kind='inconclusive_state_limit', states=1, processed=0, fact_implications=0)
    seen = [set() for _ in edges]
    seen[entry].add(start)
    queue = deque([(entry, start)])
    previous = {(entry, start): None}
    counts = {'states': 1, 'processed': 0, 'fact_implications': 0}
    while queue:
        at, state = queue.popleft()
        counts['processed'] += 1
        try:
            after = state
            if at <= n:
                cut = report['cuts'][at] if report['complete'] else None
                if compare and not ordered and cut:
                    if not cut['reachable']:
                        raise AssertionError(f'reference reached compact-unreachable cut {at}')
                    compare_facts(base, after, cut['incoming'], case, report, (at,'incoming'), counts)
                word, payload = commands_at(case, report, at, tagged=not ordered)
                for c in word:
                    after = model.transfer(after, c)
                if compare and not ordered and cut:
                    compare_facts(base, after, cut['before'], case, report, (at,'before'), counts)
                if payload:
                    after = model.transfer(after, payload)
                if compare and not ordered and cut:
                    compare_facts(base, after, cut['outgoing'], case, report, (at,'outgoing'), counts)
                if at == n and after.live:
                    raise Rejected('exit', 'notification remains live at closed exit')
        except Rejected as error:
            path = []
            cursor = (at, state)
            while cursor is not None:
                path.append(cursor[0])
                cursor = previous[cursor]
            return dict(accepted=False, kind=error.kind, detail=error.detail,
                        witness=path[::-1], **counts)
        for dest in edges[at]:
            if after not in seen[dest]:
                seen[dest].add(after)
                previous[(dest,after)] = (at,state)
                queue.append((dest, after))
                counts['states'] += 1
                if state_limit is not None and counts['states'] > state_limit:
                    return dict(accepted=None, kind='inconclusive_state_limit', **counts)
    return dict(accepted=True, kind='accepted', **counts)

def bridge(driver, case, *, state_limit=None):
    report = compact(driver, case)
    reason = unsupported(case, report)
    if reason:
        return {'name': case['name'], 'status': 'unsupported_contract', 'reason': reason}
    if not report['complete']:
        return {'name': case['name'], 'status': 'invalid_compact_input', 'reason': report['reason']}
    ref = collect(case, report, state_limit=state_limit)
    if ref['accepted'] is None:
        return {'name': case['name'], 'status': 'inconclusive', 'reference': ref}
    if report['verified'] and not ref['accepted']:
        raise AssertionError(f"compact accepted but reference rejected {case['name']}: {ref}")
    quality = collect(case, report, ordered=True, compare=False, state_limit=state_limit) if ref['accepted'] else None
    if quality and quality['accepted'] is None:
        return {'name': case['name'], 'status': 'inconclusive', 'reference': ref, 'quality': quality}
    if quality and not quality['accepted'] and quality['kind'] != 'extra_order':
        raise AssertionError(f'paired/safety disagreement: {quality}')
    return {'name': case['name'], 'status': 'checked', 'constructed': report['constructed'],
            'compact_accepted': report['verified'], 'reference': ref,
            'order_exact': quality['accepted'] if quality else None,
            'quality': quality, 'commands': report['commands']}

def op(pipe, cell=0, read=False, write=False):
    return {'pipe':pipe, 'accesses':[[cell,int(read),int(write)]] if read or write else []}

def leaf(i):return {'kind':'operation','operation':i}
def seq(*rs):return {'kind':'sequence','children':list(rs)}
def loop(r):return {'kind':'for','children':[r],'zero_trip_possible':True}
def choice(a,b):return {'kind':'choice','children':[a,b]}
def event(kind,a,b,k=0):return [kind,a,b,k]
def base(name,ops):
    return {'schema':'oahs.reference-case.v1','name':name,'cells':3,'supported':[0,1,2],
            'barriers':[0,1,2],'keys':[[a,b,k] for a in range(3) for b in range(3) if a!=b for k in range(2)],
            'operations':ops,'construct':True}

def cases():
    out=[]
    p=base('payload_free_relay',[op(0,write=True),op(2,read=True)])
    p['keys']=[[0,1,0],[1,2,0]];out.append((p,True,True))
    p=base('joint_WAW_WAR',[op(0,write=True),op(0,1,write=True),op(1,read=True),op(0,write=True),op(0,1,read=True)])
    out.append((p,True,None))
    p=base('alternative_publishers',[op(0,write=True),op(0,1,write=True),op(0,write=True),op(0,2,write=True),op(1,read=True)])
    p['body']=seq(choice(seq(leaf(0),leaf(1)),seq(leaf(2),leaf(3))),leaf(4));out.append((p,True,True))
    p=base('repeated_storage',[op(0,write=True),op(1,read=True)])
    p['body']=loop(seq(leaf(0),leaf(1)));p['keys']=[[0,1,0],[1,0,0]];out.append((p,True,None))
    p=base('mixed_writer_recurrence',[op(0,write=True),op(1,read=True),op(2,read=True,write=True),op(1,read=True)])
    p['body']=loop(seq(leaf(0),choice(leaf(1),leaf(2)),leaf(3)));out.append((p,True,True))
    p=base('early_exact',[op(0,write=True),op(0,1,write=True),op(1,read=True)])
    p['construct']=False;p['commands']=[[],[event(0,0,1)],[event(1,0,1)],[]];out.append((p,True,True))
    q=copy.deepcopy(p);q['name']='late_safe_coarser';q['commands']=[[],[],[event(0,0,1),event(1,0,1)],[]];out.append((q,True,False))
    p=base('stale_receipt',[op(0,write=True),op(0,write=True),op(1,read=True)])
    p['construct']=False;p['commands']=[[],[event(0,0,1)],[event(1,0,1)],[]];out.append((p,False,None))
    p=base('lexical_rearm',[op(0),op(1)])
    p['construct']=False;p['commands']=[[event(0,0,1),event(1,0,1)],[event(0,0,1),event(1,0,1)],[]];out.append((p,False,None))
    p=copy.deepcopy(p);p['name']='actual_acknowledgment';p['commands'][0]+=[event(0,1,0),event(1,1,0)];out.append((p,True,None))
    p=base('branch_missing_publication',[op(0,write=True),op(0),op(1,read=True)])
    p['construct']=False;p['body']=seq(choice(seq(leaf(0),leaf(1)),seq()),leaf(2))
    p['commands']=[[],[event(0,0,1)],[event(1,0,1)],[]];out.append((p,False,None))
    p=base('full_exit',[]);p['construct']=False;p['commands']=[[event(0,0,1)]];out.append((p,False,None))
    p=base('zero_and_nested',[op(0,write=True),op(1,read=True)])
    p['body']=loop(loop(seq(leaf(0),leaf(1))));p['keys']=[[0,1,0],[1,0,0]];out.append((p,True,None))
    return out


def full_history_crosscheck():
    """Use the archived independent full-history oracle, not new port semantics.

    Invalid prefixes continue algebraically ONLY after recording and comparing
    the violation. These words never become accepted programs or certificates.
    """
    rng = random.Random(2026091534)
    engines = ('p0', 'p1', 'p2')
    cells = ('x', 'y', 'z')
    keys = {a + b: (a, b) for a in engines for b in engines if a != b}
    counts = {'words': 160, 'command_verdicts': 0, 'residual_sets': 0,
              'payload_refusals': 0, 'rearm_refusals': 0}
    for trial in range(counts['words']):
        model = Interface(engines, cells, keys)
        state = model.initial()
        oracle = FullHistory(engines, keys)
        for step in range(40):
            option = rng.randrange(5)
            empty = [k for k in keys if k not in oracle.live]
            live = list(oracle.live)
            if option == 0 and empty:
                key = rng.choice(empty)
                command = Command('set', keys[key][0], key=key)
            elif option == 1 and live:
                key = rng.choice(live)
                command = Command('wait', keys[key][1], key=key)
            elif option == 2:
                command = Command('fence', rng.choice(engines))
            else:
                engine = rng.choice(engines)
                reads = tuple(c for c in cells if rng.randrange(4) == 0)
                writes = tuple(c for c in cells if rng.randrange(4) == 0)
                command = Command('op', engine, reads, writes, label=f'{trial}:{step}')
            if command.kind == 'op':
                assert set(model.payload_residual(state, command)) == oracle.residual(command)
                counts['residual_sets'] += 1
            expected = oracle.step(command)
            try:
                next_state = model.transfer(state, command)
                actual = None
            except Rejected as error:
                actual = error.kind
                next_state = model.transfer(state, command, check=False)
            assert actual == expected, (trial, step, command, actual, expected)
            state = next_state
            counts['command_verdicts'] += 1
            counts['payload_refusals'] += actual == 'payload'
            counts['rearm_refusals'] += actual == 'rearm'
    return counts


def negative_tests(driver):
    """Check that the bridge itself detects invented proof credit and bad assets."""
    detected = []
    early = next(c for c, _, _ in cases() if c['name'] == 'early_exact')
    original = compact(driver, early)
    mutations = [
        ('pending_completion', lambda r: r['cuts'][1]['incoming']['pending'][1].__setitem__(0, 0)),
        ('receipt_future_write', lambda r: r['cuts'][2]['incoming']['events'][0]['u'].__setitem__(1, 0)),
        ('consumption_return', lambda r: r['cuts'][2]['outgoing']['events'][0].__setitem__('k', 3)),
        ('occupancy', lambda r: r['cuts'][2]['incoming']['events'][0].__setitem__('o', 1)),
    ]
    for name, change in mutations:
        report = copy.deepcopy(original)
        change(report)
        try:
            collect(early, report)
        except AssertionError as error:
            assert 'compact fact contradicted' in str(error)
            detected.append(name)
        else:
            raise AssertionError(f'bridge did not detect corrupted {name}')
    p = base('stale_carried_consumption', [op(0), op(1), op(2)])
    p['construct'] = False
    p['commands'] = [[event(0,0,1), event(0,0,2)], [event(1,0,1)], [event(1,0,2)], []]
    report = compact(driver, p)
    a = report['keys'].index([0,1,0]); b = report['keys'].index([0,2,0])
    report['cuts'][2]['incoming']['events'][b]['j'][a] = 1
    try:
        collect(p, report)
    except AssertionError as error:
        assert 'carried consumption' in str(error)
        detected.append('stale_carried_consumption')
    else:
        raise AssertionError('bridge accepted a stale carried consumption')
    with tempfile.TemporaryDirectory(prefix='oahs-reference-negative-') as directory:
        root = Path(directory)
        shutil.copytree(VENDOR, root/'vendor', ignore=shutil.ignore_patterns('__pycache__'))
        with (root/'vendor'/'causal_interface.py').open('a') as stream:
            stream.write('\n# deliberately modified negative fixture\n')
        try:
            verify_vendor(root/'vendor')
        except ValueError:
            detected.append('reference_source_integrity')
        else:
            raise AssertionError('reference hash mismatch was ignored')
        example = VENDOR/'ordered_results'/'early_prefix'
        cert = json.loads((example/'certificate.json').read_text())
        cert['invariant'][cert['program']['entry']] = []
        bad_cert = root/'certificate.json'; bad_cert.write_text(json.dumps(cert))
        try:
            check_emitted(bad_cert, example/'schema.json')
        except AssertionError:
            detected.append('certificate_entry')
        else:
            raise AssertionError('missing invariant entry was accepted')
        schema = json.loads((example/'schema.json').read_text())
        schema['placements'][0]['guard'] = 'event_is_full()'
        bad_schema = root/'schema.json'; bad_schema.write_text(json.dumps(schema))
        try:
            check_emitted(example/'certificate.json', bad_schema)
        except ValueError:
            detected.append('unobservable_guard')
        else:
            raise AssertionError('event-state guard was accepted')
    return detected

def self_test(driver, output=None):
    provenance=verify_vendor()
    negatives=negative_tests(driver)
    history=full_history_crosscheck()
    closure=[]
    for directory in sorted((VENDOR/'ordered_results').iterdir()):
        if (directory/'certificate.json').exists():
            closure.append(check_emitted(directory/'certificate.json',directory/'schema.json'))
    results=[]
    for case,expected,exact in cases():
        got=bridge(driver,case)
        assert got['status']=='checked',(case['name'],got)
        assert got['compact_accepted']==expected,(case['name'],got)
        assert got['reference']['accepted']==expected,(case['name'],got)
        if case.get('construct'):assert got['constructed'],(case['name'],got)
        if exact is not None:assert got['order_exact']==exact,(case['name'],got)
        results.append(got)
    rng=random.Random(2026091533)
    # Fixed candidate words and constructor outputs; every draw is explicitly
    # sequenced in Python. Reference acceptance never uses a bounded unrolling.
    for i in range(120):
        operations=[]
        for j in range(4):
            lane=rng.randrange(3);cell=rng.randrange(3);write=bool(rng.randrange(2))
            operations.append(op(lane,cell,not write,write))
        case=base(f'generated_{i}',operations)
        if i%4==1:case['body']=seq(leaf(0),choice(seq(leaf(1),leaf(2)),seq()),leaf(3))
        elif i%4==2:case['body']=loop(seq(leaf(0),leaf(1),leaf(2),leaf(3)))
        elif i%4==3:case['body']={'kind':'while','children':[seq(leaf(0),leaf(1)),seq(leaf(2),leaf(3))]}
        got=bridge(driver,case)
        assert got['status']=='checked',(case['name'],got)
        results.append(got)
        report=compact(driver,case)
        if report['constructed']:
            mutations=[]
            for cut,word in enumerate(report['commands']):
                for offset,command in enumerate(word):
                    if command[0] in (0,1,2):mutations.append((cut,offset))
            for cut,offset in mutations[:3]:
                m=copy.deepcopy(case);m['name']+=f'_delete_{cut}_{offset}';m['construct']=False
                m['commands']=copy.deepcopy(report['commands']);del m['commands'][cut][offset]
                got=bridge(driver,m);assert got['status']=='checked',(m['name'],got);results.append(got)
    # Unsupported contracts and resource limits must not appear as verified.
    p=base('unsupported_sync',[op(0,write=True)]);p['synchronous']=[0]
    assert bridge(driver,p)['status']=='unsupported_contract'
    p=base('limit_negative',[op(0,write=True),op(1,read=True)])
    assert bridge(driver,p,state_limit=0)['status']=='inconclusive'
    out={'passed':True,'source_archive_sha256':provenance['archive_sha256'],
         'reference_files_verified':len(provenance['files']),
         'stored_certificates':len(closure),
         'stored_states':sum(x['certificate_states'] for x in closure),
         'stored_closure_edges':sum(x['closure_edges'] for x in closure),
         'cases':len(results),'compact_accepted':sum(x['compact_accepted'] for x in results),
         'reference_accepted':sum(x['reference']['accepted'] for x in results),
         'exact':sum(x['order_exact'] is True for x in results),
         'safe_coarser':sum(x['order_exact'] is False for x in results),
         'reference_states':sum(x['reference']['states'] for x in results),
         'fact_implications':sum(x['reference']['fact_implications'] for x in results),
         'constructor_successes':sum(x['constructed'] for x in results),
         'precision_only_ref_accepts':sum(x['reference']['accepted'] and not x['compact_accepted'] for x in results),
         'negative_tests':negatives,'full_history_crosscheck':history,
         'trip_count_bound':None,'results':results,'certificate_checks':closure}
    if output:Path(output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('results','certificate_checks')},indent=2))
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--case',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-states',type=int,default=None,
                        help='test-reference resource guard; exhaustion is inconclusive, never accepted')
    a=parser.parse_args()
    if sys.flags.optimize:
        parser.error('assertion-disabled Python (-O/PYTHONOPTIMIZE) cannot run this validation gate')
    if a.max_states is not None and a.max_states < 0:
        parser.error('--max-states must be nonnegative')
    verify_vendor()
    if a.self_test:self_test(a.driver,a.output);return
    if not a.case:parser.error('provide --case or --self-test')
    case=json.loads(a.case.read_text());out=bridge(a.driver,case,state_limit=a.max_states)
    text=json.dumps(out,indent=2)+'\n'
    if a.output:a.output.write_text(text)
    print(text,end='')
    if out['status']!='checked':raise SystemExit(2)

if __name__=='__main__':main()
