# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Original Event Lab reference code retains its MIT notice below and in LICENSE.
# SPDX-License-Identifier: MIT
"""v0.2 regressions: actual review failures, not native/compiler claims."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import itertools
import json
import random
import unittest

from eventlab import fixtures as f
from eventlab.intervals import Footprint
from eventlab.symbolic import Program
from eventlab.physical import lower, materialize, allocate, verify_async, compare, Allocation
from eventlab.reuse import allocate_symbolically, prove_key_rule
from eventlab.storage_keys import derive_storage_candidate

EXAMPLES = Path(__file__).parents[1] / 'examples'


def two_reads():
    return f.model('two_independent_reads', [], 'true', [
        f.op('WA', 'V', [], 'true', ['0'], writes=[f.access('VEC', '0')]),
        f.op('WB', 'V', [], 'true', ['1'], writes=[f.access('VEC', '4')]),
        f.op('RA', 'V', [], 'true', ['2'], reads=[f.access('VEC', '0')]),
        f.op('RB', 'V', [], 'true', ['3'], reads=[f.access('VEC', '4')])], [{}])


class IntervalTests(unittest.TestCase):
    def test_union_and_intersection_exact(self):
        a = Footprint([('MAT', 0, 8), ('MAT', 4, 12), ('LEFT', 0, 4)])
        b = Footprint([('MAT', 10, 16), ('RIGHT', 0, 4)])
        self.assertEqual(a.intervals, (('LEFT', 0, 4), ('MAT', 0, 12)))
        self.assertEqual((a & b).intervals, (('MAT', 10, 12),))
        self.assertEqual((a | b).byte_count, 24)

    def test_random_intervals_against_independent_byte_oracle(self):
        rng = random.Random(0xA2A3)
        def points(spans):
            return {(s, (p,)) for s, a, b in spans for p in range(a, b)}
        for _ in range(2000):
            a = [(rng.choice(['MAT', 'LEFT']), (lo := rng.randrange(-20, 20)), lo + rng.randrange(10))
                 for _ in range(rng.randrange(8))]
            b = [(rng.choice(['MAT', 'LEFT']), (lo := rng.randrange(-20, 20)), lo + rng.randrange(10))
                 for _ in range(rng.randrange(8))]
            x, y = Footprint(a), Footprint(b)
            self.assertEqual(points((x & y).intervals), points(a) & points(b))
            self.assertEqual(points((x | y).intervals), points(a) | points(b))
            self.assertEqual(x.byte_count, len(points(a)))

    def test_half_open_and_large_endpoints(self):
        start = 1 << 80
        a = Footprint([('MAT', start, start + (1 << 40))])
        b = Footprint([('MAT', start + (1 << 40), start + (1 << 40) + 1)])
        self.assertFalse(a & b)
        self.assertEqual(len(a.intervals), 1)
        self.assertEqual(a.byte_count, 1 << 40)
        self.assertTrue(a & Footprint([('MAT', start - 1, start + 1)]))

    def test_invalid_intervals_rejected(self):
        for intervals in [[('MAT', 8, 7)], [('MAT', 0.5, 1)], [('MAT', 0, True)]]:
            with self.assertRaises(ValueError):
                Footprint(intervals)

    def test_real_size_panel_no_byte_enumeration(self):
        for size in [131072, 1 << 40]:
            model = f.model('large_panel', [], 'true', [
                f.op('L', 'MTE2', [], 'true', ['0'], writes=[f.access('MAT', '0', size)]),
                f.op('X', 'MTE1', [], 'true', ['1'], reads=[f.access('MAT', '0', size)])], [{}])
            p = Program(model); plan = p.plan(); c = lower(plan, {})
            self.assertEqual(c.operations[0]['writes'].byte_count, size)
            self.assertEqual(len(c.operations[0]['writes'].intervals), 1)
            self.assertFalse(c.missing())
            self.assertEqual(verify_async(c, allocate(c))['status'], 'PASS')
            broken = materialize(c.operations, [])
            self.assertEqual(verify_async(broken, allocate(broken))['status'], 'FAIL')

    def test_interval_budget_counts_effects_not_bytes(self):
        p = Program(two_reads())
        with self.assertRaisesRegex(ValueError, 'interval budget'):
            p.instances({}, interval_limit=1)
        self.assertEqual(len(p.instances({}, interval_limit=4)), 4)

    def test_original_fixture_hazards_match_byte_oracle(self):
        # Independence check for the replacement finite footprint representation:
        # rebuild the OLD byte sets only for small fixtures, not for real panels.
        for data in f.all_models():
            p = Program(data); parameters = data['scenarios'][-1]
            c = lower(p.plan(), parameters)
            reads, writes = {}, {}
            for relation, table in [(p.reads, reads), (p.writes, writes)]:
                for instance, location in relation.points(parameters):
                    table.setdefault(instance, set()).add(location)
            byte_ops = deepcopy(c.operations)
            for op in byte_ops:
                op['reads'] = reads.get(op['identity'], set())
                op['writes'] = writes.get(op['identity'], set())
            byte_model = materialize(byte_ops, c.handoffs, c.barriers)
            self.assertEqual(c.dense_hazards(), byte_model.dense_hazards(), data['name'])


class BarrierRefinementTests(unittest.TestCase):
    def test_review_counterexample(self):
        p = Program(two_reads())
        before, after = lower(p.plan(refine_barriers=False), {}), lower(p.plan(), {})
        self.assertEqual(before.barriers, {2, 3})
        self.assertEqual(after.barriers, {2})
        quality = compare(before, after)
        self.assertEqual(quality['removed_payload_order'], [(2, 3)])
        self.assertEqual(quality['new_payload_order'], [])
        self.assertEqual(verify_async(after, allocate(after))['status'], 'PASS')

    def test_repeated_reader_occurrences_not_just_static_families(self):
        data = f.model('repeated_same_lane_reader', ['N'], '0<=N<=1000000', [
            f.op('W', 'V', [], 'true', ['0','0'], writes=[f.access('VEC', '0', 8)]),
            f.op('R', 'V', ['i'], '0<=i<N', ['1','i'], reads=[f.access('VEC', '0', 8)])], [])
        p = Program(data); before, after = p.plan(refine_barriers=False), p.plan()
        for n in [0,1,2,8]:
            a,b = lower(before, {'N':n}), lower(after, {'N':n})
            self.assertEqual(len(b.barriers), int(n>0))
            self.assertTrue(compare(a,b)['no_additional_payload_blocking'])
            self.assertEqual(verify_async(b,allocate(b),max_states=10000)['status'],'PASS')
        remote = p.relation(['R[999999] -> R[999999] : N=1000000'])
        self.assertFalse(remote.subset(after.barriers['V']))

    def test_writes_cannot_use_their_own_barrier_as_supply(self):
        data = f.model('repeated_writer', ['N'], '0<=N<=1000000', [
            f.op('W','V',['i'],'0<=i<N',['i'],writes=[f.access('VEC','0',8)])], [])
        plan = Program(data).plan()
        for n in [0,1,2,6]:
            c=lower(plan,{'N':n})
            self.assertEqual(len(c.barriers),max(n-1,0))
            self.assertFalse(c.missing())

    def test_guarded_and_empty_recurrence(self):
        data = f.model('guarded_local', ['N','TAKE'], '0<=N<=5 and 0<=TAKE<=1', [
            f.op('W','V',['i'],'0<=i<N',['i','0'],writes=[f.access('VEC','0',8)]),
            f.op('R','V',['i'],'0<=i<N and TAKE=1',['i','1'],reads=[f.access('VEC','0',8)]),
            f.op('Other','V',['i'],'0<=i<N',['i','2'],reads=[f.access('VEC','0',8)])], [])
        p=Program(data); before,after=p.plan(refine_barriers=False),p.plan()
        for n,t in itertools.product(range(4),range(2)):
            a,b=lower(before,{'N':n,'TAKE':t}),lower(after,{'N':n,'TAKE':t})
            self.assertFalse(b.missing())
            self.assertTrue(compare(a,b)['no_additional_payload_blocking'])
            self.assertEqual(verify_async(b,allocate(b))['status'],'PASS')

    def test_random_linear_memory_programs(self):
        rng=random.Random(8031)
        for trial in range(80):
            ops=[]
            for i in range(rng.randrange(2,9)):
                kind=rng.choice(['read','write','rmw'])
                access=f.access('VEC',str(rng.randrange(4)),rng.randrange(1,4))
                ops.append(f.op('P'+str(i),'V',[],'true',[str(i)],
                                reads=[access] if kind!='write' else [],
                                writes=[access] if kind!='read' else []))
            p=Program(f.model('random_'+str(trial),[],'true',ops,[{}]))
            before,after=lower(p.plan(refine_barriers=False),{}),lower(p.plan(),{})
            self.assertFalse(after.missing())
            self.assertTrue(compare(before,after)['no_additional_payload_blocking'])
            self.assertLessEqual(len(after.barriers),len(before.barriers))
            self.assertEqual(verify_async(after,allocate(after),max_states=50000)['status'],'PASS')

    def test_deleted_required_barrier_is_detected(self):
        c=lower(Program(two_reads()).plan(),{})
        bad=materialize(c.operations,c.handoffs,[])
        self.assertTrue(bad.missing())
        self.assertEqual(verify_async(bad,allocate(bad))['status'],'FAIL')


class StorageKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import time, sys
        started = time.monotonic()
        cls.plans = {}
        cls.assignments = {}
        for name in ['review_combined_gemm','review_combined_gemm_unrolled']:
            p=Program(json.loads((EXAMPLES/(name+'.json')).read_text()))
            print(f'\n  building {name}', file=sys.stderr, flush=True)
            cls.plans[name]=p.plan()
            print(f'  memory plan ready at {time.monotonic()-started:.3f}s', file=sys.stderr, flush=True)
            cls.assignments[name]=allocate_symbolically(cls.plans[name])
            print(f'  keys ready at {time.monotonic()-started:.3f}s', file=sys.stderr, flush=True)

    def test_both_representations_get_proved_slot_keys(self):
        for name,assignment in self.assignments.items():
            self.assertTrue(assignment.accepted,assignment.report())
            for domain,rule in assignment.rules.items():
                self.assertEqual(rule.name,'physical-conflict-slot')
                self.assertTrue(rule.next_same_key.subset(rule.causal_support))
                self.assertIsNotNone(rule.candidate_basis)
            self.assertEqual(assignment.rules[('MTE1','MTE2')].key_count,4)
            self.assertEqual(assignment.rules[('MTE2','MTE1')].key_count,4)

    def test_old_search_still_incomplete_not_proof_weakened(self):
        # The same proof with candidates disabled still fails the looped case.
        plan=self.plans['review_combined_gemm']
        old=allocate_symbolically(plan,storage_candidates=False)
        self.assertFalse(old.accepted)
        self.assertEqual(old.failed_domain,('MTE1','MTE2'))

    def test_unrolled_totality_and_pairing(self):
        plan=self.plans['review_combined_gemm_unrolled']
        a=self.assignments['review_combined_gemm_unrolled']
        for parameters in [{'O':1,'K':0},{'O':1,'K':1},{'O':2,'K':2},{'O':2,'K':3}]:
            c=lower(plan,parameters); keys=a.specialize(plan,c,parameters).keys
            for x,y in itertools.combinations(c.handoffs,2):
                if keys[x.id]==keys[y.id]:
                    self.assertTrue(c.graph.precedes(('consume',x.id),('publish_issue',y.id)) or
                                    c.graph.precedes(('consume',y.id),('publish_issue',x.id)))

    def test_concrete_footprints_and_schedules_equal_modulo_unrolling_names(self):
        for parameters in [{'O':0,'K':0},{'O':1,'K':1},{'O':2,'K':3}]:
            a=lower(self.plans['review_combined_gemm'],parameters)
            b=lower(self.plans['review_combined_gemm_unrolled'],parameters)
            self.assertEqual([(x['time'],x['reads'],x['writes']) for x in a.operations],
                             [(x['time'],x['reads'],x['writes']) for x in b.operations])
            self.assertEqual(a.dense_hazards(),b.dense_hazards())
            self.assertEqual(a.ordered_payload(),b.ordered_payload())

    def test_colliding_slot_rule_rejected(self):
        plan=self.plans['review_combined_gemm'];domain=('MTE1','MTE2')
        candidate,_=derive_storage_candidate(plan,domain)
        self.assertIsNone(prove_key_rule(plan,domain,candidate.keys(plan.program,1),'deliberate-collision',1))

    def test_offsets_gaps_and_renamed_iterators(self):
        data=f.ring(3)
        # Keep real mapping changes separate from just renaming coordinates.
        import re
        for op in data['statements']:
            op['iterators']=['slot_iteration']
            ren=lambda x: re.sub(r'\bi\b','slot_iteration',re.sub(r'\bo\b','tile',str(x)))
            op['domain']=ren(op['domain']);op['schedule']=[ren(x) for x in op['schedule']]
            for mode in ['reads','writes']:
                for access in op[mode]:
                    access['address']=ren(access['address'])
                    if access['space']=='VEC':
                        access['address']='8192 + 3*('+access['address']+')'
        plan=Program(data).plan();a=allocate_symbolically(plan)
        self.assertTrue(a.accepted)
        self.assertEqual(a.rules[('MTE2','V')].key_count,3)
        self.assertEqual(a.rules[('MTE2','V')].name,'physical-conflict-slot')

    def test_expensive_nested_case_returns_unproved_at_budget(self):
        from eventlab.islwrap import ISL, ISLError
        data=json.loads((EXAMPLES/'nested_three_slot_scaled.json').read_text())
        with self.assertRaisesRegex(ISLError, 'maximal number of operations'):
            Program(data,ISL(max_operations=100000)).plan()

    def test_live_token_mutation_is_not_hidden_by_storage_identity(self):
        plan=Program(f.preload()).plan();c=lower(plan,{'J':1,'K':1})
        bad=Allocation(True,{h.id:('MTE2','MTE1',0) for h in c.handoffs},{},{},'injected')
        self.assertEqual(verify_async(c,bad)['status'],'FAIL')

    def test_descriptor_budget_fallback_is_explicit(self):
        plan=Program(f.ring(4)).plan()
        candidate,reason=derive_storage_candidate(plan,('MTE2','V'),max_descriptors=2)
        self.assertIsNone(candidate)
        self.assertIn('budget',reason)
        # A failed enumeration leaves the isl context usable.
        self.assertTrue(allocate_symbolically(plan).accepted)

    def test_gemm_updates_still_require_target_model(self):
        c=lower(self.plans['review_combined_gemm'],{'O':2,'K':2})
        self.assertEqual(c.metrics()['named_barriers'],{'M':6})
        no_m=materialize(c.operations,c.handoffs,[])
        self.assertTrue(no_m.missing())

if __name__=='__main__':
    unittest.main()
