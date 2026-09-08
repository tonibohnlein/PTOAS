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
from __future__ import annotations
import itertools
import json
import unittest
from copy import deepcopy
from dataclasses import replace
from eventlab import fixtures as f
from eventlab.symbolic import Program
from eventlab.physical import lower,allocate,verify_async,materialize,Handoff,compare,Allocation

class SymbolicTests(unittest.TestCase):
    def test_ring_arbitrary_parameter_correspondence(self):
        for slots in [1,2,3,4]:
            with self.subTest(slots=slots):
                p=Program(f.ring(slots));a=p.plan();s=p.isl
                expected=s.map(f'[N] -> {{ C[i] -> L[j=i+{slots}] : N<=1000000 and 0<=i and j<N; '
                               f'S[i] -> C[j=i+{slots}] : N<=1000000 and 0<=i and j<N }}')
                self.assertTrue(a.dependencies.war.equal(expected))
                self.assertEqual(a.barriers,{})
                self.assertTrue(a.dependencies.dense.subset(a.supply))
    def test_nested_invocation_reset_not_flat_distance(self):
        p=Program(f.ring(2,True));plan=p.plan()
        relations=set(plan.dependencies.war.points({'O':2,'N':3}))
        self.assertIn((('C',(0,2)),('L',(1,0))),relations)
        self.assertIn((('S',(0,2)),('C',(1,0))),relations)
        self.assertIn((('C',(0,1)),('L',(1,1))),relations)
    def test_preload_wait_does_not_capture_second_load(self):
        p=Program(f.preload());plan=p.plan();c=lower(plan,{'J':2,'K':2})
        names=[o['identity'] for o in c.operations]
        b=names.index(('B',()));use=names.index(('UseA',(0,)))
        self.assertNotIn((b,use),c.ordered_payload())
        self.assertEqual(len(c.handoffs),2)
    def test_empty_reader_preserves_writer_obligation(self):
        p=Program(f.panel());plan=p.plan()
        empty=lower(plan,{'O':2,'J':0})
        self.assertEqual(len(empty.barriers),1)
        self.assertFalse(empty.missing())
        nonempty=lower(plan,{'O':2,'J':3})
        self.assertFalse(nonempty.barriers)
        release=[h for h in nonempty.handoffs if h.src_lane=='MTE1'][0]
        self.assertEqual(nonempty.operations[release.source]['identity'],('Extract',(0,2)))
        self.assertEqual(nonempty.operations[release.target]['identity'],('Load',(1,)))
    def test_bundle_only_when_boundaries_are_compatible(self):
        c=lower(Program(f.bundle()).plan(),{})
        self.assertEqual(len(c.handoffs),1)
        c=lower(Program(f.bundle(True)).plan(),{})
        self.assertEqual(len(c.handoffs),2)
        self.assertNotIn((2,1),c.ordered_payload())
    def test_acc_updates_keep_internal_dependencies(self):
        p=Program(f.accumulator());pl=p.plan()
        edges=set(pl.dependencies.raw.points({'O':2,'K':2}))
        self.assertIn((('Init',(0,)),('Update',(0,0))),edges)
        self.assertIn((('Update',(0,0)),('Update',(0,1))),edges)
        self.assertIn((('Update',(0,1)),('Store',(0,))),edges)
        self.assertGreater(len(lower(pl,{'O':2,'K':2}).barriers),0)
        # No fabricated MMAD intrinsic certificate is available to this prototype.
        self.assertFalse(lower(pl,{'O':2,'K':0}).missing())
    def test_partial_writes_keep_both_productions(self):
        p=Program(f.partial());pl=p.plan();edges=set(pl.dependencies.raw.points({}))
        self.assertIn((('Low',()),('ReadAll',())),edges)
        self.assertIn((('High',()),('ReadAll',())),edges)
        self.assertNotIn((('High',()),('ReadLow',())),edges)
    def test_may_write_does_not_kill_previous_writer(self):
        p=Program(f.partial(True));pl=p.plan();edges=set(pl.dependencies.raw.points({}))
        self.assertIn((('Low',()),('ReadAll',())),edges)
        self.assertIn((('High',()),('ReadAll',())),edges)
    def test_equivalent_parity_constraint_shapes(self):
        forms=['i%2=0','i%2!=1','exists a: i=2*a']
        base=None
        for form in forms:
            p=Program(f.parity(form));pl=p.plan()
            normalized={kind:set(getattr(pl.dependencies,kind).points({'N':7})) for kind in ['raw','war','waw']}
            if base is None:base=normalized
            self.assertEqual(base,normalized)
            self.assertFalse(lower(pl,{'N':7}).missing())
    def test_symbolic_proof_is_not_a_fixed_unroll(self):
        p=Program(f.ring(3));pl=p.plan()
        # Query a remote iteration without enumerating the preceding iterations.
        witness=p.isl.map('[N] -> { C[999996] -> L[999999] : N=1000000 }')
        self.assertTrue(witness.subset(pl.dependencies.war))
        self.assertTrue(witness.subset(pl.supply))
    def test_no_source_and_no_reader_are_not_event_tokens(self):
        p=Program(f.preload());pl=p.plan();c=lower(pl,{'J':0,'K':0})
        self.assertEqual(len(c.operations),2)
        self.assertEqual(c.handoffs,[])
        self.assertTrue(allocate(c).accepted)
    def test_unsupported_visibility_not_called_completion(self):
        x=f.model('bad',[],'true',[
            f.op('W','MTE3',[],'true',['0'],writes=[f.access('GM','0')]),
            f.op('R','MTE2',[],'true',['1'],reads=[f.access('GM','0')])],[{}])
        with self.assertRaisesRegex(ValueError,'visibility'):Program(x).plan()
    def test_parameter_domain_is_not_silently_empty_success(self):
        p=Program(f.ring(2));pl=p.plan()
        for value in [-1,1000001]:
            with self.assertRaisesRegex(ValueError,'outside'):lower(pl,{'N':value})
        for parameters in [{}, {'N':2,'EXTRA':1}, {'N':2.5}]:
            with self.assertRaises(ValueError):lower(pl,parameters)
    def test_opaque_and_mixed_contexts_rejected(self):
        x=f.bundle();x['statements'][0]['opaque_effects']=True
        with self.assertRaisesRegex(ValueError,'opaque'):Program(x)
        x=f.bundle();x['statements'][0]['core']='other'
        with self.assertRaisesRegex(ValueError,'contexts'):Program(x)
    def test_no_memory_rar_demand(self):
        x=f.model('readonly',[], 'true',[
            f.op('A','MTE2',[],'true',['0'],reads=[f.access('MAT','0',4)]),
            f.op('B','MTE1',[],'true',['1'],reads=[f.access('MAT','0',4)])],[{}])
        pl=Program(x).plan()
        self.assertTrue(pl.dependencies.dense.empty())
        self.assertEqual(pl.handoffs,{})
        self.assertFalse(pl.dependencies.live_in_reads.empty())
    def test_nonzero_lower_bound_nonunit_step(self):
        x=f.ring(2)
        for op in x['statements']:
            op['domain']='2<=i<2+3*N and (i-2)%3=0'
            for a in op.get('reads',[])+op.get('writes',[]):
                a['address']=a['address'].replace('i%2','floor((i-2)/3)%2')
        plan=Program(x).plan()
        expected=plan.program.relation(['C[i] -> L[j=i+6] : 2<=i and j<2+3*N and (i-2)%3=0 and 0<=N<=1000000',
                                       'S[i] -> C[j=i+6] : 2<=i and j<2+3*N and (i-2)%3=0 and 0<=N<=1000000'])
        self.assertTrue(plan.dependencies.war.equal(expected))
        self.assertFalse(lower(plan,{'N':3}).missing())
    def test_acc_read_read_cannot_be_silently_ignored(self):
        x=f.model('acc_resource',[], 'true',[
            f.op('A','M',[],'true',['0'],reads=[f.access('ACC','0',4)]),
            f.op('B','FIX',[],'true',['1'],reads=[f.access('ACC','0',4)])],[{}])
        with self.assertRaisesRegex(ValueError,'resource'):Program(x).plan()
    def test_schedule_collisions_rejected(self):
        x=f.bundle();x['statements'][1]['schedule']=['0']
        with self.assertRaisesRegex(ValueError,'distinguish'):Program(x)

class ConcreteTests(unittest.TestCase):
    def test_all_fixture_specializations_against_independent_hazards(self):
        for data in f.all_models():
            p=Program(data);plan=p.plan()
            for params in data['scenarios']:
                with self.subTest(name=data['name'],params=params):
                    c=lower(plan,params)
                    self.assertEqual(c.missing(),[])
                    a=allocate(c,6)
                    self.assertTrue(a.accepted,a)
                    for paths in a.chains.values():
                        for path in paths:
                            for x,y in zip(path,path[1:]):
                                self.assertTrue(c.graph.precedes(('consume',x),('publish_issue',y)))
    def test_ring_async_and_three_lane_overlap(self):
        c=lower(Program(f.ring(2)).plan(),{'N':3})
        report=verify_async(c,allocate(c))
        self.assertEqual(report['status'],'PASS',report)
        self.assertEqual(report['max_active_lanes'],3)
    def test_event_pool_shortage_does_not_add_order(self):
        c=lower(Program(f.preload()).plan(),{'J':1,'K':1})
        a=allocate(c,1)
        self.assertFalse(a.accepted)
        self.assertEqual(a.required['MTE2->MTE1'],2)
        self.assertEqual(a.keys,{})
    def test_source_signal_submission_is_nonblocking(self):
        c=lower(Program(f.preload()).plan(),{'J':1,'K':1})
        names=[o['identity'] for o in c.operations]
        self.assertNotIn((names.index(('A',())),names.index(('B',()))),c.ordered_payload())
        r=verify_async(c,allocate(c));self.assertEqual(r['status'],'PASS',r)
    def test_broad_handoff_is_safe_but_loses_overlap(self):
        good=lower(Program(f.preload()).plan(),{'J':1,'K':1})
        broad=materialize(good.operations,[Handoff(0,1,2,'MTE2','MTE1')])
        self.assertFalse(broad.missing())
        quality=compare(good,broad)
        self.assertIn((1,2),quality['new_payload_order'])
        self.assertFalse(quality['no_additional_payload_blocking'])
        self.assertEqual(verify_async(broad,allocate(broad))['status'],'PASS')
    def test_same_readiness_cut_merges_two_hazards(self):
        good=lower(Program(f.bundle()).plan(),{})
        separate=materialize(good.operations,[Handoff(0,0,2,'MTE1','M'),Handoff(1,1,2,'MTE1','M')])
        quality=compare(separate,good)
        self.assertEqual(quality['new_payload_order'],[])
        self.assertEqual(quality['removed_payload_order'],[])
        self.assertEqual(len(good.handoffs),1)
    def test_deleted_ready_is_detected(self):
        good=lower(Program(f.ring(2)).plan(),{'N':2})
        badhs=[replace(h,id=i) for i,h in enumerate(good.handoffs[1:])]
        bad=materialize(good.operations,badhs,good.barriers)
        self.assertTrue(bad.missing())
        report=verify_async(bad,allocate(bad))
        self.assertEqual(report['status'],'FAIL',report)
        self.assertEqual(report['failure']['kind'],'memory-order')
    def test_stale_generation_is_detected(self):
        good=lower(Program(f.ring(2)).plan(),{'N':2})
        hs=list(good.handoffs)
        index=next(i for i,h in enumerate(hs) if h.src_lane=='MTE2' and h.source==3)
        hs[index]=replace(hs[index],source=0)
        bad=materialize(good.operations,hs)
        self.assertTrue(bad.missing())
        self.assertEqual(verify_async(bad,allocate(bad))['status'],'FAIL')
    def test_deleted_release_is_detected(self):
        good=lower(Program(f.ring(1)).plan(),{'N':2})
        hs=[h for h in good.handoffs if not(h.src_lane=='V' and h.dst_lane=='MTE2')]
        bad=materialize(good.operations,[replace(h,id=i) for i,h in enumerate(hs)])
        self.assertTrue(bad.missing())
        self.assertEqual(verify_async(bad,allocate(bad))['status'],'FAIL')
    def test_forced_key_sharing_is_detected(self):
        c=lower(Program(f.preload()).plan(),{'J':1,'K':1})
        a=Allocation(True,{h.id:('MTE2','MTE1',0) for h in c.handoffs},{},{},'deliberately invalid')
        report=verify_async(c,a)
        self.assertEqual(report['status'],'FAIL',report)
        self.assertEqual(report['failure']['kind'],'event-rearm')
    def test_skipped_read_and_empty_loop(self):
        for data,params in [(f.panel(),{'O':2,'J':0}),(f.panel(),{'O':2,'J':2}),
                            (f.conditional(),{'N':3,'TAKE':1}),(f.ring(2,True),{'O':2,'N':1})]:
            c=lower(Program(data).plan(),params);r=verify_async(c,allocate(c))
            self.assertEqual(r['status'],'PASS',r)
    def test_async_budget_exhaustion_is_not_pass(self):
        c=lower(Program(f.ring(2)).plan(),{'N':3})
        self.assertEqual(verify_async(c,allocate(c),max_states=5)['status'],'LIMIT')
    def test_comparison_rejects_changed_access_semantics(self):
        from copy import deepcopy
        a=lower(Program(f.bundle()).plan(),{})
        ops=deepcopy(a.operations);ops[0]['writes']=set()
        b=materialize(ops,a.handoffs)
        with self.assertRaisesRegex(ValueError,'footprints'):compare(a,b)
    def test_allocator_minimum_under_fixed_compatibility(self):
        c=lower(Program(f.ring(2)).plan(),{'N':4});a=allocate(c)
        for domain,paths in a.chains.items():
            events=[h.id for h in c.handoffs if h.src_lane+'->'+h.dst_lane==domain]
            # Exhaustively reject every smaller coloring in this small population.
            for colors in range(1,len(paths)):
                for coloring in itertools.product(range(colors),repeat=len(events)):
                    valid=True
                    for (i,x),(j,y) in itertools.combinations(enumerate(events),2):
                        if coloring[i]==coloring[j] and not (
                            c.graph.precedes(('consume',x),('publish_issue',y)) or
                            c.graph.precedes(('consume',y),('publish_issue',x))):valid=False;break
                    self.assertFalse(valid,(domain,coloring))

class SymbolicReuseTests(unittest.TestCase):
    def test_key_rules_are_proved_for_all_fixture_contexts(self):
        from eventlab.reuse import allocate_symbolically
        for data in f.all_models():
            plan=Program(data).plan();a=allocate_symbolically(plan)
            self.assertTrue(a.accepted,data['name'])
            for rule in a.rules.values():
                self.assertTrue(rule.next_same_key.subset(rule.causal_support))
            for params in data['scenarios']:
                c=lower(plan,params)
                assigned=a.specialize(plan,c,params)
                self.assertEqual(set(assigned.keys),{h.id for h in c.handoffs})
                for x,y in itertools.combinations(c.handoffs,2):
                    if assigned.keys[x.id]==assigned.keys[y.id]:
                        self.assertTrue(c.graph.precedes(('consume',x.id),('publish_issue',y.id)) or
                                        c.graph.precedes(('consume',y.id),('publish_issue',x.id)))
    def test_modulo_assignment_derived_and_checked(self):
        from eventlab.reuse import allocate_symbolically
        for slots in [1,2,3,4]:
            plan=Program(f.ring(slots)).plan();a=allocate_symbolically(plan)
            self.assertEqual(a.rules[('MTE2','V')].key_count,slots)
            if slots>1:self.assertFalse(allocate_symbolically(plan,slots-1).accepted)
    def test_preload_pool_one_has_no_proof(self):
        from eventlab.reuse import allocate_symbolically
        plan=Program(f.preload()).plan()
        self.assertFalse(allocate_symbolically(plan,1).accepted)
        self.assertTrue(allocate_symbolically(plan,2).accepted)
    def test_symbolic_nested_key_matching(self):
        from eventlab.reuse import allocate_symbolically
        plan=Program(f.ring(2,True)).plan();a=allocate_symbolically(plan)
        c=lower(plan,{'O':2,'N':2})
        r=verify_async(c,a.specialize(plan,c,{'O':2,'N':2}))
        self.assertEqual(r['status'],'PASS',r)
    def test_declared_key_range_checked(self):
        from eventlab.reuse import prove_key_rule
        plan=Program(f.preload()).plan();p=plan.program
        bad=p.relation(['A[] -> Key[2] : J>0', 'B[] -> Key[2] : K>0'])
        with self.assertRaisesRegex(ValueError,'pool'):
            prove_key_rule(plan,('MTE2','MTE1'),bad,'bad',2)

if __name__=='__main__':unittest.main()
