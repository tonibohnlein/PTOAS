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
"""Occurrence-accurate dependence discovery and symbolic prefix-frontier synthesis.

The input is a facts interchange, not a new source IR or a claim that arbitrary PTO
can be parsed here. Iteration domains, schedules, and byte footprints use isl syntax.
The plan stays symbolic in parameters; concrete specialization is only for tests and
physical-key realization in this first prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
import re
from .islwrap import ISL, Rel
from .intervals import Footprint

_NAME=re.compile(r'^[A-Za-z_][A-Za-z_0-9]*$')
# An intentionally small same-core subset. This is not a substitute for the
# native target/version/reservation tables, instruction effects, or visibility.
LANES = {'MTE2', 'MTE1', 'M', 'FIX', 'V', 'MTE3'}
SPACES = {'GM', 'VEC', 'MAT', 'LEFT', 'RIGHT', 'ACC'}
DIRECT = {('MTE2','MTE1'), ('MTE1','MTE2'), ('MTE1','M'), ('M','MTE1'),
          ('M','FIX'), ('FIX','M'), ('MTE2','V'), ('V','MTE2'),
          ('V','MTE3'), ('MTE3','V')}

@dataclass
class Dependencies:
    raw: Rel
    war: Rel
    waw: Rel
    dense: Rel
    obligations: Rel
    live_in_reads: Rel
    def all(self): return self.raw | self.war | self.waw

@dataclass
class Plan:
    program: 'Program'
    dependencies: Dependencies
    handoffs: dict[tuple[str,str],Rel]
    barriers: dict[str,Rel]
    supply: Rel
    closure_stable: bool
    audit: list[dict]
    def primitive(self):
        return self.program.primitive_supply(self.handoffs,self.barriers)
    def report(self):
        return {'scope':'symbolic affine/presburger facts, fixed payload schedule',
                'isl_version':self.program.isl.version,
                'RAW':str(self.dependencies.raw),'WAR':str(self.dependencies.war),
                'WAW':str(self.dependencies.waw),
                'live_in_reads':str(self.dependencies.live_in_reads),
                'live_in_read_precision':('exact flow result' if self.program.writes.equal(self.program.definite_writes) else 'not computed in conservative may-write fallback'),
                'handoffs':{p+'->'+q:str(h) for (p,q),h in self.handoffs.items()},
                'barriers_before':{p:str(b) for p,b in self.barriers.items()},
                'all_memory_requirements_proved':self.dependencies.dense.subset(self.supply),
                'supply_fixed_point_reached':self.closure_stable,'audit':self.audit,
                'not_proved':['PTO native effect extraction', 'instruction-specific target legality and effect contracts','arbitrary non-affine control',
                              'complete search over all finite-key realizations',
                              'hardware visibility beyond the qualified local model',
                              'device timing']}

class Program:
    def __init__(self, data: dict, isl: ISL|None=None):
        self.data=data
        self.isl=isl or ISL()
        self.parameters=tuple(data.get('parameters',[]))
        self.context=data.get('context','true')
        self.statements=data['statements']
        if len(self.parameters) != len(set(self.parameters)):
            raise ValueError('duplicate parameter name')
        if data.get('fixed_protocols') or data.get('resource_effects'):
            raise ValueError('fixed protocols and resource effects require a native semantic adapter')
        self.by_name={x['id']:x for x in self.statements}
        if len(self.by_name)!=len(self.statements): raise ValueError('duplicate statement identity')
        if not self.statements: raise ValueError('no statements')
        self.time_dims=len(self.statements[0]['schedule'])
        for name in (*self.parameters,*self.by_name):
            if not _NAME.fullmatch(name): raise ValueError('invalid symbolic identifier: '+name)
        for op in self.statements:
            if op['lane'] not in LANES: raise ValueError('unsupported physical lane')
            if op.get('core', 'local') != 'local':
                raise ValueError('multiple/remote physical contexts are outside this prototype')
            if len(op.get('iterators', [])) != len(set(op.get('iterators', []))):
                raise ValueError('duplicate iterator identity')
            for it in op.get('iterators', []):
                if not _NAME.fullmatch(it) or it in self.parameters or it == 'byte':
                    raise ValueError('invalid or colliding iterator identity')
            if len(op['schedule'])!=self.time_dims: raise ValueError('schedule dimensions differ')
            if op.get('opaque_effects',False): raise ValueError('opaque effects must stay with production fallback')
            for a in op.get('reads',[])+op.get('writes',[]):
                if a['space'] not in SPACES: raise ValueError('unsupported physical address space; do not invent alias-separated roots')
                if not isinstance(a['size'],int) or a['size']<1: raise ValueError('positive constant byte extent required')
        self.schedule=self._schedule()
        if not self.schedule.single() or not self.schedule.injective():
            raise ValueError('schedule must be functional and distinguish coexecuting operations')
        self.reverse_schedule=self._schedule(reverse=True)
        self.reads=self._accesses('reads')
        self.writes=self._accesses('writes')
        self.definite_writes=self._accesses('writes',True)
        self.lanes=tuple(sorted({x['lane'] for x in self.statements}))
        if set(self.lanes) & {'V','MTE3'} and set(self.lanes) & {'M','MTE1','FIX'}:
            raise ValueError('mixed Cube/vector physical contexts require a native contract')
        self.lane_schedule={p:self._schedule(lane=p) for p in self.lanes}
        self.before={p:self.lane_schedule[p].order(self.lane_schedule[p]) for p in self.lanes}
        self.through={p:self.lane_schedule[p].order(self.lane_schedule[p],False) for p in self.lanes}
        self.identity=self.schedule.then(self.schedule.reverse())
        self.global_order=self.schedule.order(self.schedule)

    def relation(self,clauses):
        return self.isl.map('['+','.join(self.parameters)+'] -> { '+'; '.join(clauses)+' }')
    def empty(self): return self.relation([])
    @staticmethod
    def instance(op): return op['id']+'['+','.join(op.get('iterators',[]))+']'
    def condition(self,op): return '('+self.context+') and ('+op.get('domain','true')+')'
    def _schedule(self,lane=None,reverse=False):
        clauses=[]
        for op in self.statements:
            if lane is not None and op['lane']!=lane: continue
            t=[f'-({x})' if reverse else str(x) for x in op['schedule']]
            clauses.append(self.instance(op)+' -> T['+','.join(t)+'] : '+self.condition(op))
        return self.relation(clauses)
    def _accesses(self,mode,definite_only=False,space=None):
        clauses=[]
        for op in self.statements:
            for a in op.get(mode,[]):
                if definite_only and not a.get('definite',True): continue
                if space is not None and a['space']!=space: continue
                addr=a['address']; size=a['size']
                clauses.append(self.instance(op)+f" -> {a['space']}[byte] : "+self.condition(op)+
                               f' and ({addr}) <= byte < ({addr}) + {size}')
        return self.relation(clauses)
    def _flow(self,sinks,schedule):
        # A may-write is not a content kill. Use the complete pairwise relation
        # when exact reaching-definition flow is not applicable, rather than
        # falsely claiming a definite producer.
        if not self.writes.equal(self.definite_writes):
            order=schedule.order(schedule)
            return (self.writes.then(sinks.reverse()) & order), self.empty()
        out=self.isl.flow(sinks,self.writes,schedule)
        return out['must_dependence'] | out['may_dependence'], out['must_no_source'] | out['may_no_source']
    def discover(self):
        raw,live=self._flow(self.reads,self.schedule)
        waw,_=self._flow(self.writes,self.schedule)
        next_write,_=self._flow(self.reads,self.reverse_schedule)
        war=next_write.reverse()
        dense=((self.writes.then(self.reads.reverse()) |
                self.reads.then(self.writes.reverse()) |
                self.writes.then(self.writes.reverse())) & self.global_order)
        # Dense requirements are an independent symbolic safety contract. The
        # compiler seed uses sparse nearest-definition/reclamation relationships.
        sparse=raw|war|waw
        # Do not build a transitive closure merely to discover requirements.
        # Exact reaching/next-write flow supplies the sparse seed; the completed
        # plan is independently checked against ALL dense overlapping pairs.
        obligations=sparse
        # Completion is not an unqualified cross-pipe GM visibility contract.
        gm_w=self._accesses('writes',space='GM')
        gm_r=self._accesses('reads',space='GM')
        gm_raw=gm_w.then(gm_r.reverse()) & self.global_order
        for p in self.lanes:
            for q in self.lanes:
                if p!=q and not gm_raw.filter(self.lane_schedule[p],self.lane_schedule[q]).empty():
                    raise ValueError('cross-pipe GM publication/visibility is outside this target model')
        # Ordinary RAR is not a memory dependency. ACC can carry additional
        # target resource conflicts: do not accidentally certify a read-only
        # cross-pipe ACC case using a payload-memory-only model.
        acc_r=self._accesses('reads',space='ACC')
        acc_rr=acc_r.then(acc_r.reverse()) & self.global_order
        for p in self.lanes:
            for q in self.lanes:
                if p!=q and not acc_rr.filter(self.lane_schedule[p],self.lane_schedule[q]).subset(dense):
                    raise ValueError('cross-pipe ACC read/read resource contract is not represented')
        return Dependencies(raw,war,waw,dense,obligations,live)

    def frontier(self,required: Rel,p:str,q:str):
        """Backward needs -> latest REQUIRED source prefix; forward availability
        -> keep only prefix increases at destination demands. No producer lookahead.

        Subtracting domination by earlier targets is valid because their target
        schedule strictly decreases along any domination chain in a finite execution.
        """
        rel=required.filter(self.lane_schedule[p],self.lane_schedule[q])
        if rel.empty(): return rel
        latest=rel.reverse().then(self.schedule).lexmax().then(self.schedule.reverse()).reverse()
        earlier_supply=self.through[p].then(latest).then(self.before[q])
        handoffs=latest-earlier_supply
        if not handoffs.single() or not handoffs.injective():
            raise ValueError('frontier matching must be one publication to one acquisition per execution')
        return handoffs

    def primitive_supply(self,handoffs,barriers):
        result=self.empty()
        for (p,q),h in handoffs.items():
            result=result|self.through[p].then(h).then(self.through[q])
        for p,targets in barriers.items():
            result=result|self.before[p].then(targets).then(self.through[p])
        return result

    def plan(self, *, refine_barriers=True):
        deps=self.discover(); audit=[]
        hs={(p,q):self.frontier(deps.obligations,p,q)
            for p in self.lanes for q in self.lanes if p!=q}
        hs={d:h for d,h in hs.items() if not h.empty()}
        if any(d not in DIRECT for d in hs):
            raise ValueError('a required direct event domain is outside the prototype target subset')
        cross,stable=self.primitive_supply(hs,{}).closure(5, required=deps.dense)
        barriers={}
        for p in self.lanes:
            missing=deps.dense.filter(self.lane_schedule[p],self.lane_schedule[p])-cross
            if not missing.empty():
                targets=self.identity.filter(domain=missing.reverse())
                barriers[p]=targets
        # v0.2: interpret the selected same-lane cuts as supply, too. First
        # remove occurrence subsets dominated by earlier candidate cuts. This
        # is only a proposal: recheck the WHOLE plan without the removed cuts.
        if refine_barriers:
            barriers = self.refine_barriers(deps, hs, barriers, cross, audit)
        # Complete direction families, not singleton coverage sets. Every deletion
        # is judged in the current full plan, without the deleted family itself.
        for domain in list(sorted(hs)):
            trial=dict(hs); old=trial.pop(domain)
            coverage,st=self.prove_plan_supply(trial,barriers,deps.dense)
            if deps.dense.subset(coverage):
                hs=trial
                audit.append({'action':'remove_redundant_direction_family','direction':domain,
                              'removed':str(old),'reason':'all dense requirements supplied jointly'})
        final,stable=self.primitive_supply(hs,barriers).closure(6, required=deps.dense)
        if not deps.dense.subset(final):
            raise ValueError('bounded closure did not establish all requirements; no plan accepted')
        audit.append({'action':'accept_symbolic_payload_order',
                      'reason':'all dense memory hazards are implied, for the whole parameter context',
                      'scope':'completion order, not physical reusable-token proof'})
        return Plan(self,deps,hs,barriers,final,stable,audit)

    def prove_plan_supply(self, handoffs, barriers, required):
        """Goal-directed exact proof with a cheap *rejection-only* bound.

        All primitive completion edges advance original issue order. A path
        from u to v therefore needs a first primitive edge u->x with x<=v.
        Failure of this necessary condition proves the trial cannot supply all
        requirements. Its success does NOT prove coverage; exact composition
        remains mandatory. This avoids expensive futile closure on many removed
        ACC-barrier trials.
        """
        primitive = self.primitive_supply(handoffs, barriers)
        if required.subset(primitive):
            return primitive, False
        possible = primitive.then(self.global_order | self.identity)
        if not required.subset(possible):
            return primitive, False
        return primitive.closure(6, required=required)

    def refine_barriers(self, deps, handoffs, barriers, cross, audit, budget=64):
        """Deletion-only refinement, preserving all payload and event cuts.

        Candidate cuts may dominate one another on a finite execution. Removing
        their union is accepted only when exact path compositions still prove
        the full dense contract. No deleted barrier may supply its own proof.
        The final event assignment must use the returned plan, not cached supply.
        """
        result = dict(barriers)
        attempts = 0

        def accept(lane, proposed, reason):
            nonlocal result, attempts
            if attempts >= budget or proposed.equal(result[lane]):
                return False
            attempts += 1
            trial = dict(result)
            removed = trial[lane] - proposed
            if proposed.empty():
                del trial[lane]
            else:
                trial[lane] = proposed
            supply, _ = self.prove_plan_supply(handoffs, trial, deps.dense)
            if not deps.dense.subset(supply):
                return False
            # All primitives were only deleted. This inclusion plus unchanged
            # payload/control means no new payload ordering is introduced.
            if not self.primitive_supply(handoffs, trial).subset(
                    self.primitive_supply(handoffs, result)):
                raise ValueError('barrier deletion unexpectedly adds primitive ordering')
            result = trial
            audit.append({'action': 'remove_redundant_barrier_occurrences',
                          'lane': lane, 'removed': str(removed), 'reason': reason,
                          'all_dense_requirements_rechecked': True,
                          'no_added_primitive_order': True})
            return True

        for lane in sorted(list(result)):
            missing = deps.dense.filter(self.lane_schedule[lane], self.lane_schedule[lane]) - cross
            # Strict at both ends: a barrier does not finish its own following
            # operation, and a cut cannot justify its own removal.
            earlier = self.before[lane].then(result[lane]).then(self.before[lane])
            remaining = missing - earlier
            essential = self.identity.filter(domain=remaining.reverse()) & result[lane]
            accept(lane, essential, 'earlier selected cuts already supply these demands')

        # Bounded whole-site-family cleanup can additionally exploit paths
        # involving handoffs and several lanes. Unknown coverage keeps the cut.
        for op in reversed(self.statements):
            lane = op['lane']
            if lane not in result or attempts >= budget:
                continue
            site = self.relation([self.instance(op) + ' -> ' + self.instance(op)
                                  + ' : ' + self.condition(op)])
            accept(lane, result[lane] - site, 'full mixed plan supplies the removed barrier family')
        audit.append({'action': 'barrier_refinement_summary', 'attempts': attempts,
                      'budget': budget, 'scope': 'symbolic deletion only; allocation is re-proved later'})
        return result

    def instances(self,params,limit=4096,interval_limit=65536):
        if set(params) != set(self.parameters):
            raise ValueError('specialization parameters must exactly match declared parameters')
        if any(type(v) is not int for v in params.values()):
            raise ValueError('specialization parameters must be integers')
        context = self.relation(['__Context[] -> __Context[] : '+self.context])
        if context.specialize(params).empty():
            raise ValueError('specialization lies outside the declared analysis context')
        times=self.schedule.points(params,limit=limit)
        result=[]
        if type(interval_limit) is not int or interval_limit < 1:
            raise ValueError('positive interval budget required')
        reads = {}; writes = {}; used = 0
        for op in self.statements:
            for mode, store in [('reads', reads), ('writes', writes)]:
                for access in op.get(mode, []):
                    lo = access['address']; size = access['size']
                    bounds = self.relation([self.instance(op) +
                        f' -> Bounds[({lo}), ({lo})+{size}] : ' + self.condition(op)])
                    for instance, (_, (start, end)) in bounds.points(params, limit=interval_limit):
                        used += 1
                        if used > interval_limit:
                            raise ValueError('memory interval budget exceeded')
                        store.setdefault(instance, []).append((access['space'], start, end))
        for instance,(_,time) in sorted(times,key=lambda item:item[1][1]):
            result.append({'identity':instance,'lane':self.by_name[instance[0]]['lane'],
                           'time':time,'reads':Footprint(reads.get(instance, [])),
                           'writes':Footprint(writes.get(instance, []))})
        return result
