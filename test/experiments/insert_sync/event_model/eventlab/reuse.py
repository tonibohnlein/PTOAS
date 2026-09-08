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
"""Prove bounded candidate key functions symbolically, without sampling trip counts.

This is intentionally not a complete allocator. A fixed-plan causal chain-cover
oracle also exists in physical.py. Here periodic functions are accepted ONLY when
every next same-key publication is causally after the previous consumption over
the entire declared parameter context. Failed search is not global infeasibility.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .islwrap import Rel
from .symbolic import Plan
from .physical import Allocation
from .storage_keys import derive_storage_candidate

@dataclass
class Rule:
    name: str
    key_count: int
    publication_keys: Rel
    acquisition_keys: Rel
    next_same_key: Rel
    causal_support: Rel
    candidate_basis: dict | None = None
    def report(self):
        return {'rule':self.name,'key_count':self.key_count,
                'publication_keys':str(self.publication_keys),
                'candidate_basis':self.candidate_basis,
                'acquisition_keys':str(self.acquisition_keys),
                'next_same_key_publication':str(self.next_same_key),
                'rearm_proved_for_entire_context':self.next_same_key.subset(self.causal_support),
                'criterion':'previous consumer payload finishes before next source payload issues; sufficient, not necessary'}

@dataclass
class SymbolicAllocation:
    accepted: bool
    rules: dict[tuple[str,str],Rule]
    failed_domain: tuple[str,str]|None
    attempts: int
    diagnostics: list[dict] = field(default_factory=list)
    def report(self):
        return {'accepted':self.accepted,'attempts':self.attempts,
                'failed_domain':self.failed_domain, 'diagnostics':self.diagnostics,
                'rules':{'->'.join(d):r.report() for d,r in self.rules.items()},
                'scope':'bounded candidate-key search with symbolic proofs, not exhaustive allocation or PTO code generation'}
    def specialize(self,plan:Plan,concrete,parameters):
        if not self.accepted:raise ValueError('no complete symbolic key assignment')
        by_domain={d:dict(r.publication_keys.points(parameters)) for d,r in self.rules.items()}
        keys={}
        for h in concrete.handoffs:
            domain=(h.src_lane,h.dst_lane)
            source=concrete.operations[h.source]['identity']
            _,value=by_domain[domain][source]
            keys[h.id]=(*domain,value[0])
        return Allocation(True,keys,{}, {'->'.join(d):r.key_count for d,r in self.rules.items()},
                          'specialized from a key rule proved over the entire declared context')


def prove_key_rule(plan:Plan,domain,keys:Rel,name:str,count:int):
    p,q=domain;h=plan.handoffs[domain]
    keys=keys.filter(domain=h)
    if not keys.single():raise ValueError('a publication must have one key')
    # Totality of the key mapping over all actual publications.
    ids=plan.program.identity.filter(domain=h)
    if not ids.subset(keys.domain_identity()):
        raise ValueError('key rule misses a publication occurrence')
    all_key_values=plan.program.relation([f'Key[k] -> Key[k] : {plan.program.context} and 0<=k<{count}'])
    if not keys.then(all_key_values).equal(keys):
        raise ValueError('key rule exceeds its declared pool')
    # Same-key pairs span heterogeneous occurrence tuple dimensions after
    # unrolling. Coalescing their full Cartesian product triggers an isl-0.27
    # assertion in the review case. Coalescing is NOT a proof operation: retain
    # the exact uncoalesced relation, restrict to source order, then select next.
    same=keys.then(keys.reverse(), simplify=False).intersect(plan.program.before[p], simplify=False)
    # For each publication, find its next same-key source occurrence, if any.
    next_=same.then(plan.program.schedule).lexmin().then(plan.program.schedule.reverse())
    # H maps a publication to its matching consumer. Full completion of that
    # consumer before the next producer is a sufficient causal rearm witness:
    # consume(a) -> issue(C) -> finish(C) -> issue(P_next) -> submit(b).
    support=h.then(plan.supply)
    if not next_.subset(support):return None
    acquisition=h.reverse().then(keys)
    if not acquisition.single():raise ValueError('ambiguous acquisition key')
    return Rule(name,count,keys,acquisition,next_,support)


def allocate_symbolically(plan:Plan,pool_size=6,*,storage_candidates=True):
    if type(pool_size) is not int or pool_size<0:raise ValueError('nonnegative integer pool required')
    if pool_size>8:raise ValueError('prototype key search is bounded at eight; no target reservation claim')
    p=plan.program;rules={};attempts=0;diagnostics=[]
    for domain in sorted(plan.handoffs):
        ops=[op for op in p.statements if op['lane']==domain[0]]
        accepted=None
        storage, reason = derive_storage_candidate(plan, domain) if storage_candidates else (None, 'disabled for ablation')
        diagnostics.append({'domain': domain, 'storage_candidate': storage.report() if storage else None,
                            'reason': reason})
        for k in range(1,pool_size+1):
            if storage is not None:
                attempts += 1
                accepted = prove_key_rule(plan, domain, storage.keys(p, k), 'physical-conflict-slot', k)
                if accepted:
                    accepted.candidate_basis = storage.report()
                    break
            # These are arithmetic candidates, NOT kernel/loop-topology patterns.
            # Actual storage correspondence and all guards enter the proof.
            for mode in ['constant','static-rank','inner-index','inner-index-plus-rank','outer-index']:
                clauses=[]
                for rank,op in enumerate(ops):
                    it=op.get('iterators',[])
                    base={'constant':'0','static-rank':str(rank),
                          'inner-index':it[-1] if it else str(rank),
                          'inner-index-plus-rank':f'({it[-1]})+{rank}' if it else str(rank),
                          'outer-index':it[0] if it else str(rank)}[mode]
                    clauses.append(p.instance(op)+f' -> Key[((({base}) % {k}))] : '+p.condition(op))
                attempts+=1
                accepted=prove_key_rule(plan,domain,p.relation(clauses),mode,k)
                if accepted:break
            if accepted:break
        if not accepted:return SymbolicAllocation(False,rules,domain,attempts,diagnostics)
        rules[domain]=accepted
    return SymbolicAllocation(True,rules,None,attempts,diagnostics)
