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
"""Concrete Ascend-style cut lowering, causal key allocation, and reference checks.

This is a bounded realization oracle, NOT a native PTO emitter. Source signal
submission does not wait for source completion; publication itself does. Payload
issues on a lane are ordered, but their completions are NOT silently serialized.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, replace
from itertools import combinations
from .symbolic import Plan

@dataclass(frozen=True)
class Handoff:
    id: int
    source: int
    target: int
    src_lane: str
    dst_lane: str

class Graph:
    def __init__(self):
        self.names=[]; self.ids={}; self.pred=[]; self.succ=[]
    def node(self,name):
        if name not in self.ids:
            self.ids[name]=len(self.names); self.names.append(name)
            self.pred.append(set()); self.succ.append(set())
        return self.ids[name]
    def edge(self,a,b):
        a=self.node(a);b=self.node(b)
        self.pred[b].add(a);self.succ[a].add(b)
    def close(self):
        indeg=[len(p) for p in self.pred]
        ready=deque(i for i,d in enumerate(indeg) if not d)
        self.anc=[0]*len(indeg);self.order=[]
        while ready:
            i=ready.popleft();self.order.append(i)
            for j in sorted(self.succ[i]):
                self.anc[j]|=self.anc[i]|(1<<i)
                indeg[j]-=1
                if not indeg[j]: ready.append(j)
        if len(self.order)!=len(indeg): raise ValueError('cyclic physical event plan')
    def precedes(self,a,b): return bool(self.anc[self.ids[b]] & (1<<self.ids[a]))

@dataclass
class Concrete:
    operations: list[dict]
    handoffs: list[Handoff]
    barriers: set[int]
    graph: Graph
    streams: dict[str,list[tuple]]
    def ordered_payload(self):
        n=len(self.operations)
        return {(i,j) for j in range(n) for i in range(n)
                if self.graph.precedes(('finish',i),('issue',j))}
    def dense_hazards(self):
        # Deliberately independent of isl flow, symbolic obligations and synthesis.
        edges=set()
        for j,b in enumerate(self.operations):
            for i,a in enumerate(self.operations[:j]):
                if (a['writes'] & (b['reads']|b['writes'])) or (a['reads'] & b['writes']):
                    edges.add((i,j))
        return edges
    def missing(self): return sorted(self.dense_hazards()-self.ordered_payload())
    def metrics(self):
        return {'payload_operations':len(self.operations),'sets':len(self.handoffs),
                'waits':len(self.handoffs),
                'named_barriers':{p:sum(self.operations[i]['lane']==p for i in self.barriers)
                                  for p in self.streams if any(self.operations[i]['lane']==p for i in self.barriers)},
                'exit_policy':'separate abstract completion drain; no target ABI claim',
                'missing_memory_requirements':self.missing()}


def lower(plan:Plan,params:dict[str,int]):
    ops=plan.program.instances(params)
    ids={op['identity']:i for i,op in enumerate(ops)}
    hs=[]
    for (p,q),relation in sorted(plan.handoffs.items()):
        for a,b in relation.points(params):
            hs.append(Handoff(len(hs),ids[a],ids[b],p,q))
    barriers=set()
    for _,relation in plan.barriers.items():
        for a,_ in relation.points(params): barriers.add(ids[a])
    return materialize(ops,hs,barriers)


def materialize(ops,hs,barriers=frozenset()):
    if any(i < 0 or i >= len(ops) for i in barriers):
        raise ValueError('barrier target is not a payload occurrence')
    if any(h.source < 0 or h.target < 0 or h.source >= len(ops) or h.target >= len(ops) for h in hs):
        raise ValueError('handoff endpoint is not a payload occurrence')
    if any(h.src_lane == h.dst_lane for h in hs):
        raise ValueError('same-lane handoffs require the named-barrier model')
    if any(h.id!=i for i,h in enumerate(hs)): raise ValueError('handoff identities must be dense')
    lanes=sorted({op['lane'] for op in ops})
    streams={p:[] for p in lanes};g=Graph()
    for i,op in enumerate(ops):
        g.edge(('issue',i),('finish',i))
        q=streams[op['lane']]
        for h in hs:
            if h.target==i:
                if h.dst_lane!=op['lane']: raise ValueError('wait lane differs from payload lane')
                q.append(('wait',h.id))
        if i in barriers: q.append(('barrier',i))
        q.append(('issue',i))
        for h in hs:
            if h.source==i:
                if h.src_lane!=op['lane'] or h.source>=h.target:
                    raise ValueError('invalid forward publication cut')
                q.append(('signal',h.id))
    def name(action):
        return ({'issue':'issue','signal':'publish_issue','wait':'consume','barrier':'barrier'}[action[0]],action[1])
    for lane,commands in streams.items():
        for c in commands:g.node(name(c))
        for a,b in zip(commands,commands[1:]): g.edge(name(a),name(b))
    for h in hs:
        g.edge(('publish_issue',h.id),('publish_fire',h.id))
        for i,op in enumerate(ops[:h.source+1]):
            if op['lane']==h.src_lane:g.edge(('finish',i),('publish_fire',h.id))
        g.edge(('publish_fire',h.id),('consume',h.id))
    for j in barriers:
        for i,op in enumerate(ops[:j]):
            if op['lane']==ops[j]['lane']:g.edge(('finish',i),('barrier',j))
    g.close()
    return Concrete(ops,list(hs),set(barriers),g,streams)


def compare(reference:Concrete,candidate:Concrete):
    if reference.operations != candidate.operations:
        raise ValueError('payload identities, lane assignment, schedule, or footprints differ')
    before,after=reference.ordered_payload(),candidate.ordered_payload()
    return {'new_payload_order':sorted(after-before),'removed_payload_order':sorted(before-after),
            'missing_memory_requirements':candidate.missing(),
            'no_additional_payload_blocking':after<=before}

@dataclass
class Allocation:
    accepted: bool
    keys: dict[int,tuple[str,str,int]]
    chains: dict[str,list[list[int]]]
    required: dict[str,int]
    reason: str


def allocate(plan:Concrete,pool_size:int=6):
    """Minimum chain cover under a conservative causal rearm criterion.

    A compatible link a->b requires consume(a) BEFORE submission of signal(b),
    without any added waits. Maximum bipartite matching finds the fewest chains
    in this compatibility DAG, independently for each directed event domain.
    This is optimal only in this fixed concrete plan and sufficient-reuse model.
    """
    if pool_size<0:raise ValueError('negative event pool')
    groups={}
    for h in plan.handoffs:groups.setdefault((h.src_lane,h.dst_lane),[]).append(h.id)
    keys={};chains={};required={}
    for domain,vertices in sorted(groups.items()):
        if len(vertices) > 256:
            return Allocation(False, {}, {}, {}, 'reference allocator budget: more than 256 occurrences in one domain')
        compatible={a:[b for b in vertices if a!=b and plan.graph.precedes(('consume',a),('publish_issue',b))]
                    for a in vertices}
        # Deterministic augmenting-path maximum matching; adequate for a reference
        # tool. Production can use Hopcroft--Karp or its existing bounded allocator.
        right={}
        def augment(a,seen):
            for b in compatible[a]:
                if b in seen:continue
                seen.add(b)
                if b not in right or augment(right[b],seen):
                    right[b]=a;return True
            return False
        for a in vertices:augment(a,set())
        successor={a:b for b,a in right.items()}
        roots=[a for a in vertices if a not in right]
        paths=[]
        for a in roots:
            path=[a]
            while path[-1] in successor:path.append(successor[path[-1]])
            paths.append(path)
        if sum(map(len,paths))!=len(vertices):raise ValueError('cyclic or incomplete reuse cover')
        label='->'.join(domain);chains[label]=paths;required[label]=len(paths)
        for color,path in enumerate(paths):
            for h in path:keys[h]=(*domain,color)
    accepted=all(n<=pool_size for n in required.values())
    return Allocation(accepted,keys if accepted else {},chains,required,
                      'causal reuse only; no added payload order' if accepted else
                      'pool insufficient for this plan/model; no automatic serialization or global infeasibility claim')


def verify_async(plan:Concrete,allocation:Allocation,max_states=200000):
    """Exhaustive finite interleaving exploration, independent of graph reachability.

    It models nonblocking signal submission, delayed publication, consuming waits,
    independently completing payloads, exact token ownership, and body barriers.
    Dense reference hazards are checked when an operation issues, NOT used to gate
    execution. A violation is therefore reachable even if the planner omitted it.
    """
    if not allocation.accepted:raise ValueError('cannot verify an unassigned plan')
    if set(allocation.keys) != {h.id for h in plan.handoffs}:
        raise ValueError('allocation must assign every logical event exactly once')
    if max_states < 1: raise ValueError('positive state budget required')
    lanes=tuple(sorted(plan.streams));commands=[plan.streams[p] for p in lanes]
    all_ops=(1<<len(plan.operations))-1
    key_values=sorted(set(allocation.keys.values()))
    kidx={k:i for i,k in enumerate(key_values)}
    event_key={h:kidx[k] for h,k in allocation.keys.items()}
    prior={h.id:sum(1<<i for i,o in enumerate(plan.operations[:h.source+1]) if o['lane']==h.src_lane)
           for h in plan.handoffs}
    barriers={j:sum(1<<i for i,o in enumerate(plan.operations[:j]) if o['lane']==plan.operations[j]['lane'])
              for j in plan.barriers}
    needs=[0]*len(plan.operations)
    for i,j in plan.dense_hazards():needs[j]|=1<<i
    # pc, issued payload, finished payload, pending publications, consumed events,
    # concrete hardware flag values (event identity, -1 empty)
    initial=((0,)*len(lanes),0,0,0,0,(-1,)*len(key_values))
    queue=deque([initial]);seen={initial};terminal=0;max_active=0;examples={};failure=None
    while queue:
        state=queue.popleft();pc,issued,done,pending,consumed,tokens=state
        active=issued & ~done
        active_lanes=tuple(p for p in lanes if any(active&(1<<i) and o['lane']==p
                                                   for i,o in enumerate(plan.operations)))
        max_active=max(max_active,len(active_lanes))
        examples.setdefault(','.join(active_lanes),[i for i in range(len(plan.operations)) if active&(1<<i)])
        next_states=[]
        for lane,stream in enumerate(commands):
            if pc[lane]>=len(stream):continue
            kind,x=stream[pc[lane]]
            npc=list(pc);npc[lane]+=1;npc=tuple(npc)
            if kind=='issue':
                missing=needs[x] & ~done
                if missing:
                    failure={'kind':'memory-order','target':x,
                             'unfinished_sources':[i for i in range(len(needs)) if missing&(1<<i)]};break
                next_states.append((npc,issued|(1<<x),done,pending,consumed,tokens))
            elif kind=='signal':
                key=event_key[x]
                live=[h.id for h in plan.handoffs if event_key[h.id]==key and
                      ((pending&(1<<h.id)) or tokens[key]==h.id)]
                if live:
                    failure={'kind':'event-rearm','event':x,'still_live':live};break
                next_states.append((npc,issued,done,pending|(1<<x),consumed,tokens))
            elif kind=='wait':
                key=event_key[x]
                if tokens[key]==-1:continue
                if tokens[key]!=x:
                    failure={'kind':'wrong-event-generation','expected':x,'observed':tokens[key]};break
                ts=list(tokens);ts[key]=-1
                next_states.append((npc,issued,done,pending,consumed|(1<<x),tuple(ts)))
            elif kind=='barrier':
                if barriers[x]&~done:continue
                next_states.append((npc,issued,done,pending,consumed,tokens))
        if failure:break
        for i in range(len(plan.operations)):
            if issued&(1<<i) and not done&(1<<i):
                next_states.append((pc,issued,done|(1<<i),pending,consumed,tokens))
        for h in plan.handoffs:
            if pending&(1<<h.id) and not prior[h.id]&~done:
                key=event_key[h.id]
                if tokens[key]!=-1:
                    failure={'kind':'live-flag-overwrite','event':h.id};break
                ts=list(tokens);ts[key]=h.id
                next_states.append((pc,issued,done,pending&~(1<<h.id),consumed,tuple(ts)))
        if failure:break
        finished=all(pc[i]==len(s) for i,s in enumerate(commands))
        if finished and done==all_ops and not pending and all(t==-1 for t in tokens):
            terminal+=1
        elif not next_states:
            failure={'kind':'deadlock','pcs':pc};break
        for ns in next_states:
            if ns not in seen:
                seen.add(ns);queue.append(ns)
                if len(seen)>max_states:
                    return {'status':'LIMIT','states':len(seen),'terminal_states':terminal,
                            'max_active_lanes':max_active,'not_a_pass':True}
    return {'status':'FAIL' if failure else 'PASS','states':len(seen),'terminal_states':terminal,
            'failure':failure,'max_active_lanes':max_active,'active_lane_examples':examples,
            'scope':'exhaustive for this finite specialization and stated abstract target only'}
