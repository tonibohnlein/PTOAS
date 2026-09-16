"""Small executable operational qualification of the abstract block sequencer.

This is a second MODEL, not hardware execution. Each block has writable/writing/
readable/reading state; first pending same-role request is served first. The graph
adapter is compared with this state machine, not with physical UnitFlag silicon.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json
from examples import profile,producer,consumer


def check(blocks:int,generations:int):
    p=profile(blocks)
    actions=[]
    for g in range(generations):actions.extend([producer(p),consumer(p)])
    nodes=[];pred=[];meta=[];issue={e:None for e in p.engines};block_ops={b:[] for b in p.blocks}
    # Native operation graph ONLY, before resource edges.
    for step,cmd in enumerate(actions):
        frag=p.fragment(cmd);local={}
        for name in frag.nodes:
            local[name]=len(nodes);nodes.append((step,name));pred.append(set());meta.append(None)
        for a,b in frag.edges:pred[local[b]].add(local[a])
        if issue[cmd.engine] is not None:pred[local['I']].add(issue[cmd.engine])
        issue[cmd.engine]=local['I']
        for ac in frag.accesses:
            if ac.permission:
                # Generation association on each lane is the ordered-service
                # premise. Permission alone would not determine this mapping.
                g=step//2;bi=p.blocks.index(ac.cell);role=ac.permission
                meta[local[ac.begin]]=(bi,role,'begin',g)
                meta[local[ac.end]]=(bi,role,'end',g)
                block_ops[ac.cell].append((local[ac.begin],local[ac.end]))
    # Repeated output cells need an independent output-order contract. This
    # comparison isolates resource sequencing: add those WAW constraints as
    # fixed native test premises, NOT as inferred UnitFlag credit.
    for g in range(1,generations):
        for j in range(blocks):
            a=nodes.index((2*(g-1)+1,f'o{j}e'));b=nodes.index((2*g+1,f'o{j}b'))
            pred[b].add(a)
    native=[sum(1<<v for v in ps) for ps in pred]
    constrained=list(native)
    for operations in block_ops.values():
        for previous,current in zip(operations,operations[1:]):constrained[current[0]]|=1<<previous[1]
    n=len(nodes);allmask=(1<<n)-1
    def successors(mask,with_resource_edges):
        pp=constrained if with_resource_edges else native
        return [v for v in range(n) if not mask&(1<<v) and pp[v]&mask==pp[v]]
    graph_seen={0};q=deque([0])
    while q:
        mask=q.popleft()
        for v in successors(mask,True):
            nxt=mask|(1<<v)
            if nxt not in graph_seen:graph_seen.add(nxt);q.append(nxt)
    # (state, generation): 0 writable, 1 writing, 2 readable, 3 reading.
    initial=(0,tuple((0,0) for _ in range(blocks)))
    operational={initial};q=deque([initial])
    while q:
        mask,state=q.popleft()
        for v in successors(mask,False):
            ns=list(state);m=meta[v]
            if m is not None:
                bi,role,phase,g=m;st,expected=ns[bi]
                required={('write','begin'):0,('write','end'):1,('read','begin'):2,('read','end'):3}[role,phase]
                if st!=required or g!=expected:continue
                ns[bi]=((st+1)%4,expected+(st==3))
            nxt=(mask|(1<<v),tuple(ns))
            if nxt not in operational:operational.add(nxt);q.append(nxt)
    masks={m for m,s in operational}
    assert masks==graph_seen,(blocks,generations,len(masks),len(graph_seen))
    assert allmask in masks
    return {'blocks':blocks,'generations':generations,'observable_vertices':n,
            'native_and_profile_graph_prefixes':len(graph_seen),
            'operational_states':len(operational),'prefix_sets_equal':True,
            'output_WAW_fixed_for_isolation':generations>1}


def overtaking_counterexample():
    # Instructions P0 and P1 have issued in order, but P1's block starts first.
    # A flag-only machine accepts this schedule and first consumer gets gen1.
    sequence=[('write','begin',1),('write','end',1),('read','begin',0),('read','end',0),
              ('write','begin',0),('write','end',0),('read','begin',1),('read','end',1)]
    state=0;last=None;observed=[]
    for role,phase,g in sequence:
        req={('write','begin'):0,('write','end'):1,('read','begin'):2,('read','end'):3}[role,phase]
        assert state==req
        state=(state+1)%4
        if (role,phase)==('write','end'):last=g
        if (role,phase)==('read','begin'):observed.append((g,last))
    assert observed==[(0,1),(1,0)]
    return {'flag_only_schedule_legal':True,'consumers_and_observed_generations':observed,
            'ordered_service_rejects_first_write':True,
            'is_hardware_counterexample':False,
            'purpose':'shows the ordered-service premise cannot be inferred from permission bits alone'}

if __name__=='__main__':
    rows=[check(1,1),check(2,1),check(1,2)]
    result={'cases':rows,'necessity':overtaking_counterexample()}
    Path(__file__).with_name('operational_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
