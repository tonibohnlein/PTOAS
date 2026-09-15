"""Check emitted JSON and an inductive certificate without rerunning synthesis.
This uses the documented transfer algebra; the solver is not trusted.
"""
import json
from pathlib import Path
from causal_interface import State,Command,Interface,Rejected
from order_interface import OrderedInterface,OrderedState
from programs import Program,Node

def load_base(d):
    return State(tuple(d['reach']),d['live'],tuple(tuple(x)for x in d['history']))
def load_state(d,exact):
    if not exact:return load_base(d)
    return OrderedState(load_base(d['base']),tuple(d['reference']),tuple(tuple(x)for x in d['pairs']))
def load_cmd(d):
    return Command(d['kind'],d.get('engine',''),tuple(d.get('reads',())),tuple(d.get('writes',())),d.get('key',''),d.get('label',''))
def emitted_plan(schema,p,variables,options,slots,keys):
    if schema['keys']!={k:list(v)for k,v in keys.items()}:raise ValueError('physical key pool changed')
    plan={v:0 for o,j,v in variables};lookup={(o,j):v for o,j,v in variables};seen=set()
    for group in schema['placements']:
        o=group['observation']
        if o in seen or o not in p.predicates:raise ValueError('invalid/duplicate original observation')
        seen.add(o)
        if group['guard']!=p.predicates[o]:raise ValueError('guard is not the qualified original predicate')
        if len(group['commands'])>slots:raise ValueError('too many commands at cut')
        for j,d in enumerate(group['commands']):
            c=load_cmd(d)
            if c not in options:raise ValueError('unqualified primitive')
            plan[lookup[o,j]]=options.index(c)
    return plan

def check(certfile,schemafile):
    data=json.loads(Path(certfile).read_text());d=data['program'];p=Program(d['name'],tuple(d['engines']),tuple(d['cells']),[Node(load_cmd(n['command']),tuple(n['successors']),n['observation'],n['label'])for n in d['nodes']],d['entry'],set(d['exits']),d['predicates'],d['assumptions'])
    exact=data.get('order_exact',False);keys={k:tuple(v)for k,v in data['keys'].items()}
    options=[load_cmd(x)for x in data['options']];variables=data['variables'];slots=data['slots']
    plan=emitted_plan(json.loads(Path(schemafile).read_text()),p,variables,options,slots,keys)
    if plan!=data['plan']:raise ValueError('certificate is for a different emitted program')
    model=Interface(p.engines,p.cells,keys)
    if exact:model=OrderedInterface(model)
    lookup={(o,j):v for o,j,v in variables};obs=sorted({n.observation for n in p.nodes if n.observation})
    invariant=[{(load_state(x['state'],exact),tuple(x['origin']))for x in entries}for entries in data['invariant']]
    assert (model.initial(),tuple(-1 for _ in keys))in invariant[p.entry]
    checked=0;transfers=0
    for at,bucket in enumerate(invariant):
        n=p.nodes[at]
        for s,origin in bucket:
            origin=list(origin)
            if n.observation:
                for j in range(slots):
                    c=options[plan[lookup[n.observation,j]]];s=model.transfer(s,c);transfers+=1
                    if c.kind=='set':origin[model.ki[c.key]]=obs.index(n.observation)*slots+j
                    elif c.kind=='wait':origin[model.ki[c.key]]=-1
            s=model.transfer(s,n.command);transfers+=1
            if at in p.exits:assert not s.live,'live exit'
            out=(s,tuple(origin))
            for target in n.successors:
                assert out in invariant[target],f'certificate not closed on {at}->{target}'
                checked+=1
    return dict(name=p.name,accepted=True,order_exact=exact,certificate_states=sum(map(len,invariant)),
                closure_edges=checked,command_transfers=transfers,synthesis_rerun=False)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory');a=parser.parse_args();p=Path(a.directory)
    print(json.dumps(check(p/'certificate.json',p/'schema.json'),indent=2))
