#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Actual C++ M5 versus unchanged supplied final-block phase reference.

Control and commands are exported by the real C++ core. The reference collects
whole states independently. Ghost READ tags have I/C endpoints and no writer;
they observe whole-operation completion without adding any original conflict.
No reference source is edited. This does not qualify native hardware.
"""
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import replace
from pathlib import Path
import argparse, copy, json, random, subprocess, sys
from phase_vendor_check import verify_pin, VENDOR
verify_pin()
sys.path.insert(0, str(VENDOR/'checks/phases'))
import phase_interface as pi
import examples as ex


def normalize(profile, program):
    engines=list(profile.engines);cells=list(profile.cells);groups=list(profile.groups)
    key_ids={};pools=defaultdict(list)
    for key,(a,b) in profile.keys.items():
        pair=(engines.index(a),engines.index(b));key_ids[key]=(*pair,len(pools[pair]));pools[pair].append(key)
    operations=[];op_ids=[];payload=[];words=[];successors=[]
    for i,node in enumerate(program.nodes):
        action=node.action
        if action.kind in ('op','produce','consume'):
            action=replace(action,label='m5_op_'+str(len(operations)))
            op_ids.append(len(operations));operations.append(action);payload.append(action);words.append([])
        else:
            op_ids.append(-1);payload.append(None)
            if action.kind=='nop':words.append([])
            elif action.kind=='fence':words.append([[2,engines.index(action.engine),0,0]])
            else:
                a,b,k=key_ids[action.key];words.append([[0 if action.kind=='set' else 1,a,b,k]])
        successors.append(list(node.successors))
    # The unique invocation exit is control-only, not a dummy payload operation.
    exit_=len(words)
    for i in program.exits:successors[i].append(exit_)
    successors.append([]);words.append([]);payload.append(None);op_ids.append(-1)
    return dict(profile=profile,engines=engines,cells=cells,groups=groups,keys=key_ids,
                operations=operations,op_ids=op_ids,payload=payload,words=words,
                successors=successors,entry=program.entry,exit=exit_)


def encode(x,construct=False):
    p=x['profile'];cs=x['cells'];es=x['engines'];gs=x['groups'];rows=[str(len(es)),str(len(cs))]
    def ids(values):return str(len(values))+' '+ ' '.join(str(v) for v in values)
    for c in cs:
        v=p.cells[c];rows.append(f'{v.domain} {v.offset} {v.size}')
    # These are supplied MODEL premises, not obtained from device testing.
    rows+=['1 1 1 1',str(len(gs))]
    for name in gs:
        g=p.groups[name];rows.append(f'{name} {es.index(g.producer)} {es.index(g.consumer)} '+ids([cs.index(c) for c in g.blocks]))
    rows.append(str(len(x['keys'])))
    rows += [' '.join(map(str,v)) for v in x['keys'].values()]
    rows.append(str(len(x['operations'])))
    for a in x['operations']:
        reads=list(a.reads);writes=list(a.writes)
        if a.kind=='produce':writes+=list(a.blocks)
        if a.kind=='consume':reads+=list(a.blocks)
        row=f'{es.index(a.engine)} '+ids([cs.index(c) for c in reads])+' '+ids([cs.index(c) for c in writes])
        if a.kind=='op':row+=' 0'
        else:
            row+=f' {1 if a.kind=="produce" else 2} {gs.index(a.group)} '
            row+=ids([cs.index(c) for c in a.blocks])+' '+ids([cs.index(c) for c in a.reads])+' '+ids([cs.index(c) for c in a.writes])
        rows.append(row)
    rows.append(f'{len(x["words"])} {x["entry"]} {x["exit"]}')
    for op,succ in zip(x['op_ids'],x['successors']):rows.append(f'{op} '+ids(succ))
    for word in x['words']:rows.append(str(len(word))+' '+' '.join(' '.join(map(str,c)) for c in word))
    rows.append(str(int(construct)));return '\n'.join(rows)+'\n'


def command(x,c):
    kind,a,b,k=c
    if kind==2:return pi.Action('fence',x['engines'][a])
    inverse={tuple(v):name for name,v in x['keys'].items()}
    if kind not in (0,1) or (a,b,k) not in inverse:raise AssertionError('unqualified actual command')
    return pi.Action('set' if kind==0 else 'wait',x['engines'][a if kind==0 else b],key=inverse[a,b,k])


class TaggedProfile(pi.Profile):
    def __init__(self,x):
        base=x['profile'];self.tags={a.label:'__whole_'+str(i) for i,a in enumerate(x['operations'])}
        cells=list(base.cells.values())+[pi.Cell(t,'GHOST_WHOLE',i,1) for i,t in enumerate(self.tags.values())]
        super().__init__(base.engines,cells,base.groups.values(),base.keys)
    def fragment(self,a):
        f=super().fragment(a)
        if a.kind in ('op','produce','consume'):
            return replace(f,accesses=f.accesses+(pi.Access(self.tags[a.label],'R','I','C'),))
        return f


def collect(x,words,exact=False,tagged=False):
    profile=TaggedProfile(x) if tagged else x['profile'];m=pi.PhaseInterface(profile,exact)
    seen=[set() for _ in words];snap=[dict(incoming=set(),before=set(),outgoing=set()) for _ in words]
    initial=m.initial();seen[x['entry']].add(initial);todo=deque([(x['entry'],initial)])
    while todo:
        at,s=todo.popleft();snap[at]['incoming'].add(s)
        try:
            for c in words[at]:s=m.transfer(s,command(x,c))
            snap[at]['before'].add(s)
            if x['payload'][at]:s=m.transfer(s,x['payload'][at])
            if at==x['exit']:m.check_exit(s)
        except pi.Rejected as e:return False,e.kind,m,snap
        snap[at]['outgoing'].add(s)
        for t in x['successors'][at]:
            if s not in seen[t]:seen[t].add(s);todo.append((t,s))
    return True,'ok',m,snap


def facts(x,actual,m,snap):
    count=0;invkeys={tuple(v):k for k,v in x['keys'].items()}
    used=[invkeys[tuple(k)] for k in actual['keys']]
    def require(ok,description):
        nonlocal count
        count+=1
        if not ok:raise AssertionError(description)
    for at,point in enumerate(actual['cuts']):
        for label,refstates in snap[at].items():
            cpp=point[label]
            if not refstates:continue
            require(cpp is not None,('missing state',at,label))
            for s in refstates:
                def covered(i,port):
                    action=x['operations'][i];bucket=m.ci[m.profile.tags[action.label],action.engine,'R']
                    return all(sig&(1<<port) for sig in s.history[bucket])
                for q,engine in enumerate(x['engines']):
                    for i,bit in enumerate(cpp['pending'][q]):
                        if not bit:require(covered(i,m.pi['A:'+engine]),('invented full completion',at,label,q,i))
                for e,key in enumerate(used):
                    v=cpp['events'][e];live=bool(s.live&(1<<m.ki[key]));require(v['occupancy']&(2 if live else 1),('occupancy',at,key))
                    if v['valid']:require(live,('receipt validity',at,key))
                    for i,bit in enumerate(v['uncovered']):
                        if not bit:require(live and covered(i,m.pi['S:'+key]),('receipt completion',at,key,i))
                    for q,engine in enumerate(x['engines']):
                        if v['known']&(1<<q):require(s.reach[m.pi['D:'+key]]&(1<<m.pi['A:'+engine]),('consumption knowledge',at,key,q))
                    for f,other in enumerate(used):
                        if v['carried'][f]:require(live and s.reach[m.pi['D:'+other]]&(1<<m.pi['S:'+key]),('carried consumption',at,key,other))
                for v in cpp['resources']:
                    cell=x['cells'][v['cell']];u=m.pi['U:'+cell];readable=bool(s.readable&(1<<m.bi[cell]))
                    require(v['permission']&(2 if readable else 1),('resource permission',at,cell))
                    for i,bit in enumerate(v['completed']):
                        if bit:require(covered(i,u),('resource whole-completion',at,cell,i))
                    for f,key in enumerate(used):
                        if v['carried'][f]:require(s.reach[m.pi['D:'+key]]&(1<<u),('resource event knowledge',at,cell,key))
    return count


def fragments(x,actual):
    checks=0
    for i,a in enumerate(x['operations']):
        f=x['profile'].fragment(a);g=actual['fragments'][i]
        assert list(f.nodes)==g['nodes'],(a,g['nodes'],f.nodes)
        assert set(f.edges)=={(f.nodes[u],f.nodes[v]) for u,v in g['edges']}
        expected={(x['cells'].index(v.cell),int('R'in v.mode),int('W'in v.mode),f.nodes.index(v.begin),f.nodes.index(v.end),int(bool(v.permission))) for v in f.accesses}
        assert expected=={tuple(v) for v in g['accesses']};checks+=3
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--driver',required=True);ap.add_argument('--output');args=ap.parse_args()
    result=dict(cases=0,accepted=0,exact=0,coarser=0,construction_attempts=0,constructed=0,fact_implications=0,fragment_checks=0,records=[])
    def run(name,p,q,construct=False,mutate=None):
        x=normalize(p,q)
        if mutate:mutate(x['words'])
        process=subprocess.run([args.driver],input=encode(x,construct),capture_output=True,text=True)
        if process.returncode:raise AssertionError((name,process.returncode,process.stderr))
        actual=json.loads(process.stdout);assert actual['complete'],(name,actual['reason'])
        words=actual['commands'];safe,kind,m,states=collect(x,words)
        assert bool(actual['accepted'])==safe,(name,actual['reason'],kind)
        exact=False
        if safe:
            exact,_,_,_=collect(x,words,True)
            assert bool(actual['exact'])==exact,(name,'order mismatch')
            good,_,tagmodel,tagstates=collect(x,words,False,True);assert good
            result['fact_implications']+=facts(x,actual,tagmodel,tagstates)
            result['accepted']+=1;result['exact' if exact else 'coarser']+=1
        result['fragment_checks']+=fragments(x,actual);result['cases']+=1
        if construct:
            result['construction_attempts']+=1;result['constructed']+=bool(actual['constructed'])
            assert not actual['constructed'], 'native phase construction must remain unqualified'
            assert actual['reason'], 'refusal must explain the missing phase adapter'
        result['records'].append(dict(name=name,accepted=safe,exact=exact,kind=kind,constructed=bool(actual['constructed']),states=actual['states']))
        return x,actual
    profile=ex.profile(2);P,C=ex.producer(profile),ex.consumer(profile)
    # Original serialized profile schemas: recurrence validated without unrolling.
    for path in sorted((VENDOR/'checks/phases/certificates').glob('*.json')):
        obj=json.loads(path.read_text());p=pi.Profile.from_json(obj['profile']);q=pi.program_from_json(obj['program'])
        x,a=run('certificate/'+path.stem,p,q)
        assert a['accepted']
        for at,word in enumerate(x['words']):
            for j in range(len(word)):
                run('delete/'+path.stem+f'/{at}/{j}',p,q,mutate=lambda w,at=at,j=j:w[at].pop(j))
        run('construct/'+path.stem,p,q,construct=True)
    ordinary=pi.Action('op','L',writes=('left','right'),label='load')
    cases={
      'input_missing':[ordinary,P,C],
      'output_missing':[P,C,pi.Action('op','Q',reads=('g0','g1'))],
      'operand_false_release':[P,C,*ex.handshake(profile,'fm'),pi.Action('op','M',writes=('left',))],
      'later_unrelated':[P,pi.Action('op','M',writes=('side',)),C,*ex.handshake(profile,'fm'),pi.Action('op','M',reads=('side',))],
      'coarse_software_pair':[P,*ex.handshake(profile,'mf'),C],
      'invalid_acquire':[ordinary,ex.event(profile,'lm','wait'),P,C],
      'invalid_relay':[ordinary,ex.event(profile,'lm','wait'),*ex.handshake(profile,'mf'),P,C],
      'wrong_role':[P,P,C],
      'resource_left_live':[P],
      'stale_ack':[ex.event(profile,'qm','set'),ex.event(profile,'qm','wait'),P,C,ex.event(profile,'fq','set'),*ex.handshake(profile,'mq'),ex.event(profile,'qm','set'),ex.event(profile,'qm','wait'),ex.event(profile,'fq','wait'),ex.event(profile,'qm','set')],
    }
    for name,actions in cases.items():run(name,profile,ex.linear(actions))
    rng=random.Random(0x4D3509)
    for trial in range(100):
        actions=[]
        for i in range(rng.randrange(3,11)):
            pick=rng.randrange(6)
            if pick==0:actions.append(P)
            elif pick==1:actions.append(C)
            elif pick==2:actions.append(pi.Action('fence',rng.choice(profile.engines)))
            elif pick==3:actions.append(ex.event(profile,rng.choice(tuple(profile.keys)),rng.choice(('set','wait'))))
            else:
                cell=rng.choice(('left','right','side','g0','g1'));mode=rng.randrange(3)
                actions.append(pi.Action('op',rng.choice(profile.engines),reads=(cell,) if mode!=1 else (),writes=(cell,) if mode!=0 else ()))
        run(f'random/{trial}',profile,ex.linear(actions))
    # Positive randomized cohorts: real mixed plans, not only invalid noise.
    for trial in range(20):
        p=ex.profile(1+trial%3);prod,cons=ex.producer(p),ex.consumer(p)
        actions=[pi.Action('op','L',writes=('left','right')),*ex.handshake(p,'lm'),prod,cons,
                 *ex.handshake(p,'fq'),pi.Action('op','Q',reads=tuple(f'g{j}' for j in range(len(p.blocks)))),
                 *ex.handshake(p,'qf'),*ex.handshake(p,'ml')]
        if trial%2:actions.append(pi.Action('op','Q',reads=('side',)))
        run(f'positive-loop/{trial}',p,ex.loop(actions))
    # Fact oracle must reject an intentionally fabricated full M completion.
    x,a=run('fact-negative-control',profile,ex.linear([P,C]))
    ok,_,model,snaps=collect(x,a['commands'],False,True);assert ok
    bad=copy.deepcopy(a);bad['cuts'][1]['before']['pending'][x['engines'].index('F')][0]=0
    try:facts(x,bad,model,snaps)
    except AssertionError:result['invented_completion_rejected']=True
    else:raise AssertionError('fact checker missed false completion')
    verify_pin()
    text=json.dumps(result,indent=2)+'\n'
    if args.output:Path(args.output).write_text(text)
    print(text)
if __name__=='__main__':main()
