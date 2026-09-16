#!/usr/bin/env python3
"""Reproducible qualification, mixed-protocol, invariant and oracle checks.
Run from any directory with Python 3.10+. No solver, NPU, LLVM or network required.
"""
from __future__ import annotations
from dataclasses import replace, asdict
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import random
import time
from phase_interface import *
from examples import profile, producer, consumer, event, handshake, linear, loop, branched_loop
from full_history import FullHistory

HERE=Path(__file__).resolve().parent


def run_word(p,commands,exact=False,exit_check=True):
    m=PhaseInterface(p,exact);s=m.initial();o=FullHistory(p,exact)
    for j,cmd in enumerate(commands):
        ce=oe=None
        try:ns=m.transfer(s,cmd)
        except Rejected as e:ce=e.kind
        try:no=o.transfer(cmd)
        except Rejected as e:oe=e.kind
        assert ce==oe,(j,cmd,ce,oe)
        if ce:return {'accepted':False,'kind':ce,'index':j}
        s,o=ns,no
    if exit_check:
        ce=oe=None
        try:m.check_exit(s)
        except Rejected as e:ce=e.kind
        try:o.check_exit()
        except Rejected as e:oe=e.kind
        assert ce==oe
        if ce:return {'accepted':False,'kind':ce,'index':len(commands)}
    return {'accepted':True,'command_count':len(commands),'pair_queries':o.pair_queries}


def main():
    start=time.perf_counter();p=profile(2)
    P,C=producer(p),consumer(p)
    init=Action('op','L',writes=('left','right'),label='input_copy')
    readout=Action('op','Q',reads=('g0','g1'),label='output_user')
    named=[]
    def case(name,commands,expect,exact=False,exit_check=True):
        got=run_word(p,commands,exact,exit_check)
        assert got['accepted']==(expect=='ok'),(name,got,expect)
        if expect!='ok':assert got['kind']==expect,(name,got,expect)
        named.append({'name':name,'expected':expect,'exact_order_requested':exact,**got})

    case('block_overlap_without_software_M_F_event',[P,C],'ok',True)
    case('ordinary_input_ready_plus_block_interlock',[init,*handshake(p,'lm'),P,C],'ok',True)
    case('missing_input_readiness',[init,P,C],'payload')
    case('ordinary_output_completion_handoff',[P,C,*handshake(p,'fq'),readout],'ok')
    case('block_read_does_not_complete_output',[P,C,readout],'payload')
    case('coarse_full_operation_handshake_is_safe',[P,*handshake(p,'mf'),C],'ok')
    case('coarse_full_operation_handshake_loses_order',[P,*handshake(p,'mf'),C],'extra_order',True)
    case('operand_return_after_full_producer_completion',[init,*handshake(p,'lm'),P,C,
          *handshake(p,'ml'),init],'ok')
    case('block_path_is_not_operand_release',[P,C,*handshake(p,'fm'),
          Action('op','M',writes=('left',),label='overwrite_input')],'payload')
    case('block_return_does_not_cover_later_source_work',[P,Action('op','M',writes=('side',)),C,
          *handshake(p,'fm'),Action('op','M',reads=('side',),label='use_later_write')],'payload')
    raw_feedback=[event(p,'qm','set'),event(p,'qm','wait'),P,C,*handshake(p,'fq')]
    # The resource path carries consumption knowledge, but the identity copy
    # may begin a later GM output before it has read the next result block.
    # A repeated output cell therefore still needs explicit WAW ordering.
    case('consumption_relay_alone_does_not_order_next_output',raw_feedback*2,'payload')
    feedback=[*raw_feedback,Action('fence','F',label='ordinary_output_WAW')]
    case('resource_path_carries_consumption_to_return',feedback*2,'ok')
    case('balanced_listing_without_return_is_not_rearm',[
         event(p,'qm','set'),event(p,'qm','wait'),P,C,event(p,'qm','set')],'rearm')
    stale=[event(p,'qm','set'),event(p,'qm','wait'),P,C,event(p,'fq','set'),
           *handshake(p,'mq'),event(p,'qm','set'),event(p,'qm','wait'),
           event(p,'fq','wait'),event(p,'qm','set')]
    case('old_snapshot_does_not_acknowledge_new_consumption',stale,'rearm')
    case('two_final_producers_without_consumer',[P,P],'resource_balance')
    case('two_final_consumers_without_producer',[P,C,C],'resource_balance')
    case('fence_does_not_reset_readable_blocks',[P,Action('fence','M'),P],'resource_balance')
    case('live_ready_blocks_rejected_at_exit',[P],'exit_resource')
    case('reader_before_its_forward_producer',[C,P],'resource_balance')
    # Reuse can start after the last block read without waiting for unrelated
    # output-tail completion; leave next produced generation open for this query.
    case('block_reuse_without_full_output_completion',[P,C,P],'ok',False,False)
    case('next_output_write_still_needs_order',[P,C,P,C],'payload')
    case('local_fence_protects_repeated_output_writes',[P,C,Action('fence','F'),P,C],'ok')
    case('local_fence_may_strengthen_phase_order',[P,C,Action('fence','F'),P,C],'extra_order',True)

    # Invalid profile/import cases: these fail before synchronization reasoning.
    qualification=[]
    bad_actions=[('keep_producer',replace(P,mode='keep')),('keep_consumer',replace(C,mode='keep')),
      ('unknown_mode',replace(P,mode='disabled')),('unproved_layout',replace(C,layout='NZ2ND')),
      ('partial_consumer',replace(C,blocks=('u0',),writes=('g0',))),
      ('permuted_map',replace(C,blocks=('u1','u0'))),('wrong_engine',replace(P,engine='Q')),
      ('missing_operand',replace(P,reads=('left',))),('hidden_protected_reader',Action('op','Q',reads=('u0',))),
      ('missing_output',replace(C,writes=('g0',))),('wrong_event_endpoint',Action('wait','M',key='fq')),
      ('unexpected_event_effect',Action('set','F',key='fq',writes=('side',)))]
    for name,cmd in bad_actions:
        try:p.fragment(cmd)
        except QualificationError as e:qualification.append({'name':name,'rejected':True,'reason':str(e)})
        else:raise AssertionError(name)
    for name,cells,groups,entry in [
       ('unaligned_block',[replace(c,offset=1) if c.name=='u0' else c for c in p.cells.values()],list(p.groups.values()),True),
       ('partial_block',[replace(c,size=256) if c.name=='u0' else c for c in p.cells.values()],list(p.groups.values()),True),
       ('alias_into_protected_domain',list(p.cells.values())+[Cell('alias','ACC',0,512)],list(p.groups.values()),True),
       ('wrong_initial_permission',list(p.cells.values()),list(p.groups.values()),False)]:
        try:Profile(p.engines,cells,groups,p.keys,entry_writable=entry)
        except QualificationError as e:qualification.append({'name':name,'rejected':True,'reason':str(e)})
        else:raise AssertionError(name)
    # A short destination that would contradict the no-conversion full-block map.
    small=Profile(p.engines,[replace(c,size=256) if c.name=='g0' else c for c in p.cells.values()],p.groups.values(),p.keys)
    try:small.fragment(C)
    except QualificationError as e:qualification.append({'name':'partial_destination','rejected':True,'reason':str(e)})
    else:raise AssertionError('partial_destination')

    # Show that generalized comparison does more than checking new issue points.
    monitor=[]
    m=PhaseInterface(p,True);s=m.initial();o=FullHistory(p,True)
    first=Action('op','L',writes=('side',));s=m.transfer(s,first);o=o.transfer(first)
    second=Action('op','Q',reads=('left',),label='independent')
    for name,cmd,base,full,extra,internal in [
       ('completion_only_edge',second,s,o,(('T:L','C'),),()),
       ('access_begin_only_edge',P,s,o,(('T:L','b0'),),()),
       ('within_operation_block_serialization',P,m.initial(),FullHistory(p,True),(),(('e0','b1'),))]:
        kinds=[]
        for fn in (lambda:m.transfer(base,cmd,adversary_edges=extra,adversary_internal=internal),
                   lambda:full.transfer(cmd,adversary_edges=extra,adversary_internal=internal)):
            try:fn()
            except Rejected as e:kinds.append(e.kind)
            else:kinds.append('accepted')
        assert kinds==['extra_order','extra_order'],(name,kinds)
        monitor.append({'name':name,'compact':kinds[0],'full_history':kinds[1]})

    # Fully static collecting invariants. Every loop admits zero and arbitrarily
    # many visits. The unrelated Q-read branch does not introduce hidden guards.
    examples={
      'one_pair_exact':(linear([P,C]),True),
      'input_handoff_exact':(linear([init,*handshake(p,'lm'),P,C]),True),
      'resource_and_local_fence_loop':(loop([P,C,Action('fence','F')]),False),
      'mixed_ordinary_handoff_loop':(loop([init,*handshake(p,'lm'),P,C,*handshake(p,'ml'),Action('fence','F')]),False),
      'resource_consumption_relay_loop':(loop(feedback),False),
      'branch_loop':(branched_loop(p),False),
      'output_visibility_completion_loop':(loop([P,C,*handshake(p,'fq'),readout,*handshake(p,'qf')]),False)}
    certificates=[]
    for name,(cfg,exact) in examples.items():
        path=HERE/'certificates'/f'{name}.json';model=PhaseInterface(p,exact)
        result=analyze(cfg,model,path=path)
        assert result['accepted'],(name,result)
        checked=check_certificate(path)
        assert checked['states']==result['states']
        certificates.append({'name':name,**result,'recheck':checked})
    # Optional producer / compulsory consumer must fail on the skipped arm.
    bad=Program((Node(Action('nop'),(1,2)),Node(P,(2,)),Node(C,(3,)),Node(Action('nop'))),0,(3,))
    branch_negative=analyze(bad,PhaseInterface(p,False));assert branch_negative.get('kind')=='resource_balance'

    # Endpoint deletion mutations: count both meaningful failures and any
    # intentionally redundant placements; never assume every deletion is unsafe.
    mutations=[]
    for name,(cfg,exact) in examples.items():
        for i,node in enumerate(cfg.nodes):
            if node.action.kind not in ('set','wait','fence'):continue
            nodes=list(cfg.nodes);nodes[i]=Node(Action('nop',label='deleted_endpoint'),node.successors)
            result=analyze(replace(cfg,nodes=tuple(nodes)),PhaseInterface(p,exact))
            mutations.append({'case':name,'site':i,'removed':node.action.kind,
                              'accepted':result['accepted'],'kind':result.get('kind','accepted')})
    # Certificate mutation must not be rescued by rediscovering another invariant.
    obj=json.loads((HERE/'certificates'/'one_pair_exact.json').read_text());obj['invariant'][0]=[]
    mutant=HERE/'_bad_certificate.json';mutant.write_text(json.dumps(obj))
    try:check_certificate(mutant)
    except Rejected as e:assert e.kind=='certificate'
    else:raise AssertionError('bad certificate accepted')
    mutant.unlink()

    # Compare independent graph interpretation with compact projection on mixed
    # random words. Invalid choices do not mutate either state. Exact-order and
    # safety-only trials are recorded separately.
    rng=random.Random(20260915);statistics=Counter();total_pair_queries=0
    for trial in range(240):
        exact=trial%2==0;model=PhaseInterface(p,exact);state=model.initial();full=FullHistory(p,exact)
        oldqueries=0
        for pos in range(50):
            pick=rng.randrange(12)
            if pick<2:cmd=P if not state.readable else C
            elif pick==2:cmd=rng.choice([P,C])
            elif pick<7:
                c=rng.choice(['left','right','side','g0','g1']);eng=rng.choice(p.engines)
                mode=rng.choice(['R','W','RW']);cmd=Action('op',eng,reads=(c,) if 'R'in mode else (),writes=(c,) if 'W'in mode else (),label='random')
            elif pick==7:cmd=Action('fence',rng.choice(p.engines))
            else:
                key=rng.choice(tuple(p.keys));kind='wait' if state.live&(1<<model.ki[key]) else 'set'
                if pick==11:kind='wait' if kind=='set' else 'set'
                cmd=event(p,key,kind)
            ce=oe=None
            try:newstate=model.transfer(state,cmd)
            except Rejected as e:ce=e.kind
            try:newfull=full.transfer(cmd)
            except Rejected as e:oe=e.kind
            assert ce==oe,(trial,pos,asdict(cmd),ce,oe)
            statistics[('exact:' if exact else 'safety:')+(ce or 'accepted')]+=1
            if ce is None:
                total_pair_queries += newfull.pair_queries-full.pair_queries
                state,full=newstate,newfull
        if trial%60==0:print('random trials',trial,flush=True)

    # Explicit witness queries distinguish the block from whole operation and GM.
    full=FullHistory(p,False)
    for cmd in [P,C,P]:full=full.transfer(cmd)
    obs={name:av for name,av,rv in full.observations}
    checks={
      'producer_finish_does_not_gate_consumer_issue':not full.a.reach(obs['0:C'],obs['1:I']),
      'block1_write_does_not_gate_block0_read':not full.a.reach(obs['0:e1'],obs['1:b0']),
      'matching_block_ready_does_gate_read':full.a.reach(obs['0:e0'],obs['1:b0']),
      'block_read_releases_next_same_block_write':full.a.reach(obs['1:e0'],obs['2:b0']),
      'consumer_finish_does_not_gate_next_producer_issue':not full.a.reach(obs['1:C'],obs['2:I']),
      'GM_write_not_required_before_ACC_reuse':not full.a.reach(obs['1:o0e'],obs['2:b0'])}
    assert all(checks.values()),checks
    result={
      'revision':'0.9','profile':'final-block-pair-v1','scope':'reference-contract qualification and supplied-schema checking; not native UnitFlag validation or mode synthesis',
      'named_cases':named,'import_rejections':qualification,'generalized_monitor_adversaries':monitor,
      'certificates':certificates,'bad_branch':branch_negative,'endpoint_mutations':mutations,
      'boundary_queries':checks,'random_seed':20260915,'random_trials':240,'random_commands':12000,
      'random_verdict_counts':dict(statistics),'accepted_full_history_pair_queries':total_pair_queries,
      'certificate_total_states':sum(x['states'] for x in certificates),
      'certificate_successor_checks':sum(x['successor_checks'] for x in certificates),
      'negative_endpoint_mutations':sum(not x['accepted'] for x in mutations),
      'redundant_deletions_accepted':sum(x['accepted'] for x in mutations),
      'runtime_seconds':time.perf_counter()-start,
      'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(HERE.glob('*.py'))}}
    (HERE/'phase_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('named_cases','import_rejections','certificates','endpoint_mutations','source_sha256')},indent=2))

if __name__=='__main__':main()
