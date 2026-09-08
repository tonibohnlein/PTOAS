#!/usr/bin/env python3
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
"""Reproduce v0.2's review fixes. These are fact models, not native PTO kernels."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from eventlab import fixtures as f
from eventlab.islwrap import ISL, ISLError
from eventlab.symbolic import Program
from eventlab.reuse import allocate_symbolically
from eventlab.physical import lower, materialize, allocate, compare, verify_async, Allocation

ROOT = Path(__file__).resolve().parent


def store(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def two_reads():
    return f.model('review_same_lane', [], 'true', [
        f.op('WA','V',[],'true',['0'],writes=[f.access('VEC','0')]),
        f.op('WB','V',[],'true',['1'],writes=[f.access('VEC','4')]),
        f.op('RA','V',[],'true',['2'],reads=[f.access('VEC','0')]),
        f.op('RB','V',[],'true',['3'],reads=[f.access('VEC','4')])],[{}])


def require_pass(report):
    if report['status'] != 'PASS':
        raise RuntimeError('not an exhaustive pass: ' + repr(report))


def run(destination):
    destination.mkdir(parents=True, exist_ok=True)
    started=time.monotonic(); async_rows=[]; specializations=0; symbolic_domains=0
    p=Program(two_reads()); before=p.plan(refine_barriers=False); after=p.plan()
    a,b=lower(before,{}),lower(after,{})
    quality=compare(a,b); checked=verify_async(b,allocate(b));require_pass(checked)
    assert len(a.barriers)==2 and len(b.barriers)==1
    assert quality['removed_payload_order']==[(2,3)] and not quality['new_payload_order']
    async_rows.append({'case':'redundant_same_lane_barrier',**checked});specializations+=1
    bad=materialize(b.operations,b.handoffs,[]); bad_check=verify_async(bad,allocate(bad))
    assert bad_check['status']=='FAIL'
    store(destination/'same_lane_cleanup.json',{
        'scope':'same current code with refinement off/on, not native compiler arms',
        'before':a.metrics(),'after':b.metrics(),'quality':quality,
        'audit':after.audit,'asynchronous_check':checked,'missing_barrier_mutation':bad_check})

    # First-use barrier is derived for arbitrary N in the declared context,
    # including an empty loop. It is not only whole-static-site deletion.
    model=f.model('repeated_read', ['N'], '0<=N<=1000000', [
        f.op('W','V',[],'true',['0','0'],writes=[f.access('VEC','0',16)]),
        f.op('R','V',['i'],'0<=i<N',['1','i'],reads=[f.access('VEC','0',16)])],[])
    p=Program(model); old,new=p.plan(refine_barriers=False),p.plan();rows=[]
    for n in [0,1,2,8]:
        c0,c1=lower(old,{'N':n}),lower(new,{'N':n})
        r=verify_async(c1,allocate(c1));require_pass(r)
        rows.append({'N':n,'before':c0.metrics(),'after':c1.metrics(),'quality':compare(c0,c1),'async':r})
        async_rows.append({'case':'repeated_read','N':n,**r});specializations+=1
    store(destination/'repeated_reader.json',{'plan':new.report(),'specializations':rows})

    # These two inputs are the exact review fixture bytes (named differently
    # here). They deliberately retain the same bounded parameter context.
    for name in ['review_combined_gemm','review_combined_gemm_unrolled']:
        path=ROOT/'examples'/(name+'.json'); model=json.loads(path.read_text())
        p=Program(model); t=time.monotonic(); plan=p.plan(); planning_seconds=time.monotonic()-t
        keys=allocate_symbolically(plan)
        if not keys.accepted:
            raise RuntimeError('storage-derived key construction failed: '+name)
        symbolic_domains += len(keys.rules)
        rows=[]
        for parameters in model['scenarios']:
            c=lower(plan,parameters); assignment=keys.specialize(plan,c,parameters)
            if c.missing():raise RuntimeError('dense overlap check failed')
            # Independently check every concrete same-key reuse, including the
            # larger specializations for which state exploration is not run.
            for i,a in enumerate(c.handoffs):
                for b in c.handoffs[i+1:]:
                    if assignment.keys[a.id] == assignment.keys[b.id]:
                        if not (c.graph.precedes(('consume',a.id),('publish_issue',b.id)) or
                                c.graph.precedes(('consume',b.id),('publish_issue',a.id))):
                            raise RuntimeError('concrete rearm not implied')
            row={'parameters':parameters,'metrics':c.metrics(),'concrete_rearm_check':'PASS',
                 'assignment':asdict(assignment),'asynchronous_check':{'status':'NOT_RUN','reason':'bounded explorer size policy'}}
            if len(c.operations)<20:
                check=verify_async(c,assignment,max_states=50000);require_pass(check)
                row['asynchronous_check']=check
                async_rows.append({'case':name,'parameters':parameters,'allocator':'symbolic_storage',**check})
            rows.append(row);specializations+=1
        store(destination/(name+'.json'),{
            'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'hand-authored combined reduction, not captured native GEMM',
            'planning_seconds':planning_seconds,'plan':plan.report(),
            'symbolic_key_plan':keys.report(),'specializations':rows})

    large=[]
    for size in [131072,1<<40]:
        model=f.model('interval_size_stress',[],'true',[
            f.op('L','MTE2',[],'true',['0'],writes=[f.access('MAT','0',size)]),
            f.op('X','MTE1',[],'true',['1'],reads=[f.access('MAT','0',size)])],[{}])
        plan=Program(model).plan();keys=allocate_symbolically(plan);c=lower(plan,{})
        r=verify_async(c,keys.specialize(plan,c,{}));require_pass(r)
        large.append({'bytes':size,'represented_intervals':sum(len(o[m].intervals) for o in c.operations for m in ['reads','writes']),
                      'metrics':c.metrics(),'asynchronous_check':r})
        async_rows.append({'case':'interval_size_stress','bytes':size,**r});specializations+=1
    store(destination/'large_intervals.json',{
        'scope':'geometry/complexity test; one-terabyte extent is not a valid Ascend capacity claim',
        'specializations':large})

    # Preserve the larger query that remained expensive. A configured work limit
    # is recorded as UNPROVED, not silently changed into an accepted smaller input.
    path=ROOT/'examples/nested_three_slot_scaled.json'
    limit={'case':path.name,'max_isl_operations':100000,'status':'UNEXPECTED_ACCEPT'}
    try:
        Program(json.loads(path.read_text()),ISL(max_operations=limit['max_isl_operations'])).plan()
    except ISLError as error:
        if 'maximal number of operations' not in str(error):raise
        limit.update(status='UNPROVED_AT_CONFIGURED_BUDGET',reason=str(error))
    if limit['status']=='UNEXPECTED_ACCEPT':
        # This is not an error if a future isl improves; do not count it as the
        # expected limit regression without separately validating the plan.
        raise RuntimeError('known-limit behavior changed; review and validate it')
    store(destination/'known_limit.json',limit)

    # Strong separation of acceptance from the expected limitation.
    status={'status':'PASS_WITH_EXPLICIT_UNPROVED_CASE','version':'0.2.0','isl':p.isl.version,
            'review_model_specializations':specializations,
            'combined_model_symbolic_domains_proved':symbolic_domains,
            'exhaustive_positive_checks':len(async_rows),
            'exhaustive_positive_states':sum(r['states'] for r in async_rows),
            'new_unsafe_mutation_checks':1,'known_unproved_cases':1,
            'native_import_and_emission':'NOT_IMPLEMENTED','native_PTOAS':'NOT_RUN',
            'device_correctness':'NOT_RUN','device_timing':'NOT_RUN',
            'elapsed_seconds':round(time.monotonic()-started,3)}
    store(destination/'asynchronous_checks.json',async_rows)
    store(destination/'status.json',status)
    print(json.dumps(status,indent=2))
    return status

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('evidence/revision'))
    run(parser.parse_args().output)
