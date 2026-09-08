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
"""Reproduce the research-model campaign, without a native compiler or device."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json
import platform
import time

from eventlab import fixtures as f
from eventlab.symbolic import Program
from eventlab.reuse import allocate_symbolically
from eventlab.physical import Handoff, Allocation, lower, materialize, allocate, verify_async, compare
from eventlab.__main__ import concrete_report


def run(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter(); rows=[]; symbolic=[]; async_rows=[]; mutations=[]
    for data in f.all_models():
        program=Program(data); plan=program.plan()
        key_proof=allocate_symbolically(plan)
        if not key_proof.accepted:raise RuntimeError('symbolic key proof failed: '+data['name'])
        symbolic.append({'case':data['name'],'plan':plan.report(),'key_proof':key_proof.report()})
        for parameters in data['scenarios']:
            row=concrete_report(plan,parameters,symbolic=key_proof)
            row['case']=data['name']; rows.append(row)
            if row['mechanisms']['missing_memory_requirements'] or not row['assignment']['accepted']:
                raise RuntimeError('finite realization failed: '+data['name'])
    checks=[(f.ring(k),{'N':n}) for k in [1,2,3] for n in range(4)]
    checks += [(f.ring(2,True),{'O':o,'N':n}) for o,n in [(0,1),(2,0),(2,1),(2,2)]]
    checks += [(f.preload(),{'J':j,'K':k}) for j,k in [(0,0),(1,1),(2,1)]]
    checks += [(f.panel(),{'O':o,'J':j}) for o,j in [(1,0),(2,0),(2,1),(2,2)]]
    checks += [(f.conditional(),{'N':n,'TAKE':t}) for n,t in [(0,0),(3,0),(3,1)]]
    checks += [(f.bundle(),{}),(f.bundle(True),{}),(f.partial(),{}),(f.partial(True),{}),
               (f.accumulator(),{'O':2,'K':1})]
    for data,parameters in checks:
        plan=Program(data).plan();c=lower(plan,parameters)
        key_proof=allocate_symbolically(plan)
        for strategy,assignment in [('finite_chain_cover',allocate(c)),
                                    ('symbolic_keys',key_proof.specialize(plan,c,parameters))]:
            check=verify_async(c,assignment)
            async_rows.append({'case':data['name'],'parameters':parameters,'allocation':strategy,**check})
            if check['status']!='PASS': raise RuntimeError('asynchronous model failed: '+str(async_rows[-1]))

    precise=lower(Program(f.preload()).plan(),{'J':1,'K':1})
    broad=materialize(precise.operations,[Handoff(0,1,2,'MTE2','MTE1')])
    broad_check=verify_async(broad,allocate(broad))
    quality=compare(precise,broad)
    if broad_check['status']!='PASS' or quality['no_additional_payload_blocking']:
        raise RuntimeError('safe-but-serialized negative control failed')
    mutations.append({'case':'second_preload_captured','expected':'safe_but_adds_blocking',
                      'check':broad_check,'quality':quality})

    bundle=lower(Program(f.bundle()).plan(),{})
    separate=materialize(bundle.operations,[Handoff(0,0,2,'MTE1','M'),Handoff(1,1,2,'MTE1','M')])
    bundling=compare(separate,bundle)
    if bundling['new_payload_order'] or bundling['removed_payload_order']:
        raise RuntimeError('compatible coalescing changed ordering')

    for kind in ['deleted_ready','deleted_release','stale_generation','forced_key_reuse']:
        c=lower(Program(f.ring(1 if kind=='deleted_release' else 2)).plan(),{'N':2})
        hs=list(c.handoffs)
        if kind=='deleted_ready':
            hs.pop(next(i for i,h in enumerate(hs) if h.src_lane=='MTE2'))
        elif kind=='deleted_release':
            hs.pop(next(i for i,h in enumerate(hs) if h.src_lane=='V' and h.dst_lane=='MTE2'))
        elif kind=='stale_generation':
            i=next(i for i,h in enumerate(hs) if h.src_lane=='MTE2' and h.source==3)
            hs[i]=replace(hs[i],source=0)
        else:
            c=precise;hs=list(c.handoffs)
        hs=[replace(h,id=i) for i,h in enumerate(hs)]
        bad=materialize(c.operations,hs,c.barriers)
        assignment=allocate(bad)
        if kind=='forced_key_reuse':
            assignment=Allocation(True,{h.id:('MTE2','MTE1',0) for h in hs},{},{},'injected fault')
        check=verify_async(bad,assignment)
        if check['status']!='FAIL':raise RuntimeError('mutation not detected: '+kind)
        mutations.append({'case':kind,'expected':'FAIL','check':check})

    # Uniform-overlap versus command-count is measured separately; no combined score.
    examples={'independent_preloads':{'precise':precise.metrics(),'broad':broad.metrics(),
                                    'quality':quality},
              'compatible_bundle':{'before':separate.metrics(),'after':bundle.metrics(),
                                   'quality':bundling}}
    status={'status':'PASS','python':platform.python_version(),'isl':program.isl.version,
            'semantic_fixture_models':len(symbolic),'finite_specializations':len(rows),
            'symbolic_key_domains_proved':sum(len(x['key_proof']['rules']) for x in symbolic),
            'positive_async_checks':len(async_rows),'positive_async_states':sum(r['states'] for r in async_rows),
            'negative_controls':len(mutations),'scope':'symbolic input facts plus finite target-model realizations',
            'native_import_and_emission':'NOT_IMPLEMENTED','native_compiler':'NOT_RUN',
            'device_correctness':'NOT_RUN','device_timing':'NOT_RUN',
            'elapsed_seconds':round(time.perf_counter()-begin,3)}
    for name,result in [('campaign',{'status':status,'symbolic':symbolic,'specializations':rows,
                                     'asynchronous_checks':async_rows,'negative_controls':mutations}),
                        ('quality_examples',examples),('campaign_status',status)]:
        (destination/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(status,indent=2))
    return status

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('evidence/reproduced'))
    run(parser.parse_args().output)
