# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native S1/S2 acceptance: unchanged input hashes, strict dispatch, real mutations.
No libisl dependency. Device and <=2x compilation acceptance remain separate.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

def main():
    if not __debug__: raise RuntimeError('Assertions must be enabled')
    parser=argparse.ArgumentParser()
    parser.add_argument('--driver',type=Path,required=True)
    parser.add_argument('--opt',type=Path,required=True)
    parser.add_argument('--python-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    sys.path.insert(0,str(args.python_root))
    from observations import population,analyze,SERIAL_DRIVER
    from compare_boundaries import run as observe
    rows=[]
    def run(name,command):
        before=time.perf_counter()
        result=subprocess.run([str(x) for x in command],text=True,capture_output=True)
        elapsed=time.perf_counter()-before
        (args.output/(name+'.stdout')).write_text(result.stdout)
        (args.output/(name+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,(name,result.returncode,result.stderr[-5000:],result.stdout[-5000:])
        return result,elapsed
    for case in population():
        if case['case_id'] not in ('one_buffer','two_buffer','three_buffer','qk_matmul'): continue
        name=case['case_id']; source=case['source']
        # Actual CLI admission at the ordinary pipeline point, with quota zero:
        # the new implementation is not permitted to invoke the reference.
        output=args.output/(name+'.pto')
        command=[sys.executable,'-c',SERIAL_DRIVER,args.python_root,'--pto-arch=a3','--pto-level=level3',
                 '--enable-insert-sync','--insert-sync-planner=structured',
                 '--insert-sync-logical-work-budget=0',
                 '--insert-sync-gm-alias=assume-disjoint-arguments','--emit-pto-ir',source,'-o',output]
        _,seconds=run(name,command)
        report=analyze(output)
        unsynchronized=args.output/(name+'.unsynchronized.pto')
        run(name+'.unsynchronized',[sys.executable,'-c',SERIAL_DRIVER,args.python_root,
            '--pto-arch=a3','--pto-level=level3','--emit-pto-ir',source,'-o',unsynchronized])
        original=analyze(unsynchronized)
        for field in ('payload','allocations','views','abi'):
            assert report[field]==original[field],(name,field,'non-sync projection changed')
        executed=[]
        for scenario in case['scenarios']:
            before,_=observe(unsynchronized,scenario)
            after,metric=observe(output,scenario)
            assert before.payload==after.payload,(name,scenario['name'],'payload replay changed')
            assert not after.tokens,(name,scenario['name'],'outstanding notifications')
            if name=='qk_matmul' and scenario['arguments'][4]>0:
                first=next(i for i,p in enumerate(after.payload) if p[0]=='pto.textract')
                assert after.before[first]['completed'].get('PIPE_MTE2',-1)==0,(name,scenario['name'],'first panel publication broadened')
            executed.append(dict(scenario=scenario['name'],metrics=metric))
        assert any(x.get('pto.insert_sync.producer')=='"structured"' for x in report['status_attributes']),report
        rows.append(dict(case=name,seconds=seconds,mechanisms=report['mechanisms'],
            scalar_sites=report['sync_control'],executed=executed,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        # Use the driver on unchanged, explicitly addressed input as well.
        for mutation in ('none','drop-wait','drop-set','duplicate-set','wrong-key','drop-retirement','wrong-participation'):
            result,_=run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
            verdict=json.loads(result.stdout)
            assert verdict['expected'] and verdict['atomic'],verdict
    assert len(rows)==4,rows
    fixtures=here/'structured_inputs'
    for name,mutation in (('independent_preloads','none'),('independent_preloads','late-set'),
                          ('nested_loop','expect-unsupported'),('unknown_guard','expect-unsupported')):
        run(name+'.'+mutation,[args.driver,fixtures/(name+'.pto'),mutation,args.output/(name+'.'+mutation+'.pto')])
    boundary_rows=[]
    for name in ('boundary_preloads','boundary_periodic','boundary_periodic_equiv','boundary_mixed'):
        source=fixtures/(name+'.pto'); output=args.output/(name+'.pto')
        result,elapsed=run(name,[args.driver,source,'none',output])
        verdict=json.loads(result.stdout)
        assert verdict['accepted'] and verdict['atomic'],verdict
        before_report=analyze(source);after_report=analyze(output)
        for field in ('payload','allocations','views','abi'):
            assert before_report[field]==after_report[field],(name,field)
        executions=[]
        for trips in (-1,0,1,2,3,4,5,6,7,11,16):
            scenario={'name':'trips_'+str(trips),'arguments':['src','dst',trips,True]}
            before,_=observe(source,scenario);after,metrics=observe(output,scenario)
            assert before.payload==after.payload,(name,trips,'payload changed')
            assert not after.tokens,(name,trips,'unused notification')
            if trips>0:
                first=next(i for i,p in enumerate(after.payload) if p[0]=='pto.tabs')
                assert after.before[first]['completed'].get('PIPE_MTE2',-1)==0,(name,trips,'late preload acquired')
            executions.append(dict(trips=trips,metrics=metrics))
        boundary_rows.append(dict(case=name,seconds=elapsed,verdict=verdict,executed=executions,
                                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for mutation in ('drop-wait','drop-set','duplicate-set','wrong-key','drop-retirement',
                         'wrong-first','wrong-last','wrong-existence'):
            run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
    run('boundary_preloads.late',[args.driver,fixtures/'boundary_preloads.pto','late-boundary-set',
                                 args.output/'boundary_preloads.late.pto'])
    run('boundary_late_bound.refuse',[args.driver,fixtures/'boundary_late_bound.pto','expect-unsupported',
                                     args.output/'boundary_late_bound.refuse.pto'])
    summary=dict(boundary_rows=boundary_rows,status='passed',rows=rows,driver_sha256=hashlib.sha256(args.driver.read_bytes()).hexdigest(),
                 timing='single diagnostics only; repeated matched <=2x campaign NOT_RUN',device='NOT_RUN')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
