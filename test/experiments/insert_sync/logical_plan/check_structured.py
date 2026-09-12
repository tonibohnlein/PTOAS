# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native S1-S7 acceptance: unchanged input hashes, strict dispatch, real mutations.
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
    from s6_report import parse_reports, validate_report
    rows=[]; qk_case=None
    def run(name,command,audit=False):
        env=dict(os.environ)
        env.pop('PTOAS_STRUCTURED_PLAN_JSON',None)
        env.pop('PTOAS_LOGICAL_TRACE',None)
        if audit: env['PTOAS_STRUCTURED_PLAN_JSON']='1'
        before=time.perf_counter()
        result=subprocess.run([str(x) for x in command],text=True,capture_output=True,env=env)
        elapsed=time.perf_counter()-before
        (args.output/(name+'.stdout')).write_text(result.stdout)
        (args.output/(name+'.stderr')).write_text(result.stderr)
        assert result.returncode==0,(name,result.returncode,result.stderr[-5000:],result.stdout[-5000:])
        return result,elapsed
    run('s4_utilities',[sys.executable,here/'check_s4_utilities.py'])
    run('s6_utilities',[sys.executable,here/'check_s6_utilities.py'])
    for case in population():
        if case['case_id'] not in ('one_buffer','two_buffer','three_buffer','four_use','online_softmax','qk_matmul','q_proj'): continue
        name=case['case_id']; source=case['source']
        if name=='qk_matmul': qk_case=case
        # Actual CLI admission at the ordinary pipeline point, with quota zero:
        # the new implementation is not permitted to invoke the reference.
        output=args.output/(name+'.pto')
        command=[sys.executable,'-c',SERIAL_DRIVER,args.python_root,'--pto-arch=a3','--pto-level=level3',
                 '--enable-insert-sync','--insert-sync-planner=structured',
                 '--insert-sync-logical-work-budget=0',
                 '--insert-sync-gm-alias=assume-disjoint-arguments','--emit-pto-ir',source,'-o',output]
        _,seconds=run(name,command)
        report=analyze(output)
        # Separate diagnostic compilation: per-deletion proof queries must NOT
        # enter the reported normal compile time or the paired campaign.
        audited=args.output/(name+'.audited.pto')
        audit_command=list(command);audit_command[-1]=audited
        diagnostic,_=run(name+'.audit',audit_command,audit=True)
        reports=parse_reports(diagnostic.stderr)
        assert len(reports)==1,(name,'expected one kernel report')
        accounting=validate_report(reports[0])
        counts=report['mechanisms']
        actual_sites=counts['sets']+counts['waits']+sum(counts['named'].values())+counts['PIPE_ALL']
        assert accounting['static_sites']==actual_sites,(name,accounting,counts)
        assert output.read_bytes()==audited.read_bytes(),(name,'diagnostic mode changed IR')
        (args.output/(name+'.plan.json')).write_text(json.dumps(reports[0],indent=2)+'\n')
        if name=='q_proj':
            assert actual_sites<=76,(name,'whole-startup refinement did not improve 78-site baseline',counts)
            assert sum(u['coalesced_sites'] for u in accounting['units'])>0,(name,'startup sites not coalesced')
            for mutation in ('narrow-coalesced','duplicate-coalesced','late-coalesced-set'):
                result,_=run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
                verdict=json.loads(result.stdout)
                assert verdict['mutation_applied'] and verdict['expected'] and verdict['atomic'],verdict
        if name=='four_use':
            # Quality regression, separate from correctness: after all barriers
            # exist, remove a redundant pair without changing survivor keys.
            counts=report['mechanisms']
            assert counts['sets']==counts['waits'] and counts['sets']<=18,(name,counts)
            assert sum(counts['named'].values())<=2 and counts['PIPE_ALL']==1,(name,counts)
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
            scalar_sites=report['sync_control'],keys_by_direction=report['event_ids_by_direction'],executed=executed,
            s6_accounting=accounting,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        # Use the driver on unchanged, explicitly addressed input as well.
        for mutation in ('none','drop-wait','drop-set','duplicate-set','wrong-key','drop-retirement','wrong-participation'):
            result,_=run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
            verdict=json.loads(result.stdout)
            assert verdict['expected'] and verdict['atomic'],verdict
    assert len(rows)==7,rows
    fixtures=here/'structured_inputs'
    # A common period is not an exact truth partition for comparisons of two
    # residue-valued expressions. Refuse before dropping any physical phase.
    for name in ('ordinal_coperiodic_compare','ordinal_periodic_constant_compare'):
        result,_=run(name,[args.driver,fixtures/(name+'.pto'),'expect-unsupported',args.output/(name+'.pto')])
        verdict=json.loads(result.stdout)
        assert not verdict['accepted'] and verdict['expected'] and verdict['atomic'],verdict
        assert 'payload guard is outside' in verdict['reason'],verdict
    for name,mutation in (('independent_preloads','none'),('independent_preloads','late-set'),
                          ('nested_loop','none'),('unknown_guard','none')):
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
    # Whole-region choices are invariant through the admitted wrapper nest.
    # These are generated structural fixtures, not unchanged benchmark kernels.
    invocation_rows=[]
    shapes=[(0,0),(1,1),(2,1),(1,2),(2,3),(3,2)]
    for name in ('nested_preloads','nested_three_levels','nested_choice','nested_bypass'):
        source=fixtures/(name+'.pto');output=args.output/(name+'.pto')
        result,seconds=run(name,[args.driver,source,'none',output])
        verdict=json.loads(result.stdout)
        assert verdict['accepted'] and verdict['atomic'],verdict
        original=analyze(source);actual=analyze(output)
        for field in ('payload','allocations','views','abi'):
            assert actual[field]==original[field],(name,field)
        assert actual['mechanisms']['PIPE_ALL']==1,(name,'not one function retirement drain')
        executions=[]
        for outer,middle in shapes:
            for trips in (-1,0,1,2,5):
                for take in (False,True):
                    scenario={'name':f'{outer}_{middle}_{trips}_{take}',
                              'arguments':['src','dst',trips,take,outer,middle]}
                    before,_=observe(source,scenario);after,metrics=observe(output,scenario)
                    assert before.payload==after.payload,(name,scenario,'payload changed')
                    assert not after.tokens,(name,scenario,'outstanding invocation notifications')
                    # Every invocation has two preloads. Check the FIRST
                    # relevant reader at each reset, not only at function entry.
                    if trips>0:
                        length=2+2*trips+2
                        for start in range(0,len(after.payload),length):
                            assert after.payload[start][0]=='pto.tload'
                            assert after.payload[start+2][0]=='pto.tabs'
                            assert after.before[start+2]['completed'].get('PIPE_MTE2',-1)==start,(
                                name,scenario,start,'current independent second preload acquired')
                    executions.append(dict(scenario=scenario['name'],metrics=metrics))
        for mutation in ('drop-wait','drop-set','wrong-key','duplicate-set','drop-retirement',
                         'wrong-first','wrong-last','wrong-existence','wrong-invocation','late-nested-set'):
            response,_=run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
            evidence=json.loads(response.stdout)
            assert evidence['mutation_applied'] and evidence['expected'] and evidence['atomic'],evidence
        if name=='nested_three_levels':
            run(name+'.wrong-frame',[args.driver,source,'wrong-invocation-frame',args.output/(name+'.wrong-frame.pto')])
        invocation_rows.append(dict(case=name,seconds=seconds,verdict=verdict,executed=executions,
                                    mechanisms=actual['mechanisms'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    for name in ('nested_varying_bound','nested_varying_choice','nested_mixed_sequence'):
        run(name,[args.driver,fixtures/(name+'.pto'),'expect-unsupported',args.output/(name+'.pto')])

    # Stronger real-input anchor: the original QK above remains unchanged.
    # This ADDITIONAL derivative repeats that same operation sequence twice.
    assert qk_case is not None
    original_text=qk_case['source'].read_text()
    first=original_text.index('  pto.tload')
    last=original_text.rfind('  return')
    assert first<last and '%__oahs_outer' not in original_text
    repeated_text=(original_text[:first]+'  %__oahs_two = arith.constant 2 : index\n'
        +'  scf.for %__oahs_outer = %c0_index to %__oahs_two step %c1_index {\n'
        +original_text[first:last]+'  }\n'+original_text[last:])
    derived=args.output/'qk_nested.derived.pto';derived.write_text(repeated_text)
    result,seconds=run('qk_nested',[args.driver,derived,'none',args.output/'qk_nested.pto'])
    verdict=json.loads(result.stdout)
    assert verdict['accepted'] and verdict['atomic'],verdict
    for scenario in qk_case['scenarios']:
        before,_=observe(derived,scenario);after,metrics=observe(args.output/'qk_nested.pto',scenario)
        assert before.payload==after.payload and not after.tokens,scenario
        if scenario['arguments'][4]>0:
            assert len(after.payload)%2==0
            length=len(after.payload)//2
            for start in (0,length):
                first_reader=next(i for i in range(start,start+length) if after.payload[i][0]=='pto.textract')
                assert after.before[first_reader]['completed'].get('PIPE_MTE2',-1)==start,(
                    scenario,start,'nested QK readiness broadened')
    invocation_rows.append(dict(case='qk_nested_derivative',seconds=seconds,verdict=verdict,
        original_source_sha256=hashlib.sha256(qk_case['source'].read_bytes()).hexdigest(),
        derived_source_sha256=hashlib.sha256(derived.read_bytes()).hexdigest()))
    ordinal_rows=[]
    for name in ('ordinal_offset','ordinal_stride','ordinal_constant_slot',
                 'ordinal_shifted_mask','ordinal_signed_remainder','ordinal_startup',
                 'ordinal_scaled_startup','ordinal_slots'):
        source=fixtures/(name+'.pto');output=args.output/(name+'.pto')
        result,seconds=run(name,[args.driver,source,'none',output])
        verdict=json.loads(result.stdout)
        assert verdict['accepted'] and verdict['requirements']>0 and verdict['handoffs']>0,verdict
        original=analyze(source);actual=analyze(output)
        for field in ('payload','allocations','views','abi'):
            assert actual[field]==original[field],(name,field)
        executed=[]
        for upper in (-1,0,1,2,3,4,5,6,7,8,9,16,23):
            scenario={'name':str(upper),'arguments':['src','dst',upper]}
            before,_=observe(source,scenario);after,metrics=observe(output,scenario)
            assert before.payload==after.payload and not after.tokens,(name,upper)
            assert actual['mechanisms']['PIPE_ALL']==1,(name,'retirement policy')
            executed.append(dict(upper=upper,metrics=metrics))
        for mutation in ('drop-wait','drop-set','duplicate-set','wrong-key','drop-retirement'):
            run(name+'.'+mutation,[args.driver,source,mutation,args.output/(name+'.'+mutation+'.pto')])
        ordinal_rows.append(dict(case=name,seconds=seconds,mechanisms=actual['mechanisms'],
            keys_by_direction=actual['event_ids_by_direction'],executed=executed))
    for name,mutation in (('ordinal_startup','wrong-startup'),
                          ('ordinal_startup','wrong-empty-case'),
                          ('ordinal_scaled_startup','wrong-startup'),
                          ('ordinal_slots','wrong-ordinal')):
        result,_=run(name+'.'+mutation,[args.driver,fixtures/(name+'.pto'),mutation,
                                        args.output/(name+'.'+mutation+'.pto')])
        verdict=json.loads(result.stdout)
        assert verdict['mutation_applied'] and verdict['expected'] and verdict['atomic'],verdict
    for name in ('ordinal_negative_lower','ordinal_dynamic_step'):
        run(name,[args.driver,fixtures/(name+'.pto'),'expect-unsupported',args.output/(name+'.pto')])
    # Raw native import, not a pipeline whose canonicalization could hide the
    # signed-i1 bug. Empty imported models must fail the handoff/readiness checks.
    boolean_outputs=[]
    for name in ('signed_i1_periodic','signed_i1_control'):
        source=fixtures/(name+'.pto');output=args.output/(name+'.pto')
        result,_=run(name,[args.driver,source,'none',output]);verdict=json.loads(result.stdout)
        assert verdict['accepted'] and verdict['requirements']>0 and verdict['handoffs']>0,verdict
        observed=[]
        for upper in (0,1,2,3,7):
            scenario={'name':str(upper),'arguments':['src','dst',upper,True]}
            before,_=observe(source,scenario);after,_=observe(output,scenario)
            assert before.payload==after.payload and not after.tokens,(name,upper)
            for i,payload in enumerate(after.payload):
                if payload[0]=='pto.tabs':
                    assert after.before[i]['completed'].get('PIPE_MTE2',-1)>=i-1,(name,upper,i)
            observed.append((after.payload,after.before))
        boolean_outputs.append(observed)
    assert boolean_outputs[0]==boolean_outputs[1],'equivalent Boolean guards changed actual execution/order'
    # New synthetic split-K GEMM, NOT the frozen historical workload. It
    # exercises the exact initial/steady payload split with ordinary events.
    gemm_source=fixtures/'s6_startup_gemm.pto';gemm_output=args.output/'s6_startup_gemm.pto'
    response,gemm_seconds=run('s6_startup_gemm',[args.driver,gemm_source,'none',gemm_output],audit=True)
    verdict=json.loads(response.stdout)
    assert verdict['accepted'] and verdict['atomic'],verdict
    gemm_reports=parse_reports(response.stderr)
    assert len(gemm_reports)==1
    gemm_accounting=validate_report(gemm_reports[0])
    original=analyze(gemm_source);compiled=analyze(gemm_output)
    for field in ('payload','allocations','views','abi'):
        assert original[field]==compiled[field],('s6_startup_gemm',field)
    assert compiled['mechanisms']['PIPE_ALL']==1
    scenario={'name':'M16_N256_K2048','arguments':['a','b','c']}
    before,_=observe(gemm_source,scenario);after,metrics=observe(gemm_output,scenario)
    assert before.payload==after.payload and not after.tokens
    for mutation in ('drop-wait','drop-set','wrong-key','duplicate-set','drop-retirement',
                     'narrow-coalesced','duplicate-coalesced','late-coalesced-set'):
        result,_=run('s6_startup_gemm.'+mutation,[args.driver,gemm_source,mutation,
                       args.output/('s6_startup_gemm.'+mutation+'.pto')])
        verdict=json.loads(result.stdout)
        assert verdict['mutation_applied'] and verdict['expected'] and verdict['atomic'],verdict
    (args.output/'s6_startup_gemm.plan.json').write_text(json.dumps(gemm_reports[0],indent=2)+'\n')
    gemm_row=dict(case='s6_startup_gemm',source_sha256=hashlib.sha256(gemm_source.read_bytes()).hexdigest(),
                  synthetic=True,historical_gemm=False,accounting=gemm_accounting,executed=metrics,
                  diagnostic_seconds=gemm_seconds,device='NOT_RUN')
    summary=dict(s6_gemm=gemm_row,ordinal_rows=ordinal_rows,invocation_rows=invocation_rows,boundary_rows=boundary_rows,status='passed',rows=rows,driver_sha256=hashlib.sha256(args.driver.read_bytes()).hexdigest(),
                 timing='single diagnostics only; repeated matched <=2x campaign NOT_RUN',device='NOT_RUN')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # S7 is an explicit additional contract experiment. The seven conservative
    # cases above remain unchanged; this gate must prove real native rule hits.
    run('s7_utilities',[sys.executable,here/'check_s7_utilities.py'])
    run('s7_hardware',[sys.executable,here/'check_hardware.py',
        '--driver',args.driver,'--opt',args.opt,'--python-root',args.python_root,
        '--output',args.output/'hardware'])
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
