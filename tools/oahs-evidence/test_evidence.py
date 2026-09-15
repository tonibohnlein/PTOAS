# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tests of evidence machinery. The fake compiler is NEVER native qualification."""
from __future__ import annotations
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import evidence as e
import capture_compiler as cc
import collect_records as cr

PAYLOAD = '''module {
  func.func @kernel() {
    %x = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%src) outs(%x)
    pto.tabs ins(%x) outs(%dst)
    return
  }
}
'''
FAKE = r'''#!INTERPRETER
# Fake compiler for tests. It does not parse PTO or check any hardware property.
import pathlib, sys, time
args=sys.argv[1:]
source=pathlib.Path(next(a for a in args if pathlib.Path(a).suffix in ('.pto','.mlir')))
text=source.read_text()
if 'TEST_TIMEOUT' in text:
    time.sleep(5)
if 'TEST_ERROR' in text:
    print('synthetic diagnostic', file=sys.stderr);sys.exit(7)
if 'TEST_NO_OUTPUT' in text:sys.exit(0)
target=pathlib.Path(args[args.index('-o')+1])
if '--emit-pto-ir' in args:
    text=text.replace('    return','    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n    pto.barrier <PIPE_ALL>\n    return')
    target.write_text(text)
else:target.write_text('// synthetic compiler test output, NOT kernel lowering\n')
print('synthetic stdout');print('synthetic stderr',file=sys.stderr)
'''


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'input.pto'; self.raw.write_text(PAYLOAD)
        self.log = self.root / 'enumeration.log'; self.log.write_text('synthetic fixture inventory only\n')
        self.compiler = self.root / 'compiler'
        self.compiler.write_text(FAKE.replace('INTERPRETER', sys.executable + ' -S')); self.compiler.chmod(0o755)
        self.spec = dict(schema=e.SCHEMA, source_revisions={'PTOAS': e.PRODUCTION_REVISION, 'PyPTO':'a'*40, 'pypto-lib':'b'*40},
                         cohorts={c:dict(collected=True, collection_argv=['synthetic-enumerator', c], enumeration_log='enumeration.log') for c in e.COHORTS},
                         regressions={r:dict(case_ids=['case4'], evidence='enumeration.log') for r in ('small_gemm','attention_refusals','prefill')}, inputs=[])
        for i, c in enumerate(e.COHORTS):
            self.spec['inputs'].append(dict(id=f'case{i}',cohort=c,raw='input.pto',prepared='input.pto',
                expectation='supported',authored='none',preparation=dict(kind='identity'),
                contracts=dict(architecture='a3',hardware='test-only',alias='may-alias',abi='test-only',memory_planner='test-only'),
                compile_args=['--pto-arch=a3','--pto-level=level3']))
        self.config = dict(source_revision=e.PRODUCTION_REVISION,
            artifacts={'compiler':str(self.compiler), 'interpreter':sys.executable},
            environment={'PATH':os.defpath}, versions={k:'synthetic fixture, not qualified' for k in ('llvm_mlir','pto_isa','cann','runtime','device_profile')},
            argv=dict(sync=['{compiler}','{flags}','--enable-insert-sync','--emit-pto-ir','{input}','-o','{output}'],
                      lower=['{compiler}','{flags}','{input}','-o','{output}']))

    def tearDown(self):self.tmp.cleanup()

    def put(self, name, value):
        path=self.root/name;path.write_bytes(e.canonical(value));return path

    def capture(self, spec=None, name='capture'):
        path=self.put(name+'.spec.json', self.spec if spec is None else spec)
        result=e.capture(path,self.root/name)
        return self.root/name,result

    def baseline(self, spec=None):
        cap,_=self.capture(spec)
        cfg=self.put('toolchain.json',self.config)
        return cap,e.run_baseline(cap,cfg,self.root/'run',timeout=1)

    def receipt(self, cap, run, mode='generated'):
        proof=self.root/'proof.json';proof.write_text('{"status":"test-only"}\n')
        result=dict(schema=e.SCHEMA,kind='candidate_receipts',planner='oahs',capture_id=run['capture_id'],
                    source_revision='1'*40,toolchain_id='2'*64,cases=[])
        for row in run['cases']:
            output=self.root/'run'/row['id']/'sync.pto'
            result['cases'].append(dict(id=row['id'],handling=mode,prepared_sha256=row['prepared_sha256'],
                contracts_sha256=row['contracts_sha256'],verification_status='passed',
                output=dict(path=str(output),sha256=e.sha(output.read_bytes())),
                verification=dict(path=str(proof),sha256=e.sha(proof.read_bytes()))))
        return self.put('candidate.json',result)


class CaptureTests(Fixture):
    def test_capture_integrity_and_distinct_contexts(self):
        root,cap=self.capture();self.assertTrue(cap['inventory_complete']);self.assertEqual(5,len(cap['cases']))
        # Same bytes across contexts do not collapse five expected cases into one.
        self.assertEqual(1,len({c['prepared']['sha256'] for c in cap['cases']}));self.assertEqual(cap,e.verify_capture(root))

    def test_missing_input_remains(self):
        self.spec['inputs'][1]['prepared']='absent.pto'
        root,cap=self.capture();self.assertFalse(cap['inventory_complete']);self.assertEqual('missing',cap['cases'][1]['state']);self.assertEqual(5,len(cap['cases']))

    def test_empty_population_never_qualifies(self):
        self.spec['inputs']=[]
        _,cap=self.capture();self.assertFalse(cap['inventory_complete'])

    def test_missing_cohort_decl_rejected(self):
        del self.spec['cohorts']['pypto_ops']
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_incomplete_enumeration_retained(self):
        self.spec['cohorts']['pypto_ops']['collected']=False
        _,cap=self.capture();self.assertFalse(cap['inventory_complete']);self.assertTrue(cap['missing'])

    def test_missing_regression_inventory_blocks(self):
        self.spec['regressions']['attention_refusals']['case_ids']=[]
        _,cap=self.capture();self.assertFalse(cap['inventory_complete'])

    def test_duplicate_id_rejected(self):
        self.spec['inputs'].append(self.spec['inputs'][0])
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_unsafe_id_rejected(self):
        self.spec['inputs'][0]['id']='../case'
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_invalid_needs_reason(self):
        self.spec['inputs'][0]['expectation']='invalid'
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_unknown_classification_incomplete(self):
        self.spec['inputs'][0]['expectation']='unclassified'
        _,cap=self.capture();self.assertFalse(cap['inventory_complete'])

    def test_metadata_required(self):
        del self.spec['inputs'][0]['contracts']['alias']
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_identity_does_not_allow_rewrite(self):
        (self.root/'modified.pto').write_text(PAYLOAD+'// edit\n')
        self.spec['inputs'][0]['prepared']='modified.pto'
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_external_preparation_needs_command(self):
        self.spec['inputs'][0]['preparation']={'kind':'external'}
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_external_preparation_needs_evidence(self):
        self.spec['inputs'][0]['preparation']={'kind':'external','argv':['prepare']}
        _,cap=self.capture();self.assertFalse(cap['inventory_complete'])

    def test_mutated_object_refused(self):
        root,cap=self.capture();(root/cap['cases'][0]['raw']['path']).write_bytes(b'changed')
        with self.assertRaises(e.EvidenceError):e.verify_capture(root)

    def test_mutated_manifest_refused(self):
        root,cap=self.capture();cap['inventory_complete']=False;(root/'capture.json').write_bytes(e.canonical(cap))
        with self.assertRaises(e.EvidenceError):e.verify_capture(root)

    def test_existing_destination_refused(self):
        self.capture()
        with self.assertRaises(e.EvidenceError):self.capture()

    def test_symlink_refused(self):
        link=self.root/'link.pto';link.symlink_to(self.raw)
        self.spec['inputs'][0]['raw']='link.pto'
        _,cap=self.capture();self.assertEqual('missing',cap['cases'][0]['state'])

    def test_streaming_binary_identity(self):
        data=bytes(range(256))*12000
        path=self.root/'binary';path.write_bytes(data)
        self.assertEqual(dict(sha256=e.sha(data),bytes=len(data)),e.file_identity(path))

    def test_duplicate_json_keys_refused(self):
        path=self.root/'bad.json';path.write_text('{"x":1,"x":2}')
        with self.assertRaises(e.EvidenceError):e.load(path)

    def test_nonfinite_json_refused(self):
        path=self.root/'bad.json';path.write_text('{"x":NaN}')
        with self.assertRaises(e.EvidenceError):e.load(path)


class InventoryTests(unittest.TestCase):
    def test_comments_strings_and_attributes(self):
        text=b'''// pto.set_flag fake
module {
  func.func @x() attributes {pto.label = "pto.set_flag"} {
    /* pto.wait_flag */
    %x = "pto.tload"(%arg) : () -> ()
    pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    pto.barrier <PIPE_M>
    pto.barrier <PIPE_ALL>
    return
  }
}'''
        inv=e.text_inventory(text)
        self.assertEqual(1,inv['set_sites']);self.assertEqual(1,inv['wait_sites'])
        self.assertEqual(1,inv['named_barriers']);self.assertEqual(1,inv['pipe_all_sites'])
        self.assertEqual(1,inv['operations']['pto.tload']);self.assertIsNone(inv['executed_mechanisms'])
        self.assertEqual('unclassified',inv['terminal_retirement'])

    def test_no_reverse_pair_release_inference(self):
        inv=e.text_inventory(b'pto.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]\npto.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]')
        self.assertEqual('unclassified',inv['acknowledgment_roles'])


class RunnerTests(Fixture):
    def test_full_synthetic_campaign(self):
        cap,result=self.baseline();self.assertEqual('completed',result['status']);self.assertEqual(5,result['totals']['compiled'])
        checked=e.verify_run(self.root/'run');self.assertEqual(result,checked)
        self.assertEqual('not_executed',result['device']);self.assertEqual('not_executed',result['correctness'])
        for row in result['cases']:
            self.assertEqual(2,len(row['stages']));self.assertEqual('existing_completed',row['handling'])
            self.assertNotIn('--enable-insert-sync',row['stages'][1]['argv'])
            self.assertEqual(row['stages'][0]['output_sha256'],row['stages'][1]['input_sha256'])

    def test_expected_invalid_is_not_timeout(self):
        self.raw.write_text('TEST_ERROR\n')
        for case in self.spec['inputs']:case.update(expectation='invalid',invalid_reason='synthetic invalid',invalid_diagnostic='synthetic diagnostic')
        _,run=self.baseline();self.assertEqual('completed',run['status']);self.assertEqual({'compiler_error':5},run['totals'])

    def test_positive_failure_remains_in_denominator(self):
        self.raw.write_text('TEST_ERROR\n')
        _,run=self.baseline();self.assertEqual('incomplete',run['status']);self.assertEqual(5,len(run['cases']))

    def test_wrong_error_does_not_pass_invalid_test(self):
        self.raw.write_text('TEST_ERROR\n')
        for case in self.spec['inputs']:case.update(expectation='invalid',invalid_reason='test',invalid_diagnostic='different required error')
        _,run=self.baseline();self.assertEqual('incomplete',run['status']);self.assertFalse(run['cases'][0]['expectation_met'])

    def test_missing_output_is_failure(self):
        self.raw.write_text('TEST_NO_OUTPUT\n')
        _,run=self.baseline();self.assertEqual('incomplete',run['status']);self.assertEqual({'missing_output':5},run['totals'])

    def test_timeout_is_not_expected_refusal(self):
        self.raw.write_text('TEST_TIMEOUT\n')
        self.spec['inputs']=self.spec['inputs'][:1];self.spec['inputs'][0].update(expectation='invalid',invalid_reason='test',invalid_diagnostic='synthetic diagnostic')
        cap,_=self.capture();cfg=self.put('toolchain.json',self.config)
        run=e.run_baseline(cap,cfg,self.root/'run',0.05)
        self.assertEqual('timeout',run['cases'][0]['status']);self.assertFalse(run['cases'][0]['expectation_met'])

    def test_missing_executable_is_incomplete(self):
        self.config['artifacts']['compiler']=str(self.root/'absent')
        _,run=self.baseline();self.assertEqual('incomplete',run['status']);self.assertIn('error',run);self.assertEqual(5,len(run['cases']));self.assertEqual({'not_run':5},run['totals'])

    def test_changed_output_rejected(self):
        _,run=self.baseline();(self.root/'run/case0/sync.pto').write_text('changed')
        with self.assertRaises(e.EvidenceError):e.verify_run(self.root/'run')

    def test_changed_log_rejected(self):
        self.baseline();(self.root/'run/case0/sync.stdout').write_text('changed')
        with self.assertRaises(e.EvidenceError):e.verify_run(self.root/'run')

    def test_placeholder_version_blocks_baseline(self):
        self.config['versions']['cann']='RECORD VERSION'
        _,run=self.baseline();self.assertEqual('incomplete',run['status']);self.assertIn('placeholders',run['error'])

    def test_toolchain_change_mid_campaign_is_detected(self):
        original=e.launch
        def change(*args, **kwargs):
            result=original(*args,**kwargs)
            self.compiler.write_text(self.compiler.read_text()+'# changed during campaign\n')
            return result
        with patch.object(e,'launch',change):
            _,run=self.baseline()
        self.assertEqual('incomplete',run['status']);self.assertIn('Toolchain changed',run['error'])

    def test_experimental_baseline_revision_refused(self):
        self.config['source_revision']='c73c04fb3b7a76d10ab55b48d4dcbe0270e36da2'
        _,run=self.baseline();self.assertIn('pinned production',run['error'])

    def test_no_second_sync_during_lowering(self):
        self.config['argv']['lower'].insert(1,'--enable-insert-sync')
        _,run=self.baseline();self.assertIn('without inserting',run['error'])

    def test_no_case_planner_override(self):
        self.spec['inputs'][0]['compile_args'].append('--insert-sync-planner=composition')
        _,run=self.baseline();self.assertEqual('incomplete',run['status'])

    def test_unknown_is_not_noop(self):
        self.spec['inputs'][0]['authored']='unclassified'
        _,run=self.baseline();self.assertEqual('unclassified_input',run['cases'][0]['status'])


class CompareTests(Fixture):
    def test_complete_explicit_receipts(self):
        cap,run=self.baseline();candidate=self.receipt(cap,run)
        out=e.compare(cap,self.root/'run',candidate);self.assertEqual('passed',out['status']);self.assertEqual(5,out['numerator'])

    def test_fallback_is_not_oahs_coverage(self):
        cap,run=self.baseline();candidate=self.receipt(cap,run,'fallback')
        out=e.compare(cap,self.root/'run',candidate);self.assertEqual(0,out['numerator']);self.assertEqual(5,out['denominator']);self.assertEqual('incomplete',out['status'])

    def test_missing_receipt_does_not_shrink_denominator(self):
        cap,run=self.baseline();p=self.receipt(cap,run);c=e.load(p);c['cases'].pop();p.write_bytes(e.canonical(c))
        out=e.compare(cap,self.root/'run',p);self.assertEqual(5,out['denominator']);self.assertEqual(4,out['numerator'])

    def test_zero_baseline_success_is_not_100_percent(self):
        self.raw.write_text('TEST_ERROR\n')
        cap,run=self.baseline()
        p=self.put('candidate.json',dict(schema=e.SCHEMA,kind='candidate_receipts',planner='oahs',capture_id=run['capture_id'],source_revision='1'*40,toolchain_id='2'*64,cases=[]))
        out=e.compare(cap,self.root/'run',p);self.assertEqual(0,out['denominator']);self.assertEqual(5,out['baseline_unresolved']);self.assertEqual('incomplete',out['status'])

    def test_changed_contract_refused(self):
        cap,run=self.baseline();p=self.receipt(cap,run);c=e.load(p);c['cases'][0]['contracts_sha256']='0'*64;p.write_bytes(e.canonical(c))
        with self.assertRaises(e.EvidenceError):e.compare(cap,self.root/'run',p)

    def test_duplicate_receipt_refused(self):
        cap,run=self.baseline();p=self.receipt(cap,run);c=e.load(p);c['cases'].append(c['cases'][0]);p.write_bytes(e.canonical(c))
        with self.assertRaises(e.EvidenceError):e.compare(cap,self.root/'run',p)

    def test_authored_preserved_requires_authored_input(self):
        cap,run=self.baseline();p=self.receipt(cap,run,'authored_preserved')
        out=e.compare(cap,self.root/'run',p);self.assertEqual(0,out['numerator'])

    def test_verified_noop_is_separate(self):
        cap,run=self.baseline();p=self.receipt(cap,run,'noop_verified')
        out=e.compare(cap,self.root/'run',p);self.assertEqual({'noop_verified':5},out['handling']);self.assertEqual(5,out['numerator'])
        self.assertEqual(5, out['automatic_synthesis'])
        self.assertEqual(0, out['authored_protocol_handling'])

    def test_authored_handling_is_not_automatic_synthesis(self):
        for row in self.spec['inputs']:
            row['authored'] = 'protocol'
        cap, run = self.baseline()
        out = e.compare(cap, self.root/'run', self.receipt(cap, run, 'authored_preserved'))
        self.assertEqual(5, out['numerator'])
        self.assertEqual(0, out['automatic_synthesis'])
        self.assertEqual(5, out['authored_protocol_handling'])

    def test_failed_proof_not_coverage(self):
        cap,run=self.baseline();p=self.receipt(cap,run);c=e.load(p);c['cases'][0]['verification_status']='failed';p.write_bytes(e.canonical(c))
        out=e.compare(cap,self.root/'run',p);self.assertEqual(4,out['numerator'])


class DeviceTests(Fixture):
    def data(self):
        cap,run=self.baseline()
        log=self.root/'timing.log';log.write_text('synthetic test timings; NOT hardware measurements\n')
        records=dict(schema=e.SCHEMA,kind='device_measurements',run_id=run['run_id'],device='synthetic',measurement_protocol={'timer':'synthetic','warmup':2},
            raw_log=dict(path=str(log),sha256=e.sha(log.read_bytes())),expected_cases=['case3','case4'],
            kernels=[dict(id=r['id'],synchronized_sha256=r['synchronized_sha256'],microseconds=[10,11,12],correctness='passed',invocations=3) for r in run['cases'][3:]],
            model=dict(correctness='passed',microseconds=[100,102,101]))
        binary=self.root/'device-test.bin';binary.write_bytes(b'SYNTHETIC TEST BINARY, NOT DEVICE CODE')
        for sample in records['kernels']:
            sample['binary']=dict(path=str(binary),sha256=e.sha(binary.read_bytes()))
            build=self.put(sample['id']+'.build.json',dict(synchronized_sha256=sample['synchronized_sha256'],
                binary_sha256=sample['binary']['sha256'],argv=['synthetic-build']))
            sample['build_record']=dict(path=str(build),sha256=e.sha(build.read_bytes()))
        return cap,run,records

    def test_recorded_not_executed_here(self):
        cap,run,m=self.data();p=self.put('metrics.json',m)
        out=e.device_summary(cap,self.root/'run',p);self.assertEqual('recorded',out['status']);self.assertIn('externally',out['provenance'])

    def test_missing_kernel_is_incomplete(self):
        cap,run,m=self.data();m['kernels'].pop();p=self.put('metrics.json',m)
        out=e.device_summary(cap,self.root/'run',p);self.assertEqual('incomplete',out['status']);self.assertEqual(['case4'],out['missing_cases'])

    def test_subset_of_model_population_is_incomplete(self):
        cap,run,m=self.data();m['kernels'].pop();m['expected_cases'].pop();p=self.put('metrics.json',m)
        out=e.device_summary(cap,self.root/'run',p);self.assertEqual(['case4'],out['missing_workload_inventory'])

    def test_timing_mismatched_output_rejected(self):
        cap,run,m=self.data();m['kernels'][0]['synchronized_sha256']='0'*64;p=self.put('metrics.json',m)
        with self.assertRaises(e.EvidenceError):e.device_summary(cap,self.root/'run',p)

    def test_nonpositive_timings_rejected(self):
        cap,run,m=self.data();m['kernels'][0]['microseconds']=[0];p=self.put('metrics.json',m)
        with self.assertRaises(e.EvidenceError):e.device_summary(cap,self.root/'run',p)

    def test_failed_correctness_not_qualified(self):
        cap,run,m=self.data();m['kernels'][0]['correctness']='failed';p=self.put('metrics.json',m)
        out=e.device_summary(cap,self.root/'run',p);self.assertEqual('incomplete',out['status'])

    def test_kernel_sum_not_model_latency(self):
        cap,run,m=self.data();m['model']={};p=self.put('metrics.json',m)
        out=e.device_summary(cap,self.root/'run',p);self.assertEqual('incomplete',out['status']);self.assertEqual({},out['model'])


class ShimTests(Fixture):
    def call(self, args):
        context=self.put('context.json',{'scope':'synthetic'})
        command=[sys.executable,'-S',str(Path(cc.__file__)), '--real',str(self.compiler),'--records',str(self.root/'records'),'--context',str(context),'--']+args
        result=subprocess.run(command,capture_output=True)
        reports=list((self.root/'records').glob('invocation-*/invocation.json'))
        return result,[e.load(p) for p in reports]

    def test_passthrough_and_complete_module(self):
        args=['--enable-insert-sync','--emit-pto-ir',str(self.raw),'-o',str(self.root/'out.pto')]
        result,rows=self.call(args);self.assertEqual(0,result.returncode);self.assertEqual(b'synthetic stdout\n',result.stdout);self.assertEqual(b'synthetic stderr\n',result.stderr)
        self.assertEqual(args,rows[0]['argv'][1:]);self.assertEqual(e.sha(self.raw.read_bytes()),rows[0]['input']['sha256']);self.assertFalse(rows[0]['fallback_by_shim'])

    def test_failed_frontend_kernel_retained(self):
        self.raw.write_text('TEST_ERROR\n')
        result,rows=self.call([str(self.raw),'-o',str(self.root/'output.cpp')]);self.assertEqual(7,result.returncode);self.assertEqual(7,rows[0]['returncode']);self.assertIsNotNone(rows[0]['input'])

    def test_repeated_input_keeps_distinct_invocations(self):
        args=[str(self.raw),'-o',str(self.root/'output.cpp')]
        self.call(args);_,rows=self.call(args);self.assertEqual(2,len(rows))

    def test_stale_output_not_reused(self):
        self.raw.write_text('TEST_NO_OUTPUT\n');(self.root/'output.cpp').write_text('stale')
        result,rows=self.call([str(self.raw),'-o',str(self.root/'output.cpp')]);self.assertEqual(0,result.returncode);self.assertFalse(rows[0]['capture_complete']);self.assertIsNone(rows[0]['output'])

    def test_collector_retains_failure_and_requires_suite_inventory(self):
        self.raw.write_text('TEST_ERROR\n')
        self.call([str(self.raw),'-o',str(self.root/'out.cpp')])
        out=self.root/'collected.json'
        spec=cr.collect([('pypto_ops',self.root/'records')],out)
        self.assertEqual(1,len(spec['inputs']));self.assertEqual(7,spec['inputs'][0]['frontend_returncode'])
        self.assertFalse(spec['cohorts']['pypto_ops']['collected'])
        cap=e.capture(out,self.root/'capture');self.assertFalse(cap['inventory_complete'])

    def test_interrupted_compiler_capture_is_not_dropped(self):
        folder=self.root/'records/invocation-interrupted';folder.mkdir(parents=True)
        spec=cr.collect([('model_kernels',self.root/'records')],self.root/'collected.json')
        self.assertEqual(1,len(spec['inputs']));self.assertIsNone(spec['inputs'][0]['raw'])

    def test_interrupted_capture_with_context_is_retained(self):
        folder = self.root/'records/invocation-interrupted'
        folder.mkdir(parents=True)
        (folder/'context.json').write_bytes(e.canonical({'expectation': 'supported'}))
        spec = cr.collect([('model_kernels', self.root/'records')], self.root/'collected.json')
        self.assertEqual(1, len(spec['inputs']))
        self.assertFalse(spec['inputs'][0]['capture_complete'])
        self.assertIsNone(spec['inputs'][0]['raw'])

    def test_path_parser_separates_output(self):
        source,out=cc.infer_paths(['input.pto','-o','output.pto'],self.root)
        self.assertEqual(self.root/'input.pto',source);self.assertEqual(self.root/'output.pto',out)


if __name__ == '__main__':unittest.main(verbosity=2)
