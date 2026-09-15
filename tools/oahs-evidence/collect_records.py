#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Turn ALL captured compiler invocations into an explicitly unfinished inventory.

This does not certify that the frontend enumerated its whole suite. Supply the
frontend's separate enumeration/failure log before marking a cohort collected.
Failed compiler calls and ambiguous/missing inputs remain entries. No deduplication
by kernel name or file hash. No architecture-based filtering.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from evidence import COHORTS, SCHEMA, PRODUCTION_REVISION, EvidenceError, canonical, load, require, sha, stable_read, write_new


def collect(groups, output):
    groups = list(groups)
    require(groups, 'At least one explicit cohort directory is needed')
    spec = dict(schema=SCHEMA, source_revisions={'PTOAS':PRODUCTION_REVISION},
                cohorts={name:dict(collected=False, collection_argv=[], enumeration_log=None,
                    note='Compiler-call captures are not complete frontend enumeration evidence') for name in COHORTS},
                regressions={name:dict(case_ids=[], evidence=None, minimum_cases=minimum,
                    note='Reported regression not yet pinned to these case identities')
                    for name,minimum in [('small_gemm',1),('attention_refusals',6),('prefill',44)]}, inputs=[])
    seen_dirs = set()
    for name, root in groups:
        require(name in COHORTS, 'Unknown cohort')
        root=Path(root).resolve()
        require(root.is_dir() and root not in seen_dirs, 'Missing or repeated capture directory')
        seen_dirs.add(root)
        invocations=sorted(root.glob('invocation-*'))
        require(invocations, 'No captured calls in '+str(root))
        for folder in invocations:
            require(folder.is_dir() and not folder.is_symlink(), 'Invalid capture entry')
            report_path=folder/'invocation.json'
            # Interrupted invocation still appears, with an absent input.
            report=load(report_path) if report_path.exists() else {}
            context_path=folder/'context.json'
            context=load(context_path) if context_path.exists() else {}
            if context_path.exists() and report_path.exists():
                require(sha(stable_read(context_path)) == report.get('context_sha256'), 'Captured context changed')
            source=folder/'input.pto'
            if report.get('input'):
                require(source.is_file() and sha(stable_read(source))==report['input']['sha256'], 'Captured input changed')
            else:
                source=None
            cid=name+'-'+sha(str(folder).encode())[:20]
            row=dict(id=cid,cohort=name, raw=str(source) if source else None,
                     prepared=str(source) if source else None, expectation=context.get('expectation','unclassified'),
                     authored=context.get('authored','unclassified'),
                     preparation={'kind':'identity'}, contracts=context.get('contracts',{
                         'architecture':'not-recorded','hardware':'not-recorded','alias':'not-recorded',
                         'abi':'not-recorded','memory_planner':'not-recorded'}),
                     compile_args=context.get('compile_args',[]),
                     frontend_invocation=str(report_path), frontend_returncode=report.get('returncode'),
                     input_stage=report.get('input_stage','not captured'),capture_complete=report.get('capture_complete',False))
            for field in ('invalid_reason','invalid_diagnostic','invalid_stage'):
                if field in context:row[field]=context[field]
            spec['inputs'].append(row)
            for repo, revision in context.get('source_revisions',{}).items():
                require(repo not in spec['source_revisions'] or spec['source_revisions'][repo]==revision, 'Mixed source revisions require separate cohorts/campaigns')
                spec['source_revisions'][repo]=revision
    write_new(output,canonical(spec))
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',action='append',required=True,help='COHORT=/absolute/capture/directory')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    groups=[]
    for value in args.group:
        require('=' in value,'Expected COHORT=directory')
        groups.append(value.split('=',1))
    report=collect(groups,args.out)
    print(f'Wrote {len(report["inputs"])} expected input records. Inventory remains UNQUALIFIED until enumeration, contracts, and regression bindings are complete.')
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError) as error:
        print('Collection error:',error);raise SystemExit(1)
