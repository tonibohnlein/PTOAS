#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Replay frozen inputs with source/contract identities and publication witnesses.

Historical evidence is always labelled; a --baseline report supplies fresh
before/after admissions. No device qualification or timing gate is inferred.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from s7_corpus import load_frozen_manifest, load_frozen_lock, preflight_frozen, composition_outcome

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witness_classes(reports):
    output = []
    for report in reports:
        accesses = {a['id']: a for a in report['accesses']}
        for pair in report['publication_pairs']:
            writer, reader = accesses[pair['writer']], accesses[pair['reader']]
            known = writer['origins_complete'] and reader['origins_complete']
            common = set(writer['roots']) & set(reader['roots'])
            if not pair['native_overlap']:
                classification = ('established-disjointness-contract' if known and not common else
                                  'lost-provenance-or-range-precision')
            elif known:
                classification = 'genuine-possible-overlap'
            else:
                classification = 'unresolved-evidence'
            output.append(dict(function=report['function'], **pair, classification=classification,
                               writer_access=writer, reader_access=reader))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', type=Path, required=True)
    parser.add_argument('--cas-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publication-reference', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    manifest = HERE / 'corpus/frozen-363.jsonl'
    records, manifest_hash = load_frozen_manifest(manifest)
    lock = load_frozen_lock(HERE / 'corpus/frozen-363.lock.json', manifest_hash)
    prepared, _ = preflight_frozen(records, {}, args.cas_root)
    reference = json.loads(args.publication_reference.read_text())
    refusals = {r['id']: r for r in reference['results'] if r['category'] == 'corpus' and
                'MTE3-to-MTE2 GM publication' in (r.get('first_refusal') or '')}
    if len(refusals) != 86:
        raise ValueError('publication reference must contain the reviewed 86 first refusals')
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    before = {r['id']: r for r in baseline['rows']} if baseline else {}
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PTOAS_LOGICAL_TRACE='1', PTOAS_COMPOSITION_WITNESSES='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    binary_hash = digest(args.opt)
    cache, rows = {}, []
    for record, _, _, payload in prepared:
        key = (record['prepared']['sha256'], record['gm_contract'])
        if key not in cache:
            folder = args.output / ('input-' + key[0] + '-' + key[1])
            folder.mkdir()
            source = folder / 'input.pto'
            source.write_bytes(payload)
            arms = {}
            for arm in ('composition', 'existing'):
                command = [str(args.opt.resolve()), '--mlir-disable-threading',
                           f'--pto-insert-sync=planner={arm} structured-precision=true '
                           'logical-work-budget=0 gm-alias=' + key[1], str(source.resolve())]
                start = time.monotonic()
                try:
                    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
                    outcome = composition_outcome(result.returncode, result.stdout, result.stderr)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError('replay timeout: ' + record['id']) from exc
                stdout, stderr = folder / (arm + '.pto'), folder / (arm + '.stderr')
                stdout.write_text(result.stdout)
                stderr.write_text(result.stderr)
                reports = [json.loads(line[len('OAHS_WITNESS '):]) for line in result.stderr.splitlines()
                           if line.startswith('OAHS_WITNESS ')]
                arms[arm] = dict(command=command, outcome=outcome, seconds=time.monotonic()-start,
                                 output=str(stdout), output_sha256=digest(stdout), stderr=str(stderr),
                                 stderr_sha256=digest(stderr), witnesses=reports)
            cache[key] = arms
        arms = cache[key]
        old = before.get(record['id'])
        if old and (old['prepared_sha256'] != key[0] or old['gm_alias'] != key[1]):
            raise ValueError('before/after input or alias identity differs: ' + record['id'])
        actual = arms['composition']['outcome']
        prior = old['actual'] if old else record['current']
        row = dict(id=record['id'], index=record['index'], family=record['family'],
                   input_sha256=record['original']['sha256'], prepared_sha256=key[0], gm_alias=key[1],
                   hardware_contract='conservative', ownership_contract='none', measurement='fresh',
                   actual=actual, before=prior, before_kind='fresh-baseline' if old else 'historical-reference',
                   gained=actual['returncode'] == 0 and prior['returncode'] != 0,
                   lost=actual['returncode'] != 0 and prior['returncode'] == 0,
                   arms={a: {k: v for k, v in value.items() if k != 'witnesses'} for a, value in arms.items()})
        if record['id'] in refusals:
            row['publication_reference'] = refusals[record['id']]['first_refusal']
            row['publication_witnesses'] = witness_classes(arms['composition']['witnesses'])
            row['subsequent_blocker'] = actual['first_refusal']
        (args.output / (f'{record["index"]:04d}.witness.json')).write_text(
            json.dumps(arms['composition']['witnesses'], indent=2) + '\n')
        rows.append(row)
        with (args.output / 'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        if len(rows) % 40 == 0:
            print(f'{len(rows)}/{len(records)} replay rows complete', flush=True)
    if digest(args.opt) != binary_hash:
        raise RuntimeError('binary changed during replay')
    summary = dict(schema='oahs.coverage.campaign.v1', rows=rows, device='NOT_RUN',
                   revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                   source_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
                   manifest_sha256=manifest_hash, binary_sha256=binary_hash,
                   publication_reference=dict(path=str(args.publication_reference),
                       sha256=digest(args.publication_reference), measurement='historical-reference'),
                   baseline=dict(path=str(args.baseline), sha256=digest(args.baseline)) if baseline else None,
                   admissions=dict(Counter(r['family'] for r in rows if r['actual']['returncode'] == 0)),
                   gained=[r['id'] for r in rows if r['gained']], lost=[r['id'] for r in rows if r['lost']],
                   distinct_replays=len(cache), publication_refusals=len(refusals))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('admissions', 'gained', 'lost', 'distinct_replays')}, indent=2))
    return int(bool(summary['lost']) or any(r['actual']['status'] in ('crashed', 'timeout') for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
