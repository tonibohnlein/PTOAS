#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native conservative-composition gate and hash-frozen compatibility campaign.

No device or whole-frontend performance claims. Runs serially, records exact
binary/input hashes and commands. Corpus admission is reported, not assumed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from s7_corpus import prepare_bytes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--opt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--python-root', type=Path,
                        help='also lower the pinned historical GEMM through the complete compiler to PTO and C++')
    parser.add_argument('--corpus-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    env = dict(os.environ, PTOAS_LOGICAL_TRACE='1')
    env.pop('PTOAS_STRUCTURED_PLAN_JSON', None)

    def run(name, command, source, category, expected=None):
        before = source.read_bytes()
        start = time.perf_counter()
        try:
            result = subprocess.run(list(map(str, command)), capture_output=True,
                                    text=True, env=env, timeout=60)
            code, stdout, stderr = result.returncode, result.stdout, result.stderr
            status = 'pass' if code == 0 else 'refused' if code > 0 else 'crashed'
        except subprocess.TimeoutExpired as error:
            def decoded(data):
                return data.decode(errors='replace') if isinstance(data, bytes) else data or ''
            code, stdout, stderr = None, decoded(error.stdout), decoded(error.stderr)
            status = 'timeout'
        elapsed = time.perf_counter() - start
        if source.read_bytes() != before:
            raise RuntimeError('input changed during construction: ' + str(source))
        stem = f'{len(rows):04d}'
        (args.output / (stem + '.stdout')).write_text(stdout)
        (args.output / (stem + '.stderr')).write_text(stderr)
        row = dict(id=name, category=category, source=str(source),
                   source_sha256=digest(before), command=list(map(str, command)),
                   returncode=code, status=status, seconds=elapsed, log=stem)
        if code != 0:
            reason = re.search(r'structured construction: [^;]+; (.*?); work=', stderr)
            row['first_refusal'] = reason.group(1) if reason else next(
                (line.split('error: ', 1)[-1] for line in stderr.splitlines()
                 if 'error: ' in line), status)
        if category == 'mutation':
            row['verdict'] = json.loads(stdout) if code == 0 else None
        else:
            row['sets'] = stdout.count('pto.set_flag[')
            row['waits'] = stdout.count('pto.wait_flag[')
            row['barriers'] = stdout.count('pto.barrier')
        rows.append(row)
        if expected is not None and ((code == 0) != expected or status in ('timeout', 'crashed')):
            raise RuntimeError(f'{name}: unexpected result ({status})\n{stderr}\n{stdout}')
        return row

    def compile_case(name, source, gm='may-alias', category='native', expected=True):
        return run(name, [args.opt, '--mlir-disable-threading',
                   '--pto-insert-sync=planner=structured structured-precision=false '
                   'logical-work-budget=0 gm-alias=' + gm, source],
                   source, category, expected)

    fixtures = HERE / 'structured_inputs'
    positive = ('composition_while_forwarding', 'nested_mixed_sequence',
                'nested_varying_choice', 'nested_varying_bound', 'nested_three_levels',
                'unknown_guard', 'ordinal_dynamic_step', 'ordinal_negative_lower',
                'sequential_cross_pipe', 'sequential_same_pipe', 'section_vector')
    for name in positive:
        source = fixtures / (name + '.pto')
        compile_case(name, source, gm='assume-disjoint-arguments')
    for name in ('section_outside_config', 'section_outside_async_descriptor',
                 'section_outside_physical', 'missing_positive_contract'):
        compile_case(name, fixtures / (name + '.pto'), expected=False)
    compile_case('same-address-gm-visibility', fixtures / 'nested_mixed_sequence.pto',
                 gm='may-alias', expected=False)
    source = fixtures / 'composition_while_forwarding.pto'
    for mutation in ('none', 'drop-wait', 'drop-set', 'duplicate-set', 'wrong-key',
                     'drop-packet', 'late-packet', 'wrong-reply-key', 'drop-retirement',
                     'drop-named-barrier'):
        row = run(mutation, [args.driver, source, 'composition:' + mutation,
                  args.output / ('mutated-' + mutation + '.pto')], source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('non-atomic/incorrect reconstruction verdict')
    if args.historical:
        compile_case('pinned-historical-gemm', args.historical)
        if args.python_root:
            from observations import SERIAL_DRIVER
            prefix = [sys.executable, '-c', SERIAL_DRIVER, args.python_root.resolve(),
                      '--pto-arch=a3', '--pto-level=level3']
            synchronized = args.output.resolve() / 'historical-gemm.composed.pto'
            generated = args.output.resolve() / 'historical-gemm.composed.cpp'
            commands = (
                ('historical-gemm-pto', args.historical, synchronized,
                 ['--enable-insert-sync', '--insert-sync-planner=structured',
                  '--insert-sync-structured-precision=false', '--emit-pto-ir']),
                ('historical-gemm-cpp', synchronized, generated, []))
            for name, source, output, flags in commands:
                row = run(name, prefix + flags + [source, '-o', output],
                          source, 'frontend', True)
                row.update(artifact=str(output), artifact_sha256=digest(output.read_bytes()))

    corpus_manifests = []
    for path in args.corpus_manifest:
        manifest = json.loads(path.read_text())
        root = ROOT if manifest['root'] == '$REPO' else Path(manifest['root'])
        corpus_manifests.append(dict(path=str(path), sha256=digest(path.read_bytes())))
        for case in manifest['cases']:
            source = root / case['path']
            if digest(source.read_bytes()) != case['sha256']:
                raise RuntimeError('frozen source changed: ' + str(source))
            # Same documented a2a3->a3 binding as the existing compatibility
            # campaign; retain both raw and prepared hashes and the exact rule.
            raw = source.read_bytes()
            prepared, adaptations = prepare_bytes(raw, case['stage'])
            snapshot = args.output / ('input-' + digest(prepared) + '.pto')
            snapshot.write_bytes(prepared)
            row = compile_case(case['id'], snapshot, case['gm_contract'], 'corpus', None)
            row.update(manifest=str(path), input_sha256=case['sha256'],
                       prepared_sha256=digest(prepared), stage=case['stage'],
                       classification=case['classification'],
                       preparation=adaptations)
    binaries = [args.driver, args.opt]
    if args.python_root:
        binaries.extend(p for p in args.python_root.rglob('libPTOASCompiler*.so*') if p.is_file())
    summary = dict(schema='oahs.composition.validation.v1',
                   scope='native construction/reconstruction and raw/prepared compatibility; not device or whole-compilation performance',
                   revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                   binaries={str(p): digest(p.read_bytes()) for p in binaries},
                   manifests=corpus_manifests, results=rows,
                   totals=dict(Counter(r['category'] + ':' + r['status'] for r in rows)))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['totals'], indent=2))


if __name__ == '__main__':
    main()
