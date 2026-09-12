#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native cut reconstruction and unchanged-payload completion-boundary checks.

The normalization compiler run is not a whole-compilation timing measurement.
An optional frozen benchmark summary selects the precise prior comparison arm.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from observations import SERIAL_DRIVER, population


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--python-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--require-quality', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.python_root.resolve()))
    from compare_boundaries import compare, run

    commands, cases = [], []
    env = dict(os.environ, PTOAS_LOGICAL_TRACE='1')
    env.pop('PTOAS_STRUCTURED_PLAN_JSON', None)

    def invoke(name, command):
        result = subprocess.run(list(map(str, command)), capture_output=True,
                                text=True, env=env, timeout=60)
        prefix = args.output / name
        prefix.with_suffix('.stdout').write_text(result.stdout)
        prefix.with_suffix('.stderr').write_text(result.stderr)
        commands.append(dict(name=name, command=list(map(str, command)),
                             returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(name + ': ' + result.stderr + result.stdout)
        return result.stdout

    compiler = [sys.executable, '-c', SERIAL_DRIVER, args.python_root.resolve(),
                '--pto-arch=a3', '--pto-level=level3', '--emit-pto-ir']
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    for case in population():
        name = case['case_id']
        raw = args.output.resolve() / (name + '.native.pto')
        emitted = args.output.resolve() / (name + '.pto')
        verdict = json.loads(invoke(name + '-native', [args.driver, case['source'], 'cuts:none', raw]))
        if not verdict['accepted'] or not verdict['atomic']:
            raise RuntimeError('native construction/reconstruction failed: ' + name)
        invoke(name + '-normalize', compiler + [raw, '-o', emitted])
        if baseline:
            row = next(r for r in baseline['rows'] if r['case'] == name)
            if digest(case['source']) != row['source_sha256']:
                raise RuntimeError('frozen input changed: ' + name)
            trial = next(t for t in row['trials'] if t['arm'] == 'structured' and t['round'] == 0)
            reference = Path(trial['command'][-1])
            if digest(reference) != trial['output_sha256']:
                raise RuntimeError('frozen reference changed: ' + name)
        else:
            reference = args.output.resolve() / (name + '.existing.pto')
            invoke(name + '-reference', compiler + ['--enable-insert-sync',
                   '--insert-sync-planner=existing',
                   '--insert-sync-gm-alias=assume-disjoint-arguments',
                   case['source'], '-o', reference])
        scenarios = []
        for scenario in case['scenarios']:
            old, _ = run(reference, scenario)
            new, _ = run(emitted, scenario)
            differences = compare(old, new)  # also requires identical payload
            if old.tokens or new.tokens:
                raise RuntimeError('unconsumed final event: ' + name)
            scenarios.append(dict(name=scenario['name'],
                                  later=sum(d['automatic_requires_later_prefix'] for d in differences),
                                  differences=differences[:5]))
        cases.append(dict(case=name, input_sha256=digest(case['source']), verdict=verdict,
                          output=str(emitted), output_sha256=digest(emitted),
                          reference=str(reference), reference_sha256=digest(reference),
                          scenarios=scenarios))
        print(json.dumps(dict(case=name, later=sum(s['later'] for s in scenarios))), flush=True)
    source = next(c['source'] for c in population() if c['case_id'] == 'two_buffer')
    mutations = []
    for mutation in ('drop-set', 'drop-wait', 'duplicate-set', 'wrong-key', 'drop-retirement',
                     'early-publication', 'late-acquisition', 'drop-exit-ack'):
        verdict = json.loads(invoke(mutation, [args.driver, source, 'cuts:' + mutation,
                             args.output.resolve() / ('mutated-' + mutation + '.pto')]))
        if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
            raise RuntimeError('mutation escaped reconstruction: ' + mutation)
        mutations.append(dict(mutation=mutation, verdict=verdict))
    qualified = not any(s['later'] for c in cases for s in c['scenarios'])
    summary = dict(schema='oahs.cuts.validation.v1', cases=cases, mutations=mutations,
                   commands=commands, driver_sha256=digest(args.driver),
                   runner_sha256=digest(__file__),
                   baseline_sha256=digest(args.baseline) if args.baseline else None,
                   quality_qualified=qualified, device='NOT_RUN',
                   scope='native reconstruction and finite completion boundaries; not compilation timing')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.require_quality and not qualified:
        raise SystemExit('completion-boundary quality gate remains open')


if __name__ == '__main__':
    main()
