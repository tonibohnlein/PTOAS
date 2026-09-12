#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare frozen compatibility results and separately observe overlap changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, action='append', required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--python-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    candidate = json.loads(args.corpus.read_text())
    groups = []
    for baseline in args.baseline:
        old = json.loads(baseline.read_text())
        old_rows = {r['id']: r for r in old['cases']}
        new_rows = {r['id']: r for r in candidate['results']
                    if r['category'] == 'corpus' and r['id'] in old_rows}
        if set(new_rows) != set(old_rows):
            raise ValueError('baseline and candidate populations differ')
        for name, row in new_rows.items():
            if row['prepared_sha256'] != old_rows[name]['input_sha256']:
                raise ValueError('pre-sync bytes differ: ' + name)
            if row['input_sha256'] != old_rows[name]['sha256']:
                raise ValueError('original input bytes differ: ' + name)
        before = {name for name, row in old_rows.items()
                  if row['arms']['structured']['status'] == 'applied'}
        after = {name for name, row in new_rows.items() if row['returncode'] == 0}
        groups.append(dict(baseline=str(baseline),
                           baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                           population=len(old_rows), before=len(before), after=len(after),
                           gained=sorted(after-before), lost=sorted(before-after)))

    sys.path.insert(0, str(args.python_root.resolve()))
    from observations import population
    from compare_boundaries import run, compare
    cases = {case['case_id']: case for case in population()}
    benchmark = json.loads(args.benchmark.read_text())
    quality = []
    for row in benchmark['rows']:
        paths = {}
        for arm in ('structured', 'composition'):
            trial = next(t for t in row['trials'] if t['arm'] == arm
                         and t['round'] == 0 and t['status'] == 'applied')
            path = Path(trial['command'][-1])
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial['output_sha256']:
                raise ValueError('benchmark output changed: ' + str(path))
            paths[arm] = path
        scenarios = []
        for scenario in cases[row['case']]['scenarios']:
            old, _ = run(paths['structured'], scenario)
            new, _ = run(paths['composition'], scenario)
            if old.payload != new.payload or old.tokens or new.tokens:
                raise ValueError('payload or event participation differs')
            differences = compare(old, new)
            counts = Counter('later' if d['automatic_requires_later_prefix'] else 'earlier'
                             for d in differences)
            scenarios.append(dict(name=scenario['name'], differences=dict(counts),
                                  first_differences=differences[:3]))
        quality.append(dict(case=row['case'], scenarios=scenarios,
                            compilation_ratio=row['ratios']['composition']))
    report = dict(schema='oahs.composition.comparison.v1', corpus=groups, quality=quality,
                  scope='matched pre-sync compatibility and observed completion boundaries; not device timing',
                  overlap_qualified=False,
                  remaining='bounded precision/early publication and shared allocation integration')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(corpus=[{k: g[k] for k in ('population', 'before', 'after')} for g in groups],
                          quality=[dict(case=q['case'],
                                        later=sum(s['differences'].get('later', 0) for s in q['scenarios']))
                                   for q in quality]), indent=2))


if __name__ == '__main__':
    main()
