#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Classify frozen authored plans with fresh native reconstruction, serially.

This records verification results, not preparation equivalence, boundary
recovery, timing qualification, numerical correctness or device performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for reference in manifest['rows']:
        if 'snapshot' not in reference or reference['name'] == 'prepared-payload':
            continue
        source = args.manifest.parent / reference['snapshot']
        if digest(source) != reference['sha256']:
            raise ValueError('changed frozen source: ' + str(source))
        for hardware in ('conservative', 'a2a3-mmad-acc-v1'):
            name = reference['name'] + '.' + hardware
            output = args.output / (name + '.pto')
            command = [str(args.driver.resolve()), str(source.resolve()),
                       'authored:none', str(output.resolve()), hardware, 'may-alias']
            start = time.monotonic()
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=60)
                code, stdout, stderr = process.returncode, process.stdout, process.stderr
                try:
                    verdict = json.loads(stdout)
                except ValueError:
                    verdict = None
                classification = ('verified' if code == 0 and verdict and verdict['accepted'] else
                                  'crash' if code < 0 else 'verification-refusal' if verdict else 'import-refusal')
            except subprocess.TimeoutExpired:
                code, stdout, stderr, verdict, classification = None, '', '', None, 'timeout'
            (args.output / (name + '.stdout')).write_text(stdout)
            (args.output / (name + '.stderr')).write_text(stderr)
            rows.append(dict(reference=reference['name'], source_sha256=digest(source), command=command,
                             hardware=hardware, gm_alias='may-alias', classification=classification,
                             returncode=code, verdict=verdict, seconds=time.monotonic() - start,
                             output_sha256=digest(output) if output.exists() else None))
    result = dict(driver_sha256=digest(args.driver), manifest_sha256=digest(args.manifest), rows=rows,
                  comparison_qualified=False, device_qualified=False)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in rows:
        print(row['reference'], row['hardware'], row['classification'],
              row['verdict']['reason'] if row['verdict'] else '')
    return int(any(row['classification'] in ('crash', 'timeout') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
