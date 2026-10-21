# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Refresh production stages without running audit recognizers or extra exports."""

import argparse
import json
import os
from pathlib import Path
import resource

from assess_whole_kernel_access import PREPARED, SURVEY, access_stage, compile_stage, digest, invoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('optimizer', 'trace-tool', 'cli', 'python', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=35)
    parser.add_argument('--kernels', nargs='+', choices=PREPARED + SURVEY)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('Positive watchdog required')
    for name in ('optimizer', 'trace_tool', 'cli', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    # Keep the venv executable spelling: resolving its symlink would select
    # the base interpreter without the environment's installed packages.
    args.python = args.python.absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    result = {'scope': 'Whole production requests; no audit or missing-export construction',
              'watchdog_seconds': args.timeout, 'workers': 1,
              'tools': {str(p): digest(p) for p in (args.optimizer, args.trace_tool, args.cli)},
              'cases': []}
    root = Path.cwd()
    for kernel in PREPARED + SURVEY:
        if args.kernels and kernel not in args.kernels:
            continue
        inputs = root / ('test/lit/pto/Inputs/pypto_region_survey' if kernel in SURVEY else
                         'test/npu_validation/benchmarks/auto_sync/prepared')
        source = inputs / kernel / 'input.pto'
        for alias in ('may-not-alias', 'may-alias'):
            folder = args.output / kernel / alias
            folder.mkdir(parents=True, exist_ok=True)
            case = {'kernel': kernel, 'alias': alias, 'input': str(source),
                    'input_sha256': digest(source), 'access': access_stage(args, folder, source, alias),
                    'configurations': []}
            for mode in ('production', 'completion'):
                target = folder / mode
                target.mkdir(exist_ok=True)
                flag = f'gm-alias={alias} profile-stages=true'
                if mode == 'completion':
                    flag += ' completion-only=true'
                logical = compile_stage(args, target, 'logical', source,
                    '--pto-frontier-analysis=' + flag + ' prepare-allocation=false')
                row = {'mode': mode, 'logical': logical}
                if logical['status'] == 'success':
                    support = compile_stage(args, target, 'allocation-support', source,
                        '--pto-frontier-analysis=' + flag)
                    row['allocation-support'] = support
                    if support['status'] == 'success':
                        physical = compile_stage(args, target, 'physical', Path(support['output']),
                            '--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5')
                        row['physical'] = physical
                        if physical['status'] == 'success':
                            output = target / 'kernel.cpp'
                            row['cpp'] = invoke([args.python, args.cli, '--pto-arch=a3',
                                '--pto-level=level3', '--enable-insert-sync=false',
                                '--enable-plan-memory=false', physical['output'], '-o', output],
                                target / 'cpp.stdout', args.timeout)
                            if row['cpp']['status'] == 'success':
                                row['cpp'].update(output=str(output), sha256=digest(output))
                case['configurations'].append(row)
                print(kernel, alias, mode, {k: v.get('status') for k, v in row.items()
                                           if isinstance(v, dict)}, flush=True)
            result['cases'].append(case)
            (args.output / 'matrix.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
