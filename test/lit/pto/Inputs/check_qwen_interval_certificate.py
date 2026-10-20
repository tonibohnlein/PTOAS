# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independently validate the existing decoder's exact Qwen child assignment.

Proof facts are unconditional circuit facts. Runtime parameter specialization
happens only after one symbolic recipe/certificate has been constructed.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil

from assess_whole_kernel_access import annotate, digest, invoke
from check_closed_child_logic import analyze_trace
from check_finite_allocation import physical_check
from survey_pypto_regions import specialize


def checked(args, source, output):
    run = invoke([args.trace_tool, '--compact-region-contract', source], output, 35)
    if run['status'] != 'success':
        raise AssertionError(run)
    return run


def assign(trace, records):
    physical = copy.deepcopy(trace)
    by_label = {record['record']: record for record in records}
    for event in physical['events']:
        if event['kind'] in ('payload', 'barrier'):
            continue
        record = by_label[event['members'][0]]
        if (record['source'], record['target']) != (event['source_pipe'], event['target_pipe']):
            raise AssertionError('Wrong record direction')
        coordinates = [event['source_ordinal']] + event['members'][1:]
        if 'tuple' in record:
            rule = record['tuple']
            if rule['arity'] + 1 != len(coordinates):
                raise AssertionError('Expected the certified tuple plus one enclosing invocation coordinate')
            # These Qwen rules are coordinate independent. Do not guess the
            # permutation between enclosing and leaf coordinates for a future
            # dynamic rule; the native binding service must establish that.
            if any(stride and modulus != 1 for _, stride, _, modulus, _ in rule['terms']):
                raise AssertionError('Dynamic original-frame tuple binding needs its native adapter')
            offset = rule['base'] + sum(scale * (phase % modulus)
                for _, _, phase, modulus, scale in rule['terms'])
        else:
            offset = (record['stride'] * event['source_ordinal'] + record['phase']) % len(record['ids'])
        if not 0 <= offset < len(record['ids']):
            raise AssertionError('Out-of-table physical mapping')
        event['physical_id'] = record['ids'][offset]
    physical_check({'accepted': True, 'allocated': True, 'trace': trace, 'physical': physical}, set(range(6)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('trace-tool', 'input', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    args.trace_tool = Path(shutil.which(str(args.trace_tool)) or args.trace_tool).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    original = args.input.read_text()
    common = 'test.contract_math_only, test.contract_region = 5 : i64, test.contract_gm_alias = "may-not-alias"'
    source = args.output / 'probe.pto'
    source.write_text(annotate(original, common + ', test.contract_completion_probe, test.contract_completion_bound, '
        'test.contract_lane_pair = array<i64: 12, 24>'))
    probe = checked(args, source, args.output / 'probe.json')
    data = json.loads(Path(probe['stdout']).read_text())
    if not data.get('prepared') or data.get('outside_endpoints'):
        raise AssertionError('Original-frame preparation failed')
    rows = data['lane_pair_probe']
    if not rows or any(not row['query_available'] or not row['interval_implies'] for row in rows):
        raise AssertionError('Lane 12->24 interval implication not established')
    source = args.output / 'symbolic-input.pto'
    source.write_text(annotate(original, common + ', test.contract_completion_emit'))
    emitted = checked(args, source, args.output / 'symbolic-child.pto')
    text = Path(emitted['stdout']).read_text()
    receipt = {'tool_sha256': digest(args.trace_tool), 'input_sha256': digest(args.input),
        'probe': probe, 'emission': emitted, 'caller_bounds_installed': False,
        'certificate_decoded': data['physical_certificate_decoded'], 'cases': []}
    ks = range(18) if args.full else (0, 1, 2, 3, 4, 6, 17)
    specs = [(block, k) for block in (0, 19) for k in ks]
    specs.append((20, 17))
    if args.full:
        specs += [(block, 17) for block in range(1, 19)] + [(0, 32), (0, 64)]
    for block, k in specs:
        bindings = {'%arg3': 20, '%arg4': k, '%arg5': 0, '%arg6': 0,
                    '%arg7': 17408, '%arg8': block, '%arg9': 20}
        path = args.output / f'block{block}-K{k}.pto'
        path.write_text(annotate(specialize(text, bindings), 'test.trace_existing'))
        run = invoke([args.trace_tool, '--structured-trace', path], path.with_suffix('.json'), 35)
        if run['status'] != 'success':
            raise AssertionError(run)
        trace = json.loads(Path(run['stdout']).read_text())['trace']
        result = analyze_trace(trace, specialize(original, bindings), int(block < 20), k)
        result['scope'] = 'caller' if k == 17 else 'short-K-derivative' if k < 17 else 'outside-caller-scaling'
        if block >= 20:
            result['scope'] = 'no-work-block-outside-caller'
        if data['physical_certificate_decoded']:
            assign(trace, data['physical_assignment'])
            result['decoded_assignment_causal_reuse'] = 'passed'
        else:
            result['decoded_assignment_causal_reuse'] = 'unavailable'
        result.update(block=block, run=run)
        receipt['cases'].append(result)
        (args.output / 'results.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('interval certificate: lane 12->24 passed; decoded map ' +
          ('independently validated' if data['physical_certificate_decoded'] else 'unavailable'))


if __name__ == '__main__':
    main()
