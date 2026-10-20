# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independently check the actual symbolic Qwen completion import and mapping.

Both comparison plans are compiled before specializing runtime inputs. The
reference uses original access operands, not compiler cells or demand records.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import shutil
from types import SimpleNamespace

from assess_whole_kernel_access import annotate, compile_stage, digest, invoke
from check_closed_child_logic import child_phases, identity
from check_finite_allocation import physical_check
from compact_vector_oracle import command_order
from pypto_region_oracle import interpret, mathematical
from survey_pypto_regions import specialize


def traced(args, text, path):
    path.write_text(annotate(text, 'test.trace_existing'))
    run = invoke([args.trace_tool, '--structured-trace', path], path.with_suffix('.json'), 35)
    if run['status'] != 'success':
        raise AssertionError(run)
    result = json.loads(Path(run['stdout']).read_text())
    if not result.get('accepted') or result['trace'].get('error'):
        raise AssertionError(result)
    return result['trace'], run


def check(args, folder, logical, physical, original, bindings, decoded=None):
    source = specialize(original, bindings)
    items = interpret(source)
    required, _ = mathematical(items)
    word, logical_run = traced(args, specialize(logical, bindings), folder / 'logical.pto')
    machine, physical_run = traced(args, specialize(physical, bindings), folder / 'physical.pto')
    payloads = [e for e in word['events'] if e['kind'] == 'payload']
    if [(e['type'], e['coordinates'], e['pipe']) for e in payloads] != [
        (e['phase'], e['coordinates'], e['pipe']) for e in items
    ]:
        raise AssertionError('Original payload sequence changed')
    reach, _ = command_order(word, include_commands=True)
    if any(row >> i & 1 for i, row in enumerate(reach)):
        raise AssertionError('Command graph cycle')
    wanted = {(a, b) for a in range(len(items)) for b in range(len(items))
              if required[2*a+1] >> (2*b) & 1}
    realized = {(a, b) for a in range(len(items)) for b in range(len(items))
                if reach[2*a+1] >> (2*b) & 1}
    if wanted - realized:
        raise AssertionError(('Missing modeled prerequisites', sorted(wanted-realized)[:5]))
    # This checks the allocated output, including ALL boundaries, tail children
    # and palette transitions. No alternative coloring is supplied by the test.
    physical_check({'accepted': True, 'allocated': True, 'trace': word, 'physical': machine}, set(range(6)))
    checked_assignments = 0
    if decoded is not None:
        for old, new in zip(word['events'], machine['events']):
            if old['kind'] in ('payload', 'barrier'):
                continue
            record = decoded.get(old['members'][0])
            if record is None:
                continue  # Other retained children have their own certificates.
            if (record['source'], record['target']) != (old['source_pipe'], old['target_pipe']):
                raise AssertionError('Child record namespace or direction changed')
            rule = record.get('tuple')
            if rule:
                if any(stride and modulus != 1 for _, stride, _, modulus, _ in rule['terms']):
                    raise AssertionError('Unexpected dynamic Qwen map requires original-frame evaluation')
                offset = rule['base'] + sum(scale*(phase % modulus)
                    for _, _, phase, modulus, scale in rule['terms'])
            else:
                offset = (record['stride']*old['source_ordinal'] + record['phase']) % len(record['ids'])
            if new['physical_id'] != record['ids'][offset]:
                raise AssertionError('Import changed the decoded child assignment')
            checked_assignments += 1
    phases = child_phases(source)
    child_sites = {}
    for index, event in enumerate(payloads):
        if event['type'] in phases:
            child_sites.setdefault(event['coordinates'][0], []).append(index)
    pending, seen, child_pairs = {}, set(), 0
    use = {}
    counts = Counter(e['kind'] for e in word['events'] if e['kind'] != 'payload')
    for event in word['events']:
        if event['kind'] in ('payload', 'barrier'):
            continue
        key = identity(event)
        owners = [owner for owner, sites in child_sites.items()
                  if sites[0] <= event['gap'] <= sites[-1]+1]
        owner = owners[0] if len(owners) == 1 else None
        if event['kind'] == 'set':
            if key in seen:
                raise AssertionError('Publication identity repeated across counter resets')
            seen.add(key)
            pending[key] = owner
        else:
            if key not in pending or pending.pop(key) != owner:
                raise AssertionError('Unpublished or cross-child WAIT')
            child_pairs += int(owner is not None)
    if pending:
        raise AssertionError('Unmatched publication at exit')
    for event in machine['events']:
        if event['kind'] == 'set':
            direction = str((event['source_pipe'], event['target_pipe']))
            use.setdefault(direction, set()).add(event['physical_id'])
    return {'payloads': len(items), 'commands': dict(counts), 'child_pairs': child_pairs,
            'missing_ci': 0, 'excess_ci': len(realized-wanted),
            'matching': 'passed', 'invocation_identity': 'passed', 'acyclic': True,
            'actual_mapping_causal_reuse': 'passed',
            'directional_ids': {key: sorted(value) for key, value in use.items()},
            'decoded_child_endpoints_checked': checked_assignments,
            'logical_trace': logical_run, 'physical_trace': physical_run}, realized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trace-tool', 'optimizer', 'input', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--baseline-optimizer', type=Path)
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    for name in ('trace_tool', 'optimizer', 'baseline_optimizer'):
        value = getattr(args, name)
        if value:
            setattr(args, name, Path(shutil.which(str(value)) or value).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = {'input_sha256': digest(args.input), 'watchdog_seconds': 35,
              'caller_bounds_installed': False, 'tools': {}, 'routes': [], 'comparisons': []}
    routes = [('candidate', args.optimizer)]
    if args.baseline_optimizer:
        routes.append(('baseline', args.baseline_optimizer))
    ks = range(18) if args.full else (0, 1, 2, 3, 4, 6, 17)
    specs = [(1, block, k) for block in (0, 19) for k in ks]
    specs += [(0, 0, 2), (2, 0, 2), (3, 0, 4)]
    if args.full:
        specs += [(1, block, 17) for block in range(1, 19)] + [(1, 0, 32), (1, 0, 64)]
    probe_source = args.output / 'child-probe.pto'
    probe_source.write_text(annotate(args.input.read_text(),
        'test.contract_math_only, test.contract_region = 5 : i64, '
        'test.contract_gm_alias = "may-not-alias", test.contract_completion_probe, test.contract_completion_bound'))
    probe = invoke([args.trace_tool, '--compact-region-contract', probe_source], args.output/'child-probe.json', 35)
    if probe['status'] != 'success':
        raise AssertionError(probe)
    data = json.loads(Path(probe['stdout']).read_text())
    if not data['physical_certificate_decoded']:
        raise AssertionError('Per-invocation decoder refused')
    result['child_probe'] = probe
    result['child_preparation'] = {key: data.get(key) for key in (
        'families', 'materialized_endpoints', 'closed_child_qualified', 'child_pending_payloads',
        'child_transparent', 'child_pending_notification_completion')}
    decoded = {record['record']: record for record in data['physical_assignment']}
    orders = {}
    for name, optimizer in routes:
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        result['tools'][str(optimizer)] = digest(optimizer)
        config = SimpleNamespace(optimizer=optimizer, timeout=35)
        logical = compile_stage(config, folder, 'logical', args.input,
            '--pto-frontier-analysis=gm-alias=may-not-alias prepare-allocation=false profile-stages=true')
        if logical['status'] != 'success':
            raise AssertionError(logical)
        physical = compile_stage(config, folder, 'physical', Path(logical['output']),
            '--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5')
        if physical['status'] != 'success':
            raise AssertionError(physical)
        if name == 'candidate' and logical['static_ir']['sets'] < 100:
            raise AssertionError('Compact eventful child was not imported')
        row = {'name': name, 'logical': logical, 'physical': physical, 'cases': []}
        result['routes'].append(row)
        for visits, block, k in specs:
            target = folder / f'visits{visits}-block{block}-K{k}'
            target.mkdir(exist_ok=True)
            bindings = {'%arg3': 20*visits, '%arg4': k, '%arg5': 0, '%arg6': 0,
                        '%arg7': 17408, '%arg8': block, '%arg9': 20}
            checked, order = check(args, target, Path(logical['output']).read_text(),
                Path(physical['output']).read_text(), args.input.read_text(), bindings,
                decoded if name == 'candidate' else None)
            checked.update(visits=visits, block=block, K=k,
                scope='outside-caller-reset-stress' if visits > 1 else
                      'caller' if visits == 1 and k == 17 else 'short-or-zero-derivative')
            row['cases'].append(checked)
            orders[(name, visits, block, k)] = order
            (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    if args.baseline_optimizer:
        for visits, block, k in specs:
            before = orders[('baseline', visits, block, k)]
            after = orders[('candidate', visits, block, k)]
            result['comparisons'].append({'visits': visits, 'block': block, 'K': k,
                'removed_ci': len(before-after), 'added_ci': len(after-before)})
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print('actual completion import: modeled coverage, matching, reset identity and physical reuse passed')


if __name__ == '__main__':
    main()
