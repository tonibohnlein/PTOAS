# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare Qwen's supplied lane evidence with a retained symbolic command recipe.

The named caller domain is checked offline, not installed as a compiler premise.
No allocation certificate is manufactured from finite execution checks.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import resource
import shutil

from assess_whole_kernel_access import annotate, digest, invoke
from check_closed_child_logic import analyze_trace, identity
from check_completion_child_contract import diagnostic
from compact_vector_oracle import command_order
from survey_pypto_regions import specialize


def checked(command, path):
    result = invoke(command, path, 35)
    if result['status'] != 'success':
        raise AssertionError(result)
    return result


def lane_view(certificate):
    lanes = certificate['lane_evidence']['lanes']
    lane_ids = {(lane['group'], lane['local']): i for i, lane in enumerate(lanes)}
    records = {}
    for group_id, group in enumerate(certificate['groups']):
        for member, label in enumerate(group['records']):
            rule = group['tuple_rules'][member]
            if not isinstance(rule, dict) or 'base' not in rule:
                # The existing scalar rule also has to be constant for this diagnostic.
                if group['strides'][member]:
                    raise AssertionError('Nonconstant scalar map needs an explicit evaluator')
                local = group['phases'][member]
            else:
                if any(term[3] != 1 for term in rule['terms']):
                    raise AssertionError('Nonconstant tuple map needs an explicit evaluator')
                local = rule['base']
            if label in records or (group_id, local) not in lane_ids:
                raise AssertionError('Invalid supplied record/lane map')
            records[label] = lane_ids[(group_id, local)]
    return lanes, records


def lifetime_buckets(trace, records):
    commands = [event for event in trace['events'] if event['kind'] != 'payload']
    offset = 2 * sum(event['kind'] == 'payload' for event in trace['events'])
    pairs, owners = {}, {}
    payloads = [event for event in trace['events'] if event['kind'] == 'payload']
    for index, event in enumerate(commands):
        if event['kind'] == 'barrier':
            continue
        key = identity(event)
        pairs.setdefault(key, {})[event['kind']] = offset + index
        # Original payload coordinates, not matching-tuple positions, identify
        # this invocation. Different record shapes have different tuple arity.
        owner = payloads[min(event['gap'], len(payloads) - 1)]['coordinates'][0]
        if key in owners and owners[key] != owner:
            raise AssertionError('A logical handoff crosses child invocations')
        owners[key] = owner
    buckets = defaultdict(list)
    for key, pair in pairs.items():
        if set(pair) != {'set', 'wait'}:
            raise AssertionError('Incomplete logical handoff')
        pair['invocation'] = owners[key]
        buckets[records[key[3][0]]].append((key, pair))
    return buckets


def compare_evidence(trace, certificate):
    lanes, records = lane_view(certificate)
    buckets = lifetime_buckets(trace, records)
    reach, _ = command_order(trace, include_commands=True)
    missing = []
    conflicts = []
    inconsistent = []
    groups = certificate['groups']
    for first, left in sorted(buckets.items()):
        for second, right in sorted(buckets.items()):
            if first == second:
                continue
            a, b = groups[lanes[first]['group']], groups[lanes[second]['group']]
            if (a['source'], a['target']) != (b['source'], b['target']):
                continue
            # Lane evidence is scoped to one whole child invocation. Keep the
            # global unseparated lifetime-width test in analyze_trace separate.
            compatible = [(x, y) for _, x in left for _, y in right if x['invocation'] == y['invocation']]
            if not compatible:
                continue
            before = all(reach[x['wait']] >> y['set'] & 1 for x, y in compatible)
            supplied = second in lanes[first]['successors']
            if supplied and not before:
                # Conditional/absent bounds can differ from a global command envelope.
                inconsistent.append([first, second])
            if before and not supplied:
                missing.append({'first': first, 'second': second,
                                'last_wait': max(x['wait'] for _, x in left),
                                'first_set': min(y['set'] for _, y in right)})
            if first < second and any(not (reach[x['wait']] >> y['set'] & 1 or
                                          reach[y['wait']] >> x['set'] & 1)
                                      for x, y in compatible):
                conflicts.append([first, second])
    # An offline constant-lane coloring is only a bounded schema-expressibility
    # witness. It is never published as trusted producer evidence.
    colors = {}
    conflict_set = {tuple(pair) for pair in conflicts}
    for lane in sorted(buckets):
        used = {color for prior, color in colors.items()
                if tuple(sorted((prior, lane))) in conflict_set}
        colors[lane] = next(color for color in range(len(lanes)) if color not in used)
    return {'scope': 'within each whole child invocation; no inter-invocation reuse claim',
            'active_lanes': sorted(buckets), 'inactive_lanes': sorted(set(range(len(lanes))) - set(buckets)),
            'missing_supplied_edges': missing, 'supplied_edges_failing_bounded_envelope': inconsistent,
            'actual_lane_conflicts': conflicts, 'bounded_constant_lane_map': colors,
            'bounded_constant_lane_map_budget': max(colors.values(), default=-1) + 1,
            'first_refused_lane_active': 18 in buckets}


def sufficient_probe(args, original):
    attributes = ('test.contract_math_only, test.contract_region = 5 : i64, '
                  'test.contract_gm_alias = "may-not-alias", test.contract_completion_probe, '
                  'test.contract_completion_bound, test.contract_lane_pair = array<i64: 12, 24>')
    source = args.output / 'sufficient-probe.pto'
    source.write_text(annotate(original, attributes))
    run = checked([args.trace_tool, '--compact-region-contract', source], args.output / 'sufficient-probe.json')
    rows = json.loads(Path(run['stdout']).read_text())['lane_pair_probe']
    if not rows or any(not row['query_available'] for row in rows):
        raise AssertionError('This obligation needs a query, not an assumed empty answer')
    return {'run': run, 'pairs': len(rows),
            'query_available': all(row['query_available'] for row in rows),
            'cheap_implications_proved': sum(row['implies'] or row['constant_under'] for row in rows),
            'threshold_implications_proved': sum(row['threshold_implies'] for row in rows)}


def cases(full):
    # Caller-exact K=17; shorter K and zero visits are labeled derivatives.
    values = [('caller', block, 20, 17) for block in (range(20) if full else (0, 19))]
    values += [('short-K-derivative', 0, 20, k) for k in (range(18) if full else (0, 1, 2, 3, 4, 6))]
    values += [('zero-visit-derivative', 0, 0, 17), ('no-work-block-stress', 20, 20, 17),
               ('outside-caller-reset-stress', 0, 40, 2), ('outside-caller-reset-stress', 0, 60, 4)]
    if full:
        values += [('outside-caller-K-scaling', 0, 20, k) for k in (32, 64)]
    return values


def evaluate(args, original, emitted, certificate, spec):
    scope, block, bound, k = spec
    folder = args.output / f'{scope}-b{block}-n{bound}-K{k}'
    folder.mkdir(exist_ok=True)
    bindings = {'%arg3': bound, '%arg4': k, '%arg5': 0, '%arg6': 0,
                '%arg7': 17408, '%arg8': block, '%arg9': 20}
    source = folder / 'input.pto'
    source.write_text(annotate(specialize(emitted, bindings), 'test.trace_existing'))
    run = checked([args.trace_tool, '--structured-trace', source], folder / 'trace.json')
    trace = json.loads(Path(run['stdout']).read_text())
    if not trace.get('accepted') or trace['trace'].get('error'):
        raise AssertionError(trace)
    visits = len(range(block, bound, 20))
    result = analyze_trace(trace['trace'], specialize(original, bindings), visits, k)
    result.update(scope=scope, block=block, outer_bound=bound, run=run)
    result['certificate_comparison'] = compare_evidence(trace['trace'], certificate)
    if scope == 'caller' and not result['fits_six_for_this_execution']:
        raise AssertionError('An admitted caller obstruction requires explicit investigation')
    if result['certificate_comparison']['supplied_edges_failing_bounded_envelope']:
        raise AssertionError('Supplied evidence disagrees with the independent command order')
    (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


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
    # This probes the same unspecialized owner; it does not restrict its domain.
    bound = diagnostic(args, True)
    certificate = bound['result']['certificate']
    original = args.input.read_text()
    source, emitted = args.output / 'symbolic-input.pto', args.output / 'symbolic-child.pto'
    attributes = ('test.contract_math_only, test.contract_region = 5 : i64, '
                  'test.contract_gm_alias = "may-not-alias", test.contract_completion_emit')
    source.write_text(annotate(original, attributes))
    preparation = checked([args.trace_tool, '--compact-region-contract', source], emitted)
    receipt = {'input_sha256': digest(args.input), 'tool_sha256': digest(args.trace_tool),
               'caller_contract_installed_in_compiler': False, 'emission': preparation,
               'certificate_probe': bound, 'sufficient_probe': sufficient_probe(args, original), 'cases': []}
    for spec in cases(args.full):
        result = evaluate(args, original, emitted.read_text(), certificate, spec)
        receipt['cases'].append(result)
        (args.output / 'results.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(spec, [(row['direction'], row['minimum']) for row in result['directional_widths']], flush=True)


if __name__ == '__main__':
    main()
