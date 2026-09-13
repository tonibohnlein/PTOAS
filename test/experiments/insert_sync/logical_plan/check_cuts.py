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
import re
from pathlib import Path
import subprocess
import sys

from observations import SERIAL_DRIVER, population


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_schema(constructor):
    return {'cuts': 'oahs.cuts.validation.v1', 'demands': 'oahs.demands.validation.v1'}[constructor]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--python-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--require-quality', action='store_true')
    parser.add_argument('--constructor', choices=('cuts', 'demands'), default='cuts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.python_root.resolve()))
    from compare_boundaries import compare, run
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    from compare_boundaries import Boundaries
    from measure import children, replay

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

    def native_counters(name):
        trace = (args.output / (name + '.stderr')).read_text()
        counters = {}
        for counter in ('direct_handoffs', 'shared_acknowledgments', 'reused_acknowledgments',
                        'completion_refinements', 'rejected_refinements', 'owned_refinements',
                        'protocol_keys', 'shared_protocol_keys', 'allocation_fallback_scopes', 'allocation_fallback_keys',
                        'allocation_replays', 'rejected_allocation_replays', 'replay_commands_removed',
                        'replayed_fallback_demands', 'entry_episodes', 'entry_reply_families', 'rejected_entry_proposals',
                        'entry_summary_slots', 'entry_summary_scans', 'entry_storage_units', 'entry_candidate_pairs',
                        'entry_witness_cells', 'entry_witnesses', 'entry_source_overlap_rejections',
                        'entry_summary_skipped',
                        'late_entry_candidates', 'late_entry_families', 'late_entry_sites',
                        'rejected_late_entry_families',
                        'choice_demand_candidates', 'choice_demand_families',
                        'rejected_choice_demands', 'choice_demand_work',
                        'choice_demand_reserved_work', 'choice_demand_analysis_work',
                        'choice_demand_analysis_passes', 'choice_demand_budget_pass',
                        'child_return_candidates', 'child_return_acks_removed',
                        'rejected_child_returns', 'child_return_work',
                        'child_return_checks', 'child_return_budget_check', 'child_return_budget_exhausted',
                        'ring_candidates', 'rejected_rings', 'ring_candidate_commands_removed', 'cut_cycles',
                        'deferred_ring_candidates', 'deferred_rings', 'rejected_deferred_rings', 'deferred_protocol_steps',
                        'periodic_deferred_rings', 'periodic_write_overlap_rejections', 'periodic_scalar_work',
                        'periodic_dag_visits', 'periodic_residue_evaluations', 'deferred_eligibility_work',
                        'deferred_discovery_work', 'deferred_discovery_refusals',
                        'deferred_receipt_cells', 'deferred_skipped_families',
                        'rendezvous_packets', 'demand_fallbacks'):
            matches = re.findall(r'\b' + counter + r' (\d+)\b', trace)
            if len(matches) != 1:
                raise RuntimeError('missing/ambiguous native path counter: ' + counter)
            counters[counter] = int(matches[0])
        return counters

    compiler = [sys.executable, '-c', SERIAL_DRIVER, args.python_root.resolve(),
                '--pto-arch=a3', '--pto-level=level3', '--emit-pto-ir']
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    from benchmark_buffers import demand_population, provenance
    from observations import analyze
    selected_population = demand_population() if args.constructor == 'demands' else population()
    for case in selected_population:
        name = case['case_id']
        raw = args.output.resolve() / (name + '.native.pto')
        emitted = args.output.resolve() / (name + '.pto')
        gm = case.get('gm_contract', 'assume-disjoint-arguments')
        verdict = json.loads(invoke(name + '-native', [args.driver, case['source'],
                             args.constructor + ':none', raw, 'conservative', gm]))
        if not verdict['accepted'] or not verdict['atomic']:
            raise RuntimeError('native construction/reconstruction failed: ' + name)
        counters = native_counters(name + '-native') if args.constructor == 'demands' else {}
        if args.constructor == 'demands' and name in ('online_softmax', 'qk_matmul'):
            if counters['entry_summary_slots'] or counters['entry_summary_skipped'] != 1:
                raise RuntimeError('branch-free entry analysis allocated optional first-site population: ' + name)
        if args.constructor == 'demands' and name in ('two_buffer', 'three_buffer'):
            if not counters['periodic_deferred_rings'] or not counters['periodic_write_overlap_rejections']:
                raise RuntimeError('periodic input release / shared-output refusal not exercised: ' + name)
        invoke(name + '-normalize', compiler + [raw, '-o', emitted])
        cpp = None
        if args.constructor == 'demands' and name == 'historical_gemm':
            cpp = args.output.resolve() / (name + '.cpp')
            invoke(name + '-cpp', [arg for arg in compiler if arg != '--emit-pto-ir'] +
                   [emitted, '-o', cpp])
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
                   '--insert-sync-gm-alias=' + gm,
                   case['source'], '-o', reference])
        scenarios = []
        for scenario in case['scenarios']:
            old, old_metrics = run(reference, scenario)
            new, new_metrics = run(emitted, scenario)
            differences = compare(old, new)  # also requires identical payload
            if old.tokens or new.tokens:
                raise RuntimeError('unconsumed final event: ' + name)
            def sync_counts(metrics):
                return {op: metrics['counts'].get(op, 0)
                        for op in ('pto.set_flag', 'pto.wait_flag', 'pto.barrier')}
            old_counts, new_counts = sync_counts(old_metrics), sync_counts(new_metrics)
            def guard_counts(metrics):
                return {op: metrics['scalar_counts'].get(op, 0) for op in ('arith.cmpi', 'scf.if')}
            scenarios.append(dict(name=scenario['name'], reference_executed=old_counts,
                                  candidate_executed=new_counts,
                                  reference_guard_executed=guard_counts(old_metrics),
                                  candidate_guard_executed=guard_counts(new_metrics),
                                  executed_not_worse=sum(new_counts.values()) <= sum(old_counts.values()),
                                  later=sum(d['automatic_requires_later_prefix'] for d in differences),
                                  differences=differences[:5]))
        cases.append(dict(case=name, input_sha256=digest(case['source']), verdict=verdict,
                          gm_contract=gm, native_counters=counters,
                          candidate_static=analyze(emitted)['mechanisms'],
                          reference_static=analyze(reference)['mechanisms'],
                          output=str(emitted), output_sha256=digest(emitted),
                          cpp_output=str(cpp) if cpp else None,
                          cpp_sha256=digest(cpp) if cpp else None,
                          reference=str(reference), reference_sha256=digest(reference),
                          scenarios=scenarios))
        print(json.dumps(dict(case=name, later=sum(s['later'] for s in scenarios))), flush=True)
    source = next(c['source'] for c in population() if c['case_id'] == 'two_buffer')
    mutations = []
    mutation_names = ('drop-set', 'drop-wait', 'duplicate-set', 'wrong-key', 'drop-retirement',
                      'early-publication', 'late-acquisition')
    if args.constructor == 'cuts':
        mutation_names += ('drop-exit-ack',)
    for mutation in mutation_names:
        verdict = json.loads(invoke(mutation, [args.driver, source, args.constructor + ':' + mutation,
                             args.output.resolve() / ('mutated-' + mutation + '.pto')]))
        if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
            raise RuntimeError('mutation escaped reconstruction: ' + mutation)
        mutations.append(dict(mutation=mutation, verdict=verdict))
    qualified = not any(s['later'] or not s['executed_not_worse']
                        for c in cases for s in c['scenarios'])
    path_checks = []
    if args.constructor == 'demands':
        for name, slots in (('one_buffer', 1), ('two_buffer', 2), ('three_buffer', 3)):
            row = next(c for c in cases if c['case'] == name)
            counters = row['native_counters']
            if (counters['ring_candidates'] != 1 or counters['rejected_rings'] or
                    counters['cut_cycles'] != 2 * slots or counters['ring_candidate_commands_removed'] < slots):
                raise RuntimeError('native recurring handoff path not exercised: ' + name)
            path_checks.append(dict(name=name + '-recurring-rings', counters=counters))
        shared = Path(__file__).parent / 'structured_inputs/demand_ring_shared.pto'
        shared_raw = args.output.resolve() / 'shared-rings.native.pto'
        verdict = json.loads(invoke('shared-rings', [args.driver, shared, 'demands:none', shared_raw]))
        counters = native_counters('shared-rings')
        if (not verdict['accepted'] or not verdict['atomic'] or counters['cut_cycles'] != 2 or
                counters['rejected_rings'] or not counters['ring_candidate_commands_removed']):
            raise RuntimeError('native multi-cell recurring handoff sharing not exercised')
        shared_normalized = args.output.resolve() / 'shared-rings.pto'
        invoke('shared-rings-normalize', compiler + [shared_raw, '-o', shared_normalized])
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(shared_normalized.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            observer = Boundaries()
            for n, take in ((0, True), (3, False), (1, True), (3, True), (0, False), (2, True)):
                replay(function, ['src', 'dst', n, take], observer=observer.observe)
                if observer.tokens:
                    raise RuntimeError('shared recurring handoff exports an unconsumed publication')
        path_checks.append(dict(name='shared-rings', verdict=verdict, counters=counters,
                                source_sha256=digest(shared), output_sha256=digest(shared_normalized)))
        deferred = Path(__file__).parent / 'structured_inputs/demand_deferred_ring.pto'
        deferred_raw = args.output.resolve() / 'deferred-rings.native.pto'
        verdict = json.loads(invoke('deferred-rings', [args.driver, deferred, 'demands:none', deferred_raw]))
        counters = native_counters('deferred-rings')
        if (not verdict['accepted'] or not verdict['atomic'] or not counters['deferred_rings'] or
                counters['rejected_deferred_rings']):
            raise RuntimeError('native earliest recurring release path not exercised')
        deferred_normalized = args.output.resolve() / 'deferred-rings.pto'
        invoke('deferred-rings-normalize', compiler + [deferred_raw, '-o', deferred_normalized])
        episodes = []
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(deferred_normalized.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            observer = Boundaries()
            for lower, upper in ((0, 0), (3, 4), (-2, 1), (9, 4), (5, 7), (0, 1)):
                metrics = replay(function, ['src', 'dst', lower, upper], observer=observer.observe)
                if observer.tokens:
                    raise RuntimeError('deferred release exports an unconsumed final publication')
                if lower >= upper and (metrics['counts'].get('pto.set_flag', 0) or
                                       metrics['counts'].get('pto.wait_flag', 0)):
                    raise RuntimeError('empty deferred release owner executes an event')
                episodes.append(dict(lower=lower, upper=upper, counts=metrics['counts'],
                                     scalar_counts=metrics['scalar_counts']))
        path_checks.append(dict(name='deferred-rings', verdict=verdict, counters=counters, episodes=episodes,
                                source_sha256=digest(deferred), output_sha256=digest(deferred_normalized)))
        verdict = json.loads(invoke('deferred-rollback', [args.driver, deferred,
            'demands:reject-deferred-rings', args.output.resolve() / 'deferred-rollback.pto']))
        counters = native_counters('deferred-rollback')
        rejection = re.search(r'structured deferred_rejection stage (\S+) reason (.+)',
                              (args.output / 'deferred-rollback.stderr').read_text())
        if (not verdict['accepted'] or not verdict['atomic'] or counters['deferred_rings'] or
                not counters['rejected_deferred_rings'] or not rejection):
            raise RuntimeError('native deferred rollback or rejection diagnostic failed')
        path_checks.append(dict(name='deferred-rollback', verdict=verdict, counters=counters,
                                rejection_stage=rejection[1], rejection_reason=rejection[2]))
        for mutation in ('deferred-wrong-previous', 'deferred-wrong-exit', 'deferred-drop-previous',
                         'deferred-drop-exit', 'deferred-late-previous', 'deferred-early-exit',
                         'wrong-key', 'early-publication'):
            verdict = json.loads(invoke(mutation, [args.driver, deferred,
                'demands:' + mutation, args.output.resolve() / (mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('deferred release corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name=mutation, verdict=verdict))
        periodic = Path(__file__).parent / 'structured_inputs/demand_periodic_ring.pto'
        periodic_raw = args.output.resolve() / 'periodic-rings.native.pto'
        verdict = json.loads(invoke('periodic-rings', [args.driver, periodic, 'demands:none', periodic_raw]))
        counters = native_counters('periodic-rings')
        if not verdict['accepted'] or not verdict['atomic'] or counters['periodic_deferred_rings'] != 2:
            raise RuntimeError('native nonzero first-active release path not exercised')
        periodic_normalized = args.output.resolve() / 'periodic-rings.pto'
        invoke('periodic-rings-normalize', compiler + [periodic_raw, '-o', periodic_normalized])
        episodes = []
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(periodic_normalized.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            observer = Boundaries()
            for upper in (0, 2, 3, 4, 7, 10, 3, 4):
                metrics = replay(function, ['src', 'dst', upper], observer=observer.observe)
                visits = sum(iv % 3 == 0 for iv in range(2, upper))
                if observer.tokens:
                    raise RuntimeError('periodic release exports an unconsumed publication')
                if any(metrics['counts'].get(op, 0) != 4 * visits for op in ('pto.set_flag', 'pto.wait_flag')):
                    raise RuntimeError('periodic release has incorrect first/steady/final command count')
                episodes.append(dict(upper=upper, active_visits=visits, counts=metrics['counts'],
                                     scalar_counts=metrics['scalar_counts']))
        path_checks.append(dict(name='periodic-rings', verdict=verdict, counters=counters, episodes=episodes,
                                source_sha256=digest(periodic), output_sha256=digest(periodic_normalized)))
        periodic_text = periodic.read_text()
        declined = {
            'period-too-large': periodic_text.replace('%c3 = arith.constant 3', '%c3 = arith.constant 33'),
            'unknown-predicate': periodic_text.replace('arith.cmpi eq, %slot, %c0', 'arith.cmpi eq, %slot, %upper'),
            'never-active': periodic_text.replace('%c0 = arith.constant 0', '%c0 = arith.constant 99'),
            'first-active-overflow': periodic_text.replace('%lower = arith.constant 2',
                '%lower = arith.constant 9223372036854775806').replace('%c3 = arith.constant 3',
                '%c3 = arith.constant 32'),
        }
        for name, fixture_text in declined.items():
            fixture = args.output.resolve() / (name + '.input.pto')
            fixture.write_text(fixture_text)
            verdict = json.loads(invoke(name, [args.driver, fixture, 'demands:none',
                                               args.output.resolve() / (name + '.pto')]))
            counters = native_counters(name)
            if not verdict['accepted'] or not verdict['atomic'] or counters['periodic_deferred_rings']:
                raise RuntimeError('optional periodic refusal poisoned baseline: ' + name)
            path_checks.append(dict(name=name, verdict=verdict, counters=counters, source_sha256=digest(fixture)))
        for mutation in ('periodic-wrong-first', 'periodic-wrong-exit', 'deferred-drop-previous',
                         'deferred-drop-exit', 'deferred-late-previous', 'deferred-early-exit',
                         'wrong-key', 'early-publication'):
            verdict = json.loads(invoke('periodic-' + mutation, [args.driver, periodic,
                'demands:' + mutation, args.output.resolve() / ('periodic-' + mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('periodic release corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name='periodic-' + mutation, verdict=verdict))
        for mutation in ('wrong-key', 'drop-set', 'drop-wait', 'early-publication', 'late-acquisition'):
            verdict = json.loads(invoke('shared-rings-' + mutation, [args.driver, shared,
                'demands:' + mutation, args.output.resolve() / ('shared-rings-' + mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('shared recurring handoff corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name='shared-rings-' + mutation, verdict=verdict))
        for counter in ('direct_handoffs', 'reused_acknowledgments', 'owned_refinements',
                        'protocol_keys', 'shared_protocol_keys',
                        'rendezvous_packets', 'demand_fallbacks'):
            if not any(c['native_counters'][counter] > 0 for c in cases):
                raise RuntimeError('native demand path not exercised: ' + counter)
        if not any(c['native_counters']['completion_refinements'] >
                   c['native_counters']['rejected_refinements'] for c in cases):
            raise RuntimeError('native accepted refinement not exercised')
        # Corrupt only the optional pre-emission refinement. The initial plan
        # must survive and pass fresh native reconstruction, not an IR rollback
        # that incorrectly disguises a failed emitted-plan check as success.
        rollback_path = args.output.resolve() / 'rejected-refinement.pto'
        verdict = json.loads(invoke('rejected-refinement', [args.driver, source,
                              'demands:reject-refinement', rollback_path]))
        counters = native_counters('rejected-refinement')
        if not verdict['accepted'] or not verdict['atomic'] or counters['rejected_refinements'] != 1:
            raise RuntimeError('native rejected-refinement rollback not exercised')
        path_checks.append(dict(name='rejected-refinement', verdict=verdict, counters=counters))
        overlap = Path(__file__).parent / 'structured_inputs/demand_key_overlap.pto'
        overlap_output = args.output.resolve() / 'overlap-fallback.pto'
        verdict = json.loads(invoke('overlap-fallback', [args.driver, overlap, 'demands:without-allocation-replay', overlap_output]))
        counters = native_counters('overlap-fallback')
        if (not verdict['accepted'] or not verdict['atomic'] or counters['allocation_fallback_scopes'] != 1 or
                counters['direct_handoffs'] != 5 or counters['allocation_fallback_keys'] != 3 or
                counters['rendezvous_packets'] != 3):
            raise RuntimeError('overlapping logical publications did not exercise qualified allocation fallback')
        path_checks.append(dict(name='overlap-fallback', verdict=verdict, counters=counters,
                                source_sha256=digest(overlap), output_sha256=digest(overlap_output)))
        for mode in ('none', 'reject-allocation-replay'):
            name = 'overlap-replay-' + mode
            output = args.output.resolve() / (name + '.pto')
            verdict = json.loads(invoke(name, [args.driver, overlap, 'demands:' + mode, output]))
            counters = native_counters(name)
            if not verdict['accepted'] or not verdict['atomic'] or counters['allocation_replays'] != 1:
                raise RuntimeError('native bounded allocation replay was not exercised')
            if mode == 'none':
                if (counters['rejected_allocation_replays'] or counters['replayed_fallback_demands'] != 1 or
                        counters['replay_commands_removed'] != 10 or counters['rendezvous_packets'] != 1 or
                        counters['shared_acknowledgments'] != 0):
                    raise RuntimeError('later demands did not reuse fallback completion')
            elif counters['rejected_allocation_replays'] != 1 or digest(output) != digest(overlap_output):
                raise RuntimeError('rejected allocation replay did not preserve exact original plan')
            path_checks.append(dict(name=name, verdict=verdict, counters=counters))
        for mutation in ('wrong-key', 'duplicate-set', 'drop-wait'):
            verdict = json.loads(invoke('overlap-' + mutation, [args.driver, overlap, 'demands:' + mutation,
                args.output.resolve() / ('overlap-' + mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('mixed raw/canonical corruption escaped native reconstruction')
            path_checks.append(dict(name='overlap-' + mutation, verdict=verdict))
        fallback = Path(__file__).parent / 'structured_inputs/demand_fallback_nested.pto'
        raw = args.output.resolve() / 'fallback-native.pto'
        verdict = json.loads(invoke('fallback-native', [args.driver, fallback, 'demands:fallback:none', raw]))
        counters = native_counters('fallback-native')
        if (not verdict['accepted'] or not verdict['atomic'] or counters['direct_handoffs'] != 0 or
                counters['rendezvous_packets'] == 0 or counters['demand_fallbacks'] == 0):
            raise RuntimeError('native fallback packet path not exercised')
        fallback_verdict, fallback_counters = verdict, counters
        normalized = args.output.resolve() / 'fallback.pto'
        invoke('fallback-normalize', compiler + [raw, '-o', normalized])
        choice_residual = Path(__file__).parent / 'structured_inputs/demand_choice_prefix.pto'
        choice_prefix = args.output.resolve() / 'choice-first-only.input.pto'
        residual_text = choice_residual.read_text()
        if residual_text.count('pto.tabs ins(%b') != 2:
            raise RuntimeError('unexpected residual-B fixture population')
        choice_prefix.write_text(residual_text.replace('pto.tabs ins(%b', 'pto.tabs ins(%a'))
        choice_outputs, choice_profiles = {}, {}
        for arm, mode in (('selected', 'none'), ('disabled', 'without-choice-demands'),
                          ('rejected', 'reject-choice-demands')):
            name = 'choice-prefix-' + arm
            choice_raw = args.output.resolve() / (name + '.native.pto')
            verdict = json.loads(invoke(name, [args.driver, choice_prefix, 'demands:' + mode, choice_raw]))
            if not verdict['accepted'] or not verdict['atomic']:
                raise RuntimeError('ordinary Choice demand failed: ' + arm)
            choice_profiles[arm] = native_counters(name)
            choice_output = args.output.resolve() / (name + '.pto')
            invoke(name + '-normalize', compiler + [choice_raw, '-o', choice_output])
            choice_outputs[arm] = choice_output
        if (not choice_profiles['selected']['choice_demand_families'] or
                choice_profiles['disabled']['choice_demand_families'] or
                not choice_profiles['rejected']['rejected_choice_demands']):
            raise RuntimeError('ordinary Choice selection and rollback not exercised')
        if choice_outputs['disabled'].read_bytes() != choice_outputs['rejected'].read_bytes():
            raise RuntimeError('rejected Choice candidate changed the baseline output')
        choice_scenarios = []
        for count, take in ((0, False), (1, False), (1, True), (3, False), (3, True)):
            scenario = dict(arguments=['src', count, take])
            old, old_metrics = run(choice_outputs['disabled'], scenario)
            new, new_metrics = run(choice_outputs['selected'], scenario)
            differences = compare(old, new)
            # Both tabs use A in this derived positive input. The independent
            # second load must not gate either consumer's acquired prefix.
            for iteration in range(count):
                first = 4 * iteration + 2
                second = first + 1
                if (new.before[first]['completed'].get('PIPE_MTE2', -1) != first - 2 or
                        new.before[second]['completed'].get('PIPE_MTE2', -1) != first - 2):
                    raise RuntimeError('Choice first-prefix placement or residual B credit is wrong')
            if count and not any(not d['automatic_requires_later_prefix'] for d in differences):
                raise RuntimeError('Choice candidate has no observed earlier acquisition prefix')
            choice_scenarios.append(dict(arguments=scenario['arguments'], differences=differences,
                                         baseline_counts=old_metrics['counts'],
                                         selected_counts=new_metrics['counts'],
                                         baseline_scalar=old_metrics['scalar_counts'],
                                         selected_scalar=new_metrics['scalar_counts']))
        path_checks.append(dict(name='ordinary-choice-prefix', profiles=choice_profiles,
                                source_sha256=digest(choice_prefix),
                                outputs={arm: digest(path) for arm, path in choice_outputs.items()},
                                scenarios=choice_scenarios))
        residual_outputs, residual_profiles = {}, {}
        for arm, mode in (('selected', 'none'), ('disabled', 'without-child-returns'),
                          ('rejected', 'reject-child-returns')):
            name = 'choice-residual-' + arm
            raw = args.output.resolve() / (name + '.native.pto')
            output = args.output.resolve() / (name + '.pto')
            verdict = json.loads(invoke(name, [args.driver, choice_residual, 'demands:' + mode, raw]))
            if not verdict['accepted'] or not verdict['atomic']:
                raise RuntimeError('residual-B child-return construction failed')
            invoke(name + '-normalize', compiler + [raw, '-o', output])
            residual_outputs[arm] = output
            residual_profiles[arm] = native_counters(name)
        if (not residual_profiles['selected']['choice_demand_families'] or
                not residual_profiles['selected']['child_return_acks_removed'] or
                residual_profiles['disabled']['child_return_acks_removed'] or
                not residual_profiles['rejected']['rejected_child_returns']):
            raise RuntimeError('child-return selection or fault rollback was not exercised')
        if residual_outputs['rejected'].read_bytes() != residual_outputs['disabled'].read_bytes():
            raise RuntimeError('rejected child-return candidate changed the exact baseline')
        residual_scenarios = []
        for count, take in ((0, False), (1, False), (1, True), (3, False), (3, True)):
            scenario = dict(arguments=['src', count, take])
            old, old_metrics = run(residual_outputs['disabled'], scenario)
            new, new_metrics = run(residual_outputs['selected'], scenario)
            for iteration in range(count):
                first = 4 * iteration + 2
                if (new.before[first]['completed'].get('PIPE_MTE2', -1) != first - 2 or
                        new.before[first + 1]['completed'].get('PIPE_MTE2', -1) != first - 1):
                    raise RuntimeError('child-return candidate broadened A or lost residual B readiness')
            event_cost = lambda metrics: sum(metrics['counts'].get(op, 0)
                                             for op in ('pto.set_flag', 'pto.wait_flag'))
            if event_cost(new_metrics) - event_cost(old_metrics) > 2 * count:
                raise RuntimeError('child-return candidate exceeded the existing per-owner cost cap')
            if old_metrics['scalar_counts'] != new_metrics['scalar_counts']:
                raise RuntimeError('child-return candidate added scalar participation work')
            residual_scenarios.append(dict(arguments=scenario['arguments'], differences=compare(old, new),
                                           baseline_counts=old_metrics['counts'],
                                           selected_counts=new_metrics['counts']))
        path_checks.append(dict(name='ordinary-choice-residual-child-return', profiles=residual_profiles,
                                source_sha256=digest(choice_residual),
                                output_sha256=digest(residual_outputs['selected']), scenarios=residual_scenarios))
        for mutation in ('child-drop-return', 'child-wrong-return-key'):
            name = 'choice-residual-' + mutation
            verdict = json.loads(invoke(name, [args.driver, choice_residual, 'demands:' + mutation,
                                               args.output.resolve() / (name + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('actual child-return corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name=name, verdict=verdict))
        for mutation in ('drop-set', 'drop-wait', 'duplicate-set', 'wrong-key',
                         'early-publication', 'late-acquisition'):
            name = 'choice-prefix-' + mutation
            verdict = json.loads(invoke(name, [args.driver, choice_prefix, 'demands:' + mutation,
                                               args.output.resolve() / (name + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('ordinary Choice corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name=name, verdict=verdict))
        authored = Path(__file__).parent / 'structured_inputs/demand_authored_guard.pto'
        verdict = json.loads(invoke('entry-authored-guard', [args.driver, authored, 'demands:expect-unsupported',
            args.output.resolve() / 'entry-authored-guard.pto']))
        if (verdict['accepted'] or not verdict['expected'] or not verdict['atomic'] or
                'explicit-synchronization-input' not in verdict['reason']):
            raise RuntimeError('original guarded synchronization was mistaken for generated precision')
        path_checks.append(dict(name='entry-authored-guard', verdict=verdict, source_sha256=digest(authored)))
        entry = Path(__file__).parent / 'structured_inputs/demand_entry.pto'
        entry_raw = args.output.resolve() / 'entry-native.pto'
        verdict = json.loads(invoke('entry-native', [args.driver, entry, 'demands:none', entry_raw]))
        counters = native_counters('entry-native')
        if (not verdict['accepted'] or not verdict['atomic'] or counters['entry_episodes'] != 2 or
                counters['entry_reply_families'] != 1):
            raise RuntimeError('native incoming first-consumer episodes not exercised')
        entry_normalized = args.output.resolve() / 'entry.pto'
        invoke('entry-normalize', compiler + [entry_raw, '-o', entry_normalized])
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(entry_normalized.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            observer = Boundaries()
            for lower, upper, take in ((3, 3, True), (3, 6, False), (3, 4, True),
                                       (5, 8, True), (8, 5, True), (0, 2, True)):
                start = len(observer.payload)
                metrics = replay(function, ['src', lower, upper, take], observer=observer.observe)
                if observer.tokens:
                    raise RuntimeError('incoming episode exports an unconsumed token')
                if not take or lower >= upper:
                    if metrics['counts'].get('pto.set_flag', 0) or metrics['counts'].get('pto.wait_flag', 0):
                        raise RuntimeError('empty incoming episode executes an event')
                else:
                    reads = [i for i in range(start, len(observer.payload)) if observer.payload[i][0] == 'pto.tabs']
                    if [observer.before[i]['completed'].get('PIPE_MTE2', -1) for i in reads[:2]] != [start, start + 1]:
                        raise RuntimeError('incoming publication captured a later unrelated load')
        path_checks.append(dict(name='incoming-episodes', verdict=verdict, counters=counters,
                                source_sha256=digest(entry), output_sha256=digest(entry_normalized)))
        verdict = json.loads(invoke('entry-rejected', [args.driver, entry, 'demands:reject-entry-proposal',
            args.output.resolve() / 'entry-rejected.pto']))
        counters = native_counters('entry-rejected')
        if not verdict['accepted'] or not verdict['atomic'] or counters['entry_episodes'] or counters['rejected_entry_proposals'] != 1:
            raise RuntimeError('rejected optional entry proposal did not preserve native construction')
        path_checks.append(dict(name='entry-rejected', verdict=verdict, counters=counters))
        for mutation in ('entry-wrong-first', 'entry-wrong-nonempty', 'entry-drop-first',
                         'entry-drop-ack', 'entry-late-first', 'entry-inject-packet'):
            verdict = json.loads(invoke(mutation, [args.driver, entry, 'demands:' + mutation,
                args.output.resolve() / (mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('incoming-episode corruption escaped native reconstruction: ' + mutation)
            path_checks.append(dict(name=mutation, verdict=verdict))
        choice_entry = Path(__file__).parent / 'structured_inputs/demand_entry_choice.pto'
        choice_raw = args.output.resolve() / 'entry-choice-native.pto'
        choice_verdict = json.loads(invoke('entry-choice-native',
            [args.driver, choice_entry, 'demands:none', choice_raw]))
        choice_counters = native_counters('entry-choice-native')
        if (not choice_verdict['accepted'] or not choice_verdict['atomic'] or
                choice_counters['entry_episodes'] != 1 or not choice_counters['entry_summary_slots'] or
                not choice_counters['entry_witnesses']):
            raise RuntimeError('shared incoming Choice-cut episode not exercised')
        choice_normalized = args.output.resolve() / 'entry-choice.pto'
        invoke('entry-choice-normalize', compiler + [choice_raw, '-o', choice_normalized])
        choice_scenarios = []
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(choice_normalized.read_text())
            original = ir.Module.parse(choice_entry.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            original_function = next(op for op in children(original.operation) if op.name == 'func.func')
            observer = Boundaries()
            for outer, lower, upper, take, other in (
                    (0, 2, 5, True, True), (2, 2, 2, True, False),
                    (2, 2, 5, False, True), (1, 3, 4, True, True),
                    (3, 2, 5, True, False), (2, 5, 2, True, True),
                    (2, 1, 3, True, True), (1, 2, 4, False, False)):
                arguments = ['src', outer, lower, upper, take, other]
                start = len(observer.payload)
                metrics = replay(function, arguments, observer=observer.observe)
                original_metrics = replay(original_function, arguments)
                if metrics['payload_sha256'] != original_metrics['payload_sha256']:
                    raise RuntimeError('shared incoming Choice episode changed original payload')
                if observer.tokens:
                    raise RuntimeError('shared incoming Choice episode exports an unconsumed token')
                choice_scenarios.append(dict(arguments=arguments,
                    executed={name: metrics['counts'].get(name, 0)
                              for name in ('pto.set_flag', 'pto.wait_flag', 'pto.barrier')},
                    scalar={name: metrics['scalar_counts'].get(name, 0)
                            for name in ('arith.cmpi', 'scf.if')}))
                latest_source = start
                for index in range(start, len(observer.payload)):
                    if observer.payload[index][0] == 'pto.tload':
                        latest_source = index
                    elif observer.payload[index][0] == 'pto.tabs':
                        if observer.before[index]['completed'].get('PIPE_MTE2', -1) < latest_source:
                            raise RuntimeError('Choice consumer did not acquire incoming outer-visit history')
        path_checks.append(dict(name='incoming-choice-episode', verdict=choice_verdict,
                                counters=choice_counters, source_sha256=digest(choice_entry),
                                output_sha256=digest(choice_normalized), scenarios=choice_scenarios))
        source_inside = args.output.resolve() / 'entry-choice-source-inside.pto'
        source_inside.write_text(choice_entry.read_text().replace('          scf.if %other {',
            '          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf16>) '
            'outs(%a : !pto.tile_buf<vec, 16x16xf16>)\n          scf.if %other {'))
        verdict = json.loads(invoke('entry-choice-source-inside', [args.driver, source_inside,
            'demands:none', args.output.resolve() / 'entry-choice-source-inside-native.pto']))
        counters = native_counters('entry-choice-source-inside')
        if not verdict['accepted'] or not verdict['atomic'] or counters['entry_episodes']:
            raise RuntimeError('source effect in one arm did not disable optional incoming-only credit')
        path_checks.append(dict(name='entry-choice-source-inside', verdict=verdict, counters=counters,
                                source_sha256=digest(source_inside)))
        for mutation in ('entry-wrong-first', 'entry-wrong-nonempty', 'entry-drop-first',
                         'entry-drop-ack', 'entry-late-first', 'entry-conditional-first'):
            name = 'choice-' + mutation
            verdict = json.loads(invoke(name, [args.driver, choice_entry, 'demands:' + mutation,
                args.output.resolve() / (name + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('Choice incoming-episode corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name=name, verdict=verdict))
        late_entry = Path(__file__).parent / 'structured_inputs/demand_entry_late_choice.pto'
        late_outputs, late_profiles = {}, {}
        for arm, mode in (('selected', 'none'), ('common', 'without-late-entry'),
                          ('rejected', 'reject-late-entry')):
            name = 'entry-late-choice-' + arm
            late_raw = args.output.resolve() / (name + '.native.pto')
            verdict = json.loads(invoke(name, [args.driver, late_entry, 'demands:' + mode, late_raw]))
            if not verdict['accepted'] or not verdict['atomic']:
                raise RuntimeError('late incoming Choice arm failed: ' + arm)
            late_profiles[arm] = native_counters(name)
            late_normalized = args.output.resolve() / (name + '.pto')
            invoke(name + '-normalize', compiler + [late_raw, '-o', late_normalized])
            late_outputs[arm] = late_normalized
        if (late_profiles['selected']['late_entry_families'] != 1 or
                late_profiles['selected']['late_entry_sites'] != 3 or
                late_profiles['common']['late_entry_families'] or
                not late_profiles['rejected']['rejected_late_entry_families']):
            raise RuntimeError('late incoming Choice selection/fallback not exercised')
        if late_outputs['common'].read_bytes() != late_outputs['rejected'].read_bytes():
            raise RuntimeError('rejected late placement did not retain the exact common plan')
        static_guards = {arm: path.read_text().count('arith.cmpi eq') for arm, path in late_outputs.items()}
        if static_guards != dict(selected=3, common=1, rejected=1):
            raise RuntimeError('unexpected late First static guard population')
        late_scenarios = []
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            modules = {arm: ir.Module.parse(path.read_text()) for arm, path in late_outputs.items()}
            functions = {arm: next(op for op in children(module.operation) if op.name == 'func.func')
                         for arm, module in modules.items()}
            observers = {arm: Boundaries() for arm in functions}
            original_module = ir.Module.parse(late_entry.read_text())
            original_function = next(op for op in children(original_module.operation) if op.name == 'func.func')
            for outer, lower, upper, take, other in (
                    (0, 2, 5, True, True), (2, 2, 2, False, False),
                    (2, 2, 5, False, True), (1, 3, 4, True, True),
                    (3, 2, 5, True, False), (2, 5, 2, True, True),
                    (2, 1, 3, True, True), (1, 2, 4, False, False)):
                arguments = ['src', outer, lower, upper, take, other]
                original_metrics = replay(original_function, arguments)
                metrics = {arm: replay(function, arguments, observer=observers[arm].observe)
                           for arm, function in functions.items()}
                for arm, metric in metrics.items():
                    if metric['payload_sha256'] != original_metrics['payload_sha256'] or observers[arm].tokens:
                        raise RuntimeError('late First changed payload or exported a token: ' + arm)
                executed = {arm: {name: metric['counts'].get(name, 0)
                                  for name in ('pto.set_flag', 'pto.wait_flag', 'pto.barrier')}
                            for arm, metric in metrics.items()}
                if executed['selected'] != executed['common']:
                    raise RuntimeError('exclusive First sites changed executed event counts')
                scalar = {arm: {name: metric['scalar_counts'].get(name, 0)
                                for name in ('arith.cmpi', 'scf.if')}
                          for arm, metric in metrics.items()}
                late_scenarios.append(dict(arguments=arguments, executed=executed, scalar=scalar))
        path_checks.append(dict(name='late-incoming-choice', profiles=late_profiles,
                                source_sha256=digest(late_entry),
                                outputs={arm: digest(path) for arm, path in late_outputs.items()},
                                static_first_guards=static_guards, scenarios=late_scenarios))
        for mutation in ('entry-wrong-first', 'entry-drop-first', 'entry-drop-ack',
                         'entry-late-first', 'entry-stack-first'):
            name = 'late-choice-' + mutation
            verdict = json.loads(invoke(name, [args.driver, late_entry, 'demands:' + mutation,
                args.output.resolve() / (name + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('late incoming Choice corruption escaped reconstruction: ' + mutation)
            path_checks.append(dict(name=name, verdict=verdict))
        with ir.Context() as context:
            context.enable_multithreading(False)
            pto.register_dialect(context, load=True)
            module = ir.Module.parse(normalized.read_text())
            function = next(op for op in children(module.operation) if op.name == 'func.func')
            observer = Boundaries()
            for n, take in ((0, True), (3, False), (1, True), (3, True), (0, False), (2, True)):
                replay(function, ['src', n, take], observer=observer.observe)
                if observer.tokens:
                    raise RuntimeError('fallback exports an unconsumed token')
        path_checks.append(dict(name='nested-fallback', verdict=fallback_verdict, counters=fallback_counters,
                                source_sha256=digest(fallback), output_sha256=digest(normalized)))
        for mutation in ('wrong-key', 'duplicate-set', 'drop-wait'):
            verdict = json.loads(invoke('fallback-' + mutation, [args.driver, fallback,
                'demands:fallback:' + mutation, args.output.resolve() / ('fallback-' + mutation + '.pto')]))
            if verdict['accepted'] or not verdict['expected'] or not verdict['atomic']:
                raise RuntimeError('fallback corruption escaped native reconstruction')
            path_checks.append(dict(name='fallback-' + mutation, verdict=verdict))
    summary = dict(schema=report_schema(args.constructor), cases=cases, mutations=mutations,
                   path_checks=path_checks,
                   provenance=provenance(args.python_root.resolve()),
                   manifest_sha256=digest(Path(__file__).parent /
                       ('demand_manifest.json' if args.constructor == 'demands' else 'checkpoint/manifest.json')),
                   commands=commands, driver_sha256=digest(args.driver),
                   runner_sha256=digest(__file__),
                   baseline_sha256=digest(args.baseline) if args.baseline else None,
                   constructor=args.constructor, quality_qualified=qualified, device='NOT_RUN',
                   scope='native reconstruction and finite completion boundaries; not compilation timing')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.require_quality and not qualified:
        raise SystemExit('completion-boundary quality gate remains open')


if __name__ == '__main__':
    main()
