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
                        'ring_candidates', 'rejected_rings', 'ring_candidate_commands_removed', 'cut_cycles',
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
            scenarios.append(dict(name=scenario['name'], reference_executed=old_counts,
                                  candidate_executed=new_counts,
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
        normalized = args.output.resolve() / 'fallback.pto'
        invoke('fallback-normalize', compiler + [raw, '-o', normalized])
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
        path_checks.append(dict(name='nested-fallback', verdict=verdict, counters=counters,
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
