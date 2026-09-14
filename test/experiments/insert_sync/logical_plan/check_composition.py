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
    parser.add_argument('--corpus-constructor', choices=('conservative', 'demands'), default='conservative',
                        help='corpus-only arm; native baseline mutation checks remain conservative')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    historical_evidence = None
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
                   returncode=code, status=status, seconds=elapsed, log=stem,
                   output_sha256=digest(stdout.encode()), stderr_sha256=digest(stderr.encode()),
                   measurement="fresh", hardware_contract="conservative")
        if code != 0:
            reason = re.search(r'structured construction: [^;]+; (.*?); work=', stderr)
            row['first_refusal'] = reason.group(1) if reason else next(
                (line.split('error: ', 1)[-1] for line in stderr.splitlines()
                 if 'error: ' in line), status)
        if category == 'mutation':
            row['verdict'] = json.loads(stdout) if code == 0 else None
            if row['verdict']:
                for contract in ('hardware_contract', 'gm_alias', 'ownership_contract', 'ownership_credit'):
                    if contract in row['verdict']:
                        row[contract] = row['verdict'][contract]
        else:
            row['sets'] = stdout.count('pto.set_flag[')
            row['waits'] = stdout.count('pto.wait_flag[')
            row['barriers'] = stdout.count('pto.barrier')
            row['cmo'] = stdout.count('pto.cmo.cacheinvalid')
            row['fences'] = stdout.count('pto.fence.barrier_all')
        rows.append(row)
        if expected is not None and ((code == 0) != expected or status in ('timeout', 'crashed')):
            raise RuntimeError(f'{name}: unexpected result ({status})\n{stderr}\n{stdout}')
        return row

    def compile_case(name, source, gm='may-alias', category='native', expected=True):
        demands = category == 'corpus' and args.corpus_constructor == 'demands'
        planner = 'composition' if demands else 'structured'
        precision = 'true' if demands else 'false'
        row = run(name, [args.opt, '--mlir-disable-threading',
                   f'--pto-insert-sync=planner={planner} structured-precision={precision} '
                   'logical-work-budget=0 gm-alias=' + gm, source],
                   source, category, expected)
        row.update(planner=planner, precision=demands)
        return row

    fixtures = HERE / 'structured_inputs'
    positive = ('composition_while_forwarding', 'nested_mixed_sequence',
                'nested_varying_choice', 'nested_varying_bound', 'nested_three_levels',
                'unknown_guard', 'ordinal_dynamic_step', 'ordinal_negative_lower',
                'sequential_cross_pipe', 'sequential_same_pipe', 'section_vector',
                'composition_tci', 'composition_tconcat', 'composition_scalar_memory',
                'composition_scalar_visibility', 'composition_tmrgsort',
                'composition_remote_wait_scalar_visibility',
                'composition_remote_wait_loop_scalar_visibility',
                'composition_tput_macro', 'composition_tget_macro')
    positive += ('composition_authored_remote_signal',)
    for name in positive:
        source = fixtures / (name + '.pto')
        compile_case(name, source, gm='assume-disjoint-arguments')
    persistent_outputs = {}
    for name in ('persistent_early_vector', 'persistent_multiple_readers',
                 'persistent_bias_table'):
        source = fixtures / (name + '.pto')
        output = args.output / (name + '-demands.pto')
        row = run(name + '-demands',
                  [args.driver, source, 'demands:none', output],
                  source, 'mutation', True)
        if not row['verdict']['accepted']:
            raise RuntimeError(name + ': persistent demand construction was not accepted')
        persistent_outputs[name] = output.read_text()
    if any(run_name != 'persistent_bias_table' and
           next(row['verdict']['barriers'] for row in rows
                if row['id'] == run_name + '-demands') != 0
           for run_name in persistent_outputs):
        raise RuntimeError('persistent vector/multi-reader bodies retained a barrier')

    early = persistent_outputs['persistent_early_vector']
    first_tabs = early.find('pto.tabs')
    second_tabs = early.find('pto.tabs', first_tabs + 1)
    release = early.find('pto.set_flag[<PIPE_V>, <PIPE_MTE2>', first_tabs)
    if min(first_tabs, second_tabs, release) < 0 or not first_tabs < release < second_tabs:
        raise RuntimeError('persistent-early-vector: release was delayed past unrelated vector work')
    multi = persistent_outputs['persistent_multiple_readers']
    first_store = multi.find('pto.tstore')
    second_store = multi.find('pto.tstore', first_store + 1)
    release = multi.find('pto.set_flag[<PIPE_MTE3>, <PIPE_MTE2>', first_store)
    if min(first_store, second_store, release) < 0 or not first_store < release < second_store:
        raise RuntimeError('persistent-multiple-readers: MTE3 release was delayed past unrelated work')
    bias = persistent_outputs['persistent_bias_table']
    biased = bias.find('pto.tmatmul.bias')
    plain = bias.find('pto.tmatmul ins', biased + 1)
    release = bias.find('pto.set_flag[<PIPE_M>, <PIPE_MTE1>', biased)
    if min(biased, plain, release) < 0 or not biased < release < plain:
        raise RuntimeError('persistent-bias-table: release was delayed past the unrelated MMAD')
    for name, mutation in (('persistent_early_vector', 'persistent-drop-cleanup'),
                           ('persistent_multiple_readers', 'persistent-drop-mte3-release')):
        source = fixtures / (name + '.pto')
        row = run(name + '-' + mutation,
                  [args.driver, source, 'demands:' + mutation,
                   args.output / (name + '-' + mutation + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['mutation_applied'] or row['verdict']['accepted']:
            raise RuntimeError(name + ': persistent endpoint mutation escaped verification')
    for name, qualified in (('mmad_boundary_10', False), ('mmad_large_11', True)):
        source = HERE / 'hardware_inputs' / (name + '.pto')
        counts = []
        for hardware in ('conservative', 'a2a3-mmad-acc-v1'):
            row = run(name + '-composition-' + hardware,
                      [args.driver, source, 'demands:none',
                       args.output / (name + '-' + hardware + '.pto'), hardware],
                      source, 'mutation', True)
            counts.append(row['verdict']['barriers'])
        if (counts[1] < counts[0]) != qualified:
            raise RuntimeError('composition accumulator qualification crossed its exact block threshold')
    hardware_source = HERE / 'hardware_inputs' / 'mmad_large_11.pto'
    for mutation in ('drop-m-to-mte1', 'drop-m-to-fix'):
        run('composition-accumulator-' + mutation,
            [args.driver, hardware_source, 'demands:' + mutation,
             args.output / ('composition-accumulator-' + mutation + '.pto'), 'a2a3-mmad-acc-v1'],
            hardware_source, 'mutation', True)
    unitflag = fixtures / 'unitflag_paired.pto'
    ownership_rows = []
    for credit in ('true', 'false'):
        row = run('unitflag-paired-credit-' + credit,
                  [args.driver, unitflag, 'demands:none',
                   args.output / ('unitflag-paired-credit-' + credit + '.pto'),
                   'conservative', 'assume-disjoint-arguments',
                   'a2a3-unitflag-paired-v1', credit],
                  unitflag, 'mutation', True)
        ownership_rows.append(row)
    command_cost = lambda row: row['verdict']['barriers'] + 2 * row['verdict']['handoffs']
    if command_cost(ownership_rows[0]) >= command_cost(ownership_rows[1]):
        raise RuntimeError('UnitFlag credit did not remove its exact ACC ownership protocol')
    for mutation in ('drop-m-to-mte1', 'drop-m-to-mte2'):
        row = run('unitflag-' + mutation,
                  [args.driver, unitflag, 'demands:' + mutation,
                   args.output / ('unitflag-' + mutation + '.pto'),
                   'conservative', 'assume-disjoint-arguments',
                   'a2a3-unitflag-paired-v1', 'true'],
                  unitflag, 'mutation', True)
        if not row['verdict']['mutation_applied'] or row['verdict']['accepted']:
            raise RuntimeError('UnitFlag credit incorrectly implied operand release or GM completion')
    run('unitflag-missing-profile',
        [args.driver, unitflag, 'demands:expect-unsupported',
         args.output / 'unitflag-missing-profile.pto'], unitflag, 'mutation', True)
    mismatched_geometry = fixtures / 'unitflag_mismatched_geometry.pto'
    run('unitflag-mismatched-geometry',
        [args.driver, mismatched_geometry, 'demands:expect-unsupported',
         args.output / 'unitflag-mismatched-geometry.pto',
         'conservative', 'assume-disjoint-arguments',
         'a2a3-unitflag-paired-v1', 'true'],
        mismatched_geometry, 'mutation', True)
    for label, old, new in (
            ('bad-entry-range', 'array<i64: 0, 1024>', 'array<i64: 512, 1024>'),
            ('partial-producer', '#pto<acc_phase final>', '#pto<acc_phase partial>'),
            ('partial-store', '#pto<st_phase final>', '#pto<st_phase partial>')):
        broken = args.output / ('unitflag-' + label + '-input.pto')
        broken.write_text(unitflag.read_text().replace(old, new, 1))
        run('unitflag-' + label,
            [args.driver, broken, 'demands:expect-unsupported',
             args.output / ('unitflag-' + label + '.pto'),
             'conservative', 'assume-disjoint-arguments',
             'a2a3-unitflag-paired-v1', 'true'],
            broken, 'mutation', True)
    authored = fixtures / 'authored_open_section.pto'
    run('authored-open-section', [args.driver, authored, 'authored:none',
        args.output / 'authored-open-section.pto'], authored, 'mutation', True)
    for label, fragment in (
            ('missing-prime', '    pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]\n'),
            ('missing-cleanup', '    pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID1>]\n'),
            ('missing-drain', '      pto.barrier <PIPE_ALL>\n')):
        broken = args.output / ('authored-' + label + '.pto')
        # Only the exact outer indentation matches: inside-section endpoints
        # have six spaces and are retained by this mutation.
        text = authored.read_text()
        text = text.replace('\n' + fragment, '\n', 1)
        broken.write_text(text)
        run('authored-' + label, [args.driver, broken, 'authored:expect-unsupported',
            args.output / ('authored-' + label + '-checked.pto')], broken, 'mutation', True)
    source = fixtures / 'composition_tput_macro.pto'
    row = run('tput-macro-drop-original',
              [args.driver, source, 'composition:drop-macro',
               args.output / 'tput-macro-drop-original.pto'],
              source, 'mutation', True)
    if not row['verdict']['atomic'] or not row['verdict']['expected']:
        raise RuntimeError('opaque P2P macro deletion was not rejected atomically')
    for mutation in ('drop-macro-prerequisite', 'late-macro-prerequisite',
                     'macro-hidden-key0', 'macro-hidden-key1'):
        row = run('tput-macro-' + mutation,
                  [args.driver, source, 'composition:' + mutation,
                   args.output / ('tput-macro-' + mutation + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('opaque P2P macro prerequisite was not reconstructed')
    row = run('tput-macro-cuts-fallback',
              [args.driver, source, 'cuts:none',
               args.output / 'tput-macro-cuts-fallback.pto'],
              source, 'mutation', True)
    if not row['verdict']['atomic'] or not row['verdict']['expected']:
        raise RuntimeError('cut precision did not use the conservative atomic macro fallback')
    for mode in ('none', 'drop-macro-prerequisite', 'late-macro-prerequisite',
                 'macro-hidden-key0', 'macro-hidden-key1'):
        row = run('tput-macro-demands-' + mode,
                  [args.driver, source, 'demands:' + mode,
                   args.output / ('tput-macro-demands-' + mode + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('demand placement lost atomic macro prerequisites')
    aic_source = args.output / 'composition_tput_macro_aic.pto'
    aic_source.write_text(source.read_text().replace(
        '#pto.kernel_kind<vector>', '#pto.kernel_kind<cube>', 1))
    row = compile_case('composition-tput-macro-aic', aic_source,
                       gm='assume-disjoint-arguments', expected=False)
    if 'multi-phase-needs-endpoints:pto.comm.tput' not in row.get('first_refusal', ''):
        raise RuntimeError('P2P macro did not remain fail-closed on AIC')
    source = fixtures / 'composition_authored_remote_signal.pto'
    for mutation in ('none', 'drop-authored-notify', 'drop-authored-wait'):
        row = run('authored-remote-' + mutation,
                  [args.driver, source, 'demands:' + mutation,
                   args.output / ('authored-remote-' + mutation + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('authored remote signal was not preserved')
    row = compile_case('authored-remote-signal-alias', source, gm='may-alias', expected=False)
    if 'authored remote signal may alias local payload' not in row.get('first_refusal', ''):
        raise RuntimeError('remote signal aliasing did not remain fail-closed')
    source = fixtures / 'composition_remote_wait_scalar_visibility.pto'
    for fixture in ('composition_remote_wait_scalar_visibility',
                    'composition_remote_wait_loop_scalar_visibility'):
        source = fixtures / (fixture + '.pto')
        for mutation in ('none', 'drop-invalidate-cmo', 'drop-visibility-fence'):
            row = run(fixture + '-' + mutation,
                      [args.driver, source, 'demands:' + mutation,
                       args.output / (fixture + '-' + mutation + '.pto')],
                      source, 'mutation', True)
            if not row['verdict']['atomic'] or not row['verdict']['expected']:
                raise RuntimeError('post-wait scalar-cache visibility was not reconstructed')
    source = fixtures / 'composition_retained_helper.pto'
    for mode in ('none', 'drop-wait', 'wrong-key'):
        row = run('retained-helper-' + mode,
                  [args.driver, source, 'composition:' + mode,
                   args.output / ('helper-' + mode + '.pto')], source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('retained helper reconstruction failed')
    helper_text = source.read_text()
    for name, text, mode in (
        ('missing-helper-contract', helper_text.replace(', pto.tileop.effects = ["read"]', '', 1),
         'expect-unsupported'),
        ('invalid-helper-effect', helper_text.replace('pto.tileop.effects = ["read"]',
                                                      'pto.tileop.effects = ["unknown"]', 1),
         'expect-unsupported'),
        ('shadowed-helper', helper_text.replace(
            '  func.func @test',
            '  func.func private @read_tile() attributes {pto.tileop.effects = ["unknown"]}\n'
            '  module @nested attributes {pto.target_arch = "a3"} {\n  func.func @test', 1)
         .rstrip() + '\n}\n', 'none')):
        prepared = args.output / (name + '.pto')
        prepared.write_text(text)
        row = run(name, [args.driver, prepared, 'composition:' + mode,
                        args.output / (name + '.output.pto')], prepared, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('staged helper lookup/contract regression: ' + name)
    for name in ('section_outside_config', 'section_outside_async_descriptor',
                 'section_outside_physical', 'missing_positive_contract',
                 'composition_tmrgsort_observed'):
        compile_case(name, fixtures / (name + '.pto'), expected=False)
    row = compile_case('composition_authored_remote_notify_unreleased',
                       fixtures / 'composition_authored_remote_notify_unreleased.pto',
                       gm='assume-disjoint-arguments', expected=False)
    if 'authored remote notify has an unfinished local producer prefix' not in row.get('first_refusal', ''):
        raise RuntimeError('unreleased remote notification did not fail closed at its local release contract')
    row = compile_case('composition_authored_remote_signal_aic',
                       fixtures / 'composition_authored_remote_signal_aic.pto', expected=False)
    if 'authored remote notify requires the qualified AIV contract' not in row.get('first_refusal', ''):
        raise RuntimeError('remote notification did not remain fail-closed on AIC')
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
    source = fixtures / 'composition_scalar_visibility.pto'
    for mutation in ('none', 'drop-clean-cmo', 'drop-invalidate-cmo',
                     'drop-visibility-fence',
                     'reverse-clean-visibility', 'reverse-invalidate-visibility'):
        row = run('visibility-' + mutation,
                  [args.driver, source, 'demands:' + mutation,
                   args.output / ('visibility-' + mutation + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('GM visibility reconstruction failed')
    source = fixtures / 'composition_authored_fixed.pto'
    for mutation in ('none', 'drop-authored-barrier', 'drop-authored-cmo',
                     'drop-authored-fence',
                     'mix-generated-authored-visibility',
                     'mix-generated-authored-barrier'):
        row = run('authored-fixed-' + mutation,
                  [args.driver, source, 'demands:' + mutation,
                   args.output / ('authored-fixed-' + mutation + '.pto')],
                  source, 'mutation', True)
        if not row['verdict']['atomic'] or not row['verdict']['expected']:
            raise RuntimeError('authored fixed synchronization was not preserved')
    if args.historical:
        raw = args.historical.read_bytes()
        contract_path = args.historical.parent / 'abi-preconditions.json'
        if not contract_path.exists():
            raise RuntimeError('historical GEMM requires adjacent abi-preconditions.json')
        contract = json.loads(contract_path.read_text())
        quadruples = []
        for argument, bounds in sorted(contract['arguments'].items(), key=lambda item: int(item[0])):
            multiple = int(bounds['multiple'])
            maximum = int(bounds['max']) // multiple * multiple
            quadruples.extend((int(argument), int(bounds['min']), maximum, multiple))
        metadata = 'pto.scalar_argument_preconditions = array<i64: ' + \
            ', '.join(map(str, quadruples)) + '>'
        marker = 'pto.noalias_pairs = array<i64: 0, 1, 0, 2, 1, 2>'
        text = raw.decode()
        if text.count(marker) != 1 or 'pto.scalar_argument_preconditions' in text:
            raise RuntimeError('historical source has an unexpected ABI metadata boundary')
        qualified = args.output / 'historical-gemm.abi-qualified.pto'
        qualified.write_text(text.replace(marker, marker + ', ' + metadata, 1))
        historical_evidence = dict(
            source=str(args.historical), source_sha256=digest(raw),
            contract=str(contract_path), contract_sha256=digest(contract_path.read_bytes()),
            prepared=str(qualified), prepared_sha256=digest(qualified.read_bytes()),
            preparation='add recorded scalar arg/min/max/multiple ABI metadata; payload, addresses, and control unchanged',
            scalar_argument_preconditions=quadruples, arms={}, boundaries=[])

        # These four arms keep the hardware and persistent-lifetime effects
        # visible as separate static observations.  Only the final arm is the
        # combined acceptance target.
        for name, source, constructor, hardware in (
                ('ordinary-conservative', args.historical, 'composition', 'conservative'),
                ('ordinary-accumulator', args.historical, 'composition', 'a2a3-mmad-acc-v1'),
                ('persistent-unqualified', args.historical, 'demands', 'a2a3-mmad-acc-v1'),
                ('persistent-qualified', qualified, 'demands', 'a2a3-mmad-acc-v1')):
            output = args.output / ('historical-gemm.' + name + '.pto')
            row = run('historical-gemm-' + name,
                      [args.driver, source, constructor + ':none', output,
                       hardware, 'assume-disjoint-arguments'],
                      source, 'mutation', True)
            if not row['verdict']['accepted']:
                raise RuntimeError(name + ': historical construction was not accepted')
            emitted = output.read_text()
            historical_evidence['arms'][name] = dict(
                output=str(output), output_sha256=digest(output.read_bytes()),
                verdict=row['verdict'], sets=emitted.count('pto.set_flag['),
                waits=emitted.count('pto.wait_flag['),
                barriers=emitted.count('pto.barrier'),
                terminal_drains=emitted.count('pto.barrier <PIPE_ALL>'))
        final = historical_evidence['arms']['persistent-qualified']
        if final['verdict']['barriers'] != 0 or final['terminal_drains'] != 1:
            raise RuntimeError('qualified historical GEMM did not recover a barrier-free body plus terminal drain')

        # Re-import the actual emitted population without constructor metadata.
        # The terminal drain orders consumed events across invocations; it must
        # never compensate for a deleted event or an absent/misplaced drain.
        final_source = Path(final['output'])
        final_lines = final_source.read_text().splitlines(keepends=True)
        mutations = [(f'drop-event-{index}', final_lines[:index] + final_lines[index + 1:])
                     for index, line in enumerate(final_lines)
                     if 'pto.set_flag[' in line or 'pto.wait_flag[' in line]
        drain_index = next(index for index, line in enumerate(final_lines)
                           if 'pto.barrier <PIPE_ALL>' in line)
        no_drain = final_lines[:drain_index] + final_lines[drain_index + 1:]
        mutations.append(('drop-terminal', no_drain))
        early = next(index for index, line in enumerate(no_drain) if 'pto.section.cube {' in line) + 1
        mutations.append(('early-terminal', no_drain[:early] + [final_lines[drain_index]] + no_drain[early:]))
        late_cleanup = list(final_lines)
        cleanup_index = max(index for index in range(drain_index) if 'pto.wait_flag[' in late_cleanup[index])
        cleanup = late_cleanup.pop(cleanup_index)
        section_exit = next(index for index in range(drain_index, len(late_cleanup))
                            if late_cleanup[index].strip() == '}')
        late_cleanup.insert(section_exit + 1, cleanup)
        mutations.append(('cleanup-after-terminal', late_cleanup))
        for name, lines in [('intact', final_lines)] + mutations:
            source = args.output / ('historical-gemm.reconstructed-' + name + '.pto')
            source.write_text(''.join(lines))
            mode = 'none' if name == 'intact' else 'expect-unsupported'
            row = run('historical-gemm-reconstructed-' + name,
                      [args.driver, source, 'authored:' + mode,
                       source.with_suffix('.output.pto'), 'a2a3-mmad-acc-v1', 'assume-disjoint-arguments'],
                      source, 'mutation', True)
            if not row['verdict']['expected'] or not row['verdict']['atomic']:
                raise RuntimeError('historical actual-command reconstruction failed: ' + name)
            if name == 'cleanup-after-terminal' and 'synchronization follows' not in row['verdict']['reason']:
                raise RuntimeError('post-section cleanup did not fail at the retirement boundary')
        historical_evidence['actual_command_mutations'] = len(mutations)

        # Metadata syntax/type errors are fatal.  A valid but insufficient
        # lower bound merely declines the optional nonempty fact and retains a
        # verified conservative result.
        malformed = args.output / 'historical-gemm.bad-scalar-metadata.pto'
        malformed.write_text(qualified.read_text().replace(
            metadata, 'pto.scalar_argument_preconditions = array<i64: 3, 0, 128>', 1))
        row = run('historical-gemm-bad-scalar-metadata',
                  [args.driver, malformed, 'demands:expect-unsupported',
                   args.output / 'historical-gemm.bad-scalar-metadata.output.pto',
                   'a2a3-mmad-acc-v1', 'assume-disjoint-arguments'],
                  malformed, 'mutation', True)
        if row['verdict']['accepted']:
            raise RuntimeError('malformed scalar ABI metadata was accepted')
        insufficient = args.output / 'historical-gemm.insufficient-scalar-bound.pto'
        insufficient_values = list(quadruples)
        for index in range(1, len(insufficient_values), 4):
            insufficient_values[index] = 0
        insufficient_metadata = 'pto.scalar_argument_preconditions = array<i64: ' + \
            ', '.join(map(str, insufficient_values)) + '>'
        insufficient.write_text(qualified.read_text().replace(metadata, insufficient_metadata, 1))
        row = run('historical-gemm-insufficient-scalar-bound',
                  [args.driver, insufficient, 'demands:none',
                   args.output / 'historical-gemm.insufficient-scalar-bound.output.pto',
                   'a2a3-mmad-acc-v1', 'assume-disjoint-arguments'],
                  insufficient, 'mutation', True)
        if not row['verdict']['accepted'] or row['verdict']['barriers'] == 0:
            raise RuntimeError('insufficient scalar bound did not retain the conservative zero-trip path')

        reference = args.historical.parent / 'prototype.pto'
        if reference.exists():
            from compare_boundaries import compare, run as observe_boundaries
            manifest = json.loads((HERE / 'demand_manifest.json').read_text())
            case = next(item for item in manifest['cases'] if item['case_id'] == 'historical_gemm')
            final_path = Path(historical_evidence['arms']['persistent-qualified']['output'])
            for scenario in case['scenarios']:
                expected, expected_metrics = observe_boundaries(reference, scenario)
                actual, actual_metrics = observe_boundaries(final_path, scenario)
                differences = compare(expected, actual)
                historical_evidence['boundaries'].append(dict(
                    scenario=scenario['name'], differences=differences,
                    reference_metrics=expected_metrics, actual_metrics=actual_metrics))
                if differences:
                    raise RuntimeError('historical handoff boundaries differ in ' + scenario['name'])
        if args.python_root:
            from observations import SERIAL_DRIVER
            prefix = [sys.executable, '-c', SERIAL_DRIVER, args.python_root.resolve(),
                      '--pto-arch=a3', '--pto-level=level3']
            synchronized = args.output.resolve() / 'historical-gemm.composed.pto'
            generated = args.output.resolve() / 'historical-gemm.composed.cpp'
            commands = (
                ('historical-gemm-pto', qualified, synchronized,
                 ['--enable-insert-sync', '--insert-sync-planner=composition',
                  '--insert-sync-structured-precision=true',
                  '--insert-sync-hardware-contract=a2a3-mmad-acc-v1',
                  '--insert-sync-gm-alias=assume-disjoint-arguments', '--emit-pto-ir']),
                ('historical-gemm-cpp', synchronized, generated, []))
            for name, source, output, flags in commands:
                row = run(name, prefix + flags + [source, '-o', output],
                          source, 'frontend', True)
                row.update(artifact=str(output), artifact_sha256=digest(output.read_bytes()))
                row.update(hardware_contract='a2a3-mmad-acc-v1', gm_alias='assume-disjoint-arguments',
                           ownership_contract='none')
                if name == 'historical-gemm-pto':
                    text = output.read_text()
                    if (text.count('pto.set_flag[') != final['sets'] or
                            text.count('pto.wait_flag[') != final['waits'] or
                            text.count('pto.barrier') != final['terminal_drains']):
                        raise RuntimeError('compiler pipeline changed the qualified native GEMM event population')

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
                   corpus_constructor=args.corpus_constructor,
                   scope='native construction/reconstruction and raw/prepared compatibility; not device or whole-compilation performance',
                   revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   dirty=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                   binaries={str(p): digest(p.read_bytes()) for p in binaries},
                   manifests=corpus_manifests, historical=historical_evidence, results=rows,
                   totals=dict(Counter(r['category'] + ':' + r['status'] for r in rows)))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['totals'], indent=2))


if __name__ == '__main__':
    main()
