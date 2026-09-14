#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native regressions for origin, geometry and residual handoff coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HEADER = '''module attributes {pto.target_arch = "a3"} {
  func.func @coverage(%a: !pto.ptr<f16>, %b: !pto.ptr<f16>, %take: i1, %offset: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %huge = arith.constant 9223372036854775807 : index
    %false = arith.constant false
    %addr = arith.constant 0 : i64
    %tile = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf16>
'''
FOOTER = '    return\n  }\n}\n'
VIEW = '''    %{name}v = pto.make_tensor_view %{root}, shape = [%c32, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf16>
    %{name} = pto.partition_view %{name}v, offsets = [%{offset}, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>
'''
LOAD = '    pto.tload ins(%{name} : !pto.partition_tensor_view<16x16xf16>) outs(%tile : !pto.tile_buf<vec, 16x16xf16>)\n'
STORE = '    pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>) outs(%{name} : !pto.partition_tensor_view<16x16xf16>)\n'


def cases():
    for label, offset, expected in [('disjoint', 'c16', True), ('overlap', 'c8', False),
                                     ('unknown', 'offset', False)]:
        body = VIEW.format(name='write', root='a', offset='c0')
        body += VIEW.format(name='read', root='a', offset=offset)
        body += STORE.format(name='write') + LOAD.format(name='read')
        yield 'gm-range-' + label, HEADER + body + FOOTER, 'may-alias', expected
    body = '    %shift = pto.addptr %a, %huge : !pto.ptr<f16> -> !pto.ptr<f16>\n'
    body += VIEW.format(name='write', root='a', offset='c0')
    body += VIEW.format(name='read', root='shift', offset='c0')
    body += STORE.format(name='write') + LOAD.format(name='read')
    yield 'gm-range-overflow', HEADER + body + FOOTER, 'may-alias', False
    for label, elements, expected in [('disjoint', 256, True), ('overlap', 128, False)]:
        body = f'    %delta = arith.constant {elements} : index\n'
        body += '    %shift = pto.addptr %a, %delta : !pto.ptr<f16> -> !pto.ptr<f16>\n'
        body += VIEW.format(name='write', root='a', offset='c0')
        body += VIEW.format(name='read', root='shift', offset='c0')
        body += STORE.format(name='write') + LOAD.format(name='read')
        yield 'gm-addptr-' + label, HEADER + body + FOOTER, 'may-alias', expected
    # A third access must not merge A's distinct intervals through a may-alias
    # root. Unknown B geometry still overlaps every possible A interval.
    for label, peer_offset in [('exact', 'c0'), ('unknown', 'offset')]:
        body = VIEW.format(name='peer', root='b', offset=peer_offset) + LOAD.format(name='peer')
        body += VIEW.format(name='write', root='a', offset='c0')
        body += VIEW.format(name='read', root='a', offset='c16')
        body += STORE.format(name='write') + LOAD.format(name='read')
        yield 'gm-range-peer-' + label, HEADER + body + FOOTER, 'may-alias', True
        # Moving the B load after the A store creates a genuine possible
        # cross-root publication requirement under the same alias contract.
        reordered = body.replace(LOAD.format(name='peer'), '') + LOAD.format(name='peer')
        yield 'gm-range-peer-' + label + '-overlap', HEADER + reordered + FOOTER, 'may-alias', False
    header = HEADER.replace('%take: i1', '%c: !pto.ptr<f16>, %take: i1').replace(
        'attributes {pto.kernel_kind', 'attributes {pto.noalias_pairs = array<i64: 0, 1>, pto.kernel_kind')
    body = VIEW.format(name='peerb', root='b', offset='c0') + LOAD.format(name='peerb')
    body += VIEW.format(name='peerc', root='c', offset='offset') + LOAD.format(name='peerc')
    body += VIEW.format(name='write', root='a', offset='c0')
    body += VIEW.format(name='read', root='a', offset='c16')
    body += STORE.format(name='write') + LOAD.format(name='read')
    yield 'gm-range-partial-alias-contract', header + body + FOOTER, 'may-alias', True
    header = HEADER.replace('%take: i1', '%c: !pto.ptr<f16>, %take: i1').replace(
        'attributes {pto.kernel_kind',
        'attributes {pto.noalias_pairs = array<i64: 0, 2, 1, 2>, pto.kernel_kind')
    body = VIEW.format(name='write', root='c', offset='c0') + STORE.format(name='write')
    body += '''    %result, %finished = scf.while (%p = %a, %active = %take) : (!pto.ptr<f16>, i1) -> (!pto.ptr<f16>, i1) {
      scf.condition(%active) %p, %active : !pto.ptr<f16>, i1
    } do {
    ^bb0(%q: !pto.ptr<f16>, %again: i1):
      scf.yield %b, %false : !pto.ptr<f16>, i1
    }
'''
    body += VIEW.format(name='read', root='result', offset='c0') + LOAD.format(name='read')
    yield 'while-union-partial-contract', header + body + FOOTER, 'may-alias', True
    yield 'while-union-incomplete-contract', header.replace('0, 2, 1, 2', '0, 2') + body + FOOTER, 'may-alias', False
    for label, roots, count in [('root-budget', ('a',), 130), ('pair-budget', ('a', 'b'), 10)]:
        body = '    %extent = arith.constant 8192 : index\n'
        wide_view = VIEW.replace('shape = [%c32,', 'shape = [%extent,')
        for root in roots:
            for slot in range(count):
                name = f'{root}{slot}'
                body += f'    %{name}offset = arith.constant {32 * slot} : index\n'
                body += wide_view.format(name=name, root=root, offset=name + 'offset')
                body += LOAD.format(name=name)
        body += VIEW.format(name='write', root='a', offset='c0')
        body += VIEW.format(name='read', root='a', offset='c16')
        body += STORE.format(name='write') + LOAD.format(name='read')
        # Coarsening may lose this otherwise provable disjointness; it must
        # never silently omit the affected group's physical obligations.
        yield 'gm-range-' + label, HEADER + body + FOOTER, 'may-alias', False
    for label, condition, backedge, expected in [('forward', 'p', 'q', True),
                                                 ('backedge-union', 'p', 'b', False),
                                                 ('zero-trip-condition', 'b', 'q', False)]:
        body = VIEW.format(name='write', root='b', offset='c0') + STORE.format(name='write')
        body += '''    %result, %finished = scf.while (%p = %a, %active = %take) : (!pto.ptr<f16>, i1) -> (!pto.ptr<f16>, i1) {
      scf.condition(%active) %COND, %active : !pto.ptr<f16>, i1
    } do {
    ^bb0(%q: !pto.ptr<f16>, %again: i1):
      scf.yield %BACK, %false : !pto.ptr<f16>, i1
    }
'''.replace('COND', condition).replace('BACK', backedge)
        body += VIEW.format(name='read', root='result', offset='c0') + LOAD.format(name='read')
        yield 'while-' + label, HEADER + body + FOOTER, 'assume-disjoint-arguments', expected
    # Before and after signatures differ; the exit is condition.args[0], B.
    body = VIEW.format(name='write', root='b', offset='c0') + STORE.format(name='write')
    body += '''    %result = scf.while (%p = %a, %n = %c0) : (!pto.ptr<f16>, index) -> !pto.ptr<f16> {
      %first = arith.cmpi ult, %n, %c1 : index
      %continue = arith.andi %take, %first : i1
      scf.condition(%continue) %b : !pto.ptr<f16>
    } do {
    ^bb0(%q: !pto.ptr<f16>):
      scf.yield %q, %c1 : !pto.ptr<f16>, index
    }
'''
    body += VIEW.format(name='read', root='result', offset='c0') + LOAD.format(name='read')
    yield 'while-different-signatures', HEADER + body + FOOTER, 'assume-disjoint-arguments', False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', type=Path, required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output', type=Path)
    output.add_argument('--output-root', type=Path, help='create a fresh report directory on every test run')
    args = parser.parse_args()
    if args.output_root:
        args.output = args.output_root / ('coverage-' + str(time.time_ns()))
    args.output.mkdir(parents=True, exist_ok=False)
    binary_sha256 = hashlib.sha256(args.opt.read_bytes()).hexdigest()
    env = dict(os.environ, PTOAS_COMPOSITION_WITNESSES='1', PTOAS_LOGICAL_TRACE='1')
    rows = []
    for name, text, alias, expected in cases():
        source = args.output / (name + '.pto')
        source.write_text(text)
        command = [str(args.opt.resolve()), '--mlir-disable-threading',
                   '--pto-insert-sync=planner=composition structured-precision=true '
                   'logical-work-budget=0 gm-alias=' + alias, str(source.resolve())]
        start = time.monotonic()
        p = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
        (args.output / (name + '.stdout')).write_text(p.stdout)
        (args.output / (name + '.stderr')).write_text(p.stderr)
        reports = [json.loads(line.removeprefix('OAHS_WITNESS ')) for line in p.stderr.splitlines()
                   if line.startswith('OAHS_WITNESS ')]
        passed = (p.returncode == 0 if expected else p.returncode > 0) and bool(reports)
        if not expected:
            passed &= 'MTE3-to-MTE2 GM publication' in p.stderr and not p.stdout.strip()
        rows.append(dict(name=name, command=command, expected=expected, passed=passed,
                         gm_alias=alias, hardware_contract='conservative', ownership_contract='none',
                         returncode=p.returncode, seconds=time.monotonic()-start,
                         input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         output_sha256=hashlib.sha256(p.stdout.encode()).hexdigest(), witnesses=reports))
        print(name, 'PASS' if passed else 'FAIL', flush=True)
    if hashlib.sha256(args.opt.read_bytes()).hexdigest() != binary_sha256:
        raise RuntimeError('native binary changed during regression run')
    report = dict(schema='oahs.coverage.regressions.v1', measurement='fresh',
                  binary_sha256=binary_sha256,
                  source_revision=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[4], text=True).strip(),
                  source_diff_sha256=hashlib.sha256(subprocess.check_output(
                      ['git', 'diff', 'HEAD'], cwd=Path(__file__).resolve().parents[4])).hexdigest(),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, passed=all(row['passed'] for row in rows), device='NOT_RUN')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
