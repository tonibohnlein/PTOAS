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
    for label, condition, backedge, expected in [('forward', 'p', 'q', True),
                                                 ('backedge-union', 'p', 'b', False),
                                                 ('zero-trip-condition', 'b', 'q', False)]:
        body = VIEW.format(name='write', root='b', offset='c0') + STORE.format(name='write')
        body += '''    %result = scf.while (%p = %a) : (!pto.ptr<f16>) -> !pto.ptr<f16> {
      scf.condition(%take) %COND : !pto.ptr<f16>
    } do {
    ^bb0(%q: !pto.ptr<f16>):
      scf.yield %BACK : !pto.ptr<f16>
    }
'''.replace('COND', condition).replace('BACK', backedge)
        body += VIEW.format(name='read', root='result', offset='c0') + LOAD.format(name='read')
        yield 'while-' + label, HEADER + body + FOOTER, 'assume-disjoint-arguments', expected
    # Before and after signatures differ; the exit is condition.args[0], B.
    body = VIEW.format(name='write', root='b', offset='c0') + STORE.format(name='write')
    body += '''    %result = scf.while (%p = %a, %n = %c0) : (!pto.ptr<f16>, index) -> !pto.ptr<f16> {
      scf.condition(%take) %b : !pto.ptr<f16>
    } do {
    ^bb0(%q: !pto.ptr<f16>):
      scf.yield %q, %c0 : !pto.ptr<f16>, index
    }
'''
    body += VIEW.format(name='read', root='result', offset='c0') + LOAD.format(name='read')
    yield 'while-different-signatures', HEADER + body + FOOTER, 'assume-disjoint-arguments', False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
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
        passed = (p.returncode == 0) == expected and bool(reports)
        if not expected:
            passed &= 'MTE3-to-MTE2 GM publication' in p.stderr
        rows.append(dict(name=name, command=command, expected=expected, passed=passed,
                         returncode=p.returncode, seconds=time.monotonic()-start,
                         input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                         output_sha256=hashlib.sha256(p.stdout.encode()).hexdigest(), witnesses=reports))
        print(name, 'PASS' if passed else 'FAIL', flush=True)
    report = dict(schema='oahs.coverage.regressions.v1', measurement='fresh',
                  binary_sha256=hashlib.sha256(args.opt.read_bytes()).hexdigest(),
                  rows=rows, passed=all(row['passed'] for row in rows), device='NOT_RUN')
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
