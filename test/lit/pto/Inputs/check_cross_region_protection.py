# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check fixed accumulator protection across sequences and structured control."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    prefix = Path(fixture).read_text().split('    scf.for')[0]
    prefix = prefix.replace('@accumulation()', '@accumulation(%n: index, %flag: i1)')
    prefix = prefix.replace(') {', ') attributes {test.trace_arguments = array<i64: 3, 1>} {')
    init = '    pto.tmatmul ins(%a, %b : !a, !b) outs(%c : !c)'
    acc = '    pto.tmatmul.acc ins(%c, %a, %b : !c, !a, !b) outs(%c : !c)'

    def tagged(op, group):
        return op + ' {test.protection_group = ' + str(group) + ' : i64}\n'

    def loop(body, bound='%n'):
        return '    scf.for %t = %zero to ' + bound + ' step %one {\n' + body + '    }\n'

    def program(body):
        return prefix + body + '    return\n  }\n}\n'

    positive = program(tagged(init, 1) + loop(tagged(acc, 1)) + tagged(acc, 1))
    cases = [positive,
        program(loop(tagged(acc, 1)) + tagged(acc, 1)),
        program(tagged(init, 1) + loop(tagged(init, 1) + tagged(acc, 1)) + tagged(acc, 1)),
        program(tagged(init, 1) + loop(loop(tagged(acc, 1), '%two').replace('%t =', '%u =')) + tagged(acc, 1)),
        program(tagged(init, 1) + '    scf.if %flag {\n' + tagged(acc, 1) +
                '    } else {\n' + tagged(acc, 1) + '    }\n' + tagged(acc, 1)),
        program(tagged(init, 1) + '    scf.if %flag {\n' + tagged(acc, 1) + '    }\n' + tagged(acc, 1)),
        # Initializers keep incoming demands but publish the same protected output.
        program(tagged(init, 1) + '    scf.if %flag {\n' + tagged(acc, 1) +
                '    } else {\n' + tagged(init, 1) + '    }\n' + tagged(acc, 1)),
    ]
    alias = '    %alias = pto.alloc_tile addr = %base : !c\n'
    cases.append(program(tagged(init, 1) + alias + loop(tagged(acc.replace('%c', '%alias'), 1)) + tagged(acc, 1)))
    other = '    %other = arith.constant 32768 : i64\n' + alias.replace('%base', '%other')
    cases.append(program(tagged(init, 1) + other + loop(tagged(acc.replace('%c', '%alias'), 2)) + tagged(acc, 3)))
    cases.append(positive.replace('80', '64').replace('test.protection_group = 1', 'test.protection_group = 0'))
    cases.append(positive.replace('%n: index,', '%base: i64, %n: index,').replace(
        '    %base = arith.constant 0 : i64\n', ''))
    cases.append(positive.replace('"a3"', '"a5"').replace('left, 32x16xf16, slayout=',
                 'left, 32x16xf16, blayout=col_major, slayout=').replace(
                 'test.protection_group = 1', 'test.protection_group = 0'))
    # An external reader breaks protection. A later initializer starts a
    # distinct group.
    output = ('    %m = arith.constant 32 : index\n    %width = arith.constant 80 : index\n'
              '    %view = pto.make_tensor_view %out, shape = [%m, %width], strides = [%width, %one] '
              '{layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>\n'
              '    %part = pto.partition_view %view, offsets = [%zero, %zero], sizes = [%m, %width] '
              ': !pto.tensor_view<?x?xf32>\n')
    store = '    pto.tstore ins(%c : !c) outs(%part : !pto.partition_tensor_view<32x80xf32>)\n'
    for interference in (store, loop(store), '    scf.if %flag {\n' + store + '    }\n'):
        value = program(output + tagged(init, 1) + interference + tagged(acc, 2) +
                        tagged(init, 2) + loop(tagged(acc, 2)))
        cases.append(value.replace('%n: index,', '%out: !pto.ptr<f32, gm>, %n: index,'))
    # Descriptor rebinding invalidates fixed geometry even when a later
    # initializer uses the original SSA value. Never export invocation facts.
    rebind = ('    %new_address = arith.constant 32768 : i64\n'
              '    %rebound = pto.tassign %c, %new_address : !c -> !c\n')
    for mutation in (rebind, loop(tagged(acc, 0) + rebind),
                     '    scf.if %flag {\n' + tagged(acc, 0) + rebind + '    }\n'):
        cases.append(program(tagged(init, 0) + mutation + tagged(init, 0) + loop(tagged(acc, 0))))
    cases.append(program(rebind + tagged(init, 0) + loop(tagged(acc, 0))))
    with tempfile.TemporaryDirectory(prefix='cross-region-protection-') as directory:
        path = Path(directory) / 'case.pto'
        for number, source in enumerate(cases):
            path.write_text(source)
            try:
                invoke(tool, '--region-contract-checks', path)
            except AssertionError as error:
                raise AssertionError((number, source, str(error))) from error
        # Check actual insertion on both sides of an accumulation-only loop,
        # including the zero-trip path connecting prologue to epilogue.
        traces = [(source, trips, 1, width * trips + 2)
                  for source, width in ((positive, 1), (cases[3], 2)) for trips in (0, 1, 3)]
        traces += [(cases[4], 3, flag, 3) for flag in (0, 1)]
        traces += [(cases[5], 3, flag, flag + 2) for flag in (0, 1)]
        # A symbolic, invocation-invariant ACC base must use the same shared
        # protection at residual-access crossings as at materialized cells.
        symbolic = cases[4].replace('@accumulation(%n:', '@accumulation(%base: i64, %n:').replace(
            '    %base = arith.constant 0 : i64\n', '')
        symbolic = symbolic.replace('array<i64: 3, 1>', 'array<i64: 0, 3, 1>')
        traces.append((symbolic, 3, 1, 3))
        for source, trips, flag, count in traces:
            path.write_text(source.replace('array<i64: 3, 1>', f'array<i64: {trips}, {flag}>'))
            document = json.loads(invoke(tool, '--structured-trace', path))
            assert document['accepted'], document
            trace = document['trace']
            assert not trace['error'], trace
            payloads = [event for event in trace['events'] if event['kind'] == 'payload']
            assert len(payloads) == count, payloads
            assert all(event['kind'] == 'payload' or
                       (event['kind'] == 'barrier' and event['pipe'] == 6) for event in trace['events']), trace
    print(f'cross-region protection: {len(cases)} contracts and {len(traces)} composed insertion traces passed')


if __name__ == '__main__':
    main()
