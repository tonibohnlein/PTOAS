# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Equivalent executed accumulation words must have identical command order."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def main():
    tool, fixture = sys.argv[1:]
    prefix = Path(fixture).read_text().split('    scf.for')[0]
    prefix = prefix.replace('@accumulation()', '@accumulation(%flag: i1)')
    prefix = prefix.replace(') {', ') attributes {test.trace_arguments = array<i64: 1>} {')
    init = 'pto.tmatmul ins(%a, %b : !a, !b) outs(%c : !c)\n'
    acc = 'pto.tmatmul.acc ins(%c, %a, %b : !c, !a, !b) outs(%c : !c)\n'

    def loop(iv, body, bound='%two'):
        return f'scf.for %{iv} = %zero to {bound} step %one {{\n' + body + '}\n'

    def choose(condition, yes, no=''):
        return f'scf.if %{condition} {{\n' + yes + '} else {\n' + no + '}\n'

    first = '%first = arith.cmpi eq, %i, %zero : index\n'
    nested_first = (first + '%second = arith.cmpi eq, %j, %zero : index\n'
                    '%both = arith.andi %first, %second : i1\n')
    cases = [
        (init + acc * 3, [True, False, False, False]),
        (choose('flag', init + acc, init + acc) + acc * 2, [True, False, False, False]),
        (loop('i', first + choose('first', init, acc)) + loop('j', acc), [True, False, False, False]),
        (loop('i', loop('j', nested_first + choose('both', init, acc))), [True, False, False, False]),
        (init + loop('i', choose('flag', acc, acc)) + acc, [True, False, False, False]),
        # Repeated initialization is a real boundary even within uniform loops.
        (loop('i', init + acc), [True, False, True, False]),
        (loop('i', loop('j', init + acc, '%one')), [True, False, True, False]),
        # An existing accumulator can enter the function; finding its original
        # initializer is not necessary to protect subsequent accumulations.
        (acc * 4, [False] * 4),
        (loop('i', loop('j', acc)), [False] * 4),
        (init + loop('i', acc, '%zero') + acc, [True, False]),
        (loop('i', loop('j', init + acc), '%zero'), []),
    ]
    # Nonuniform outer visits contain a real cross-pipe accumulator reader.
    # Inner uniform loops must still protect accumulation without losing the
    # store readiness or the release before the next visit's initialization.
    store_setup = ('%m = arith.constant 32 : index\n%width = arith.constant 80 : index\n'
                   '%view = pto.make_tensor_view %out, shape = [%m, %width], strides = [%width, %one] '
                   '{layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>\n'
                   '%part = pto.partition_view %view, offsets = [%zero, %zero], sizes = [%m, %width] '
                   ': !pto.tensor_view<?x?xf32>\n')
    store = 'pto.tstore ins(%c : !c) outs(%part : !pto.partition_tensor_view<32x80xf32>)\n'
    stored_cases = [
        (loop('tile', init + acc + store), [True, False, False] * 2),
        (loop('tile', loop('i', first + choose('first', init, acc)) + store), [True, False, False] * 2),
    ]
    checks = 0
    with tempfile.TemporaryDirectory(prefix='structured-protection-') as directory:
        path = Path(directory) / 'case.pto'
        for flag in (0, 1):
            for body, resets in cases + stored_cases:
                head = prefix.replace('array<i64: 1>', f'array<i64: {flag}>')
                if (body, resets) in stored_cases:
                    head = head.replace('@accumulation(%flag:', '@accumulation(%out: !pto.ptr<f32, gm>, %flag:')
                    head = head.replace(f'array<i64: {flag}>', f'array<i64: 0, {flag}>') + store_setup
                path.write_text(head + body + 'return\n}\n}\n')
                run = subprocess.run([tool, '--structured-trace', str(path)], capture_output=True,
                                     text=True, timeout=60, check=False)
                assert run.returncode == 0, run.stderr
                document = json.loads(run.stdout)
                assert document['accepted'], (body, document)
                trace = document['trace']
                assert not trace['error'], trace
                payloads = [e for e in trace['events'] if e['kind'] == 'payload']
                assert len(payloads) == len(resets), (body, trace)
                commands = []
                for event in trace['events']:
                    if event['kind'] == 'payload':
                        continue
                    command = dict(event)
                    if event['kind'] != 'barrier':
                        command['identity'] = (event['plan'], event['record'], event['source_ordinal'],
                                               tuple(event.get('members', [])))
                    commands.append(command)
                pipes = [e['pipe'] for e in payloads]
                edges = native(pipes)
                edges.update((2*a+1, 2*b) for b, reset in enumerate(resets) for a in range(b)
                             if reset or pipes[a] != 2 or pipes[b] != 2)
                required, _ = closure(2*len(pipes), edges)
                actual = closure_with_commands(pipes, commands)
                assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (body, trace)
                checks += 1
    print(f'structured protection: {checks} equivalent-word/reset/empty execution checks passed')


if __name__ == '__main__':
    main()
