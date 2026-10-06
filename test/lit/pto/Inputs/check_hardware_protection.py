# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Target eligibility and independent residual-conflict closure checks."""
import json
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
from check_periodic_demands import check_scan, run
from check_logical_insertion import invoke, recognized, validate


def access(atom, read=False, write=False, group=0):
    return dict(atom=atom, read=read, write=write, protection_group=group)


def scan_checks(tool):
    def op(ident, *effects, pipe=2):
        return dict(id=ident, pipe=pipe, accesses=list(effects))
    cases = [
        # Removing the reduced A->B edge would also lose U->B on atom zero.
        dict(scan=[op(0, access(0, write=True)), op(1, access(1, write=True, group=1)),
                   op(2, access(0, read=True), access(1, read=True, write=True, group=1))]),
        # One pair has both protected and unprotected storage reasons.
        dict(scan=[op(0, access(0, write=True, group=1), access(1, write=True)),
                   op(1, access(0, read=True, write=True, group=1), access(1, read=True))]),
        dict(scan=[op(0, access(0, write=True, group=1)),
                   op(1, access(0, read=True, write=True, group=1))], prerequisites=[[0, 1, 0]]),
        # An outside reader survives entry into a protected writer chain.
        dict(scan=[op(0, access(0, read=True), pipe=4), op(1, access(0, write=True, group=1)),
                   op(2, access(0, read=True, write=True, group=1)), op(3, access(0, read=True), pipe=10)]),
    ]
    rng = random.Random(12349)
    for _ in range(300):
        values = []
        for ident in range(rng.randrange(1, 20)):
            pipe = rng.randrange(4)
            effects = []
            for atom in range(4):
                mode = rng.randrange(4)
                if mode:
                    group = 1 + 3 * pipe + rng.randrange(3) if mode & 2 and rng.randrange(2) else 0
                    effects.append(access(atom, bool(mode & 1), bool(mode & 2), group))
            values.append(op(ident, *effects, pipe=pipe))
        cases.append(dict(scan=values))
    results = run(tool, cases)
    for case, result in zip(cases, results):
        check_scan(case, result)
    assert results[0]['generators'] == [[0, 2]]
    assert results[1]['generators'] == results[2]['generators'] == [[0, 1]]
    invalid = [dict(scan=[op(0, access(0, read=True, group=1))]),
               dict(scan=[op(0, access(0, write=True, group=1)), op(1, access(0, write=True, group=1), pipe=3)])]
    for result in run(tool, invalid):
        assert result['error'] and not result['generators']
    return len(cases)


def ir_checks(tool, source):
    original = source.read_text()
    with tempfile.TemporaryDirectory(prefix='pto-protection-') as scratch:
        path = Path(scratch) / 'case.pto'
        for width, expected in [(64, False), (80, True), (96, True)]:
            path.write_text(original.replace('80', str(width)))
            template = recognized(tool, path)
            groups = [dict(p['hardware_protection']) for p in template['payloads']]
            assert len(groups) == 4
            if expected:
                assert all(groups) and groups[0] == groups[1] and groups[2] == groups[3]
                assert groups[0] != groups[2], 'initializer must end the previous group'
            else:
                assert not any(groups), 'below-threshold shapes must retain barriers'
            trace = json.loads(invoke(tool, '--insertion-trace', path))
            validate(template, trace)
            barriers = sum(e['kind'] == 'barrier' and e['pipe'] != 6 for e in trace['events'])
            assert barriers == (3 if expected else 7), (width, barriers)
        # A5 geometry uses a different left-tile layout; A2/A3 protection does not apply.
        text = original.replace('"a3"', '"a5"').replace('left, 32x16xf16, slayout=',
                                                           'left, 32x16xf16, blayout=col_major, slayout=')
        path.write_text(text)
        template = recognized(tool, path)
        assert not any(p['hardware_protection'] for p in template['payloads'])
        validate(template, json.loads(invoke(tool, '--insertion-trace', path)))
        # A store reading the accumulator interrupts the protected chain, even
        # though the following MMAD has the same accumulator and dimensions.
        text = original.replace('@accumulation()', '@accumulation(%out: !pto.ptr<f32, gm>)')
        text = text.replace('    scf.for', '''    %m = arith.constant 32 : index
    %n = arith.constant 80 : index
    %rows = arith.constant 64 : index
    %view = pto.make_tensor_view %out, shape = [%rows, %n], strides = [%n, %one]
      {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf32>
    scf.for''')
        text = text.replace('scf.for %t = %zero to %two step %one {', '''scf.for %t = %zero to %two step %one {
      %row = arith.muli %t, %m : index
      %part = pto.partition_view %view, offsets = [%row, %zero], sizes = [%m, %n] : !pto.tensor_view<?x?xf32>''')
        acc = '      pto.tmatmul.acc ins(%c, %a, %b : !c, !a, !b) outs(%c : !c)'
        store = '      pto.tstore ins(%c : !c) outs(%part : !pto.partition_tensor_view<32x80xf32>)'
        path.write_text(text.replace(acc, store + '\n' + acc, 1))
        template = recognized(tool, path)
        groups = [dict(p['hardware_protection']) for p in template['payloads']]
        assert groups[0] and not groups[1] and not groups[2]
        assert groups[3] and groups[3] == groups[4] and groups[0] != groups[3]
        validate(template, json.loads(invoke(tool, '--insertion-trace', path)))
        # Symbolic accumulator addresses retain the same native protection.
        head = original[:original.index("    scf.for")]
        head = head.replace("@accumulation()", "@accumulation(%address: i64, %g: i1)")
        head = head.replace("%c = pto.alloc_tile addr = %base", "%c = pto.alloc_tile addr = %address")
        pair = ("pto.tmatmul ins(%a, %b : !a, !b) outs(%c : !c)\n"
                "pto.tmatmul.acc ins(%c, %a, %b : !c, !a, !b) outs(%c : !c)\n")
        path.write_text(head + pair + "return\n}\n}\n")
        explicit = next(json.loads(line) for line in invoke(tool, "--explicit-analysis", path).splitlines()
                        if line.startswith("{"))
        assert not explicit["error"] and explicit["retained"] == [], explicit
        for guard in (0, 1):
            guarded_head = head.replace("%g: i1) {", "%g: i1) attributes {test.trace_arguments = "
                                        f"array<i64: 0, {guard}>" + "} {")
            path.write_text(guarded_head + "scf.if %g {\n" + pair + "}\nreturn\n}\n}\n")
            report = json.loads(invoke(tool, "--structured-trace", path))
            assert report["accepted"] and not report["trace"]["error"], report
            events = report["trace"]["events"]
            assert sum(e["kind"] == "payload" for e in events) == 2 * guard
            assert all(e["kind"] == "payload" or (e["kind"] == "barrier" and e["pipe"] == 6)
                       for e in events), events
        # Constant valid operands recover the effective MMAD dimensions even
        # when the descriptor type marks those dimensions dynamic.
        text = original.replace('32x16xf16, slayout=', '32x16xf16, valid=?x16, slayout=')
        text = text.replace('%a = pto.alloc_tile addr = %base : !a',
                            '%m = arith.constant 32 : index\n'
                            '    %a = pto.alloc_tile addr = %base valid_row = %m : !a')
        path.write_text(text)
        template = recognized(tool, path)
        groups = [dict(p['hardware_protection']) for p in template['payloads']]
        assert all(groups) and groups[0] == groups[1] and groups[2] == groups[3]
        validate(template, json.loads(invoke(tool, '--insertion-trace', path)))
        # Check all three operands, not only a dynamic lhs dimension. Capacity
        # must not supply the threshold when current valid dimensions are small.
        dynamic = re.sub(r'(tile_buf<\w+, \d+x\d+xf\d+)', r'\1, valid=?x?', original)
        dynamic = dynamic.replace('    %a =', '    %m = arith.constant 32 : index\n'
                                             '    %k = arith.constant 16 : index\n'
                                             '    %n = arith.constant 80 : index\n    %a =')
        for tile, row, col in [('a', 'm', 'k'), ('b', 'k', 'n'), ('c', 'm', 'n')]:
            dynamic = dynamic.replace(f'%{tile} = pto.alloc_tile addr = %base :',
                                      f'%{tile} = pto.alloc_tile addr = %base valid_row = %{row} valid_col = %{col} :')
        for size, expected in [(80, True), (64, False)]:
            path.write_text(dynamic.replace('constant 80 : index', f'constant {size} : index'))
            template = recognized(tool, path)
            assert all(bool(p['hardware_protection']) == expected for p in template['payloads'])
            validate(template, json.loads(invoke(tool, '--insertion-trace', path)))
        # Shared scalar normalization folds constant dimension expressions.
        folded = dynamic.replace('%n = arith.constant 80 : index',
                                 '%n = arith.addi %m, %k : index\n'
                                 '    %nn = arith.addi %n, %m : index')
        path.write_text(folded.replace('valid_col = %n :', 'valid_col = %nn :'))
        template = recognized(tool, path)
        assert all(p['hardware_protection'] for p in template['payloads'])
        # A reaching metadata update, including one before each loop visit,
        # takes precedence over allocation operands. Unknown metadata from
        # another block must not be replaced by those initial dimensions.
        loop = '    scf.for %t = %zero to %two step %one {'
        for size in (80, 64):
            update = (f'\n      %width = arith.constant {size} : index\n'
                      '      pto.set_validshape %b, %k, %width : !b\n'
                      '      pto.set_validshape %c, %m, %width : !c')
            updated = dynamic.replace(loop, loop + update)
            # Metadata operations currently prevent template recognition; check
            # the shared dimensions directly without relying on that backend.
            updated = updated.replace('outs(%c : !c)', 'outs(%c : !c) '
                                      '{test.tile_valid_shapes = array<i64: 1, 16, ' + str(size) + '>}')
            # acc's operand 1 is lhs, unlike the initializer's rhs operand 1.
            updated = re.sub(r'(pto.tmatmul.acc[^\n]*test.tile_valid_shapes = array<i64:) 1, 16, \d+',
                             r'\1 2, 16, ' + str(size), updated)
            path.write_text(updated)
            invoke(tool, '--region-contract-checks', path)
        # A late update to a descriptor allocated outside the loop reaches the
        # next visit. Recovering the initial allocation size would be unsound.
        late = dynamic.replace('    }\n    return',
                               '      pto.set_validshape %b, %k, %m : !b\n'
                               '    }\n    return')
        late = late.replace('outs(%c : !c)', 'outs(%c : !c) '
                            '{test.tile_valid_shapes = array<i64: 1, -1, -1>}')
        late = re.sub(r'(pto.tmatmul.acc[^\n]*test.tile_valid_shapes = array<i64:) 1,', r'\1 2,', late)
        path.write_text(late)
        invoke(tool, '--region-contract-checks', path)
        # Unknown row extents use the shared buffer bound without inferring hardware protection.
        unknown = text.replace('@accumulation()', '@accumulation(%m: index)')
        unknown = unknown.replace('    %m = arith.constant 32 : index\n', '')
        path.write_text(unknown)
        docs = [json.loads(line) for line in invoke(tool, '--recognize', path).splitlines()
                if line.startswith('{')]
        attempts = [a for d in docs for n in d['nodes'] for a in n['attempts']
                    if a['route'] == 'numeric-template']
        assert attempts and any(a['state'] == 'applicable' for a in attempts)
        template = recognized(tool, path)
        assert not any(p['hardware_protection'] for p in template['payloads'])
        validate(template, json.loads(invoke(tool, '--insertion-trace', path)))


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    count = scan_checks(tool)
    ir_checks(tool, Path(sys.argv[2]))
    print(f'hardware protection: {count} independent scan checks, target/shape/reset/interference checks passed')


if __name__ == '__main__':
    main()
