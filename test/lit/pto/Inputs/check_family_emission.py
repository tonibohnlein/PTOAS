# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check family emission against independent ordering and cyclic phase oracles."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from check_logical_insertion import invoke, recognized
from check_physical_allocation import check_physical, opt



def synthetic_plan(positions, records, barrier=False):
    """Exercise old saved-IR compatibility; this is not a storage-analysis certificate."""
    lines = ["module {", "func.func @emission() attributes {pto.cyclic_allocation = {",
             "version = 2 : i64, plan = 0 : i64, directions = [{source = 4 : i64,",
             "target = 3 : i64, budget = 2 : i64, records = array<i64: " +
             ", ".join(map(str, records)) + ">}]}} {",
             "%zero = arith.constant 0 : index", "%one = arith.constant 1 : index",
             "%end = arith.constant 5 : index", "scf.for %k = %zero to %end step %one {"]
    for record, position in enumerate(positions):
        lines += [f"%c{record} = arith.constant {position} : index",
                  f"%g{record} = arith.cmpi eq, %k, %c{record} : index"]
    for kind, cut in [("set", 0), ("wait", 1)]:
        for record in range(len(positions)):
            lines += [f"scf.if %g{record} {{",
                      f"pto.logical_{kind} [<PIPE_MTE2>, <PIPE_MTE1>] "
                      f"plan 0 record {record} ordinal %zero",
                      f"}} {{pto.endpoint_cut = {cut} : i64}}"]
            if barrier and kind == "set" and record == 0:
                lines.append("pto.barrier <PIPE_MTE2>")
    return "\n".join(lines + ["}", "return", "}", "}"])


def compatibility_checks(optimizer, path):
    """Saved singleton-record IR adapts each existing record to a singleton family."""
    passes = ["--pto-frontier-allocate=eligible-ids=1,3"]
    for positions, records, barrier in [([0, 1, 2], [0, 2, 1], False),
                                       ([0, 1, 2], [0, 1, 2], True),
                                       ([0, 1, 3, 4], [0, 1, 3, 2], False)]:
        path.write_text(synthetic_plan(positions, records, barrier))
        result = opt(optimizer, path, passes).stdout
        assert result.count("pto.set_flag") == len(records)
        assert result.count("pto.wait_flag") == len(records)
        assert result.count("pto.barrier") == int(barrier)
        assert "pto.endpoint_cut" not in result


def check_phase_mapping(template, report, eligible):
    """Use analysis records, independently of emitted guards and ID arithmetic."""
    owners = {}
    for family in template['logical_endpoints']['families']:
        for member, record in enumerate(family['members']):
            owners[family['id'], member] = record['record']
    # The certificate's order is the periodic allocation handoff order.
    phases = {}
    base = 0
    for direction in template['allocation']['directions']:
        for phase, handoff in enumerate(direction['handoffs']):
            phases[handoff['record']] = (phase, len(direction['handoffs']), direction['uniform_budget'], base)
        base += direction['uniform_budget']
    for logical, physical in zip(report['logical']['events'], report['physical']['events']):
        if logical['kind'] not in ('set', 'wait'):
            continue
        member = logical.get('members', [0])[0]
        record = member if logical.get('record_label') else owners[logical['record'], member]
        phase, stride, budget, base = phases[record]
        expected = eligible[base + (logical['source_ordinal'] * stride + phase) % budget]
        assert physical['physical_id'] == expected, (logical, physical, expected)


def main():
    tool, optimizer = shutil.which(sys.argv[1]), shutil.which(sys.argv[2])
    source = Path(sys.argv[3]).read_text().replace("array<i64: 1, 3>", "array<i64: 0, 1, 3, 5>")
    checked = 0
    with tempfile.TemporaryDirectory(prefix="family-emission-") as scratch:
        path = Path(scratch) / "case.pto"
        for inner_end in [1, 2, 5, 8, 9, 17]:
            for outer_end in [1, 2, 8]:
                text = source.replace("constant 9 : index", f"constant {inner_end} : index")
                text = text.replace("array<i64: 8>", f"array<i64: {outer_end}>")
                path.write_text(text)
                compact = json.loads(invoke(tool, "--physical-trace", path))
                template = recognized(tool, path)
                check_physical(template, compact, {0, 1, 3, 5})
                check_phase_mapping(template, compact, [0, 1, 3, 5])
                checked += 1
        # The same static payload cut is visited at 1, 3 and 7, leaving a hole at 5.
        # Preserve selected-arm execution as well as the sparse coordinates.
        hole = source.replace("%inner_end = arith.constant 9 : index",
                              "%inner_end = arith.constant 9 : index\n    %five = arith.constant 5 : index")
        first = "        pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile0 : !tile)"
        last = "        pto.textract ins(%tile1, %zero, %zero : !tile, index, index) outs(%left1 : !left)"
        hole = hole.replace(first, "        %keep = arith.cmpi ne, %k, %five : index\n        scf.if %keep {\n" + first)
        hole = hole.replace(last, last + "\n        }")
        path.write_text(hole)
        compact = json.loads(invoke(tool, "--physical-trace", path))
        template = recognized(tool, path)
        check_physical(template, compact, {0, 1, 3, 5})
        check_phase_mapping(template, compact, [0, 1, 3, 5])
        checked += 1
        # Typed, nonzero coordinates must be normalized without narrowing.
        typed = source.replace("%inner_end = arith.constant 9 : index", """%inner_end = arith.constant 10 : i32
    %inner_low = arith.constant 1 : i32
    %inner_step = arith.constant 2 : i32""")
        typed = typed.replace("%k = %one to %inner_end step %two {",
                              "%k = %inner_low to %inner_end step %inner_step : i32 {")
        path.write_text(typed)
        compact = json.loads(invoke(tool, "--physical-trace", path))
        template = recognized(tool, path)
        check_physical(template, compact, {0, 1, 3, 5})
        check_phase_mapping(template, compact, [0, 1, 3, 5])
        checked += 1
        # Numerical nested bounds may depend on an enclosing induction value.
        # Include a context-dependent positive step.
        for low, end, step in [('0', '3', '%one'), ('1', '3', '%i')]:
            nested = source.replace('%inner_end = arith.constant 9 : index',
                f'%inner_end = arith.constant 4 : index\n'
                f'    %nest_low = arith.constant {low} : index\n    %nest_end = arith.constant {end} : index')
            nested = nested.replace('scf.for %k = %one to %inner_end step %two {',
                f'scf.for %i = %nest_low to %nest_end step %one {{\n      scf.for %k = %i to %inner_end step {step} {{')
            nested = nested.replace('      }\n    }\n    return', '      }\n      }\n    }\n    return')
            path.write_text(nested)
            template = recognized(tool, path)
            report = json.loads(invoke(tool, "--physical-trace", path))
            check_physical(template, report, {0, 1, 3, 5})
            check_phase_mapping(template, report, [0, 1, 3, 5])
            checked += 1
        # One source cut reaches two distinct consumer cuts. Its SET emission
        # must be shared independently of those WAIT locations.
        asymmetric = source.replace(
            '        pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile1 : !tile)\n', '')
        asymmetric = asymmetric.replace(
            '        pto.textract ins(%tile1, %zero, %zero : !tile, index, index) outs(%left1 : !left)\n', '')
        consumer = '        pto.textract ins(%tile0, %zero, %zero : !tile, index, index) outs(%left0 : !left)'
        asymmetric = asymmetric.replace(consumer,
            '        %first = arith.cmpi eq, %k, %one : index\n        scf.if %first {\n' +
            consumer + '\n        } else {\n' + consumer + '\n        }')
        path.write_text(asymmetric)
        template = recognized(tool, path)
        report = json.loads(invoke(tool, "--physical-trace", path))
        check_physical(template, report, {0, 1, 3, 5})
        check_phase_mapping(template, report, [0, 1, 3, 5])
        logical = opt(optimizer, path, ["--pto-frontier-analysis"]).stdout
        # The common producer cut contributes one readiness SET, while each
        # mutually exclusive consumer arm has its own readiness WAIT.
        assert logical.count('pto.logical_set[<PIPE_MTE2>, <PIPE_MTE1>]') == 1
        assert logical.count('pto.logical_wait[<PIPE_MTE2>, <PIPE_MTE1>]') == 2
        checked += 1
        compatibility_checks(optimizer, path)
        path.write_text(source)
        logical = opt(optimizer, path, ["--pto-frontier-analysis"]).stdout
        physical = opt(optimizer, path, ["--pto-frontier-analysis",
                                         "--pto-frontier-allocate=eligible-ids=0,1,3,5"]).stdout
        plan = recognized(tool, path)['logical_endpoints']
        pairs = sum(recipe['kind'] == 'set' for recipe in plan['recipes'])
        assert logical.count("pto.logical_set") < pairs, "family structure lost before insertion"
        assert physical.count("pto.set_flag") == logical.count("pto.logical_set"), "allocation changed family sites"
        assert "pto.endpoint_cut" not in physical
        assert physical.count("<PIPE_ALL>") == 1
        for name in ["scf.for", "pto.tload", "pto.textract"]:
            assert physical.count(name) == source.count(name)
    print(f"family emission: {checked} independent ordering checks, family sites, terminal completion")


if __name__ == "__main__":
    main()
