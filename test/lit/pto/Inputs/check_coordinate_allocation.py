# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check code growth and exact IDs independently of record numbering."""
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

from check_logical_insertion import invoke
from check_physical_allocation import opt


def fixture(size, threshold=False, sparse=False):
    """Saved certificates isolate lowering; no recognizer chooses these labels."""
    entries, members, records, even, odd, expected = [], [], [], [], [], []
    for row in range(size):
        for column in range(4):
            if sparse and column == 2:
                continue
            record = (4 * row + column) ** 2 + 13
            lane = int(row >= size // 2) if threshold else (row + column) % 2
            records.append(record)
            (even if column % 2 == 0 else odd).append(record)
            expected.append(["set", [1, 5][lane]])
            expected.append(["wait", [1, 5][lane]])
            entries.append(f"{{lane_begin = {lane} : i64, lane_count = 1 : i64, offset = 0 : i64, "
                           f"record = {record} : i64, source = 4 : i64, target = 3 : i64}}")
            point = f"[array<i64: 0, {1 + 2 * row}>, array<i64: 1, {column}>]"
            members.append(f"{{record = {record} : i64, source = {point}, target = {point}}}")
    pieces = []
    for piece, kind, cut, selected in [(0, 0, 0, records), (1, 2, 1, even), (2, 2, 1, odd)]:
        pieces.append(f"{{id = {piece} : i64, family = 13 : i64, kind = {kind} : i64, cut = {cut} : i64, "
                      "source_pipe = 4 : i64, target_pipe = 3 : i64, displacement = 0 : i64, "
                      "records = array<i64: " + ", ".join(map(str, selected)) + ">}")
    text = """module {
  func.func @coordinate_ids(%end: index) attributes {
    test.trace_existing, test.trace_arguments = array<i64: END>,
    pto.cyclic_allocation = {version = 3 : i64, plan = 0 : i64,
      strategy = "directed-cycle-cover", order_exact = true, budget = 2 : i64, entries = [ENTRIES]},
    pto.endpoint_families = {version = 3 : i64, plan = 0 : i64, families = [
      {id = 13 : i64, local = false, source_pipe = 4 : i64, target_pipe = 3 : i64,
       displacement = 0 : i64, source_cut = 0 : i64, target_cut = 1 : i64,
       source_order = 0 : i64, target_order = 0 : i64, members = [MEMBERS]}], pieces = [PIECES]}
  } {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %thirteen = arith.constant 13 : index
    %true = arith.constant true
    scf.for %i = %one to %end step %two {
      %offset = arith.subi %i, %one : index
      %row = arith.divui %offset, %two : index
      %base = arith.muli %row, %four : index
      scf.for %j = %zero to %four step %one {
        %flat = arith.addi %base, %j : index
        %square = arith.muli %flat, %flat : index
        %label = arith.addi %square, %thirteen : index
        %parity = arith.remui %j, %two : index
        %even = arith.cmpi eq, %parity, %zero : index
        %odd = arith.cmpi ne, %parity, %zero : index
        %keep = arith.cmpi ne, %j, %two : index
        scf.if KEEP {
          scf.if %true {
            pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 0 : i64}
          } {pto.endpoint_cut = 0 : i64}
          scf.if %even {
            pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 1 : i64}
          } {pto.endpoint_cut = 1 : i64}
          scf.if %odd {
            pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 13 ordinal %zero members(%label)
              {pto.endpoint_piece = 2 : i64}
          } {pto.endpoint_cut = 1 : i64}
        }
      } {pto.family_loop = 1 : i64}
    } {pto.family_loop = 0 : i64}
    return
  }
}
"""
    for token, replacement in [("END", str(1 + 2 * size)), ("ENTRIES", ", ".join(entries)),
                               ("MEMBERS", ", ".join(members)), ("PIECES", ", ".join(pieces)),
                               ("KEEP", "%keep" if sparse else "%true")]:
        text = text.replace(token, replacement)
    return text, expected


def commands(tool, path):
    trace = json.loads(invoke(tool, "--structured-trace", path))["trace"]
    assert not trace["error"], trace["error"]
    return [[event["kind"], event["physical_id"]] for event in trace["events"]
            if event["kind"] in ("set", "wait")]


def candidate_checks(optimizer, interpreter, path):
    # These saved, producer-trusted certificates isolate candidate consumption.
    # The production physical-allocation suite separately checks causal reuse.
    source, expected = fixture(8)
    records = [(4 * row + column) ** 2 + 13 for row in range(8) for column in range(4)]
    def alternative(budget):
        return ('regular_candidate = {version = 2 : i64, plan = 0 : i64, directions = ['
                '{source = 4 : i64, target = 3 : i64, budget = ' + str(budget) +
                ' : i64, records = array<i64: ' + ', '.join(map(str, records)) + '>}]}, ')
    # A cheaper certified constant map wins; an oversized candidate is ignored.
    for budget in (1, 3):
        path.write_text(source.replace('strategy = "directed-cycle-cover",',
                                       alternative(budget) + 'strategy = "directed-cycle-cover",'))
        physical = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,5"]).stdout
        path.write_text(physical)
        wanted = [[kind, 1] for kind, _ in expected] if budget == 1 else expected
        assert commands(interpreter, path) == wanted
        if budget == 1:
            assert 'strategy = "regular"' in physical
            assert 'pto.set_flag_dyn' not in physical and 'pto.wait_flag_dyn' not in physical


def sharing_checks(optimizer, interpreter, path):
    source, _ = fixture(8)
    # Every member has the same dynamic mapping. Arbitrary labels must disappear
    # before profitability is judged, even for nested tuple-family exports.
    source = re.sub(r'lane_begin = [01] : i64, lane_count = 1',
                    'lane_begin = 0 : i64, lane_count = 2', source)
    source = source.replace('ordinal %zero', 'ordinal %row')
    source = source.replace('pto.endpoint_families = {version = 3',
                            'pto.endpoint_families = {version = 4')
    records = [(4 * row + column) ** 2 + 13 for row in range(8) for column in range(4)]
    values = ', '.join(map(str, records))
    zeros = ', '.join('0' for _ in records)
    rule = '{coordinate_count = 1 : i64, base = 0 : i64, terms = [array<i64: 0, 1, 0, 2, 1>]}'
    certificate = ('pto.finite_allocation = {version = 2 : i64, plan = 0 : i64, kind = "finite", '
                   'strategy = "regional-palettes", groups = [{source = 4 : i64, target = 3 : i64, '
                   'budget = 2 : i64, records = array<i64: ' + values + '>, strides = array<i64: ' + zeros +
                   '>, phases = array<i64: ' + zeros + '>, conflicts = array<i64>, tuple_rules = [' +
                   ', '.join(rule for _ in records) + ']}]},\n    ')
    start = source.index('pto.cyclic_allocation =')
    end = source.index('pto.endpoint_families =', start)
    source = source[:start] + certificate + source[end:]
    path.write_text(source)
    physical = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,5"]).stdout
    # A common final palette expression is shared across the three sibling
    # endpoint guards. Count its result, not unrelated original label selects.
    palette_selects = re.findall(r' = arith.select .* : index', physical)
    assert len(palette_selects) == 1, physical
    path.write_text(physical)
    expected = [[kind, [1, 5][row % 2]] for row in range(8) for _ in range(4) for kind in ('set', 'wait')]
    assert commands(interpreter, path) == expected
    # Non-power-of-two division is outside the cheap-sharing policy. Its
    # complete selector must remain inside each command's original guard.
    source = source.replace('budget = 2 : i64', 'budget = 3 : i64')
    source = source.replace('array<i64: 0, 1, 0, 2, 1>', 'array<i64: 0, 1, 0, 3, 1>')
    path.write_text(source)
    physical = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,5"]).stdout
    threes = set(re.findall(r'(%[\w]+) = arith.constant 3 : index', physical))
    remainders = [line for line in physical.splitlines() if ' = arith.remui ' in line
                  and any(re.search(r', ' + re.escape(value) + r' : index', line) for value in threes)]
    assert len(remainders) == 6 and all(len(line) - len(line.lstrip()) >= 12 for line in remainders), physical
    path.write_text(physical)
    expected = [[kind, [1, 3, 5][row % 3]] for row in range(8) for _ in range(4) for kind in ('set', 'wait')]
    assert commands(interpreter, path) == expected


def common_tuple_checks(optimizer, interpreter, path):
    """Compose differing nonconstant mappings; preserve every supplied ID rule."""
    observed = []
    cases = [(False, False, False, False, 2), (False, True, True, False, 2),
             (True, False, False, False, 2), (False, False, False, True, 2), (False, False, False, False, 3),
             (False, False, False, "regular", 2), (True, False, False, "regular", 2),
             (False, True, True, "regular", 2), (False, False, False, "regular", 3),
             (True, False, True, "duplicate", 2), (False, False, False, "stride", 3)]
    for nested, sparse, huge, different, modulus in cases:
        sizes = []
        # Duplicate skeletons can require a non-affine phase selection after
        # sorting. Check exact semantics there, without promising compact fits.
        residual = different is True or different in ("stride", "duplicate")
        for size in ((8,) if residual else (8, 32, 128)):
            source, _ = fixture(size, sparse=sparse)
            records, rules, expected = [], [], []
            for row in range(size):
                for column in range(4):
                    if sparse and column == 2:
                        continue
                    record = (4 * row + column) ** 2 + 13
                    records.append(record)
                    base = (row + column) % 2
                    # Vary spelling and term order without changing the complete
                    # function. Constant terms fold into the certified base.
                    shifted = different in ("regular", "duplicate")
                    phase = (row + column) % modulus if shifted else int(
                        different and row == 0 and column == 0)
                    inner_phase = row % 2 if shifted else 0
                    stride = 1 + (column % 2 if different == "stride" else 0)
                    terms = [f"array<i64: 0, {modulus + stride}, {modulus + phase}, {modulus}, 2>"]
                    if nested:
                        coordinate = 0 if different == "duplicate" else 1
                        terms.append(f"array<i64: {coordinate}, 1, {inner_phase}, 2, 2>")
                    terms.append(f"array<i64: 0, 0, {base}, 2, 1>")
                    if column % 2:
                        terms.reverse()
                    rules.append("{coordinate_count = " + str(2 if nested else 1) +
                                 " : i64, base = 0 : i64, terms = [" + ", ".join(terms) + "]}")
                    ordinal = (1 << 64) - 1 if huge else row
                    inner_coordinate = ordinal if different == "duplicate" else column
                    offset = base + 2 * ((stride * ordinal + phase) % modulus) + (
                        2 * ((inner_coordinate + inner_phase) % 2) if nested else 0)
                    palette = list(range(6)) if nested or modulus == 3 else [0, 2, 3, 5]
                    expected.extend([[kind, palette[offset]] for kind in ("set", "wait")])
            values = ", ".join(map(str, records))
            zeros = ", ".join("0" for _ in records)
            certificate = ('pto.finite_allocation = {version = 2 : i64, plan = 0 : i64, kind = "finite", '
                           'strategy = "regional-palettes", groups = [{source = 4 : i64, target = 3 : i64, '
                           'budget = ' + str(len(palette)) + ' : i64, records = array<i64: ' + values +
                           '>, strides = array<i64: ' + zeros + '>, phases = array<i64: ' + zeros +
                           '>, conflicts = array<i64>, tuple_rules = [' + ', '.join(rules) + ']}]},\n    ')
            start = source.index('pto.cyclic_allocation =')
            end = source.index('pto.endpoint_families =', start)
            source = source[:start] + certificate + source[end:]
            source = source.replace('pto.endpoint_families = {version = 3',
                                    'pto.endpoint_families = {version = 4')
            source = source.replace('ordinal %zero', 'ordinal %huge' if huge else 'ordinal %row')
            if huge:
                source = source.replace('%zero = arith.constant',
                                        '%huge = arith.constant -1 : index\n    %zero = arith.constant', 1)
            if nested:
                source = source.replace('members(%label)', 'members(%label, %j)')
            path.write_text(source)
            flags = ['--pto-frontier-allocate=eligible-ids=' + ','.join(map(str, palette))]
            physical = opt(optimizer, path, flags).stdout
            sizes.append(len(re.findall(r" = arith\.", physical)))
            if modulus == 3:
                threes = set(re.findall(r'(%[\w]+) = arith.constant 3 : index', physical))
                remainders = [line for line in physical.splitlines() if ' = arith.remui ' in line
                              and any(re.search(r', ' + re.escape(value) + r' : index', line) for value in threes)]
                assert remainders and all(len(line) - len(line.lstrip()) >= 12 for line in remainders)
            assert physical.count('pto.set_flag') == 1 and physical.count('pto.wait_flag') == 2
            path.write_text(physical)
            assert commands(interpreter, path) == expected, (size, nested, sparse, huge, different)
            path.write_text(physical.replace(f"array<i64: {1 + 2 * size}>", "array<i64: 1>"))
            assert commands(interpreter, path) == []
        if not residual:
            assert max(sizes) < 150 and max(sizes) - min(sizes) <= 8, sizes
        observed.append(sizes)
    print("shared nonconstant tuple maps:", observed)


def main():
    optimizer, interpreter = map(shutil.which, sys.argv[1:3])
    scratch = Path(sys.argv[3])
    scratch.mkdir(parents=True, exist_ok=True)
    observed = []
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        path = Path(directory) / "case.pto"
        candidate_checks(optimizer, interpreter, path)
        sharing_checks(optimizer, interpreter, path)
        common_tuple_checks(optimizer, interpreter, path)
        for threshold, sparse, nested in [(t, s, n) for t, s in [(False, False), (False, True), (True, False)]
                                         for n in (False, True)]:
            sizes = []
            for size in (8, 32, 128):
                source, expected = fixture(size, threshold, sparse)
                if nested:
                    source = source.replace("pto.endpoint_families = {version = 3",
                                            "pto.endpoint_families = {version = 4")
                path.write_text(source)
                physical = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,5"]).stdout
                sizes.append(len(re.findall(r" = arith\.", physical)))
                assert physical.count("pto.set_flag") == 1
                assert physical.count("pto.wait_flag") == 2
                path.write_text(physical)
                assert commands(interpreter, path) == expected
                # Each actual command is checked, including noncontiguous IDs,
                # independent consumer pieces, nonunit coordinates and holes.
                path.write_text(physical.replace(f"array<i64: {1 + 2 * size}>", "array<i64: 1>"))
                assert commands(interpreter, path) == []
            assert max(sizes) < 150, sizes
            assert max(sizes) - min(sizes) <= 8, sizes
            observed.append(sizes)
    print("coordinate allocation: exact traces and bounded generated arithmetic", observed)


if __name__ == "__main__":
    main()
