# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check allocated family phase maps and reject corrupted allocation interfaces."""
from pathlib import Path
import re
import shutil
import sys
import tempfile

from check_physical_allocation import opt


SSA = r"%[A-Za-z0-9_]+"
MASK = (1 << 64) - 1


def evaluate(text, ordinal, member, *visits):
    """Interpret the emitted integer expressions independently of the allocator."""
    arguments = re.search(r"func.func @irregular\(([^)]*)\)", text)
    names = re.findall(SSA, arguments[1])
    values = dict(zip(names, (ordinal, member, *visits)))
    commands = []
    active = True
    for line in text.splitlines():
        assignment = re.search(rf"({SSA}) = arith\.(\w+) (.*)", line)
        if assignment:
            name, kind, rest = assignment.groups()
            operands = [values[operand] for operand in re.findall(SSA, rest)]
            if kind == "constant":
                value = int(rest.split()[0])
            elif kind == "addi":
                value = sum(operands) & MASK
            elif kind == "muli":
                value = operands[0] * operands[1] & MASK
            elif kind == "remui":
                value = operands[0] % operands[1]
            elif kind == "ori":
                value = operands[0] | operands[1]
            elif kind == "andi":
                value = operands[0] & operands[1]
            elif kind == "cmpi":
                predicate = rest.split(",", 1)[0]
                assert predicate in ("eq", "ult"), predicate
                value = operands[0] == operands[1] if predicate == "eq" else operands[0] < operands[1]
            elif kind == "select":
                value = operands[1] if operands[0] else operands[2]
            else:
                raise AssertionError(("unexpected expression", line))
            values[name] = value
        branch = re.search(rf"scf.if ({SSA})", line)
        if branch:
            active = bool(values[branch[1]])
        command = re.search(rf"pto\.(set|wait)_flag_dyn.*({SSA})\]", line)
        if command and active:
            commands.append((command[1], values[command[2]]))
        if line.strip().startswith("}"):
            active = True
    return commands


def phase_checks(optimizer, path, text):
    path.write_text(text)
    emitted = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,4,5"]).stdout
    assert emitted.count("pto.set_flag_dyn") == 2 and emitted.count("pto.wait_flag_dyn") == 2
    assert "pto.logical_" not in emitted and "pto.endpoint_families" not in emitted
    ids, phases = [1, 3, 4, 5], [0, 2, 1, 3]
    # Five records rotate the four eligible IDs; the four-member family has a
    # non-affine phase permutation, while the other family is a singleton.
    for ordinal in [0, 1, 2, 17, (1 << 63) - 1, MASK]:
        for member, phase in enumerate(phases):
            selected = [ids[(ordinal + phase) % 4], ids[ordinal % 4]]
            expected = list(zip(["set", "set", "wait", "wait"], selected + selected))
            assert evaluate(emitted, ordinal, member) == expected, (ordinal, member)
    return 24


def rejection_checks(optimizer, path, source):
    member13 = "{record = 13 : i64, source = [], target = []}"
    cases = [
        source.replace(", " + member13, ""),
        source.replace(member13, "{record = 11 : i64, source = [], target = []}"),
        source.replace("source_cut = 0 : i64", "source_cut = 9 : i64"),
        source.replace("members(%member)", "members(%four)"),
        source.replace("version = 2 : i64", "version = 7 : i64"),
    ]
    for text in cases:
        path.write_text(text)
        result = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,4,5"], success=False)
        assert "error:" in result.stderr
    return len(cases)


def piece_source(source):
    """Use independent source/consumer partitions with original record labels."""
    source = source.replace("pto.endpoint_families = {version = 2 : i64", "pto.endpoint_families = {version = 3 : i64")
    closing = "members = [{record = 20 : i64, source = [], target = []}]}]}"
    pieces = """members = [{record = 20 : i64, source = [], target = []}]}], pieces = [
      {id = 0 : i64, family = 10 : i64, kind = 0 : i64, cut = 0 : i64,
       source_pipe = 4 : i64, target_pipe = 3 : i64, displacement = 0 : i64,
       records = array<i64: 10, 11, 12, 13, 20>},
      {id = 1 : i64, family = 10 : i64, kind = 2 : i64, cut = 1 : i64,
       source_pipe = 4 : i64, target_pipe = 3 : i64, displacement = 0 : i64,
       records = array<i64: 10, 12, 20>},
      {id = 2 : i64, family = 10 : i64, kind = 2 : i64, cut = 1 : i64,
       source_pipe = 4 : i64, target_pipe = 3 : i64, displacement = 0 : i64,
       records = array<i64: 11, 13>}]}"""
    assert closing in source
    source = source.replace(closing, pieces)
    body = """    %ten = arith.constant 10 : index
    %eleven = arith.constant 11 : index
    %twelve = arith.constant 12 : index
    %thirteen = arith.constant 13 : index
    %twenty = arith.constant 20 : index
    %p10 = arith.cmpi eq, %member, %ten : index
    %p11 = arith.cmpi eq, %member, %eleven : index
    %p12 = arith.cmpi eq, %member, %twelve : index
    %p13 = arith.cmpi eq, %member, %thirteen : index
    %p20 = arith.cmpi eq, %member, %twenty : index
    %even0 = arith.ori %p10, %p12 : i1
    %even = arith.ori %even0, %p20 : i1
    %odd = arith.ori %p11, %p13 : i1
    %present = arith.ori %even, %odd : i1
    scf.if %present {
      pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 10 ordinal %ordinal members(%member)
        {pto.endpoint_piece = 0 : i64}
    } {pto.endpoint_cut = 0 : i64}
    scf.if %even {
      pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 10 ordinal %ordinal members(%member)
        {pto.endpoint_piece = 1 : i64}
    } {pto.endpoint_cut = 1 : i64}
    scf.if %odd {
      pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 10 ordinal %ordinal members(%member)
        {pto.endpoint_piece = 2 : i64}
    } {pto.endpoint_cut = 1 : i64}
    return
  }
}
"""
    return source[:source.index("    %four =")] + body


def piece_checks(optimizer, path, source):
    text = piece_source(source)
    path.write_text(text)
    emitted = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,4,5"]).stdout
    assert emitted.count("pto.set_flag_dyn") == 1 and emitted.count("pto.wait_flag_dyn") == 2
    ids, phases = [1, 3, 4, 5], {10: 0, 11: 2, 12: 1, 13: 3, 20: 0}
    for ordinal in [0, 1, 2, 17, MASK]:
        for label, phase in phases.items():
            expected = ids[(ordinal + phase) % 4]
            assert evaluate(emitted, ordinal, label) == [("set", expected), ("wait", expected)]
    assert evaluate(emitted, 0, 14) == [], "nonmember label acquired a notification"
    # Even-label and odd-label pieces have differences noninvertible modulo four.
    # Their affine maps still exist and must not be rejected by inverse-based fitting.
    regular = text.replace("records = array<i64: 10, 12, 11, 13, 20>",
                           "records = array<i64: 10, 11, 12, 13, 20>")
    path.write_text(regular)
    fitted = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,4,5"]).stdout
    for label, phase in {10: 0, 11: 1, 12: 2, 13: 3, 20: 0}.items():
        expected = ids[(2 + phase) % 4]
        assert evaluate(fitted, 2, label) == [("set", expected), ("wait", expected)]
    cases = [
        text.replace("records = array<i64: 10, 12, 20>", "records = array<i64: 10, 12>"),
        text.replace("records = array<i64: 11, 13>", "records = array<i64: 10, 11, 13>"),
        text.replace("family = 10 : i64", "family = 20 : i64"),
        text.replace("pto.endpoint_piece = 2 : i64", "pto.endpoint_piece = 1 : i64"),
        text.replace("members(%member)", "members(%twenty)"),
        text.replace("kind = 0 : i64, cut = 0 : i64", "kind = 0 : i64, cut = 2 : i64"),
        text.replace("{record = 10 : i64, source = [], target = []}",
                     "{record = 10 : i64, source = [array<i64: 99, 0>], target = []}"),
        text.replace("{record = 10 : i64, source = [], target = []}",
                     "{record = 10 : i64, source = [array<i64: 99>], target = []}"),
    ]
    for broken in cases:
        path.write_text(broken)
        opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,4,5"], success=False)
    return 31, len(cases)

def tuple_checks(optimizer, path, source):
    """Exercise certified source tuples, independent pieces and differing rules."""
    text = piece_source(source).replace("version = 3 : i64", "version = 4 : i64")
    text = text.replace("%member: index)", "%member: index, %visit: index)")
    text = text.replace("members(%member)", "members(%member, %visit)")
    start = text.index("    pto.cyclic_allocation =")
    end = text.index("    pto.endpoint_families =", start)
    certificate = """    pto.finite_allocation = {version = 2 : i64, plan = 0 : i64, kind = "finite",
      strategy = "regional-palettes", groups = [
      {source = 4 : i64, target = 3 : i64, budget = 4 : i64,
       records = array<i64: 10, 11, 12, 13>, strides = array<i64: 0, 0, 0, 0>,
       phases = array<i64: 0, 0, 0, 0>, conflicts = array<i64>, tuple_rules = [
       {coordinate_count = 2 : i64, base = 0 : i64,
        terms = [array<i64: 0, 1, 0, 2, 1>, array<i64: 1, 1, 0, 2, 2>]},
       {coordinate_count = 2 : i64, base = 0 : i64,
        terms = [array<i64: 0, 1, 1, 2, 1>, array<i64: 1, 1, 1, 2, 2>]},
       {coordinate_count = 2 : i64, base = 0 : i64, terms = [array<i64: 1, 1, 0, 4, 1>]},
       {coordinate_count = 2 : i64, base = 1 : i64, terms = [array<i64: 0, 1, 0, 3, 1>]}]},
      {source = 4 : i64, target = 3 : i64, budget = 2 : i64,
       records = array<i64: 20>, strides = array<i64: 0>, phases = array<i64: 1>,
       conflicts = array<i64: 0>, tuple_rules = [{}]}]},
"""
    text = text[:start] + certificate + text[end:]
    path.write_text(text)
    allocate = ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]
    emitted = opt(optimizer, path, allocate).stdout
    assert emitted.count("pto.set_flag_dyn") == 1 and emitted.count("pto.wait_flag_dyn") == 2
    assert "pto.logical_" not in emitted
    checked = 0
    for ordinal in (0, 1, 2, 17, (1 << 63) - 1, MASK):
        for visit in (0, 1, 2, 9, MASK):
            expected = {10: ordinal % 2 + 2 * (visit % 2),
                        11: (ordinal + 1) % 2 + 2 * ((visit + 1) % 2),
                        12: visit % 4, 13: 1 + ordinal % 3, 20: 5}
            for label, event_id in expected.items():
                assert evaluate(emitted, ordinal, label, visit) == [("set", event_id), ("wait", event_id)]
                checked += 1
    cases = [
        text.replace("coordinate_count = 2", "coordinate_count = 1", 1),
        text.replace("array<i64: 0, 1, 0, 2, 1>", "array<i64: 2, 1, 0, 2, 1>", 1),
        text.replace("array<i64: 0, 1, 0, 2, 1>", "array<i64: 0, 1, 0, 0, 1>", 1),
        text.replace("array<i64: 0, 1, 0, 2, 1>", "array<i64: 0, 1, 0, 2, 9223372036854775807>", 1),
        text.replace("base = 0 : i64", "base = 4 : i64", 1),
        text.replace("members(%member, %visit)", "members(%member)", 1),
        text.replace("members(%member, %visit)\n        {pto.endpoint_piece = 1",
                     "members(%member, %visit, %visit)\n        {pto.endpoint_piece = 1", 1),
        text.replace("strides = array<i64: 0>, phases = array<i64: 1>",
                     "strides = array<i64: 1>, phases = array<i64: 1>", 1),
    ]
    for broken in cases:
        assert broken != text
        path.write_text(broken)
        result = opt(optimizer, path, allocate, success=False)
        assert "error:" in result.stderr
    return checked, len(cases)


def mixed_palette_checks(optimizer, path, source):
    """Keep v2 families and v3 pieces compact across different cyclic palettes."""
    certificate = """    pto.finite_allocation = {version = 2 : i64, plan = 0 : i64, kind = "finite",
      strategy = "regional-palettes", groups = [
      {source = 4 : i64, target = 3 : i64, budget = 3 : i64,
       records = array<i64: 10, 12, 20>, strides = array<i64: 1, 2, 1>,
       phases = array<i64: 0, 2, 1>, conflicts = array<i64>},
      {source = 4 : i64, target = 3 : i64, budget = 2 : i64,
       records = array<i64: 11, 13>, strides = array<i64: 1, 1>,
       phases = array<i64: 1, 0>, conflicts = array<i64: 0>}]},
"""
    checked = 0
    for pieces, text in [(False, source), (True, piece_source(source))]:
        start = text.index("    pto.cyclic_allocation =")
        end = text.index("    pto.endpoint_families =", start)
        text = text[:start] + certificate + text[end:]
        path.write_text(text)
        emitted = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=1,3,5,0,2"]).stdout
        assert "pto.logical_" not in emitted
        assert emitted.count("pto.set_flag_dyn") == (1 if pieces else 2)
        assert emitted.count("pto.wait_flag_dyn") == 2
        for ordinal in (0, 1, 2, 17, (1 << 63) - 1, MASK):
            expected = {10: [1, 3, 5][ordinal % 3], 11: [0, 2][(ordinal + 1) % 2],
                        12: [1, 3, 5][(2 * ordinal + 2) % 3], 13: [0, 2][ordinal % 2],
                        20: [1, 3, 5][(ordinal + 1) % 3]}
            if pieces:
                for record, event_id in expected.items():
                    assert evaluate(emitted, ordinal, record) == [("set", event_id), ("wait", event_id)]
                    checked += 1
            else:
                for member, record in enumerate((10, 11, 12, 13)):
                    values = [expected[record], expected[20]]
                    commands = list(zip(("set", "set", "wait", "wait"), values + values))
                    assert evaluate(emitted, ordinal, member) == commands
                    checked += 1
        assert evaluate(emitted, 0, 14 if pieces else 4) == []
    return checked


def legacy_tuple_rejection(optimizer, path):
    """A legacy flat endpoint cannot supply an enclosing-visit coordinate."""
    path.write_text('''module {
  func.func @legacy(%ordinal: index) attributes {
    pto.finite_allocation = {version = 2 : i64, plan = 0 : i64, kind = "finite",
      strategy = "regional-palettes", groups = [
      {source = 4 : i64, target = 3 : i64, budget = 2 : i64,
       records = array<i64: 0>, strides = array<i64: 0>, phases = array<i64: 0>,
       conflicts = array<i64>, tuple_rules = [
        {coordinate_count = 2 : i64, base = 0 : i64,
         terms = [array<i64: 1, 1, 0, 2, 1>]}]}]},
    pto.endpoint_families = {version = 1 : i64, plan = 0 : i64}
  } {
    pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %ordinal
    pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %ordinal
    return
  }
}''')
    result = opt(optimizer, path, ["--pto-frontier-allocate=eligible-ids=0,1"], success=False)
    assert "one-coordinate allocation rule" in result.stderr, result.stderr
    return 1


def main():
    optimizer = shutil.which(sys.argv[1])
    assert optimizer, "test optimizer must be available"
    source = Path(sys.argv[2]).read_text()
    with tempfile.TemporaryDirectory(prefix="family-allocation-") as directory:
        path = Path(directory) / "case.pto"
        phases = phase_checks(optimizer, path, source)
        rejected = rejection_checks(optimizer, path, source)
        piece_phases, piece_rejected = piece_checks(optimizer, path, source)
        tuple_phases, tuple_rejected = tuple_checks(optimizer, path, source)
        mixed_phases = mixed_palette_checks(optimizer, path, source)
        legacy_rejected = legacy_tuple_rejection(optimizer, path)
    print(f"family allocation: {phases + piece_phases + tuple_phases + mixed_phases} phase evaluations, "
          f"{rejected + piece_rejected + tuple_rejected + legacy_rejected} rejected interfaces")


if __name__ == "__main__":
    main()
