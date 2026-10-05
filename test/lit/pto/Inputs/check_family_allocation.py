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


def evaluate(text, ordinal, member):
    """Interpret the emitted integer expressions independently of the allocator."""
    arguments = re.search(r"func.func @irregular\(([^)]*)\)", text)
    names = re.findall(SSA, arguments[1])
    values = dict(zip(names, (ordinal, member)))
    commands = []
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
            elif kind == "cmpi":
                predicate = rest.split(",", 1)[0]
                assert predicate in ("eq", "ult"), predicate
                value = operands[0] == operands[1] if predicate == "eq" else operands[0] < operands[1]
            elif kind == "select":
                value = operands[1] if operands[0] else operands[2]
            else:
                raise AssertionError(("unexpected expression", line))
            values[name] = value
        command = re.search(rf"pto\.(set|wait)_flag_dyn.*({SSA})\]", line)
        if command:
            commands.append((command[1], values[command[2]]))
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


def main():
    optimizer = shutil.which(sys.argv[1])
    assert optimizer, "test optimizer must be available"
    source = Path(sys.argv[2]).read_text()
    with tempfile.TemporaryDirectory(prefix="family-allocation-") as directory:
        path = Path(directory) / "case.pto"
        phases = phase_checks(optimizer, path, source)
        rejected = rejection_checks(optimizer, path, source)
    print(f"family allocation: {phases} irregular phase evaluations, {rejected} rejected interfaces")


if __name__ == "__main__":
    main()
