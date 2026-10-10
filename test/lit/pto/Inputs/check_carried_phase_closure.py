# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare actual carried-bank synchronization with independent byte closures."""
import json
from pathlib import Path
import sys
import tempfile

from check_composed_phases import check, invoke


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    carried = source.replace(
        "scf.for %visit = %zero to %n step %one {",
        "%last = scf.for %visit = %zero to %n step %one iter_args(%bank = %one) -> index {")
    carried = carried.replace("%slot = arith.remui %visit, %two : index",
                              "%sum = arith.addi %bank, %one : index\n"
                              "      %slot = arith.remui %sum, %two : index")
    carried = carried.replace("    }\n    return", "      scf.yield %slot : index\n    }\n    return")
    with tempfile.TemporaryDirectory(prefix="carried-phase-") as scratch:
        path = Path(scratch) / "input.pto"
        for n, m in ((0, 0), (0, 3), (1, 0), (1, 2), (2, 1), (3, 2), (5, 3)):
            path.write_text(carried.replace("array<i64: 3, 2>", f"array<i64: {n}, {m}>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), n, m)
        # Exercise the phase composer itself so a different exact producer
        # cannot hide a broken carried-state admission or specialization.
        phase_lengths = carried.replace("      scf.for %i = %zero to %m step %one {",
                                        "      %inner_limit = arith.muli %m, %slot : index\n"
                                        "      scf.for %i = %zero to %inner_limit step %one {")
        path.write_text(phase_lengths.replace("attributes {test.trace_arguments",
                                              "attributes {test.phased_insertion, test.trace_arguments"))
        phased = json.loads(invoke(tool, "--structured-trace", path))
        assert phased["phase_descriptions"] > 0, phased
        check(phased, 3, 2, phase_lengths=(0, 2))
        nested = carried.replace("    %last = scf.for %visit",
                                 "    scf.for %outer = %zero to %two step %one {\n    %last = scf.for %visit")
        nested = nested.replace("    return", "    }\n    return")
        path.write_text(nested)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2, repeats=2)
        nonunit = carried.replace("    %scalar =", "    %three = arith.constant 3 : index\n    %scalar =")
        nonunit = nonunit.replace("%visit = %zero to %n step %one", "%visit = %three to %n step %two")
        for upper in (0, 3, 4, 6, 8):
            path.write_text(nonunit.replace("array<i64: 3, 2>", f"array<i64: {upper}, 2>"))
            check(json.loads(invoke(tool, "--structured-trace", path)), max(0, (upper - 2) // 2), 2,
                  visit_lower=3, visit_step=2)
        orbit = carried.replace("count=2", "count=4")
        orbit = orbit.replace("    %scalar =", "    %four = arith.constant 4 : index\n    %scalar =")
        orbit = orbit.replace("iter_args(%bank = %one)", "iter_args(%bank = %zero)")
        orbit = orbit.replace("arith.addi %bank, %one", "arith.addi %bank, %two")
        orbit = orbit.replace("arith.remui %sum, %two", "arith.remui %sum, %four")
        path.write_text(orbit)
        check(json.loads(invoke(tool, "--structured-trace", path)), 3, 2, bank_sequence=(2, 0))
    print("carried phases: command closures, nonunit coordinates, zero/partial trips and nested resets passed")


if __name__ == "__main__":
    main()
