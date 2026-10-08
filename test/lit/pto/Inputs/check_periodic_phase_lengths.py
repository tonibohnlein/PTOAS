# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check periodic inner domains against independent unfolded byte-conflict graphs."""
import json
from pathlib import Path
import sys
import tempfile

from check_composed_phases import check, invoke


def variants(source):
    """The four-phase length requires a larger period than the two storage banks."""
    loop = "      scf.for %i = %zero to %m step %one {"
    unequal = source.replace("    %two = arith.constant 2 : index",
                             "    %two = arith.constant 2 : index\n"
                             "    %four = arith.constant 4 : index")
    unequal = unequal.replace(loop,
        "      %length_phase = arith.remui %visit, %four : index\n"
        "      %inner_limit = arith.addi %m, %length_phase : index\n"
        "      scf.for %i = %zero to %inner_limit step %one {")
    alternating = source.replace(loop,
        "      %inner_limit = arith.muli %m, %slot : index\n"
        "      scf.for %i = %zero to %inner_limit step %one {")
    compute = ('        pto.tadds ins(%selected, %value : !cell, f32) '
               'outs(%selected : !cell) {test.label = "compute"}')
    sliced = unequal.replace(compute,
        "        %first_inner = arith.cmpi eq, %i, %zero : index\n"
        "        scf.if %first_inner {\n" + compute + "\n        } else {\n" + compute + "\n        }")
    return (("unequal", unequal, lambda m: (m, m + 1, m + 2, m + 3)),
            ("alternating_empty", alternating, lambda m: (0, m)),
            ("unequal_sliced", sliced, lambda m: (m, m + 1, m + 2, m + 3)))


def check_compact(tool, path, source):
    """Changing runtime trip values must not expand loops or compiler work."""
    path.write_text(source)
    report = json.loads(invoke(tool, "--sequence-analysis", path))
    assert not report["error"] and report["prepared"], report
    assert report["numeric_visits"] == 0, report
    assert report["repeated_regions"] > 0 and report["phase_descriptions"] > 0, report
    small = invoke(tool, "--insert-logical", path)
    path.write_text(source.replace("array<i64: 3, 2>", "array<i64: 1000000001, 1000000000>"))
    large = invoke(tool, "--insert-logical", path)
    assert small == large.replace("1000000001, 1000000000", "3, 2")
    assert sum(line.lstrip().startswith("scf.for ") for line in small.splitlines()) == 2


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="periodic-phase-lengths-") as scratch:
        path = Path(scratch) / "case.pto"
        for name, case, lengths in variants(source):
            # Includes no visits, empty inner phases, full periods and partial
            # final periods. Every byte conflict and SSA edge uses actual visits.
            for n, m in ((0, 0), (1, 0), (2, 1), (3, 0), (5, 2), (6, 1)):
                path.write_text(case.replace("array<i64: 3, 2>", f"array<i64: {n}, {m}>"))
                document = json.loads(invoke(tool, "--structured-trace", path))
                check(document, n, m, phase_lengths=lengths(m))
            check_compact(tool, path, case)
            print(f"periodic phase lengths: {name} exact closures and compact emission passed")
        # A raw outer IV is not a periodic length. Other exact adapters may
        # handle this nest, but no phase representative can replace its bound.
        nonperiodic = source.replace("      scf.for %i = %zero to %m step %one {",
            "      %inner_limit = arith.addi %m, %visit : index\n"
            "      scf.for %i = %zero to %inner_limit step %one {")
        path.write_text(nonperiodic)
        report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert report.get("phase_descriptions", 0) == 0, report


if __name__ == "__main__":
    main()
