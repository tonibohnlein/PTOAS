# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check nested ID matching and causal reuse against unfolded command graphs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_allocation import physical_check
from check_repeated_region import check


def trace(tool, path):
    result = subprocess.run([tool, "--structured-trace", str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads(result.stdout)


def main():
    tool, fixture = sys.argv[1:]
    original = Path(fixture).read_text().replace(
        "test.trace_arguments =", "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
    tested = 0
    with tempfile.TemporaryDirectory(prefix="nested-allocation-") as directory:
        path = Path(directory) / "case.pto"
        for period in (1, 2, 3):
            source = original
            if period == 1:
                source = source.replace("arith.remui %i, %two", "arith.remui %zero, %two")
            else:
                source = source.replace("count=2", f"count={period}")
                source = source.replace("%two = arith.constant 2", f"%two = arith.constant {period}")
                source = source.replace("arith.remui %i, %two", "arith.remui %visit, %two")
            for begin, end, inner in ((0, 0, 2), (0, 2, 0), (0, 1, 1), (0, period + 2, 3),
                                       (1, 1, 2), (1, period + 2, 2)):
                text = source.replace("array<i64: 2, 3>", f"array<i64: {end}, {inner}>")
                if begin:
                    text = text.replace("%visit = %zero", "%visit = %one")
                path.write_text(text)
                report = trace(tool, path)
                check(report, [max(0, end - begin), inner],
                      outer_slots=period, outer_lower=begin)
                assert report.get("allocated"), (period, begin, end, inner)
                physical_check(report, set(range(6)))
                tested += 1
        # Repeated composition must retain the source visit at three levels.
        triple = original.replace("arith.remui %i, %two", "arith.remui %zero, %two")
        triple = triple.replace("%n: index, %m: index", "%p: index, %n: index, %m: index")
        triple = triple.replace("    scf.for %visit",
                                "    scf.for %outer = %zero to %p step %one {\n    scf.for %visit")
        triple = triple.replace("    }\n    pto.textract", "    }\n    }\n    pto.textract")
        for dimensions in ((0, 2, 2), (2, 2, 0), (2, 2, 2)):
            numeric = triple.replace("to %n", "to %two").replace(
                "to %m", "to %two" if dimensions[2] else "to %zero")
            path.write_text(numeric.replace("array<i64: 2, 3>",
                                           "array<i64: " + ", ".join(map(str, dimensions)) + ">"))
            report = trace(tool, path)
            check(report, dimensions, outer_slots=1)
            physical_check(report, set(range(6)))
            tested += 1
        # A symbolic child palette may lack per-lane extrema at the next level.
        # Preserve the valid logical plan and report allocation failure explicitly.
        path.write_text(triple.replace("array<i64: 2, 3>", "array<i64: 2, 2, 3>"))
        unsupported = trace(tool, path)
        assert unsupported["accepted"] and not unsupported["allocated"]
        assert unsupported["allocation_unchanged_on_failure"]
    print(f"nested allocation: {tested} independent matching and causal-reuse traces passed")


if __name__ == "__main__":
    main()
