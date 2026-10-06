# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check compact physical assignments against logical identities and causal reuse."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_finite_allocation import physical_check


def trace(tool, path):
    result = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def eligible(text, ids="0, 1, 2, 3, 4, 5"):
    return text.replace("test.trace_arguments =", f"test.eligible_ids = array<i64: {ids}>, test.trace_arguments =")


def main():
    tool, opt, directory = sys.argv[1:]
    directory = Path(directory)
    nested = (directory / "sync_arithmetic_compaction.pto").read_text()
    rotating = (directory / "sync_rotating_analysis.pto").read_text()
    rotating = rotating[:rotating.index("  func.func @ring_short")] + "}\n"
    guarded = (directory / "sync_guarded_rotating_analysis.pto").read_text()
    count = 0
    with tempfile.TemporaryDirectory(prefix="compact-allocation-") as scratch:
        path = Path(scratch) / "input.pto"
        for trips in [0, 1, 2, 5, 10]:
            path.write_text(eligible(nested.replace("constant 10 : index", f"constant {trips} : index")))
            physical_check(trace(tool, path), set(range(6)))
            path.write_text(eligible(rotating.replace("array<i64: 0>", f"array<i64: {trips}>")))
            physical_check(trace(tool, path), set(range(6)))
            count += 2
        for g, h, trips, offset in itertools.product((0, 1), (0, 1), (0, 1, 4), (0, 1)):
            text = guarded.replace("%i, %offset", "%i, %zero" if offset == 0 else "%i, %one")
            text = text.replace("array<i64: 1, 1, 3, 0>", f"array<i64: {g}, {h}, {trips}, {offset}>")
            path.write_text(eligible(text))
            physical_check(trace(tool, path), set(range(6)))
            count += 1
        # Compile, do not execute, a billion-trip nest. Physical IR size must
        # stay independent of the numerical trip count.
        sizes = []
        for trips in [10, 1000000000]:
            path.write_text(nested.replace("constant 10 : index", f"constant {trips} : index"))
            result = subprocess.run([opt, "--mlir-disable-threading", "--pto-frontier-analysis",
                                     "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", str(path)],
                                    capture_output=True, text=True, timeout=45)
            assert result.returncode == 0, result.stderr
            assert "pto.logical_" not in result.stdout and result.stdout.count("scf.for") == 2
            sizes.append(len(result.stdout.splitlines()))
        assert max(sizes) <= min(sizes) + 5, sizes
        # Symbolic offsets remain unsupported by this sufficient constant-cycle
        # proof. Logical endpoints survive allocation failure unchanged.
        path.write_text(eligible(guarded))
        report = trace(tool, path)
        assert report["accepted"] and not report["allocated"] and report["allocation_unchanged_on_failure"]
        # Known two-bank numerical pipeline cannot fit its certified budget in one ID.
        path.write_text(eligible(rotating.replace("array<i64: 0>", "array<i64: 5>"), "0"))
        report = trace(tool, path)
        assert report["accepted"] and not report["allocated"] and report["allocation_unchanged_on_failure"]
        # Contradictory immutable guards cannot leave allocated dead records.
        text = guarded.replace("%i, %offset", "%i, %zero")
        text = text.replace("      scf.if %g {", "      %yes = arith.constant true\n"
                            "      %notg = arith.xori %g, %yes : i1\n"
                            "      %never = arith.andi %g, %notg : i1\n      scf.if %never {")
        path.write_text(eligible(text))
        physical_check(trace(tool, path), set(range(6)))
    print(f"compact allocation: {count + 1} causal matching traces and two transactional rejections passed")


if __name__ == "__main__":
    main()
