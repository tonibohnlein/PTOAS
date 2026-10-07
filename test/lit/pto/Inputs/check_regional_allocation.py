# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate composed cyclic IDs against command reachability, including loop boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_allocation import physical_check
from check_sequence_analysis import validate
from check_finite_guarded_analysis import load, compute, render


def invoke(tool, arguments, path):
    return subprocess.run([tool, *arguments, str(path)], capture_output=True, text=True, timeout=90)


def main():
    tool, opt, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    start = source.index("    scf.for")
    end = source.index("    }", start) + len("    }\n")
    second = source[start:end].replace("%i", "%j").replace("%n", "%m")
    two = (source[:end] + second + source[end:]).replace("%n: index)", "%n: index, %m: index)")
    cases = [(source, [n]) for n in [-2, 0, 1, 2, 3, 5, 9]]
    cases += [(two, [n, m]) for n, m in [(0, 0), (0, 3), (4, 0), (1, 1), (2, 3), (5, 4)]]
    with tempfile.TemporaryDirectory(prefix="regional-allocation-") as directory:
        path = Path(directory) / "case.pto"
        for text, trips in cases:
            text = text.replace("array<i64: 5>", "array<i64: " + ", ".join(map(str, trips)) + ">")
            text = text.replace("test.trace_arguments =",
                                "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
            path.write_text(text)
            run = invoke(tool, ["--structured-trace"], path)
            assert run.returncode == 0, run.stderr
            report = json.loads(run.stdout)
            validate(report, trips)
            physical_check(report, set(range(6)))
        # L->D is retained only when the optional C is absent; endpoint
        # presence alone misses that it is exclusive with L->C. One ID suffices
        # because the regional adapter simplifies the integer rank guards.
        guarded = (Path(fixture).parent / "sync_finite_guarded_analysis.pto").read_text()
        body = load("L") + "\nscf.if %g {\n" + compute("C") + "\n}\n" + compute("D")
        for guard in [0, 1]:
            text = render(guarded, body, guard, 0).replace(
                "test.trace_arguments =", "test.eligible_ids = array<i64: 0>, test.trace_arguments =")
            path.write_text(text)
            run = invoke(tool, ["--structured-trace"], path)
            assert run.returncode == 0, run.stderr
            physical_check(json.loads(run.stdout), {0})
        path.write_text(source)
        logical = invoke(opt, ["--mlir-disable-threading", "--pto-frontier-analysis"], path)
        assert logical.returncode == 0, logical.stderr
        assert 'strategy = "regional-palettes"' in logical.stdout
        path.write_text(logical.stdout)
        args = ["--mlir-disable-threading", "--pto-frontier-allocate=eligible-ids=0,3,5"]
        physical = invoke(opt, args, path)
        assert physical.returncode == 0, physical.stderr
        assert "pto.logical_" not in physical.stdout and physical.stdout.count("scf.for") == 1
        # Reject malformed palette references before changing logical IR.
        for old, new in [('version = 2 : i64', 'version = 1 : i64'),
                         ('budget = 2 : i64', 'budget = 0 : i64'),
                         ('conflicts = array<i64>', 'conflicts = array<i64: 999999>'),
                         ('phases = array<i64: 0>', 'phases = array<i64: -1>'),
                         ('strides = array<i64: 1, 1>', 'strides = array<i64: -1, 1>'),
                         ('strides = array<i64: 1, 1>', 'strides = array<i64>'),
                         ('local = 1 : i64', 'local = 0 : i64'),
                         ('successors = array<i64>', 'successors = array<i64: 999999>'),
                         ('sources = array<i64: 3, 2>', 'sources = array<i64: 3>')]:
            assert old in logical.stdout, old
            path.write_text(logical.stdout.replace(old, new, 1))
            broken = invoke(opt, args, path)
            assert broken.returncode != 0 and "error:" in broken.stderr
        path.write_text(source.replace("test.trace_arguments =",
                                       "test.eligible_ids = array<i64: 0>, test.trace_arguments ="))
        report = json.loads(invoke(tool, ["--structured-trace"], path).stdout)
        assert report["accepted"] and not report["allocated"] and report["allocation_unchanged_on_failure"]
    print(f"regional allocation: {len(cases) + 2} independent causal-reuse traces; "
          "sparse IDs and failure checks passed")


if __name__ == "__main__":
    main()
