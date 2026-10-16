# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Finite regional semantic and inspected-work scaling gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_guarded_analysis import compute, load, render
from check_deferred_family_output import executable


def main():
    tool, fixture = sys.argv[1:]
    tool = executable(tool)
    template = Path(fixture).read_text(encoding="utf-8")
    reports = []
    with tempfile.TemporaryDirectory(prefix="finite-qualification-") as scratch:
        path = Path(scratch) / "finite.pto"

        def run(body, mode):
            source = render(template, body, 1, 1).replace(
                "test.trace_arguments =", f'test.finite_qualification = "{mode}", test.trace_arguments =')
            path.write_text(source, encoding="utf-8")
            completed = subprocess.run([tool, "--gm-alias=may-not-alias", "--finite-guarded-analysis", str(path)],
                                       capture_output=True, text=True, timeout=120, check=False)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr + completed.stdout)
            report = json.loads(completed.stdout)
            reports.append({"mode": mode, **report})
            return report

        for body in (
            "",
            f"scf.if %g {{ {load()} scf.if %h {{ {compute()} }} }} else {{ {load('B')} }} "
            f"{compute('R', True)} {load('E')}",
            f"{load()} scf.if %g {{ }} else {{ {compute()} }} scf.if %h {{ {load('E')} }} {compute('R', True)}",
        ):
            report = run(body, "partitions")
            assert report["partitions"] >= 3 and report["comparisons"] > 0

        for n in (32, 64, 128, 256, 512, 1024, 2048, 4096):
            body = "\n".join(load(f"L{i}") if i % 2 == 0 else compute(f"C{i}") for i in range(n))
            report = run(body, "concrete")
            assert report["sites"] == n and report["query_nodes"] == 0
            assert report["numerical_queries"] == n and report["recipe_hits"] == 1
            assert report["incidences"] <= 3 * n and report["hazard_visits"] <= 3 * n
            assert report["residual_phases"] == report["residual_effects"] == 0
            assert report["slice_root_copies"] <= n + 20

        for n in (16, 32, 64, 128, 256, 512, 1024):
            # One shared source writer, many readers with disjoint private results.
            lines = [load()]
            for i in range(n):
                lines += [f"%addr{i} = arith.constant {1024 * (i + 1)} : i64",
                          f"%out{i} = pto.alloc_tile addr = %addr{i} : !acc",
                          "scf.if %g { " + compute(f"C{i}").replace("outs(%acc", f"outs(%out{i}") + " }"]
            report = run("\n".join(lines), "guarded")
            assert report["sites"] == n + 1
            assert report["W"] <= 4 * (n + 1) and report["pair_visits"] <= 8 * (n + 1)
            assert report["expressions"] <= 80 * (n + 1)
            assert report["residual_phases"] == report["residual_effects"] == 0

        for n in (8, 16, 32, 64, 128):
            body = "\n".join(f"scf.if %{'g' if i % 2 else 'h'} {{ {load(f'L{i}')} }}" for i in range(n))
            report = run(body, "guarded")
            assert report["sites"] == n and report["W"] == n * n
            assert report["pair_visits"] == 2 * n * (n - 1)
            assert report["candidates"] == n * (n - 1) // 2
            assert report["expressions"] <= 80 * (n * n + n)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
