# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exact emitted-command comparisons for explicitly phased compact repeats."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_repeated_region import check, run


def main():
    tool, fixture = sys.argv[1:]
    original = Path(fixture).read_text()
    tested = 0
    with tempfile.TemporaryDirectory(prefix="repeated-phases-") as scratch:
        path = Path(scratch) / "case.pto"
        for q in (2, 3):
            source = original.replace("count=2", f"count={q}")
            source = source.replace("%two = arith.constant 2", f"%two = arith.constant {q}")
            source = source.replace("arith.remui %i, %two", "arith.remui %visit, %two")
            for n in sorted({0, 1, q-1, q, q+1}):
                for m in (0, 1, 3):
                    path.write_text(source.replace("array<i64: 2, 3>", f"array<i64: {n}, {m}>"))
                    check(json.loads(run(tool, "--structured-trace", path)), [n, m], outer_slots=q)
                    tested += 1
            path.write_text(source)
            report = json.loads(run(tool, "--sequence-analysis", path))
            assert not report["error"] and report["prepared"] and report["numeric_visits"] == 0, report
            small = run(tool, "--insert-logical", path)
            path.write_text(source.replace("array<i64: 2, 3>", "array<i64: 1000000000, 1000000001>"))
            large = run(tool, "--insert-logical", path)
            assert small == large.replace("1000000000, 1000000001", "2, 3")
            assert small.count("scf.for") == 2 and "version = 4" in small
        # Enclosing q=1 composition must preserve the child's period-coordinate
        # divisor and original phase binding at every emitted endpoint.
        triple = source.replace("%n: index, %m: index", "%p: index, %n: index, %m: index")
        triple = triple.replace("    scf.for %visit",
                                "    scf.for %outer = %zero to %p step %one {\n    scf.for %visit")
        triple = triple.replace("    }\n    pto.textract", "    }\n    }\n    pto.textract")
        for p, n, m in ((0, 4, 2), (2, 1, 2)):
            path.write_text(triple.replace("array<i64: 2, 3>", f"array<i64: {p}, {n}, {m}>"))
            check(json.loads(run(tool, "--structured-trace", path)), [p, n, m], outer_slots=3)
            tested += 1
        # Keep the combined external-boundary stress case. Its known large
        # scalar circuit gets an explicit bounded test-only allowance; ordinary
        # cases retain the interpreter's unchanged default ceiling.
        stress = triple.replace("test.trace_arguments =",
                                "test.trace_visit_limit = 8000000 : i64, test.trace_arguments =")
        path.write_text(stress.replace("array<i64: 2, 3>", "array<i64: 2, 4, 2>"))
        stress_report = json.loads(run(tool, "--structured-trace", path))
        check(stress_report, [2, 4, 2], outer_slots=3)
        assert stress_report["trace"]["visit_limit"] == 8000000, stress_report
        tested += 1
        for invalid_limit in (0, 20000001):
            invalid = stress.replace("8000000 : i64", f"{invalid_limit} : i64")
            path.write_text(invalid.replace("array<i64: 2, 3>", "array<i64: 0, 0, 0>"))
            failure = subprocess.run([tool, "--structured-trace", str(path)],
                                     capture_output=True, text=True, timeout=90, check=False)
            assert failure.returncode != 0, failure.stdout
            rejected_limit = json.loads(failure.stdout)
            assert "test.trace_visit_limit" in rejected_limit["trace"]["error"], rejected_limit
        print(f"combined phase stress: {stress_report['trace']['visits']} scalar visits "
              f"under explicit {stress_report['trace']['visit_limit']} ceiling")
        # Also isolate nested matching from the external boundary queries.
        core = triple.replace(
            "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)\n", "")
        path.write_text(core.replace("array<i64: 2, 3>", "array<i64: 2, 4, 2>"))
        check(json.loads(run(tool, "--structured-trace", path)), [2, 4, 2], outer_slots=3, external_work=False)
        tested += 1
        shifted = source.replace("%visit = %zero to %n step %one", "%visit = %one to %n step %stride")
        shifted = shifted.replace("    %one =", "    %stride = arith.constant 2 : index\n    %one =")
        path.write_text(shifted.replace("array<i64: 2, 3>", "array<i64: 8, 2>"))
        check(json.loads(run(tool, "--structured-trace", path)), [4, 2],
              outer_slots=3, outer_lower=1, outer_step=2)
        tested += 1
        # A writer discharged along the inner IV must not silently disappear
        # from the outer re-entry storage obligations.
        writer = source.replace("%n: index, %m: index", "%out: !pto.ptr<f32, gm>, %n: index, %m: index")
        writer = writer.replace("    %base =", "    %value = arith.constant 1.0 : f32\n    %base =")
        writer = writer.replace("      pto.tmatmul",
                                "      pto.store %value, %out[%i] : !pto.ptr<f32, gm>, f32\n      pto.tmatmul")
        path.write_text(writer)
        rejected = json.loads(run(tool, "--sequence-analysis", path))
        assert rejected["error"] and not rejected["prepared"], rejected
        print(f"repeated phases: {tested} exact command closures, partial periods and compact size passed")


if __name__ == "__main__":
    main()
