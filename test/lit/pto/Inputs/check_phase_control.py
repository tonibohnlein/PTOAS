# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check supplied-phase Boolean control through emission and causal ID reuse."""
import json
from pathlib import Path
import sys
import tempfile

from check_deferred_family_output import executable, invoke
from check_finite_allocation import physical_check
from check_repeated_region import check, run


def phase_source(original, period, mixed):
    source = original.replace("count=2", f"count={period}")
    source = source.replace("%two = arith.constant 2", f"%two = arith.constant {period}")
    source = source.replace("arith.remui %i, %two", "arith.remui %visit, %two")
    guard = """      %phase = arith.remui %visit, %two : index
      %periodic = arith.cmpi eq, %phase, %zero : index
"""
    if mixed:
        guard += """      %before = arith.cmpi slt, %visit, %two : index
      %active = arith.andi %periodic, %before : i1
"""
    else:
        # Exercise Boolean composition and select, not just a modulo comparison.
        guard += """      %false = arith.constant false
      %active = arith.select %periodic, %periodic, %false : i1
"""
    source = source.replace("      scf.for %i", guard + "      scf.if %active {\n      scf.for %i")
    source = source.replace("    }\n    pto.textract", "      }\n    }\n    pto.textract")
    return source


def main():
    tool, optimizer, fixture = sys.argv[1:]
    tool, optimizer = map(executable, (tool, optimizer))
    original = Path(fixture).read_text()
    tested = 0
    physical = 0
    combined_checks = 0
    with tempfile.TemporaryDirectory(prefix="phase-control-") as scratch:
        path = Path(scratch) / "case.pto"
        for period in (2, 3):
            for mixed in (False, True):
                source = phase_source(original, period, mixed)
                # Irrelevant narrow arithmetic must not become an extra
                # qualification premise merely because it is outer-periodic.
                source = source.replace("      %active =", "      %narrow = arith.index_cast %phase : index to i32\n"
                    "      %narrow_zero = arith.constant 0 : i32\n"
                    "      %unused = arith.cmpi eq, %narrow, %narrow_zero : i32\n      %active =")
                path.write_text(source)
                report = json.loads(run(tool, "--sequence-analysis", path))
                assert not report["error"] and report["prepared"], report
                assert report["numeric_visits"] == 0 and report["relational_compositions"] == 0, report
                assert report["phase_descriptions"] >= period, report
                small = run(tool, "--insert-logical", path)
                path.write_text(source.replace("array<i64: 2, 3>", "array<i64: 1000000000, 1000000001>"))
                large = run(tool, "--insert-logical", path)
                assert small == large.replace("1000000000, 1000000001", "2, 3")
                command = [optimizer, "--mlir-disable-threading"]
                logical = invoke([*command, "--pto-frontier-analysis", "-"], source).stdout
                combined = invoke([*command,
                    "--pto-frontier-analysis=physical-output=true allocation-ids=0,1,2,3,4,5", "-"], source).stdout
                assert "pto.logical_" not in combined
                for trips, inner in ((0, 2), (1, 0), (1, 1), (period, 2), (period + 2, 2)):
                    program = source.replace("array<i64: 2, 3>", f"array<i64: {trips}, {inner}>")
                    program = program.replace("test.trace_arguments =",
                        "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
                    path.write_text(program)
                    trace = json.loads(run(tool, "--structured-trace", path))
                    visits = [(visit, i) for visit in range(trips) for i in range(inner)
                              if visit % period == 0 and (not mixed or visit < period)]
                    check(trace, [trips, inner], outer_slots=period, visits=visits)
                    tested += 1
                    physical_check(trace, set(range(6)))
                    physical += 1
                    emitted = []
                    for text in (logical, combined):
                        bound = text.replace("test.trace_arguments = array<i64: 2, 3>",
                            f"test.trace_existing, test.trace_arguments = array<i64: {trips}, {inner}>")
                        emitted.append(json.loads(invoke([tool, "--structured-trace", "-"], bound).stdout))
                    check(emitted[0], [trips, inner], outer_slots=period, visits=visits)
                    physical_check({**emitted[0], "allocated": True, "physical": emitted[1]["trace"]}, set(range(6)))
                    combined_checks += 1
        # Outer-periodic does not imply constant during an inner visit. Keep
        # j<2 as an inner slice, for both outer-phase polarities.
        for inverted in (False, True):
            source = original.replace("arith.remui %i, %two", "arith.remui %visit, %two")
            # Keep same-pipe hazards adjacent for the order-equality oracle.
            # External bank-zero writes surrounding a bank-one visit would
            # instead require the documented nonadjacent-barrier qualification.
            source = source.replace(
                "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)\n", "")
            predicate = "ne" if inverted else "eq"
            control = f"""      %phase = arith.remui %visit, %two : index
      %periodic = arith.cmpi {predicate}, %phase, %zero : index
      %inner_first = arith.cmpi slt, %i, %two : index
      %active = arith.andi %periodic, %inner_first : i1
      scf.if %active {{
"""
            source = source.replace("      %slot =", control + "      %slot =")
            source = source.replace("      }\n    }", "      }\n      }\n    }")
            for trips, inner in ((0, 4), (3, 0), (3, 1), (3, 4)):
                program = source.replace("array<i64: 2, 3>", f"array<i64: {trips}, {inner}>")
                path.write_text(program)
                trace = json.loads(run(tool, "--structured-trace", path))
                visits = [(visit, i) for visit in range(trips) for i in range(inner)
                          if (visit % 2 != 0) == inverted and i < 2]
                check(trace, [trips, inner], outer_slots=2, visits=visits, external_work=False)
                tested += 1
    print(f"phase control: {tested} independent event closures, {physical} causal allocation checks, "
          f"{combined_checks} combined-production traces")


if __name__ == "__main__":
    main()
