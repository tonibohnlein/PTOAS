# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise public synchronization passes, route selection, and transactional failure."""
from pathlib import Path
import shutil
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(optimizer, program, flags, success=True):
    result = subprocess.run([optimizer, *flags, "-"], input=program, capture_output=True,
                            text=True, check=False, timeout=60)
    require((result.returncode == 0) == success, result.stdout + result.stderr)
    return result


def unchanged_failure(optimizer, program, flags):
    baseline = run(optimizer, program, []).stdout
    result = run(optimizer, program, [*flags, "--mlir-disable-threading", "--mlir-print-ir-after-failure",
                                     "--mlir-print-ir-module-scope"], False)
    marker = "// -----// IR Dump After "
    require(marker in result.stderr, result.stderr)
    dump = result.stderr.rsplit(marker, 1)[1].split("\n", 1)[1]
    restored = run(optimizer, dump, []).stdout
    require(restored == baseline, "failed public pass changed original IR")
    return result


def main():
    optimizer = shutil.which(sys.argv[1])
    require(optimizer is not None, "optimizer executable is unavailable")
    fixtures = Path(sys.argv[2]).resolve()
    allocate = "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"
    cases = (
        ("sync_analysis_session.pto", "explicit"),
        ("sync_finite_guarded_analysis.pto", "sequence"),
        ("sync_physical_allocation.pto", "numerical-periodic"),
        ("sync_bounded_allocation_session.pto", "bounded-lifetime"),
        ("sync_general_arithmetic_insertion.pto", "identity-sequence"),
    )
    for filename, backend in cases:
        program = (fixtures / filename).read_text(encoding="utf-8")
        logical = run(optimizer, program, ["--pto-frontier-analysis"]).stdout
        require('selected_logical_backend = "' + backend + '"' in logical, logical)
        if filename == "sync_general_arithmetic_insertion.pto":
            require("arithmetic_region_constructions = 1 : i64" in logical, logical)
        require("pto.logical_set" in logical and "pto.logical_wait" in logical, logical)
        physical = run(optimizer, logical, [allocate]).stdout
        require("pto.logical_" not in physical, physical)
        require("pto.set_flag" in physical and "pto.wait_flag" in physical, physical)
        require("pto.finite_allocation" not in physical and "pto.cyclic_allocation" not in physical, physical)

    finite = (fixtures / "sync_finite_guarded_analysis.pto").read_text(encoding="utf-8")
    checkpoint = unchanged_failure(optimizer, finite, ["--pto-frontier-analysis=analysis-only=true"])
    require("emission and allocation were not requested" in checkpoint.stderr, checkpoint.stderr)
    logical = run(optimizer, finite, ["--pto-frontier-analysis"]).stdout
    capacity = unchanged_failure(optimizer, logical, ["--pto-frontier-allocate"])
    require("allocation" in capacity.stderr, capacity.stderr)

    guarded = (fixtures / "sync_loop_session_cache.pto").read_text(encoding="utf-8")
    retained = run(optimizer, guarded, ["--pto-frontier-analysis"]).stdout
    require('selected_logical_backend = "guarded-rotating"' in retained, retained)
    require("pto.logical_set" in retained and "pto.logical_wait" in retained, retained)
    unsupported = unchanged_failure(optimizer, retained, [allocate])
    require("requires a supported finite or uniform allocation export" in unsupported.stderr,
            unsupported.stderr)

    broken = finite[finite.index("  func.func "):finite.rfind("\n}")]
    broken = broken.replace("@guarded", "@broken")
    broken = broken.replace("// BODY", "scf.while : () -> () { scf.condition(%g) } do {")
    broken = broken.replace("// END_BODY", "scf.yield\n    }")
    combined = finite[:finite.rfind("\n}")] + "\n" + broken + "\n}\n"
    unchanged_failure(optimizer, combined, ["--pto-frontier-analysis"])

    sections = (fixtures / "frontier_sections.pto").read_text(encoding="utf-8")
    profile = "--pto-frontier-analysis=arithmetic-profile=8:9:3:4096"
    profiled = run(optimizer, sections, [profile, allocate]).stdout
    require("pto.logical_" not in profiled and "pto.wait_flag" in profiled, profiled)
    arithmetic = (fixtures / "sync_general_arithmetic_insertion.pto").read_text(encoding="utf-8")
    declared = run(optimizer, arithmetic, [profile]).stdout
    require("period = 3 : i64" in declared and "dimensions = 9 : i64" in declared, declared)
    invalid = unchanged_failure(optimizer, sections, ["--pto-frontier-analysis=arithmetic-profile=8:0:3:4096"])
    require("arithmetic-profile must be positive" in invalid.stderr, invalid.stderr)
    print("public Frontier pipeline: five route fixtures, physical allocation, profiles and rollback passed")


if __name__ == "__main__":
    main()
