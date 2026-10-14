# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check repeated recipes through production analysis and independent closures."""
import json
from pathlib import Path
import re
import sys
import tempfile
from check_repeated_region import check, run
from check_frontier_stage_profile import run as compile_profile, split_report


def main():
    tool, optimizer, fixture = sys.argv[1:]
    original = Path(fixture).read_text()
    cases = ((0, 3, True), (1, 1, True), (2, 0, True), (2, 2, True),
             (3, 3, True), (2, 5, True), (2, 3, False))
    checked = 0
    with tempfile.TemporaryDirectory(prefix="repeated-crossing-families-") as scratch:
        path = Path(scratch) / "trace.pto"
        for surrounding in (False, True):
            for outer, inner, enabled in cases:
                source = original.replace("%m: index)", "%m: index, %enabled: i1)")
                source = source.replace("array<i64: 2, 3>", f"array<i64: {outer}, {inner}, {int(enabled)}>")
                source = source.replace("    %two =", f"    %fixed = arith.constant {inner} : index\n    %two =")
                source = source.replace("%i = %zero to %m", "%i = %zero to %fixed")
                source = source.replace("      scf.for %i", "      scf.if %enabled {\n      scf.for %i")
                source = source.replace("    }\n    pto.textract", "      }\n    }\n    pto.textract")
                if not surrounding:
                    source = source.replace(
                        "    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)\n", "")
                compiled = compile_profile(optimizer, source, False, True)
                if compiled.returncode:
                    raise RuntimeError(compiled.stderr)
                traced, count = re.subn(r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
                                        r"\1attributes {test.trace_existing, ", compiled.stdout, count=1)
                if count != 1:
                    raise RuntimeError("missing original invocation")
                path.write_text(traced)
                check(json.loads(run(tool, "--structured-trace", path)),
                      [outer if enabled else 0, inner], external_work=surrounding)
                _, control, _ = split_report(compiled.stderr)
                prepared = control[0]
                # Without external work the dispatcher chooses an unmigrated
                # guarded rotating leaf. This fixture with surrounding work
                # exercises the repeated producer and its retained crossings.
                if surrounding and inner == 3:
                    if prepared["endpoint_sites"] > 5 or prepared["recipe_sites"] < 7:
                        raise RuntimeError("repeated crossing recipes were not retained through composition")
                checked += 1
    print(f"repeated crossings: {checked} independent closures, zero trips, guards and delayed preparation passed")


if __name__ == "__main__":
    main()
