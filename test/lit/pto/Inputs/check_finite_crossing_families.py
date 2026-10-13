# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check fixed-coordinate regional crossings against an unfolded effect graph."""
import json
import re
from pathlib import Path
import sys
import tempfile
from check_repeated_region import check, run
from check_frontier_stage_profile import run as compile_profile, split_report


def fixed_source(source, outer, inner):
    constants = (f"    %nend = arith.constant {outer} : index\n"
                 f"    %mend = arith.constant {inner} : index\n    %two =")
    return (source.replace("    %two =", constants)
            .replace("%zero to %n step", "%zero to %nend step")
            .replace("%zero to %m step", "%zero to %mend step"))


def main():
    tool, optimizer, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    cases = ((0, 3), (2, 0), (2, 2), (2, 3), (3, 4), (2, 5))
    with tempfile.TemporaryDirectory(prefix="finite-crossing-families-") as scratch:
        path = Path(scratch) / "case.pto"
        for outer, inner in cases:
            current = fixed_source(source, outer, inner)
            compiled = compile_profile(optimizer, current, False, True)
            if compiled.returncode:
                raise RuntimeError(compiled.stderr)
            traced, count = re.subn(r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
                                    r"\1attributes {test.trace_existing, ", compiled.stdout, count=1)
            if count != 1:
                raise RuntimeError("missing original invocation in compiled plan")
            path.write_text(traced)
            check(json.loads(run(tool, "--structured-trace", path)), [outer, inner])
        # Production preparation, not a hand-constructed family test: the
        # original payload and both loop boundaries remain in the compiled IR.
        _, control, _ = split_report(compiled.stderr)
        prepared = control[0]
        if prepared["stage"] != "prepared" or prepared["recipe_sites"] == 0:
            raise RuntimeError("finite regional families materialized before insertion")
    print(f"finite crossings: {len(cases)} independent closures and retained production recipes passed")


if __name__ == "__main__":
    main()
