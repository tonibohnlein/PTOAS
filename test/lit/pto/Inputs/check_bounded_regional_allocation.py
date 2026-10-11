# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check bounded child allocation with finite tails and original loop ordinals."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_finite_allocation import physical_check


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    marker = "test.require_bounded_regional_allocation"
    attributes = (marker + ", "
                  "test.trace_arguments = array<i64: 0>, "
                  "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>")
    source = source.replace(marker, attributes)
    domain = "scf.for %arg4 = %c0 to %c8 step %c1"
    cases = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1),
             (0, 5, 1), (0, 8, 1), (1, 8, 1), (1, 8, 2),
             (2, 8, 3), (3, 3, 2), (4, 2, 1)]
    with tempfile.TemporaryDirectory(prefix="bounded-regional-allocation-") as directory:
        path = Path(directory) / "input.pto"
        for lower, upper, step in cases:
            definitions = (f"%test_lower = arith.constant {lower} : index\n"
                           f"    %test_upper = arith.constant {upper} : index\n"
                           f"    %test_step = arith.constant {step} : index\n    ")
            text = source.replace(domain, definitions +
                                  "scf.for %arg4 = %test_lower to %test_upper step %test_step")
            path.write_text(text)
            run = subprocess.run([tool, "--gm-alias=may-not-alias", "--structured-trace", str(path)],
                                 capture_output=True, text=True, timeout=90)
            if run.returncode:
                raise RuntimeError(f"bounded allocation case {lower, upper, step}: {run.stderr}")
            report = json.loads(run.stdout)
            physical_check(report, set(range(6)))
    print(f"bounded regional allocation: {len(cases)} independent causal-reuse traces passed")


if __name__ == "__main__":
    main()
