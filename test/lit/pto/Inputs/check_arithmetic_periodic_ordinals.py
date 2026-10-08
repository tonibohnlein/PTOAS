# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise original-IV to ordinal substitution on the corpus-derived load/store fixture."""
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    total = 0
    with tempfile.TemporaryDirectory(prefix="periodic-ordinals-") as directory:
        path = Path(directory) / "case.pto"
        # The fixture contains upper bounds 0..6. Includes gcd(step, period)>1,
        # negative origins, empty visits and
        # partial final periods. The C++ checker executes the emitted commands
        # and compares all adjacent conflict obligations and matching identities.
        for dynamic, origin, step in ((False, -3, 2), (False, 5, 3), (False, -4, 3),
                                      (True, -3, 2), (True, 2, 2), (True, 5, 3)):
            text = source.replace("scf.for %arg2 = %c0 to %c10 step %c1 {",
                f"%stride = arith.constant {step} : index\n"
                + ("" if dynamic else f"    %origin = arith.constant {origin} : index\n")
                + "    scf.for %arg2 = %origin to %c10 step %stride {")
            if dynamic:
                text = text.replace("%arg1: !pto.ptr<f32, gm>)", "%arg1: !pto.ptr<f32, gm>, %origin: index)")
                text = text.replace("array<i64: 0, 0>", f"array<i64: {origin}>")
            path.write_text(text)
            result = subprocess.run([tool, "--arithmetic-periodic-input-checks", str(path)],
                                    text=True, capture_output=True, check=False)
            if result.returncode:
                raise AssertionError((dynamic, origin, step, result.stdout, result.stderr))
            total += result.stdout.count("arithmetic periodic importer/continuation/paired endpoints passed")
    assert total == 42, total
    print(f"{total} counted-loop origin/step endpoint checks passed")


if __name__ == "__main__":
    main()
