#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
"""Run the two main-project step-10 targets; missing binaries are failures."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
build = Path(os.environ.get("PTOAS_BUILD_DIR", str(root / "build"))).resolve()
for name in ("pto-frontier-exact-core-test", "pto-frontier-exact-test"):
    candidates = (build / "bin" / name, build / "tools" / "pto-test-opt" / name)
    binary = next((p for p in candidates if p.is_file() and os.access(p, os.X_OK)), None)
    if binary is None:
        raise SystemExit(f"Missing {name}. Rebuild the step-10 targets in {build} before running this test.")
    subprocess.run([str(binary)], check=True)
