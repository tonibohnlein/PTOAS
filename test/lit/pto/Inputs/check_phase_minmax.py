# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check signed phase extrema with explicit 64/32-bit index layouts."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing phase executable")
    source = Path(sys.argv[2]).read_text()
    with tempfile.TemporaryDirectory(prefix="phase-minmax-") as directory:
        fixture = Path(directory) / "phase.pto"
        for width in (64, 32):
            variant = source.replace("module {", "module attributes {dlti.dl_spec = "
                                     f"#dlti.dl_spec<#dlti.dl_entry<index, {width} : i32>>" + "} {")
            if width == 32:
                variant = variant.replace("test.phase_minmax}", "test.phase_minmax, test.narrow_phase}")
            fixture.write_text(variant)
            result = subprocess.run([tool, "--certify-regions", str(fixture)], capture_output=True,
                                    text=True, timeout=60, check=False)
            assert result.returncode == 0, result.stderr
            assert "phase-minmax: signed-extrema cached narrow-layout-refused" in result.stdout
    print("phase min/max: signed values cached; 32-bit index refused")


if __name__ == "__main__":
    main()
