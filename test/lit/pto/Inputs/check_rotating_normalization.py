# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare ordinal slot patterns with direct loop iteration, without using the matcher."""
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    source = Path(sys.argv[2]).read_text()
    aliases = source[source.index("!tile ="):source.index("module {")]
    start = source.index("  func.func @normalized_loop")
    end = source.index("  func.func @unknown_step", start)
    template = source[start:end]
    cases = {}
    functions = []
    for lower in range(4):
        for step in range(1, 5):
            name = f"ordinal_{lower}_{step}"
            cases[name] = (lower, step)
            functions.append(template.replace("normalized_loop", name)
                             .replace("%i1 = arith.constant 1", f"%i1 = arith.constant {lower}")
                             .replace("%i3 = arith.constant 3", f"%i3 = arith.constant {step}"))
    with tempfile.TemporaryDirectory(prefix="pto-rotation-check-") as directory:
        path = Path(directory) / "ordinals.pto"
        path.write_text(aliases + "module {\n" + "".join(functions) + "}\n")
        run = subprocess.run([tool, "--recognize", str(path)], check=True, capture_output=True,
                             text=True, timeout=45)
    blocks = re.split(r"^recognition ", run.stdout, flags=re.MULTILINE)[1:]
    assert len(blocks) == len(cases)
    checks = 0
    for block in blocks:
        name = block.splitlines()[0]
        lower, step = cases[name]
        assert "recognize rotating: applicable" in block, name
        match = re.search(r"rotation family=0 slots=2 stride=(\d+) offset=(\d+) refresh=(\d+) atom=\[0,4\)", block)
        assert match, name
        stride, offset, refresh = map(int, match.groups())
        assert refresh == 2 // math.gcd(2, step)
        for upper in range(10):
            for ordinal, iv in enumerate(range(lower, upper, step)):
                assert (stride * ordinal + offset) % 2 == iv % 2, (name, upper, ordinal)
                checks += 1
    print("rotating ordinal checks:", len(cases), "patterns;", checks, "occurrences")


if __name__ == "__main__":
    main()
