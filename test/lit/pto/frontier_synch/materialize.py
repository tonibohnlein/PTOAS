#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Activate the reference synchronization annotations without changing payload.

HAND records local source flags; CROSS records the external C/V protocol;
LOWERING records conservative A3 vector barriers required by this PTO port.
The payload files remain the input to automatic synchronization and Phase A.
"""

import argparse
from pathlib import Path
import sys


def materialize(source: str) -> str:
    """Return manual-reference PTO, retaining all unannotated lines verbatim."""
    lines = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        for marker in ("// HAND: ", "// CROSS: ", "// LOWERING: "):
            if stripped.startswith(marker):
                line = line[:len(line) - len(stripped)] + stripped[len(marker):]
                break
        lines.append(line)
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    source = args.input.resolve().read_text(encoding="utf-8")
    sys.stdout.write(materialize(source))


if __name__ == "__main__":
    main()
