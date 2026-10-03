# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare the complete pre-dispatch shared SyncIR dumps from both constructors."""

import argparse
from pathlib import Path


def shared_dump(path):
    """Extract one shared dump, excluding later constructor diagnostics."""
    text = path.read_text(encoding="utf-8")
    marker = "// === [PTOInsertSync Debug] After Shared Translator === //"
    begin = text.index(marker)
    end = text.index("// ========================================= //", begin)
    return text[begin:end]


def main():
    """Require byte-identical phase, pipe, effect and memory records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("existing", type=Path)
    parser.add_argument("frontier", type=Path)
    args = parser.parse_args()
    if shared_dump(args.existing.resolve()) != shared_dump(args.frontier.resolve()):
        raise ValueError("constructors received different shared translation")
    print("identical shared phases, pipes, effects and memory")


if __name__ == "__main__":
    main()
