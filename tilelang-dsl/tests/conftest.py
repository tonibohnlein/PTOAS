# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Pytest bootstrap for the TileLang DSL test tree."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_tilelang_dsl_import_path() -> None:
    # Always prefer the in-tree Python sources so pytest exercises the current
    # workspace edits rather than stale build artifacts. Keep the build-tree
    # path as a fallback when the source package is unavailable.
    repo_root = Path(__file__).resolve().parents[2]
    source_path = repo_root / "tilelang-dsl" / "python"
    build_path = repo_root / "build" / "python"

    source_text = str(source_path)
    if source_path.exists():
        if source_text in sys.path:
            sys.path.remove(source_text)
        sys.path.insert(0, source_text)
        return

    build_text = str(build_path)
    if build_path.exists() and build_text not in sys.path:
        sys.path.insert(0, build_text)


_ensure_tilelang_dsl_import_path()
