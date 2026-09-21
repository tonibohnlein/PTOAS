#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify the overnight bundle and its explicit source-tree manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe manifest path: " + name)
    return root / relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    root = args.package.resolve()
    entries = (root / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        expected, name = line.split(maxsplit=1)
        path = safe(root, name.lstrip("*"))
        if digest(path) != expected:
            raise ValueError("checksum mismatch: " + name)
    manifest = json.loads((root / "manifest.json").read_text())
    for case in manifest["cases"]:
        assert digest(safe(root, case["input"])) == case["sha256"], case["id"]
        if "handoff_pin" in case:
            assert digest(safe(root, case["handoff_pin"])) == case["handoff_pin_sha256"], case["id"]
    if args.source_dir:
        rows = json.loads((root / "sources/compiler-tree.json").read_text())
        assert isinstance(rows, list), "source manifest must be a list"
        for row in rows:
            path = safe(args.source_dir, row["path"])
            mode = stat.S_IMODE(path.lstat().st_mode)
            assert mode == row["mode"], (row["path"], mode, row["mode"])
            assert path.is_symlink() == (row["type"] == "symlink"), row["path"]
            actual = hashlib.sha256(os.readlink(path).encode()).hexdigest() if path.is_symlink() else digest(path)
            assert actual == row["sha256"], row["path"]
        print("Verified source entries and exact modes:", len(rows))
    print("Verified package entries:", len(entries), "modules:", len(manifest["cases"]))


if __name__ == "__main__":
    main()
