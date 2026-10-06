# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise shared ordinary footprints through exact analysis and insertion."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def run(tool, path, mode, alias="may-not-alias"):
    return subprocess.run([tool, "--gm-alias=" + alias, mode, str(path)],
                          capture_output=True, text=True, timeout=30)


def document(output):
    lines = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    require(len(lines) == 1, output)
    return lines[0]


def main():
    tool, fixture = sys.argv[1:]
    original = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="ordinary-access-") as directory:
        path = Path(directory) / "case.pto"
        path.write_text(original)
        for alias, accepted in [("may-not-alias", True), ("may-alias", True)]:
            analysis = run(tool, path, "--explicit-analysis", alias)
            require(analysis.returncode == 0, analysis.stderr)
            doc = document(analysis.stdout)
            require(bool(doc["error"]) != accepted, doc)
            inserted = run(tool, path, "--insert-logical", alias)
            require((inserted.returncode == 0) == accepted, inserted.stderr)
            if accepted:
                require(doc["retained"] == [[0, 1], [1, 2]], doc)
                require(inserted.stdout.count("pto.logical_set") == 2, inserted.stdout)
                require(inserted.stdout.count("pto.logical_wait") == 2, inserted.stdout)
        # Effect discharge must not hide a prerequisite outside the payload span.
        path.write_text(original.replace("    %base =", "    pto.barrier #pto.pipe<PIPE_ALL>\n    %base ="))
        require(run(tool, path, "--insert-logical").returncode != 0, "existing barrier accepted")
        # The same base's read/write aliases remain exact under either policy.
        same = original.replace("pto.make_tensor_view %output", "pto.make_tensor_view %input")
        path.write_text(same)
        for alias in ["may-alias", "may-not-alias"]:
            doc = document(run(tool, path, "--explicit-analysis", alias).stdout)
            require(not doc["error"] and doc["retained"] == [[0, 1], [1, 2]], doc)
        # A conditional transfer with a live outside store needs a guarded
        # handoff and based-GM storage, not an independent-GM discharge.
        guarded = same.replace("%scalar: f32)", "%scalar: f32, %guard: i1)")
        guarded = guarded.replace("    pto.tload", "    scf.if %guard {\n      pto.tload")
        guarded = guarded.replace("    pto.tadds ins(%tile, %scalar : !tile, f32) outs(%tile : !tile)", "    }")
        path.write_text(guarded)
        inserted = run(tool, path, "--insert-logical", "may-alias")
        require(inserted.returncode == 0, inserted.stderr)
        require(inserted.stdout.count("pto.logical_set") == 1, inserted.stdout)
        require(inserted.stdout.count("pto.logical_wait") == 1, inserted.stdout)
        # Runtime metadata and vector tails retain their shared buffer bound when
        # a narrower map is unavailable. Mutable metadata is a separate structural limit.
        changed = original.replace("    pto.tadds", "    pto.set_validshape %tile, %one, %cols : !tile\n    pto.tadds")
        oversized = original.replace("2x64", "1024x4096").replace("constant 2 : index", "constant 1024 : index")
        oversized = oversized.replace("constant 64 : index", "constant 4096 : index")
        wide = original.replace("2x64", "2x16384").replace("constant 64 : index", "constant 16384 : index")
        tail = original.replace("partition_tensor_view<2x64", "partition_tensor_view<2x56")
        tail = tail.replace("constant 64 : index", "constant 56 : index")
        for text in [changed, tail, oversized, wide]:
            path.write_text(text)
            effects = run(tool, path, "--storage-effects")
            require(effects.returncode == 0 and "all-materialized=" in effects.stdout, effects.stderr + effects.stdout)
            inserted = run(tool, path, "--insert-logical")
            if text == changed:
                recognition = run(tool, path, "--recognize")
                require("unmodeled-operation" in recognition.stdout, recognition.stdout)
                require(inserted.returncode != 0, inserted.stdout)
            else:
                require(inserted.returncode == 0, inserted.stderr)
    print("ordinary access routes: exact, alias, guarded and metadata cases passed")


if __name__ == "__main__":
    main()
