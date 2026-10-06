# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check canonical GM identities across finite cells and arithmetic export."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def run(tool, path, mode, policy):
    result = subprocess.run([tool, "--gm-alias=" + policy, mode, str(path)],
                            capture_output=True, text=True, timeout=30)
    require(result.returncode == 0, result.stderr)
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    original = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="based-gm-cells-") as directory:
        path = Path(directory) / "case.pto"
        for source, bases, cells in [("%b", {0, 1}, [[1, 0, 512], [1, 0, 512]]),
                                     ("%a", {0}, [[1, 0, 512]]),
                                     ("%alias", {0}, [[1, 0, 256], [1, 256, 512], [1, 512, 768]])]:
            path.write_text(original.replace("pto.make_tensor_view %b", "pto.make_tensor_view " + source))
            for policy in ["may-alias", "may-not-alias"]:
                output = run(tool, path, "--explicit-analysis", policy)
                documents = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
                require(len(documents) == 1, output)
                doc = documents[0]
                require(not doc["error"], doc)
                disjoint = source == "%b" and policy == "may-not-alias"
                require(doc["retained"] == ([] if disjoint else [[0, 1]]), doc)
                require([cell for cell in doc["cells"] if cell[0] == 1] == cells, doc)
                text = run(tool, path, "--arithmetic", policy)
                lines = [line.removeprefix("arithmetic-json ") for line in text.splitlines()
                         if line.startswith("arithmetic-json ")]
                require(len(lines) == 1, text)
                arithmetic = json.loads(lines[0])
                actual = {relation["base_argument"] for relation in arithmetic["relations"]
                          if relation["space"] == 1}
                require(actual == bases, arithmetic)
    print("based GM cells: canonical aliases, distinct roots and arithmetic bases passed")


if __name__ == "__main__":
    main()
