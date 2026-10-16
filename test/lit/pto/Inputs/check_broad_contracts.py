# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check scoped certification fields and transactional manual-ID refusal."""
import json
from pathlib import Path
import re
import shutil
import sys

from check_frontier_public_pipeline import require, unchanged_failure


def certification(text):
    documents = [json.loads(line.removeprefix("certification-json "))
                 for line in text.splitlines() if line.startswith("certification-json ")]
    matches = [document for document in documents if document.get("function") == "native_counted"]
    require(len(matches) == 1, "expected one native_counted certification document")
    regions = [region for region in matches[0]["regions"] if region.get("node") == 0]
    require(len(regions) == 1, "expected one root certification region")
    root = regions[0]
    require(isinstance(root.get("class"), str) and re.fullmatch(r"[a-z-]+", root["class"]) is not None,
            "root class is missing or malformed")
    require(root.get("exact_demands") is True and root.get("status") == "recognized",
            "root lost recognized exact demands")
    # Retain the existing textual checks too; additional JSON fields may appear
    # between these keys, but cannot satisfy them from a different region.
    sys.stdout.write(text)


def manual_refusal(tool, fixture):
    executable = shutil.which(tool)
    require(executable is not None, "optimizer executable unavailable")
    executable = str(Path(executable).resolve(strict=True))
    program = Path(fixture).resolve(strict=True).read_text()
    result = unchanged_failure(executable, program,
                               ["--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"])
    require("physical allocation cannot share its supplied IDs with existing synchronization" in result.stderr,
            "missing precise manual synchronization refusal")
    print("manual-ID refusal: source unchanged")


def main():
    if sys.argv[1:] == ["certification"]:
        certification(sys.stdin.read())
    elif len(sys.argv) == 4 and sys.argv[1] == "manual-refusal":
        manual_refusal(sys.argv[2], sys.argv[3])
    else:
        raise RuntimeError("expected certification or manual-refusal TOOL FIXTURE")


if __name__ == "__main__":
    main()
