# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check scalar SSA ordering without dropping asynchronous storage hazards."""
import json
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    ("direct", "sum"): ({("x", True, True), ("y", True, True)}, False),
    ("direct", "fanout"): ({("x", True, True)}, False),
    ("conditional_result", "merged"): ({("x", True, False)}, False),
    ("mixed_control", "mixed"): ({("x", True, False)}, False),
    ("mixed_result", "mixed"): ({("x", True, False)}, False),
    ("native_bound", "bound"): ({("x", True, False)}, False),
    ("native_bound", "loop"): ({("x", True, False)}, False),
    ("carried", "carried"): (set(), True),
    ("carried", "result"): (set(), False),
}

def main():
    binary, fixture = sys.argv[1:]
    original = Path(fixture).read_bytes()
    for policy in ["may-not-alias", "may-alias"]:
        result = subprocess.run(
            [binary, "--gm-alias=" + policy, "--phase-index", fixture],
            capture_output=True, text=True, timeout=30, check=True,
        )
        actual = {}
        for line in result.stdout.splitlines():
            if not line.startswith("prerequisite-json "):
                continue
            row = json.loads(line.split(" ", 1)[1])
            key = (row["function"], row["consumer"])
            assert key not in actual, key
            actual[key] = ({(e["source"], e["native"], e["direct_ssa"])
                            for e in row["edges"]}, row["mapping_obligation"])
        assert actual == EXPECTED, (policy, actual, EXPECTED)
        assert Path(fixture).read_bytes() == original
    print("scalar prerequisites: 18 source-derived provenance checks passed")

if __name__ == "__main__":
    main()
