# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent byte-membership checks for guarded tensor access origins."""
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing arithmetic executable")
    result = subprocess.run([tool, "--arithmetic", sys.argv[2]],
                            check=True, capture_output=True, text=True, timeout=60)
    documents = [json.loads(line.removeprefix("arithmetic-json "))
                 for line in result.stdout.splitlines() if line.startswith("arithmetic-json ")]
    assert len(documents) == 5
    checks = 0
    for document in documents:
        name = document["function"]
        if name == "unsupported":
            assert not document["relations"], "unsupported select branch was discarded"
            continue
        assert document["sites"] and document["relations"], name
        assert not document["parameters"], document["parameters"]
        contains = membership_index(document)
        for i in range(-3, 6):
            if name == "maximum":
                origin = max(i, 0)
            elif name == "nested":
                origin = max(min(i, 1), 0)
            else:
                origin = min(i, 0)
            begin = origin * 8
            for byte in range(-40, 73):
                expected = -2 <= i < 4 and begin <= byte < begin + 32
                actual = contains((5, 0, -1, 0, 0), (i, byte))
                assert actual == expected, (name, i, byte)
                checks += 1
    print("piecewise access byte comparisons:", checks)


if __name__ == "__main__":
    main()
