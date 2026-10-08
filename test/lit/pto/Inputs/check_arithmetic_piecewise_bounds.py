# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare piecewise occurrence and native-order relations with executed loops."""
import itertools
import json
import shutil
import subprocess
import sys
from arithmetic_membership import membership_index


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    result = subprocess.run([tool, "--arithmetic", sys.argv[2]], check=True,
                            capture_output=True, text=True, timeout=60)
    documents = [json.loads(line.removeprefix("arithmetic-json "))
                 for line in result.stdout.splitlines() if line.startswith("arithmetic-json ")]
    assert len(documents) == 4
    count = 0
    for document in documents:
        name = document["function"]
        if name == "unsigned_unproved":
            assert not document["sites"] and not document["relations"]
            continue
        assert document["sites"] and document["relations"], name
        assert document["parameters"] == [0, 1], (name, document["parameters"])
        contains = membership_index(document)
        points = list(itertools.product(range(-1, 5), repeat=2))
        for n, k in itertools.product(range(-1, 5), repeat=2):
            executed = []
            for i in range(n):
                if name == "minimum":
                    js = range(min(i, k))
                else:
                    lower = max(i, k) if name == "maximum" else (i if i < k else k)
                    js = range(lower, 4, 2)
                executed.extend((i, j) for j in js)
            ranks = {point: rank for rank, point in enumerate(executed)}
            for a in points:
                assert contains((1, 0, -1, 0, 0), a + (n, k)) == (a in ranks), (name, n, k, a)
                for b in points:
                    expected = a in ranks and b in ranks and ranks[a] < ranks[b]
                    assert contains((2, 0, 0, 0, 0), a + b + (n, k)) == expected, (name, n, k, a, b)
                    count += 1
    print("piecewise bound comparisons:", count)


if __name__ == "__main__":
    main()
