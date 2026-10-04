# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check extended arithmetic domains and shared byte maps against source executions."""
import itertools
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index



def trace(name, n):
    if name == "empty_rotating":
        return []
    if name == "normalized_view":
        return [(i,) for i in range(1, n + 1, 2)]
    if name == "affine_nested_bound":
        return [(i, j) for i in range(n) for j in range(i + 1)]
    return [(i,) for i in range(n)]


def byte_origin(name, coords):
    if name == "affine_nested_bound":
        return 0
    if name == "normalized_view":
        return 1024 * (coords[0] % 2) + 80
    if name == "swapped_slot":
        return 1024 * ((coords[0] + 1) % 2)
    return 4 * (coords[-1] + (1 if name == "shifted_access" else 0))


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    run = subprocess.run([tool, "--arithmetic", sys.argv[2]], check=True,
                         capture_output=True, text=True, timeout=45)
    documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in run.stdout.splitlines()
                 if line.startswith("arithmetic-json ")]
    assert len(documents) == 8
    checks = 0
    for document in documents:
        name = document["function"]
        if name in {"unsupported_period", "wrapped_bound", "narrowed_dynamic_access"}:
            assert not document["sites"] and not document["relations"], name
            continue
        expected_parameters = [] if name == "empty_rotating" else [0]
        assert len(document["sites"]) == 1 and document["parameters"] == expected_parameters, name
        contains = membership_index(document)
        candidates = list(itertools.product(range(-1, 5), repeat=document["sites"][0]["depth"]))
        for n in range(-1, 5):
            parameters = (n,) if expected_parameters else ()
            execution = trace(name, n)
            rank = {coords: i for i, coords in enumerate(execution)}
            for a in candidates:
                assert contains((1, 0, -1, 0, 0), a + parameters) == (a in rank), (name, a, n)
                origin = byte_origin(name, a)
                for byte in set([-1, 0, 4, 80, 1024, 1104, origin - 1, origin, origin + 3, origin + 4]):
                    actual = contains((4, 0, -1, 0, 0), a + (byte,) + parameters)
                    assert actual == (a in rank and origin <= byte < origin + 4), (name, "read", a, n, byte)
                    assert not contains((5, 0, -1, 0, 0), a + (byte,) + parameters), (name, "write")
                    checks += 2
                for b in candidates:
                    before = a in rank and b in rank and rank[a] < rank[b]
                    values = a + b + parameters
                    assert contains((2, 0, 0, 0, 0), values) == before, (name, "order", a, b, n)
                    for source, target in itertools.product((1, 2), repeat=2):
                        expected = (before and (source, target) != (2, 1)) or (a == b and a in rank and
                                                                                  (source, target) == (1, 2))
                        assert contains((3, 0, 0, source, target), values) == expected, (name, "native", a, b, n)
                        checks += 1
    print("extended arithmetic: 5 accepted, 3 rejected; semantic comparisons:", checks)


if __name__ == "__main__":
    main()
