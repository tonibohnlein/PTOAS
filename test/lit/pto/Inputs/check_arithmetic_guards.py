# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check arithmetic branch domains against direct control-flow executions."""
import itertools
import json
import shutil
import subprocess
import sys
from arithmetic_membership import membership_index


def trace(name, parameters):
    if name == "repeated_boolean_dag":
        return [(0 if i != 1 else 1, (i,)) for i in range(parameters[0])]
    if name == "boolean_context":
        flag, = parameters
        return ([(0 if flag else 1, ()), (2, ())] if flag in (0, 1) else [])
    if name == "boolean_nested":
        n, flag = parameters
        if flag not in (0, 1):
            return []
        result = []
        for i in range(n):
            if not (1 <= i <= 2 or flag):
                result.append((0, (i,)))
            elif i != 1:
                result.append((1, (i,)))
        return result
    if name == "guarded_reset":
        n, k = parameters
        return [(0 if i > 0 else 1, (t, i)) for t in range(n) for i in range(k)]
    n, = parameters
    if name == "empty_arm":
        return [(0, (i,)) for i in range(n) if i != 0]
    result = []
    for i in range(n):
        if i == 0:
            result.append((0, (i,)))
        result.append((1, (i,)))
        if i + 1 < n:
            result.append((2, (i,)))
    return result


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    run = subprocess.run([tool, "--arithmetic", sys.argv[2]], check=True,
                         capture_output=True, text=True, timeout=60)
    documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in run.stdout.splitlines()
                 if line.startswith("arithmetic-json ")]
    assert len(documents) == 9
    contracts = {"first_last_prefetch": ([1, 1, 1], [0], {0, 2}),
                 "boolean_nested": ([1, 1], [0, 1], {0}),
                 "guarded_reset": ([2, 2], [0, 1], {1}),
                 "boolean_context": ([0, 0, 0], [0], {0}),
                 "empty_arm": ([1], [0], set()),
                 "repeated_boolean_dag": ([1, 1], [0], {1})}
    checks = 0
    for document in documents:
        name = document["function"]
        if name not in contracts:
            assert not document["sites"] and not document["relations"], name
            continue
        depths, parameter_ids, writes = contracts[name]
        assert [site["depth"] for site in document["sites"]] == depths, name
        assert document["parameters"] == parameter_ids, name
        contains = membership_index(document)
        candidates = [(site, coords) for site, depth in enumerate(depths)
                      for coords in itertools.product(range(-1, 4), repeat=depth)]
        for parameters in itertools.product(range(-1, 4), repeat=len(parameter_ids)):
            execution = trace(name, parameters)
            ranks = {occurrence: rank for rank, occurrence in enumerate(execution)}
            for a in candidates:
                site, coords = a
                assert contains((1, site, -1, 0, 0), coords + parameters) == (a in ranks), (name, a, parameters)
                for kind in (4, 5):
                    for byte in (-1, 0, 3, 4):
                        expected = a in ranks and (kind == 5) == (site in writes) and 0 <= byte < 4
                        actual = contains((kind, site, -1, 0, 0), coords + (byte,) + parameters)
                        assert actual == expected, (name, "access", a, parameters, kind, byte)
                        checks += 1
                for b in candidates:
                    before = a in ranks and b in ranks and ranks[a] < ranks[b]
                    values = coords + b[1] + parameters
                    assert contains((2, site, b[0], 0, 0), values) == before, (name, "order", a, b, parameters)
                    for source, target in itertools.product((1, 2), repeat=2):
                        expected = (before and (source, target) != (2, 1)) or (
                            a == b and a in ranks and (source, target) == (1, 2))
                        actual = contains((3, site, b[0], source, target), values)
                        assert actual == expected, (name, "native", a, b, parameters, source, target)
                        checks += 1
    print("arithmetic guards: 6 accepted, 3 rejected; semantic comparisons:", checks)


if __name__ == "__main__":
    main()
