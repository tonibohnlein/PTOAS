# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check target-qualified scalar hazards against independent all-pair graphs."""
import random
import sys
from check_explicit_reduction import access, occurrence, conflicts, edge_positions, closure, check, run
import check_rotating_extraction as rotating


def main():
    tool = sys.argv[1]
    rng = random.Random(91837)
    cases = []
    for _ in range(300):
        items = [occurrence(i, rng.randrange(3),
                            [access(cell, bool(mode & 1), bool(mode & 2))
                             for cell in range(3) if (mode := rng.randrange(4))])
                 for i in range(rng.randrange(1, 13))]
        cases.append({"scan": items, "scalar_pipe": 0, "reduce": True})
    # An explicit completion prerequisite is not a storage hazard. Its source
    # and target remain ordered even when their memory interaction is protected.
    items = [occurrence(0, 0, [access(0, write=True)]),
             occurrence(1, 0, [access(0, read=True)])]
    cases.extend([{"scan": items, "reduce": True},
                  {"scan": items, "scalar_pipe": 0, "reduce": True},
                  {"scan": items, "scalar_pipe": 0, "reduce": True,
                   "prerequisites": [[0, 1, 0]]}])
    for case, result in zip(cases, run(tool, cases)):
        assert not result["error"], result
        items = case["scan"]
        scalar = case.get("scalar_pipe")
        expected = {(a, b) for a, b in conflicts(items)
                    if scalar is None or items[a]["pipe"] != scalar or items[b]["pipe"] != scalar}
        expected |= edge_positions(case.get("prerequisites", []), items)
        actual = edge_positions(result["generators"], items)
        assert closure(items, actual) == closure(items, expected), (case, result, expected)
        oracle = dict(case, reduction_records=[[items[a]["id"], items[b]["id"], 0]
                                               for a, b in expected])
        check(oracle, result)
    # Explicit policy selection must not change the interpretation of arbitrary
    # numerical pipe labels in the standalone, target-neutral algorithms.
    patterns = [dict(case, scalar_pipe=0) for case in rotating.requests()]
    for case, result in zip(patterns, rotating.run(tool, patterns)):
        rotating.check(case, result)
    print(f"scalar protection: {len(cases)} scans and {len(patterns)} rotating prefixes passed")


if __name__ == "__main__":
    main()
