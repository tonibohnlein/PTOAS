# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check real shared translation of handwritten nested-if compiler fixtures."""
import json
import shutil
import subprocess
import sys

from check_demand_analysis import expect
from check_guarded_analysis import check_case, guarded


def examples():
    """Expected effects are fixture facts, not a compiler extraction registry."""
    return {
        "exclusive_writers": guarded([1, 2, 1], [[], [], [0]], [[0], [0], []], [[1], [-1], []]),
        "conditional_relay": guarded([1, 2, 1], [[], [0], [0, 1]], [[0], [1], []],
                                     [[], [1], []], footprints=2),
        "conditional_adjacency": guarded([1, 1, 1], [[], [], [0]], [[0], [1], []],
                                         [[], [1], []], footprints=2),
        "nested": guarded([1, 2, 1, 1, 2], [[], [0], [0], [0], []], [[0], [], [0], [], [0]],
                          [[], [1, 2], [1, -2], [-1], []], atoms=2),
        "repeated_condition": guarded([1, 2, 2, 1], [[], [0], [], [0]], [[0], [], [0], []],
                                      [[], [1], [1, -1], []]),
        "partial_overlap": guarded([1, 2, 1], [[], [], [0]], [[0], [1], []],
                                   [[], [1], []], footprints=2, aliases=[[0, 1]]),
        "constant_branch": guarded([1, 2, 1], [[], [], [0]], [[0], [0], []],
                                   [[], [False], []], atoms=0),
        "empty_arms": guarded([], [], [], [], atoms=1, footprints=0),
    }


def main():
    """Extract stable JSON dumps and independently check every small context."""
    if len(sys.argv) != 3:
        raise ValueError("usage: check_guarded_mlir.py pto-frontier-analysis-test input.pto")
    executable = shutil.which(sys.argv[1])
    if executable is None:
        raise ValueError("analysis runner not found")
    completed = subprocess.run([executable, "--dump-guarded-demands", sys.argv[2]],
                               check=True, capture_output=True, text=True, timeout=60)
    found = {}
    name = None
    for line in completed.stdout.splitlines():
        if ": phases=" in line:
            name = line.split(": phases=", 1)[0]
        if line.strip().startswith("guarded="):
            found[name] = json.loads(line.strip()[len("guarded="):])
    expected = examples()
    expect(set(found), set(expected), "compiler guarded fixture names")
    visits = sum(check_case(expected[name], result) for name, result in found.items())
    expect("original-ir-unchanged; construction-not-run" in completed.stdout, True, "source IR immutability")
    print("verified {} guarded MLIR functions and {} branch valuations".format(len(found), visits))


if __name__ == "__main__":
    main()
