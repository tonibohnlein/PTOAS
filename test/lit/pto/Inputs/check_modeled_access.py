# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare modeled demand reduction with fixture-defined alias relationships."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_explicit_reduction import occurrence, closure, require


def check(tool, path, order, modes):
    result = subprocess.run([tool, "--explicit-analysis", str(path)], check=True,
                            capture_output=True, text=True, timeout=30)
    report = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith("{"))
    require(not report["error"], report)
    # Independent model: U has an unresolved address; A and B have distinct
    # known ranges. Neither emitted cell IDs nor exported generators define it.
    edges = {(a, b) for a in range(3) for b in range(a + 1, 3)
             if (order[a] == "U" or order[b] == "U") and
             (modes[order[a]] or modes[order[b]])}
    occurrences = [occurrence(i, item["pipe"], []) for i, item in enumerate(report["occurrences"])]
    reach = closure(occurrences, edges)
    for source, row in enumerate(report["event_reachable"]):
        require(row == [source == target or target in reach[source] for target in range(6)], report)
    covers = {(a, b) for a, b in edges if not any(2 * b in reach[z] for z in reach[2 * a + 1])}
    require(covers == set(map(tuple, report["retained"])), report)


def main():
    tool = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="modeled-access-") as directory:
        path = Path(directory) / "input.pto"
        count = 0
        for order in itertools.permutations("UAB"):
            for flags in itertools.product([False, True], repeat=3):
                modes = dict(zip("UAB", flags))
                ops = []
                for name in order:
                    if modes[name]:
                        ops.append(f"pto.tsetval ins(%zero, %value : index, f32) outs(%{name} : !tile)")
                    else:
                        ops.append(f"%r{name} = pto.tgetval ins(%{name}, %zero : !tile, index) outs : f32")
                path.write_text("""!tile = !pto.tile_buf<vec, 4x8xf32>
module attributes {pto.target_arch = "a3"} {
  func.func @modeled(%address: i64) {
    %zero = arith.constant 0 : index
    %value = arith.constant 1.0 : f32
    %x = arith.constant 1024 : i64
    %y = arith.constant 2048 : i64
    %U = pto.alloc_tile addr = %address : !tile
    %A = pto.alloc_tile addr = %x : !tile
    %B = pto.alloc_tile addr = %y : !tile
""" + "\n".join(ops) + "\n return\n }\n}\n")
                check(tool, path, order, modes)
                count += 1
                if modes["A"]:
                    cross = path.read_text().replace(
                        "pto.tsetval ins(%zero, %value : index, f32) outs(%A : !tile)",
                        "pto.tadds ins(%A, %value : !tile, f32) outs(%A : !tile)")
                    path.write_text(cross)
                    check(tool, path, order, modes)
                    count += 1
    print(f"modeled access: {count} independent alias/order checks passed")


if __name__ == "__main__":
    main()
