# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check rank-one DMA footprints against original pointer-coordinate bytes."""
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from check_transfer_access_exports import evaluate


def source(arch, columns, stride=1, rows=1, layout="nd", column_major=False):
    tile = f"!pto.tile_buf<vec, {rows}x8xi32, valid=1x{columns}"
    tile += ">"
    if column_major:
        tile = f"!pto.tile_buf<vec, 8x1xi32, valid={columns}x1, blayout=col_major>"
    return f"""module attributes {{pto.target_arch = "{arch}", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @rank_one(%src: !pto.ptr<i32, gm>, %dst: !pto.ptr<i32, gm>, %offset: index, %extent: index) {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    %stride = arith.constant {stride} : index
    %columns = arith.constant {columns} : index
    %base = arith.constant 256 : i64
    %input = pto.make_tensor_view %src, shape = [%extent], strides = [%stride]
        {{layout = #pto.layout<{layout}>}} : !pto.tensor_view<?xi32>
    %output = pto.make_tensor_view %dst, shape = [%extent], strides = [%stride]
        {{layout = #pto.layout<{layout}>}} : !pto.tensor_view<?xi32>
    %tile = pto.alloc_tile addr = %base : {tile}
    scf.for %i = %zero to %four step %one {{
      %at = arith.addi %offset, %i : index
      %read = pto.partition_view %input, offsets = [%at], sizes = [%columns]
          : !pto.tensor_view<?xi32> -> !pto.partition_tensor_view<{columns}xi32>
      %write = pto.partition_view %output, offsets = [%at], sizes = [%columns]
          : !pto.tensor_view<?xi32> -> !pto.partition_tensor_view<{columns}xi32>
      pto.tload ins(%read : !pto.partition_tensor_view<{columns}xi32>) outs(%tile : {tile})
      pto.tstore ins(%tile : {tile}) outs(%write : !pto.partition_tensor_view<{columns}xi32>)
    }}
    return
  }}
}}"""


def check(tool, path, arch, columns, **options):
    path.write_text(source(arch, columns, **options), encoding="utf-8")
    result = subprocess.run([tool, "--step1-json", str(path)], capture_output=True,
                            text=True, check=False, timeout=10)
    if result.returncode:
        raise RuntimeError(f"{arch} columns={columns} {options}: {result.stderr}")
    report = json.loads(result.stdout)
    assert len(report["access_records"]) == 4, report
    supported = not options
    comparisons = 0
    for record in report["access_records"]:
        if record["space"] != 1:
            if supported and columns == 8:
                assert record["declared_selections"] == 1, record
            else:
                assert record["declared_selections"] == 0, record
            # The short local burst remains an allocation bound, not a claim
            # that exactly the first seven elements are touched locally.
            assert record["ranges"] or record["regions"], record
            if supported and columns == 7:
                assert record["allocation_bytes"] == 32, record
                assert [(part["begin"], part["end"]) for part in record["ranges"]] == [(256, 288)], record
            continue
        if not supported:
            assert record["declared_selections"] == 0 and not record["regions"], record
            continue
        assert record["declared_selections"] == 1 and len(record["regions"]) == 1, record
        region = record["regions"][0]
        assert region["extents"] == ["1", str(columns)], region
        assert region["element_bytes"] == 4, region
        assert region["base"]["number"] == (0 if record["mode"] == "read" else 1), region
        # The sum of the function parameter and original IV may remain one
        # symbol; assign that original scalar expression independently.
        for offset, visit, col in itertools.product([-19, 0, 13], range(4), range(columns)):
            actual = evaluate(region["offset"], {"d0": 0, "d1": col, "s0": offset + visit})
            assert actual == (offset + visit + col) * 4, (record, actual)
            comparisons += 1
    return comparisons


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise FileNotFoundError(sys.argv[1])
    comparisons = 0
    with tempfile.TemporaryDirectory(prefix="rank-one-transfer-") as directory:
        path = Path(directory) / "case.pto"
        for arch in ["a3", "a5"]:
            for columns in [7, 8]:
                comparisons += check(tool, path, arch, columns)
            for options in [{"stride": 2}, {"rows": 2}, {"layout": "dn"}, {"column_major": True}]:
                check(tool, path, arch, 8, **options)
    print(f"rank-one transfers: {comparisons} original-coordinate byte checks and eight refusal cases passed")


if __name__ == "__main__":
    main()
