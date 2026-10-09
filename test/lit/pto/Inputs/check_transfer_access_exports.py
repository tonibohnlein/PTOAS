# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check transfer footprints from independently computed byte addresses."""
import ast
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def evaluate(expression, values):
    node = ast.parse(expression.replace("floordiv", "//").replace("mod", "%"), mode="eval")

    def visit(part):
        if isinstance(part, ast.Expression):
            return visit(part.body)
        if isinstance(part, ast.Constant):
            return part.value
        if isinstance(part, ast.Name):
            return values[part.id]
        left, right = visit(part.left), visit(part.right)
        operations = {ast.Add: lambda: left + right, ast.Mult: lambda: left * right,
                      ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}
        return operations[type(part.op)]()

    return visit(node)


def source(arch, dtype, width, columns, dynamic, atomic=False, column_major=False):
    capacity = ((columns * width + 31) // 32) * 32 // width
    tensor = f"!pto.tensor_view<{'?x?' if dynamic else '32x64'}x{dtype}>"
    partition = f"!pto.partition_tensor_view<{'?x?' if dynamic else f'2x{columns}'}x{dtype}>"
    tile = f"!pto.tile_buf<vec, 2x{capacity}x{dtype}, valid=2x{columns}>"
    store = " {atomicType = #pto<atomic_type atomic_add>}" if atomic else ""
    text = f"""module attributes {{pto.target_arch = "{arch}", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @transfer(%src: !pto.ptr<{dtype}, gm>, %dst: !pto.ptr<{dtype}, gm>) {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %seven = arith.constant 7 : index
    %rows = arith.constant 32 : index
    %cols = arith.constant 64 : index
    %stride = arith.constant 96 : index
    %width = arith.constant {columns} : index
    %base = arith.constant 256 : i64
    %input = pto.make_tensor_view %src, shape = [%rows, %cols], strides = [%stride, %one] : {tensor}
    %output = pto.make_tensor_view %dst, shape = [%rows, %cols], strides = [%stride, %one] : {tensor}
    %tile = pto.alloc_tile addr = %base : {tile}
    scf.for %i = %one to %seven step %two {{
      %read = pto.partition_view %input, offsets = [%i, %one], sizes = [%two, %width] : {tensor} -> {partition}
      %write = pto.partition_view %output, offsets = [%i, %one], sizes = [%two, %width] : {tensor} -> {partition}
      pto.tload ins(%read : {partition}) outs(%tile : {tile})
      pto.tstore ins(%tile : {tile}) outs(%write : {partition}){store}
    }}
    return
  }}
}}"""
    if column_major:
        text = text.replace(tensor, f"!pto.tensor_view<{'?x?' if dynamic else '64x32'}x{dtype}>")
        text = text.replace(partition, f"!pto.partition_tensor_view<{'?x?' if dynamic else f'{columns}x2'}x{dtype}>")
        text = text.replace(tile, f"!pto.tile_buf<vec, {capacity}x2x{dtype}, valid={columns}x2, blayout=col_major>")
        text = text.replace("shape = [%rows, %cols], strides = [%stride, %one]",
                            "shape = [%cols, %rows], strides = [%one, %stride] {layout = #pto.layout<dn>}")
        text = text.replace("offsets = [%i, %one], sizes = [%two, %width]",
                            "offsets = [%one, %i], sizes = [%width, %two]")
    return text


def check(tool, path, arch, dtype, width, columns, dynamic, atomic=False, column_major=False):
    path.write_text(source(arch, dtype, width, columns, dynamic, atomic, column_major))
    result = subprocess.run([tool, "--step1-json", str(path)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert len(report["access_records"]) == (5 if atomic else 4), report
    for record in report["access_records"]:
        if atomic and record["operation"] == "pto.tstore":
            if record["space"] == 1:
                assert not record["regions"], record
            continue
        if record["space"] != 1:
            if columns * width % 32:
                assert record["declared_selections"] == 0, record
                assert record["regions"] or record["ranges"], record
            else:
                assert record["declared_selections"] == 1, record
            continue
        assert record["declared_selections"] == 1, record
        assert len(record["regions"]) == 1, record
        region = record["regions"][0]
        expected_extents = [str(columns), "2"] if column_major else ["2", str(columns)]
        assert region["extents"] == expected_extents, region
        assert region["element_bytes"] == width, region
        expected_root = 0 if record["mode"] == "read" else 1
        assert region["base"]["number"] == expected_root, region
        for iteration, row, col in itertools.product([1, 3, 5], [0, 1], [0, columns - 1]):
            coords = {"d0": col, "d1": row} if column_major else {"d0": row, "d1": col}
            actual = evaluate(region["offset"], {"s0": iteration, **coords})
            assert actual == ((iteration + row) * 96 + 1 + col) * width, (record, actual)


def check_single_column(tool, path, arch):
    # Both minor-layout interpretations are legal when the second extent is
    # one; the column-major consumer determines the contiguous burst axis.
    text = source(arch, "f32", 4, 16, False, column_major=True)
    text = text.replace(" {layout = #pto.layout<dn>}", "")
    text = text.replace("64x32xf32", "64x1xf32").replace("16x2xf32", "16x1xf32")
    text = text.replace("valid=16x2", "valid=16x1")
    text = text.replace("%rows = arith.constant 32", "%rows = arith.constant 1")
    text = text.replace("%two = arith.constant 2", "%two = arith.constant 1")
    text = text.replace("%stride = arith.constant 96", "%stride = arith.constant 1")
    text = text.replace("offsets = [%one, %i]", "offsets = [%one, %zero]")
    path.write_text(text)
    result = subprocess.run([tool, "--step1-json", str(path)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["unresolved_accesses"] == 0, report
    for record in report["access_records"]:
        assert record["declared_selections"] == 1, record
        if record["space"] != 1:
            continue
        region = record["regions"][0]
        assert region["extents"] == ["16", "1"], region
        for row in [0, 1, 15]:
            assert evaluate(region["offset"], {"d0": row, "d1": 0}) == (row + 1) * 4, region


def check_runtime_extent(tool, path):
    text = source("a3", "f32", 4, 3, True)
    text = text.replace("%dst: !pto.ptr<f32, gm>)", "%dst: !pto.ptr<f32, gm>, %n: index)")
    text = text.replace("%width = arith.constant 3 : index", "%width = arith.addi %n, %zero : index")
    path.write_text(text)
    result = subprocess.run([tool, "--step1-json", str(path)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert len(report["access_records"]) == 4, report
    for record in report["access_records"]:
        if record["space"] == 1:
            assert record["declared_selections"] == 0 and not record["regions"], record
            assert record["root"]["block_argument"], record


def main():
    total = 0
    with tempfile.TemporaryDirectory(prefix="transfer-access-") as directory:
        path = Path(directory) / "case.pto"
        for arch, element, columns, dynamic, column_major in itertools.product(
                ["a3", "a5"], [("i8", 1), ("f16", 2), ("f32", 4)],
                [1, 3, 7, 8, 9, 15, 16, 17], [False, True], [False, True]):
            dtype, width = element
            check(sys.argv[1], path, arch, dtype, width, columns, dynamic, column_major=column_major)
            total += 1
        check(sys.argv[1], path, "a3", "f32", 4, 3, True, atomic=True)
        check_runtime_extent(sys.argv[1], path)
        for arch in ["a3", "a5"]:
            check_single_column(sys.argv[1], path, arch)
    print(f"{total} generic transfer cases plus singleton-column, runtime-extent, and atomic-store checks passed")


if __name__ == "__main__":
    main()
