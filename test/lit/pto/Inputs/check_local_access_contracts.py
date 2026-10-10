# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare local instruction contracts against target-specific address oracles."""
import ast
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def offset(expression, row, col):
    tree = ast.parse(expression, mode="eval")

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return {"d0": row, "d1": col}[node.id]
        left, right = evaluate(node.left), evaluate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise AssertionError(expression)

    return evaluate(tree)


def fixture(arch, op, dtype, columns, changed=False, reduction=False, destination_type=None):
    other = "f16" if dtype == "f32" else "f32"
    output_type = destination_type or (other if op == "tcvt" else dtype)
    rows = 1 if reduction else 2
    out_columns = 1 if reduction else columns
    source = f"!pto.tile_buf<vec, {rows}x256x{dtype}, valid={rows}x{columns}>"
    if changed:
        source = f"!pto.tile_buf<vec, {rows}x256x{dtype}, valid=?x?>"
    initial = " valid_row = %rows valid_col = %initial" if changed else ""
    target = f"!pto.tile_buf<vec, {rows}x256x{output_type}, valid={rows}x{out_columns}>"
    binary = op in ["tadd", "tsub", "tmul"]
    scalar = op in ["tadds", "tmuls"]
    inputs, types = ["%a"], [source]
    if binary or reduction:
        inputs.append("%b")
        types.append(source)
    if scalar:
        inputs.append("%scalar")
        types.append(dtype)
    mutation = f"pto.set_validshape %a, %rows, %wrong : {source}" if changed else ""
    return f"""module attributes {{pto.target_arch = "{arch}"}} {{
  func.func @local_access() {{
    %zero = arith.constant 0 : i64
    %second = arith.constant 8192 : i64
    %third = arith.constant 16384 : i64
    %rows = arith.constant {rows} : index
    %initial = arith.constant {columns} : index
    %wrong = arith.constant {max(1, columns - 1)} : index
    %scalar = arith.constant 1.0 : {dtype}
    %a = pto.alloc_tile addr = %zero{initial} : {source}
    %b = pto.alloc_tile addr = %second{initial} : {source}
    %c = pto.alloc_tile addr = %third : {target}
    {mutation}
    pto.{op} ins({', '.join(inputs)} : {', '.join(types)}) outs(%c : {target})
    return
  }}
}}""", output_type


def check(tool, path, arch, op, dtype, columns, changed=False, destination_type=None):
    reduction = op == "trowsum"
    text, output_type = fixture(arch, op, dtype, columns, changed, reduction, destination_type)
    path.write_text(text)
    run = subprocess.run([tool, "--step1-json", str(path)], capture_output=True,
                         text=True, timeout=30)
    assert run.returncode == 0, (arch, op, changed, run.stderr)
    report = json.loads(run.stdout)
    records = [e for e in report["access_records"] if e["operation"] == "pto." + op]
    assert len(records) == (4 if reduction and arch == "a3" else
                            3 if op in ["tadd", "tsub", "tmul"] else 2), report
    for record in records:
        descriptor = record["descriptor"]
        base = offset(descriptor["offset"], 0, 0)
        width = 2 if (output_type if base == 16384 else dtype) in ["f16", "bf16"] else 4
        logical_columns = 1 if reduction and base == 16384 else columns
        rows = 1 if reduction else 2
        row_bytes = columns * (2 if dtype in ["f16", "bf16"] else 4)
        if reduction:
            selected = arch == "a3" and row_bytes <= 256 and base != 8192
        elif op == "tcvt":
            selected = columns % 128 == 0
        elif op == "tmov" and arch == "a3":
            selected = row_bytes % 32 == 0
        else:
            selected = arch == "a3" or record["mode"] == "write" or row_bytes % 256 == 0
        # Runtime metadata can prevent export or invalidate a joint guard.
        # In either case, dependent selections must retain their bounds.
        if changed:
            selected = op == "tmul"  # Existing aligned contract, guarded at the access.
        exact = selected and not changed
        assert bool(record["declared_selections"]) == selected, (arch, op, record)
        region = record["regions"][0]
        if not exact:
            assert region["element_bytes"] == 1, (arch, op, record)
            assert region["extents"] == [str(rows * 256 * width)], record
            assert offset(region["offset"], 0, 0) == base, record
            continue
        assert region["extents"] == [str(rows), str(logical_columns)], record
        assert region["element_bytes"] == width, record
        for row, col in itertools.product(range(rows), [0, logical_columns - 1]):
            assert offset(region["offset"], row, col) == base + (row * 256 + col) * width, record


def check_padding_overlap(tool, path, arch, index):
    text, _ = fixture(arch, "tsqrt", "f32", 1)
    tile = "!pto.tile_buf<vec, 2x256xf32, valid=2x1>"
    consumer = f"""%index = arith.constant {index} : index
    %value = pto.tgetval ins(%c, %index : {tile}, index) outs : f32
    return"""
    text = text.replace("func.func @local_access()",
                        "func.func @local_access() attributes {test.overlap = array<i64: 1, 2>}")
    path.write_text(text.replace("return", consumer))
    run = subprocess.run([tool, "--storage-effects", str(path)], capture_output=True,
                         text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    # Producer result is record 1, scalar consumer is record 2. Adjacent
    # padding must be disjoint, while the first valid element still overlaps.
    assert f"overlap 1,2={int(index == 0)}" in run.stdout, run.stdout


def check_repeat_overflow(tool, path):
    text, _ = fixture("a5", "tmul", "f32", 2040)
    text = text.replace("2x256", "4095x2040").replace("valid=2x2040", "valid=4095x2040")
    path.write_text(text)
    run = subprocess.run([tool, "--step1-json", str(path)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert len(report["access_records"]) == 3, report
    assert all(e["declared_selections"] == 0 for e in report["access_records"]), report


def main():
    count = 0
    with tempfile.TemporaryDirectory(prefix="local-contract-") as directory:
        path = Path(directory) / "case.pto"
        for arch, dtype, columns, op in itertools.product(
                ["a3", "a5"], ["f16", "f32"], [1, 7, 8, 16, 63, 64, 65, 127, 128],
                ["tadd", "tsub", "tmul", "tadds", "tmuls", "texp", "tsqrt", "tlog"]):
            check(sys.argv[1], path, arch, op, dtype, columns)
            count += 1
        for arch, dtype, columns, op in itertools.product(
                ["a3", "a5"], ["f16", "f32"], [1, 16, 64, 128, 256],
                ["tcvt", "tmov", "trowsum"]):
            check(sys.argv[1], path, arch, op, dtype, columns)
            count += 1
        for arch, columns, types in itertools.product(
                ["a3", "a5"], [1, 64, 128, 256], [("bf16", "f32"), ("f32", "bf16")]):
            check(sys.argv[1], path, arch, "tcvt", types[0], columns, destination_type=types[1])
            count += 1
        for arch, op in itertools.product(["a3", "a5"], ["tmul", "tcvt", "tmov", "trowsum"]):
            check(sys.argv[1], path, arch, op, "f32", 64 if op == "trowsum" else 256, changed=True)
            count += 1
        check_repeat_overflow(sys.argv[1], path)
        count += 1
        for arch, index in itertools.product(["a3", "a5"], [0, 1]):
            check_padding_overlap(sys.argv[1], path, arch, index)
            count += 1
    print(f"{count} local access contract cases passed")


if __name__ == "__main__":
    main()
