# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check ownership binding through equivalent native physical representations.

The driver performs actual import, construction and independent reconstruction.
Activation is separate from correctness; these checks make no timing claim.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile


def views(source):
    """Replace each direct allocation use with an equivalent identity view."""
    pattern = re.compile(r"^(\s*)(%\w+) = (pto\.alloc_tile[^\n]+ : )(!pto\.tile_buf<[^\n]+>)$", re.MULTILINE)

    def replace(match):
        indent, value, allocation, tile_type = match.groups()
        root = "%ownership_alloc_" + value[1:]
        return (f"{indent}{root} = {allocation}{tile_type}\n"
                f"{indent}{value} = pto.treshape {root} : {tile_type} -> {tile_type}")

    result, count = pattern.subn(replace, source)
    if count == 0:
        raise RuntimeError("fixture has no direct physical allocation to vary")
    return result


def variants(source):
    """Vary spelling and independent producer/descriptor/control context."""
    if source.count("    %c8224_i64 =") != 1 or "addr = %c32_i64" not in source:
        raise RuntimeError("arithmetic variant lost its physical address anchor")
    if source.count("    %3 = arith.index_cast") != 1:
        raise RuntimeError("unrelated-context variant lost its original insertion boundary")
    arithmetic = source.replace(
        "    %c8224_i64 =", "    %ownership_addr = arith.addi %c0_i64, %c32_i64 : i64\n    %c8224_i64 ="
    ).replace("addr = %c32_i64", "addr = %ownership_addr")
    independent = source.replace("    %3 = arith.index_cast", """
    %ownership_spare_addr = arith.constant 32768 : i64
    %ownership_spare = pto.alloc_tile addr = %ownership_spare_addr valid_row = %c1 valid_col = %c8
      : !pto.tile_buf<vec, 1x8xf32, valid=?x?>
    pto.set_validshape %ownership_spare, %c1, %c8 : !pto.tile_buf<vec, 1x8xf32, valid=?x?>
    %ownership_slice = pto.partition_view %1, offsets = [%c0, %c0], sizes = [%c1, %c8] : !pto.tensor_view<?x?xf32>
    pto.tload ins(%ownership_slice : !pto.partition_tensor_view<1x8xf32>)
      outs(%ownership_spare : !pto.tile_buf<vec, 1x8xf32, valid=?x?>)
    %ownership_cond = arith.cmpi eq, %arg6, %arg7 : i32
    scf.if %ownership_cond { }
    %3 = arith.index_cast""")
    return {"views": views(source), "arithmetic": arithmetic, "unrelated": independent}


def main():
    """Run each bounded native regression serially."""
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("native OAHS test driver was not found")
    source = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="oahs-owner-variants-") as temporary:
        for name, text in variants(source).items():
            path = pathlib.Path(temporary) / (name + ".pto")
            path.write_text(text, encoding="utf-8")
            result = subprocess.run([tool, "--construct", str(path)], check=False,
                                    capture_output=True, text=True, timeout=120)
            records = [dict(re.findall(r"(\w+)=([^\s]*)", line))
                       for line in result.stderr.splitlines() if line.startswith("function=")]
            valid = result.returncode == 0 and records and all(
                row.get("construction") == row.get("reconstruction") == "1"
                and int(row.get("ownership_bindings", "0")) > 0
                and row.get("observation_declined") == row.get("recurring_declined") == "0"
                for row in records
            )
            if not valid:
                raise RuntimeError(f"{name}: native ownership variant failed: {result.stderr}")
            print(f"{name}: constructed, reconstructed, ownership bound without retry")


if __name__ == "__main__":
    main()
