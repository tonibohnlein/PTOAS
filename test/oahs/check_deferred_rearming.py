# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise deferred rearming through the native importer and reconstruction."""
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

from check_dormant_ownership import views


def variants(source):
    """Equivalent storage/address forms and independent source context."""
    if source.count("    %c16448_i64 =") != 1 or "addr = %c64_i64" not in source:
        raise RuntimeError("arithmetic variant lost its address anchor")
    if source.count("    %3 = pto.alloc_tile") != 1:
        raise RuntimeError("context variant lost its insertion boundary")
    arithmetic = source.replace(
        "    %c16448_i64 =", "    %deferred_addr = arith.addi %c0_i64, %c64_i64 : i64\n    %c16448_i64 ="
    ).replace("addr = %c64_i64", "addr = %deferred_addr")
    context = source.replace("    %3 = pto.alloc_tile", """
    %spare_addr = arith.constant 32768 : i64
    %spare = pto.alloc_tile addr = %spare_addr valid_row = %c1 valid_col = %c16
      : !pto.tile_buf<vec, 1x16xf32, valid=?x?>
    pto.set_validshape %spare, %c1, %c16 : !pto.tile_buf<vec, 1x16xf32, valid=?x?>
    %spare_slice = pto.partition_view %2, offsets = [%c0, %c0], sizes = [%c1, %c16]
      : !pto.tensor_view<?x?xf32>
    pto.tload ins(%spare_slice : !pto.partition_tensor_view<1x16xf32>)
      outs(%spare : !pto.tile_buf<vec, 1x16xf32, valid=?x?>)
    %spare_cond = arith.cmpi eq, %arg3, %c0 : index
    scf.if %spare_cond { }
    %3 = pto.alloc_tile""")
    return {"views": views(source), "arithmetic": arithmetic, "context": context}


def main():
    """Run serial variants; numerical/device behavior is not claimed here."""
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("native OAHS test driver was not found")
    source = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="oahs-deferred-") as temporary:
        for name, value in variants(source).items():
            path = pathlib.Path(temporary) / (name + ".pto")
            path.write_text(value, encoding="utf-8")
            result = subprocess.run([tool, "--construct", str(path)], check=False,
                                    capture_output=True, text=True, timeout=120)
            rows = [dict(re.findall(r"(\w+)=([^\s]*)", line))
                    for line in result.stderr.splitlines() if line.startswith("function=")]
            valid = result.returncode == 0 and rows and all(
                row.get("construction") == row.get("reconstruction") == "1"
                and int(row.get("rearming_deferred", "0")) > 0
                and row.get("observation_declined") == row.get("recurring_declined") == "0"
                for row in rows
            )
            if not valid:
                raise RuntimeError(f"{name}: native deferred rearming failed: {result.stderr}")
            print(f"{name}: deferred rearming constructed and reconstructed without retry")


if __name__ == "__main__":
    main()
