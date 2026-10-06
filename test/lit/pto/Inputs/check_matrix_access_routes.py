# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check matrix access precision against full rectangles and current metadata."""
import re
from pathlib import Path
import subprocess
import sys
import tempfile


def run(tool, path, mode):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    text = Path(fixture).read_text()
    text = text[:text.index("  func.func @partial_shapes")] + "}\n"
    with tempfile.TemporaryDirectory(prefix="matrix-access-") as directory:
        path = Path(directory) / "case.pto"
        for element, width in [("f16", 2), ("f32", 4)]:
            base = text.replace("f16", element).replace("constant 512 : i64", "constant 1024 : i64")
            # Retain all native layouts; resolve valid extents at the use.
            dynamic = re.sub(r"(tile_buf<\w+, 16x16xf(?:16|32))", r"\1, valid=?x?", base)
            dynamic = re.sub(r"(pto.alloc_tile addr = %\w+)", r"\1 valid_row = %i16 valid_col = %i16", dynamic)
            for source in [base, dynamic]:
                path.write_text(source)
                effects = run(tool, path, "--storage-effects")
                assert "all-materialized=1" in effects, effects
                assert len(re.findall(r"effect \d+ .* intervals", effects)) == 17, effects
                # Whole 16x16 operands occupy exactly these physical byte spans.
                assert f"space=2 [0,{256 * width})" in effects, effects
                assert f"space=3 [0,{256 * width})" in effects, effects
                assert "space=5 [0,1024)" in effects, effects
                assert "effect 11 pto.tmatmul.acc read intervals" in effects
                assert "effect 14 pto.tmatmul.acc write intervals" in effects
                # Existing InsertSync consumes the same shared declarations.
                existing = run(tool, path, "--existing-dump")
                assert "pto.set_flag" in existing and "pto.wait_flag" in existing, existing
            # A later descriptor update cannot reuse the original full extent.
            changed = dynamic.replace("    pto.tmatmul ins",
                                      "    pto.set_validshape %al, %i1, %i16 : !left\n    pto.tmatmul ins")
            path.write_text(changed)
            effects = run(tool, path, "--storage-effects")
            assert "effect 5 pto.textract write intervals" in effects, effects
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
            assert "effect 15 pto.tstore read intervals" in effects, effects
            restored = changed.replace("    pto.tmatmul.acc",
                                       "    pto.set_validshape %al, %i16, %i16 : !left\n    pto.tmatmul.acc")
            path.write_text(restored)
            effects = run(tool, path, "--storage-effects")
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
            assert "effect 11 pto.tmatmul.acc read intervals" in effects, effects
            # An unknown extent remains conservative even with fixed capacity.
            unknown = dynamic.replace("%a: !pto.ptr", "%rows: index, %a: !pto.ptr")
            unknown = unknown.replace("%al = pto.alloc_tile addr = %base0 valid_row = %i16",
                                      "%al = pto.alloc_tile addr = %base0 valid_row = %rows")
            path.write_text(unknown)
            effects = run(tool, path, "--storage-effects")
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
    print("matrix accesses: F16/F32 rectangles, RMW, dynamic metadata and existing consumer passed")


if __name__ == "__main__":
    main()
