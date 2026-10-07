# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check carried-state relevance on the original proj_b_mm structure."""
import pathlib
import subprocess
import sys
import tempfile


def check(tool, path, text, carried):
    path.write_text(text)
    run = subprocess.run([tool, "--recognize", str(path)], capture_output=True,
                         text=True, check=True, timeout=60)
    outer = run.stdout.split("  loop ", 1)[1].split("\n  loop ", 1)[0]
    found = "issue loop-carried-state" in outer
    if found != carried:
        raise AssertionError(f"{path.name}: carried={found}, expected {carried}\n{run.stdout}")


def main():
    tool, source = sys.argv[1:]
    text = pathlib.Path(source).read_text()
    # The original unused carry may be analyzed without changing its source IR.
    with tempfile.TemporaryDirectory(prefix="pto-carried-relevance-") as directory:
        root = pathlib.Path(directory)
        check(tool, root / "unused.pto", text, False)
        # A live loop result is relevant even though its iterarg is unused.
        live = text.replace(") attributes {pto.kernel_kind", ") -> index attributes {pto.kernel_kind", 1)
        live = live.replace("    return\n", "    return %6 : index\n")
        check(tool, root / "live-result.pto", live, True)
        # Zero trips forward the original init operand to that same live result.
        zero = live.replace("%arg8 = %c0 to %c2", "%arg8 = %c0 to %c0")
        check(tool, root / "zero-trip.pto", zero, True)
        # Relevance reaches the outer carry through the zero-trip initializer,
        # even though the inner body's yield is independent of that initializer.
        init = text.replace("      %7 =", """      %from_init = scf.for %z = %c0 to %c0 step %c1
          iter_args(%unused = %arg9) -> (index) {
        scf.yield %c1 : index
      }
      %7 =""", 1)
        init = init.replace("%arg10 = %c0 to %c4", "%arg10 = %c0 to %from_init")
        check(tool, root / "zero-trip-initializer.pto", init, True)
        # Dead cross-carried cycles terminate, while a control use makes both live.
        cycle = text.replace("%6 = scf.for", "%6:2 = scf.for")
        cycle = cycle.replace("iter_args(%arg9 = %arg3) -> (index)",
                              "iter_args(%arg9 = %arg3, %other = %arg3) -> (index, index)")
        cycle = cycle.replace("scf.yield %8 : index", "scf.yield %other, %arg9 : index, index")
        check(tool, root / "dead-cycle.pto", cycle, False)
        cycle = cycle.replace("%arg10 = %c0 to %c4", "%arg10 = %c0 to %arg9")
        check(tool, root / "live-cycle.pto", cycle, True)
    print("carried relevance: corpus, live result, zero trips, and carry cycles passed")


if __name__ == "__main__":
    main()
