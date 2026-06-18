# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import sys
import numpy as np

from cases import CASES
from st_common import result_cmp, style_fail, style_pass


def main():
    case_filter = sys.argv[1] if len(sys.argv) > 1 else None

    all_passed = True
    for case in CASES:
        if case_filter is not None and case["name"] != case_filter:
            continue

        case_dir = case["name"]
        m, n = case["m"], case["n"]
        dtype_out = case["dtype_out"]

        if not case.get("has_output", False):
            print(style_pass(f"[INFO] {case['name']}: compile-only (no output comparison)"))
            continue

        golden_path = os.path.join(case_dir, "golden.bin")
        output_path = os.path.join(case_dir, "output.bin")

        if not os.path.exists(output_path):
            print(style_fail(f"[ERROR] {case['name']}: output.bin not found"))
            all_passed = False
            continue

        if case.get("path") == "acc2vec_nz":
            golden = np.fromfile(golden_path, dtype=dtype_out).astype(np.float64)
            output = np.fromfile(output_path, dtype=dtype_out).astype(np.float64)
            if golden.size != output.size:
                print(style_fail(
                    f"[ERROR] {case['name']}: size mismatch golden={golden.size} output={output.size}"
                ))
                all_passed = False
                continue
            ok = result_cmp(golden, output, case["eps"])
        else:
            golden = np.fromfile(golden_path, dtype=dtype_out).astype(np.float32).reshape(m, n)
            output = np.fromfile(output_path, dtype=dtype_out).astype(np.float32)

            if output.shape != (m, n):
                if output.size == m * n:
                    output = output.reshape(m, n)
                else:
                    print(style_fail(
                        f"[ERROR] {case['name']}: size mismatch golden={golden.size} output={output.size}"
                    ))
                    all_passed = False
                    continue

            ok = result_cmp(golden, output, case["eps"])
        if ok:
            print(style_pass(f"[INFO] {case['name']}: compare passed"))
        else:
            print(style_fail(f"[ERROR] {case['name']}: compare failed"))
            all_passed = False

    if not all_passed:
        sys.exit(2)
    print(style_pass("[INFO] all cases passed"))


if __name__ == "__main__":
    main()
