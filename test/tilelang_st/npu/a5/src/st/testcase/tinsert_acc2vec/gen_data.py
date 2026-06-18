# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import os
import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


def pack_nd_to_nz_f32(mat):
    """Pack 2D ND matrix to NZ fractal flat layout (N1 M0 N0).

    For 16x16 f32, the hardware preserves L0C's N1-M0-N0 physical order:
      [CB0 row0, CB1 row0, CB0 row1, CB1 row1, ..., CB0 row15, CB1 row15]
    where each CBi rowj is c0=8 consecutive f32 elements.
    """
    rows, cols = mat.shape
    c0 = 32 // mat.dtype.itemsize
    num_col_blocks = cols // c0
    return mat.reshape(rows, num_col_blocks, c0).reshape(-1)


for case in CASES:
    setup_case_rng(case)
    m, k, n = case["m"], case["k"], case["n"]
    dtype = case["dtype"]
    dtype_out = case["dtype_out"]

    A = np.random.uniform(-1.0, 1.0, size=(m, k)).astype(dtype)
    B = np.random.uniform(-1.0, 1.0, size=(k, n)).astype(dtype)
    golden_f32 = np.matmul(A.astype(np.float32), B.astype(np.float32))

    if case.get("path") == "acc2vec_nz":
        golden = pack_nd_to_nz_f32(golden_f32.astype(dtype_out))
    else:
        golden = golden_f32.astype(dtype_out)

    data = {"input1": A, "input2": B, "golden": golden}

    save_case_data(case["name"], data)
    print(
        f"[INFO] gen_data: {case['name']} A=({m},{k}) B=({k},{n}) "
        f"dtype={dtype.__name__} dtype_out={dtype_out.__name__}"
    )
