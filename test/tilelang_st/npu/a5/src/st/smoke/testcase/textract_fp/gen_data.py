# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np

from cases import CASES
from st_common import setup_case_rng, save_case_data


for case in CASES:
    setup_case_rng(case)
    name = case["name"]

    if name.startswith("fp"):
        a = np.random.uniform(-1.0, 1.0, size=case["shape_src"]).astype(case["dtype_src"])
        b = np.random.uniform(-1.0, 1.0, size=case["shape_src"]).astype(case["dtype_src"])
        fb = np.ones(case["shape_scaling"], dtype=case["dtype_scaling"])
        id_mat = np.eye(case["shape_src"][0], dtype=case["dtype_src"])
        matmul_f32 = np.matmul(a.astype(np.float32), b.astype(np.float32))
        quantized_f16 = matmul_f32.astype(np.float16)
        golden = np.matmul(quantized_f16.astype(np.float32), id_mat.astype(np.float32))
        save_case_data(name, {"input1": a, "input2": b, "input3": fb, "input4": id_mat, "golden": golden})

    print(f"[INFO] gen_data: {name} done")
