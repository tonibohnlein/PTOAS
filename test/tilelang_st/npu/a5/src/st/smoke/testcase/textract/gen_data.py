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


import os


for case in CASES:
    setup_case_rng(case)
    name = case["name"]

    if name.startswith("mat2left"):
        src = np.random.uniform(-1.0, 1.0, size=case["shape_src"]).astype(case["dtype_src"])
        id_mat = np.eye(case["shape_id"][0], case["shape_id"][1], dtype=case["dtype_id"])
        golden = np.matmul(src.astype(np.float32), id_mat.astype(np.float32)).astype(np.float32)
        save_case_data(name, {"input1": src, "input2": id_mat, "golden": golden})

    elif name.startswith("mat2right"):
        id_mat = np.eye(case["shape_id"][0], case["shape_id"][1], dtype=case["dtype_id"])
        src = np.random.uniform(-1.0, 1.0, size=case["shape_src"]).astype(case["dtype_src"])
        golden = src.astype(np.float32).T.copy()
        save_case_data(name, {"input1": id_mat, "input2": src, "golden": golden})

    print(f"[INFO] gen_data: {name} done")
