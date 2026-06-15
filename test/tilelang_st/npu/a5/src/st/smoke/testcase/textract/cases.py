# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import numpy as np


CASES = [
    {
        "name": "mat2left_f16_16x16",
        "dtype_src": np.float16,
        "dtype_id": np.float16,
        "shape_src": (16, 16),
        "shape_id": (16, 16),
        "shape_out": (16, 16),
        "eps": 1e-2,
    },
    {
        "name": "mat2right_f16_16x16",
        "dtype_src": np.float16,
        "dtype_id": np.float16,
        "shape_src": (16, 16),
        "shape_id": (16, 16),
        "shape_out": (16, 16),
        "eps": 1e-2,
    },
]