# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

"""Single source of truth for tmov ST test cases.

Each case defines:
  - name:        case identifier, used as subdirectory name and by main.cpp kCases[].
  - dtype:       numpy dtype (e.g. np.float32).
  - shape:       (rows, cols) — allocated tile dimensions.
  - valid_shape: (valid_rows, valid_cols) — effective computation region.
  - eps:         tolerance for numpy.allclose (atol and rtol).

Based on pto-isa tmov_vect test cases:
  - float, half, uint8 types
  - shapes: 64x64, 32x32, 128x128, 128x32, 128x64
"""

import numpy as np

CASES = [
    {
        "name": "f32_64x64",
        "dtype": np.float32,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-6,
    },
    {
        "name": "f32_32x32",
        "dtype": np.float32,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-6,
    },
    {
        "name": "f32_128x128",
        "dtype": np.float32,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 1e-6,
    },
    {
        "name": "f32_128x32",
        "dtype": np.float32,
        "shape": (128, 32),
        "valid_shape": (128, 32),
        "eps": 1e-6,
    },
    {
        "name": "f32_128x64",
        "dtype": np.float32,
        "shape": (128, 64),
        "valid_shape": (128, 64),
        "eps": 1e-6,
    },
    {
        "name": "f16_64x64",
        "dtype": np.float16,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 1e-3,
    },
    {
        "name": "f16_32x32",
        "dtype": np.float16,
        "shape": (32, 32),
        "valid_shape": (32, 32),
        "eps": 1e-3,
    },
    {
        "name": "f16_128x128",
        "dtype": np.float16,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 1e-3,
    },
    {
        "name": "u8_64x64",
        "dtype": np.uint8,
        "shape": (64, 64),
        "valid_shape": (64, 64),
        "eps": 0,
    },
    {
        "name": "u8_128x128",
        "dtype": np.uint8,
        "shape": (128, 128),
        "valid_shape": (128, 128),
        "eps": 0,
    },
]