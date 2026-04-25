# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.tpartmax"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tpartmax",
    advanced=True,
)
def template_tpartmax(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape
    src0_valid_rows, src0_valid_cols = src0.valid_shape
    src1_valid_rows, src1_valid_cols = src1.valid_shape
    lanes = pto.get_lanes(dtype)

    pad_scalar = pto.PadValue.MIN.eval(dtype)
    pad_vec = pto.vbr(pad_scalar)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(pad_vec, dst[row, col:], mask)

    for row in range(0, src0_valid_rows, 1):
        remained = src0_valid_cols
        for col in range(0, src0_valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            vec0 = pto.vlds(src0[row, col:])
            pto.vsts(vec0, dst[row, col:], mask)

    for row in range(0, src1_valid_rows, 1):
        remained = src1_valid_cols
        for col in range(0, src1_valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            vec_dst = pto.vlds(dst[row, col:])
            vec1 = pto.vlds(src1[row, col:])
            result = pto.vmax(vec_dst, vec1, mask)
            pto.vsts(result, dst[row, col:], mask)

    return