# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TileLang DSL template for pto.trowexpandexpdif"""

import sys
from pathlib import Path
import tilelang_dsl as pto


def _constraint_trowexpandexpdif_row_major(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile) -> bool:
    """Constraint for RowMajor layout trowexpandexpdif template."""
    # All tiles must be RowMajor layout
    src0_row_major = src0.config.b_layout == pto.BLayout.ROW_MAJOR
    src1_row_major = src1.config.b_layout == pto.BLayout.ROW_MAJOR
    dst_row_major = dst.config.b_layout == pto.BLayout.ROW_MAJOR
    return src0_row_major and src1_row_major and dst_row_major


@pto.vkernel(
    target="a5",
    op="pto.trowexpandexpdif",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    constraints=[_constraint_trowexpandexpdif_row_major],
)
def template_trowexpandexpdif_f32(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpandexpdif with f32 dtype.

    Compute exp(src0 - scalar) for each row using per-row scalars from src1[row, 0].
    Semantics: dst[row, col] = exp(src0[row, col] - src1[row, 0])
    Used in numerically stable softmax computation.
    """
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(pto.f32)):
            mask, remained = pto.make_mask(pto.f32, remained)
            scalar_vec = pto.vlds(src1[row, :])
            broadcasted = pto.vdup(scalar_vec, mask)
            lhs = pto.vlds(src0[row, col:])
            result = pto.vexpdif(lhs, broadcasted, mask, pto.VcvtPartMode.EVEN)
            pto.vsts(result, dst[row, col:], mask)
    return


@pto.vkernel(
    target="a5",
    op="pto.trowexpandexpdif",
    dtypes=[(pto.f16, pto.f16, pto.f16)],
    constraints=[_constraint_trowexpandexpdif_row_major],
)
def template_trowexpandexpdif_f16(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
    """Template for pto.trowexpandexpdif with f16 dtype.

    Compute exp(src0 - scalar) for each row using per-row scalars from src1[row, 0].
    Semantics: dst[row, col] = exp(src0[row, col] - src1[row, 0])
    Used in numerically stable softmax computation.
    """
    dtype = pto.f16
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            scalar_vec = pto.vlds(src1[row, :])
            broadcasted = pto.vdup(scalar_vec, mask)
            lhs = pto.vlds(src0[row, col:])
            diff = pto.vsub(lhs, broadcasted, mask)
            result = pto.vexp(diff, mask)
            pto.vsts(result, dst[row, col:], mask)
    return
