#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np

from validation_runtime import (
    float32_to_bf16,
    load_case_meta,
    load_int32_assignments,
    rng,
    write_buffers,
    write_golden,
)

SUPPORTED_CASES = frozenset({
    "attention_csa_test_refresh_incore_81",
    "attention_hca_test_incore_54",
    "attention_swa_test_incore_40",
    "decode_csa_test_incore_81",
    "decode_hca_test_incore_54",
    "decode_swa_test_incore_40",
    "sparse_attn_test_incore_7",
})

OUTPUT_ROWS = 1024
OUTPUT_COLS = 4096
INPUT_ROWS = 8192
INPUT_COLS = 64
BLOCK_GROUP = 8
DT_PER_GROUP = 32
HH_PER_TILE = 8
OUTPUT_ROW_GROUP = 128
OUTPUT_COL_STRIDE = 512
OUTPUT_COL_BASE = 448


def _require_count(meta, name: str, expected: int) -> None:
    actual = int(meta.elem_counts[name])
    if actual != expected:
        raise ValueError(f"{name}: expected {expected} elements, got {actual}")


def _make_bf16_zeros(meta, name: str, expected: int) -> np.ndarray:
    _require_count(meta, name, expected)
    return np.zeros(expected, dtype=meta.np_types[name])


def _make_fp32_input(meta, name: str, generator, expected: int) -> np.ndarray:
    _require_count(meta, name, expected)
    values = generator.uniform(-0.5, 0.5, size=expected).astype(np.float32)
    return values.astype(meta.np_types[name], copy=False)


def build_case(meta, generator, ints):
    if meta.outputs != ["v1"]:
        raise ValueError(f"unexpected outputs: {meta.outputs}")
    if meta.read_order != ["v1", "v2", "v3"]:
        raise ValueError(f"unexpected read order: {meta.read_order}")
    if len(ints) < 2:
        raise ValueError(f"expected block_idx/block_num int32 params, got {ints}")

    block_idx, block_num = ints[:2]
    if block_num <= 0:
        raise ValueError(f"invalid block_num={block_num}")
    if block_idx < 0 or block_idx >= block_num:
        raise ValueError(f"invalid block_idx={block_idx} for block_num={block_num}")

    output_elems = OUTPUT_ROWS * OUTPUT_COLS
    input_elems = INPUT_ROWS * INPUT_COLS
    buffers = {
        "v1": _make_bf16_zeros(meta, "v1", output_elems),
        "v2": _make_fp32_input(meta, "v2", generator, input_elems),
        "v3": _make_fp32_input(meta, "v3", generator, input_elems),
    }

    out = np.array(buffers["v1"], copy=True).reshape(OUTPUT_ROWS, OUTPUT_COLS)
    rope_even = np.asarray(buffers["v2"], dtype=np.float32).reshape(INPUT_ROWS, INPUT_COLS)
    rope_odd = np.asarray(buffers["v3"], dtype=np.float32).reshape(INPUT_ROWS, INPUT_COLS)

    group_idx = block_idx // BLOCK_GROUP
    lane_idx = block_idx % BLOCK_GROUP
    dt_base = group_idx * DT_PER_GROUP
    out_row_base = lane_idx * OUTPUT_ROW_GROUP
    src_row_lane_offset = lane_idx * HH_PER_TILE

    for dt in range(DT_PER_GROUP):
        dt_idx = dt_base + dt
        src_row = dt_idx * INPUT_COLS + src_row_lane_offset
        tile = rope_even[src_row:src_row + HH_PER_TILE, :] + rope_odd[src_row:src_row + HH_PER_TILE, :]
        tile_bf16 = float32_to_bf16(tile)
        dst_row = out_row_base + dt_idx
        for hh in range(HH_PER_TILE):
            col0 = OUTPUT_COL_BASE + hh * OUTPUT_COL_STRIDE
            out[dst_row, col0:col0 + INPUT_COLS] = tile_bf16[hh]

    return buffers, {"v1": out.reshape(-1)}


def run_case(case_name: str):
    if case_name not in SUPPORTED_CASES:
        raise KeyError(f"unsupported case: {case_name}")
    meta = load_case_meta()
    generator = rng()
    ints = load_int32_assignments()
    buffers, golden = build_case(meta, generator, ints)
    write_buffers(meta, buffers)
    write_golden(meta, golden)
