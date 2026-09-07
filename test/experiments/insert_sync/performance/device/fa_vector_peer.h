// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Benchmark peer adapted to the frozen 16x16 cube extraction. The original
// softmax/GU arithmetic is retained in the pinned reference headers.
// This peer is FIXED across compiler arms and is not an InsertSync input.
#pragma once
#include <cstdint>
#include "pto_macro_fa_softmax.hpp"
#include "pto_macro_fa_gu.hpp"
using namespace pto;

AICORE inline void benchmark_fa_vector(__gm__ float* qk, __gm__ half* p,
                                      __gm__ float* pv, __gm__ float* out,
                                      int64_t tiles) {
    using QKPipe = TPipe<0, Direction::DIR_C2V, 1024, 8, 8, false>;
    using PPipe = TPipe<2, Direction::DIR_V2C, 512, 8, 8, false>;
    using PVPipe = TPipe<4, Direction::DIR_C2V, 1024, 8, 8, false>;
    using F = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using H = Tile<TileType::Vec, half, 8, 16, BLayout::RowMajor, 8, 16>;
    using R = Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 8, 1>;
    using GF = GlobalTensor<float, Shape<1,1,1,8,16>, Stride<1,1,1,16,1>>;
    using GH = GlobalTensor<half, Shape<1,1,1,8,16>, Stride<1,1,1,16,1>>;
    QKPipe qp(qk, 0, 0);
    PPipe pp(p, 0, 0);
    PVPipe vp(pv, 0, 0);
    F x, scratch, running, product, unused_mask;
    H probability;
    R local_max, local_sum, global_max, global_sum, rescale;
    TASSIGN(x, 0); TASSIGN(scratch, 512); TASSIGN(running, 1024);
    TASSIGN(product, 1536); TASSIGN(unused_mask, 2048);
    TASSIGN(probability, 2560);
    TASSIGN(local_max, 2816); TASSIGN(local_sum, 2848);
    TASSIGN(global_max, 2880); TASSIGN(global_sum, 2912); TASSIGN(rescale, 2944);
    GF qslot, pvslot;
    GH pslot;
    for (int64_t i = 0; i < tiles; ++i) {
        TPOP<QKPipe, GF, TileSplitAxis::TILE_UP_DOWN>(qp, qslot);
        TLOAD(x, qslot);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TFREE<QKPipe, GF, TileSplitAxis::TILE_UP_DOWN>(qp, qslot);
        if (i == 0) {
            pto_macro_fa_softmax<true, 16, false>(probability, x, local_max, local_sum,
                global_max, global_sum, rescale, scratch, x, unused_mask, 0, 0);
        } else {
            pto_macro_fa_softmax<false, 16, false>(probability, x, local_max, local_sum,
                global_max, global_sum, rescale, scratch, x, unused_mask, 0, i * 16);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TALLOC<PPipe, GH, TileSplitAxis::TILE_UP_DOWN>(pp, pslot);
        TSTORE(pslot, probability);
        TPUSH<PPipe, GH, TileSplitAxis::TILE_UP_DOWN>(pp, pslot);
        TPOP<PVPipe, GF, TileSplitAxis::TILE_UP_DOWN>(vp, pvslot);
        if (i == 0) TLOAD(running, pvslot);
        else TLOAD(product, pvslot);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TFREE<PVPipe, GF, TileSplitAxis::TILE_UP_DOWN>(vp, pvslot);
        if (i != 0) pto_macro_fa_gu(running, product, rescale);
        pipe_barrier(PIPE_V);
        // Includes the one-tile case. A zero-tile launch leaves output untouched.
        if (i + 1 == tiles) {
            pto_macro_fa_gu_single_and_last_tile(running, global_sum);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            GF output(out + get_subblockid() * 8 * 16);
            TSTORE(output, running);
        }
        // Fixed peer's scratch reuse and probability store completion. These
        // barriers are reported separately from the generated cube's counts.
        pipe_barrier(PIPE_ALL);
    }
}
