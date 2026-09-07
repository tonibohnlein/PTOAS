// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Benchmark-only dispatch. Generated kernel.cpp is copied byte for byte.
#include <cstdint>
#include <pto/pto-inst.hpp>
#if defined(BENCH_CONV2D)
// Tile-valued extern prototypes in kernel.cpp also need this guard: guarding
// only the helper definitions does not make the host compilation parse them.
#if defined(__DAV_CUBE__)
#include "conv2d_helpers.h"
#include "kernel.cpp"
#endif
#else
#include "kernel.cpp"
#endif
#if defined(BENCH_FA) && defined(__DAV_VEC__)
#include "fa_vector_peer.h"
#endif

// One mixed launch schedules one cube and its two associated vector subblocks.
// Every pointer is a DEVICE pointer; ffts is the runtime control address value.
__global__ AICORE void benchmark_entry(
    __gm__ uint8_t* b0, __gm__ uint8_t* b1, __gm__ uint8_t* b2,
    __gm__ uint8_t* b3, __gm__ uint8_t* b4, __gm__ uint8_t* b5,
    __gm__ uint8_t* b6, __gm__ uint8_t* b7, __gm__ uint8_t* b8,
    uint64_t ffts, int64_t count, bool tail) {
#if defined(BENCH_CONV2D) && defined(__DAV_CUBE__)
    conv2d_interior_tile((__gm__ half*)b0, (__gm__ half*)b1, (__gm__ half*)b2, count);
#elif defined(BENCH_FA)
#if defined(__DAV_CUBE__) || defined(__DAV_VEC__)
    set_ffts_base_addr(ffts);
#endif
#if defined(__DAV_CUBE__)
    flash_attention_cube((__gm__ half*)b0, (__gm__ half*)b1, (__gm__ half*)b2,
                         (__gm__ float*)b3, (__gm__ half*)b4, (__gm__ float*)b5, count);
#elif defined(__DAV_VEC__)
    benchmark_fa_vector((__gm__ float*)b3, (__gm__ half*)b4,
                        (__gm__ float*)b5, (__gm__ float*)b6, count);
#endif
#elif defined(BENCH_GDN) || defined(BENCH_KDA)
#if defined(BENCH_GDN)
#define WY_CUBE gdn_wy_cube
#define WY_VECTOR gdn_wy_vector
#else
#define WY_CUBE kda_wy_cube
#define WY_VECTOR kda_wy_vector
#endif
#define WY_ARGS (__gm__ half*)b0, (__gm__ half*)b1, (__gm__ half*)b2, (__gm__ float*)b3, (__gm__ half*)b4, (__gm__ half*)b5, (__gm__ half*)b6, (__gm__ half*)b7, (__gm__ half*)b8, (__gm__ int64_t*)ffts, count, tail
#if defined(__DAV_CUBE__)
    WY_CUBE(WY_ARGS);
#elif defined(__DAV_VEC__)
    WY_VECTOR(WY_ARGS, static_cast<int64_t>(get_subblockid()));
#if defined(BENCH_GDN)
    // The extraction consumes previous-chunk credits, leaving the final two
    // credits to its caller. Drain them on EACH vector stripe before returning.
    if (count > 0) {
        __builtin_cce_wait_flag_dev(3);
        __builtin_cce_wait_flag_dev(4);
    }
#endif
#endif
#undef WY_ARGS
#undef WY_CUBE
#undef WY_VECTOR
#endif
}

extern "C" void LaunchCase(void** b, uint64_t ffts, int64_t count, bool tail, void* stream) {
    benchmark_entry<<<1, nullptr, stream>>>(
        (__gm__ uint8_t*)b[0], (__gm__ uint8_t*)b[1], (__gm__ uint8_t*)b[2],
        (__gm__ uint8_t*)b[3], (__gm__ uint8_t*)b[4], (__gm__ uint8_t*)b[5],
        (__gm__ uint8_t*)b[6], (__gm__ uint8_t*)b[7], (__gm__ uint8_t*)b[8],
        ffts, count, tail);
}
