// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void TEXTRACT_M2L_f16_16x16(__gm__ uint16_t *src, __gm__ uint16_t *id, __gm__ float *out);
extern "C" __global__ AICORE void TEXTRACT_M2R_f16_16x16(__gm__ uint16_t *id, __gm__ uint16_t *src, __gm__ float *out);

void LaunchTEXTRACT_M2L_f16_16x16(uint16_t *src, uint16_t *id, float *out, void *stream) {
    TEXTRACT_M2L_f16_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)src, (__gm__ uint16_t *)id, (__gm__ float *)out);
}

void LaunchTEXTRACT_M2R_f16_16x16(uint16_t *id, uint16_t *src, float *out, void *stream) {
    TEXTRACT_M2R_f16_16x16<<<1, nullptr, stream>>>((__gm__ uint16_t *)id, (__gm__ uint16_t *)src, (__gm__ float *)out);
}