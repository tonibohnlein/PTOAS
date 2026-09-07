// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Synchronized host wall time. The Python/ctypes call is outside this interval.
#include <acl/acl.h>
#include <chrono>
#include <cstdint>
extern "C" void LaunchCase(void**, uint64_t, int64_t, bool, void*);
extern "C" int RunCase(void** buffers, uint64_t ffts, int64_t count, bool tail,
                       void* stream, int64_t launches, double* host_us) {
    if (launches < 1 || host_us == nullptr) return -1;
    int status = aclrtSynchronizeStream(stream);
    if (status) return status;
    auto start = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < launches; ++i)
        LaunchCase(buffers, ffts, count, tail, stream);
    status = aclrtSynchronizeStream(stream);
    auto end = std::chrono::steady_clock::now();
    *host_us = std::chrono::duration<double, std::micro>(end - start).count() / launches;
    return status;
}
