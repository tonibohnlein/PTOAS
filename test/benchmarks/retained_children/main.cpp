// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// correct SEED: four independent queued slots, checked after one synchronize.
// time SEED: correctness first, then 10 warmups and 10 single-launch event samples.
#include "acl/acl.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void LaunchRetained(void*, void*, void*, void*);
int RetainedOuter();
int RetainedReaders();
#define CK(e) do { aclError rc = (e); if (rc != ACL_SUCCESS) { \
    std::fprintf(stderr, "%s failed: %d\n", #e, int(rc)); return 3; } } while (0)
uint64_t hashBytes(const std::vector<float>& x) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < x.size() * sizeof(float); ++i) {
        h ^= reinterpret_cast<const unsigned char*>(x.data())[i]; h *= 1099511628211ull;
    }
    return h;
}
int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[1], "correct") && std::strcmp(argv[1], "time"))) return 2;
    const bool timing = std::strcmp(argv[1], "time") == 0;
    char* end = nullptr;
    const auto seed = std::strtoull(argv[2], &end, 10);
    if (!end || *end) return 2;
    const int outer = RetainedOuter(), readers = RetainedReaders();
    if (!((outer == 2 && readers == 1) || (outer == 4 && readers == 2))) return 2;
    constexpr size_t guard = 256;
    constexpr float guardValue = 1.25e30f, sentinel = -7.5e30f;
    const size_t n = size_t(outer) * 32 * 256, total = n + 2 * guard, bytes = total * sizeof(float);
    const unsigned slots = timing ? 1 : 4;
    uint64_t random = seed ? seed : 17;
    auto value = [&]() {
        random ^= random >> 12; random ^= random << 25; random ^= random >> 27;
        const auto bits = random * 2685821657736338717ull;
        return float(int(bits % 31) - 15); // Small integers: every reference addition is exact.
    };
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream stream = nullptr; CK(aclrtCreateStream(&stream));
    struct Slot { void *a = nullptr, *b = nullptr, *out = nullptr; std::vector<float> ha, hb, want; };
    std::vector<Slot> data(slots);
    auto payload = [&](void* p) { return static_cast<float*>(p) + guard; };
    for (unsigned s = 0; s < slots; ++s) {
        auto& d = data[s]; d.ha.assign(total, guardValue); d.hb.assign(total, guardValue);
        d.want.assign(total, guardValue);
        std::vector<float> initial(total, guardValue);
        for (size_t i = guard; i < n + guard; ++i) {
            const float a = d.ha[i] = value(), b = d.hb[i] = value();
            d.want[i] = std::fabs(a) + readers * a + std::fabs(b) + readers * b;
            initial[i] = sentinel;
        }
        std::printf("case=%d,%d seed=%llu slot=%u input_fnv=%016llx,%016llx\n", outer, readers,
                    seed, s, (unsigned long long)hashBytes(d.ha), (unsigned long long)hashBytes(d.hb));
        CK(aclrtMalloc(&d.a, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&d.b, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&d.out, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(d.a, bytes, d.ha.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(d.b, bytes, d.hb.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(d.out, bytes, initial.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    auto launch = [&](Slot& d) { LaunchRetained(payload(d.a), payload(d.b), payload(d.out), stream); };
    for (auto& d : data) launch(d);
    CK(aclrtSynchronizeStream(stream));
    auto verify = [&](Slot& d) {
        std::vector<float> got(total);
        if (aclrtMemcpy(got.data(), bytes, d.out, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) return false;
        for (size_t i = 0; i < total; ++i) if (!std::isfinite(got[i]) || got[i] != d.want[i]) {
            std::fprintf(stderr, "FAIL index=%zu got=%g expected=%g\n", i, got[i], d.want[i]); return false;
        }
        if (aclrtMemcpy(got.data(), bytes, d.a, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS || got != d.ha) return false;
        if (aclrtMemcpy(got.data(), bytes, d.b, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS || got != d.hb) return false;
        return true;
    };
    for (auto& d : data) if (!verify(d)) return 1;
    std::puts("PASS correctness");
    if (timing) {
        for (unsigned i = 0; i < 10; ++i) launch(data[0]);
        CK(aclrtSynchronizeStream(stream));
        aclrtEvent first, last; CK(aclrtCreateEvent(&first)); CK(aclrtCreateEvent(&last));
        for (unsigned sample = 0; sample < 10; ++sample) {
            CK(aclrtRecordEvent(first, stream)); launch(data[0]); CK(aclrtRecordEvent(last, stream));
            CK(aclrtSynchronizeStream(stream));
            float ms = 0; CK(aclrtEventElapsedTime(&ms, first, last));
            std::printf("sample=%u device_us=%.6f\n", sample, double(ms) * 1000);
        }
        if (!verify(data[0])) return 1;
        CK(aclrtDestroyEvent(first)); CK(aclrtDestroyEvent(last));
    }
    for (auto& d : data) { CK(aclrtFree(d.a)); CK(aclrtFree(d.b)); CK(aclrtFree(d.out)); }
    CK(aclrtDestroyStream(stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
}
