// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace PtoTestCommon;

void LaunchTEXTRACT_V2V_ND_f32_16x16(float *src, float *out, void *stream);

using LaunchFn = void (*)(float *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t srcRows;
    size_t srcCols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"v2v_f32_16x16", LaunchTEXTRACT_V2V_ND_f32_16x16, 16, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t srcElems = tc.srcRows * tc.srcCols;
    const size_t outElems = tc.outRows * tc.outCols;
    const size_t srcBytes = srcElems * sizeof(float);
    const size_t outBytes = outElems * sizeof(float);
    size_t srcFileSize = srcBytes;

    std::printf(
        "[INFO] === case: %s (src=%zux%zu, out=%zux%zu) ===\n",
        tc.name, tc.srcRows, tc.srcCols, tc.outRows, tc.outCols
    );

    std::string caseDir = std::string("./") + tc.name;

    void *srcHost = nullptr;
    void *outHost = nullptr;
    void *srcDevice = nullptr;
    void *outDevice = nullptr;

    aclrtMallocHost(&srcHost, srcBytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), srcFileSize, srcHost, srcBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(
            static_cast<float *>(srcDevice),
            static_cast<float *>(outDevice),
            stream
        );

        aclrtSynchronizeStream(stream);
        aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    }

    if (rc == 0 && !WriteFile((caseDir + "/output.bin").c_str(), outHost, outBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write %s/output.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (srcDevice != nullptr) aclrtFree(srcDevice);
    if (outDevice != nullptr) aclrtFree(outDevice);
    if (srcHost != nullptr) aclrtFreeHost(srcHost);
    if (outHost != nullptr) aclrtFreeHost(outHost);

    if (rc == 0)
        std::printf("[INFO] case %s done\n", tc.name);
    return rc;
}

int main(int argc, char *argv[]) {
    const char *caseFilter = (argc > 1) ? argv[1] : nullptr;

    int rc = 0;
    int deviceId = 0;
    aclrtStream stream = nullptr;

    aclInit(nullptr);
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    for (size_t i = 0; i < kNumCases; ++i) {
        if (caseFilter != nullptr && std::strcmp(kCases[i].name, caseFilter) != 0) {
            continue;
        }
        int ret = RunCase(kCases[i], deviceId, stream);
        if (ret != 0) {
            std::fprintf(stderr, "[ERROR] case %s failed\n", kCases[i].name);
            rc = 1;
            break;
        }
    }

    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return rc;
}