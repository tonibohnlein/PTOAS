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

void LaunchTEXTRACT_M2L_f16_16x16(uint16_t *src, uint16_t *id, float *out, void *stream);
void LaunchTEXTRACT_M2R_f16_16x16(uint16_t *id, uint16_t *src, float *out, void *stream);

using LaunchFn = void (*)(uint16_t *, uint16_t *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t input1Rows;
    size_t input1Cols;
    size_t input2Rows;
    size_t input2Cols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"mat2left_f16_16x16",  LaunchTEXTRACT_M2L_f16_16x16, 16, 16, 16, 16, 16, 16},
    {"mat2right_f16_16x16", LaunchTEXTRACT_M2R_f16_16x16, 16, 16, 16, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t i1Elems = tc.input1Rows * tc.input1Cols;
    const size_t i2Elems = tc.input2Rows * tc.input2Cols;
    const size_t outElems = tc.outRows * tc.outCols;
    const size_t i1Bytes = i1Elems * sizeof(uint16_t);
    const size_t i2Bytes = i2Elems * sizeof(uint16_t);
    const size_t outBytes = outElems * sizeof(float);
    size_t i1FileSize = i1Bytes;
    size_t i2FileSize = i2Bytes;

    std::printf(
        "[INFO] === case: %s (i1=%zux%zu, i2=%zux%zu, out=%zux%zu) ===\n",
        tc.name, tc.input1Rows, tc.input1Cols, tc.input2Rows, tc.input2Cols, tc.outRows, tc.outCols
    );

    std::string caseDir = std::string("./") + tc.name;

    void *i1Host = nullptr;
    void *i2Host = nullptr;
    void *outHost = nullptr;
    void *i1Device = nullptr;
    void *i2Device = nullptr;
    void *outDevice = nullptr;

    aclrtMallocHost(&i1Host, i1Bytes);
    aclrtMallocHost(&i2Host, i2Bytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&i1Device, i1Bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&i2Device, i2Bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), i1FileSize, i1Host, i1Bytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), i2FileSize, i2Host, i2Bytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(i1Device, i1Bytes, i1Host, i1Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(i2Device, i2Bytes, i2Host, i2Bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(
            static_cast<uint16_t *>(i1Device),
            static_cast<uint16_t *>(i2Device),
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

    if (i1Device != nullptr) aclrtFree(i1Device);
    if (i2Device != nullptr) aclrtFree(i2Device);
    if (outDevice != nullptr) aclrtFree(outDevice);
    if (i1Host != nullptr) aclrtFreeHost(i1Host);
    if (i2Host != nullptr) aclrtFreeHost(i2Host);
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