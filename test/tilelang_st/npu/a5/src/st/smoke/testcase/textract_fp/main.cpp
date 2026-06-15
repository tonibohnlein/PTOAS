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

void LaunchTEXTRACT_FP_f32_f16_16x16(uint16_t *a, uint16_t *b, float *fb, uint16_t *id, float *out, void *stream);

using LaunchFn = void (*)(uint16_t *, uint16_t *, float *, uint16_t *, float *, void *);

struct TestCase {
    const char *name;
    LaunchFn launch;
    size_t aRows;
    size_t aCols;
    size_t bRows;
    size_t bCols;
    size_t fbRows;
    size_t fbCols;
    size_t idRows;
    size_t idCols;
    size_t outRows;
    size_t outCols;
};

static const TestCase kCases[] = {
    {"fp_f32_f16_16x16", LaunchTEXTRACT_FP_f32_f16_16x16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16},
};
static constexpr size_t kNumCases = sizeof(kCases) / sizeof(kCases[0]);

static int RunCase(const TestCase &tc, int deviceId, aclrtStream stream) {
    (void)deviceId;
    int rc = 0;
    const size_t aElems = tc.aRows * tc.aCols;
    const size_t bElems = tc.bRows * tc.bCols;
    const size_t fbElems = tc.fbRows * tc.fbCols;
    const size_t idElems = tc.idRows * tc.idCols;
    const size_t outElems = tc.outRows * tc.outCols;
    const size_t aBytes = aElems * sizeof(uint16_t);
    const size_t bBytes = bElems * sizeof(uint16_t);
    const size_t fbBytes = fbElems * sizeof(float);
    const size_t idBytes = idElems * sizeof(uint16_t);
    const size_t outBytes = outElems * sizeof(float);
    size_t aFileSize = aBytes;
    size_t bFileSize = bBytes;
    size_t fbFileSize = fbBytes;
    size_t idFileSize = idBytes;

    std::printf(
        "[INFO] === case: %s (a=%zux%zu, b=%zux%zu, fb=%zux%zu, id=%zux%zu, out=%zux%zu) ===\n",
        tc.name, tc.aRows, tc.aCols, tc.bRows, tc.bCols, tc.fbRows, tc.fbCols, tc.idRows, tc.idCols, tc.outRows, tc.outCols
    );

    std::string caseDir = std::string("./") + tc.name;

    void *aHost = nullptr;
    void *bHost = nullptr;
    void *fbHost = nullptr;
    void *idHost = nullptr;
    void *outHost = nullptr;
    void *aDevice = nullptr;
    void *bDevice = nullptr;
    void *fbDevice = nullptr;
    void *idDevice = nullptr;
    void *outDevice = nullptr;

    aclrtMallocHost(&aHost, aBytes);
    aclrtMallocHost(&bHost, bBytes);
    aclrtMallocHost(&fbHost, fbBytes);
    aclrtMallocHost(&idHost, idBytes);
    aclrtMallocHost(&outHost, outBytes);

    aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&fbDevice, fbBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&idDevice, idBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    if (!ReadFile((caseDir + "/input1.bin").c_str(), aFileSize, aHost, aBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input1.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input2.bin").c_str(), bFileSize, bHost, bBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input2.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input3.bin").c_str(), fbFileSize, fbHost, fbBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input3.bin\n", caseDir.c_str());
        rc = 1;
    }
    if (rc == 0 && !ReadFile((caseDir + "/input4.bin").c_str(), idFileSize, idHost, idBytes)) {
        std::fprintf(stderr, "[ERROR] failed to read %s/input4.bin\n", caseDir.c_str());
        rc = 1;
    }

    if (rc == 0) {
        aclrtMemcpy(aDevice, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(bDevice, bBytes, bHost, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(fbDevice, fbBytes, fbHost, fbBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(idDevice, idBytes, idHost, idBytes, ACL_MEMCPY_HOST_TO_DEVICE);

        tc.launch(
            static_cast<uint16_t *>(aDevice),
            static_cast<uint16_t *>(bDevice),
            static_cast<float *>(fbDevice),
            static_cast<uint16_t *>(idDevice),
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

    if (aDevice != nullptr) aclrtFree(aDevice);
    if (bDevice != nullptr) aclrtFree(bDevice);
    if (fbDevice != nullptr) aclrtFree(fbDevice);
    if (idDevice != nullptr) aclrtFree(idDevice);
    if (outDevice != nullptr) aclrtFree(outDevice);
    if (aHost != nullptr) aclrtFreeHost(aHost);
    if (bHost != nullptr) aclrtFreeHost(bHost);
    if (fbHost != nullptr) aclrtFreeHost(fbHost);
    if (idHost != nullptr) aclrtFreeHost(idHost);
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
