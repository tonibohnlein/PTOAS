// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGEXTRACTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGEXTRACTION_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
struct RotatingFragment {
    uint32_t payload = 0;
    uint32_t family = 0;
    uint32_t atom = 0;
    uint64_t slots = 0;
    uint64_t stride = 0;
    uint64_t offset = 0;
    bool read = false;
    bool write = false;
    uint64_t protectionGroup = 0;
};
struct RotatingExtraction {
    std::string error;
    std::vector<PeriodicRecord> generators;
    uint64_t refreshBound = 0;
    uint64_t protectedHazards = 0;
};
// Payload indices follow body order. Family/slot pairs and distinct atoms must
// denote disjoint exact storage. In iteration i a fragment accesses atom in
// slot (stride*i + offset) mod slots. A family has a common slots/stride pair.
// Modes at identical payload/family/atom/offset are merged before extraction.
// Nonzero protection groups certify writer pairs on one pipe within a visit.
// Groups with HardwareProtection.h's invocationProtectionBit additionally
// certify pairs across visits; their producer must prove that stronger scope.
// Returns a period-one generating relation; filter both endpoints to finite
// trips. O(A log A) arithmetic/comparison work and O(A) records, plus GCD and
// modular inverse work per family; integer costs depend on encoded bit lengths.
// No partial result is returned on invalid input or identity overflow.
RotatingExtraction extractRotatingGenerators(llvm::ArrayRef<PeriodicPayload> payloads,
                                             llvm::ArrayRef<RotatingFragment> fragments);
} // namespace mlir::pto::frontiersynch
#endif
