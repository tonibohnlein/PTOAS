// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared-cycle certificates for an unconditional periodic payload skeleton.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICSHAREDCERTIFICATE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICSHAREDCERTIFICATE_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedAllocation.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
struct PeriodicSharedAssignment {
    PeriodicSharedAllocation allocation;
    std::vector<uint32_t> records; // Phase index -> original canonical record.
    bool orderExact = true;
};
// Numerical result shared by whole-invocation and regional adapters. Empty
// means invalid input, checked arithmetic overflow, or no finite cycle cover.
std::optional<PeriodicSharedAssignment> buildPeriodicSharedAssignment(const PeriodicAnalysis& analysis);
// Every retained record executes precisely when both endpoints belong to the
// same finite prefix of the unconditional skeleton. The caller guarantees a
// fresh entry, paired logical commands, closed exit, and the existing policy
// for hidden macro reservations. Arbitrary indexed guards are not supported.
// Null means invalid input, an unrepresentable threshold/budget, or no finite
// cycle cover. No scarcity repair or logical-plan mutation is performed.
// The version-two CyclicAllocationAttr uses strategy "shared-cycle-cover",
// budget, order_exact, and entries {record,source,target,lane_begin,lane_count,
// offset}. Its minimum is for the supplied cycle-cover graph only. When startup
// guards prevent uniform barrier reconstruction, order_exact is false.
DictionaryAttr encodePeriodicSharedAllocation(
    const PeriodicAnalysis& analysis, int64_t plan, MLIRContext* context);
FailureOr<PhysicalAllocationPlan> decodePeriodicSharedAllocation(
    func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
