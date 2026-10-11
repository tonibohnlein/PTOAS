// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed ordinary palette proof shared by whole-invocation and regional exports.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATIONINTERNAL_H
#define PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATIONINTERNAL_H
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeAllocation.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicSharedAllocation.h"
namespace mlir::pto::frontiersynch::detail {
struct BoundedLifetimePalette {
    uint32_t source, target, site;
    uint64_t gap = 1;
    SmallVector<int64_t> records;
};
struct BoundedLifetimeAllocationAssignment {
    std::vector<BoundedLifetimePalette> palettes;
    std::optional<PeriodicSharedAllocation> shared;
};
// The assignment is a proof about immutable modeled occurrences. It contains
// original sourceDemand record IDs, not endpoint or physical-ID namespaces.
std::optional<BoundedLifetimeAllocationAssignment> boundedLifetimeAllocationAssignment(
    RegionExpressions& expressions, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands);
} // namespace mlir::pto::frontiersynch::detail
#endif
