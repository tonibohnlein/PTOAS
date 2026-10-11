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
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include <functional>
#include <memory>
#include <optional>
namespace mlir::pto::frontiersynch::detail {
class BoundedLifetimeAllocationContext;
// Exact original-coordinate presence in the explicitly identified output arena.
// The callback owns its evaluation context; rollbackable proof attempts must
// not write generated IDs into another retained interface's mutable cache.
struct BoundedAllocationPresence {
    std::shared_ptr<RegionExpressions> expressions;
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent)> presence;
    std::shared_ptr<BoundedLifetimeAllocationContext> invocation;
};
std::shared_ptr<RegionalAllocationSummary> boundedLifetimeRegionalAllocationWithPresence(
    RegionExpressions& producerExpressions, RegionExpressions& outputExpressions,
    const LifetimeWindowInput& window, llvm::ArrayRef<uint8_t> unconditional,
    llvm::ArrayRef<GuardedRankEdge> sourceDemands, RegionExpressions::Id trips,
    const BoundedAllocationPresence& boundary);
struct BoundedLifetimePalette {
    uint32_t source, target, site;
    uint64_t gap = 1;
    SmallVector<int64_t> records;
    bool inactive = false;
};
struct BoundedLifetimeAllocationAssignment {
    std::vector<BoundedLifetimePalette> palettes;
    std::optional<PeriodicSharedAllocation> shared;
    // Uniformly false original guards. These records still require physical
    // mappings, but supply neither live reuse obligations nor lane envelopes.
    std::vector<uint8_t> inactiveRecords;
};
std::optional<BoundedLifetimeAllocationAssignment> boundedLifetimeAllocationAssignmentWithInactive(
    RegionExpressions& expressions, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands,
    llvm::ArrayRef<uint8_t> invocationInactive);
// The assignment is a proof about immutable modeled occurrences. It contains
// original sourceDemand record IDs, not endpoint or physical-ID namespaces.
std::optional<BoundedLifetimeAllocationAssignment> boundedLifetimeAllocationAssignment(
    RegionExpressions& expressions, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> sourceDemands);
} // namespace mlir::pto::frontiersynch::detail
#endif
