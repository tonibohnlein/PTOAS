// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical command-reuse cycle covers, independent of IR and hardware capacity.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICSHAREDALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICSHAREDALLOCATION_H
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::pto::frontiersynch {
using PeriodicReuseMatrix = std::vector<std::vector<std::optional<uint64_t>>>;
enum class PeriodicSharedAllocationStatus {
    Success, InvalidInput, ZeroWeightCycle, NoFiniteCover, Overflow
};
struct PeriodicSharedPhase {
    uint32_t successor = 0, cycle = 0;
    uint64_t laneBegin = 0, laneCount = 0, offset = 0;
};
struct PeriodicSharedCycle {
    uint32_t firstPhase = 0;
    uint64_t laneBegin = 0, laneCount = 0;
};
struct PeriodicSharedAllocation {
    PeriodicSharedAllocationStatus status = PeriodicSharedAllocationStatus::Success;
    std::string error;
    uint64_t budget = 0;
    std::vector<PeriodicSharedPhase> phases;
    std::vector<PeriodicSharedCycle> cycles;
};
// Each finite weight w[a][b] certifies reuse from handoff a at ordinal i to
// handoff b at i+w[a][b]. Missing edges remain unavailable. The supplier proves
// command ordering, compatible activity, and finite-invocation boundary rules.
// Nonnegative weights must give strictly positive total on EVERY finite cycle;
// this routine checks the equivalent absence of a directed zero-weight cycle.
//
// Finds a minimum-weight cycle cover in O(c^3) arbitrary-precision integer
// operations, O(c^2) input storage and O(c) auxiliary words. Neither numerical
// weights nor trip counts are unfolded. The budget is minimal for the supplied
// cycle-cover problem, not for all possible hardware allocation strategies.
//
// A phase's lane is laneBegin + ((ordinal + offset) mod laneCount). Evaluate
// that expression using checked modular addition or a wider integer. All
// phases on one cycle share its disjoint lane range.
// NoFiniteCover is distinct from a valid cover exceeding uint64 representation.
// Failed results contain no partial assignment. Empty input succeeds at budget 0.
PeriodicSharedAllocation allocatePeriodicShared(const PeriodicReuseMatrix& weights);
} // namespace mlir::pto::frontiersynch
#endif
