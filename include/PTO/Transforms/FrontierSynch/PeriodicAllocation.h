// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Periodic canonical-handoff budgets and direction-local cyclic offsets.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PERIODICALLOCATION_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
enum class AllocationError { None, InvalidInput, Overflow };
struct AllocationNumber {
    AllocationError error = AllocationError::None;
    std::optional<uint64_t> value = uint64_t(0); // Null means proven infinity, not error.
};
struct AllocationDecision {
    AllocationError error = AllocationError::None;
    bool sufficient = false;
};
struct DirectedHandoff {
    uint32_t record = 0; // Canonical periodic generator ID, matching the logical endpoint identity.
    uint32_t source = 0;
    uint32_t target = 0;
    uint64_t displacement = 0;
    std::optional<uint64_t> firstReuse; // nu_r; null means no future reusable handoff.
};
struct DirectedAllocation {
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    uint32_t payloadTypes = 0;
    std::vector<DirectedHandoff> handoffs; // Source order; phase r is its array index.
    std::optional<uint64_t> uniformBudget = uint64_t(0);
    AllocationNumber handoffCount(uint64_t payloadPrefixLength) const;
    AllocationNumber finiteBudget(uint64_t payloadPrefixLength) const;
    // Missing prefix requests uniform certification over arbitrarily long prefixes.
    AllocationDecision capacitySuffices(uint64_t capacity,
                                       std::optional<uint64_t> payloadPrefixLength = std::nullopt) const;
    // Computes a local offset ONLY. Caller must separately certify capacity and
    // map offsets to eligible hardware resources. This is not a physical ID.
    AllocationNumber localOffset(uint32_t phase, uint64_t sourcePeriod, uint64_t capacity) const;
};
struct PeriodicAllocation {
    std::string error;
    std::vector<DirectedAllocation> directions;
};
// For the canonical direct plan on these covers, under the model's local
// adjacency and command contracts. No exceptional entry/interface reuse paths.
// Validates ordered/unique cross-pipe endpoints, including periodic wrap.
// No assumed hardware capacity or disjointness between actual ID resources.
// A supplied arbitrary logical plan is outside this interface's contract.
PeriodicAllocation buildPeriodicAllocation(const PeriodicAnalysis& analysis);
} // namespace mlir::pto::frontiersynch
#endif
