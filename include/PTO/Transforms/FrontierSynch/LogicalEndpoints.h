// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Direct logical endpoint recipes; no physical ID or emitted instruction.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALENDPOINTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALENDPOINTS_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
enum class EndpointKind { Set, Barrier, Wait }; // Canonical order at one cut.
enum class EndpointError { None, InvalidInput, Overflow };
struct CountedLoopResult {
    EndpointError error = EndpointError::None;
    uint64_t value = 0;
};
CountedLoopResult countedTrips(int64_t lower, int64_t upper, int64_t step);
CountedLoopResult countedOrdinal(int64_t lower, int64_t upper, int64_t step, int64_t induction);
struct EndpointRecipe {
    uint32_t record = 0; // Canonical PeriodicAnalysis generator identity.
    uint32_t source = 0;
    uint32_t target = 0;
    uint32_t pipe = 0;
    uint64_t displacement = 0;
    EndpointKind kind = EndpointKind::Set;
    // Set is AFTER source, Barrier/Wait BEFORE target. Template-type coordinate
    // selection precedes the ordinal guard; both are part of endpoint code.
};
// Identity namespace is one plan; composing plans must preserve that namespace.
struct LogicalIdentity {
    uint32_t record = 0;
    uint64_t sourceOrdinal = 0;
};
struct EndpointInstance {
    EndpointKind kind = EndpointKind::Set;
    uint32_t pipe = 0;
    uint32_t type = 0;
    uint64_t ordinal = 0;
    LogicalIdentity identity;
};
struct EndpointEvaluation {
    EndpointError error = EndpointError::None;
    bool active = false;
    EndpointInstance instance;
};
struct LogicalEndpointPlan {
    std::string error;
    std::vector<EndpointRecipe> recipes;
    // Out-of-domain ordinals are inactive. Invalid recipe indices or a failed
    // plan are errors; inactive must not be interpreted as an analysis failure.
    EndpointEvaluation evaluate(uint32_t recipe, uint64_t trips, uint64_t ordinal) const;
};
// Local demands emit barriers before their consumers, including nonadjacent pairs.
// Nonadjacent barriers may order intervening payloads beyond the demand relation.
LogicalEndpointPlan buildLogicalEndpoints(const PeriodicAnalysis& analysis);
// Caller has identified ONE actual legal cut. Sorts SET, barrier,
// WAIT and coalesces duplicate identities and one barrier per pipe. Rejects
// malformed instances without changing input. Does not infer cut equivalence from
// consecutive template types or across region boundaries.
EndpointError canonicalizeCoincidentCut(std::vector<EndpointInstance>& instances);
} // namespace mlir::pto::frontiersynch
#endif
