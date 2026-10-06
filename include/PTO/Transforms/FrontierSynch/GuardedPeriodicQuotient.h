// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared weighted quotient circuits for immutable guarded periodic generators.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICQUOTIENT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICQUOTIENT_H
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include <memory>
namespace mlir::pto::frontiersynch {
struct GuardedPeriodicPayload {
    uint32_t pipe = 0;
    RegionExpressions::Id presence = RegionExpressions::invalid;
};
struct GuardedPeriodicRecord {
    uint32_t source = 0, target = 0;
    RegionExpressions::Id displacement = RegionExpressions::invalid;
    RegionExpressions::Id active = RegionExpressions::invalid;
    uint64_t maxDisplacement = 0;
};
struct GuardedPeriodicThreshold {
    RegionExpressions::Id reachable = RegionExpressions::invalid;
    RegionExpressions::Id distance = RegionExpressions::invalid;
};
struct GuardedPeriodicQuotient {
    std::shared_ptr<RegionExpressions> expressions;
    std::string error;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<GuardedPeriodicRecord> records;
    std::vector<GuardedPeriodicRecord> nativePrerequisites;
    // Parallel to original records. Coincident enabled records choose the first
    // minimum-distance representative; guards select exactly the lifted covers.
    std::vector<RegionExpressions::Id> retained;
    // Row-major all-event thresholds; event 2*a is I_a, event 2*a+1 is C_a.
    // Unreachable distances are zero and must be interpreted with reachable.
    std::vector<GuardedPeriodicThreshold> thresholds;
    uint64_t graphEdges = 0;
    std::optional<GuardedPeriodicThreshold> eventThreshold(PeriodicEvent source, PeriodicEvent target) const;
};
// The caller certifies that guards and distances use immutable invocation
// parameters, all presence/active predicates are defined, every enabled
// distance is defined and <= maxDisplacement, and every enabled
// lifted edge is reference-forward (distance > 0 if source >= target).
// Enabled generator instances must generate exactly the invocation's required
// order after endpoint truncation. This routine intersects active with endpoint
// presence and checks constant violations, but does not prove symbolic premises.
// All arithmetic uses checked static bounds for uint64_t circuit intermediates.
// No iterations, slots, distance values or guard valuations are enumerated.
// Construction is O(g + h + m^3), with h fixed native prerequisites, beyond
// the supplied expression DAG. Native prerequisites are not removable demands. No IR mutation.
// Threshold queries are reflexive on present types; finite-prefix clients also
// check both occurrence domains and compare targetIteration-sourceIteration.
GuardedPeriodicQuotient analyzeGuardedPeriodicQuotient(
    std::shared_ptr<RegionExpressions> expressions,
    llvm::ArrayRef<GuardedPeriodicPayload> payloads, llvm::ArrayRef<GuardedPeriodicRecord> records,
    llvm::ArrayRef<GuardedPeriodicRecord> nativePrerequisites = {});
} // namespace mlir::pto::frontiersynch
#endif
