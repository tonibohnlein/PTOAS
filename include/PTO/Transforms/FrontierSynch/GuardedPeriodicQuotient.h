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
struct GuardedPeriodicFrontier {
    uint32_t pipe = 0, count = 0;
    // Event 2*a is I_a and 2*a+1 is C_a. These are shifted scores
    // rho(v)=min_a(count-rank(a)+count*dist(a,v)), not event distances.
    // Separate seeds supply start-origin and completion-origin queries.
    std::vector<GuardedPeriodicThreshold> starts, completions;
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
    // O(kV) entries. Potential ranks are fixed even when sites are absent.
    std::vector<GuardedPeriodicFrontier> frontiers;
    std::vector<uint32_t> sourceRows, localRanks;
    uint64_t graphEdges = 0;
    uint64_t relaxationCandidates = 0, exclusionCandidates = 0, scalingAdditions = 0;
    // Constructs only the requested scalar threshold. Exporting all pairs is
    // an explicit consumer cost; this index never materializes a pair matrix.
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
// With V=2m vertices, k pipes and E normalized/native edges, construction is
// O(g+kVE) circuit gates beyond supplied expressions and endpoint grouping.
// Grouping uses O(g log(g+1)) comparisons. The shared DAG retains O(g+kVE)
// gates; its frontier outputs occupy O(kV) entries, not O(V^2) pair entries.
// Native prerequisites are not removable demands. No IR mutation.
// Threshold queries are reflexive on present types; finite-prefix clients also
// check both occurrence domains and compare targetIteration-sourceIteration.
GuardedPeriodicQuotient analyzeGuardedPeriodicQuotient(
    std::shared_ptr<RegionExpressions> expressions,
    llvm::ArrayRef<GuardedPeriodicPayload> payloads, llvm::ArrayRef<GuardedPeriodicRecord> records,
    llvm::ArrayRef<GuardedPeriodicRecord> nativePrerequisites = {});
} // namespace mlir::pto::frontiersynch
#endif
