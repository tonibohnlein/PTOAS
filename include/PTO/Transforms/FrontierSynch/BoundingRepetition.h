// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGREPETITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGREPETITION_H
#include "PTO/Transforms/FrontierSynch/BoundingSequence.h"
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
namespace mlir::pto::frontiersynch {
struct BoundingRepeatedCrossing {
    RepeatedCrossing edge;
    std::vector<RequirementRecordKey> owners;
};
struct BoundingRepetitionSpecification {
    BoundingSequenceBridges bridges = BoundingSequenceBridges::PhysicalSelectors;
    std::vector<BoundingRepeatedCrossing> crossings;
};
struct BoundingRepetitionInput {
    // A common invariant executed period, with explicit phase descriptions.
    // Every phase supplies both selected queries and sign-specific boundaries.
    // Original occurrence identities are never collapsed into an atomic body.
    std::vector<BoundingSequenceChild> phases;
    BoundingRepetitionSpecification lower, upper;
    RegionExpressions::Id trips = RegionExpressions::invalid;
    RegionExpressions::Id begin = RegionExpressions::invalid;
    std::optional<uint64_t> maximumLength;
    SmallVector<scf::ForOp> enclosing;
    // TRUSTED uniformity/coverage premise: control, presence, inner counts and
    // graph/guard expressions are invariant across periods; all lower internal
    // and crossing edges are genuine, upper edges cover ALL actual demands,
    // including nonconsecutive periods and carried prerequisites. Native first/
    // last extrema are complete. Each present pipe recurs in every full period.
    // SuppliedCrossings additionally owns complete sign-specific storage bridges.
    // Distance-zero relations belong to the selected period body, not this list.
    InputOrderGuarantee guarantee = InputOrderGuarantee::InputOrderCovering;
};
struct BoundingPeriodDistance {
    RegionExpressions::Id reachable, representable, value;
    // reachable && !representable means finite beyond UINT64_MAX, not infinity.
    // value is meaningful only under representable. Guards are in the body arena.
};
struct BoundingRepetitionCost {
    uint64_t bodyQueries = 0, relaxations = 0, deletionTests = 0;
};
class BoundingRepeatedQuery {
public:
    const RegionalAnalysis& body() const;
    llvm::ArrayRef<RegionalEvent> ports() const;
    const BoundingRepetitionCost& cost() const;
    // Minimum period displacement, including zero when the body path exists.
    std::optional<BoundingPeriodDistance> distance(RegionalEvent source, RegionalEvent target);
    std::optional<RegionExpressions::Id> across(
        const RegionalEvent& source, const RegionalEvent& target, RegionExpressions::Id positiveGap);
private:
    struct State;
    explicit BoundingRepeatedQuery(std::shared_ptr<State> implementation);
    std::shared_ptr<State> state;
    friend std::shared_ptr<BoundingRepeatedQuery> buildBoundingRepeatedQuery(
        RegionalAnalysis, std::vector<RegionalEvent>, std::vector<RepeatedCrossing>, std::string&);
};
// Exact graph adapter, not an input-coverage recognizer. Ports include every
// crossing endpoint, may contain semantically coincident guarded aliases, and
// retain original body identities. The body and crossings include native wraps.
// O(P^2) body queries and O(P^3) min-plus gates; each arbitrary-event query adds
// O(P^2+C) shared circuit work. No trip count or distance range is enumerated.
std::shared_ptr<BoundingRepeatedQuery> buildBoundingRepeatedQuery(
    RegionalAnalysis body, std::vector<RegionalEvent> ports,
    std::vector<RepeatedCrossing> crossings, std::string& error);
struct BoundingRepetitionSign {
    std::string exportError;
    std::optional<RepeatedRegionAnalysis> repeated;
    std::shared_ptr<BoundingRepeatedQuery> query;
    // Includes all temporary deletion snapshots and initial final-index build.
    // Later arbitrary queries are additionally charged by query->cost().
    BoundingRepetitionCost constructionCost;
};
struct BoundingRepetitionResult {
    std::string error;
    std::shared_ptr<const BoundingRepetitionInput> original;
    BoundingRegionalResult bounds;
    BoundingRepetitionSign lower, upper;
    // Separate from mathematical closure/reduction, as for BoundingSequence.
    bool placementMayStrengthen = true;
};
// Sequential guarded deletion excludes each whole candidate template. Every
// accepted deletion preserves closure even with coincident endpoint aliases;
// the general route reports Partial rather than claiming unique pointwise covers.
// With C crossing templates and P common ports: O((C+1)*P^3+C^2*P)
// circuit work, O((C+1)*P^2+C^2) body-query requests before memo sharing.
// Body backend cost and all retained arena gates are charged separately.
// Original records and schema owners survive unavailable exports. Existing
// repeated preparation preserves source tuples; nonunit allocation is unavailable.
BoundingRepetitionResult repeatBoundingRegion(func::FuncOp function, scf::ForOp loop,
    const SyncInput& input, BoundingRepetitionInput specification);
} // namespace mlir::pto::frontiersynch
#endif
