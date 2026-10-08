// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGSEQUENCE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGSEQUENCE_H
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
namespace mlir::pto::frontiersynch {
struct BoundingSequenceChild {
    BoundingRegionalResult bounds;
    // TRUSTED producer sidecars for the corresponding selected graph. Storage
    // extrema describe that sign's access specification, not the original
    // effect inventory or storage metadata inherited by a query-only view.
    // The common occurrence callbacks are taken from bounds.context instead.
    std::optional<RegionalAnalysis> lowerExports, upperExports;
    // Own the producer's typed mathematical records independently of exports.
    std::shared_ptr<const void> mathematicalOwner;
    // False requires a producer proof that endpoint placement preserves its
    // selected closure. True never prevents mathematical composition/preparation.
    bool placementMayStrengthen = true;
};
struct BoundingSequenceCrossing {
    SequenceEvent source, target; // Completion -> Start, in strictly later child.
    RegionExpressions::Id active = RegionExpressions::invalid;
    std::vector<RequirementRecordKey> owners; // Original owners before global merging.
};
enum class BoundingSequenceBridges { PhysicalSelectors, SuppliedCrossings };
struct BoundingSequenceSpecification {
    // SuppliedCrossings certifies that crossings already include every required
    // storage bridge for this sign, independently of any physical byte map.
    // Native extrema must still be exact. No physical-storage completeness is
    // exported from this route; a qualified schema owner must adapt it again.
    BoundingSequenceBridges bridges = BoundingSequenceBridges::PhysicalSelectors;
    std::vector<BoundingSequenceCrossing> crossings;
};
struct BoundingSequenceInput {
    std::vector<BoundingSequenceChild> children;
    BoundingSequenceSpecification lower, upper;
    SmallVector<scf::ForOp> enclosing;
    // Original value prerequisites are genuine lower facts too. False means
    // the producer has supplied every needed value link through its sign specs.
    bool reconstructPrerequisites = true;
    // TRUSTED composition premise: each child sandwich holds and the supplied
    // sign-specific access bridges/crossings give Hlower <= Hinput <= Hupper.
    // Equivalent additionally certifies exact upper crossings/access bridges;
    // it is propagated only when every child's upper is equivalent too.
    InputOrderGuarantee guarantee = InputOrderGuarantee::InputOrderCovering;
};
struct BoundingSequenceSign {
    std::string exportError;
    std::optional<SequenceAnalysis> sequence;
    std::optional<RegionalAnalysis> regional;
};
struct BoundingSequenceResult {
    std::string error;
    std::shared_ptr<const BoundingSequenceInput> original;
    BoundingRegionalResult bounds;
    BoundingSequenceSign lower, upper;
    // Mathematical selected bounds do not bound extra order introduced by a
    // consumer-adjacent local barrier. Its weaker reuse proofs remain sound.
    // Never report selected excess as actual-plan excess when this is true.
    bool placementMayStrengthen = true;
};
// No execution expansion. Each sign uses the existing regional composition
// cost plus one reclosure after adding supplied crossings. Child queries remain
// exact selected-graph queries; unavailable exports preserve original snapshots.
// Input/IR are borrowed unchanged, as in RegionalOrderContext.
BoundingSequenceResult composeBoundingSequence(
    func::FuncOp function, const SyncInput& input, BoundingSequenceInput specification);
} // namespace mlir::pto::frontiersynch
#endif
