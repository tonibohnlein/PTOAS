// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Matching on actual guarded occurrences; never reduce an unguarded potential graph.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDCOMPACTMATCHING_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDCOMPACTMATCHING_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "llvm/ADT/APInt.h"
namespace mlir::pto::frontiersynch {
struct GuardedCompactPresence {
    RegionExpressions::Id guard = RegionExpressions::invalid;
    // Original enclosing branch decisions, evaluated at this ordinal. The
    // conjunction must be exactly guard. Empty means no structural evidence.
    std::vector<RegionExpressions::Id> enclosing;
};
using GuardedCompactGuards = std::function<std::optional<GuardedCompactPresence>(
    uint32_t, RegionExpressions::Id)>;
enum class GuardedCompactProof { MandatorySource, IdenticalGuard, EnclosingBranch, Unknown, Local };
struct GuardedCompactRecipe {
    PeriodicRecord original;
    GuardedCompactProof proof = GuardedCompactProof::Unknown;
    RegionExpressions::Id publication = RegionExpressions::invalid;
    RegionExpressions::Id acquisition = RegionExpressions::invalid;
    RegionExpressions::Id sourceIdentity = RegionExpressions::invalid;
    RegionExpressions::Id targetIdentity = RegionExpressions::invalid;
};
struct GuardedCompactMatching {
    std::string error;
    std::vector<GuardedCompactRecipe> records; // Original order/identities, no deletion/deduplication.
    uint64_t guardEvaluations = 0, addedCircuitNodes = 0, constantBits = 0;
    bool sourcePresenceComplete = false;
};
// Supplied guards denote exact original site participation at the requested
// ordinal, uniformly over immutable parameters. Structural conjuncts are
// original enclosing branch facts, not inferred implication assertions.
// Each represented actual lifetime source is no later than the selected source
// on the SAME source-site completion chain; supplied origin-distance intervals
// alone never certify that the selected source executes.
// Unknown presence/local matching preserves every record. No SAT/valuation
// search, potential-graph reduction or runtime enumeration. Guard construction
// is charged separately through the callback; each record evaluates two guards
// and substitutes no more than its supplied circuit/branch-path representation.
GuardedCompactMatching buildGuardedCompactMatching(RegionExpressions& arena,
    RegionExpressions::Id ordinal, RegionExpressions::Id trips, llvm::ArrayRef<PeriodicPayload> payloads,
    llvm::ArrayRef<PeriodicRecord> records, const GuardedCompactGuards& guards);
struct GuardedCompactPreparation {
    GuardedCompactMatching matching;
    std::shared_ptr<RegionExpressions> expressions;
    // Keeps iteration-predicate placeholder values referenced by recipes alive.
    std::shared_ptr<void> expressionOwner;
    std::unique_ptr<PreparedLogicalPlan> plan;
    std::string placementError;
};
// One original counted loop invocation without enclosing loops; sites are supplied in static reference
// order and each names one original phase. Only scf.if ancestors within the loop
// are supported. Exact original branch paths are collected here. Scalar guard
// recovery uses existing cut rules; payloads/future loads are never cloned.
// All preparation is detached and transactional. A placement/presence gap keeps
// matching records and reports no plan. Local records require a separate actual
// executed-adjacency interface; this function deliberately does not place them.
GuardedCompactPreparation prepareGuardedCompactMatching(func::FuncOp function, scf::ForOp loop,
    const PhaseIndex& index, llvm::ArrayRef<const CompoundInstanceElement*> phases,
    llvm::ArrayRef<PeriodicRecord> records);
struct OptionalIndirectReadExcess {
    RegionExpressions::Id contribution = RegionExpressions::invalid;
    // Sum contribution(j) over executed consumers; this circuit is one summand,
    // never a sampled or unrolled sum. allPresent is also a valid skip bound.
    llvm::APInt allPresent{128, 0}, deltaTimesTrips{128, 0};
};
// Only the bounded-indirect-read theorem: mandatory P_i writes disjoint x[i],
// optional Q_j reads x[f(j)], max(0,j-delta)<=f(j)<=j, no other effects/prereqs,
// and publication guard is available. Returns a symbolic participation summand
// plus closed universal bounds. It does not certify those supplied premises or
// extend the fixed-body Phi certificate to arbitrary optional payloads.
OptionalIndirectReadExcess optionalIndirectReadExcess(RegionExpressions& arena,
    RegionExpressions::Id ordinal, RegionExpressions::Id participation, uint64_t delta, uint64_t trips);
} // namespace mlir::pto::frontiersynch
#endif
