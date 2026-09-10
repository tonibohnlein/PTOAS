// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCRELATIONS_H
#define PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCRELATIONS_H

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <variant>

namespace mlir::pto::logical_sync {
using Relation = presburger::PresburgerRelation;
// NotEstablished is a negative result of this sufficient query, not proof of
// impossibility. Algebra results are exact; Proved does not certify a program.
enum class QueryStatus { Proved, NotEstablished, Unsupported, BudgetExhausted };
struct RelationResult {
    QueryStatus status = QueryStatus::Unsupported;
    std::optional<Relation> relation;
    std::string reason;
    explicit operator bool() const { return status == QueryStatus::Proved && relation.has_value(); }
};

// A point in a common periodic publication population. Rank is the strict
// within-iteration schedule rank, not the numeric phase identity. One phase
// may have several residues; identical atoms are harmless duplicates.
struct PeriodicPublication {
    int64_t phase, rank, residue;
};

// All operations retain integer existential variables. Never use projectOut,
// rational shadows, or the pinned union PWMA lexopt implementation here.
// Work accounting bounds query requests/term growth, not time inside MLIR.
class RelationQueries {
public:
    // Own the RHS so cached coordinate witnesses can never outlive or silently
    // refer to a different relation. The index is built only after the ordinary
    // composition charge succeeds. CompletionQueries invalidates it whenever a
    // committed monotone addition changes its primitive relation.
    class CompositionRHS {
        friend class RelationQueries;
        friend class CompletionQueries;
        Relation relation;
        struct Index {
            using Constants = std::vector<std::optional<llvm::DynamicAPInt>>;
            std::vector<Constants> joined;
            std::map<llvm::DynamicAPInt, std::vector<unsigned>> fixed;
            std::vector<unsigned> wildcard, all;
            uint64_t chargeCost = 0;
        };
        std::optional<Index> index;
    public:
        explicit CompositionRHS(Relation value) : relation(std::move(value)) {}
        const Relation& value() const { return relation; }
    };
    // Opt-in diagnostics, enabled by PTOAS_LOGICAL_TRACE when this instance is
    // created. Times include nested queries and all result statuses; they must
    // not be summed as exclusive pipeline time. maxInputPieces is the largest
    // total number of input disjuncts in one call, across all of its operands.
    struct PrimitiveStats {
        uint64_t calls = 0, wallNanoseconds = 0, maxInputPieces = 0;
    };
    struct Profile {
        PrimitiveStats normalize, compose, subtract, contains, restrictEndpoints;
        // Shared subtraction/Boolean-containment partitioner attribution;
        // implication is included in partition. These times are not additive
        // exclusive totals and Boolean search is included in contains.
        PrimitiveStats subtractQualification, subtractPartition, subtractImplication;
    };

private:
    class DifferenceEngine;
    friend class CompletionQueries;
    uint64_t remaining, used = 0;
    uint64_t endpointComparisons = 0;
    uint64_t compositionIndexBuilds = 0, compositionIndexPieces = 0;
    uint64_t endpointProjections = 0, endpointProjectionPieces = 0;
    uint64_t differenceCommonRows = 0, differencePartitionPieces = 0, differenceImplicationTests = 0;
    uint64_t booleanPartitionNodes = 0, booleanWitnessLeaves = 0, booleanMaxDepth = 0;
    uint64_t relationEndpointIndexPieces = 0, relationEndpointBucketLookups = 0;
    uint64_t differenceEndpointComparisons = 0, containmentEndpointComparisons = 0;
    uint64_t periodicAtomVisits = 0, periodicSortComparisons = 0, periodicOutputPieces = 0;
    bool profiling;
    Profile queryProfile;
    bool charge(const Relation& relation, uint64_t* chargedCost = nullptr);
    RelationResult composeImpl(const Relation& first, const Relation& second,
                               std::optional<CompositionRHS::Index>& index);
    RelationResult restrictCandidateOrder(const Relation& order, const Relation& candidates, bool sources);

public:
    explicit RelationQueries(uint64_t budget = 8000000);
    uint64_t work() const { return used; }
    uint64_t remainingWork() const { return remaining; }
    uint64_t endpointComparisonCount() const { return endpointComparisons; }
    uint64_t compositionIndexBuildCount() const { return compositionIndexBuilds; }
    uint64_t compositionIndexPieceCount() const { return compositionIndexPieces; }
    // Completion-owned primitive/reached/additional endpoint caches only;
    // arbitrary import and per-demand projections are outside these counters.
    uint64_t endpointProjectionCount() const { return endpointProjections; }
    uint64_t endpointProjectionPieceCount() const { return endpointProjectionPieces; }
    uint64_t differenceCommonRowCount() const { return differenceCommonRows; }
    uint64_t differencePartitionPieceCount() const { return differencePartitionPieces; }
    uint64_t differenceImplicationTestCount() const { return differenceImplicationTests; }
    // Exact Boolean-containment DFS only. Depth counts selected RHS pieces,
    // not input dimensions; matrix size can grow along a retained branch.
    uint64_t booleanPartitionNodeCount() const { return booleanPartitionNodes; }
    uint64_t booleanWitnessLeafCount() const { return booleanWitnessLeaves; }
    uint64_t booleanMaximumDepth() const { return booleanMaxDepth; }
    uint64_t relationEndpointIndexPieceCount() const { return relationEndpointIndexPieces; }
    uint64_t relationEndpointBucketLookupCount() const { return relationEndpointBucketLookups; }
    uint64_t differenceEndpointComparisonCount() const { return differenceEndpointComparisons; }
    uint64_t containmentEndpointComparisonCount() const { return containmentEndpointComparisons; }
    uint64_t periodicAtomVisitCount() const { return periodicAtomVisits; }
    uint64_t periodicSortComparisonCount() const { return periodicSortComparisons; }
    uint64_t periodicOutputPieceCount() const { return periodicOutputPieces; }
    bool profilingEnabled() const { return profiling; }
    const Profile& profile() const { return queryProfile; }
    bool spend(uint64_t amount)
    {
        if (amount > remaining) {
            used += remaining;
            remaining = 0;
            return false;
        }
        remaining -= amount;
        used += amount;
        return true;
    }
    RelationResult compose(const Relation& first, const Relation& second);
    RelationResult compose(const Relation& first, CompositionRHS& second);
    RelationResult normalize(const Relation& relation);
    // Prune only endpoint coordinates proved incompatible with the candidate
    // sets. The result may retain extra edges; it never removes a matching one.
    RelationResult restrictEndpoints(
        const Relation& relation, const presburger::PresburgerSet& sources, const presburger::PresburgerSet& targets);
    RelationResult subtract(const Relation& from, const Relation& remove);
    QueryStatus contains(const Relation& supply, const Relation& requirement);

    // R: source -> sink. Both Before arguments must be actual strict schedule
    // order over qualified occurrence domains in the SAME
    // execution (including guards and invocation coordinates). Keep byte in
    // the sink coordinates until per-byte latest producers have been chosen.
    // Coverage of every original sink is checked: an unbounded predecessor
    // family must not silently become an empty set of requirements.
    RelationResult latestSources(const Relation& requirements, const Relation& sourceBefore);
    RelationResult firstTargets(const Relation& requirements, const Relation& targetBefore);
    // Exact next-publication relation for the WHOLE population described by
    // common and atoms. common is one set disjunct: phase is unconstrained;
    // every other occurrence coordinate except iteration is a fixed constant;
    // iteration has a finite contiguous integer interval given by +/-1 rows
    // with shared parameter coefficients. Parameter-only guards/existential
    // locals are allowed; iteration-dependent locals or holes are unsupported.
    // Template coefficients must fit int64; output arithmetic remains exact.
    // The helper checks this grammar, residues and phase/rank consistency.
    // A zero-disjunct common set returns empty before validating unused atoms.
    // It does not certify that these atoms describe a native event program:
    // the caller must establish exact equality with ALL selected publications
    // and supply actual ranks, coordinate meaning and shared SSA bindings.
    // Construction sorts K atoms and links cyclic neighbours: O(K log K) plus
    // O(K * template size) output, at most K pieces, independent of period's
    // magnitude. Native selected-domain equivalence checking is separate work.
    // Empty/clipped populations have no invented terminal successors. Failure
    // returns no partial relation and never proves event reuse by itself.
    RelationResult commonPeriodSuccessors(
        const presburger::PresburgerSet& common, unsigned phaseCoordinate,
        unsigned iterationCoordinate, int64_t period, llvm::ArrayRef<PeriodicPublication> atoms);
    // latest: source -> target; sourceThrough is inclusive source order.
    // Only acquisitions in that execution can dominate a later demand.
    RelationResult staircase(const Relation& latest, const Relation& sourceThrough,
                             const Relation& targetBefore);
};

// An exact underapproximation of completion for ONE immutable logical plan.
// Early success does not mark a fixed point. A subsequent reuse query can
// continue composition. Replacing a plan requires a new instance.
class CompletionQueries {
    Relation known;
    std::shared_ptr<const Relation> issueOrder;
    std::shared_ptr<const Relation> globalIssueOrder;
    RelationQueries::CompositionRHS primitive;
    std::optional<presburger::PresburgerSet> primitiveSources, primitiveTargets;
    Relation expanded;
    bool fixed = false;
    struct MaterializedFrontier { Relation delta; };
    struct DeferredFrontier {
        // Exactly generated \ before. before is the PRE-extension reached
        // relation; using the current reached union would erase the delta.
        Relation generated, before;
        uint64_t retainedCells;
    };
    struct SourceState {
        Relation reached;
        std::variant<MaterializedFrontier, DeferredFrontier> frontier;
        bool saturated = false;
        std::optional<presburger::PresburgerSet> reachedTargets;
    };
    uint64_t frontierExtensions = 0, frontierDifferences = 0;
    uint64_t frontierDeferrals = 0, frontierResolutions = 0, frontierResets = 0;
    uint64_t deferredSources = 0, deferredCells = 0, peakDeferredCells = 0;
    QueryStatus resolveFrontier(SourceState& state, RelationQueries& queries);
    // A key restricts only the first source coordinate. All other coordinates,
    // guards and parameters remain in the actual issue-order relation. This
    // partition is algebraic; coordinate zero need not be a native phase ID.
    std::map<std::optional<llvm::DynamicAPInt>, SourceState> sources;
    std::optional<RelationQueries::CompositionRHS> transitions;
    using Coordinate = std::optional<llvm::DynamicAPInt>;
    using OrderKey = std::pair<Coordinate, Coordinate>;
    struct OrderBlocks {
        std::map<OrderKey, Relation> values;
        std::map<Coordinate, std::vector<OrderKey>> bySource, byTarget;
    };
    std::shared_ptr<std::optional<OrderBlocks>> indexedIssues = std::make_shared<std::optional<OrderBlocks>>();
    std::shared_ptr<std::optional<OrderBlocks>> indexedGlobal = std::make_shared<std::optional<OrderBlocks>>();
    RelationResult selectOrder(
        bool global, const presburger::PresburgerSet& source, const presburger::PresburgerSet& target,
        RelationQueries& queries);
    QueryStatus proveSparse(const Relation& requirement, RelationQueries& queries, unsigned rounds);
    const presburger::PresburgerSet& primitiveEndpoints(bool sources, RelationQueries& queries);

public:
    // An owning callback for ONE immutable occurrence universe. A successful
    // answer contains only actual inclusive issue-order edges and includes
    // EVERY such edge whose endpoints lie in the candidate sets. It may keep
    // extra actual edges, like the explicit relation selector. Preserve all
    // guards, parameters and invocation coordinates; global order is used only
    // by rejection screens and must never supply completion on its own.
    // The provider owns (or shares ownership of) every captured program fact,
    // charges its work through queries, and reports incomplete queries explicitly.
    using OrderSelection = std::function<RelationResult(
        bool global, const presburger::PresburgerSet& sources,
        const presburger::PresburgerSet& targets, RelationQueries& queries)>;

private:
    std::shared_ptr<const OrderSelection> orderSelection;
    bool hasGlobalSelection = false;
    uint64_t indexedLookups = 0;
    bool sparse() const { return bool(issueOrder) || bool(orderSelection); }
    bool hasGlobalOrder() const { return bool(globalIssueOrder) || hasGlobalSelection; }

public:
    explicit CompletionQueries(Relation primitiveSupply)
        : known(std::move(primitiveSupply)), primitive(known), expanded(known)
    {}
    // A compact handoff maps source completion to destination issue. Issue
    // order only extends/joins those handoffs; it is never completion by itself.
    CompletionQueries(Relation handoffs, Relation orderedIssues, std::optional<Relation> globalOrder = {})
        : known(std::move(handoffs)),
          issueOrder(std::make_shared<const Relation>(std::move(orderedIssues))),
          globalIssueOrder(globalOrder ? std::make_shared<const Relation>(std::move(*globalOrder)) : nullptr),
          primitive(known),
          expanded(Relation::getEmpty(known.getSpace()))
    {}
    CompletionQueries(Relation handoffs, OrderSelection select, bool globalOrderAvailable = false)
        : known(std::move(handoffs)), primitive(known),
          expanded(Relation::getEmpty(known.getSpace())),
          orderSelection(std::make_shared<const OrderSelection>(std::move(select))),
          hasGlobalSelection(globalOrderAvailable)
    {}
    QueryStatus prove(const Relation& requirement, RelationQueries& queries, unsigned rounds = 8);
    // New plan, identical immutable issue-order universe. Reuse only the order
    // storage/index/provider; completion, pending paths and saturation reset.
    CompletionQueries withHandoffs(Relation handoffs) const
    {
        CompletionQueries next(std::move(handoffs));
        next.issueOrder = issueOrder;
        next.globalIssueOrder = globalIssueOrder;
        next.indexedIssues = indexedIssues;
        next.indexedGlobal = indexedGlobal;
        next.orderSelection = orderSelection;
        next.hasGlobalSelection = hasGlobalSelection;
        if (sparse())
            next.expanded = Relation::getEmpty(next.known.getSpace());
        return next;
    }
    // Monotonic construction only: retain already proved paths while adding
    // actual handoffs. Seed new direct paths and paths entering added handoffs
    // for compact frontiers, preserving deferred/materialized state; large
    // frontiers may use the bounded conservative replay. Removal/movement
    // must still use withHandoffs().
    QueryStatus addHandoffs(const Relation& additional, RelationQueries& queries);
    // Compact mode exposes proved completion in the most recent query scope.
    // Absence remains unproved; it is not evidence of required synchronization.
    const Relation& supply() const { return sparse() ? expanded : known; }
    bool fixedPoint() const { return fixed; }
    // Individual source/target block probes, excluding group lookups and
    // algebra/solver work. Narrow queries must not scan the entire index.
    uint64_t orderIndexLookups() const { return indexedLookups; }
    uint64_t frontierExtensionCount() const { return frontierExtensions; }
    uint64_t frontierDifferenceCount() const { return frontierDifferences; }
    uint64_t frontierDeferralCount() const { return frontierDeferrals; }
    uint64_t frontierResolutionCount() const { return frontierResolutions; }
    uint64_t frontierResetCount() const { return frontierResets; }
    uint64_t deferredSourceCount() const { return deferredSources; }
    uint64_t deferredCellCount() const { return deferredCells; }
    uint64_t peakDeferredCellCount() const { return peakDeferredCells; }
    // Test/diagnostic inspection only; not part of ordinary query evaluation.
    unsigned saturatedSourceCount() const {
        unsigned count = 0;
        for (const auto& [scope, state] : sources) { (void)scope; count += state.saturated; }
        return count;
    }
};
} // namespace mlir::pto::logical_sync
#endif
