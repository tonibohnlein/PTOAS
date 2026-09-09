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
#include <map>
#include <memory>
#include <optional>
#include <string>

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

// All operations retain integer existential variables. Never use projectOut,
// rational shadows, or the pinned union PWMA lexopt implementation here.
// Work accounting bounds query requests/term growth, not time inside MLIR.
class RelationQueries {
    uint64_t remaining, used = 0;
    bool charge(const Relation& relation);
    RelationResult restrictCandidateOrder(const Relation& order, const Relation& candidates, bool sources);

public:
    explicit RelationQueries(uint64_t budget = 8000000) : remaining(budget) {}
    uint64_t work() const { return used; }
    uint64_t remainingWork() const { return remaining; }
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
    std::optional<Relation> issueOrder;
    std::optional<Relation> globalIssueOrder;
    Relation primitive;
    Relation expanded;
    bool fixed = false;
    struct SourceState {
        Relation reached, pending;
        bool saturated = false;
    };
    // A key restricts only the first source coordinate. All other coordinates,
    // guards and parameters remain in the actual issue-order relation. This
    // partition is algebraic; coordinate zero need not be a native phase ID.
    std::map<std::optional<llvm::DynamicAPInt>, SourceState> sources;
    std::optional<Relation> transitions;
    using Coordinate = std::optional<llvm::DynamicAPInt>;
    using OrderBlocks = std::map<std::pair<Coordinate, Coordinate>, Relation>;
    std::shared_ptr<std::optional<OrderBlocks>> indexedIssues = std::make_shared<std::optional<OrderBlocks>>();
    std::shared_ptr<std::optional<OrderBlocks>> indexedGlobal = std::make_shared<std::optional<OrderBlocks>>();
    RelationResult selectOrder(
        bool global, const presburger::PresburgerSet& source, const presburger::PresburgerSet& target,
        RelationQueries& queries);
    QueryStatus proveSparse(const Relation& requirement, RelationQueries& queries, unsigned rounds);

public:
    explicit CompletionQueries(Relation primitiveSupply)
        : known(std::move(primitiveSupply)), primitive(known), expanded(known)
    {}
    // A compact handoff maps source completion to destination issue. Issue
    // order only extends/joins those handoffs; it is never completion by itself.
    CompletionQueries(Relation handoffs, Relation orderedIssues, std::optional<Relation> globalOrder = {})
        : known(std::move(handoffs)),
          issueOrder(std::move(orderedIssues)),
          globalIssueOrder(std::move(globalOrder)),
          primitive(known),
          expanded(Relation::getEmpty(known.getSpace()))
    {}
    QueryStatus prove(const Relation& requirement, RelationQueries& queries, unsigned rounds = 8);
    // New plan, identical immutable issue-order universe. Reuse only the order
    // index; all completion, pending-path and saturation state starts fresh.
    CompletionQueries withHandoffs(Relation handoffs) const
    {
        CompletionQueries next(std::move(handoffs));
        next.issueOrder = issueOrder;
        next.globalIssueOrder = globalIssueOrder;
        next.indexedIssues = indexedIssues;
        next.indexedGlobal = indexedGlobal;
        if (issueOrder)
            next.expanded = Relation::getEmpty(next.known.getSpace());
        return next;
    }
    // Monotonic construction only: retain already proved paths while adding
    // actual handoffs. Reopen every source's delta search, including formerly
    // saturated scopes. Removal/movement must still use withHandoffs().
    QueryStatus addHandoffs(const Relation& additional, RelationQueries& queries);
    // Compact mode exposes proved completion in the most recent query scope.
    // Absence remains unproved; it is not evidence of required synchronization.
    const Relation& supply() const { return issueOrder ? expanded : known; }
    bool fixedPoint() const { return fixed; }
};
} // namespace mlir::pto::logical_sync
#endif
