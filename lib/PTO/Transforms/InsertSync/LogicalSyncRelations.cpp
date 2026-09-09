// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"

using namespace mlir::pto::logical_sync;
namespace {
RelationResult failure(QueryStatus status, const char* reason) { return {status, std::nullopt, reason}; }
bool sameSpace(const Relation& a, const Relation& b) {
    return a.getSpace().isCompatible(b.getSpace());
}
}

bool RelationQueries::charge(const Relation& relation) {
    uint64_t cost = 1;
    for (const auto& piece : relation.getAllDisjuncts()) {
        uint64_t rows = uint64_t(piece.getNumEqualities()) + piece.getNumInequalities() + 1;
        uint64_t cols = uint64_t(piece.getNumVars()) + 1;
        if (rows > remaining || cols > remaining || rows > (remaining - std::min(cost, remaining)) / cols) {
            used += remaining; remaining = 0; return false;
        }
        cost += rows * cols;
    }
    if (cost > remaining) { used += remaining; remaining = 0; return false; }
    remaining -= cost; used += cost; return true;
}

RelationResult RelationQueries::compose(const Relation& first, const Relation& second) {
    if (!first.getSpace().getRangeSpace().isCompatible(second.getSpace().getDomainSpace()))
        return failure(QueryStatus::Unsupported, "incompatible composition spaces");
    if (!charge(first) || !charge(second)) return failure(QueryStatus::BudgetExhausted, "composition budget");
    // Charge the Cartesian product BEFORE asking MLIR to construct it.
    for (unsigned i = 0; i < first.getNumDisjuncts(); ++i)
        if (!charge(second)) return failure(QueryStatus::BudgetExhausted, "composition product budget");
    Relation result = first;
    result.compose(second); // IntegerRelation::compose keeps joined coordinates as locals.
    if (!charge(result)) return failure(QueryStatus::BudgetExhausted, "composition result budget");
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult RelationQueries::subtract(const Relation& from, const Relation& remove) {
    if (!sameSpace(from, remove)) return failure(QueryStatus::Unsupported, "incompatible difference spaces");
    if (!charge(from) || !charge(remove)) return failure(QueryStatus::BudgetExhausted, "difference budget");
    // Exact conversion of existential integer witnesses to division locals.
    // This satisfies difference's documented RHS precondition. This conversion
    // uses per-polyhedron lexmin internally, not union PWMA lexopt.
    Relation qualified = remove.computeReprWithOnlyDivLocals();
    if (!charge(qualified)) return failure(QueryStatus::BudgetExhausted, "division normalization budget");
    Relation result = from.subtract(qualified);
    if (!charge(result)) return failure(QueryStatus::BudgetExhausted, "difference result budget");
    return {QueryStatus::Proved, std::move(result), {}};
}

QueryStatus RelationQueries::contains(const Relation& supply, const Relation& requirement) {
    auto missing = subtract(requirement, supply);
    if (!missing) return missing.status;
    return missing.relation->isIntegerEmpty() ? QueryStatus::Proved : QueryStatus::NotEstablished;
}

RelationResult RelationQueries::latestSources(const Relation& requirements, const Relation& sourceBefore) {
    auto dominated = compose(sourceBefore, requirements);
    if (!dominated) return dominated;
    auto latest = subtract(requirements, *dominated.relation);
    if (!latest) return latest;
    auto complete = contains(latest.relation->getRangeSet(), requirements.getRangeSet());
    if (complete != QueryStatus::Proved)
        return failure(complete, "not every sink has an established latest producer");
    return latest;
}

RelationResult RelationQueries::firstTargets(const Relation& requirements, const Relation& targetBefore) {
    auto dominated = compose(requirements, targetBefore);
    if (!dominated) return dominated;
    auto first = subtract(requirements, *dominated.relation);
    if (!first) return first;
    auto complete = contains(first.relation->getDomainSet(), requirements.getDomainSet());
    if (complete != QueryStatus::Proved)
        return failure(complete, "not every source has an established next overwrite");
    return first;
}

RelationResult RelationQueries::staircase(const Relation& latest, const Relation& sourceThrough,
                                         const Relation& targetBefore) {
    auto prefix = compose(sourceThrough, latest);
    if (!prefix) return prefix;
    auto dominated = compose(*prefix.relation, targetBefore);
    if (!dominated) return dominated;
    return subtract(latest, *dominated.relation);
}

QueryStatus CompletionQueries::prove(const Relation& requirement, RelationQueries& queries, unsigned rounds) {
    auto covered = queries.contains(known, requirement);
    if (covered != QueryStatus::NotEstablished || fixed) return covered;
    for (unsigned round = 0; round < rounds; ++round) {
        auto paths = queries.compose(known, known);
        if (!paths) return paths.status;
        auto added = queries.subtract(*paths.relation, known);
        if (!added) return added.status;
        if (added.relation->isIntegerEmpty()) { fixed = true; return QueryStatus::NotEstablished; }
        known.unionInPlace(*added.relation);
        covered = queries.contains(known, requirement);
        if (covered != QueryStatus::NotEstablished) return covered;
    }
    return QueryStatus::NotEstablished;
}
