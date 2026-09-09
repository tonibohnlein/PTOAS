// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.


#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"
#include <map>
#include <set>

using namespace mlir::pto::logical_sync;
namespace {
RelationResult failure(QueryStatus status, const char* reason) { return {status, std::nullopt, reason}; }
bool sameSpace(const Relation& a, const Relation& b) {
    return a.getSpace().isCompatible(b.getSpace());
}
// Substitute only unit-coefficient local definitions. In particular, a local
// witnessing x=2*k must survive: rational projection would lose slot parity.
struct CompactPiece : mlir::presburger::IntegerRelation {
    explicit CompactPiece(const IntegerRelation& piece) : IntegerRelation(piece) {}
    void compact()
    {
        removeRedundantLocalVars();
        removeDuplicateDivs();
        removeTrivialRedundancy();
        simplify();
    }
};
using Constants = std::vector<std::optional<llvm::DynamicAPInt>>;
Constants fixedCoordinates(const mlir::presburger::IntegerRelation& piece, unsigned offset, unsigned count)
{
    Constants result(count);
    for (unsigned i = 0; i < piece.getNumEqualities(); ++i) {
        auto row = piece.getEquality(i);
        for (unsigned j = 0; j < count; ++j) {
            if (row[offset + j] != 1 && row[offset + j] != -1)
                continue;
            bool isolated = true;
            for (unsigned k = 0; k < piece.getNumVars(); ++k)
                if (k != offset + j && row[k] != 0) {
                    isolated = false;
                    break;
                }
            if (isolated)
                result[j] = row[offset + j] == 1 ? -row.back() : row.back();
        }
    }
    return result;
}
bool incompatible(const Constants& a, const Constants& b)
{
    for (unsigned i = 0; i < a.size(); ++i)
        if (a[i] && b[i] && *a[i] != *b[i])
            return true;
    return false;
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

RelationResult RelationQueries::normalize(const Relation& relation)
{
    if (!charge(relation))
        return failure(QueryStatus::BudgetExhausted, "normalization budget");
    auto result = Relation::getEmpty(relation.getSpace());
    for (const auto& input : relation.getAllDisjuncts()) {
        CompactPiece piece(input);
        piece.compact();
        if (piece.isEmpty())
            continue;
        bool duplicate = false;
        for (const auto& old : result.getAllDisjuncts()) {
            if (!spend(1))
                return failure(QueryStatus::BudgetExhausted, "normalization deduplication budget");
            if (piece.isObviouslyEqual(old)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
            result.unionInPlace(Relation(piece));
    }
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult RelationQueries::compose(const Relation& first, const Relation& second) {
    if (!first.getSpace().getRangeSpace().isCompatible(second.getSpace().getDomainSpace()))
        return failure(QueryStatus::Unsupported, "incompatible composition spaces");
    if (!first.getNumDisjuncts() || !second.getNumDisjuncts()) {
        auto empty = Relation::getEmpty(
            mlir::presburger::PresburgerSpace::getRelationSpace(
                first.getNumDomainVars(), second.getNumRangeVars(), first.getNumSymbolVars()));
        if (!charge(empty))
            return failure(QueryStatus::BudgetExhausted, "empty composition budget");
        return {QueryStatus::Proved, std::move(empty), {}};
    }
    if (!charge(first) || !charge(second)) return failure(QueryStatus::BudgetExhausted, "composition budget");
    std::vector<Constants> joined;
    std::map<llvm::DynamicAPInt, std::vector<unsigned>> indexed;
    std::vector<unsigned> wildcard, all;
    for (const auto& piece : second.getAllDisjuncts()) {
        joined.push_back(fixedCoordinates(piece, 0, piece.getNumDomainVars()));
        unsigned index = joined.size() - 1;
        all.push_back(index);
        if (!joined.back().empty() && joined.back().front())
            indexed[*joined.back().front()].push_back(index);
        else
            wildcard.push_back(index);
    }
    auto result = Relation::getEmpty(
        mlir::presburger::PresburgerSpace::getRelationSpace(
            first.getSpace().getNumDomainVars(), second.getSpace().getNumRangeVars(),
            first.getSpace().getNumSymbolVars()));
    std::map<size_t, std::vector<unsigned>> fingerprints;
    for (const auto& left : first.getAllDisjuncts()) {
        auto constants = fixedCoordinates(left, left.getNumDomainVars(), left.getNumRangeVars());
        auto candidates = wildcard;
        if (constants.empty() || !constants.front())
            candidates = all;
        else if (auto found = indexed.find(*constants.front()); found != indexed.end())
            candidates.insert(candidates.end(), found->second.begin(), found->second.end());
        for (unsigned j : candidates) {
            const auto& right = second.getAllDisjuncts()[j];
            // Bound enumeration even when the joined coordinates cannot match.
            if (!remaining)
                return failure(QueryStatus::BudgetExhausted, "composition enumeration budget");
            --remaining;
            ++used;
            if (incompatible(constants, joined[j]))
                continue;
            if (!charge(Relation(left)) || !charge(Relation(right)))
                return failure(QueryStatus::BudgetExhausted, "composition product budget");
            CompactPiece piece(left);
            piece.compose(right); // Joined coordinates remain integer locals.
            if (!charge(Relation(piece)))
                return failure(QueryStatus::BudgetExhausted, "composition local budget");
            piece.compact();
            if (piece.isEmpty())
                continue;
            auto fingerprint =
                llvm::hash_combine(piece.getNumEqualities(), piece.getNumInequalities(), piece.getNumVars());
            for (unsigned row = 0; row < piece.getNumEqualities(); ++row)
                fingerprint = llvm::hash_combine(
                    fingerprint,
                    llvm::hash_combine_range(piece.getEquality(row).begin(), piece.getEquality(row).end()));
            for (unsigned row = 0; row < piece.getNumInequalities(); ++row)
                fingerprint = llvm::hash_combine(
                    fingerprint,
                    llvm::hash_combine_range(piece.getInequality(row).begin(), piece.getInequality(row).end()));
            bool duplicate = false;
            auto& bucket = fingerprints[size_t(fingerprint)];
            for (unsigned index : bucket) {
                if (!remaining)
                    return failure(QueryStatus::BudgetExhausted, "composition deduplication budget");
                --remaining;
                ++used;
                if (piece.isObviouslyEqual(result.getAllDisjuncts()[index])) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                bucket.push_back(result.getNumDisjuncts());
                result.unionInPlace(Relation(piece));
            }
        }
    }
    // Different witnesses often describe the same endpoint relation. Coalesce
    // only small local-free buckets; the pinned implementation does not merge
    // local-bearing pieces. Bucket separation is solely a work optimization.
    using EndpointKey = std::pair<std::optional<llvm::DynamicAPInt>, std::optional<llvm::DynamicAPInt>>;
    std::map<EndpointKey, Relation> groups;
    for (const auto& piece : result.getAllDisjuncts()) {
        auto a = fixedCoordinates(piece, 0, piece.getNumDomainVars());
        auto b = fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
        EndpointKey key{a.empty() ? std::nullopt : a.front(), b.empty() ? std::nullopt : b.front()};
        auto [it, inserted] = groups.try_emplace(key, Relation::getEmpty(result.getSpace()));
        (void)inserted;
        it->second.unionInPlace(Relation(piece));
    }
    auto compact = Relation::getEmpty(result.getSpace());
    for (auto& [key, group] : groups) {
        (void)key;
        if (group.getNumDisjuncts() > 1 && group.getNumDisjuncts() <= 32 &&
            llvm::all_of(group.getAllDisjuncts(), [](const auto& p) { return p.getNumLocalVars() == 0; })) {
            for (unsigned i = 0; i < group.getNumDisjuncts(); ++i)
                if (!charge(group))
                    return failure(QueryStatus::BudgetExhausted, "composition coalescing budget");
            group = group.coalesce();
        }
        compact.unionInPlace(group);
    }
    result = std::move(compact);
    if (!charge(result)) return failure(QueryStatus::BudgetExhausted, "composition result budget");
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult RelationQueries::subtract(const Relation& from, const Relation& remove) {
    if (!sameSpace(from, remove)) return failure(QueryStatus::Unsupported, "incompatible difference spaces");
    if (!from.getNumDisjuncts()) {
        if (!charge(from))
            return failure(QueryStatus::BudgetExhausted, "empty difference budget");
        return {QueryStatus::Proved, from, {}};
    }
    if (!charge(from) || !charge(remove)) return failure(QueryStatus::BudgetExhausted, "difference budget");
    // Distribute difference over the left union, ignoring only right pieces
    // proved disjoint by fixed endpoint coordinates. Qualify each distinct
    // right subset once; unrelated phase pairs need no integer elimination.
    std::vector<Constants> endpoints;
    unsigned dimensions = from.getNumDomainVars() + from.getNumRangeVars();
    for (const auto& piece : remove.getAllDisjuncts())
        endpoints.push_back(fixedCoordinates(piece, 0, dimensions));
    std::map<std::vector<unsigned>, Relation> qualifiedSubsets;
    auto result = Relation::getEmpty(from.getSpace());
    for (const auto& piece : from.getAllDisjuncts()) {
        auto coordinates = fixedCoordinates(piece, 0, dimensions);
        std::vector<unsigned> selected;
        for (unsigned i = 0; i < endpoints.size(); ++i) {
            if (!spend(1))
                return failure(QueryStatus::BudgetExhausted, "difference endpoint budget");
            if (!incompatible(coordinates, endpoints[i]))
                selected.push_back(i);
        }
        if (selected.empty()) {
            result.unionInPlace(Relation(piece));
            continue;
        }
        auto found = qualifiedSubsets.find(selected);
        if (found == qualifiedSubsets.end()) {
            auto subset = Relation::getEmpty(remove.getSpace());
            for (unsigned i : selected)
                subset.unionInPlace(Relation(remove.getAllDisjuncts()[i]));
            // Exact conversion satisfies difference's division-local RHS
            // precondition; this is not rational projection or union lexopt.
            auto qualified = subset.computeReprWithOnlyDivLocals();
            if (!charge(qualified))
                return failure(QueryStatus::BudgetExhausted, "division normalization budget");
            found = qualifiedSubsets.emplace(std::move(selected), std::move(qualified)).first;
        }
        result.unionInPlace(Relation(piece).subtract(found->second));
    }
    return normalize(result);
}

QueryStatus RelationQueries::contains(const Relation& supply, const Relation& requirement) {
    auto missing = subtract(requirement, supply);
    if (!missing) return missing.status;
    return missing.relation->isIntegerEmpty() ? QueryStatus::Proved : QueryStatus::NotEstablished;
}

RelationResult RelationQueries::latestSources(const Relation& requirements, const Relation& sourceBefore) {
    auto order = restrictCandidateOrder(sourceBefore, requirements, true);
    if (!order)
        return order;
    auto dominated = compose(*order.relation, requirements);
    if (!dominated) return dominated;
    auto latest = subtract(requirements, *dominated.relation);
    if (!latest) return latest;
    auto complete = contains(latest.relation->getRangeSet(), requirements.getRangeSet());
    if (complete != QueryStatus::Proved)
        return failure(complete, "not every sink has an established latest producer");
    return latest;
}

RelationResult RelationQueries::firstTargets(const Relation& requirements, const Relation& targetBefore) {
    auto order = restrictCandidateOrder(targetBefore, requirements, false);
    if (!order)
        return order;
    auto dominated = compose(requirements, *order.relation);
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
    auto sources = restrictCandidateOrder(sourceThrough, latest, true);
    if (!sources)
        return sources;
    auto targets = restrictCandidateOrder(targetBefore, latest, false);
    if (!targets)
        return targets;
    auto prefix = compose(*sources.relation, latest);
    if (!prefix) return prefix;
    auto dominated = compose(*prefix.relation, *targets.relation);
    if (!dominated) return dominated;
    return subtract(latest, *dominated.relation);
}

RelationResult RelationQueries::restrictCandidateOrder(const Relation& order, const Relation& candidates, bool sources)
{
    auto endpoints = sources ? candidates.getDomainSet() : candidates.getRangeSet();
    return restrictEndpoints(order, endpoints, endpoints);
}

RelationResult RelationQueries::restrictEndpoints(
    const Relation& order, const mlir::presburger::PresburgerSet& sources,
    const mlir::presburger::PresburgerSet& targets)
{
    if (order.getNumDomainVars() != sources.getNumRangeVars() || order.getNumRangeVars() != targets.getNumRangeVars())
        return failure(QueryStatus::Unsupported, "candidate endpoint space mismatch");
    if (!sources.getNumDisjuncts() || !targets.getNumDisjuncts()) {
        auto empty = Relation::getEmpty(order.getSpace());
        if (!charge(empty))
            return failure(QueryStatus::BudgetExhausted, "empty endpoint budget");
        return {QueryStatus::Proved, std::move(empty), {}};
    }
    if (!charge(order) || !charge(sources) || !charge(targets))
        return failure(QueryStatus::BudgetExhausted, "candidate endpoint budget");
    auto constants = [](const Relation& set) {
        std::vector<Constants> result;
        for (const auto& piece : set.getAllDisjuncts())
            result.push_back(fixedCoordinates(piece, 0, piece.getNumRangeVars()));
        return result;
    };
    auto sourceConstants = constants(sources), targetConstants = constants(targets);
    auto result = Relation::getEmpty(order.getSpace());
    for (const auto& piece : order.getAllDisjuncts()) {
        auto left = fixedCoordinates(piece, 0, piece.getNumDomainVars());
        auto right = fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
        bool possibleLeft = false, possibleRight = false;
        for (const auto& endpoint : sourceConstants) {
            if (!remaining)
                return failure(QueryStatus::BudgetExhausted, "candidate endpoint enumeration budget");
            --remaining;
            ++used;
            if (!incompatible(left, endpoint)) {
                possibleLeft = true;
                break;
            }
        }
        if (!possibleLeft)
            continue;
        for (const auto& endpoint : targetConstants) {
            if (!remaining)
                return failure(QueryStatus::BudgetExhausted, "candidate endpoint enumeration budget");
            --remaining;
            ++used;
            if (!incompatible(right, endpoint)) {
                possibleRight = true;
                break;
            }
        }
        if (possibleRight)
            result.unionInPlace(Relation(piece));
    }
    return {QueryStatus::Proved, std::move(result), {}};
}

RelationResult CompletionQueries::selectOrder(
    bool global, const mlir::presburger::PresburgerSet& source, const mlir::presburger::PresburgerSet& target,
    RelationQueries& queries)
{
    auto& index = *(global ? indexedGlobal : indexedIssues);
    const auto& original = global ? *globalIssueOrder : *issueOrder;
    if (!index) {
        auto value = queries.normalize(original);
        if (!value)
            return value;
        OrderBlocks blocks;
        for (const auto& piece : value.relation->getAllDisjuncts()) {
            auto a = fixedCoordinates(piece, 0, piece.getNumDomainVars());
            auto b = fixedCoordinates(piece, piece.getNumDomainVars(), piece.getNumRangeVars());
            auto key = std::make_pair(a.empty() ? std::nullopt : a.front(), b.empty() ? std::nullopt : b.front());
            auto [it, inserted] = blocks.try_emplace(key, Relation::getEmpty(original.getSpace()));
            (void)inserted;
            it->second.unionInPlace(Relation(piece));
        }
        index = std::move(blocks);
    }
    auto coordinates = [](const Relation& set) {
        std::set<Coordinate> result;
        for (const auto& piece : set.getAllDisjuncts()) {
            auto constants = fixedCoordinates(piece, 0, piece.getNumRangeVars());
            result.insert(constants.empty() ? std::nullopt : constants.front());
        }
        return result;
    };
    auto from = coordinates(source), to = coordinates(target);
    auto possible = [](Coordinate key, const auto& selected) {
        return !selected.empty() && (!key || selected.count(std::nullopt) || selected.count(key));
    };
    auto selected = Relation::getEmpty(original.getSpace());
    for (const auto& [key, block] : *index) {
        if (!queries.spend(1))
            return failure(QueryStatus::BudgetExhausted, "indexed issue order budget");
        if (possible(key.first, from) && possible(key.second, to))
            selected.unionInPlace(block);
    }
    return queries.restrictEndpoints(selected, source, target);
}

QueryStatus CompletionQueries::proveSparse(const Relation& requirement, RelationQueries& queries, unsigned rounds)
{
    using Scope = std::optional<llvm::DynamicAPInt>;
    std::map<Scope, Relation> demands;
    for (const auto& piece : requirement.getAllDisjuncts()) {
        auto coordinates = fixedCoordinates(piece, 0, piece.getNumDomainVars());
        Scope key = coordinates.empty() ? std::nullopt : coordinates.front();
        auto [it, inserted] = demands.try_emplace(key, Relation::getEmpty(requirement.getSpace()));
        (void)inserted;
        it->second.unionInPlace(Relation(piece));
    }
    expanded = Relation::getEmpty(requirement.getSpace());
    QueryStatus overall = QueryStatus::Proved;
    for (const auto& [scope, required] : demands) {
        auto found = sources.find(scope);
        if (found == sources.end()) {
            // Cache the FULL issue-order source scope, not just this demand's
            // possibly narrower guard/invocation domain. O itself retains all
            // qualified execution conditions; this filter never invents edges.
            mlir::presburger::IntegerRelation filter(issueOrder->getSpace().getDomainSpace());
            if (scope)
                filter.addBound(mlir::presburger::BoundType::EQ, 0, *scope);
            auto before = selectOrder(
                false, mlir::presburger::PresburgerSet(Relation(filter)), primitive.getDomainSet(), queries);
            if (!before)
                return before.status;
            auto initial = queries.compose(*before.relation, primitive);
            if (!initial)
                return initial.status;
            found = sources.emplace(scope, SourceState{*initial.relation, *initial.relation, false}).first;
        }
        auto& state = found->second;
        Relation available = Relation::getEmpty(requirement.getSpace());
        auto coverage = [&]() -> QueryStatus {
            auto after = selectOrder(false, state.reached.getRangeSet(), required.getRangeSet(), queries);
            if (!after)
                return after.status;
            auto value = queries.compose(state.reached, *after.relation);
            if (!value)
                return value.status;
            available = std::move(*value.relation);
            return queries.contains(available, required);
        };
        auto covered = coverage();
        if (covered == QueryStatus::NotEstablished && !state.saturated && globalIssueOrder) {
            // Rejection-only upper bounds: every completion path must enter
            // and leave through a real handoff. Global issue order is never
            // inserted into reached completion or used to claim saturation.
            auto before = selectOrder(true, required.getDomainSet(), primitive.getDomainSet(), queries);
            auto after = selectOrder(false, primitive.getRangeSet(), required.getRangeSet(), queries);
            if (!before || !after)
                return !before ? before.status : after.status;
            auto prefix = queries.compose(*before.relation, primitive);
            if (!prefix)
                return prefix.status;
            auto possible = queries.compose(*prefix.relation, *after.relation);
            if (!possible)
                return possible.status;
            auto necessary = queries.contains(*possible.relation, required);
            if (necessary == QueryStatus::Proved) {
                auto firstBefore = selectOrder(false, required.getDomainSet(), primitive.getDomainSet(), queries);
                auto firstAfter = selectOrder(true, primitive.getRangeSet(), required.getRangeSet(), queries);
                if (!firstBefore || !firstAfter)
                    return !firstBefore ? firstBefore.status : firstAfter.status;
                auto first = queries.compose(*firstBefore.relation, primitive);
                if (!first)
                    return first.status;
                auto possibleFirst = queries.compose(*first.relation, *firstAfter.relation);
                if (!possibleFirst)
                    return possibleFirst.status;
                necessary = queries.contains(*possibleFirst.relation, required);
            }
            if (necessary == QueryStatus::Unsupported || necessary == QueryStatus::BudgetExhausted)
                return necessary;
            if (necessary == QueryStatus::NotEstablished) {
                expanded.unionInPlace(available);
                overall = QueryStatus::NotEstablished;
                continue;
            }
        }
        // Previous squaring allowed 2^rounds handoffs. Delta propagation uses
        // that many extensions per query without constructing unrelated roots.
        // Resumed queries retain earlier reachability and pending work.
        unsigned steps = (1u << std::min(rounds, 8u)) - 1;
        for (unsigned step = 0; covered == QueryStatus::NotEstablished && !state.saturated && step < steps; ++step) {
            if (!transitions) {
                auto between = selectOrder(false, primitive.getRangeSet(), primitive.getDomainSet(), queries);
                if (!between)
                    return between.status;
                auto value = queries.compose(*between.relation, primitive);
                if (!value)
                    return value.status;
                transitions = std::move(*value.relation);
            }
            auto next = queries.compose(state.pending, *transitions);
            if (!next)
                return next.status;
            auto fresh = queries.subtract(*next.relation, state.reached);
            if (!fresh)
                return fresh.status;
            state.pending = std::move(*fresh.relation);
            if (state.pending.isIntegerEmpty()) {
                state.saturated = true;
                break;
            }
            state.reached.unionInPlace(state.pending);
            covered = coverage();
        }
        if (covered == QueryStatus::Unsupported || covered == QueryStatus::BudgetExhausted)
            return covered;
        expanded.unionInPlace(available);
        if (covered != QueryStatus::Proved)
            overall = covered;
    }
    // A saturated source query says nothing about unqueried source scopes.
    // The public whole-plan fixed-point bit is therefore never set here.
    return overall;
}

QueryStatus CompletionQueries::prove(const Relation& requirement, RelationQueries& queries, unsigned rounds) {
    if (issueOrder)
        return proveSparse(requirement, queries, rounds);
    auto covered = queries.contains(known, requirement);
    if (covered != QueryStatus::NotEstablished || fixed)
        return covered;
    for (unsigned round = 0; round < rounds; ++round) {
        auto paths = queries.compose(known, known);
        if (!paths)
            return paths.status;
        auto added = queries.subtract(*paths.relation, known);
        if (!added)
            return added.status;
        if (added.relation->isIntegerEmpty()) {
            fixed = true;
            return QueryStatus::NotEstablished;
        }
        known.unionInPlace(*added.relation);
        covered = queries.contains(known, requirement);
        if (covered != QueryStatus::NotEstablished)
            return covered;
    }
    return QueryStatus::NotEstablished;
}
