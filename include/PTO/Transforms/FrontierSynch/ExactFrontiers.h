// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_EXACTFRONTIERS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_EXACTFRONTIERS_H

#include "PTO/Transforms/FrontierSynch/ExactFrontierCore.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include <map>
#include <set>

namespace mlir::pto::frontiersynch {

// No new SSA values are created during Phase A. Bounds refer to retained original
// values; the min/max chains and conditional last coordinate are an executable
// recipe for a later emitter, not speculative arith operations in the input IR.
struct OriginalIntervalParticipation {
    OriginalProgramVersion version;
    std::size_t owner = NoControlId, operation = NoControlId;
    Value induction;
    std::vector<Value> references;
    exact_frontier::IntervalRecipe recipe;
    std::vector<std::pair<Value, bool>> invariantConditions;
    using Key = std::tuple<OriginalProgramVersion, std::size_t, std::size_t, unsigned, bool,
                           std::vector<std::uintptr_t>, std::vector<exact_frontier::Bound>,
                           std::vector<exact_frontier::Bound>, std::vector<std::pair<std::uintptr_t, bool>>>;
    Key key() const
    {
        std::vector<std::uintptr_t> refs;
        std::vector<std::pair<std::uintptr_t, bool>> conditions;
        for (auto value : references) {
            refs.push_back(reinterpret_cast<std::uintptr_t>(value.getAsOpaquePointer()));
        }
        for (const auto& condition : invariantConditions) {
            conditions.emplace_back(reinterpret_cast<std::uintptr_t>(condition.first.getAsOpaquePointer()),
                                    condition.second);
        }
        return {version, owner, operation, recipe.integer.width, recipe.integer.isUnsigned,
                std::move(refs), recipe.lower, recipe.upper, std::move(conditions)};
    }
};
// Evaluates the recipe under one already-admitted original valuation. This is
// a reference interpretation, not a compiler-side enumeration. The last atom
// has a conditional subtraction even when used inside an eager Boolean DAG.
template <class Lookup>
std::optional<bool> evaluateIntervalParticipation(
    const OriginalIntervalParticipation& interval, ObservationAtom::Kind kind, Lookup lookup)
{
    bool participates = true;
    for (const auto& condition : interval.invariantConditions) {
        const auto value = lookup(condition.first);
        if (!value || *value > 1) {
            return {};
        }
        participates &= bool(*value) == condition.second;
    }
    const auto domain = exact_frontier::intersection(interval.recipe, [&](std::size_t id) {
        return id < interval.references.size() ? lookup(interval.references[id]) : std::optional<uint64_t>{};
    });
    if (!domain.valid) {
        return {};
    }
    if (!participates || !domain.nonempty) {
        return false;
    }
    if (kind == ObservationAtom::IntervalNonEmpty) {
        return true;
    }
    if (kind != ObservationAtom::IntervalFirst && kind != ObservationAtom::IntervalLast) {
        return {};
    }
    const auto selected = exact_frontier::coordinate(interval.recipe.integer, domain,
                                                     kind == ObservationAtom::IntervalLast);
    const auto iv = lookup(interval.induction);
    return iv && selected ? std::optional<bool>(*iv == *selected) : std::nullopt;
}
struct OriginalExactFrontiers {
    enum class Status { Unknown, NoHit, Exact } status = Status::Unknown;
    OriginalInterval interval;
    bool readOnly = false;
    // Structural exactness is independent of availability of either direction.
    // A public boundary is Exact only after its own endpoint qualification.
    std::size_t nonempty = 0, noHit = NoControlId, first = 0, last = 0;
    std::string reason;
};
struct OriginalExactFrontierStats {
    std::size_t summaryQueries = 0, allQueries = 0, compositionParts = 0;
    std::size_t intervalRecipes = 0, boundReferences = 0;
};

inline OriginalValueQualification qualifyIntervalParticipation(
    const OriginalStructure& original, const OriginalValueQueries& values,
    const OriginalIntervalParticipation& interval, ObservationAtom::Kind kind, OriginalValueCut cut)
{
    using Result = OriginalValueQualification;
    auto fail = [](std::string reason) {
        Result r;
        r.obstructions.push_back(std::move(reason));
        return r;
    };
    if (!values.current() || interval.version != original.version ||
        interval.owner >= original.originalSites.size() || !values.legal(cut)) {
        return fail("interval predicate has a stale version or no executable original cut");
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(original.originalSites[interval.owner]);
    if (!loop || loop.getInductionVar() != interval.induction || !interval.recipe.integer.valid()) {
        return fail("interval predicate has no qualified original induction value");
    }
    auto result = values.counted(loop);
    std::vector<Value> lower, upper;
    for (auto side : {false, true}) {
        const auto& bounds = side ? interval.recipe.upper : interval.recipe.lower;
        auto& operands = side ? upper : lower;
        for (auto bound : bounds) {
            if (bound.reference >= interval.references.size()) {
                return fail("interval bound has no original value reference");
            }
            operands.push_back(interval.references[bound.reference]);
        }
    }
    result = OriginalValueQueries::combine(
        std::move(result), values.extremum(lower, cut, false, interval.recipe.integer.isUnsigned));
    result = OriginalValueQueries::combine(
        std::move(result), values.extremum(upper, cut, true, interval.recipe.integer.isUnsigned));
    for (const auto& condition : interval.invariantConditions) {
        result = OriginalValueQueries::combine(std::move(result), values.qualify(condition.first, cut));
    }
    if (kind != ObservationAtom::IntervalNonEmpty) {
        result = OriginalValueQueries::combine(std::move(result), values.qualify(interval.induction, cut));
        if (!loop->isProperAncestor(original.originalSites[cut.site])) {
            Result missing;
            missing.status = Result::Status::NotObservableHere;
            missing.obstructions.emplace_back("interval coordinate is not observable outside its original loop");
            result = OriginalValueQueries::combine(std::move(result), missing);
        }
    }
    // The recipe index is carried by ObservationAtom::parameter, not a new
    // runtime counter. Last uses the true branch of ell<h before computing h-1.
    result.recipe.kind = kind == ObservationAtom::IntervalNonEmpty ? OriginalValueRecipe::IntervalNonEmpty :
                         kind == ObservationAtom::IntervalFirst ? OriginalValueRecipe::IntervalFirst :
                                                                 OriginalValueRecipe::IntervalLast;
    result.recipe.width = interval.recipe.integer.width;
    result.recipe.unsignedComparison = interval.recipe.integer.isUnsigned;
    result.recipe.operands = interval.references;
    if (kind != ObservationAtom::IntervalNonEmpty) {
        result.recipe.operands.push_back(interval.induction);
    }
    return result;
}

// The shared value service remains authoritative for original SSA arithmetic,
// guard identity, dominance and completion prerequisites. This adapter adds only
// the declared interval recipe. And/Or remain eager, not implicit short-circuit
// guards enabling otherwise-unavailable operands.
template <class Predicate, class Interval>
OriginalValueQualification qualifyExactFrontierPredicate(
    const OriginalStructure& original, const OriginalValueQueries& values,
    std::size_t root, Predicate get, Interval lookup, OriginalValueCut cut)
{
    using Result = OriginalValueQualification;
    std::map<std::size_t, Result> memo;
    std::set<std::size_t> active;
    std::function<Result(std::size_t)> visit = [&](std::size_t id) -> Result {
        const auto prior = memo.find(id);
        if (prior != memo.end()) {
            return prior->second;
        }
        Result result;
        if (!active.insert(id).second) {
            result.obstructions.emplace_back("cyclic exact-frontier predicate");
            return result;
        }
        const auto p = get(id);
        if (p.kind == ParticipationExpression::True || p.kind == ParticipationExpression::False) {
            result.status = values.legal(cut) ? Result::Status::Available : Result::Status::NotObservableHere;
            if (!values.legal(cut)) {
                result.obstructions.emplace_back("frontier has no executable original endpoint");
            }
        } else if (p.kind == ParticipationExpression::Atom) {
            if (p.atom.kind == ObservationAtom::IntervalNonEmpty || p.atom.kind == ObservationAtom::IntervalFirst ||
                p.atom.kind == ObservationAtom::IntervalLast) {
                const auto* interval = lookup(p.atom.parameter);
                if (!interval || interval->owner != p.atom.owner || p.atom.value > 1) {
                    result.obstructions.emplace_back("interval observation has no matching immutable recipe");
                } else {
                    result = qualifyIntervalParticipation(original, values, *interval, p.atom.kind, cut);
                    result.recipe.parameter = p.atom.parameter;
                    result.recipe.expected = p.atom.value;
                }
            } else {
                result = values.atom(p.atom, cut);
            }
        } else if (p.kind == ParticipationExpression::Not) {
            result = visit(p.left);
        } else if (p.kind == ParticipationExpression::And || p.kind == ParticipationExpression::Or) {
            result = OriginalValueQueries::combine(visit(p.left), visit(p.right));
        } else {
            result.obstructions.emplace_back("invalid exact-frontier predicate");
        }
        active.erase(id);
        memo.emplace(id, result);
        return result;
    };
    return visit(root);
}
} // namespace mlir::pto::frontiersynch
#endif
