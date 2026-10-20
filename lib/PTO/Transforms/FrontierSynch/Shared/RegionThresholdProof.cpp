// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exhaustive threshold cells retain symbolic integer values and opaque guards.
// Only source topology survives cell rollback. Rebuilt IDs and affine-shift
// facts belong to one transaction; unaffected source subgraphs retain identity.
// Certificate proofs may partition a derived value using unconditional machine
// ranges. This abstracts its value, never assumes an external caller domain.
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
void count(uint64_t& value) { if (value != UINT64_MAX) { ++value; } }
}
struct RegionExpressions::ThresholdPreparation {
    SmallVector<Id> order;
    // Pack operand lists once; a SmallVector in every hash bucket retains
    // inline capacity even for constants and unary nodes across many roots.
    llvm::DenseMap<Id, std::pair<std::size_t, std::size_t>> ranges;
    SmallVector<Id> edges;
    ArrayRef<Id> operands(Id id) const
    {
        const auto range = ranges.find(id)->second;
        return ArrayRef<Id>(edges).slice(range.first, range.second);
    }
    std::map<Id, SmallVector<Id>> families;
};
bool RegionExpressions::inactiveByThresholds(Id expression, ThresholdProofCost* cost)
{
    return inactiveThresholdProof(expression, cost, false);
}
bool RegionExpressions::impliesByIntervals(Id premise, Id consequence, ThresholdProofCost* cost)
{
    if (!isBoolean(premise) || !isBoolean(consequence)) { return false; }
    Transaction transaction(*this);
    return inactiveThresholdProof(land(premise, lnot(consequence)), cost, true);
}
bool RegionExpressions::inactiveThresholdProof(Id expression, ThresholdProofCost* cost, bool derived)
{
    if (!constructionMessage.empty() || !isBoolean(expression)) { return false; }
    if (auto value = constantValue(expression)) { return *value == 0; }
    auto found = thresholdPreparations.find(expression);
    if (found == thresholdPreparations.end()) {
        auto prepared = std::make_shared<ThresholdPreparation>();
        SmallVector<Id> pending{expression};
        llvm::DenseSet<Id> seen;
        while (!pending.empty()) {
            const auto id = pending.pop_back_val();
            if (!seen.insert(id).second) { continue; }
            if (cost) { count(cost->dagVisits); }
            const auto& node = nodes[id];
            SmallVector<Id, 3> operands;
            appendOperands(node, operands);
            prepared->ranges.try_emplace(id, std::make_pair(prepared->edges.size(), operands.size()));
            prepared->edges.append(operands.begin(), operands.end());
            pending.append(operands.begin(), operands.end());
            prepared->order.push_back(id);
            if (node.kind != Kind::Eq && node.kind != Kind::Lt && node.kind != Kind::Le &&
                node.kind != Kind::SLt && node.kind != Kind::SLe) { continue; }
            for (auto pair : {std::make_pair(node.a, node.b), std::make_pair(node.b, node.a)}) {
                if (nodes[pair.first].kind == Kind::Input && !isBoolean(pair.first) && constantValue(pair.second)) {
                    prepared->families[pair.first].push_back(id);
                }
            }
        }
        llvm::sort(prepared->order);
        found = thresholdPreparations.emplace(expression, std::move(prepared)).first;
    } else if (cost) { count(cost->preparationHits); }
    const auto retained = found->second;
    const auto& preparation = *retained;
    std::map<Id, std::pair<uint64_t, uint64_t>> derivedBounds;
    std::map<Id, SmallVector<Id>> derivedFamilies;
    if (derived) {
        for (auto id : preparation.order) {
            if (cost) { count(cost->dagVisits); }
            const auto& node = nodes[id];
            if (node.kind != Kind::Eq && node.kind != Kind::Lt && node.kind != Kind::Le &&
                node.kind != Kind::SLt && node.kind != Kind::SLe) { continue; }
            for (auto pair : {std::make_pair(node.a, node.b), std::make_pair(node.b, node.a)}) {
                if (!isBoolean(pair.first) && !constantValue(pair.first) && constantValue(pair.second)) {
                    derivedFamilies[pair.first].push_back(id);
                }
            }
        }
        // Charge both range transfer and the resulting unsigned hull pass.
        if (cost) {
            for (auto id : preparation.order) { (void)id; count(cost->dagVisits); count(cost->dagVisits); }
        }
        derivedBounds = unsignedProofBounds(preparation.order);
    }
    const auto& families = derived ? derivedFamilies : preparation.families;
    // Prefer the outer derived count to its original bound, without exploring
    // combinations of independent inputs or changing any expression identity.
    SmallVector<Id> candidates;
    for (const auto& family : families) { candidates.push_back(family.first); }
    if (derived) { std::reverse(candidates.begin(), candidates.end()); }
    for (auto input : candidates) {
        const auto& atoms = families.at(input);
        if (cost) { count(cost->families); }
        const auto key = std::make_pair(expression, input);
        auto dependencies = thresholdDependencies.find(key);
        if (dependencies == thresholdDependencies.end()) {
            auto dependent = std::make_shared<llvm::DenseSet<Id>>();
            dependent->insert(input);
            for (auto id : preparation.order) {
                if (cost) { count(cost->dependencyVisits); }
                if (llvm::any_of(preparation.operands(id),
                                [&](Id operand) { return dependent->contains(operand); })) {
                    dependent->insert(id);
                }
            }
            dependencies = thresholdDependencies.emplace(key, std::move(dependent)).first;
        }
        const auto bounds = derived ? derivedBounds.at(input) : std::make_pair(uint64_t(0), UINT64_MAX);
        if (inactiveOverThresholds(expression, input, atoms, preparation, *dependencies->second,
                                   cost, bounds.first, bounds.second)) {
            return true;
        }
    }
    return false;
}
bool RegionExpressions::inactiveOverThresholds(
    Id expression, Id input, ArrayRef<Id> atoms, const ThresholdPreparation& preparation,
    const llvm::DenseSet<Id>& dependent, ThresholdProofCost* cost, uint64_t minimum, uint64_t maximum)
{
    std::set<uint64_t> starts{minimum};
    auto insert = [&](uint64_t point) {
        if (minimum <= point && point <= maximum) { starts.insert(point); }
    };
    for (auto atom : atoms) {
        const auto node = nodes[atom];
        const auto bound = *constantValue(node.a == input ? node.b : node.a);
        insert(bound);
        if (bound != UINT64_MAX) { insert(bound + 1); }
        if (node.kind == Kind::SLt || node.kind == Kind::SLe) { insert(uint64_t(1) << 63); }
    }
    for (auto position = starts.begin(); position != starts.end();) {
        if (cost) { count(cost->cells); }
        const auto next = std::next(position);
        const auto lower = *position, upper = next == starts.end() ? maximum : *next - 1;
        Transaction transaction(*this);
        for (auto atom : atoms) {
            const auto node = nodes[atom];
            const auto bound = *constantValue(node.a == input ? node.b : node.a);
            const auto truth = node.a == input ? fold(node.kind, lower, bound) : fold(node.kind, bound, lower);
            // Every direct comparison is constant throughout this exact cell.
            // The endpoint check also fails closed if the partition is extended.
            const auto end = node.a == input ? fold(node.kind, upper, bound) : fold(node.kind, bound, upper);
            if (!truth || truth != end) { return false; }
        }
        SmallVector<uint64_t> refinements;
        const auto specialized = specializeThresholdCell(
            expression, input, lower, upper, preparation, dependent, refinements, cost);
        if (specialized == invalid) { return false; }
        if (constantValue(specialized) == 0) { ++position; continue; }
        bool refined = false;
        for (auto point : refinements) {
            if (lower < point && point <= upper) { refined |= starts.insert(point).second; }
        }
        // Retest this same cell after an exposed comparison splits it. Set
        // iterators remain valid. Each retry adds a strictly interior boundary.
        if (refined) { continue; }
        const auto domain = land(le(constant(lower), input), le(input, constant(upper)));
        if (cost) { count(cost->proofRequests); }
        if (constantUnder(domain, specialized, cost ? &cost->proofVisits : nullptr) != 0) {
            return false;
        }
        ++position;
    }
    return true;
}
RegionExpressions::Id RegionExpressions::specializeThresholdCell(
    Id expression, Id input, uint64_t lower, uint64_t upper,
    const ThresholdPreparation& preparation, const llvm::DenseSet<Id>& dependent,
    SmallVectorImpl<uint64_t>& refinements, ThresholdProofCost* cost)
{
    if (!valid(expression)) { return invalid; }
    SmallVector<Id> pending{expression};
    llvm::DenseMap<Id, Id> memo;
    struct Shift { bool variable; uint64_t offset; };
    std::map<Id, std::optional<Shift>> shifts;
    std::function<std::optional<Shift>(Id)> shift = [&](Id id) -> std::optional<Shift> {
        if (auto found = shifts.find(id); found != shifts.end()) { return found->second; }
        if (cost) { count(cost->substitutionVisits); }
        std::optional<Shift> result;
        const auto node = nodes[id];
        if (id == input) { result = Shift{true, 0}; }
        else if (auto value = constantValue(id)) { result = Shift{false, *value}; }
        else if (preparation.ranges.contains(id) && !dependent.contains(id)) {
            // A nonconstant source subtree independent of this input cannot
            // have a constant affine offset. Do not rediscover it per cell.
            result = std::nullopt;
        }
        else if (node.kind == Kind::Add || node.kind == Kind::Sub) {
            auto a = shift(node.a), b = shift(node.b);
            if (a && b) {
                if (node.kind == Kind::Add && !(a->variable && b->variable)) {
                    result = Shift{a->variable || b->variable, a->offset + b->offset};
                } else if (node.kind == Kind::Sub && (a->variable || !b->variable)) {
                    result = Shift{a->variable && !b->variable, a->offset - b->offset};
                }
            }
        }
        shifts.emplace(id, result); return result;
    };
    while (!pending.empty()) {
        const auto id = pending.back();
        if (memo.contains(id)) { pending.pop_back(); continue; }
        if (!dependent.contains(id)) {
            memo[id] = id;
            if (cost) { count(cost->reusedNodes); }
            pending.pop_back();
            continue;
        }
        // A partitioned derived value is atomic in this cell. Its operands
        // remain independent unless used elsewhere; no inverse is assumed.
        if (id == input && nodes[id].kind != Kind::Input) {
            memo[id] = lower == upper ? constant(lower) : id;
            pending.pop_back();
            continue;
        }
        const auto node = nodes[id];
        // Select evaluates only the selected arm once its condition is known.
        // Do not traverse the discarded arm or publish any of its local IDs.
        if (node.kind == Kind::Select) {
            if (!memo.contains(node.a)) { pending.push_back(node.a); continue; }
            if (auto condition = constantValue(memo.lookup(node.a))) {
                const auto selected = *condition ? node.b : node.c;
                if (!memo.contains(selected)) { pending.push_back(selected); continue; }
                memo[id] = memo.lookup(selected);
                if (cost) { count(cost->skippedBranches); count(cost->substitutionVisits); }
                pending.pop_back();
                continue;
            }
        }
        bool ready = true;
        for (auto operand : preparation.operands(id)) {
            if (!memo.contains(operand)) { pending.push_back(operand); ready = false; }
        }
        if (!ready) { continue; }
        pending.pop_back();
        if (cost) { count(cost->substitutionVisits); }
        auto result = id;
        if (id == input && lower == upper) { result = constant(lower); }
        else if (node.kind == Kind::Integer) { result = rebuildInteger(node, memo); }
        else if (node.kind == Kind::Select) {
            result = select(memo.lookup(node.a), memo.lookup(node.b), memo.lookup(node.c));
        } else if (node.kind == Kind::Not) { result = lnot(memo.lookup(node.a)); }
        else if (node.a != invalid && node.b != invalid) {
            result = binary(node.kind, memo.lookup(node.a), memo.lookup(node.b));
        }
        if (result == invalid) { return invalid; }
        const auto rebuilt = nodes[result];
        const bool comparison = rebuilt.kind == Kind::Eq || rebuilt.kind == Kind::Lt ||
            rebuilt.kind == Kind::Le || rebuilt.kind == Kind::SLt || rebuilt.kind == Kind::SLe;
        // A select can expose new comparisons of input+offset expressions.
        // On a cell where neither operand wraps (or crosses the signed cut),
        // these are affine functions with slope zero or one. Order is monotone
        // or constant; equality additionally needs to exclude an interior root.
        if (comparison) {
            auto a = shift(rebuilt.a), b = shift(rebuilt.b);
            if (a && b) {
                // Discover thresholds after select specialization. These cuts
                // are exact modular preimages, never sampled representatives.
                for (auto operand : {*a, *b}) {
                    if (!operand.variable) { continue; }
                    refinements.push_back(uint64_t(0) - operand.offset);
                    if (rebuilt.kind == Kind::SLt || rebuilt.kind == Kind::SLe) {
                        refinements.push_back((uint64_t(1) << 63) - operand.offset);
                    }
                }
                if (a->variable != b->variable) {
                    const auto point = a->variable ? b->offset - a->offset : a->offset - b->offset;
                    refinements.push_back(point);
                    if (point != UINT64_MAX) { refinements.push_back(point + 1); }
                }
                const auto firstA = a->offset + (a->variable ? lower : 0);
                const auto lastA = a->offset + (a->variable ? upper : 0);
                const auto firstB = b->offset + (b->variable ? lower : 0);
                const auto lastB = b->offset + (b->variable ? upper : 0);
                const bool signedOrder = rebuilt.kind == Kind::SLt || rebuilt.kind == Kind::SLe;
                const auto continuous = [&](uint64_t first, uint64_t last) {
                    return first <= last && (!signedOrder || (first >> 63) == (last >> 63));
                };
                const auto first = fold(rebuilt.kind, firstA, firstB);
                const auto last = fold(rebuilt.kind, lastA, lastB);
                const auto root = a->variable ? b->offset - a->offset : a->offset - b->offset;
                const bool interiorEquality = rebuilt.kind == Kind::Eq && a->variable != b->variable &&
                    lower < root && root < upper;
                if (continuous(firstA, lastA) && continuous(firstB, lastB) &&
                    first && first == last && !interiorEquality) { result = boolean(*first != 0); }
            }
        }
        memo[id] = result;
    }
    return memo.lookup(expression);
}
} // namespace mlir::pto::frontiersynch
