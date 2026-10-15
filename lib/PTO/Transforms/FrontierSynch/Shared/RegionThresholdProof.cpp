// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exhaustive threshold cells retain symbolic integer values and opaque guards.
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
void count(uint64_t& value) { if (value != UINT64_MAX) { ++value; } }
}
bool RegionExpressions::inactiveByThresholds(Id expression, ThresholdProofCost* cost)
{
    if (!constructionMessage.empty() || !isBoolean(expression)) { return false; }
    if (auto value = constantValue(expression)) { return *value == 0; }
    SmallVector<Id> pending{expression};
    llvm::DenseSet<Id> seen;
    std::map<Id, SmallVector<Id>> families;
    while (!pending.empty()) {
        const auto id = pending.pop_back_val();
        if (!seen.insert(id).second) { continue; }
        if (cost) { count(cost->dagVisits); }
        const auto& node = nodes[id];
        appendOperands(node, pending);
        if (node.kind != Kind::Eq && node.kind != Kind::Lt && node.kind != Kind::Le &&
            node.kind != Kind::SLt && node.kind != Kind::SLe) { continue; }
        for (auto pair : {std::make_pair(node.a, node.b), std::make_pair(node.b, node.a)}) {
            if (nodes[pair.first].kind == Kind::Input && !isBoolean(pair.first) && constantValue(pair.second)) {
                families[pair.first].push_back(id);
            }
        }
    }
    for (const auto& [input, atoms] : families) {
        if (cost) { count(cost->families); }
        if (inactiveOverThresholds(expression, input, atoms, cost)) { return true; }
    }
    return false;
}
bool RegionExpressions::inactiveOverThresholds(
    Id expression, Id input, ArrayRef<Id> atoms, ThresholdProofCost* cost)
{
    std::set<uint64_t> starts{0};
    for (auto atom : atoms) {
        const auto node = nodes[atom];
        const auto bound = *constantValue(node.a == input ? node.b : node.a);
        starts.insert(bound);
        if (bound != UINT64_MAX) { starts.insert(bound + 1); }
        if (node.kind == Kind::SLt || node.kind == Kind::SLe) { starts.insert(uint64_t(1) << 63); }
    }
    for (auto position = starts.begin(); position != starts.end();) {
        if (cost) { count(cost->cells); }
        const auto next = std::next(position);
        const auto lower = *position, upper = next == starts.end() ? UINT64_MAX : *next - 1;
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
        const auto specialized = specializeThresholdCell(expression, input, lower, upper, refinements, cost);
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
    SmallVectorImpl<uint64_t>& refinements, ThresholdProofCost* cost)
{
    if (!valid(expression)) { return invalid; }
    SmallVector<Id> pending{expression}, order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        const auto id = pending.pop_back_val();
        if (!seen.insert(id).second) { continue; }
        if (cost) { count(cost->substitutionVisits); }
        order.push_back(id);
        appendOperands(nodes[id], pending);
    }
    llvm::sort(order);
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
    for (auto id : order) {
        if (cost) { count(cost->substitutionVisits); }
        const auto node = nodes[id];
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
