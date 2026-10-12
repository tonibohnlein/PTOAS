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
    for (auto position = starts.begin(); position != starts.end(); ++position) {
        if (cost) { count(cost->cells); }
        const auto next = std::next(position);
        const auto lower = *position, upper = next == starts.end() ? UINT64_MAX : *next - 1;
        Transaction transaction(*this);
        SmallVector<std::pair<Id, Id>> facts;
        for (auto atom : atoms) {
            const auto node = nodes[atom];
            const auto bound = *constantValue(node.a == input ? node.b : node.a);
            const auto truth = node.a == input ? fold(node.kind, lower, bound) : fold(node.kind, bound, lower);
            // Every direct comparison is constant throughout this exact cell.
            // The endpoint check also fails closed if the partition is extended.
            const auto end = node.a == input ? fold(node.kind, upper, bound) : fold(node.kind, bound, upper);
            if (!truth || truth != end) { return false; }
            facts.push_back({atom, boolean(*truth != 0)});
        }
        if (lower == upper) { facts.push_back({input, constant(lower)}); }
        Substitution substitution(facts);
        const auto specialized = substitute(expression, substitution);
        if (specialized == invalid) { return false; }
        if (constantValue(specialized) == 0) { continue; }
        const auto domain = land(le(constant(lower), input), le(input, constant(upper)));
        if (constantUnder(domain, specialized) != 0) { return false; }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch
