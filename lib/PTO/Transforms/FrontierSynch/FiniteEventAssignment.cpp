// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteEventAssignment.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
FiniteEventAssignment assignOrderedFiniteEvents(llvm::ArrayRef<OrderedHandoffSummary> summaries,
    llvm::ArrayRef<unsigned> eligibleIds)
{
    FiniteEventAssignment result;
    llvm::DenseSet<unsigned> distinct;
    for (auto id : eligibleIds) {
        if (!distinct.insert(id).second) { result.reason = "duplicate eligible physical ID"; return result; }
    }
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& row = summaries[i];
        if (!row.sourceRank || !row.targetRank || row.sourceTargetPrefix >= row.targetRank ||
            (i && (summaries[i - 1].sourceRank >= row.sourceRank ||
                   summaries[i - 1].targetRank >= row.targetRank ||
                   summaries[i - 1].sourceTargetPrefix > row.sourceTargetPrefix))) {
            result.reason = "ordered handoff summary premises unmet"; return result;
        }
    }
    std::size_t next = 0;
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        next = std::max(next, i + 1);
        while (next < summaries.size() &&
               summaries[next].sourceTargetPrefix < summaries[i].targetRank) { ++next; }
        result.thresholds.push_back(next);
        result.required = std::max(result.required, next - i);
    }
    if (result.required > eligibleIds.size()) {
        result.reason = "selected finite plan requires " + std::to_string(result.required) +
                        " IDs; available " + std::to_string(eligibleIds.size());
        return result;
    }
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        result.ids.push_back(eligibleIds[i % result.required]);
    }
    result.certified = true;
    return result;
}
FiniteEventAssignment assignFiniteEvents(llvm::ArrayRef<FiniteHandoff> handoffs,
    llvm::ArrayRef<unsigned> eligibleIds, llvm::ArrayRef<llvm::BitVector> reach)
{
    FiniteEventAssignment result;
    for (std::size_t i = 0; i < reach.size(); ++i) {
        if (reach[i].size() != reach.size() || reach[i].test(i)) {
            result.reason = "finite command relation is not strict and acyclic"; return result;
        }
    }
    // Validate every row before following edges into later rows.
    for (std::size_t i = 0; i < reach.size(); ++i) {
        for (auto j = reach[i].find_first(); j >= 0; j = reach[i].find_next(j)) {
            if (reach[j].test(i) || !llvm::all_of(reach[j].set_bits(), [&](unsigned k) { return reach[i].test(k); })) {
                result.reason = "finite command relation is not a transitive causal order"; return result;
            }
        }
    }
    llvm::DenseSet<unsigned> distinct;
    for (auto id : eligibleIds) {
        if (!distinct.insert(id).second) { result.reason = "duplicate eligible physical ID"; return result; }
    }
    for (const auto& handoff : handoffs) {
        if (handoff.publication >= reach.size() || handoff.acquisition >= reach.size() ||
            !reach[handoff.publication].test(handoff.acquisition)) {
            result.reason = "finite handoff lacks a causally matched acquisition"; return result;
        }
    }
    const auto count = handoffs.size(), absent = count;
    llvm::SmallVector<std::size_t> left(count, absent), right(count, absent);
    // Maximum bipartite matching gives a minimum chain partition of the causal
    // rearm partial order. A chain uses one ID; no payload order is added.
    std::function<bool(std::size_t, llvm::BitVector&)> augment;
    augment = [&](std::size_t source, llvm::BitVector& seen) {
        for (std::size_t target = 0; target < count; ++target) {
            if (seen.test(target) ||
                !reach[handoffs[source].acquisition].test(handoffs[target].publication)) { continue; }
            seen.set(target);
            if (right[target] == absent || augment(right[target], seen)) {
                left[source] = target; right[target] = source; return true;
            }
        }
        return false;
    };
    std::size_t matches = 0;
    for (std::size_t i = 0; i < count; ++i) {
        llvm::BitVector seen(count);
        if (augment(i, seen)) { ++matches; }
    }
    result.required = count - matches;
    if (result.required > eligibleIds.size()) {
        result.reason = "selected finite plan requires " + std::to_string(result.required) +
            " IDs; eligible pool contains " + std::to_string(eligibleIds.size()); return result;
    }
    result.ids.resize(count);
    std::size_t color = 0;
    for (std::size_t first = 0; first < count; ++first) {
        if (right[first] != absent) { continue; }
        for (auto node = first; node != absent; node = left[node]) { result.ids[node] = eligibleIds[color]; }
        ++color;
    }
    result.certified = true;
    return result;
}
} // namespace mlir::pto::frontiersynch
