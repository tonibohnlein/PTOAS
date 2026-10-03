// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include <algorithm>

namespace mlir::pto::frontiersynch {
namespace {
using EdgeLists = SmallVector<SmallVector<std::size_t>>;
FailureOr<EdgeLists> orderIncoming(std::size_t count, ArrayRef<Demand> generators)
{
    EdgeLists outgoing(count), incoming(count);
    for (auto [id, edge] : llvm::enumerate(generators)) {
        const bool valid = edge.source < edge.consumer && edge.consumer < count;
        if (!valid) {
            return failure();
        }
        outgoing[edge.source].push_back(id);
    }
    // Linear ordering by decreasing source reference position, not pipe rank.
    for (std::size_t end = count; end > 0; --end) {
        for (auto id : outgoing[end - 1]) {
            incoming[generators[id].consumer].push_back(id);
        }
    }
    return incoming;
}

void join(SmallVectorImpl<std::size_t>& target, ArrayRef<std::size_t> source)
{
    for (std::size_t pipe = 0; pipe < target.size(); ++pipe) {
        target[pipe] = std::max(target[pipe], source[pipe]);
    }
}
} // namespace

LogicalResult RankReduction::build(ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<Demand> generators)
{
    columns.clear();
    ranks.clear();
    minimum.clear();
    nonadjacent.clear();
    RankReduction pending;
    DenseMap<PipelineType, std::size_t> pipeIds;
    for (const auto* phase : sequence) {
        if (!phase) {
            return failure();
        }
        auto [found, added] = pipeIds.try_emplace(phase->kPipeValue, pending.columns.size());
        if (added) {
            pending.columns.push_back(phase->kPipeValue);
        }
        pending.ranks.push_back({phase, found->second, 0, {}, {}});
    }
    auto incoming = orderIncoming(sequence.size(), generators);
    if (failed(incoming)) {
        return failure();
    }
    SmallVector<std::optional<std::size_t>> previous(pending.columns.size());
    for (auto [site, summary] : llvm::enumerate(pending.ranks)) {
        summary.S.assign(pending.columns.size(), 0);
        summary.T.assign(pending.columns.size(), 0);
        auto prior = previous[summary.pipe];
        summary.rank = prior ? pending.ranks[*prior].rank + 1 : 1;
        if (prior) {
            summary.S = pending.ranks[*prior].S;
        }
        for (auto id : (*incoming)[site]) {
            const auto& source = pending.ranks[generators[id].source];
            if (summary.S[source.pipe] >= source.rank) {
                continue;
            }
            pending.minimum.push_back(id);
            join(summary.S, source.T);
            const bool local = source.pipe == summary.pipe;
            if (local && source.rank != summary.rank - 1) {
                pending.nonadjacent.push_back(id);
            }
        }
        summary.T = summary.S;
        if (prior) {
            join(summary.T, pending.ranks[*prior].T);
        }
        summary.T[summary.pipe] = summary.rank;
        previous[summary.pipe] = site;
    }
    *this = std::move(pending);
    return success();
}
} // namespace mlir::pto::frontiersynch
