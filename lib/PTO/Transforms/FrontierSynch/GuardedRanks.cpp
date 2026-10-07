// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
#include "PTO/Transforms/FrontierSynch/GuardedRanks.h"
#include <map>
namespace mlir::pto::frontiersynch {
GuardedRanks reduceGuardedRanks(
    RegionExpressions& e, llvm::ArrayRef<GuardedRankPayload> payloads, llvm::ArrayRef<GuardedRankEdge> generators,
    llvm::ArrayRef<GuardedRankEdge> native)
{
    GuardedRanks out;
    out.payloads.assign(payloads.begin(), payloads.end());
    if (payloads.size() > UINT32_MAX) {
        out.error = "guarded rank identity overflow";
        return out;
    }
    std::map<uint32_t, uint32_t> pipes;
    for (auto payload : payloads) {
        if (payload.present >= e.size() || !e.isBoolean(payload.present)) {
            out.error = "invalid guarded presence";
            return out;
        }
        out.columns.push_back(pipes.emplace(payload.pipe, pipes.size()).first->second);
    }
    using Incoming = std::vector<std::map<uint32_t, RegionExpressions::Id, std::greater<uint32_t>>>;
    Incoming incoming(payloads.size()), fixed(payloads.size());
    auto bucket = [&](llvm::ArrayRef<GuardedRankEdge> edges, Incoming& lists) {
        for (auto edge : edges) {
            if (edge.source >= edge.target || edge.target >= payloads.size() || edge.guard >= e.size() ||
                !e.isBoolean(edge.guard)) {
                out.error = "guarded reduction requires valid forward edges";
                return false;
            }
            auto guard = e.land(edge.guard, e.land(payloads[edge.source].present, payloads[edge.target].present));
            auto [found, added] = lists[edge.target].emplace(edge.source, guard);
            if (!added) {
                found->second = e.lor(found->second, guard);
            }
        }
        return true;
    };
    if (!bucket(generators, incoming) || !bucket(native, fixed)) {
        return out;
    }
    using Row = std::vector<RegionExpressions::Id>;
    Row zero(pipes.size(), e.constant(0)), counters = zero;
    std::vector<Row> runningStarts(pipes.size(), zero), runningCompletions(pipes.size(), zero);
    auto join = [&](Row& to, const Row& from, RegionExpressions::Id guard) {
        for (std::size_t p = 0; p < to.size(); ++p) {
            to[p] = e.select(e.land(guard, e.lt(to[p], from[p])), from[p], to[p]);
        }
    };
    for (uint32_t i = 0; i < payloads.size(); ++i) {
        auto column = out.columns[i], present = payloads[i].present;
        auto rank = e.add(counters[column], e.select(present, e.constant(1), e.constant(0)));
        counters[column] = rank;
        out.ranks.push_back(rank);
        Row start = runningStarts[column];
        for (auto [source, guard] : fixed[i]) {
            join(start, out.completions[source], guard);
        }
        for (auto [source, guard] : incoming[i]) {
            auto keep = e.land(guard, e.lt(start[out.columns[source]], out.ranks[source]));
            if (e.constantValue(keep) != 0) {
                out.retained.push_back({source, i, keep});
            }
            join(start, out.completions[source], keep);
        }
        Row completion = start;
        join(completion, runningCompletions[column], present);
        completion[column] = rank;
        for (std::size_t p = 0; p < zero.size(); ++p) {
            runningStarts[column][p] = e.select(present, start[p], runningStarts[column][p]);
            runningCompletions[column][p] = e.select(present, completion[p], runningCompletions[column][p]);
        }
        out.starts.push_back(std::move(start));
        out.completions.push_back(std::move(completion));
    }
    out.error = e.constructionError();
    return out;
}
std::optional<RegionExpressions::Id> GuardedRanks::query(RegionExpressions& e, PeriodicEvent a, PeriodicEvent b) const
{
    auto valid = [](PeriodicEventKind kind) {
        return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
    };
    if (!error.empty() || a.type >= payloads.size() || b.type >= payloads.size() || !valid(a.kind) || !valid(b.kind)) {
        return std::nullopt;
    }
    auto present = e.land(payloads[a.type].present, payloads[b.type].present);
    if (a.kind == PeriodicEventKind::Start && b.kind == PeriodicEventKind::Start &&
        columns[a.type] == columns[b.type] && a.type <= b.type) {
        return present;
    }
    const auto& row = b.kind == PeriodicEventKind::Start ? starts[b.type] : completions[b.type];
    return e.land(present, e.le(ranks[a.type], row[columns[a.type]]));
}
} // namespace mlir::pto::frontiersynch
