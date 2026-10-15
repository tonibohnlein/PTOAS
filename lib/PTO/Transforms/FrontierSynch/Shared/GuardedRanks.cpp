// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
#include "PTO/Transforms/FrontierSynch/Shared/GuardedRanks.h"
#include <map>
namespace mlir::pto::frontiersynch {
GuardedRankFrontier initGuardedRankFrontier(RegionExpressions& e, uint32_t columns)
{
    GuardedRankFrontier result;
    result.counters.assign(columns, e.constant(0));
    result.starts.assign(columns, result.counters); result.completions = result.starts;
    return result;
}
namespace {
GuardedRankStep advanceRank(
    RegionExpressions& e, GuardedRankFrontier& frontier, uint32_t column, RegionExpressions::Id present,
    llvm::ArrayRef<GuardedRankIncoming> candidates, llvm::ArrayRef<GuardedRankIncoming> native,
    std::optional<RegionExpressions::Id> referenceRank, bool knownEndpointOrder = false)
{
    GuardedRankStep out;
    const auto k = frontier.counters.size();
    auto integer = [&](RegionExpressions::Id value) { return value < e.size() && !e.isBoolean(value); };
    auto row = [&](llvm::ArrayRef<RegionExpressions::Id> values) {
        return values.size() == k && llvm::all_of(values, integer);
    };
    if (column >= k || frontier.starts.size() != k || frontier.completions.size() != k || present >= e.size() ||
        !e.isBoolean(present) || !integer(frontier.counters[column]) || !row(frontier.starts[column]) ||
        !row(frontier.completions[column])) {
        out.error = "invalid guarded frontier registers";
        return out;
    }
    auto valid = [&](llvm::ArrayRef<GuardedRankIncoming> edges) {
        return llvm::all_of(edges, [&](const auto& edge) {
            return edge.column < k && integer(edge.rank) && edge.guard < e.size() && e.isBoolean(edge.guard) &&
                   edge.completion.size() == k;
        });
    };
    if (!valid(candidates) || !valid(native)) {
        out.error = "invalid guarded frontier source";
        return out;
    }
    if (!referenceRank && e.constantValue(frontier.counters[column]) == UINT64_MAX && e.constantValue(present) == 1) {
        out.error = "guarded frontier rank overflow";
        return out;
    }
    auto join = [&](std::vector<RegionExpressions::Id>& to, llvm::ArrayRef<RegionExpressions::Id> from,
                    RegionExpressions::Id guard) {
        if (e.constantValue(guard) == 0) {
            return true;
        }
        if (!row(from)) {
            return false;
        }
        for (std::size_t p = 0; p < k; ++p) {
            to[p] = e.select(e.land(guard, e.lt(to[p], from[p])), from[p], to[p]);
        }
        return true;
    };
    out.rank = referenceRank ? *referenceRank :
                               e.add(frontier.counters[column], e.select(present, e.constant(1), e.constant(0)));
    out.start = frontier.starts[column];
    for (const auto& edge : native) {
        if (!join(out.start, edge.completion, e.land(present, edge.guard))) {
            out.error = "invalid native frontier row";
            return out;
        }
    }
    std::vector<bool> sourcesPerPipe(k);
    bool groupedIncoming = false;
    if (knownEndpointOrder) {
        for (const auto& edge : candidates) {
            groupedIncoming |= sourcesPerPipe[edge.column];
            sourcesPerPipe[edge.column] = true;
        }
    }
    if (groupedIncoming) {
        // These finite inputs provide original source maps in decreasing
        // reference order. Native completion chains imply every earlier
        // active incoming source through the latest one on its pipe. Select
        // that endpoint once, rather than repeatedly joining its predecessors.
        struct Source {
            RegionExpressions::Id active, rank;
            std::vector<RegionExpressions::Id> completion;
        };
        const auto no = e.boolean(false), zero = e.constant(0);
        std::vector<Source> selected(k, Source{no, zero, std::vector<RegionExpressions::Id>(k, zero)});
        std::vector<RegionExpressions::Id> latest;
        latest.reserve(candidates.size());
        for (const auto& edge : candidates) {
            auto& source = selected[edge.column];
            auto active = e.land(present, edge.guard);
            auto choose = e.land(active, e.lnot(source.active));
            latest.push_back(choose);
            if (e.constantValue(choose) == 0) {
                continue;
            }
            if (!row(edge.completion)) {
                out.error = "invalid candidate frontier row";
                return out;
            }
            source.rank = e.select(choose, edge.rank, source.rank);
            for (std::size_t p = 0; p < k; ++p) {
                source.completion[p] = e.select(choose, edge.completion[p], source.completion[p]);
            }
            source.active = e.lor(source.active, active);
        }
        // Restricted incoming-cut queries use H0, before this consumer's
        // edges. An alternative path ends through the preceding executed
        // native start (including fixed prerequisites) or another selected
        // incoming completion. No test can use the edge being reduced.
        std::vector<RegionExpressions::Id> keep(k, no);
        for (std::size_t p = 0; p < k; ++p) {
            const auto& source = selected[p];
            if (e.constantValue(source.active) == 0) {
                continue;
            }
            auto required = e.land(source.active, e.lt(out.start[p], source.rank));
            for (std::size_t q = 0; q < k; ++q) {
                if (p == q || e.constantValue(selected[q].active) == 0) {
                    continue;
                }
                auto independent =
                    e.select(selected[q].active, e.lt(selected[q].completion[p], source.rank), e.boolean(true));
                required = e.land(required, independent);
            }
            keep[p] = required;
        }
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            out.retained.push_back(e.land(latest[i], keep[candidates[i].column]));
        }
        // Reachability depends on original active generators, not their
        // cover-selection guards. Keep reduction out of subsequent H0 rows.
        for (const auto& source : selected) {
            if (!join(out.start, source.completion, source.active)) {
                out.error = "invalid selected frontier row";
                return out;
            }
        }
    } else {
        for (const auto& edge : candidates) {
            const auto active = e.land(present, edge.guard);
            auto keep = e.land(active, e.lt(out.start[edge.column], edge.rank));
            out.retained.push_back(keep);
            // If this source already reaches the accumulated prefix, its entire
            // completion row is contained in that prefix. Joining an active
            // redundant source therefore changes nothing. Keep cover selection
            // out of the reachability circuit; otherwise each later row embeds
            // the earlier reduction tests in every component. Statically rejected
            // candidates still require no join construction at all.
            if (e.constantValue(keep) == 0) {
                continue;
            }
            if (!join(out.start, edge.completion, active)) {
                out.error = "invalid candidate frontier row";
                return out;
            }
        }
    }
    out.completion = out.start;
    join(out.completion, frontier.completions[column], present);
    out.completion[column] = out.rank;
    if (!e.constructionError().empty()) {
        out.error = e.constructionError();
        return out;
    }
    auto nextStart = frontier.starts[column], nextCompletion = frontier.completions[column];
    for (std::size_t p = 0; p < k; ++p) {
        nextStart[p] = e.select(present, out.start[p], nextStart[p]);
        nextCompletion[p] = e.select(present, out.completion[p], nextCompletion[p]);
    }
    if (!e.constructionError().empty()) {
        out.error = e.constructionError();
        return out;
    }
    frontier.starts[column] = std::move(nextStart);
    frontier.completions[column] = std::move(nextCompletion);
    frontier.counters[column] = out.rank;
    return out;
}
} // namespace
GuardedRankStep advanceGuardedRank(
    RegionExpressions& e, GuardedRankFrontier& frontier, uint32_t column, RegionExpressions::Id present,
    llvm::ArrayRef<GuardedRankIncoming> candidates, llvm::ArrayRef<GuardedRankIncoming> native)
{
    return advanceRank(e, frontier, column, present, candidates, native, std::nullopt);
}
GuardedRanks reduceGuardedRanks(
    RegionExpressions& e, llvm::ArrayRef<GuardedRankPayload> payloads, llvm::ArrayRef<GuardedRankEdge> generators,
    llvm::ArrayRef<GuardedRankEdge> native, GuardedRankLabels labels)
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
    auto frontier = initGuardedRankFrontier(e, pipes.size());
    std::vector<uint64_t> referenceRanks(pipes.size());
    for (uint32_t i = 0; i < payloads.size(); ++i) {
        std::vector<GuardedRankIncoming> candidates, nativeSources;
        for (auto [source, guard] : incoming[i]) {
            candidates.push_back({out.columns[source], out.ranks[source], guard, out.completions[source]});
        }
        for (auto [source, guard] : fixed[i]) {
            nativeSources.push_back({out.columns[source], out.ranks[source], guard, out.completions[source]});
        }
        // Any strictly increasing labels on each present pipe chain induce the
        // same prefix comparisons. Fixed potential-site labels remove the
        // branch-dependent counting circuit; presence still gates every edge,
        // query and frontier update. The finite input bounds labels by UINT32_MAX.
        std::optional<RegionExpressions::Id> referenceRank;
        if (labels == GuardedRankLabels::Reference) {
            referenceRank = e.constant(++referenceRanks[out.columns[i]]);
        }
        auto step = advanceRank(
            e, frontier, out.columns[i], payloads[i].present, candidates, nativeSources, referenceRank, true);
        if (!step.error.empty()) { out.error = step.error; return out; }
        std::size_t at = 0;
        for (auto [source, guard] : incoming[i]) {
            auto keep = step.retained[at++];
            if (e.constantValue(keep) != 0) { out.retained.push_back({source, i, keep}); }
        }
        out.ranks.push_back(step.rank);
        out.starts.push_back(std::move(step.start)); out.completions.push_back(std::move(step.completion));
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
