// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
#include "PTO/Transforms/FrontierSynch/Shared/GuardedRanks.h"
#include "GuardedQueryProjection.h"
#include "FixedRankThresholds.h"
#include <map>
#include <set>
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
    std::optional<RegionExpressions::Id> referenceRank, bool knownEndpointOrder = false,
    detail::FixedRankThresholds* thresholds = nullptr)
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
            const auto previous = to[p];
            to[p] = e.select(e.land(guard, e.lt(previous, from[p])), from[p], previous);
            if (thresholds) { thresholds->record(to[p], guard, from[p], previous, true); }
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
                const auto previous = source.completion[p];
                source.completion[p] = e.select(choose, edge.completion[p], previous);
                if (thresholds) { thresholds->record(source.completion[p], choose, edge.completion[p], previous); }
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
            if (thresholds || e.constantValue(source.active) == 0) {
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
            const auto& edge = candidates[i];
            const auto incumbent = e.land(latest[i], keep[edge.column]);
            if (!thresholds || e.constantValue(latest[i]) == 0) {
                out.retained.push_back(incumbent);
                continue;
            }
            // Under latest[i], the selected source on this pipe is precisely
            // this candidate. Use its fixed label, never a selected rank DAG.
            const auto label = e.constantValue(edge.rank);
            if (!label) { out.error = "reference rank is not a fixed label"; return out; }
            auto required = thresholds->below(out.start[edge.column], *label);
            for (std::size_t q = 0; q < k; ++q) {
                if (q == edge.column || e.constantValue(selected[q].active) == 0) { continue; }
                required = e.land(required, e.select(selected[q].active,
                    thresholds->below(selected[q].completion[edge.column], *label), e.boolean(true)));
            }
            out.retained.push_back(e.land(latest[i], required));
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
            const auto incumbent = e.land(active, e.lt(out.start[edge.column], edge.rank));
            const auto label = e.constantValue(edge.rank);
            if (thresholds && !label) { out.error = "reference rank is not a fixed label"; return out; }
            auto uncovered = thresholds ? thresholds->below(out.start[edge.column], *label) :
                                          e.lt(out.start[edge.column], edge.rank);
            auto keep = e.land(active, uncovered);
            out.retained.push_back(keep);
            // If this source already reaches the accumulated prefix, its entire
            // completion row is contained in that prefix. Joining an active
            // redundant source therefore changes nothing. Keep cover selection
            // out of the reachability circuit; otherwise each later row embeds
            // the earlier reduction tests in every component. Statically rejected
            // candidates still require no join construction at all.
            if (e.constantValue(incumbent) == 0) {
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
        if (thresholds) {
            thresholds->record(nextStart[p], present, out.start[p], frontier.starts[column][p]);
            thresholds->record(nextCompletion[p], present, out.completion[p], frontier.completions[column][p]);
        }
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
    llvm::ArrayRef<GuardedRankEdge> native, GuardedRankLabels labels, bool retainQueryGraph,
    bool projectFixedThresholds)
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
    if (retainQueryGraph && labels == GuardedRankLabels::Reference) {
        auto graph = std::make_shared<detail::GuardedQueryGraph>();
        graph->arena = &e; graph->payloads = out.payloads; graph->columns = out.columns;
        graph->columnCount = pipes.size(); graph->incoming.resize(payloads.size());
        SmallVector<RegionExpressions::Id> guards;
        for (const auto& payload : payloads) { guards.push_back(payload.present); }
        for (uint32_t target = 0; target < payloads.size(); ++target) {
            for (const auto* directory : {&incoming, &fixed}) {
                for (const auto& [source, guard] : (*directory)[target]) {
                    graph->incoming[target].push_back({source, target, guard});
                    guards.push_back(guard);
                }
            }
        }
        graph->totalGuards = detail::RegionConstructionAccess::totalScalarDag(e, guards, graph->qualificationNodes);
        auto& work = detail::RegionConstructionAccess::alternativeConstructionWork(e);
        detail::RegionAlternativeWork::add(work.graphs);
        detail::RegionAlternativeWork::add(work.graphOccurrences, payloads.size());
        detail::RegionAlternativeWork::add(work.graphEdges, guards.size() - payloads.size());
        detail::RegionAlternativeWork::add(work.guardQualification, graph->qualificationNodes);
        out.queryGraph = std::move(graph);
    }
    bool project = false;
    if (projectFixedThresholds && labels == GuardedRankLabels::Reference) {
        if (out.queryGraph) { project = out.queryGraph->totalGuards; }
        else {
            SmallVector<RegionExpressions::Id> sourceGuards;
            for (const auto& payload : payloads) { sourceGuards.push_back(payload.present); }
            for (auto edge : generators) { sourceGuards.push_back(edge.guard); }
            for (auto edge : native) { sourceGuards.push_back(edge.guard); }
            uint64_t qualificationVisits = 0;
            project = detail::RegionConstructionAccess::totalScalarDag(e, sourceGuards, qualificationVisits);
            auto& work = detail::RegionConstructionAccess::alternativeConstructionWork(e);
            detail::RegionAlternativeWork::add(work.guardQualification, qualificationVisits);
        }
    }
    if (project) {
        // Fixed-threshold projection is economical for pipe-uniform presence:
        // each pipe has one atomic presence predicate and edge activity uses
        // only that same vocabulary. Independent per-occurrence guards retain
        // the numeric representation. Sharing in this arena alone is not a
        // cost model for lowering at distinct original cuts.
        std::map<uint32_t, RegionExpressions::Id> presence;
        std::set<RegionExpressions::Id> inputs;
        uint64_t qualificationVisits = 0;
        for (const auto& payload : payloads) {
            const auto [it, inserted] = presence.emplace(payload.pipe, payload.present);
            ++qualificationVisits;
            if (!inserted && it->second != payload.present) { project = false; break; }
            if (!inserted) { continue; }
            if (e.constantValue(payload.present)) { continue; }
            uint64_t atomVisits = 0;
            const auto atoms = e.referencedInputs(ArrayRef<RegionExpressions::Id>{payload.present}, &atomVisits);
            detail::RegionAlternativeWork::add(qualificationVisits, atomVisits);
            if (atoms.size() != 1 || atoms.front().first != payload.present) { project = false; break; }
            inputs.insert(payload.present);
        }
        if (project) {
            SmallVector<RegionExpressions::Id> guards;
            for (auto edge : generators) { guards.push_back(edge.guard); }
            for (auto edge : native) { guards.push_back(edge.guard); }
            uint64_t guardVisits = 0;
            const auto atoms = e.referencedInputs(guards, &guardVisits);
            detail::RegionAlternativeWork::add(qualificationVisits, guardVisits);
            for (const auto& atom : atoms) {
                if (!inputs.count(atom.first)) { project = false; break; }
            }
        }
        auto& work = detail::RegionConstructionAccess::alternativeConstructionWork(e);
        detail::RegionAlternativeWork::add(work.guardQualification, qualificationVisits);
    }
    detail::FixedRankThresholds thresholds(e, out.projectionWork);
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
            e, frontier, out.columns[i], payloads[i].present, candidates, nativeSources, referenceRank, true,
            project ? &thresholds : nullptr);
        if (!step.error.empty()) { out.error = step.error; return out; }
        std::size_t at = 0;
        for (auto [source, guard] : incoming[i]) {
            auto keep = step.retained[at];
            ++at;
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
