// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_GUARDEDQUERYPROJECTION_H
#define PTO_FRONTIER_GUARDEDQUERYPROJECTION_H
#include "PTO/Transforms/FrontierSynch/Shared/GuardedRanks.h"
#include "RegionExpressionAlternatives.h"
#include <map>
namespace mlir::pto::frontiersynch::detail {
struct GuardedQueryGraph {
    const RegionExpressions* arena = nullptr;
    std::vector<GuardedRankPayload> payloads;
    std::vector<uint32_t> columns;
    std::vector<std::vector<GuardedRankEdge>> incoming;
    std::size_t columnCount = 0;
    bool totalGuards = false;
    uint64_t qualificationNodes = 0;
};
enum class GuardedQueryProjectionStatus {
    Projected, OriginalConstant, InvalidOriginal, MissingGraph, ForeignArena, PartialGuard,
    AdditionalInput, NoReduction, InvalidProjection
};
inline const char* guardedQueryProjectionName(GuardedQueryProjectionStatus status)
{
    switch (status) {
        case GuardedQueryProjectionStatus::Projected: return "projected";
        case GuardedQueryProjectionStatus::OriginalConstant: return "original-constant";
        case GuardedQueryProjectionStatus::InvalidOriginal: return "invalid-original-query";
        case GuardedQueryProjectionStatus::MissingGraph: return "reference-graph-unavailable";
        case GuardedQueryProjectionStatus::ForeignArena: return "foreign-expression-owner";
        case GuardedQueryProjectionStatus::PartialGuard: return "guard-totality-unqualified";
        case GuardedQueryProjectionStatus::AdditionalInput: return "additional-input-dependency";
        case GuardedQueryProjectionStatus::NoReduction: return "no-prebind-structural-reduction";
        case GuardedQueryProjectionStatus::InvalidProjection: return "invalid-projected-expression";
    }
    return "invalid-status";
}
struct GuardedQueryProjection {
    std::optional<RegionExpressions::Id> answer;
    GuardedQueryProjectionStatus status = GuardedQueryProjectionStatus::InvalidOriginal;
    RegionExpressions::Id additionalInput = RegionExpressions::invalid;
    uint64_t occurrences = 0, edges = 0, originalNodes = 0, projectedNodes = 0, constructedNodes = 0;
};
struct GuardedQueryProjectionWork {
    std::map<GuardedQueryProjectionStatus, uint64_t> outcomes;
    uint64_t occurrences = 0, edges = 0, dependencyNodes = 0, constructedNodes = 0, acceptedConstructedNodes = 0;
    void record(const GuardedQueryProjection& query)
    {
        auto add = [](uint64_t& to, uint64_t from) { to = from > UINT64_MAX - to ? UINT64_MAX : to + from; };
        add(outcomes[query.status], 1); add(occurrences, query.occurrences); add(edges, query.edges);
        add(dependencyNodes, query.originalNodes); add(dependencyNodes, query.projectedNodes);
        add(constructedNodes, query.constructedNodes);
        if (query.status == GuardedQueryProjectionStatus::Projected) {
            add(acceptedConstructedNodes, query.constructedNodes);
        }
    }
};
// Candidate construction only. Production retains the original numeric answer
// and can register this equivalent spelling for independently qualified late
// emission; input subsets alone do not certify historical availability.
// Compute only the requested Boolean reachability, leaving numeric ranks,
// retained cover and original graph guards untouched. All generated state is
// query-local and rolls back before selecting an unqualified original answer.
// The producer retains this graph with its unchanged numeric result; modifying
// public rank/row fields afterward is not a supported graph-owner operation.
inline GuardedQueryProjection projectGuardedRankQuery(
    RegionExpressions& e, const GuardedRanks& ranks, PeriodicEvent source, PeriodicEvent target,
    std::optional<RegionExpressions::Id> original = {})
{
    GuardedQueryProjection result;
    result.answer = original ? original : ranks.query(e, source, target);
    if (!result.answer || !e.constructionError().empty()) { return result; }
    if (e.constantValue(*result.answer)) {
        result.status = GuardedQueryProjectionStatus::OriginalConstant; return result;
    }
    const auto graph = ranks.queryGraph;
    if (!graph) { result.status = GuardedQueryProjectionStatus::MissingGraph; return result; }
    if (graph->arena != &e) { result.status = GuardedQueryProjectionStatus::ForeignArena; return result; }
    if (!graph->totalGuards) { result.status = GuardedQueryProjectionStatus::PartialGuard; return result; }
    RegionExpressions::Transaction transaction(e);
    const auto before = e.size();
    const auto no = e.boolean(false), yes = e.boolean(true);
    std::vector<RegionExpressions::Id> starts(graph->columnCount, no), completions(graph->columnCount, no);
    std::vector<RegionExpressions::Id> completed(target.type + 1, no);
    auto answer = no;
    for (uint32_t site = 0; site <= target.type; ++site) {
        ++result.occurrences;
        const auto column = graph->columns[site];
        const auto present = graph->payloads[site].present;
        auto start = starts[column];
        for (const auto& edge : graph->incoming[site]) {
            ++result.edges;
            start = e.lor(start, e.select(edge.guard, completed[edge.source], no));
        }
        const bool nativeSource = site >= source.type && column == graph->columns[source.type];
        if (nativeSource && source.kind == PeriodicEventKind::Start) { start = yes; }
        auto completion = e.lor(start, completions[column]);
        if (nativeSource) { completion = yes; }
        completed[site] = e.select(present, completion, no);
        if (site == target.type) {
            answer = target.kind == PeriodicEventKind::Start ? e.select(present, start, no) : completed[site];
        }
        // Absence skips this potential occurrence in both native chains.
        starts[column] = e.select(present, start, starts[column]);
        completions[column] = e.select(present, completion, completions[column]);
    }
    answer = e.select(graph->payloads[source.type].present, answer, no);
    result.constructedNodes = e.size() - before;
    if (answer == RegionExpressions::invalid || !e.constructionError().empty()) {
        result.status = GuardedQueryProjectionStatus::InvalidProjection;
        return result;
    }
    const auto oldInputs = e.referencedInputs(ArrayRef<RegionExpressions::Id>(*result.answer), &result.originalNodes);
    const auto newInputs = e.referencedInputs(ArrayRef<RegionExpressions::Id>(answer), &result.projectedNodes);
    llvm::DenseSet<RegionExpressions::Id> permitted;
    for (auto input : oldInputs) { permitted.insert(input.first); }
    for (auto input : newInputs) {
        if (!permitted.count(input.first)) {
            result.additionalInput = input.first;
            result.status = GuardedQueryProjectionStatus::AdditionalInput;
            return result;
        }
    }
    if (result.projectedNodes >= result.originalNodes) {
        result.status = GuardedQueryProjectionStatus::NoReduction;
        return result;
    }
    transaction.commit();
    result.answer = answer;
    result.status = GuardedQueryProjectionStatus::Projected;
    return result;
}
// Keep analysis/query identity numeric. The optional owned spelling is tried
// only by the historical materializer, with exact original-route fallback.
inline std::optional<RegionExpressions::Id> retainGuardedRankAlternative(
    RegionExpressions& e, const GuardedRanks& ranks, PeriodicEvent source, PeriodicEvent target)
{
    auto& work = RegionConstructionAccess::alternativeConstructionWork(e);
    RegionAlternativeWork::add(work.queries);
    const auto before = e.size();
    auto original = ranks.query(e, source, target);
    RegionAlternativeWork::add(work.numericConstructed, e.size() - before);
    if (!original || !e.constructionError().empty()) {
        RegionAlternativeWork::add(work.outcomes["invalid-original-query"]);
        return original;
    }
    RegionExpressions::Transaction scratch(e);
    const auto projected = projectGuardedRankQuery(e, ranks, source, target, original);
    RegionAlternativeWork::add(work.outcomes[guardedQueryProjectionName(projected.status)]);
    RegionAlternativeWork::add(work.occurrences, projected.occurrences);
    RegionAlternativeWork::add(work.edges, projected.edges);
    RegionAlternativeWork::add(work.dependencies, projected.originalNodes);
    RegionAlternativeWork::add(work.dependencies, projected.projectedNodes);
    RegionAlternativeWork::add(work.candidateConstructed, projected.constructedNodes);
    if (projected.status == GuardedQueryProjectionStatus::Projected && projected.answer &&
        RegionConstructionAccess::retainAlternative(e, *original, *projected.answer)) {
        RegionAlternativeWork::add(work.retainedCandidateConstructed, projected.constructedNodes);
        scratch.commit();
    }
    return original;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
