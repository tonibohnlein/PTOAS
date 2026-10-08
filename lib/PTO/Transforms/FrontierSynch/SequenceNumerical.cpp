// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical specialization only; symbolic guards/order keep the general path.
#include "SequenceAnalysisInternal.h"
#include "ChainInterfaceInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
using EventIdentity = std::tuple<uint32_t, RegionExpressions::Id, PeriodicEventKind>;
using ChainKey = std::pair<uint32_t, PeriodicEventKind>;
std::shared_ptr<const RegionalNumericalInterface> leafInterface(const RegionalAnalysis& region,
    const std::vector<RegionalEvent>& events)
{
    if (region.referenceBefore) { return {}; }
    auto result = std::make_shared<RegionalNumericalInterface>();
    result->events = events;
    std::map<ChainKey, std::vector<uint32_t>> lists;
    for (uint32_t id = 0; id < events.size(); ++id) {
        const auto& event = events[id];
        auto* phase = region.anchors[event.type].phase;
        if (!phase) { return {}; }
        lists[{static_cast<uint32_t>(phase->kPipeValue), event.kind}].push_back(id);
    }
    std::vector<std::vector<uint32_t>> chains;
    for (auto& [key, list] : lists) {
        std::stable_sort(list.begin(), list.end(), [&](uint32_t a, uint32_t b) {
            const auto& x = events[a]; const auto& y = events[b];
            return std::make_pair(*region.expressions->constantValue(x.ordinal), x.type) <
                   std::make_pair(*region.expressions->constantValue(y.ordinal), y.type);
        });
        for (std::size_t i = 1; i < list.size(); ++i) {
            const auto& a = events[list[i - 1]]; const auto& b = events[list[i]];
            if (a.type == b.type && a.ordinal == b.ordinal) { return {}; }
        }
        result->chainKeys.push_back(key); chains.push_back(std::move(list));
    }
    auto index = buildNumericalChainInterface(std::move(chains), [&](uint32_t a, uint32_t b) -> std::optional<bool> {
        auto answer = regionalReachability(region, events[a], events[b]);
        auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
        return value ? std::optional<bool>(*value != 0) : std::nullopt;
    });
    if (!index.error.empty()) { return {}; }
    result->index = std::make_shared<const NumericalChainInterface>(std::move(index));
    result->query = [region](RegionalEvent a, RegionalEvent b, NumericalChainQueryCost& cost) -> std::optional<bool> {
        if (!validRegionalEvent(region, a) || !validRegionalEvent(region, b)) { return std::nullopt; }
        ++cost.leafQueries;
        auto answer = region.reachability(a, b);
        auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
        return value ? std::optional<bool>(*value != 0) : std::nullopt;
    };
    result->thresholds = [region, events, index = result->index](RegionalEvent event, bool reverse,
        NumericalChainQueryCost& cost) -> std::optional<std::vector<uint32_t>> {
        if (!validRegionalEvent(region, event)) { return std::nullopt; }
        return numericalLeafThresholds(*index, [&](uint32_t port, bool back) -> std::optional<bool> {
            auto answer = back ? regionalReachability(region, events[port], event) :
                                 regionalReachability(region, event, events[port]);
            auto value = answer ? region.expressions->constantValue(*answer) : std::nullopt;
            return value ? std::optional<bool>(*value != 0) : std::nullopt;
        }, reverse, cost);
    };
    return result;
}
bool childSelection(const RegionalNumericalInterface& index, const std::vector<RegionalEvent>& events,
                    std::vector<uint32_t>& selected)
{
    if (!index.index || !index.index->error.empty() || index.events.size() != index.index->chain.size() ||
        index.chainKeys.size() != index.index->chains.size() || !index.thresholds || !index.query) { return false; }
    std::map<EventIdentity, uint32_t> ids;
    for (uint32_t id = 0; id < index.events.size(); ++id) {
        const auto& event = index.events[id];
        if (!event.visits.empty() || !ids.emplace(EventIdentity{event.type, event.ordinal, event.kind}, id).second) {
            return false;
        }
    }
    for (const auto& event : events) {
        auto found = ids.find({event.type, event.ordinal, event.kind});
        if (found == ids.end()) { return false; }
        selected.push_back(found->second);
    }
    return true;
}
} // namespace
bool SequenceAnalysisState::numericalCrossingReduction()
{
    if (children.size() != 2 || !portChoices.empty()) { return false; }
    // Decline symbolic guards before any leaf callbacks. The symbolic reducer
    // proves mutually exclusive candidates without expanding child queries.
    for (const auto& edge : crossings) {
        if (!expressions.constantValue(edge.guard)) { return false; }
    }
    if (auto found = incoming.find(1); found != incoming.end()) {
        for (const auto& link : found->second) { if (!expressions.constantValue(link.guard)) { return false; } }
    }
    std::array<std::vector<RegionalEvent>, 2> events;
    std::vector<NumericalChainSelection> selection;
    for (uint32_t p = 0; p < ports.size(); ++p) {
        const auto& port = ports[p];
        if (port.child > 1 || !expressions.constantValue(port.ordinal) || !port.visits.empty() ||
            (!children[port.child].regional.numerical && expressions.constantValue(present(p)) != 1)) { return false; }
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            selection.push_back({port.child, static_cast<uint32_t>(events[port.child].size())});
            events[port.child].push_back(port.event(kind));
        }
    }
    std::array<std::shared_ptr<const RegionalNumericalInterface>, 2> indices;
    std::array<std::vector<uint32_t>, 2> selected;
    uint64_t leafQueries = 0, leafOperations = 0, reused = 0;
    for (unsigned child = 0; child < 2; ++child) {
        indices[child] = children[child].regional.numerical;
        if (indices[child]) { ++reused; }
        else { indices[child] = leafInterface(children[child].regional, events[child]); }
        // A cached hierarchy missing a requested bridge endpoint declines; it
        // must not become a "leaf" that recursively expands semantic queries.
        if (!indices[child] || !childSelection(*indices[child], events[child], selected[child])) { return false; }
        if (!children[child].regional.numerical) {
            leafQueries += indices[child]->index->queries; leafOperations += indices[child]->index->operations;
        }
    }
    for (auto& event : selection) { event.event = selected[event.child][event.event]; }
    std::vector<NumericalCrossing> links;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> lookup;
    if (auto found = incoming.find(1); found != incoming.end()) {
        for (const auto& link : found->second) {
            if (link.source >= selection.size() || link.target >= selection.size() ||
                selection[link.source].child != 0 || selection[link.target].child != 1) { return false; }
            if (expressions.constantValue(link.guard) == 0) { continue; }
            auto pair = std::make_pair(selection[link.source].event, selection[link.target].event);
            if (lookup.emplace(pair, links.size()).second) { links.push_back({pair.first, pair.second}); }
        }
    }
    auto retained = chain::reduce(*indices[0]->index, *indices[1]->index, links, leafOperations);
    if (!retained) { return false; }
    std::set<std::pair<uint32_t, uint32_t>> nativePairs;
    for (const auto& edge : nativeValueCrossings) {
        if (expressions.constantValue(edge.guard) == 1) {
            nativePairs.emplace(selection[2 * edge.source + 1].event, selection[2 * edge.target].event);
        }
    }
    std::vector<Expr> guards;
    for (const auto& edge : crossings) {
        auto guard = expressions.constantValue(edge.guard);
        if (!guard || ports[edge.source].child != 0 || ports[edge.target].child != 1) { return false; }
        auto found = lookup.find({selection[2 * edge.source + 1].event, selection[2 * edge.target].event});
        if (*guard && found == lookup.end()) { return false; }
        const auto pair = std::make_pair(selection[2 * edge.source + 1].event, selection[2 * edge.target].event);
        guards.push_back(expressions.boolean(*guard && (*retained)[found->second] && !nativePairs.count(pair)));
    }
    std::map<ChainKey, uint32_t> directory;
    for (const auto& index : indices) {
        for (const auto& key : index->chainKeys) { directory.emplace(key, 0); }
    }
    std::vector<ChainKey> keys;
    for (auto& [key, id] : directory) { id = keys.size(); keys.push_back(key); }
    std::array<std::vector<uint32_t>, 2> maps;
    for (unsigned child = 0; child < 2; ++child) {
        for (const auto& key : indices[child]->chainKeys) { maps[child].push_back(directory.at(key)); }
    }
    auto merged = buildNumericalChainMerge(indices[0]->index, indices[1]->index, links, selection, maps[0], maps[1]);
    if (!merged.index.error.empty()) { return false; }
    for (std::size_t i = 0; i < guards.size(); ++i) { crossings[i].guard = guards[i]; }
    numerical = std::make_shared<NumericalChainMerge>(std::move(merged));
    numericalChildren = std::move(indices); numericalSelection = std::move(selection);
    numericalChainKeys = std::move(keys);
    costs.numericalLeafQueries += leafQueries;
    costs.numericalIndexOperations += leafOperations + numerical->index.operations;
    costs.numericalReusedChildren += reused; ++costs.numericalMerges;
    // Keep only reduced links for queries that later require symbolic fallback.
    // The numerical query path itself never scans these links or caches pairs.
    std::vector<EntryLink> reduced;
    std::set<uint32_t> seen;
    if (auto found = incoming.find(1); found != incoming.end()) {
        for (const auto& link : found->second) {
            auto edge = lookup.find({numericalSelection[link.source].event, numericalSelection[link.target].event});
            if (expressions.constantValue(link.guard) == 1 && edge != lookup.end() &&
                (*retained)[edge->second] && seen.insert(edge->second).second) { reduced.push_back(link); }
        }
    }
    incoming.clear(); incoming.emplace(1, std::move(reduced)); reachabilityCache.clear();
    std::vector<Crossing> kept;
    for (const auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) != 0) { kept.push_back(edge); }
    }
    crossings.swap(kept); // Release capacity for discarded candidates too.
    crossingIds.clear(); std::vector<Crossing>().swap(nativeValueCrossings);
    return true;
}
std::optional<std::vector<uint32_t>> SequenceAnalysisState::numericalThresholds(
    SequenceEvent event, bool reverse, NumericalChainQueryCost& cost)
{
    if (!numerical || event.child > 1) { return std::nullopt; }
    auto port = portIds.find({event.child, event.type, event.ordinal, event.visits});
    if (port != portIds.end()) {
        auto id = 2 * port->second + (event.kind == PeriodicEventKind::Completion);
        ++cost.indexOperations;
        return reverse ? numerical->index.reverse[id] : numerical->index.forward[id];
    }
    auto child = numericalChildren[event.child]->thresholds(
        {event.type, event.ordinal, event.kind, event.visits}, reverse, cost);
    return child ? numerical->propagate(event.child, *child, reverse, cost) : std::nullopt;
}
std::optional<Expr> SequenceAnalysisState::numericalReachability(
    SequenceEvent source, SequenceEvent target, NumericalChainQueryCost& cost)
{
    if (!numerical || source.child > 1 || target.child > 1) { return std::nullopt; }
    if (source.child > target.child) { return no(); }
    auto aPort = portIds.find({source.child, source.type, source.ordinal, source.visits});
    auto bPort = portIds.find({target.child, target.type, target.ordinal, target.visits});
    if (aPort != portIds.end() && bPort != portIds.end()) {
        ++cost.indexOperations;
        return expressions.boolean(numerical->index.reaches(
            2 * aPort->second + (source.kind == PeriodicEventKind::Completion),
            2 * bPort->second + (target.kind == PeriodicEventKind::Completion)));
    }
    if (source.child == target.child) {
        auto answer = numericalChildren[source.child]->query(
            {source.type, source.ordinal, source.kind, source.visits},
            {target.type, target.ordinal, target.kind, target.visits}, cost);
        return answer ? std::optional<Expr>(expressions.boolean(*answer)) : std::nullopt;
    }
    auto a = numericalChildren[0]->thresholds(
        {source.type, source.ordinal, source.kind, source.visits}, false, cost);
    auto b = numericalChildren[1]->thresholds(
        {target.type, target.ordinal, target.kind, target.visits}, true, cost);
    if (!a || !b) { return std::nullopt; }
    auto answer = numerical->crosses(*a, *b, cost);
    return answer ? std::optional<Expr>(expressions.boolean(*answer)) : std::nullopt;
}
} // namespace mlir::pto::frontiersynch
