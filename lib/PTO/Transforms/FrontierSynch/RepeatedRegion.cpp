// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#include "RepeatedRegionInternal.h"
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
bool RepeatedRegionState::buildBoundary()
{
    // Two copies are analysis descriptions with distinct child identities.
    // All body SSA prerequisites are internal, by the repetition contract;
    // running valueBridges here would mistake them for cross-visit edges.
    SequenceAnalysisState pair(function, body.expressions);
    for (unsigned i = 0; i < 2; ++i) {
        Child child;
        child.regional = body;
        child.anchors = body.anchors;
        pair.children.push_back(std::move(child));
    }
    if (!pair.importSummaries(false)) { error = pair.error; return false; }
    pair.bridges();
    if (!pair.error.empty() || !pair.closure()) { error = pair.error; return false; }
    for (const auto& edge : pair.crossings) {
        const auto& a = pair.ports[edge.source];
        const auto& b = pair.ports[edge.target];
        if (a.child != 0 || b.child != 1) { error = "invalid repeated boundary direction"; return false; }
        if (e().constantValue(edge.guard) != 0) {
            crossings.push_back({a.event(PeriodicEventKind::Completion), b.event(), edge.guard, false});
        }
    }
    for (const auto& [pipe, lasts] : body.lastPayloads) {
        auto found = body.firstPayloads.find(pipe);
        if (found == body.firstPayloads.end()) { error = "missing repeated native first selector"; return false; }
        for (const auto& last : lasts) {
            for (const auto& first : found->second) {
                for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                    auto a = last.event, b = first.event;
                    a.kind = kind; b.kind = kind;
                    crossings.push_back({a, b, e().land(last.present, first.present), true});
                }
            }
        }
    }
    return true;
}
bool RepeatedRegionState::closePorts()
{
    using Key = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
    std::map<Key, std::size_t> ids;
    auto port = [&](const RegionalEvent& event) {
        auto [it, added] = ids.emplace(Key{event.type, event.ordinal, event.kind, event.visits}, slots.size());
        if (added) { slots.push_back(event); }
        return it->second;
    };
    for (const auto& crossing : crossings) { port(crossing.source); port(crossing.target); }
    const auto count = slots.size();
    if (count >= UINT32_MAX || (count && count > distances.max_size() / count)) {
        error = "repeated port matrix size overflow"; return false;
    }
    infinity = count + 1; // Every finite shortest zero/one path has a simple representative.
    const auto absent = e().constant(infinity), zero = e().constant(0);
    distances.assign(count * count, absent);
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            auto reachable = regionalReachability(body, slots[a], slots[b]);
            if (!reachable) { error = "repeated body port query unavailable"; return false; }
            distances[a * count + b] = e().select(*reachable, zero, absent);
        }
    }
    for (const auto& crossing : crossings) {
        auto a = port(crossing.source), b = port(crossing.target);
        auto pa = regionalPresence(body, crossing.source), pb = regionalPresence(body, crossing.target);
        if (!pa || !pb) { error = "repeated crossing endpoint unavailable"; return false; }
        auto active = e().land(crossing.guard, e().land(*pa, *pb));
        auto candidate = e().select(active, e().constant(1), absent);
        auto& value = distances[a * count + b];
        value = e().select(e().lt(candidate, value), candidate, value);
    }
    for (std::size_t via = 0; via < count; ++via) {
        for (std::size_t a = 0; a < count; ++a) {
            for (std::size_t b = 0; b < count; ++b) {
                // Values are at most P+1; the checked P<UINT32_MAX makes addition total.
                auto candidate = e().add(distances[a * count + via], distances[via * count + b]);
                auto& value = distances[a * count + b];
                value = e().select(e().lt(candidate, value), candidate, value);
            }
        }
    }
    return e().constructionError().empty();
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::present(RegionalEvent event)
{
    if (event.visits.empty()) { return std::nullopt; }
    auto visit = event.visits.front();
    event.visits.erase(event.visits.begin());
    auto local = regionalPresence(body, event);
    if (!local) { return std::nullopt; }
    return e().land(e().lt(visit, trips), *local);
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::query(RegionalEvent source, RegionalEvent target)
{
    const auto key = std::make_pair(EventKey{source.type, source.ordinal, source.kind, source.visits},
                                    EventKey{target.type, target.ordinal, target.kind, target.visits});
    if (auto found = queryMemo.find(key); found != queryMemo.end()) { return found->second; }
    auto result = computeQuery(std::move(source), std::move(target));
    queryMemo.emplace(key, result);
    return result;
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::computeQuery(RegionalEvent source, RegionalEvent target)
{
    auto ps = present(source), pt = present(target);
    if (!ps || !pt) { return std::nullopt; }
    auto i = source.visits.front(), j = target.visits.front();
    source.visits.erase(source.visits.begin()); target.visits.erase(target.visits.begin());
    auto local = regionalReachability(body, source, target);
    if (!local) { return std::nullopt; }
    auto result = e().land(e().eq(i, j), *local), across = e().boolean(false);
    std::vector<Id> entries, exits;
    for (const auto& slot : slots) {
        auto exit = regionalReachability(body, source, slot), entry = regionalReachability(body, slot, target);
        if (!exit || !entry) { return std::nullopt; }
        exits.push_back(*exit); entries.push_back(*entry);
    }
    for (std::size_t a = 0; a < slots.size(); ++a) {
        for (std::size_t b = 0; b < slots.size(); ++b) {
            const auto distance = distances[a * slots.size() + b];
            auto reached = e().land(e().lt(distance, e().constant(infinity)), e().le(distance, e().sub(j, i)));
            across = e().lor(across, e().land(reached, e().land(exits[a], entries[b])));
        }
    }
    return e().land(e().land(*ps, *pt), e().lor(result, e().land(e().lt(i, j), across)));
}
} // namespace mlir::pto::frontiersynch
