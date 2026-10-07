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
    for (const auto& edge : pair.crossings) {
        if (e().constantValue(edge.guard) == 0) { continue; }
        const auto& a = pair.ports[edge.source];
        const auto& b = pair.ports[edge.target];
        queryCrossings.push_back({a.event(PeriodicEventKind::Completion), b.event(), edge.guard, false});
    }
    if (!pair.error.empty() || !pair.closure()) { error = pair.error; return false; }
    for (const auto& edge : pair.crossings) {
        if (e().constantValue(edge.guard) == 0) { continue; }
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
                    auto native = RepeatedCrossing{a, b, e().land(last.present, first.present), true};
                    if (e().constantValue(native.guard) == 0) { continue; }
                    crossings.push_back(native);
                    queryCrossings.push_back(native);
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
    for (const auto& crossing : queryCrossings) { port(crossing.target); }
    const auto count = slots.size();
    if (count >= UINT32_MAX || (count && count > distances.max_size() / count)) {
        error = "repeated port matrix size overflow"; return false;
    }
    // A shortest path after its first crossing visits at most R distinct
    // crossing targets. Thus every finite minimum needs at most R crossings.
    infinity = count + 1;
    SmallVector<Id> presences;
    for (const auto& crossing : queryCrossings) {
        for (const auto& endpoint : {crossing.source, crossing.target}) {
            auto present = regionalPresence(body, endpoint);
            if (!present) { error = "repeated boundary presence unavailable"; return false; }
            presences.push_back(*present);
        }
    }
    portEnable = e().boolean(true);
    SmallVector<std::pair<Id, Id>> bindings;
    for (auto conjunct : e().commonBooleanConjuncts(presences)) {
        portEnable = e().land(portEnable, conjunct);
        bindings.emplace_back(conjunct, e().boolean(true));
    }
    if (!bindings.empty()) { portCondition = std::make_unique<RegionExpressions::Substitution>(bindings); }
    distances.assign(count * count, RegionExpressions::invalid);
    targetCrossings.resize(count);
    for (std::size_t i = 0; i < queryCrossings.size(); ++i) {
        targetCrossings[port(queryCrossings[i].target)].push_back(i);
    }
    return e().constructionError().empty();
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::portDistance(std::size_t source, std::size_t target)
{
    // D(a,b,k) permits only ports [0,k) as intermediate vertices. Memoized
    // requests share exactly the Floyd recurrence, without constructing body
    // queries for pairs that no exported reachability query needs.
    const auto count = slots.size();
    if (source >= count || target >= count) { return std::nullopt; }
    const auto absent = e().constant(infinity), zero = e().constant(0);
    using Key = std::tuple<std::size_t, std::size_t, std::size_t>;
    const Key requested{source, target, count};
    SmallVector<Key> pending{requested};
    while (!pending.empty()) {
        const auto key = pending.back();
        if (distanceMemo.count(key)) { pending.pop_back(); continue; }
        const auto [a, b, level] = key;
        auto finish = [&](Id answer) { distanceMemo.emplace(key, answer); pending.pop_back(); };
        if (a == b) { finish(zero); continue; }
        if (level == 0) {
            auto& base = distances[a * count + b];
            if (base == RegionExpressions::invalid) {
                auto reachable = crossingStep(slots[a], b);
                if (!reachable) { error = "repeated body port query unavailable"; return std::nullopt; }
                base = *reachable;
            }
            finish(base); continue;
        }
        const auto via = level - 1;
        const Key direct{a, b, via};
        auto old = distanceMemo.find(direct);
        if (old == distanceMemo.end()) { pending.push_back(direct); continue; }
        if (a == via || b == via || old->second == zero) { finish(old->second); continue; }
        const Key left{a, via, via};
        auto prefix = distanceMemo.find(left);
        if (prefix == distanceMemo.end()) { pending.push_back(left); continue; }
        if (prefix->second == absent) { finish(old->second); continue; }
        const Key right{via, b, via};
        auto suffix = distanceMemo.find(right);
        if (suffix == distanceMemo.end()) { pending.push_back(right); continue; }
        finish(e().boundedMinPlus(old->second, prefix->second, suffix->second, infinity));
    }
    if (!e().constructionError().empty()) { return std::nullopt; }
    return distanceMemo.at(requested);
}
RepeatedRegionState::Id RepeatedRegionState::underPortEnable(Id expression)
{
    return portCondition ? e().substitute(expression, *portCondition) : expression;
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::bodyPortQuery(
    const RegionalEvent& source, const RegionalEvent& target)
{
    const auto key = std::make_pair(EventKey{source.type, source.ordinal, source.kind, source.visits},
                                   EventKey{target.type, target.ordinal, target.kind, target.visits});
    if (auto found = bodyPortMemo.find(key); found != bodyPortMemo.end()) { return found->second; }
    auto answer = regionalReachability(body, source, target);
    if (answer) { *answer = underPortEnable(*answer); }
    bodyPortMemo.emplace(key, answer);
    return answer;
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::crossingStep(
    const RegionalEvent& source, std::size_t target)
{
    if (target >= targetCrossings.size()) { return std::nullopt; }
    auto enabled = e().boolean(false);
    for (auto index : targetCrossings[target]) {
        const auto& crossing = queryCrossings[index];
        auto present = regionalPresence(body, crossing.target);
        if (!present) { return std::nullopt; }
        auto guard = underPortEnable(e().land(crossing.guard, *present));
        if (e().constantValue(guard) == 0) { continue; }
        auto prefix = bodyPortQuery(source, crossing.source);
        if (!prefix) { return std::nullopt; }
        enabled = e().lor(enabled, e().select(guard, *prefix, e().boolean(false)));
    }
    return e().select(enabled, e().constant(1), e().constant(infinity));
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
    // Absent endpoints cannot participate in a path. Establish this before
    // requesting any body query or quotient distance for their coordinates.
    if (e().constantValue(*ps) == 0 || e().constantValue(*pt) == 0) {
        return e().boolean(false);
    }
    auto i = source.visits.front(), j = target.visits.front();
    source.visits.erase(source.visits.begin()); target.visits.erase(target.visits.begin());
    const auto sameVisit = e().eq(i, j), laterVisit = e().lt(i, j);
    Id result = e().boolean(false), across = e().boolean(false);
    if (e().constantValue(sameVisit) != 0) {
        auto local = regionalReachability(body, source, target);
        if (!local) { return std::nullopt; }
        result = e().land(sameVisit, *local);
    }
    if (e().constantValue(sameVisit) == 1 || e().constantValue(laterVisit) == 0) {
        return e().land(e().land(*ps, *pt), result);
    }
    const auto gap = e().sub(j, i);
    if (e().constantValue(gap) == 1) {
        // Reference-forward paths between adjacent visits cross their boundary
        // exactly once. Answer this query directly instead of constructing all
        // shortest-distance alternatives and then comparing them with one.
        for (const auto& crossing : queryCrossings) {
            auto guard = underPortEnable(crossing.guard);
            if (e().constantValue(guard) == 0) { continue; }
            auto prefix = bodyPortQuery(source, crossing.source);
            if (!prefix) { return std::nullopt; }
            if (e().constantValue(*prefix) == 0) { continue; }
            auto suffix = bodyPortQuery(crossing.target, target);
            if (!suffix) { return std::nullopt; }
            auto path = e().select(guard, e().land(*prefix, *suffix), e().boolean(false));
            across = e().lor(across, path);
        }
        across = e().select(portEnable, across, e().boolean(false));
        return e().land(e().land(*ps, *pt), e().lor(result, e().land(laterVisit, across)));
    }
    const auto absent = e().constant(infinity);
    auto distance = absent;
    // The contracted quotient stops immediately after a crossing. Every
    // query therefore appends its final body path, even when target is itself
    // a quotient vertex: it may be reached from another crossing target.
    for (std::size_t b = 0; b < slots.size(); ++b) {
        auto entry = bodyPortQuery(slots[b], target);
        if (!entry) { return std::nullopt; }
        if (e().constantValue(*entry) == 0) { continue; }
        auto selected = distanceFrom(source, b);
        if (!selected) { return std::nullopt; }
        if (*selected == absent) { continue; }
        auto candidate = e().select(*entry, *selected, absent);
        distance = e().minimum(candidate, distance);
    }
    across = e().select(portEnable,
        e().land(e().lt(distance, absent), e().le(distance, gap)), e().boolean(false));
    return e().land(e().land(*ps, *pt), e().lor(result, e().land(laterVisit, across)));
}
std::optional<RepeatedRegionState::Id> RepeatedRegionState::distanceFrom(
    const RegionalEvent& source, std::size_t column)
{
    const auto count = slots.size();
    if (column >= count) { return std::nullopt; }
    const EventKey key{source.type, source.ordinal, source.kind, source.visits};
    auto [memo, added] = sourceDistances.try_emplace(key);
    if (added) { memo->second.assign(2 * count, RegionExpressions::invalid); }
    auto& values = memo->second;
    if (values[column] != RegionExpressions::invalid) { return values[column]; }
    const auto absent = e().constant(infinity);
    auto best = absent;
    for (std::size_t a = 0; a < count; ++a) {
        // Charge the first crossing explicitly. Taking a zero-cost diagonal
        // for a source that is itself a target port would revive same-visit
        // paths without establishing any crossing support.
        auto& first = values[count + a];
        if (first == RegionExpressions::invalid) {
            auto query = crossingStep(source, a);
            if (!query) { return std::nullopt; }
            first = *query;
        }
        if (first == absent) { continue; }
        auto distance = portDistance(a, column);
        if (!distance) { return std::nullopt; }
        if (*distance == absent) { continue; }
        best = e().boundedMinPlus(best, first, *distance, infinity);
    }
    // Each original path alternates complete body paths and unit crossings.
    // Contracting the former leaves exactly this first step and a shortest
    // path between crossing targets. Guards remain symbolic and correlated;
    // only the final scalar threshold sees the requested visit displacement.
    values[column] = best;
    return best;
}

} // namespace mlir::pto::frontiersynch
