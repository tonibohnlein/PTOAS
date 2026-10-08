// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteOverlay.h"
#include <limits>
#include <tuple>
namespace mlir::pto::frontiersynch {
using OverlayExpr = RegionExpressions::Id;
struct FiniteOverlayState {
    RegionalAnalysis base;
    std::shared_ptr<FiniteOverlayCost> cost = std::make_shared<FiniteOverlayCost>();
    std::vector<RegionalEvent> ports;
    std::vector<OverlayExpr> presence, active;
    std::vector<std::vector<OverlayExpr>> closure;
    using EventKey = std::tuple<uint32_t, OverlayExpr, PeriodicEventKind, std::vector<OverlayExpr>>;
    std::map<std::pair<EventKey, EventKey>, OverlayExpr> baseCache;
    RegionExpressions& arena() { return *base.expressions; }
    OverlayExpr no() { return arena().boolean(false); }
    OverlayExpr both(OverlayExpr a, OverlayExpr b) { return arena().land(a, b); }
    OverlayExpr either(OverlayExpr a, OverlayExpr b) { return arena().lor(a, b); }
    EventKey key(RegionalEvent event) const { return {event.type, event.ordinal, event.kind, event.visits}; }
    std::optional<OverlayExpr> baseQuery(RegionalEvent source, RegionalEvent target)
    {
        const auto pair = std::make_pair(key(source), key(target));
        if (auto found = baseCache.find(pair); found != baseCache.end()) { return found->second; }
        auto value = regionalReachability(base, source, target);
        if (!value || !arena().isBoolean(*value)) { return std::nullopt; }
        ++cost->baseQueries;
        baseCache.emplace(pair, *value);
        return value;
    }
    OverlayExpr equal(RegionalEvent a, RegionalEvent b)
    {
        ++cost->identityTests;
        if (a.type != b.type || a.kind != b.kind || a.visits.size() != b.visits.size()) { return no(); }
        auto result = arena().eq(a.ordinal, b.ordinal);
        for (std::size_t i = 0; i < a.visits.size(); ++i) {
            result = both(result, arena().eq(a.visits[i], b.visits[i]));
        }
        return result;
    }
    struct EndpointQueries {
        OverlayExpr direct;
        std::vector<OverlayExpr> incoming, outgoing, baseOutgoing;
    };
    std::optional<EndpointQueries> endpoints(RegionalEvent source, RegionalEvent target)
    {
        auto direct = baseQuery(source, target);
        if (!direct) { return std::nullopt; }
        EndpointQueries result{*direct, {}, {}, {}};
        std::vector<OverlayExpr> baseIncoming;
        for (const auto& port : ports) {
            auto before = baseQuery(source, port), after = baseQuery(port, target);
            if (!before || !after) { return std::nullopt; }
            baseIncoming.push_back(*before);
            result.baseOutgoing.push_back(*after);
        }
        for (std::size_t j = 0; j < ports.size(); ++j) {
            auto incoming = no(), outgoing = no();
            for (std::size_t i = 0; i < ports.size(); ++i) {
                incoming = either(incoming, both(baseIncoming[i], closure[i][j]));
                outgoing = either(outgoing, both(closure[j][i], result.baseOutgoing[i]));
            }
            result.incoming.push_back(incoming);
            result.outgoing.push_back(outgoing);
        }
        return result;
    }
    std::optional<OverlayExpr> query(RegionalEvent source, RegionalEvent target)
    {
        auto values = endpoints(source, target);
        if (!values) { return std::nullopt; }
        auto result = values->direct;
        for (std::size_t j = 0; j < ports.size(); ++j) {
            result = either(result, both(values->incoming[j], values->baseOutgoing[j]));
        }
        return arena().constructionError().empty() ? std::optional<OverlayExpr>(result) : std::nullopt;
    }
    std::optional<OverlayExpr> retain(RegionalEvent source, RegionalEvent target)
    {
        auto values = endpoints(source, target);
        if (!values) { return std::nullopt; }
        auto intermediate = no();
        for (std::size_t j = 0; j < ports.size(); ++j) {
            auto strict = both(arena().lnot(equal(ports[j], source)), arena().lnot(equal(ports[j], target)));
            intermediate = either(intermediate, both(strict, both(values->incoming[j], values->outgoing[j])));
        }
        auto result = arena().lnot(intermediate);
        return arena().constructionError().empty() ? std::optional<OverlayExpr>(result) : std::nullopt;
    }
};
FiniteOverlayAnalysis analyzeFiniteOverlay(RegionalAnalysis base,
    std::vector<FiniteOverlayDemand> demands, RegionalOrderQuery referenceBefore)
{
    FiniteOverlayAnalysis result;
    result.base = base;
    result.demands = std::move(demands);
    auto state = std::make_shared<FiniteOverlayState>();
    state->base = std::move(base);
    result.cost = state->cost;
    auto fail = [&](const char* message) {
        result.error = message;
        result.retained.clear();
    };
    if (!state->base.expressions || !state->base.capabilities.exactQueries ||
        !state->base.presence || !state->base.reachability || !referenceBefore ||
        result.demands.size() > std::numeric_limits<uint32_t>::max() / 2) {
        fail("finite overlay requires exact base queries, order and representable port identities"); return result;
    }
    auto& arena = state->arena();
    for (const auto& demand : result.demands) {
        if (demand.source.kind != PeriodicEventKind::Completion || demand.target.kind != PeriodicEventKind::Start ||
            !arena.isBoolean(demand.guard)) {
            fail("finite overlay requires guarded completion-to-start demands"); return result;
        }
        auto source = regionalPresence(state->base, demand.source);
        auto target = regionalPresence(state->base, demand.target);
        auto order = referenceBefore(demand.source, demand.target);
        if (!source || !target || !order || !arena.isBoolean(*source) || !arena.isBoolean(*target) ||
            !arena.isBoolean(*order)) {
            fail("finite overlay endpoint identity or reference-order query is invalid"); return result;
        }
        auto active = arena.land(demand.guard, arena.land(*source, *target));
        if (!arena.implies(active, *order)) {
            fail("finite overlay demand lacks a proved forward reference order"); return result;
        }
        state->ports.push_back(demand.source);
        state->ports.push_back(demand.target);
        state->presence.push_back(*source);
        state->presence.push_back(*target);
        state->active.push_back(active);
    }
    const auto count = state->ports.size();
    state->cost->ports = count;
    state->closure.assign(count, std::vector<OverlayExpr>(count, state->no()));
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < count; ++j) {
            auto query = state->baseQuery(state->ports[i], state->ports[j]);
            if (!query) { fail("finite overlay base port query is unavailable"); return result; }
            state->closure[i][j] = arena.land(*query, arena.land(state->presence[i], state->presence[j]));
        }
    }
    for (std::size_t i = 0; i < result.demands.size(); ++i) {
        state->closure[2 * i][2 * i + 1] = arena.lor(state->closure[2 * i][2 * i + 1], state->active[i]);
    }
    for (std::size_t via = 0; via < count; ++via) {
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < count; ++j) {
                state->closure[i][j] = arena.lor(state->closure[i][j],
                    arena.land(state->closure[i][via], state->closure[via][j]));
                ++state->cost->closureUpdates;
            }
        }
    }
    for (std::size_t i = 0; i < result.demands.size(); ++i) {
        const auto& demand = result.demands[i];
        auto baseReaches = state->baseQuery(demand.source, demand.target);
        auto retained = state->retain(demand.source, demand.target);
        if (!baseReaches || !retained) { fail("finite overlay cover query is unavailable"); return result; }
        auto guard = arena.land(state->active[i], arena.land(arena.lnot(*baseReaches), *retained));
        for (std::size_t earlier = 0; earlier < i; ++earlier) {
            const auto& previous = result.demands[earlier];
            auto equal = arena.land(state->equal(previous.source, demand.source),
                                    state->equal(previous.target, demand.target));
            guard = arena.land(guard, arena.lnot(arena.land(state->active[earlier], equal)));
        }
        result.retained.push_back(guard);
    }
    if (!arena.constructionError().empty()) { fail("finite overlay expression construction failed"); return result; }
    result.state = state;
    result.retainBase = [state](RegionalEvent source, RegionalEvent target) { return state->retain(source, target); };
    result.regional = state->base;
    result.regional.numerical.reset();
    result.regional.arithmeticRelations.reset();
    result.regional.symbolicStorage.reset();
    result.regional.reachability = [state](RegionalEvent source, RegionalEvent target) {
        return state->query(source, target);
    };
    result.regional.prepare = {};
    result.regional.prepareWithVisits = {};
    result.regional.prepareFiltered = {};
    result.regional.capabilities.endpointRecipes = false;
    result.regional.capabilities.contextualGuards = true;
    // Adding selected requirements changes query semantics, not memory effects
    // or storage boundary occurrences. The caller must certify the full input.
    return result;
}
} // namespace mlir::pto::frontiersynch
