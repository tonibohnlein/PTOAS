// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BoundingRepetition.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Distance = BoundingPeriodDistance;
using Key = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
Key key(const RegionalEvent& event) { return {event.type, event.ordinal, event.kind, event.visits}; }
}
struct BoundingRepeatedQuery::State {
    RegionalAnalysis body;
    std::vector<RegionalEvent> ports;
    std::vector<RepeatedCrossing> crossings;
    std::map<Key, std::size_t> indices;
    using LocalMemo = std::map<std::pair<Key, Key>, std::optional<Id>>;
    std::shared_ptr<LocalMemo> localMemo = std::make_shared<LocalMemo>();
    std::map<std::pair<Key, Key>, Distance> positiveMemo;
    std::map<std::pair<Key, Key>, Id> unitMemo;
    std::vector<std::vector<Distance>> closure;
    bool closureAttempted = false, closureReady = false;
    BoundingRepetitionCost cost;
    std::string error;
    RegionExpressions& e() { return *body.expressions; }
    Distance absent() { return {e().boolean(false), e().boolean(false), e().constant(0)}; }
    Distance edge(Id guard, uint64_t weight) { return {guard, guard, e().constant(weight)}; }
    Distance minimum(Distance a, Distance b)
    {
        if (e().constantValue(a.reachable) == 0) { return b; }
        if (e().constantValue(b.reachable) == 0) { return a; }
        auto choose = e().land(a.representable,
            e().lor(e().lnot(b.representable), e().le(a.value, b.value)));
        return {e().lor(a.reachable, b.reachable), e().lor(a.representable, b.representable),
                e().select(choose, a.value, b.value)};
    }
    Distance plus(Distance a, Distance b)
    {
        auto reachable = e().land(a.reachable, b.reachable);
        if (e().constantValue(reachable) == 0) { return absent(); }
        auto fits = e().land(e().land(a.representable, b.representable),
                            e().le(a.value, e().sub(e().constant(UINT64_MAX), b.value)));
        // Overflow preserves reachability but cannot satisfy a uint64 gap.
        // The value is inspected only when representable holds; off-guard
        // wrapped addition is total and irrelevant. Keeping it unmasked also
        // preserves constant path weights through guarded min-plus closure.
        return {reachable, fits, e().add(a.value, b.value)};
    }
    std::optional<Id> local(RegionalEvent a, RegionalEvent b)
    {
        const auto pair = std::make_pair(key(a), key(b));
        if (auto found = localMemo->find(pair); found != localMemo->end()) { return found->second; }
        if (cost.bodyQueries == UINT64_MAX) { return {}; }
        ++cost.bodyQueries;
        auto answer = regionalReachability(body, std::move(a), std::move(b));
        if (answer && (*answer >= e().size() || !e().isBoolean(*answer))) { answer.reset(); }
        localMemo->emplace(pair, answer);
        return answer;
    }
    bool initialize()
    {
        const auto n = ports.size();
        if (n > UINT32_MAX) {
            error = "weighted repeated port work exceeds representation"; return false;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (!validRegionalEvent(body, ports[i]) || !indices.emplace(key(ports[i]), i).second) {
                error = "weighted repeated ports need distinct valid body identities"; return false;
            }
        }
        for (auto& crossing : crossings) {
            auto a = indices.find(key(crossing.source)), b = indices.find(key(crossing.target));
            if (!crossing.displacement || a == indices.end() || b == indices.end() ||
                crossing.guard >= e().size() || !e().isBoolean(crossing.guard)) {
                error = "weighted crossing needs positive distance and valid guarded port identities"; return false;
            }
            auto pa = regionalPresence(body, crossing.source), pb = regionalPresence(body, crossing.target);
            if (!pa || !pb) { error = "weighted crossing presence unavailable"; return false; }
            crossing.guard = e().land(crossing.guard, e().land(*pa, *pb));
        }
        return e().constructionError().empty();
    }
    bool ensureClosure()
    {
        if (closureAttempted) { return closureReady; }
        closureAttempted = true;
        const uint64_t n = ports.size();
        if (n && (n > UINT64_MAX / n || n*n > UINT64_MAX / n ||
                  n*n*n > UINT64_MAX - cost.relaxations)) {
            error = "weighted repeated port work exceeds representation"; return false;
        }
        closure.assign(n, std::vector<Distance>(n, absent()));
        for (std::size_t a = 0; a < n; ++a) {
            for (std::size_t b = 0; b < n; ++b) {
                auto query = local(ports[a], ports[b]);
                if (!query) { error = "weighted repeated body query unavailable"; return false; }
                closure[a][b] = edge(*query, 0);
            }
        }
        for (const auto& crossing : crossings) {
            auto a = indices.at(key(crossing.source)), b = indices.at(key(crossing.target));
            closure[a][b] = minimum(closure[a][b], edge(crossing.guard, crossing.displacement));
        }
        // Zero-weight aliases may form SCCs. The nonnegative min-plus Floyd
        // recurrence is valid without ordering those aliases or inventing payloads.
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = 0; b < n; ++b) {
                    ++cost.relaxations;
                    closure[a][b] = minimum(closure[a][b], plus(closure[a][k], closure[k][b]));
                }
            }
        }
        closureReady = e().constructionError().empty();
        return closureReady;
    }
    std::optional<Id> unit(const RegionalEvent& source, const RegionalEvent& target)
    {
        if (!validRegionalEvent(body, source) || !validRegionalEvent(body, target)) { return {}; }
        const auto pair = std::make_pair(key(source), key(target));
        if (auto found = unitMemo.find(pair); found != unitMemo.end()) { return found->second; }
        auto result = e().boolean(false);
        // Every crossing has positive displacement. A path of total distance
        // one therefore uses exactly one retained unit crossing, with arbitrary
        // zero-distance body paths before and after it. Keep this Boolean proof
        // instead of constructing an integer shortest-distance comparison.
        for (const auto& crossing : crossings) {
            if (cost.relaxations == UINT64_MAX) { return {}; }
            ++cost.relaxations;
            if (crossing.displacement != 1 || e().constantValue(crossing.guard) == 0) { continue; }
            auto prefix = local(source, crossing.source);
            if (!prefix) { return {}; }
            if (e().constantValue(*prefix) == 0) { continue; }
            auto suffix = local(crossing.target, target);
            if (!suffix) { return {}; }
            result = e().lor(result, e().land(crossing.guard, e().land(*prefix, *suffix)));
        }
        if (!e().constructionError().empty()) { return {}; }
        unitMemo.emplace(pair, result);
        return result;
    }
    std::optional<Distance> positive(const RegionalEvent& source, const RegionalEvent& target)
    {
        if (!validRegionalEvent(body, source) || !validRegionalEvent(body, target)) { return {}; }
        const auto pair = std::make_pair(key(source), key(target));
        if (auto found = positiveMemo.find(pair); found != positiveMemo.end()) { return found->second; }
        if (!ensureClosure()) { return {}; }
        std::vector<Distance> first(ports.size(), absent());
        for (const auto& crossing : crossings) {
            auto prefix = local(source, crossing.source);
            auto present = regionalPresence(body, crossing.target);
            if (!prefix || !present) { return {}; }
            auto guard = e().land(crossing.guard, e().land(*prefix, *present));
            const auto at = indices.at(key(crossing.target));
            if (cost.relaxations == UINT64_MAX) { return {}; }
            ++cost.relaxations;
            first[at] = minimum(first[at], edge(guard, crossing.displacement));
        }
        auto answer = absent();
        for (std::size_t b = 0; b < ports.size(); ++b) {
            auto suffix = local(ports[b], target);
            if (!suffix) { return {}; }
            if (e().constantValue(*suffix) == 0) { continue; }
            auto reached = absent();
            for (std::size_t a = 0; a < ports.size(); ++a) {
                if (cost.relaxations == UINT64_MAX) { return {}; }
                ++cost.relaxations;
                reached = minimum(reached, plus(first[a], closure[a][b]));
            }
            answer = minimum(answer, plus(reached, edge(*suffix, 0)));
        }
        if (!e().constructionError().empty()) { return {}; }
        positiveMemo.emplace(pair, answer);
        return answer;
    }
};
BoundingRepeatedQuery::BoundingRepeatedQuery(std::shared_ptr<State> implementation)
    : state(std::move(implementation)) {}
const RegionalAnalysis& BoundingRepeatedQuery::body() const { return state->body; }
llvm::ArrayRef<RegionalEvent> BoundingRepeatedQuery::ports() const { return state->ports; }
const BoundingRepetitionCost& BoundingRepeatedQuery::cost() const { return state->cost; }
std::shared_ptr<BoundingRepeatedQuery> BoundingRepeatedQuery::withCrossings(
    std::vector<RepeatedCrossing> crossings, std::string& error) const
{
    error.clear();
    auto next = std::make_shared<State>();
    next->body = state->body; next->ports = state->ports;
    next->crossings = std::move(crossings); next->localMemo = state->localMemo;
    if (!next->initialize()) {
        error = next->error.empty() ? next->e().constructionError() : next->error; return {};
    }
    return std::shared_ptr<BoundingRepeatedQuery>(new BoundingRepeatedQuery(std::move(next)));
}
std::optional<BoundingPeriodDistance> BoundingRepeatedQuery::distance(RegionalEvent source, RegionalEvent target)
{
    if (!validRegionalEvent(state->body, source) || !validRegionalEvent(state->body, target)) { return {}; }
    auto direct = state->local(source, target);
    auto positive = state->positive(source, target);
    if (!direct || !positive) { return {}; }
    return state->minimum(state->edge(*direct, 0), *positive);
}
std::optional<Id> BoundingRepeatedQuery::across(const RegionalEvent& source, const RegionalEvent& target, Id gap)
{
    if (gap >= state->e().size() || state->e().isBoolean(gap)) { return {}; }
    if (state->e().constantValue(gap) == 1) { return state->unit(source, target); }
    auto distance = state->positive(source, target);
    if (!distance) { return {}; }
    // Complete invariant native chains allow delaying the source by any number
    // of whole periods. Thus the minimum positive displacement characterizes
    // all later separations; this premise is supplied by the graph adapter.
    return state->e().land(distance->representable, state->e().le(distance->value, gap));
}
std::shared_ptr<BoundingRepeatedQuery> buildBoundingRepeatedQuery(RegionalAnalysis body,
    std::vector<RegionalEvent> ports, std::vector<RepeatedCrossing> crossings, std::string& error)
{
    error.clear();
    if (!body.expressions || !body.expressions->constructionError().empty() || !body.reachability || !body.presence) {
        error = "weighted repetition needs a valid exact body query and presence"; return {};
    }
    auto state = std::make_shared<BoundingRepeatedQuery::State>();
    state->body = std::move(body); state->ports = std::move(ports); state->crossings = std::move(crossings);
    if (!state->initialize()) {
        error = state->error.empty() ? state->e().constructionError() : state->error; return {};
    }
    return std::shared_ptr<BoundingRepeatedQuery>(new BoundingRepeatedQuery(std::move(state)));
}
} // namespace mlir::pto::frontiersynch
