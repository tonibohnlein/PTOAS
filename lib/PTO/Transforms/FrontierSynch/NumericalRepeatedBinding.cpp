// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "NumericalRepeatedBinding.h"
#include <algorithm>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Key = std::tuple<uint32_t, RegionExpressions::Id, PeriodicEventKind,
                       std::vector<RegionExpressions::Id>>;
Key key(const RegionalEvent& event) { return {event.type, event.ordinal, event.kind, event.visits}; }
// An optional comparison must never become an inconsistent std::sort predicate.
// Merge sorted runs, abandoning context evaluation immediately if unavailable.
bool order(std::vector<uint32_t>& values, const std::function<std::optional<bool>(uint32_t, uint32_t)>& before)
{
    for (std::size_t width = 1; width < values.size(); width *= 2) {
        auto previous = values;
        for (std::size_t first = 0; first < values.size(); first += 2 * width) {
            auto middle = std::min(first + width, values.size()), end = std::min(first + 2 * width, values.size());
            auto a = first, b = middle, at = first;
            while (a < middle && b < end) {
                auto earlier = before(previous[b], previous[a]);
                if (!earlier) { return false; }
                values[at++] = *earlier ? previous[b++] : previous[a++];
            }
            while (a < middle) { values[at++] = previous[a++]; }
            while (b < end) { values[at++] = previous[b++]; }
        }
    }
    return true;
}
} // namespace
std::optional<NumericalRepeatedBinding> bindNumericalRepeated(
    const RegionalAnalysis& body, const std::vector<RegionalEvent>& ports,
    const std::vector<RepeatedCrossing>& crossings, NumericalBindingWork& work)
{
    NumericalRepeatedBinding out;
    auto& e = *body.expressions;
    if (ports.size() > UINT32_MAX) { return {}; }
    for (const auto& crossing : crossings) {
        if (!e.constantValue(crossing.guard)) { return {}; }
    }
    std::map<Key, uint32_t> identities, originalIds;
    std::vector<RegionalEvent> events;
    std::map<std::pair<uint32_t, PeriodicEventKind>, std::vector<uint32_t>> groups;
    for (const auto& port : ports) {
        originalIds.emplace(key(port), out.ports.size());
        auto present = regionalPresence(body, port);
        auto active = present ? e.constantValue(*present) : std::nullopt;
        if (!active) { return {}; }
        if (!*active) { out.ports.push_back(UINT32_MAX); continue; }
        if (port.type >= body.anchors.size() || !body.anchors[port.type].phase) { return {}; }
        auto normalized = port;
        auto normalize = [&](RegionExpressions::Id& expression) {
            auto value = e.constantValue(expression);
            if (!value) { return false; }
            expression = e.constant(*value); return true;
        };
        if (!normalize(normalized.ordinal)) { return {}; }
        for (auto& visit : normalized.visits) { if (!normalize(visit)) { return {}; } }
        auto [found, inserted] = identities.emplace(key(normalized), events.size());
        out.ports.push_back(found->second);
        if (inserted) {
            const auto pipe = static_cast<uint32_t>(body.anchors[port.type].phase->kPipeValue);
            groups[{pipe, port.kind}].push_back(events.size());
            events.push_back(std::move(normalized));
        }
    }
    std::vector<std::vector<uint32_t>> chains;
    for (auto& [group, list] : groups) {
        if (!order(list, [&](uint32_t a, uint32_t b) -> std::optional<bool> {
            if (work.orderingQueries == UINT64_MAX) { return {}; }
            ++work.orderingQueries;
            if (body.referenceBefore) {
                auto before = body.referenceBefore(events[a], events[b]);
                auto value = before ? e.constantValue(*before) : std::nullopt;
                return value ? std::optional<bool>(*value != 0) : std::nullopt;
            }
            if (!events[a].visits.empty() || !events[b].visits.empty()) { return {}; }
            return std::make_pair(*e.constantValue(events[a].ordinal), events[a].type) <
                   std::make_pair(*e.constantValue(events[b].ordinal), events[b].type);
        })) { return {}; }
        chains.push_back(std::move(list));
    }
    auto child = buildNumericalChainInterface(std::move(chains), [&](uint32_t a, uint32_t b) -> std::optional<bool> {
        if (work.bodyQueries == UINT64_MAX) { return {}; }
        ++work.bodyQueries;
        auto query = regionalReachability(body, events[a], events[b]);
        auto value = query ? e.constantValue(*query) : std::nullopt;
        return value ? std::optional<bool>(*value != 0) : std::nullopt;
    });
    if (!child.error.empty()) { return {}; }
    std::vector<NumericalWeightedCrossing> links;
    std::vector<std::size_t> originals;
    for (std::size_t i = 0; i < crossings.size(); ++i) {
        const auto& crossing = crossings[i];
        if (e.constantValue(crossing.guard) == 0) { continue; }
        auto a = originalIds.find(key(crossing.source)), b = originalIds.find(key(crossing.target));
        if (a == originalIds.end() || b == originalIds.end() ||
            out.ports[a->second] == UINT32_MAX || out.ports[b->second] == UINT32_MAX) { return {}; }
        links.push_back({out.ports[a->second], out.ports[b->second], crossing.displacement, crossing.native});
        originals.push_back(i);
    }
    out.analysis = buildNumericalWeightedRepetition(child, links);
    work.index = out.analysis.cost;
    if (!out.analysis.error.empty()) { return {}; }
    out.covers = crossings;
    for (auto& crossing : out.covers) { crossing.guard = e.boolean(false); }
    for (std::size_t canonical = 0; canonical < out.analysis.crossings.size(); ++canonical) {
        if (out.analysis.retained[canonical]) {
            const auto original = originals[out.analysis.representatives[canonical]];
            out.covers[original].guard = crossings[original].guard;
        }
    }
    out.events = std::move(events);
    return out;
}
} // namespace mlir::pto::frontiersynch
