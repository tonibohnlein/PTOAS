// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RepeatedRegionInternal.h"
#include "NumericalRepeatedBinding.h"
#include "PTO/Transforms/FrontierSynch/NumericalRepeatedSquaring.h"
#include "PTO/Transforms/FrontierSynch/RegionalNumericalInterface.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Key = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
bool charge(uint64_t& total, uint64_t amount)
{
    if (amount > UINT64_MAX - total) { return false; }
    total += amount; return true;
}
struct NumericalRepeatedExport {
    RegionalAnalysis body;
    std::vector<RegionalEvent> events;
    NumericalRepeatedSquaring repeated;
    std::optional<std::vector<uint32_t>> leaf(RegionalEvent event, bool reverse,
                                            NumericalChainQueryCost& cost) const
    {
        if (!validRegionalEvent(body, event)) { return {}; }
        return numericalLeafThresholds(*repeated.leaf,
            [&](uint32_t port, bool back) -> std::optional<bool> {
                if (body.numerical && body.numerical->query) {
                    auto answer = back ? body.numerical->query(events[port], event, cost) :
                                         body.numerical->query(event, events[port], cost);
                    if (answer) { return answer; }
                }
                auto answer = back ? regionalReachability(body, events[port], event) :
                                     regionalReachability(body, event, events[port]);
                auto value = answer ? body.expressions->constantValue(*answer) : std::nullopt;
                return value ? std::optional<bool>(*value != 0) : std::nullopt;
            }, reverse, cost);
    }
    std::optional<uint64_t> split(RegionalEvent& event) const
    {
        if (event.visits.empty()) { return {}; }
        auto copy = body.expressions->constantValue(event.visits.front());
        if (!copy) { return {}; }
        event.visits.erase(event.visits.begin());
        return validRegionalEvent(body, event) ? copy : std::nullopt;
    }
    std::optional<bool> query(RegionalEvent source, RegionalEvent target, NumericalChainQueryCost& cost) const
    {
        auto a = split(source), b = split(target);
        if (!a || !b) { return {}; }
        if (*a >= repeated.root->copies || *b >= repeated.root->copies || *b < *a) { return false; }
        if (*a == *b) {
            if (body.numerical && body.numerical->query) {
                if (auto answer = body.numerical->query(source, target, cost)) { return answer; }
            }
            if (!charge(cost.leafQueries, 1)) { return {}; }
            auto answer = regionalReachability(body, source, target);
            auto value = answer ? body.expressions->constantValue(*answer) : std::nullopt;
            return value ? std::optional<bool>(*value != 0) : std::nullopt;
        }
        auto from = leaf(source, false, cost), to = leaf(target, true, cost);
        if (!from || !to) { return {}; }
        NumericalSquaringQueryCost work;
        auto answer = repeated.across(*a, *from, *b, *to, work);
        if (!charge(cost.indexOperations, work.nodes) ||
            !charge(cost.indexOperations, work.indexOperations)) { return {}; }
        return answer;
    }
    std::optional<std::vector<uint32_t>> thresholds(RegionalEvent event, bool reverse,
                                                   NumericalChainQueryCost& cost) const
    {
        auto copy = split(event);
        if (!copy || *copy >= repeated.root->copies) { return {}; }
        auto values = leaf(event, reverse, cost);
        if (!values) { return {}; }
        NumericalSquaringQueryCost work;
        auto answer = repeated.thresholds(*copy, *values, reverse, work);
        if (!charge(cost.indexOperations, work.nodes) ||
            !charge(cost.indexOperations, work.indexOperations)) { return {}; }
        return answer;
    }
};
} // namespace
// This adapter is reached only after the invariant physical-copy contract was
// established by the repetition recognizer. Numerical counts alone do not
// establish that contract. Existing compact endpoint recipes remain shared.
void attachNumericalRepeatedExports(const std::shared_ptr<RepeatedRegionState>& state, RegionalAnalysis& out)
{
    auto count = state->e().constantValue(state->trips);
    if (!count || state->phaseCount != 1) { return; }
    std::map<Key, RegionalEvent> directory;
    auto event = [&](RegionalEvent value) {
        for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
            value.kind = kind;
            directory.emplace(Key{value.type, value.ordinal, value.kind, value.visits}, value);
        }
    };
    auto selectors = [&](const auto& list) { for (const auto& selector : list) { event(selector.event); } };
    for (const auto& edge : state->queryCrossings) {
        if (edge.displacement != 1) { return; }
        event(edge.source); event(edge.target);
    }
    for (const auto& boundary : state->body.storageBoundary) {
        selectors(boundary.firstWriters); selectors(boundary.lastWriters);
        for (const auto& item : boundary.firstReaders) { selectors(item.second); }
        for (const auto& item : boundary.lastReaders) { selectors(item.second); }
    }
    for (const auto& boundary : state->body.accessBoundary) { event(boundary.first.event); event(boundary.last.event); }
    for (const auto& boundary : state->body.deferredAccessBoundary) {
        event(boundary.first.event); event(boundary.last.event);
    }
    for (const auto& item : state->body.firstPayloads) { selectors(item.second); }
    for (const auto& item : state->body.lastPayloads) { selectors(item.second); }
    for (const auto& item : state->body.firstSitePayloads) { selectors(item.second); }
    std::vector<RegionalEvent> ports;
    for (const auto& item : directory) { ports.push_back(item.second); }
    NumericalBindingWork work;
    auto frame = evaluateNumericalRepeatedFrame(state->body, ports, state->queryCrossings, work);
    if (!charge(out.cost.numericalLeafQueries, work.bodyQueries) ||
        !charge(out.cost.numericalIndexOperations, work.orderingQueries) ||
        !charge(out.cost.numericalIndexOperations, work.nestedIndexOperations) || !frame) { return; }
    std::vector<NumericalRepeatedLink> links;
    for (const auto& edge : frame->links) { links.push_back({edge.source, edge.target, edge.native}); }
    auto owner = std::make_shared<NumericalRepeatedExport>();
    owner->body = state->body; owner->events = std::move(frame->events);
    owner->repeated = buildNumericalRepeatedSquaring(
        std::make_shared<const NumericalChainInterface>(std::move(frame->child)), links, *count);
    const auto& repeated = owner->repeated;
    if (!charge(out.cost.numericalMerges, repeated.cost.merges) ||
        !charge(out.cost.numericalIndexOperations, repeated.cost.indexOperations)) { return; }
    if (!repeated.error.empty() || !repeated.root) { return; }
    auto numerical = std::make_shared<RegionalNumericalInterface>();
    numerical->index = repeated.root->index;
    for (const auto& port : repeated.root->ports) {
        auto value = owner->events[port.port];
        value.visits.insert(value.visits.begin(), state->e().constant(port.copy));
        numerical->events.push_back(std::move(value));
    }
    for (const auto& chain : numerical->index->chains) {
        const auto& value = numerical->events[chain.front()];
        numerical->chainKeys.emplace_back(static_cast<uint32_t>(out.anchors[value.type].phase->kPipeValue), value.kind);
    }
    numerical->query = [owner](RegionalEvent a, RegionalEvent b, NumericalChainQueryCost& cost) {
        return owner->query(std::move(a), std::move(b), cost);
    };
    numerical->thresholds = [owner](RegionalEvent event, bool reverse, NumericalChainQueryCost& cost) {
        return owner->thresholds(std::move(event), reverse, cost);
    };
    // Keep the live state query: later overlays may replace its crossings and
    // explicitly clear this optional numerical snapshot. Consumers use the
    // numerical interface only while that snapshot remains installed.
    out.numerical = std::move(numerical);
}
} // namespace mlir::pto::frontiersynch
