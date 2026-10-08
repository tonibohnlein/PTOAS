// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common exact regional contract. All expressions belong to the supplied arena.
#include "RepeatedRegionInternal.h"
#include "RepeatedReadOnlyStorage.h"
namespace mlir::pto::frontiersynch {
namespace {
std::string invalidRepeatedFrames(const RegionalAnalysis& body)
{
    if (body.occurrenceLoops.size() != body.anchors.size() ||
        (!body.outerLoops.empty() && body.outerLoops.size() != body.anchors.size()) ||
        (!body.outerDivisors.empty() && (body.outerLoops.empty() ||
            body.outerDivisors.size() != body.anchors.size()))) {
        return "repeated coordinate frames and divisors must match payload types";
    }
    for (std::size_t type = 0; type < body.outerDivisors.size(); ++type) {
        if (body.outerDivisors[type].size() != body.outerLoops[type].size() ||
            llvm::any_of(body.outerDivisors[type], [](uint64_t value) { return value == 0; })) {
            return "repeated coordinate divisors must be positive and match their frames";
        }
    }
    return {};
}
} // namespace
void liftRepeatedSelectors(RegionalAnalysis& out, RegionExpressions::Id trips)
{
    auto& e = *out.expressions;
    auto nonempty = e.lt(e.constant(0), trips), last = e.sub(trips, e.constant(1));
    auto lift = [&](RegionalSelector& selector, bool final) {
        selector.event.visits.insert(selector.event.visits.begin(), final ? last : e.constant(0));
        selector.present = e.land(nonempty, selector.present);
    };
    for (auto& cell : out.storageBoundary) {
        for (auto& value : cell.firstWriters) { lift(value, false); }
        for (auto& value : cell.lastWriters) { lift(value, true); }
        for (auto& [pipe, values] : cell.firstReaders) { for (auto& value : values) { lift(value, false); } }
        for (auto& [pipe, values] : cell.lastReaders) { for (auto& value : values) { lift(value, true); } }
    }
    for (auto& access : out.accessBoundary) { lift(access.first, false); lift(access.last, true); }
    for (auto& access : out.deferredAccessBoundary) { lift(access.first, false); lift(access.last, true); }
    for (auto& [pipe, values] : out.firstPayloads) { for (auto& value : values) { lift(value, false); } }
    for (auto& [pipe, values] : out.lastPayloads) { for (auto& value : values) { lift(value, true); } }
    for (auto& [type, values] : out.firstSitePayloads) { for (auto& value : values) { lift(value, false); } }
}
RepeatedRegionAnalysis repeatInvariantRegion(func::FuncOp function, scf::ForOp loop,
    RegionalAnalysis body, RegionExpressions::Id trips)
{
    RepeatedRegionAnalysis result;
    if (body.storageSelectors || !body.symbolicStorageEffects.empty()) {
        return repeatSymbolicStorageRegion(function, loop, std::move(body), trips);
    }
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function || !body.expressions ||
        trips >= body.expressions->size() || body.expressions->isBoolean(trips)) {
        result.error = "repetition requires original loop and integer trip expression"; return result;
    }
    result.error = invalidRepeatedFrames(body);
    if (!result.error.empty()) { return result; }
    auto state = std::make_shared<RepeatedRegionState>();
    state->function = function; state->loop = loop; state->body = std::move(body); state->trips = trips;
    if (!state->buildBoundary() || !state->closePorts()) {
        result.error = state->error.empty() ? state->e().constructionError() : state->error;
        return result;
    }
    auto exported = exportRepeatedRegion(state);
    if (exported.error.empty()) { attachNumericalRepeatedExports(state, exported.regional); }
    return exported;
}
RepeatedRegionAnalysis exportRepeatedRegion(std::shared_ptr<RepeatedRegionState> state)
{
    RepeatedRegionAnalysis result;
    if (!state || !state->function || !state->loop || !state->body.expressions) {
        result.error = "repeated export requires a valid original loop and expression arena"; return result;
    }
    result.error = invalidRepeatedFrames(state->body);
    if (!result.error.empty()) { return result; }
    auto loop = state->loop;
    auto trips = state->trips;
    auto& out = result.regional;
    out = state->body;
    out.numerical.reset();
    out.prepare = {}; out.prepareFiltered = {}; out.prepareWithVisits = {};
    out.capabilities.endpointRecipes = false;
    out.firstOrdinal.reset();
    if (out.outerLoops.empty()) { out.outerLoops.resize(out.anchors.size()); }
    if (out.outerDivisors.empty()) {
        out.outerDivisors.resize(out.anchors.size());
        for (std::size_t type = 0; type < out.anchors.size(); ++type) {
            out.outerDivisors[type].assign(out.outerLoops[type].size(), 1);
        }
    }
    for (std::size_t type = 0; type < out.anchors.size(); ++type) {
        auto& frame = out.outerLoops[type];
        auto* payload = out.anchors[type].phase ? out.anchors[type].phase->elementOp : nullptr;
        if (!payload || !loop->isProperAncestor(payload) ||
            (!frame.empty() && !loop->isProperAncestor(frame.front())) ||
            (out.occurrenceLoops[type] && !loop->isProperAncestor(out.occurrenceLoops[type]))) {
            result.error = "repeated body coordinates must lie inside its original loop"; return result;
        }
        frame.insert(frame.begin(), loop);
        out.outerDivisors[type].insert(out.outerDivisors[type].begin(), 1);
    }
    auto& e = state->e();
    liftRepeatedSelectors(out, trips);
    out.presence = [state](RegionalEvent event) { return state->present(std::move(event)); };
    if (state->body.endpointEventGuard) {
        out.endpointEventGuard = [state](RegionalEvent event) -> std::optional<RegionExpressions::Id> {
            if (event.visits.empty()) { return std::nullopt; }
            event.visits.erase(event.visits.begin());
            return state->body.endpointEventGuard(std::move(event));
        };
    }
    out.reachability = [state](RegionalEvent a, RegionalEvent b) { return state->query(std::move(a), std::move(b)); };
    out.referenceBefore = [state](RegionalEvent a, RegionalEvent b) -> std::optional<RegionExpressions::Id> {
        if (a.visits.empty() || b.visits.empty()) { return std::nullopt; }
        auto i = a.visits.front(), j = b.visits.front();
        a.visits.erase(a.visits.begin()); b.visits.erase(b.visits.begin());
        auto inner = regionalReferenceBefore(state->body, a, b);
        if (!inner) { return std::nullopt; }
        auto& e = state->e();
        return e.lor(e.lt(i, j), e.land(e.eq(i, j), *inner));
    };
    out.cost.ports += state->slots.size();
    ++out.cost.repeatedRegions;
    out.cost.crossings += state->crossings.size();
    out.cost.expressionNodes = e.size();
    if (state->body.capabilities.endpointRecipes && (state->body.prepare || state->body.prepareWithVisits)) {
        out.prepareWithVisits = [state](ArrayRef<scf::ForOp> enclosing) { return state->prepare(enclosing); };
        out.prepare = [state]() { return state->prepare({}); };
        out.capabilities.endpointRecipes = true;
    }
    result.state = std::move(state);
    return result;
}
} // namespace mlir::pto::frontiersynch
