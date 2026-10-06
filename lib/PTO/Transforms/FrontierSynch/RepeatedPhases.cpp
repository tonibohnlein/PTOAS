// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "RepeatedRegionInternal.h"
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Selectors = std::vector<RegionalSelector>;
Id any(RegionExpressions& e, const Selectors& values)
{
    auto result = e.boolean(false);
    for (const auto& value : values) { result = e.lor(result, value.present); }
    return result;
}
void restrictSelectors(RegionExpressions& e, Selectors& values, Id guard)
{
    for (auto& value : values) { value.present = e.land(value.present, guard); }
}
void appendFirst(RegionExpressions& e, Selectors& first, Selectors next, Id guard)
{
    restrictSelectors(e, next, e.land(guard, e.lnot(any(e, first))));
    first.insert(first.end(), next.begin(), next.end());
}
void appendLast(RegionExpressions& e, Selectors& last, const Selectors& next, Id guard)
{
    restrictSelectors(e, last, e.land(guard, e.lnot(any(e, next))));
    last.insert(last.end(), next.begin(), next.end());
}
void mergeCell(RegionExpressions& e, RegionalStorageBoundary& left, const RegionalStorageBoundary& right)
{
    auto noLeftWriter = e.lnot(any(e, left.firstWriters));
    auto noRightWriter = e.lnot(any(e, right.firstWriters));
    for (const auto& [pipe, values] : right.firstReaders) {
        appendFirst(e, left.firstReaders[pipe], values, noLeftWriter);
    }
    for (auto& [pipe, values] : left.lastReaders) { restrictSelectors(e, values, noRightWriter); }
    for (const auto& [pipe, values] : right.lastReaders) {
        appendLast(e, left.lastReaders[pipe], values, e.boolean(true));
    }
    appendFirst(e, left.firstWriters, right.firstWriters, e.boolean(true));
    appendLast(e, left.lastWriters, right.lastWriters, e.boolean(true));
}
void liftSelectors(RegionExpressions& e, Selectors& values, uint32_t offset, Id period, Id guard)
{
    for (auto& value : values) {
        value.event.type += offset;
        value.event.visits.insert(value.event.visits.begin(), period);
        value.present = e.land(value.present, guard);
    }
}
RegionalStorageBoundary liftCell(RegionExpressions& e, RegionalStorageBoundary cell,
    uint32_t offset, Id period, Id guard)
{
    liftSelectors(e, cell.firstWriters, offset, period, guard);
    liftSelectors(e, cell.lastWriters, offset, period, guard);
    for (auto& [pipe, values] : cell.firstReaders) { liftSelectors(e, values, offset, period, guard); }
    for (auto& [pipe, values] : cell.lastReaders) { liftSelectors(e, values, offset, period, guard); }
    return cell;
}
void appendPartial(RegionalAnalysis& out, const std::vector<RegionalAnalysis>& phases,
    const std::vector<uint32_t>& starts, Id periods, Id remainder)
{
    auto& e = *out.expressions;
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
        auto enabled = e.lt(e.constant(phase), remainder);
        for (auto& destination : out.storageBoundary) {
            for (const auto& source : phases[phase].storageBoundary) {
                if (!sameStorageDomain(destination.cell, source.cell) ||
                    source.cell.begin > destination.cell.begin || source.cell.end < destination.cell.end) { continue; }
                mergeCell(e, destination, liftCell(e, source, starts[phase], periods, enabled));
            }
        }
        for (const auto& [pipe, source] : phases[phase].firstPayloads) {
            auto values = source;
            liftSelectors(e, values, starts[phase], periods, enabled);
            appendFirst(e, out.firstPayloads[pipe], std::move(values), e.boolean(true));
        }
        for (const auto& [pipe, source] : phases[phase].lastPayloads) {
            auto values = source;
            liftSelectors(e, values, starts[phase], periods, enabled);
            appendLast(e, out.lastPayloads[pipe], values, e.boolean(true));
        }
    }
}
} // namespace
RepeatedRegionAnalysis repeatPhasedRegions(func::FuncOp function, scf::ForOp loop,
    std::vector<RegionalAnalysis> phases, Id trips, ArrayRef<scf::ForOp> enclosing)
{
    RepeatedRegionAnalysis failure;
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function || phases.empty() ||
        phases.size() > maxRegionalSlotVisits || !phases.front().expressions) {
        failure.error = "repeated region requires a valid loop and supported explicit phase count"; return failure;
    }
    auto arena = phases.front().expressions;
    auto& e = *arena;
    if (trips >= e.size() || e.isBoolean(trips) || !e.constructionError().empty()) {
        failure.error = "repeated phases require a valid integer trip expression"; return failure;
    }
    auto lower = sequenceInteger(loop.getLowerBound()), step = sequenceInteger(loop.getStep());
    if (!lower || *lower < 0 || !step || *step <= 0) {
        failure.error = "phase endpoint binding requires a positive constant index step"; return failure;
    }
    const auto q = phases.size();
    auto ordinal = e.div(e.sub(e.input(loop.getInductionVar()), e.constant(*lower)), e.constant(*step));
    auto period = e.div(ordinal, e.constant(q)), phaseValue = e.rem(ordinal, e.constant(q));
    auto periods = e.div(trips, e.constant(q)), remainder = e.rem(trips, e.constant(q));
    std::vector<uint32_t> starts, typePhases;
    for (std::size_t phase = 0; phase < q; ++phase) {
        if (phases[phase].expressions != arena || phases[phase].anchors.size() > UINT32_MAX - typePhases.size()) {
            failure.error = "phase views require a common arena and bounded type identities"; return failure;
        }
        starts.push_back(static_cast<uint32_t>(typePhases.size()));
        typePhases.insert(typePhases.end(), phases[phase].anchors.size(), static_cast<uint32_t>(phase));
        phases[phase].endpointSiteGuard = e.eq(phaseValue, e.constant(phase));
        phases[phase].endpointInvocationGuard = e.lor(e.lt(period, periods),
            e.land(e.eq(period, periods), e.lt(e.constant(phase), remainder)));
    }
    bool endpoints = llvm::all_of(phases, [](const RegionalAnalysis& phase) {
        return phase.capabilities.endpointRecipes && (phase.prepare || phase.prepareWithVisits);
    });
    auto composed = composeRegionalSequence(function, arena, phases, false, false);
    if (!composed.error.empty()) { failure.error = composed.error; return failure; }
    composed.state->completeInvocation = false;
    composed.state->requiredOuterLoops.assign(enclosing.begin(), enclosing.end());
    composed.state->requiredOuterLoops.push_back(loop);
    auto body = sequenceRegionalResult(composed);
    if (!endpoints) {
        body.prepare = {}; body.prepareWithVisits = {}; body.prepareFiltered = {};
        body.capabilities.endpointRecipes = false;
    }
    auto count = e.add(periods, e.select(e.lt(e.constant(0), remainder), e.constant(1), e.constant(0)));
    auto result = repeatInvariantRegion(function, loop, body, count);
    if (!result.error.empty()) { return result; }
    result.state->phaseCount = q;
    result.state->originalTrips = trips;
    result.state->typePhases = typePhases;
    auto full = body;
    liftRepeatedSelectors(full, periods);
    auto& out = result.regional;
    for (auto& divisors : out.outerDivisors) { divisors.front() = q; }
    auto oldSite = out.endpointEventGuard;
    out.endpointEventGuard = [oldSite, arena, phaseValue, typePhases](RegionalEvent event) -> std::optional<Id> {
        if (event.type >= typePhases.size()) { return std::nullopt; }
        auto other = oldSite ? oldSite(event) : std::optional<Id>(arena->boolean(true));
        if (!other) { return std::nullopt; }
        return arena->land(*other, arena->eq(phaseValue, arena->constant(typePhases[event.type])));
    };
    out.storageBoundary = std::move(full.storageBoundary);
    out.firstPayloads = std::move(full.firstPayloads);
    out.lastPayloads = std::move(full.lastPayloads);
    appendPartial(out, phases, starts, periods, remainder);
    auto originalPresence = out.presence;
    auto originalQuery = out.reachability;
    auto actual = [arena, periods, remainder, typePhases](RegionalEvent event) -> std::optional<Id> {
        if (event.type >= typePhases.size() || event.visits.empty()) { return std::nullopt; }
        auto& e = *arena;
        auto n = event.visits.front();
        return e.lor(e.lt(n, periods), e.land(e.eq(n, periods), e.lt(e.constant(typePhases[event.type]), remainder)));
    };
    out.presence = [originalPresence, actual, arena](RegionalEvent event) -> std::optional<Id> {
        auto p = originalPresence(event), a = actual(event);
        if (!p || !a) { return std::nullopt; }
        return arena->land(*p, *a);
    };
    out.reachability = [originalQuery, actual, arena](RegionalEvent a, RegionalEvent b) -> std::optional<Id> {
        auto value = originalQuery(a, b), pa = actual(a), pb = actual(b);
        if (!value || !pa || !pb) { return std::nullopt; }
        return arena->land(*value, arena->land(*pa, *pb));
    };
    for (auto* accesses : {&out.accessBoundary, &out.deferredAccessBoundary}) {
        for (auto& access : *accesses) {
            const auto phase = typePhases[access.last.event.type];
            auto exists = e.lt(e.constant(phase), trips);
            access.first.present = e.land(access.first.present, exists);
            access.last.present = e.land(access.last.present, exists);
            access.last.event.visits.front() = e.select(e.lt(e.constant(phase), remainder), periods,
                e.sub(periods, e.constant(1)));
        }
    }
    out.cost.phaseDescriptions += q;
    out.cost.expressionNodes = e.size();
    return result;
}
} // namespace mlir::pto::frontiersynch
