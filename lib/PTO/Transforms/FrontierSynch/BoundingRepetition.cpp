// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BoundingRepetition.h"
#include "RepeatedRegionInternal.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using EventKey = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
void clearStorage(RegionalAnalysis& region)
{
    region.storageBoundary.clear(); region.accessBoundary.clear(); region.deferredAccessBoundary.clear();
    region.storageSelectors = {}; region.symbolicStorageEffects.clear();
}
std::optional<RegionalAnalysis> phaseExport(const BoundingSequenceChild& phase, bool upper, std::string& error)
{
    const auto& view = upper ? phase.bounds.upper : phase.bounds.lower;
    const auto& supplied = upper ? phase.upperExports : phase.lowerExports;
    if (!view || !supplied) {
        error = "repeated selected query or sign-specific boundary export unavailable"; return {};
    }
    const auto& canonical = view->regional();
    if (canonical.expressions != supplied->expressions || canonical.accessModel != supplied->accessModel ||
        canonical.gmAliasPolicy != supplied->gmAliasPolicy || canonical.anchors.size() != supplied->anchors.size() ||
        canonical.occurrenceLoops != supplied->occurrenceLoops || canonical.outerLoops != supplied->outerLoops ||
        canonical.outerDivisors != supplied->outerDivisors) {
        error = "repeated selected sidecar has a different occurrence frame"; return {};
    }
    for (std::size_t i = 0; i < canonical.anchors.size(); ++i) {
        if (!canonical.anchors[i].phase || canonical.anchors[i].phase != supplied->anchors[i].phase) {
            error = "repeated selected pipe/cut binding unavailable"; return {};
        }
    }
    auto result = *supplied;
    result.anchors = canonical.anchors; result.presence = canonical.presence;
    result.referenceBefore = canonical.referenceBefore; result.firstOrdinal = canonical.firstOrdinal;
    result.endpointSiteGuard = canonical.endpointSiteGuard;
    result.endpointInvocationGuard = canonical.endpointInvocationGuard;
    result.endpointEventGuard = canonical.endpointEventGuard;
    result.reachability = canonical.reachability; result.numerical = canonical.numerical;
    result.capabilities.exactQueries = true;
    return result;
}
BoundingRepetitionSign baseSign(func::FuncOp function, scf::ForOp loop,
                               const BoundingRepetitionInput& specification, bool upper)
{
    BoundingRepetitionSign result;
    const auto& selected = upper ? specification.upper : specification.lower;
    if (selected.bridges != BoundingSequenceBridges::PhysicalSelectors &&
        selected.bridges != BoundingSequenceBridges::SuppliedCrossings) {
        result.exportError = "invalid repeated storage bridge mode"; return result;
    }
    std::vector<RegionalAnalysis> phases;
    for (const auto& phase : specification.phases) {
        auto region = phaseExport(phase, upper, result.exportError);
        if (!region) { return result; }
        if (selected.bridges == BoundingSequenceBridges::SuppliedCrossings) {
            clearStorage(*region);
            region->capabilities.completeStorageModel = true;
            region->capabilities.exactSelectors = true;
        }
        phases.push_back(std::move(*region));
    }
    auto repeated = repeatPhasedRegions(function, loop, std::move(phases), specification.trips,
        specification.enclosing, specification.begin, specification.maximumLength);
    if (!repeated.error.empty()) { result.exportError = repeated.error; return result; }
    result.repeated = std::move(repeated);
    return result;
}
bool appendCrossings(BoundingRepetitionSign& result, const BoundingRepetitionSpecification& specification)
{
    if (!result.repeated) { return false; }
    auto& state = *result.repeated->state;
    auto& arena = state.e();
    for (const auto& supplied : specification.crossings) {
        auto edge = supplied.edge;
        if (!edge.displacement || !validRegionalEvent(state.body, edge.source) ||
            !validRegionalEvent(state.body, edge.target) || edge.guard >= arena.size() ||
            !arena.isBoolean(edge.guard) || (!edge.native &&
            (edge.source.kind != PeriodicEventKind::Completion || edge.target.kind != PeriodicEventKind::Start))) {
            result.exportError = "invalid positive-distance repeated crossing"; result.repeated.reset(); return false;
        }
        auto source = regionalPresence(state.body, edge.source), target = regionalPresence(state.body, edge.target);
        if (!source || !target) {
            result.exportError = "repeated crossing endpoint presence unavailable";
            result.repeated.reset(); return false;
        }
        edge.guard = arena.land(edge.guard, arena.land(*source, *target));
        state.crossings.push_back(edge); state.queryCrossings.push_back(edge);
    }
    return true;
}
std::vector<RegionalEvent> commonPorts(const BoundingRepetitionSign& lower, const BoundingRepetitionSign& upper)
{
    std::map<EventKey, RegionalEvent> events;
    for (const auto* sign : {&lower, &upper}) {
        if (!sign->repeated) { continue; }
        // Keep certificate-only ports from the original unreduced crossings,
        // including early releases that a later reduction might bypass.
        for (const auto* records : {&sign->repeated->state->queryCrossings, &sign->repeated->state->crossings}) {
            for (const auto& edge : *records) {
                for (const auto& event : {edge.source, edge.target}) {
                    events.emplace(EventKey{event.type, event.ordinal, event.kind, event.visits}, event);
                }
            }
        }
    }
    std::vector<RegionalEvent> result;
    for (const auto& item : events) { result.push_back(item.second); }
    return result;
}
bool addCost(BoundingRepetitionSign& result, const BoundingRepetitionCost& cost, bool deletion)
{
    auto& total = result.constructionCost;
    if (cost.bodyQueries > UINT64_MAX-total.bodyQueries || cost.relaxations > UINT64_MAX-total.relaxations ||
        (deletion && total.deletionTests == UINT64_MAX)) {
        result.exportError = "weighted repeated construction counter overflow";
        result.repeated.reset(); return false;
    }
    total.bodyQueries += cost.bodyQueries; total.relaxations += cost.relaxations;
    total.deletionTests += deletion;
    return true;
}
bool selectCrossings(BoundingRepetitionSign& result, const BoundingRepetitionSpecification& specification,
                     const std::vector<RegionalEvent>& ports)
{
    if (!result.repeated) { return false; }
    auto& state = *result.repeated->state;
    auto records = state.crossings;
    auto& arena = state.e();
    uint64_t deletions = 0;
    for (std::size_t candidate = 0; candidate < records.size(); ++candidate) {
        if (records[candidate].native || arena.constantValue(records[candidate].guard) == 0) { continue; }
        // Remove the whole edge template from the alternative graph. Sequential
        // guard refinement preserves closure even when distinct descriptions
        // denote one event under a parameter valuation (e.g. first==last at K=1).
        auto alternatives = records;
        alternatives[candidate].guard = arena.boolean(false);
        auto query = buildBoundingRepeatedQuery(state.body, ports, std::move(alternatives), result.exportError);
        if (!query) { result.repeated.reset(); return false; }
        auto implied = query->across(records[candidate].source, records[candidate].target,
                                      arena.constant(records[candidate].displacement));
        if (!implied) {
            result.exportError = "weighted repeated alternative-path query unavailable";
            result.repeated.reset(); return false;
        }
        ++deletions;
        if (!addCost(result, query->cost(), true)) { return false; }
        records[candidate].guard = arena.land(records[candidate].guard, arena.lnot(*implied));
    }
    result.query = buildBoundingRepeatedQuery(state.body, ports, records, result.exportError);
    if (!result.query) { result.repeated.reset(); return false; }
    if (!addCost(result, result.query->cost(), false)) { result.query.reset(); return false; }
    state.crossings = records; state.queryCrossings = std::move(records);
    state.queryMemo.clear(); state.bodyPortMemo.clear(); state.sourceDistances.clear(); state.distanceMemo.clear();
    state.weightedAcross = [query = result.query](const RegionalEvent& a, const RegionalEvent& b, Id gap) {
        return query->across(a, b, gap);
    };
    auto& out = result.repeated->regional;
    out.numerical.reset();
    out.cost.ports += ports.size(); out.cost.implicationChecks += deletions;
    out.cost.expressionNodes = arena.size();
    if (specification.bridges == BoundingSequenceBridges::SuppliedCrossings) {
        clearStorage(out); out.capabilities.completeStorageModel = false; out.capabilities.exactSelectors = false;
    }
    return true;
}
bool validInput(const SyncInput& input, const BoundingRepetitionInput& specification, std::string& error)
{
    if (specification.phases.empty() || specification.phases.size() > maxRegionalSlotVisits ||
        (specification.guarantee != InputOrderGuarantee::InputOrderEquivalent &&
         specification.guarantee != InputOrderGuarantee::InputOrderCovering)) {
        error = "invalid bounding repetition specification"; return false;
    }
    OrderContext common;
    for (const auto& phase : specification.phases) {
        if (failed(validateBoundingRegionalResult(phase.bounds, error))) { return false; }
        const auto& context = phase.bounds.context;
        if (&context->input() != &input || (common && (common->expressions() != context->expressions() ||
            common->accessModel() != context->accessModel() || common->gmAliasPolicy() != context->gmAliasPolicy()))) {
            error = "bounding repeated phases have different shared input contexts"; return false;
        }
        common = context;
    }
    return true;
}
} // namespace
BoundingRepetitionResult repeatBoundingRegion(func::FuncOp function, scf::ForOp loop,
    const SyncInput& input, BoundingRepetitionInput specification)
{
    BoundingRepetitionResult result;
    result.original = std::make_shared<const BoundingRepetitionInput>(std::move(specification));
    const auto& original = *result.original;
    if (!validInput(input, original, result.error)) { return result; }
    result.lower = baseSign(function, loop, original, false);
    result.upper = baseSign(function, loop, original, true);
    appendCrossings(result.lower, original.lower); appendCrossings(result.upper, original.upper);
    const auto ports = commonPorts(result.lower, result.upper);
    selectCrossings(result.lower, original.lower, ports); selectCrossings(result.upper, original.upper, ports);
    auto* domain = result.upper.repeated ? &result.upper.repeated->regional :
        (result.lower.repeated ? &result.lower.repeated->regional : nullptr);
    if (!domain) { return result; }
    auto context = captureRegionalOrderContext(input, *domain, result.error);
    if (failed(context)) { return result; }
    result.bounds.context = *context; result.bounds.guarantee = original.guarantee;
    result.bounds.reduction = ReductionQuality::Partial;
    result.placementMayStrengthen = false;
    for (const auto& phase : original.phases) {
        result.placementMayStrengthen |= phase.placementMayStrengthen;
        if (phase.bounds.guarantee != InputOrderGuarantee::InputOrderEquivalent) {
            result.bounds.guarantee = InputOrderGuarantee::InputOrderCovering;
        }
        if (phase.bounds.reduction == ReductionQuality::Generators) {
            result.bounds.reduction = ReductionQuality::Generators;
        }
    }
    auto exportView = [&](BoundingRepetitionSign& sign, std::optional<RegionalOrderView>& destination) {
        if (!sign.repeated) { return; }
        const auto& region = sign.repeated->regional;
        auto view = makeRegionalOrderView(*context, {region.reachability, {}}, sign.exportError);
        if (succeeded(view)) { destination = std::move(*view); }
    };
    exportView(result.lower, result.bounds.lower); exportView(result.upper, result.bounds.upper);
    if (result.upper.repeated) {
        const auto& state = *result.upper.repeated->state;
        for (const auto& edge : state.crossings) {
            if (!edge.native && state.body.expressions->constantValue(edge.guard) != 0 &&
                state.body.anchors[edge.source.type].phase->kPipeValue ==
                    state.body.anchors[edge.target.type].phase->kPipeValue) { result.placementMayStrengthen = true; }
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
