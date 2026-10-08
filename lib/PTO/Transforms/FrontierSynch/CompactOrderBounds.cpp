// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "llvm/ADT/DenseSet.h"
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
bool validEvent(const RegionExpressions& arena, std::size_t types, const RegionalEvent& event)
{
    return event.type < types && event.visits.empty() && event.ordinal < arena.size() &&
        !arena.isBoolean(event.ordinal) &&
        (event.kind == PeriodicEventKind::Start || event.kind == PeriodicEventKind::Completion);
}
std::optional<RegionalOrderView> exportOrder(const PeriodicExcessSnapshot& snapshot, std::string& error)
{
    if (!snapshot) { error = "compact order export has no immutable graph snapshot"; return std::nullopt; }
    RegionalOrderQueries queries;
    queries.reachability = [snapshot](RegionalEvent source, RegionalEvent target) -> std::optional<Expr> {
        auto context = snapshot->context();
        auto& arena = *context->expressions();
        if (!validRegionalEvent(context->domain(), source) || !validRegionalEvent(context->domain(), target)) {
            return std::nullopt;
        }
        auto first = regionalPresence(context->domain(), source), second = regionalPresence(context->domain(), target);
        if (!first || !second) { return std::nullopt; }
        auto threshold = snapshot->analysis().eventThreshold({source.type, source.kind}, {target.type, target.kind});
        if (threshold.error != PeriodicQueryError::None) { return std::nullopt; }
        if (!threshold.displacement) { return arena.boolean(false); }
        // The unsigned subtraction is interpreted only when source<=target.
        // With both endpoints present, a reference-forward periodic path stays
        // in this finite prefix; no omitted intermediate iteration is assumed.
        auto distance = arena.sub(target.ordinal, source.ordinal);
        auto ordered = arena.land(arena.le(source.ordinal, target.ordinal),
                                   arena.le(arena.constant(*threshold.displacement), distance));
        auto answer = arena.land(arena.land(*first, *second), ordered);
        return arena.constructionError().empty() ? std::optional<Expr>(answer) : std::nullopt;
    };
    auto view = makeRegionalOrderView(snapshot->context(), std::move(queries), error);
    return succeeded(view) ? std::optional<RegionalOrderView>(std::move(*view)) : std::nullopt;
}
} // namespace
CompactFixedBodyContext::CompactFixedBodyContext(OrderContext context, std::vector<PeriodicPayload> payloads,
                                                 Expr trips)
    : owner(std::move(context)), word(std::move(payloads)), count(trips) {}
std::shared_ptr<const CompactFixedBodyContext> captureCompactFixedBodyContext(
    scf::ForOp loop, const SyncInput& input, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> arena, Expr exactTrips, std::string& error)
{
    error.clear();
    if (!loop || !arena || !arena->constructionError().empty() || exactTrips >= arena->size() ||
        arena->isBoolean(exactTrips)) {
        error = "compact fixed context requires an original loop, valid arena and exact integer trip circuit";
        return {};
    }
    const auto contract = recognizeExplicit(*loop.getBody(), index, input.accesses());
    for (const auto& diagnostic : contract.diagnostics) {
        if (diagnostic.issue != RecognitionIssue::AdditionalPrerequisite) {
            error = "compact order export needs a fixed all-sites-present shared body";
            return {};
        }
    }
    auto phases = index.explicitSequence(*loop.getBody());
    if (failed(phases) || phases->size() > UINT32_MAX) {
        error = "compact order export has no fixed representable phase sequence";
        return {};
    }
    RegionalAnalysis domain;
    domain.expressions = arena;
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    domain.firstOrdinal = arena->constant(0);
    std::vector<PeriodicPayload> payloads;
    for (const auto* phase : *phases) {
        if (!phase || !phase->elementOp || phase->elementOp->getBlock() != loop.getBody()) {
            error = "compact order export phase is not in the original loop body";
            return {};
        }
        auto* operation = phase->elementOp;
        domain.anchors.push_back({phase, {}, {operation->getBlock(), operation},
                                  {operation->getBlock(), operation->getNextNode()}});
        domain.occurrenceLoops.push_back(loop);
        payloads.push_back({static_cast<uint32_t>(phase->kPipeValue)});
    }
    // Verify the supplied index did not omit any shared modeled phase.
    llvm::DenseSet<const CompoundInstanceElement*> members(phases->begin(), phases->end());
    for (const auto* phase : input.instructions()) {
        if (phase->elementOp->getBlock() == loop.getBody() && !members.contains(phase)) {
            error = "compact order export omitted an original shared phase";
            return {};
        }
    }
    const auto types = payloads.size();
    domain.presence = [arena, exactTrips, types](RegionalEvent event) -> std::optional<Expr> {
        if (!validEvent(*arena, types, event)) { return std::nullopt; }
        return arena->lt(event.ordinal, exactTrips);
    };
    domain.referenceBefore = [arena, types](RegionalEvent source, RegionalEvent target) -> std::optional<Expr> {
        if (!validEvent(*arena, types, source) || !validEvent(*arena, types, target)) { return std::nullopt; }
        return arena->lor(arena->lt(source.ordinal, target.ordinal),
            arena->land(arena->eq(source.ordinal, target.ordinal), arena->boolean(source.type < target.type)));
    };
    auto context = captureRegionalOrderContext(input, domain, error);
    if (failed(context)) { return {}; }
    return std::shared_ptr<const CompactFixedBodyContext>(
        new CompactFixedBodyContext(std::move(*context), std::move(payloads), exactTrips));
}
CompactOrderBounds buildCompactOrderBounds(std::shared_ptr<const CompactFixedBodyContext> domain,
    const CompactWriterReaderAnalysis& upper, const CompactLowerFacts& lower, uint64_t certificateTrips)
{
    CompactOrderBounds result;
    result.mathematical = std::make_shared<const CompactWriterReaderAnalysis>(upper);
    result.lowerFacts = std::make_shared<const CompactLowerFacts>(lower);
    result.domain = std::move(domain);
    if (!result.domain) {
        result.exportError = "compact mathematical records have no qualified fixed-body order context";
        return result;
    }
    result.bounds.context = result.domain->context();
    result.bounds.reduction = ReductionQuality::Covers;
    if (!upper.error.empty()) { result.exportError = upper.error; return result; }
    result.upperGraph = bindPeriodicExcessGraph(result.domain, upper.upper, ReductionQuality::Covers,
                                                result.exportError);
    if (!result.upperGraph) { return result; }
    result.bounds.upper = exportOrder(result.upperGraph, result.exportError);
    if (!result.bounds.upper) { return result; }
    if (!lower.error.empty()) { result.exportError = lower.error; return result; }
    // Preserve fixed native prerequisites in both graphs. The absence of
    // certain storage facts is a native lower graph, never an empty event order.
    auto lowerIndex = analyzePeriodicDemands(upper.upper.payloads, lower.records, upper.upper.nativePrerequisites);
    if (!lowerIndex.error.empty()) { result.exportError = lowerIndex.error; return result; }
    result.lowerGraph = bindPeriodicExcessGraph(result.domain, lowerIndex, ReductionQuality::Covers,
                                                result.exportError);
    if (!result.lowerGraph) { return result; }
    result.bounds.lower = exportOrder(result.lowerGraph, result.exportError);
    if (!result.bounds.lower) { return result; }
    result.certificate = certifyPeriodicExcess(result.upperGraph, result.lowerGraph, certificateTrips);
    if (result.certificate.error.empty()) { result.bounds.guarantee = result.certificate.guarantee; }
    else {
        // A contradicted lower/upper containment binding cannot publish valid
        // paired bounds. Keep its graph and records for diagnosis, and retain
        // the independently constructed upper query capability.
        result.exportError = result.certificate.error;
        result.bounds.lower.reset();
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
