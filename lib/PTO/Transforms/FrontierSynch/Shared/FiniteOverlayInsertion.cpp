// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve filtered child recipes and append finite overlay families before SSA.
#include "PTO/Transforms/FrontierSynch/Shared/FiniteOverlayInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "llvm/Support/SaveAndRestore.h"
#include "OccurrenceBinding.h"
#include "AllocationInputCache.h"
#include "PTO/Transforms/FrontierSynch/Shared/FiniteOverlayAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <map>
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteOverlayLogicalInsertion(
    func::FuncOp function, const FiniteOverlayAnalysis& analysis,
    std::string& error)
{
    const auto& base = analysis.base;
    if (!function || !analysis.error.empty() || !analysis.retainBase || !base.prepareFiltered ||
        !base.expressions || analysis.demands.size() != analysis.retained.size() ||
        base.anchors.size() != base.occurrenceLoops.size()) {
        error = "finite overlay lacks filtered base endpoint recipes"; return failure();
    }
    for (const auto& anchor : base.anchors) {
        if (!anchor.phase || !anchor.phase->elementOp ||
            anchor.phase->elementOp->getParentOfType<func::FuncOp>() != function ||
            !anchor.before.before || !anchor.after.before ||
            anchor.before.before->getParentOfType<func::FuncOp>() != function ||
            anchor.after.before->getParentOfType<func::FuncOp>() != function) {
            error = "finite overlay anchors belong to another function or invalid cuts"; return failure();
        }
    }
    auto supplied = base.prepareFiltered(analysis.retainBase);
    if (failed(supplied)) { error = "finite overlay base endpoint refinement failed"; return failure(); }
    auto plan = std::move(*supplied);
    auto baseView = takeRegionalAllocationView(*plan, true);
    if (!baseView || (plan->familyRecipe && !plan->independentPieces)) {
        error = "finite overlay base lacks composable matching metadata"; return failure();
    }
    plan->allocationCertificate = {};
    // Retain the typed base proof, then rebuild the certificate against augmented queries.
    auto& arena = *base.expressions;
    using Expr = RegionExpressions::Id;
    // Compatibility adapters for still-concrete legacy children only.
    std::map<Operation*, Block*> blocks;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    auto emit = [&](Expr value, Operation* cut) -> FailureOr<Value> {
        if (!cut) { return failure(); }
        auto found = blocks.find(cut);
        if (found == blocks.end()) { found = blocks.emplace(cut, &plan->addPreparation(cut)).first; }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(found->second);
        return arena.emitContextual(value, builder, cut, contexts[cut]);
    };
    uint64_t nextRecord = 0;
    for (const auto& family : plan->families) {
        nextRecord = std::max(nextRecord, uint64_t(family.id) + 1);
        for (const auto& member : family.members) { nextRecord = std::max(nextRecord, uint64_t(member.record) + 1); }
    }
    if (plan->families.empty() && !plan->endpoints.empty()) {
        error = "finite overlay base has no endpoint provenance"; return failure();
    }
    if (!plan->independentPieces) {
        for (auto& endpoint : plan->endpoints) {
            if (endpoint.record < 0 || static_cast<uint64_t>(endpoint.record) >= nextRecord) {
                error = "finite overlay base has invalid record identity"; return failure();
            }
            if (plan->groupedFamilies && !endpoint.memberCoordinates.empty()) {
                error = "finite overlay needs a filtered base with ungrouped or independent recipes"; return failure();
            }
            endpoint.records = {static_cast<uint32_t>(endpoint.record)};
            if (endpoint.kind != LogicalCommandKind::Barrier) {
                auto member = emit(arena.constant(endpoint.record), endpoint.before);
                if (failed(member)) { return failure(); }
                endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin(), *member);
            }
        }
    }
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    auto cutGuard = [&](RegionalEvent event, Expr retained) -> std::optional<Expr> {
        if (!validRegionalEvent(base, event)) { return std::nullopt; }
        auto indexLoop = [](scf::ForOp loop) { return !loop || loop.getInductionVar().getType().isIndex(); };
        if (!indexLoop(base.occurrenceLoops[event.type]) ||
            (!base.outerLoops.empty() && !llvm::all_of(base.outerLoops[event.type], indexLoop))) {
            return std::nullopt;
        }
        auto conditions = regionalOccurrenceConstraints(base, event, error);
        if (failed(conditions)) { return std::nullopt; }
        for (auto condition : *conditions) { retained = arena.land(retained, condition); }
        // A finite leaf has only ordinal zero. Keep the overlay's existing
        // validity check in addition to the shared original-frame constraints.
        if (!base.occurrenceLoops[event.type]) {
            retained = arena.land(retained, arena.eq(event.ordinal, arena.constant(0)));
        }
        return retained;
    };
    std::vector<ContextualEndpointFamily> endpoints;
    std::vector<std::optional<uint32_t>> overlayRecords(analysis.demands.size());
    for (std::size_t i = 0; i < analysis.demands.size(); ++i) {
        auto retained = analysis.retained[i];
        if (arena.constantValue(retained) == 0) { continue; }
        const auto& demand = analysis.demands[i];
        if (demand.source.type >= base.anchors.size() || demand.target.type >= base.anchors.size() ||
            nextRecord >= UINT32_MAX) { error = "finite overlay endpoint identity overflow"; return failure(); }
        const auto& a = base.anchors[demand.source.type]; const auto& b = base.anchors[demand.target.type];
        if (!a.phase || !b.phase) { error = "finite overlay has no payload anchors"; return failure(); }
        auto p = static_cast<uint32_t>(a.phase->kPipeValue), q = static_cast<uint32_t>(b.phase->kPipeValue);
        auto record = static_cast<uint32_t>(nextRecord++);
        overlayRecords[i] = record;
        EndpointFamily family;
        family.id = record; family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = a.after; family.targetCut = b.before;
        family.members.push_back({record, demand.source.type, demand.target.type, a.coordinates, b.coordinates});
        plan->families.push_back(std::move(family));
        for (bool publish : {true, false}) {
            if (publish && p == q) { continue; }
            auto* cut = publish ? a.after.before : b.before.before;
            auto expression = cutGuard(publish ? demand.source : demand.target, retained);
            if (!expression) { error = "finite overlay endpoint coordinate unsupported"; return failure(); }
            ContextualEndpointFamily endpoint;
            endpoint.piece.cut = {cut->getBlock(), cut};
            endpoint.piece.kind = p == q ? EndpointKind::Barrier : publish ? EndpointKind::Set : EndpointKind::Wait;
            endpoint.piece.namespaceId = record;
            endpoint.piece.sourcePipe = p; endpoint.piece.targetPipe = q;
            // Do not add exclusion evidence beyond this producer's incumbent
            // exact guard. The occurrence constraints remain its authority.
            endpoint.piece.members.push_back({record, {}});
            endpoint.activity = *expression; endpoint.ordinal = arena.constant(0);
            endpoints.push_back(std::move(endpoint));
        }
    }
    if (!endpoints.empty()) {
        auto recipe = buildContextualFamilyEndpointRecipe(base.expressions, endpoints);
        if (failed(recipe)) { error = "finite overlay endpoint guard unavailable"; return failure(); }
        auto joined = appendFamilyEndpointRecipe(plan->familyRecipe, *recipe, 0, plan->endpoints.size());
        if (failed(joined)) { error = "finite overlay recipe composition failed"; return failure(); }
        plan->familyRecipe = std::move(*joined);
    }
    const auto namespaces = familyRecordNamespaces(plan->families);
    int64_t piece = 0;
    for (auto& endpoint : plan->endpoints) {
        if (endpoint.records.empty() || !namespaces.count(endpoint.records.front())) {
            error = "finite overlay endpoint lacks record provenance"; return failure();
        }
        endpoint.record = namespaces.at(endpoint.records.front()); endpoint.piece = piece++;
    }
    if (!arena.constructionError().empty()) { error = arena.constructionError(); return failure(); }
    plan->allocationPreparation = AllocationPreparation::candidates(
        [owned = std::make_shared<FiniteOverlayAnalysis>(analysis), baseView,
         overlayRecords = std::move(overlayRecords), cache = AllocationInputCache{}](
             PreparedLogicalPlan& prepared, AllocationStrength strength) mutable {
        auto allocation = requestAllocationSummary(*baseView, strength, prepared.allocationProfile);
        return cache.request({allocation}, [&]() {
            // The overlay queries matching shape from recipes. Its temporary input
            // does not replace the incumbent selected on the enclosing plan.
            llvm::SaveAndRestore<std::shared_ptr<RegionalAllocationSummary>> input(
                prepared.regionalAllocation, allocation);
            return finiteOverlayAllocation(*owned, prepared, overlayRecords);
        });
    });
    return plan;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteOverlayInsertion(
    func::FuncOp function, const FiniteOverlayAnalysis& analysis, std::string& error)
{
    auto plan = prepareFiniteOverlayLogicalInsertion(function, analysis, error);
    if (succeeded(plan)) {
        prepareAllocationSupport(**plan);
        (*plan)->allocationCertificate = regionalAllocationCertificate(analysis.regional, **plan);
    }
    return plan;
}
RegionalAnalysis finiteOverlayRegionalResult(func::FuncOp function,
    const FiniteOverlayAnalysis& analysis)
{
    auto result = analysis.regional;
    if (!function || !analysis.error.empty() || !analysis.base.prepareFiltered || !analysis.retainBase) {
        return result;
    }
    auto owned = std::make_shared<FiniteOverlayAnalysis>(analysis);
    result.prepare = [owned, function]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        std::string error;
        auto plan = prepareFiniteOverlayLogicalInsertion(function, *owned, error);
        if (succeeded(plan)) { (*plan)->completeInvocation = false; }
        return plan;
    };
    result.capabilities.endpointRecipes = true;
    return result;
}
} // namespace mlir::pto::frontiersynch
