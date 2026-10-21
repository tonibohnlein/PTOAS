// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain finite guarded protocols as family recipes. Allocation consumes their
// fixed matching and terminal lifetimes; only the shared materializer emits SSA.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticProgram.h"
#include "llvm/ADT/DenseSet.h"
#include "../Shared/OccurrenceBinding.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
namespace {
FailureOr<std::vector<TemplateEndpointAnchor>> originalPoints(FiniteGuardedState& state,
    const ArithmeticProgram* program)
{
    std::vector<TemplateEndpointAnchor> result;
    if (!program) { return result; }
    result = state.anchors;
    if (!program->context.root || program->sites.size() != result.size() ||
        program->context.function != state.function) {
        state.insertionError = "expanded endpoints require the retained original occurrence map"; return failure();
    }
    for (auto [id, site] : llvm::enumerate(program->sites)) {
        auto* op = result[id].phase ? result[id].phase->elementOp : nullptr;
        if (!op || site.phase != result[id].phase || !site.loops.empty()) {
            state.insertionError = "expanded endpoints contain an unbound occurrence frame"; return failure();
        }
        llvm::DenseSet<Operation*> fixed;
        for (auto point : site.fixedCoordinates) {
            if (!point.loop || !point.loop->isProperAncestor(op) || point.induction < 0 ||
                !fixed.insert(point.loop).second) {
                state.insertionError = "expanded endpoint point has an invalid or duplicate original frame";
                return failure();
            }
            result[id].coordinates.push_back({point.loop, point.induction});
        }
        for (auto* parent = op->getParentOp(); parent && parent != state.function; parent = parent->getParentOp()) {
            if (!isa<scf::ForOp>(parent)) { continue; }
            const bool internal = program->context.roots.empty() ? program->context.root->isAncestor(parent) :
                llvm::any_of(program->context.roots, [&](Operation* root) { return root->isAncestor(parent); });
            if (internal && !fixed.count(parent)) {
                state.insertionError = "expanded endpoint is missing an internal original coordinate";
                return failure();
            }
        }
    }
    return result;
}
bool stableActivity(RegionExpressions& arena, RegionExpressions::Id activity,
    const TemplateEndpointAnchor& source, const TemplateEndpointAnchor& target)
{
    // A uniform record predicate must retain its meaning at both fixed visits.
    // Do not replay an intra-loop SSA evaluation as another visit's history.
    for (const auto& input : arena.referencedInputs(activity)) {
        if (!input.second) { return false; }
        for (const auto* anchor : {&source, &target}) {
            for (auto point : anchor->coordinates) {
                if (!point.loop.isDefinedOutsideOfLoop(input.second)) { return false; }
            }
        }
    }
    return true;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FiniteGuardedState::prepare(
    const RegionalDemandFilter& filter, const ArithmeticProgram* expanded)
{
    accumulateCost(preparationWork.requests, 1);
    insertionError.clear();
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        insertionError = "finite guarded endpoint arithmetic requires a 64-bit index representation";
        return failure();
    }
    auto selectedAnchors = originalPoints(*this, expanded);
    if (failed(selectedAnchors)) { return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    accumulateCost(preparationWork.inventories, 1);
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    std::vector<RegionalAllocationGroup> allocationGroups;
    std::vector<ContextualEndpointFamily> endpoints;
    ContextualFamilyExpressions expressions(arena);
    const auto zero = arena->constant(0);
    for (auto demand : retained) {
        accumulateCost(preparationWork.demandVisits, 1);
        if (filter) {
            accumulateCost(preparationWork.filterQueries, 1);
            auto condition = filter({demand.source, zero, PeriodicEventKind::Completion},
                                    {demand.target, zero, PeriodicEventKind::Start});
            if (!condition || !arena->isBoolean(*condition)) {
                insertionError = "finite guarded overlay filter has no exact predicate"; return failure();
            }
            demand.guard = both(demand.guard, *condition);
        }
        if (arena->constantValue(demand.guard) == 0) { continue; }
        if (plan->families.size() >= UINT32_MAX) {
            insertionError = "finite guarded record identity overflow"; return failure();
        }
        const auto record = static_cast<uint32_t>(plan->families.size());
        const auto p = pipe(demand.source), q = pipe(demand.target);
        const auto& a = expanded ? (*selectedAnchors)[demand.source] : anchors[demand.source];
        const auto& b = expanded ? (*selectedAnchors)[demand.target] : anchors[demand.target];
        if (expanded && (!stableActivity(*arena, demand.guard, a, b) ||
            !stableActivity(*arena, presence[demand.source], a, b) ||
            !stableActivity(*arena, presence[demand.target], a, b))) {
            insertionError = "expanded record activity requires unqualified intra-visit history";
            return failure();
        }
        EndpointFamily family;
        family.id = record; family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = a.after; family.targetCut = b.before;
        if (expanded) { family.sourceOrder = demand.source; family.targetOrder = demand.target; }
        family.members.push_back({record, demand.source, demand.target, a.coordinates, b.coordinates});
        auto sourcePoint = originalFixedPointGuard(*arena, a.coordinates, a.after.before, insertionError);
        auto targetPoint = originalFixedPointGuard(*arena, b.coordinates, b.before.before, insertionError);
        if (failed(sourcePoint) || failed(targetPoint)) { return failure(); }
        const auto sourceGuard = expanded ? both(demand.guard, *sourcePoint) : demand.guard;
        const auto consumerGuard = expanded ? both(demand.guard, *targetPoint) : demand.guard;
        const auto sourcePresent = expanded ? both(presence[demand.source], *sourcePoint) : demand.guard;
        const auto selectedSource = expanded ? std::optional<SelectedPublication>(
            SelectedPublication{presence[demand.source], sourcePresent}) : std::nullopt;
        accumulateCost(preparationWork.publicationRequests, 1);
        auto direct = prepareDirectPublicationFamily(function, expressions, std::move(family),
            sourceGuard, demand.guard, consumerGuard, insertionError, selectedSource);
        if (failed(direct)) { return failure(); }
        if (p != q) {
            allocationGroups.push_back({p, q, 1, {{record, 0, 0,
                {demand.source, zero, PeriodicEventKind::Start},
                {demand.target, zero, PeriodicEventKind::Completion}, direct->publicationGuard}}, {},
                direct->terminalLifetime});
        }
        for (auto& endpoint : direct->endpoints) { endpoints.push_back(std::move(endpoint)); }
        plan->families.push_back(std::move(direct->family));
    }
    // Record provenance is retained while stream namespaces remain shared by
    // direction/displacement, exactly as in the concrete endpoint path.
    const auto namespaces = familyRecordNamespaces(plan->families);
    for (auto& endpoint : endpoints) {
        endpoint.piece.namespaceId = namespaces.at(endpoint.piece.members.front().record);
    }
    if (!endpoints.empty()) {
        accumulateCost(preparationWork.recipeBuilds, 1);
        auto recipe = buildContextualFamilyEndpointRecipe(expressions, endpoints);
        if (failed(recipe)) {
            insertionError = "finite guarded family predicate or identity unavailable: " + arena->lastEmissionError();
            return failure();
        }
        plan->familyRecipe = std::move(*recipe);
    }
    plan->allocationPreparation = [owner = arena, groups = std::move(allocationGroups)](
        PreparedLogicalPlan& prepared) {
        auto summary = std::make_shared<RegionalAllocationSummary>();
        summary->groups = groups;
        prepared.regionalAllocation = std::move(summary);
    };
    return plan;
}
} // namespace mlir::pto::frontiersynch
