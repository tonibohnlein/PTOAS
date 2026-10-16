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
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FiniteGuardedState::prepare(const RegionalDemandFilter& filter)
{
    insertionError.clear();
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        insertionError = "finite guarded endpoint arithmetic requires a 64-bit index representation";
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    std::vector<RegionalAllocationGroup> allocationGroups;
    std::vector<ContextualEndpointFamily> endpoints;
    ContextualFamilyExpressions expressions(arena);
    const auto zero = arena->constant(0);
    for (auto demand : retained) {
        if (filter) {
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
        const auto& a = anchors[demand.source];
        const auto& b = anchors[demand.target];
        EndpointFamily family;
        family.id = record; family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = a.after; family.targetCut = b.before;
        family.members.push_back({record, demand.source, demand.target, {}, {}});
        auto direct = prepareDirectPublicationFamily(function, expressions, std::move(family),
            demand.guard, demand.guard, demand.guard, insertionError);
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
