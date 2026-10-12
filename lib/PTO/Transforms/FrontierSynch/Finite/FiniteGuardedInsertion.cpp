// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <tuple>
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FiniteGuardedState::prepare(const RegionalDemandFilter& filter)
{
    insertionError.clear();
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    const bool unsupportedWidth = bits.isScalable() || bits.getFixedValue() != 64;
    if (unsupportedWidth) {
        insertionError = "finite guarded endpoint arithmetic requires a 64-bit index representation";
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    std::vector<RegionalAllocationGroup> allocationGroups;
    plan->groupedFamilies = true;
    plan->independentPieces = true;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    std::map<Operation*, Block*> blocks;
    auto emit = [&](Expr expression, Operation* cut) -> FailureOr<Value> {
        auto found = blocks.find(cut);
        if (found == blocks.end()) { found = blocks.emplace(cut,&plan->addPreparation(cut)).first; }
        OpBuilder builder(function.getContext());
        builder.setInsertionPointToEnd(found->second);
        return arena->emitContextual(expression,builder,cut,contexts[cut]);
    };
    for (auto demand : retained) {
        if (filter) {
            auto zero = arena->constant(0);
            auto condition = filter({demand.source, zero, PeriodicEventKind::Completion},
                                    {demand.target, zero, PeriodicEventKind::Start});
            if (!condition || !arena->isBoolean(*condition)) {
                insertionError = "finite guarded overlay filter has no exact predicate"; return failure();
            }
            demand.guard = both(demand.guard, *condition);
        }
        if (arena->constantValue(demand.guard) == 0) { continue; }
        auto p = pipe(demand.source), q = pipe(demand.target);
        if (plan->families.size() >= UINT32_MAX) {
            insertionError = "finite guarded record identity overflow"; return failure();
        }
        auto record = static_cast<uint32_t>(plan->families.size());
        const auto& a = anchors[demand.source]; const auto& b = anchors[demand.target];
        EndpointFamily family;
        family.id = record; family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = a.after; family.targetCut = b.before;
        family.members.push_back({record,demand.source,demand.target,{}, {}});
        plan->families.push_back(std::move(family));
        auto targetGuard = emit(demand.guard,b.before.before);
        if (failed(targetGuard)) {
            insertionError = "finite guarded WAIT/barrier predicate is unavailable: " + arena->lastEmissionError();
            return failure();
        }
        if (p == q) {
            PreparedLogicalEndpoint barrier{b.before.before,LogicalCommandKind::Barrier,p,q,record,*targetGuard,{}};
            barrier.records.push_back(record); barrier.piece = plan->endpoints.size();
            plan->endpoints.push_back(std::move(barrier));
            continue;
        }
        auto zero = arena->constant(0);
        auto publication = prepareGuardedPublication(function, *arena, a.after, b.before,
            demand.guard, demand.guard, emit, insertionError, *targetGuard);
        if (failed(publication)) { return failure(); }
        plan->families.back().targetChoices = publication->targetChoices;
        allocationGroups.push_back({p, q, 1, {{record, 0, 0,
            {demand.source, zero, PeriodicEventKind::Start},
            {demand.target, zero, PeriodicEventKind::Completion}, publication->publicationGuard}}, {},
            bool(publication->cleanupCut)});
        auto sourceIdentity = emit(arena->constant(0),a.after.before);
        auto targetIdentity = emit(arena->constant(0),b.before.before);
        auto sourceMember = emit(arena->constant(record),a.after.before);
        auto targetMember = emit(arena->constant(record),b.before.before);
        if (failed(sourceIdentity) || failed(targetIdentity) ||
            failed(sourceMember) || failed(targetMember)) {
            insertionError = "finite guarded SET predicate or identity is unavailable"; return failure();
        }
        PreparedLogicalEndpoint set{a.after.before,LogicalCommandKind::Set,p,q,record,
            publication->sourceGuard,*sourceIdentity};
        set.memberCoordinates.push_back(*sourceMember);
        set.records.push_back(record); set.piece = plan->endpoints.size();
        plan->endpoints.push_back(std::move(set));
        PreparedLogicalEndpoint wait{b.before.before,LogicalCommandKind::Wait,p,q,record,*targetGuard,*targetIdentity};
        wait.memberCoordinates.push_back(*targetMember);
        wait.records.push_back(record); wait.piece = plan->endpoints.size();
        plan->endpoints.push_back(std::move(wait));
        if (failed(appendPublicationCleanup(*publication, record, p, q, {zero},
            *arena, emit, *plan, insertionError))) { return failure(); }
    }
    // Stream namespaces are shared by pipe pair and displacement. Keep the
    // original demand identities in records and memberCoordinates.
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    for (const auto& family : plan->families) {
        namespaces.try_emplace({family.sourcePipe, family.targetPipe, family.displacement}, family.id);
    }
    for (auto& endpoint : plan->endpoints) {
        const auto& family = plan->families[endpoint.records.front()];
        endpoint.record = namespaces.at({family.sourcePipe, family.targetPipe, family.displacement});
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
