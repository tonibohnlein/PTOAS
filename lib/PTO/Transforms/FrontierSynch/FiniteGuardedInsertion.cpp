// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FiniteGuardedInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FiniteGuardedState::prepare(const RegionalDemandFilter& filter)
{
    insertionError.clear();
    bool terminalCleanup = false;
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->regionalAllocation = std::make_shared<RegionalAllocationSummary>();
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
        plan->regionalAllocation->groups.push_back({p, q, 1, {{record, 0, 0,
            {demand.source, zero, PeriodicEventKind::Start},
            {demand.target, zero, PeriodicEventKind::Completion}, demand.guard}}});
        auto sourceGuard = emit(demand.guard,a.after.before);
        std::optional<Expr> cleanupGuard;
        Operation* terminal = function.front().getTerminator();
        if (failed(sourceGuard)) {
            // Publish once when the source executes. Exactly one of the
            // consumer predicate and its source-qualified complement consumes
            // the notification. Never put cleanup at the conditional join.
            SmallVector<TemplateEndpointCut> alternatives{
                b.before, {terminal->getBlock(), terminal}};
            auto choices = qualifyTerminalCleanupCuts(alternatives);
            bool finiteSource = true;
            for (auto* parent = a.after.before->getParentOp(); parent != function; parent = parent->getParentOp()) {
                if (!isa<scf::IfOp>(parent)) { finiteSource = false; break; }
            }
            if (!choices || !finiteSource) {
                insertionError = "late SET guard needs a finite terminal cleanup interface"; return failure();
            }
            sourceGuard = emit(presence[demand.source], a.after.before);
            cleanupGuard = both(presence[demand.source], negate(demand.guard));
            plan->families.back().targetChoices = std::move(choices);
            terminalCleanup = true;
        }
        auto sourceIdentity = emit(arena->constant(0),a.after.before);
        auto targetIdentity = emit(arena->constant(0),b.before.before);
        auto sourceMember = emit(arena->constant(record),a.after.before);
        auto targetMember = emit(arena->constant(record),b.before.before);
        if (failed(sourceGuard) || failed(sourceIdentity) || failed(targetIdentity) ||
            failed(sourceMember) || failed(targetMember)) {
            insertionError = "finite guarded SET predicate or identity is unavailable"; return failure();
        }
        PreparedLogicalEndpoint set{a.after.before,LogicalCommandKind::Set,p,q,record,*sourceGuard,*sourceIdentity};
        set.memberCoordinates.push_back(*sourceMember);
        set.records.push_back(record); set.piece = plan->endpoints.size();
        plan->endpoints.push_back(std::move(set));
        PreparedLogicalEndpoint wait{b.before.before,LogicalCommandKind::Wait,p,q,record,*targetGuard,*targetIdentity};
        wait.memberCoordinates.push_back(*targetMember);
        wait.records.push_back(record); wait.piece = plan->endpoints.size();
        plan->endpoints.push_back(std::move(wait));
        if (cleanupGuard) {
            auto guard = emit(*cleanupGuard, terminal);
            auto identity = emit(arena->constant(0), terminal);
            auto member = emit(arena->constant(record), terminal);
            if (failed(guard) || failed(identity) || failed(member)) {
                insertionError = "terminal cleanup predicate is unavailable: " + arena->lastEmissionError();
                return failure();
            }
            PreparedLogicalEndpoint cleanup{terminal, LogicalCommandKind::Wait, p, q, record, *guard, *identity};
            cleanup.memberCoordinates.push_back(*member);
            cleanup.records.push_back(record); cleanup.piece = plan->endpoints.size();
            plan->endpoints.push_back(std::move(cleanup));
        }
    }
    // The ordinary envelope ends at the consumer and would allow premature
    // reuse. Finite allocation recognizes the terminal alternative explicitly.
    if (terminalCleanup) { plan->regionalAllocation.reset(); }
    return plan;
}
} // namespace mlir::pto::frontiersynch
