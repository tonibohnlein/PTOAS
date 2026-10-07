// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Closed-callee composition owns module-level dependency decisions. Recognition
// is linear in the call graph and IR size and never invokes a demand backend.
#include "PTO/Transforms/FrontierSynch/ClosedCallees.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/Transforms/FrontierSynch/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/Verifier.h"
#include "../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/SymbolTable.h"
namespace mlir::pto::frontiersynch {
namespace {
unsigned cores(func::FuncOp function)
{
    switch (recoverSyncPhysicalCore(function)) {
        case SyncPhysicalCore::AIC: return 1;
        case SyncPhysicalCore::AIV: return 2;
        case SyncPhysicalCore::Unknown: return 3;
        case SyncPhysicalCore::Conflict: return 0;
    }
    return 0;
}
bool wrapperBody(func::FuncOp function)
{
    if (function.isDeclaration() || function.getNumResults() || !function.getBody().hasOneBlock()) {
        return false;
    }
    bool calls = false;
    for (Operation& op : function.front()) {
        if (auto call = dyn_cast<func::CallOp>(op)) {
            if (call.getNumResults()) { return false; }
            calls = true;
        } else if (!isa<func::ReturnOp>(op) && !mlir::pto::detail::canReplayScalar(&op)) {
            return false;
        }
    }
    return calls;
}
}
ClosedCalleeModule recognizeClosedCallees(ModuleOp module)
{
    ClosedCalleeModule output;
    for (auto owner : module.getOps<func::FuncOp>()) {
        owner.walk([&](func::CallOp call) {
            if (auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr())) {
                if (function->getParentOp() == module) {
                    output.calledFunctions.insert(function);
                    output.callers[function].push_back(owner);
                }
            }
        });
    }
    for (auto function : module.getOps<func::FuncOp>()) {
        if (!wrapperBody(function)) { continue; }
        auto& candidate = output.wrappers[function];
        if (!cores(function)) { candidate.result.note(RecognitionIssue::UnsupportedControl, function); }
        for (auto call : function.front().getOps<func::CallOp>()) {
            auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            if (!callee || callee->getParentOp() != module || callee.isDeclaration() || callee.getNumResults() ||
                hasManualOnCoreSynchronization(callee)) {
                candidate.result.note(RecognitionIssue::UnmodeledOperation, call);
                continue;
            }
            candidate.callees.push_back(callee);
        }
    }
    // Reverse dependency counts avoid recursion and are independent of symbol
    // order. Duplicate calls remain duplicate edges for participation checking.
    DenseMap<Operation*, unsigned> pending;
    DenseMap<Operation*, SmallVector<Operation*>> parents;
    SmallVector<Operation*> ready;
    for (auto function : module.getOps<func::FuncOp>()) {
        auto found = output.wrappers.find(function);
        if (found == output.wrappers.end()) { continue; }
        unsigned count = 0;
        for (auto callee : found->second.callees) {
            if (output.wrappers.count(callee)) { ++count; parents[callee].push_back(function); }
        }
        pending[function] = count;
        if (!count) { ready.push_back(function); }
    }
    DenseSet<Operation*> processed;
    while (!ready.empty()) {
        auto* operation = ready.pop_back_val();
        processed.insert(operation);
        auto function = cast<func::FuncOp>(operation);
        auto& candidate = output.wrappers.find(operation)->second;
        unsigned occupied = 0;
        for (auto callee : candidate.callees) {
            unsigned active = cores(callee);
            if (auto nested = output.wrappers.find(callee); nested != output.wrappers.end()) {
                if (nested->second.result.state != RecognitionState::Applicable) {
                    candidate.result.note(RecognitionIssue::UnmodeledOperation, callee);
                }
                active = nested->second.activeCores;
            }
            if (!active) { candidate.result.note(RecognitionIssue::UnsupportedControl, callee); }
            active &= cores(function);
            if (occupied & active) {
                // Multiple communicating invocations on one core need a richer
                // interprocedural protocol proof, not a textual call-order edge.
                candidate.result.note(RecognitionIssue::UnsupportedControl, function, true);
            }
            occupied |= active;
        }
        candidate.activeCores = occupied;
        for (auto* parent : parents[operation]) {
            if (!--pending[parent]) { ready.push_back(parent); }
        }
    }
    for (auto& item : output.wrappers) {
        if (!processed.contains(item.first)) {
            item.second.result.note(RecognitionIssue::UnsupportedControl, item.first, true);
        }
    }
    return output;
}
namespace {
LogicalResult checkCalleeEvents(ModuleOp module, const ClosedCalleeModule& delegation,
                               DenseSet<Operation*> newEvents)
{
    // Calls inherit both newly inserted and pre-existing event use. Hidden
    // macro events are supplied by the shared model, not rediscovered here.
    auto usesEvents = newEvents;
    for (auto function : module.getOps<func::FuncOp>()) {
        auto hidden = function.walk([](Operation* op) {
            auto model = getSyncMacroModel(op);
            return model && !model->hiddenEvents.empty() ? WalkResult::interrupt() : WalkResult::advance();
        });
        if (hidden.wasInterrupted() || hasManualOnCoreSynchronization(function)) { usesEvents.insert(function); }
    }
    auto propagate = [&](DenseSet<Operation*>& events) {
        SmallVector<Operation*> work(events.begin(), events.end());
        while (!work.empty()) {
            auto* callee = work.pop_back_val();
            auto found = delegation.callers.find(callee);
            if (found == delegation.callers.end()) { continue; }
            for (auto* caller : found->second) {
                if (events.insert(caller).second) { work.push_back(caller); }
            }
        }
    };
    propagate(usesEvents);
    propagate(newEvents);
    // An ordinary caller may hold an event across a helper call. Independent
    // allocation then requires a composed lifetime contract. Manual callers
    // are unchanged, but their callees must not acquire new event use either.
    for (auto function : module.getOps<func::FuncOp>()) {
        if (function.isDeclaration() || delegation.wrappers.count(function)) { continue; }
        const bool manual = hasManualOnCoreSynchronization(function);
        auto invalid = function.walk([&](func::CallOp call) {
            auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            if (callee && (manual ? newEvents : usesEvents).contains(callee)) {
                call.emitError("ordinary helper call requires an interprocedural event-lifetime contract; "
                               "callee synchronization is not event-free");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (invalid.wasInterrupted()) { return failure(); }
    }
    return success();
}
} // namespace
LogicalResult insertModuleSynchronization(ModuleOp module, GMAliasPolicy policy, PrepareCallee prepare)
{
    // Module ownership is essential: a function pass cannot inspect callee
    // bodies concurrently with another function pass inserting synchronization.
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(module->clone()));
    for (Operation& operation : *copy->getBody()) {
        if (auto nested = dyn_cast<ModuleOp>(operation)) {
            if (failed(insertModuleSynchronization(nested, policy, prepare))) { return failure(); }
        } else if (!isa<func::FuncOp>(operation) && operation.getNumRegions()) {
            return operation.emitError("closed-callee analysis requires an explicit module symbol scope");
        }
    }
    auto delegation = recognizeClosedCallees(*copy);
    SmallVector<Operation*> excludedDelegations;
    for (const auto& item : delegation.wrappers) {
        if (item.second.result.state != RecognitionState::Applicable) { excludedDelegations.push_back(item.first); }
    }
    // Failure of this route is not failure of the program: for example, two
    // synchronous scalar helpers can still use the ordinary scalar analysis.
    for (auto* excluded : excludedDelegations) { delegation.wrappers.erase(excluded); }
    struct Pending {
        func::FuncOp function;
        std::unique_ptr<PreparedLogicalPlan> plan;
    };
    SmallVector<Pending> pending;
    DenseSet<Operation*> newEvents;
    SmallVector<func::FuncOp> contexts;
    for (auto function : copy->getOps<func::FuncOp>()) {
        if (function.isDeclaration() || hasManualOnCoreSynchronization(function) ||
            delegation.wrappers.count(function)) { continue; }
        if (hasPhysicalSections(function)) { contexts.push_back(function); continue; }
        const bool called = delegation.calledFunctions.contains(function);
        auto plan = prepare(function, called ? GMAliasPolicy::MayAlias : policy);
        if (failed(plan)) {
            if (llvm::is_contained(excludedDelegations, function.getOperation())) {
                function.emitError("closed-callee composition requires defined acyclic callees, "
                    "no manual synchronization, and at most one active invocation per physical core");
            }
            return failure();
        }
        // A successful whole-function producer accounts for every payload.
        // Closed calls additionally require its fixed completion interface.
        if (called) { (*plan)->completeInvocation = true; }
        if (llvm::any_of((*plan)->endpoints, [](const auto& endpoint) {
                return endpoint.kind != LogicalCommandKind::Barrier;
            })) { newEvents.insert(function); }
        pending.push_back({function, std::move(*plan)});
    }
    // Prepare every ordinary function before editing any body: scalar helper
    // inference must see the original callee effects, not inserted commands.
    for (auto function : contexts) {
        const bool called = delegation.calledFunctions.contains(function);
        if (failed(insertContextSynchronization(function, [&](func::FuncOp projected) {
                auto plan = prepare(projected, called ? GMAliasPolicy::MayAlias : policy);
                if (succeeded(plan) && called) { (*plan)->completeInvocation = true; }
                return plan;
            }))) { return failure(); }
        auto events = function.walk([](Operation* op) {
            return isa<LogicalSetOp, LogicalWaitOp>(op) ? WalkResult::interrupt() : WalkResult::advance();
        });
        if (events.wasInterrupted()) { newEvents.insert(function); }
    }
    if (failed(checkCalleeEvents(*copy, delegation, newEvents))) { return failure(); }
    for (auto& item : pending) {
        if (failed(insertLogicalSynchronization(item.function, *item.plan))) { return failure(); }
    }
    for (auto function : copy->getOps<func::FuncOp>()) {
        if (!delegation.wrappers.count(function)) { continue; }
        // The wrapper adds no ordering between cores and publishes no events.
        // Its invoked leaves are closed. Keep calls and participation guards
        // intact; allocation still runs on every leaf and can fail separately.
        PreparedLogicalPlan empty(0);
        ExplicitAnalysis noDemands;
        empty.allocationCertificate = explicitAllocationCertificate(noDemands, 0, function.getContext());
        if (failed(insertLogicalSynchronization(function, empty))) { return failure(); }
    }
    if (failed(verify(*copy))) { return failure(); }
    module.getBodyRegion().takeBody(copy->getBodyRegion());
    return success();
}
} // namespace mlir::pto::frontiersynch
