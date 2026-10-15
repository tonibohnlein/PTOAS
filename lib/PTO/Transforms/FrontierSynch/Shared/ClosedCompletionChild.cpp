// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve selectively synchronized children through a closed matching and
// coarse completion contract, without importing their reachability selectors.
// Children without local events may retain pending effects. Their resources
// are empty or owned by closed descendants, independently of payload completion.
#include "ClosedCompletionChild.h"
#include "ProtocolSeal.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/FrontierSynch/Shared/ExecutionContexts.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
namespace mlir::pto::frontiersynch {
namespace {
constexpr llvm::StringLiteral closedAttr = "pto.closed_completion";
bool completed(func::FuncOp function)
{
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) {
        return false;
    }
    auto terminal = dyn_cast<func::ReturnOp>(function.front().getTerminator());
    auto all = terminal ? dyn_cast_or_null<BarrierOp>(terminal->getPrevNode()) : BarrierOp{};
    return all && all.getPipe().getPipe() == PIPE::PIPE_ALL;
}
bool callable(func::FuncOp function, std::shared_ptr<SyncInput>* source = nullptr, bool closedCalls = false)
{
    if (!function || function.isDeclaration() || function.getNumResults() || !function.getBody().hasOneBlock() ||
        !isa<ModuleOp>(function->getParentOp()) || getTargetArch(function) != PTOArch::A3) {
        return false;
    }
    auto core = recoverSyncPhysicalCore(function);
    if (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV) {
        return false;
    }
    if (hasPhysicalSections(function) || function->hasAttr(ActiveContextAttr)) {
        return false;
    }
    // Sealed exports remain leaves. Private qualification may own closed calls.
    const bool basic = !function
                            .walk([&](Operation* op) {
                                return (isa<SymbolUserOpInterface>(op) && !(closedCalls && isa<func::CallOp>(op))) ||
                                               getSyncMacroModel(op).has_value() ||
                                               recoverSyncPhysicalCore(op) != core ?
                                           WalkResult::interrupt() :
                                           WalkResult::advance();
                            })
                            .wasInterrupted();
    if (!basic) {
        return false;
    }
    auto input = std::make_shared<SyncInput>(GMAliasPolicy::MayAlias);
    if (failed(input->build(function, SyncInstructionView::PipeEnvelopes, SyncInputDetail::CoarseCompletion))) {
        return false;
    }
    for (const auto* phase : input->instructions()) {
        if (getSyncPipelinePresence("a3", core, static_cast<PIPE>(phase->kPipeValue)) != std::optional<bool>(true)) {
            return false;
        }
    }
    if (source) {
        *source = std::move(input);
    }
    return true;
}
uint8_t modes(const SyncInput& input)
{
    uint8_t result = 0;
    for (const auto* phase : input.instructions()) {
        if (!phase->useVec.empty()) {
            result |= 1;
        }
        if (!phase->defVec.empty()) {
            result |= 2;
        }
    }
    return result;
}
bool validSummary(const CompletionSummary& summary)
{
    return summary.all.size() == 1 && summary.exposed.size() == 1 && summary.suffix.size() == 1 &&
           summary.all[0] <= 3 && summary.exposed[0] <= 3 && summary.suffix[0] <= 3 &&
           !(summary.exposed[0] & ~summary.all[0]) && !(summary.suffix[0] & ~summary.all[0]);
}
} // namespace
std::optional<CompletionSummary> closedCompletionSummary(const PreparedLogicalPlan& plan)
{
    if (!plan.completeInvocation) {
        return std::nullopt;
    }
    uint8_t all = 0;
    if (plan.completionOwner) {
        all = plan.completionOwner->transfer().summary.all[0];
    } else if (
        plan.mathematicalOwner && plan.mathematicalOwner->input &&
        plan.mathematicalOwner->input->memory().gmPolicy() == GMAliasPolicy::MayAlias) {
        all = modes(*plan.mathematicalOwner->input);
    } else {
        return std::nullopt;
    }
    return CompletionSummary{{all}, {all}, {0}, false};
}
LogicalResult emitClosedCompletionExport(func::FuncOp function, const CompletionSummary& summary)
{
    if (!callable(function) || !completed(function) || !validSummary(summary) || summary.transparent ||
        summary.suffix[0] || summary.exposed != summary.all) {
        return failure();
    }
    Builder b(function.getContext());
    NamedAttrList fields;
    fields.set("version", b.getI64IntegerAttr(1));
    fields.set("all", b.getI64IntegerAttr(summary.all[0]));
    fields.set("closed_matching", b.getUnitAttr());
    fields.set("completed_exit", b.getUnitAttr());
    function->setAttr(closedAttr, fields.getDictionary(function.getContext()));
    fields.set("seal", b.getStringAttr(closedCompletionSeal(function)));
    function->setAttr(closedAttr, fields.getDictionary(function.getContext()));
    return success();
}
FailureOr<CompletionSummary> decodeClosedCompletionExport(func::FuncOp function, std::string& error)
{
    auto certificate = function->getAttrOfType<DictionaryAttr>(closedAttr);
    auto version = certificate ? certificate.getAs<IntegerAttr>("version") : IntegerAttr{};
    auto all = certificate ? certificate.getAs<IntegerAttr>("all") : IntegerAttr{};
    auto seal = certificate ? certificate.getAs<StringAttr>("seal") : StringAttr{};
    if (!callable(function) || !completed(function) || !version || !version.getType().isInteger(64) ||
        version.getInt() != 1 || !all || !all.getType().isInteger(64) || all.getInt() < 0 || all.getInt() > 3 ||
        !seal || !certificate.getAs<UnitAttr>("closed_matching") || !certificate.getAs<UnitAttr>("completed_exit") ||
        seal.getValue() != closedCompletionSeal(function)) {
        error = "closed completion child has stale or unsupported matching/completion evidence";
        return failure();
    }
    return CompletionSummary{{uint8_t(all.getInt())}, {uint8_t(all.getInt())}, {0}, false};
}
std::shared_ptr<const ClosedCompletionChild> ClosedCompletionChild::import(func::CallOp call, std::string& error)
{
    ClosedCompletionImportContext context;
    return import(call, error, context);
}
std::shared_ptr<const ClosedCompletionChild> ClosedCompletionChild::import(
    func::CallOp call, std::string& error, ClosedCompletionImportContext& context)
{
    error.clear();
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    auto parent = call->getParentOfType<func::FuncOp>();
    std::shared_ptr<SyncInput> coarse;
    if (!callee || !parent || call.getNumResults() ||
        !callable(callee, &coarse, callee.isPrivate() && !callee->hasAttr(closedAttr)) ||
        getTargetArch(parent) != getTargetArch(callee) ||
        recoverSyncPhysicalCore(parent) != recoverSyncPhysicalCore(callee) ||
        call.getOperandTypes() != callee.getArgumentTypes() || callee == parent || failed(verify(callee))) {
        error = "completion call lacks a same-core closed child without result-readiness obligations";
        return {};
    }
    if (context.active.contains(callee)) {
        error = "completion child has recursive call ownership";
        return {};
    }
    if (auto found = context.retained.find(callee); found != context.retained.end()) { return found->second; }
    context.active.insert(callee);
    auto leave = llvm::make_scope_exit([&] { context.active.erase(callee); });
    auto owner = std::shared_ptr<ClosedCompletionChild>(new ClosedCompletionChild);
    owner->callee = callee;
    owner->module = callee->getParentOp();
    if (callee->hasAttr(closedAttr)) {
        auto summary = decodeClosedCompletionExport(callee, error);
        if (failed(summary)) {
            return {};
        }
        // Physical IDs and hidden reservations are jointly closed by the child's
        // existing completed exit. The barrier-only parent allocates no IDs.
        bool logical = false;
        callee.walk([&](Operation* op) { logical |= isa<LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op); });
        if (logical) {
            error = "completion child requires physical allocation before closed import";
            return {};
        }
        owner->effects = *summary;
        owner->resources = ResourceClosure::CompletedExit;
    } else {
        if (!callee.isPrivate()) {
            error = "completion child has unqualified event lifetimes";
            return {};
        }
        auto input = std::move(coarse);
        PhaseIndex index;
        if (failed(index.build(callee, *input))) {
            error = "completion child lacks a qualified phase index";
            return {};
        }
        owner->qualification = CompletionRegion::qualifyExistingChild(callee, input, index, error, context);
        if (!owner->qualification) { return {}; }
        owner->effects = owner->qualification->transfer().summary;
        if (!owner->qualification->importedFunctions().empty()) {
            owner->resources = ResourceClosure::ClosedDescendants;
        }
    }
    owner->seal = closedCompletionSeal(callee);
    context.retained[callee] = owner;
    return owner;
}
bool ClosedCompletionChild::unchanged() const
{
    DenseSet<const ClosedCompletionChild*> checked;
    return unchanged(checked);
}
bool ClosedCompletionChild::unchanged(DenseSet<const ClosedCompletionChild*>& checked) const
{
    if (checked.contains(this)) { return true; }
    auto function = callee;
    const bool valid = function && !function.isDeclaration() && function.getBody().hasOneBlock() &&
        function->getParentOp() == module && seal == closedCompletionSeal(function) &&
        (!qualification || qualification->validateClosedChildren(checked));
    if (valid) { checked.insert(this); }
    return valid;
}
} // namespace mlir::pto::frontiersynch
