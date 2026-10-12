// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate a complete staged invocation before consuming it with one body transfer.
#include "PrivateStatePlan.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
std::string text(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    operation->print(stream);
    return result;
}
std::string symbolContext(ModuleOp module)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    module->getAttrDictionary().print(stream);
    for (Operation& operation : *module.getBody()) {
        if (auto function = dyn_cast<func::FuncOp>(operation)) {
            NamedAttrList attributes(function->getAttrs());
            for (auto name : {"pto.frontier_recognition", "pto.prefix_protocol", "pto.endpoint_families",
                              "pto.cyclic_allocation", "pto.finite_allocation", "pto.sync_context_plans",
                              "pto.sync_subblock_guard"}) {
                attributes.erase(name);
            }
            attributes.getDictionary(module.getContext()).print(stream);
            stream << (function.isDeclaration() ? "declaration" : "definition");
        } else {
            // Globals and nested symbol scopes are not changed by this phase.
            operation.print(stream);
        }
    }
    return result;
}
std::vector<Operation*> operations(func::FuncOp function)
{
    std::vector<Operation*> result;
    function.walk([&](Operation* operation) { result.push_back(operation); });
    return result;
}
bool clean(const PreparedLogicalPlan& plan)
{
    return plan.completeInvocation && plan.preparation.empty() && plan.endpoints.empty() && plan.families.empty() &&
        !plan.nestedIdentities && !plan.independentPieces && !plan.groupedFamilies &&
        !plan.allocationCertificate && !plan.allocationPreparation && !plan.regionalAllocation &&
        !plan.compactBoundingOwner;
}
bool prefixMetadata(func::FuncOp function, int64_t planId)
{
    auto protocol = function->getAttrOfType<DictionaryAttr>("pto.prefix_protocol");
    auto version = protocol ? protocol.getAs<IntegerAttr>("version") : IntegerAttr{};
    auto plan = protocol ? protocol.getAs<IntegerAttr>("plan") : IntegerAttr{};
    auto directions = protocol ? protocol.getAs<ArrayAttr>("directions") : ArrayAttr{};
    return version && version.getInt() == 1 && plan && plan.getInt() == planId && directions &&
        !function->hasAttr("pto.endpoint_families") && !function->hasAttr(CyclicAllocationAttr) &&
        !function->hasAttr(FiniteAllocationAttr);
}
} // namespace
PrivateStatePlan::~PrivateStatePlan() = default;
FailureOr<std::unique_ptr<PreparedLogicalPlan>> preparePrivateStatePlan(
    func::FuncOp original, const InvocationPrefixRecipe& recipe,
    std::shared_ptr<const MathematicalResult> mathematics, PrivateStateTarget target, PrivateStateError& error)
{
    error = {};
    const bool valid = original && isa<ModuleOp>(original->getParentOp()) && mathematics &&
        mathematics->input && mathematics->region == 0 && target == PrivateStateTarget::ScalarControl64 &&
        recipe.privateScalarControl;
    if (!valid) {
        error = {PrivateStateFailure::InvalidInput, original,
                 "private publication requires complete root mathematics and explicit scalar-control target"};
        return failure();
    }
    auto owner = std::unique_ptr<PrivateStatePlan>(new PrivateStatePlan);
    owner->original = original;
    owner->originalModule = cast<ModuleOp>(original->getParentOp());
    owner->originalText = text(original->getParentOp());
    owner->functionText = text(original);
    owner->contextText = symbolContext(owner->originalModule);
    owner->originalOperations = operations(original);
    owner->mathematics = mathematics;
    owner->planId = recipe.planId;
    auto staged = materializeInvocationPrefix(original, recipe, error);
    if (failed(staged)) { return failure(); }
    owner->staged = std::move(*staged);
    // Preserve the existing complete-invocation exit interface, after all prefix
    // cleanup acquisitions and before the original return.
    auto* terminal = owner->staged->function.front().getTerminator();
    auto previous = dyn_cast_or_null<BarrierOp>(terminal->getPrevNode());
    const auto all = PipeAttr::get(original.getContext(), PIPE::PIPE_ALL);
    const bool needsExit = !previous || previous.getPipe() != all;
    if (needsExit) {
        OpBuilder builder(terminal);
        builder.create<BarrierOp>(terminal->getLoc(), all);
    }
    const bool intact = owner->originalText == text(original->getParentOp()) &&
        owner->originalOperations == operations(original);
    const bool metadata = prefixMetadata(owner->staged->function, recipe.planId);
    const bool verified = succeeded(verify(*owner->staged->module));
    if (!intact || !metadata || !verified) {
        error = {PrivateStateFailure::Verification, original,
                 "private publication changed source or protocol metadata"};
        return failure();
    }
    // Actual materialized work, not an estimate or an admission budget.
    uint64_t publications = 0;
    owner->staged->function.walk([&](LogicalSetOp) { ++publications; });
    Builder builder(original.getContext());
    NamedAttrList protocol(owner->staged->function->getAttrOfType<DictionaryAttr>("pto.prefix_protocol"));
    protocol.set("state_slots", builder.getI64IntegerAttr(owner->staged->finalState.size()));
    protocol.set("consumer_sites", builder.getI64IntegerAttr(recipe.consumers.size()));
    protocol.set("publication_sites", builder.getI64IntegerAttr(publications));
    owner->staged->function->setAttr("pto.prefix_protocol", protocol.getDictionary(original.getContext()));
    owner->stagedText = text(owner->staged->module->getOperation());
    auto plan = std::make_unique<PreparedLogicalPlan>(recipe.planId);
    plan->completeInvocation = true;
    plan->mathematicalOwner = std::move(mathematics);
    plan->privateState = std::move(owner);
    return plan;
}
LogicalResult PrivateStatePlan::validate(func::FuncOp function, PreparedLogicalPlan& plan, bool fullModule)
{
    const bool owned = !consumed && staged && function == original && plan.planId == planId &&
        function->getParentOp() == originalModule && plan.mathematicalOwner == mathematics &&
        plan.privateState.get() == this && clean(plan);
    if (!owned) {
        return function.emitError(
            "private publication requires its unconsumed original root plan without mixed endpoints");
    }
    const bool moduleUnchanged = !fullModule || originalText == text(function->getParentOp());
    const bool unchanged = moduleUnchanged && functionText == text(function) &&
        contextText == symbolContext(originalModule) && originalOperations == operations(function) &&
        stagedText == text(staged->module->getOperation());
    const bool metadata = prefixMetadata(staged->function, planId);
    if (!unchanged || !metadata || failed(verify(*staged->module))) {
        return function.emitError("private publication source or staged protocol changed after preparation");
    }
    return success();
}
LogicalResult PrivateStatePlan::preflightSiblingSynchronization(func::FuncOp function, PreparedLogicalPlan& plan)
{
    if (failed(validate(function, plan, true))) { return failure(); }
    siblingSynchronization = true;
    return success();
}
LogicalResult PrivateStatePlan::insert(func::FuncOp function, PreparedLogicalPlan& plan)
{
    if (failed(validate(function, plan, !siblingSynchronization))) { return failure(); }
    auto protocol = staged->function->getAttr("pto.prefix_protocol");
    function.getBody().takeBody(staged->function.getBody());
    function->setAttr("pto.prefix_protocol", protocol);
    if (plan.recognitionReport) { function->setAttr("pto.frontier_recognition", plan.recognitionReport); }
    consumed = true;
    // Mapping keys refer to the old body. Destroy the mapping before releasing
    // staged symbol context, and never expose it through the attached plan.
    staged.reset();
    originalOperations.clear();
    return success();
}
} // namespace mlir::pto::frontiersynch
