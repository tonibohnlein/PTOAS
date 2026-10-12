// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Root attachment is transactional and never implies physical allocation support.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/PrivateStatePlan.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
std::string printed(Operation* operation)
{
    std::string result; llvm::raw_string_ostream stream(result); operation->print(stream); return result;
}
struct Fixture {
    explicit Fixture(MLIRContext& context) : module(ModuleOp::create(UnknownLoc::get(&context)))
    {
        OpBuilder b(&context); auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        function = b.create<func::FuncOp>(loc, "private_plan", b.getFunctionType({}, {b.getIndexType()}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        source = b.create<arith::AddIOp>(loc, one, one);
        auto consumer = b.create<arith::AddIOp>(loc, source->getResult(0), one);
        b.create<func::ReturnOp>(loc, consumer.getResult());
        auto input = std::make_shared<pto::SyncInput>();
        mathematics = std::make_shared<fs::MathematicalResult>(); mathematics->input = input;
        recipe.planId = 7; recipe.privateScalarControl = true;
        fs::PrefixDirection direction{static_cast<uint32_t>(pto::PIPE::PIPE_V),
                                      static_cast<uint32_t>(pto::PIPE::PIPE_MTE3)};
        recipe.directions.push_back(direction); recipe.sourceBounds.emplace(direction.sourcePipe, 1);
        recipe.sources.push_back({source, direction.sourcePipe, {}});
        recipe.consumers.push_back({consumer, direction,
            [](OpBuilder& b, const IRMapping&, ArrayRef<Value>, std::string&) -> FailureOr<Value> {
                return Value(b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1));
            }});
    }
    FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> prepare(fs::PrivateStateError& error)
    {
        return fs::preparePrivateStatePlan(function, recipe, mathematics,
                                           fs::PrivateStateTarget::ScalarControl64, error);
    }
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    Operation* source;
    std::shared_ptr<fs::MathematicalResult> mathematics;
    fs::InvocationPrefixRecipe recipe;
};
bool rejectionAndRetry(MLIRContext& context)
{
    Fixture fixture(context), foreign(context);
    fs::PrivateStateError error;
    const auto original = printed(fixture.module->getOperation());
    auto plan = fixture.prepare(error);
    if (failed(plan)) { return false; }
    auto wrongTarget = fs::preparePrivateStatePlan(fixture.function, fixture.recipe, fixture.mathematics,
                                                 fs::PrivateStateTarget::Unavailable, error);
    if (succeeded(wrongTarget)) { return false; }
    const bool foreignRejected = failed(fs::insertLogicalSynchronization(foreign.function, **plan));
    if (!foreignRejected) { return false; }
    fixture.source->setAttr("changed", UnitAttr::get(&context));
    const auto changed = printed(fixture.module->getOperation());
    const bool staleRejected = failed(fs::insertLogicalSynchronization(fixture.function, **plan));
    if (!staleRejected || changed != printed(fixture.module->getOperation())) { return false; }
    fixture.source->removeAttr("changed");
    (*plan)->endpoints.push_back({});
    const bool mixedRejected = failed(fs::insertLogicalSynchronization(fixture.function, **plan));
    if (!mixedRejected) { return false; }
    (*plan)->endpoints.clear();
    if (original != printed(fixture.module->getOperation())) { return false; }
    auto view = fs::takeRegionalAllocationView(**plan);
    if (view) { return false; }
    fs::prepareAllocationSupport(**plan);
    const bool attached = succeeded(fs::insertLogicalSynchronization(fixture.function, **plan));
    if (!attached || failed(verify(*fixture.module))) { return false; }
    const auto logical = printed(fixture.module->getOperation());
    const bool twiceRejected = failed(fs::insertLogicalSynchronization(fixture.function, **plan));
    const bool allocationRejected = failed(fs::allocatePhysicalEventIds(fixture.function, {0, 1, 2, 3, 4, 5}));
    return twiceRejected && allocationRejected && logical == printed(fixture.module->getOperation()) &&
        (*plan)->mathematicalOwner == fixture.mathematics;
}
func::FuncOp addSibling(Fixture& fixture, StringRef name, IRMapping& mapping)
{
    auto sibling = cast<func::FuncOp>(fixture.function->clone(mapping));
    sibling.setSymName(name);
    fixture.module->getBody()->push_back(sibling);
    return sibling;
}
bool batchAttachment(MLIRContext& context)
{
    Fixture fixture(context); fs::PrivateStateError error;
    IRMapping secondMap, ordinaryMap;
    auto second = addSibling(fixture, "second", secondMap);
    auto ordinary = addSibling(fixture, "ordinary", ordinaryMap);
    auto recipe = fixture.recipe;
    for (auto& source : recipe.sources) { source.original = secondMap.lookup(source.original); }
    for (auto& consumer : recipe.consumers) { consumer.original = secondMap.lookup(consumer.original); }
    auto firstPlan = fixture.prepare(error);
    auto secondPlan = fs::preparePrivateStatePlan(second, recipe, fixture.mathematics,
                                                  fs::PrivateStateTarget::ScalarControl64, error);
    const bool prepared = succeeded(firstPlan) && succeeded(secondPlan);
    if (!prepared) { return false; }
    const bool authorized = succeeded((*firstPlan)->privateState->preflightSiblingSynchronization(
        fixture.function, **firstPlan)) &&
        succeeded((*secondPlan)->privateState->preflightSiblingSynchronization(second, **secondPlan));
    if (!authorized) { return false; }
    fs::PreparedLogicalPlan plain(0); plain.completeInvocation = true;
    if (failed(fs::insertLogicalSynchronization(ordinary, plain))) { return false; }
    fixture.source->setAttr("changed_after_preflight", UnitAttr::get(&context));
    const auto changed = printed(fixture.module->getOperation());
    const bool stale = failed(fs::insertLogicalSynchronization(fixture.function, **firstPlan));
    if (!stale || changed != printed(fixture.module->getOperation())) { return false; }
    fixture.source->removeAttr("changed_after_preflight");
    const bool attached = succeeded(fs::insertLogicalSynchronization(fixture.function, **firstPlan)) &&
        succeeded(fs::insertLogicalSynchronization(second, **secondPlan));
    return attached && succeeded(verify(*fixture.module)) &&
        fixture.function->hasAttr("pto.prefix_protocol") && second->hasAttr("pto.prefix_protocol");
}
bool standaloneSiblingRefusal(MLIRContext& context)
{
    Fixture fixture(context); fs::PrivateStateError error; IRMapping mapping;
    auto sibling = addSibling(fixture, "sibling", mapping);
    auto plan = fixture.prepare(error);
    if (failed(plan)) { return false; }
    fs::PreparedLogicalPlan ordinary(0); ordinary.completeInvocation = true;
    if (failed(fs::insertLogicalSynchronization(sibling, ordinary))) { return false; }
    const auto changed = printed(fixture.module->getOperation());
    const bool refused = failed(fs::insertLogicalSynchronization(fixture.function, **plan));
    return refused && changed == printed(fixture.module->getOperation());
}
bool malformedMetadata(MLIRContext& context)
{
    Fixture fixture(context); fs::PrivateStateError error;
    const auto original = printed(fixture.module->getOperation());
    auto source = fixture.function.getOperation();
    fixture.recipe.actions.push_back({fixture.source, fs::PrivateStateCut::Before,
        [source](OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value>, std::string&) {
            mapping.lookup(source)->setAttr("pto.endpoint_families", b.getDictionaryAttr({}));
            return success();
        }});
    auto bad = fixture.prepare(error);
    const bool refused = failed(bad) && error.kind == fs::PrivateStateFailure::Verification;
    if (!refused || original != printed(fixture.module->getOperation())) { return false; }
    fixture.recipe.actions.clear();
    auto retry = fixture.prepare(error);
    return succeeded(retry) && original == printed(fixture.module->getOperation());
}
} // namespace
bool runPrivateStatePlanChecks()
{
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect, pto::PTODialect>();
    ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic&) { return success(); });
    const bool passed = rejectionAndRetry(context) && malformedMetadata(context) &&
        batchAttachment(context) && standaloneSiblingRefusal(context);
    if (!passed) { return false; }
    llvm::outs() << "private state plan checks passed\n";
    return true;
}
