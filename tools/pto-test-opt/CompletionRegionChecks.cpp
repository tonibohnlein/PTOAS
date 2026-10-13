// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Original-cut oracle and transactional checks for conservative completion.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
std::string render(Operation* operation)
{
    std::string text; llvm::raw_string_ostream stream(text); operation->print(stream); return text;
}
}
LogicalResult runCompletionRegionChecks(func::FuncOp function)
{
    if (function.isDeclaration()) { return success(); }
    const auto original = render(function);
    auto input = std::make_shared<pto::SyncInput>();
    fs::PhaseIndex index;
    if (failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, *input))) { return failure(); }
    std::string error;
    auto completion = fs::CompletionRegion::build(function, input, index, error);
    if (function->hasAttr("test.completion_refuse")) {
        if (completion || error.empty() || render(function) != original) { return failure(); }
        llvm::outs() << "completion refused: " << function.getSymName() << "\n";
        return success();
    }
    if (!completion || render(function) != original) {
        return function.emitError("completion construction failed: ") << error;
    }
    auto expected = function->getAttrOfType<DenseI64ArrayAttr>("test.completion_cuts");
    if (!expected) { return failure(); }
    fs::PreparedLogicalPlan plan(0);
    plan.completeInvocation = true;
    plan.completionOwner = completion;
    // Mixing a second realization must fail before changing any original IR.
    plan.endpoints.emplace_back();
    {
        ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
        if (succeeded(fs::insertLogicalSynchronization(function, plan)) || render(function) != original) {
            return failure();
        }
    }
    plan.endpoints.clear();
    function->setAttr("test.changed_after_completion", UnitAttr::get(function.getContext()));
    const auto changed = render(function);
    {
        ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
        if (succeeded(fs::insertLogicalSynchronization(function, plan)) || render(function) != changed) {
            return function.emitError("completion accepted changed source IR");
        }
    }
    function->removeAttr("test.changed_after_completion");
    // A different source input's phases cannot be combined with this index.
    auto other = std::make_shared<pto::SyncInput>();
    if (failed(other->build(function, pto::SyncInstructionView::PipeEnvelopes))) { return failure(); }
    if (!other->instructions().empty() && fs::CompletionRegion::build(function, other, index, error)) {
        return function.emitError("completion accepted another input's phase identity");
    }
    if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) { return failure(); }
    std::set<int64_t> observed;
    SmallVector<Operation*> erase;
    bool good = true;
    function.walk([&](pto::BarrierOp barrier) {
        if (!barrier->hasAttr("pto.completion_boundary")) { return; }
        auto* next = barrier->getNextNode();
        auto id = next ? next->getAttrOfType<IntegerAttr>("test.cut") : IntegerAttr{};
        if (!id || barrier.getPipe().getPipe() != pto::PIPE::PIPE_ALL || !observed.insert(id.getInt()).second) {
            good = false;
        }
        erase.push_back(barrier);
    });
    const std::set<int64_t> wanted(expected.asArrayRef().begin(), expected.asArrayRef().end());
    if (!good || observed != wanted) {
        return function.emitError("completion cuts differ from independent expectation");
    }
    for (auto* operation : erase) { operation->erase(); }
    function->removeAttr("pto.completion_protocol");
    function->removeAttr("pto.endpoint_families");
    if (render(function) != original) { return function.emitError("completion changed original payload/control IR"); }
    llvm::outs() << "completion region: " << function.getSymName() << " cuts=" << observed.size()
                 << " payload-control-preserved\n";
    return success();
}
