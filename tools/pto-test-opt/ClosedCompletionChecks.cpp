// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Production closed-child/coarse request tests. Existing transfer oracles check
// concrete pending-state traces independently of this adapter and its seals.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/ClosedCompletionChild.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
std::string render(Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    operation->print(out);
    return text;
}
bool failedAt(unsigned line)
{
    llvm::errs() << "closed completion check failed at line " << line << "\n";
    return false;
}
bool compile(ModuleOp module, bool completion)
{
    pto::PTOFrontierAnalysisOptions options;
    options.completionOnly = completion;
    options.gmAlias = "may-alias";
    PassManager manager(module.getContext());
    manager.addPass(pto::createPTOFrontierAnalysisPass(options));
    return succeeded(manager.run(module));
}
constexpr const char* eventful = R"mlir(
!tile = !pto.tile_buf<mat, 16x16xf16, blayout=col_major, slayout=row_major>
!left = !pto.tile_buf<left, 16x16xf16, slayout=row_major>
module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<cube>} {
  func.func @child(%a: !pto.ptr<f16, gm>) attributes {pto.export_closed_completion} {
    %base = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    %view = pto.make_tensor_view %a, shape = [%sixteen, %sixteen], strides = [%sixteen, %one]
        {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf16>
    %part = pto.partition_view %view, offsets = [%zero, %zero], sizes = [%sixteen, %sixteen] : !pto.tensor_view<?x?xf16>
    %tile = pto.alloc_tile addr = %base : !tile
    %left = pto.alloc_tile addr = %base : !left
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile : !tile)
    pto.textract ins(%tile, %zero, %zero : !tile, index, index) outs(%left : !left)
    return
  }
})mlir";
constexpr const char* caller = R"mlir(
func.func @parent(%a: !pto.ptr<f16, gm>, %take: i1, %n: index) {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.if %take { func.call @child(%a) : (!pto.ptr<f16, gm>) -> () }
  scf.for %i = %zero to %n step %one {
    func.call @child(%a) : (!pto.ptr<f16, gm>) -> ()
    func.call @child(%a) : (!pto.ptr<f16, gm>) -> ()
  }
  return
})mlir";
bool eventfulImport(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(eventful, &context);
    if (!module || !compile(*module, false)) {
        return failedAt(__LINE__);
    }
    auto child = module->lookupSymbol<func::FuncOp>("child");
    if (!child || !child->hasAttr("pto.closed_completion")) {
        return failedAt(__LINE__);
    }
    auto parentOwner = parseSourceString<ModuleOp>(
        std::string("module {\nfunc.func private @child(!pto.ptr<f16, gm>)\n") + caller + "\n}", &context);
    if (!parentOwner) {
        return failedAt(__LINE__);
    }
    auto parent = parentOwner->lookupSymbol<func::FuncOp>("parent");
    parent->remove();
    module->getBody()->push_back(parent);
    func::CallOp first;
    parent.walk([&](func::CallOp call) {
        if (!first) {
            first = call;
        }
    });
    std::string error;
    if (fs::ClosedCompletionChild::import(first, error)) {
        return failedAt(__LINE__);
    }
    if (failed(fs::allocatePhysicalEventIds(child, {0, 1, 2, 3, 4, 5}))) {
        return failedAt(__LINE__);
    }
    auto imported = fs::ClosedCompletionChild::import(first, error);
    if (!imported || imported->summary().all != fs::CompletionFacts{3} ||
        imported->summary().suffix != fs::CompletionFacts{0} || imported->summary().transparent) {
        llvm::errs() << error << "\n";
        return failedAt(__LINE__);
    }
    Region savedBody;
    savedBody.takeBody(child.getBody());
    if (imported->unchanged()) {
        return failedAt(__LINE__);
    }
    child.getBody().takeBody(savedBody);
    auto* extraBlock = new Block;
    child.getBody().push_back(extraBlock);
    if (imported->unchanged()) {
        return failedAt(__LINE__);
    }
    extraBlock->erase();
    // Summary provenance binds target, ABI and implementation, not only SSA.
    auto architecture = module->getOperation()->getAttr("pto.target_arch");
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
    if (imported->unchanged() || fs::ClosedCompletionChild::import(first, error)) {
        return failedAt(__LINE__);
    }
    module->getOperation()->setAttr("pto.target_arch", architecture);
    auto signature = child.getFunctionType();
    child.setType(FunctionType::get(&context, signature.getInputs(), {IntegerType::get(&context, 32)}));
    if (imported->unchanged() || fs::ClosedCompletionChild::import(first, error)) {
        return failedAt(__LINE__);
    }
    child.setType(signature);
    OpBuilder recursive(child.front().getTerminator());
    auto nested = recursive.create<func::CallOp>(child.getLoc(), child.getSymName(), TypeRange{}, child.getArguments());
    if (fs::ClosedCompletionChild::import(first, error)) {
        return failedAt(__LINE__);
    }
    nested.erase();
    // A manual eventful body without producer evidence cannot be imported.
    auto savedContract = child->getAttr("pto.closed_completion");
    child->removeAttr("pto.closed_completion");
    if (fs::ClosedCompletionChild::import(first, error)) {
        return failedAt(__LINE__);
    }
    child->setAttr("pto.closed_completion", savedContract);
    auto certificate = child->getAttrOfType<DictionaryAttr>("pto.closed_completion");
    child->setAttr("pto.closed_completion", UnitAttr::get(&context));
    if (imported->unchanged()) {
        return failedAt(__LINE__);
    }
    child->setAttr("pto.closed_completion", certificate);
    NamedAttrList changed(certificate);
    changed.set("all", IntegerAttr::get(IntegerType::get(&context, 64), 0));
    child->setAttr("pto.closed_completion", changed.getDictionary(&context));
    const auto stale = render(module->getOperation());
    {
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::decodeClosedCompletionExport(child, error)) ||
            succeeded(fs::allocatePhysicalEventIds(child, {0, 1, 2, 3, 4, 5})) ||
            stale != render(module->getOperation())) {
            return failedAt(__LINE__);
        }
    }
    child->setAttr("pto.closed_completion", certificate);
    auto completion = fs::CompletionRegion::buildCoarse(parent, pto::GMAliasPolicy::MayAlias, error);
    if (!completion || completion->inputOwner()->hasFineStorage() || completion->inputOwner()->descriptorState() ||
        completion->importedFunctions().size() != 1) {
        llvm::errs() << error << "\n";
        return failedAt(__LINE__);
    }
    fs::PreparedLogicalPlan plan(0);
    plan.completeInvocation = true;
    plan.completionOwner = completion;
    auto sourceText = render(module->getOperation());
    child->setAttr("test.changed_child", UnitAttr::get(&context));
    const auto modified = render(module->getOperation());
    {
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::insertLogicalSynchronization(parent, plan)) || modified != render(module->getOperation())) {
            return failedAt(__LINE__);
        }
    }
    child->removeAttr("test.changed_child");
    if (sourceText != render(module->getOperation())) {
        return failedAt(__LINE__);
    }
    // Public coordinator also preserves precompiled child flags and handles
    // guarded, repeated and sibling calls without allocating parent event IDs.
    auto publicCopy = cast<ModuleOp>(module->clone());
    OwningOpRef<ModuleOp> publicOwner(publicCopy);
    if (!compile(publicCopy, true)) {
        return failedAt(__LINE__);
    }
    auto publicParent = publicCopy.lookupSymbol<func::FuncOp>("parent");
    auto publicChild = publicCopy.lookupSymbol<func::FuncOp>("child");
    unsigned barriers = 0, flags = 0, calls = 0;
    publicParent.walk([&](Operation* op) {
        barriers += isa<pto::BarrierOp>(op);
        calls += isa<func::CallOp>(op);
        flags +=
            isa<pto::LogicalSetOp, pto::LogicalWaitOp, pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp,
                pto::WaitFlagDynOp>(op);
    });
    if (barriers != 1 || flags || calls != 3 || render(publicChild) != render(child)) {
        return failedAt(__LINE__);
    }
    if (failed(fs::insertLogicalSynchronization(parent, plan)) || failed(verify(*module))) {
        return failedAt(__LINE__);
    }
    return true;
}
constexpr const char* native = R"mlir(
!tile = !pto.tile_buf<vec, 1x8xf32>
module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func private @writer() {
    %address = arith.constant 0 : i64
    %one = arith.constant 1.0 : f32
    %tile = pto.alloc_tile addr = %address : !tile
    pto.texpands ins(%one : f32) outs(%tile : !tile)
    return
  }
  func.func private @reader() {
    %address = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %tile = pto.alloc_tile addr = %address : !tile
    %read = "pto.tgetval"(%tile, %zero) : (!tile, index) -> f32
    return
  }
  func.func @parent() {
    func.call @writer() : () -> ()
    func.call @reader() : () -> ()
    func.call @reader() : () -> ()
    return
  }
})mlir";
bool pendingPrimitive(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(native, &context);
    if (!module) {
        return failedAt(__LINE__);
    }
    auto parent = module->lookupSymbol<func::FuncOp>("parent");
    auto writer = module->lookupSymbol<func::FuncOp>("writer");
    std::string error;
    auto call = *parent.front().getOps<func::CallOp>().begin();
    auto child = fs::ClosedCompletionChild::import(call, error);
    if (!child || child->summary().all != fs::CompletionFacts{2} || child->summary().suffix != fs::CompletionFacts{2} ||
        !child->summary().transparent) {
        llvm::errs() << error << "\n";
        return failedAt(__LINE__);
    }
    auto coarse = std::make_shared<pto::SyncInput>();
    fs::PhaseIndex index;
    if (failed(
            coarse->build(writer, pto::SyncInstructionView::PipeEnvelopes, pto::SyncInputDetail::CoarseCompletion)) ||
        failed(index.build(writer, *coarse)) || coarse->hasFineStorage() || coarse->descriptorState() ||
        !coarse->accesses().cells().empty() || fs::analyzeExplicit(writer.front(), index, *coarse).error.empty() ||
        fs::analyzeFiniteGuarded(writer, {}, index, *coarse, {}).error.empty() ||
        succeeded(fs::recognizeProgram(writer, *coarse, index))) {
        return failedAt(__LINE__);
    }
    const auto writerText = render(writer);
    const auto readerText = render(module->lookupSymbol<func::FuncOp>("reader"));
    if (!compile(*module, true)) {
        return failedAt(__LINE__);
    }
    writer = module->lookupSymbol<func::FuncOp>("writer");
    parent = module->lookupSymbol<func::FuncOp>("parent");
    unsigned barriers = 0;
    parent.walk([&](pto::BarrierOp) { ++barriers; });
    if (barriers != 2 || writerText != render(writer) ||
        readerText != render(module->lookupSymbol<func::FuncOp>("reader"))) {
        return failedAt(__LINE__);
    }
    return true;
}
} // namespace
int runClosedCompletionChecks()
{
    MLIRContext context;
    context.disableMultithreading();
    context.loadDialect<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
    if (!eventfulImport(context) || !pendingPrimitive(context)) {
        return 1;
    }
    llvm::outs()
        << "closed completion checks: selective children, pending primitives, seals and coarse requests passed\n";
    return 0;
}
