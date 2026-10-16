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
        imported->summary().suffix != fs::CompletionFacts{0} || imported->summary().transparent ||
        imported->resourceClosure() != fs::ClosedCompletionChild::ResourceClosure::CompletedExit) {
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
        completion->importedFunctions().size() != 1 || completion->childContracts().completedExit != 1 ||
        completion->childContracts().eventFree || completion->childContracts().closedDescendants ||
        !completion->completesInvocation()) {
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
// The selective leaf is produced and physically allocated above. These private
// wrappers own no events; their pending tail is an ordinary asynchronous load.
constexpr const char* descendants = R"mlir(
!tile = !pto.tile_buf<mat, 16x16xf16, blayout=col_major, slayout=row_major>
module {
  func.func private @child(!pto.ptr<f16, gm>)
  func.func private @pending(%a: !pto.ptr<f16, gm>) {
    func.call @child(%a) : (!pto.ptr<f16, gm>) -> ()
    %base = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    %view = pto.make_tensor_view %a, shape = [%sixteen, %sixteen], strides = [%sixteen, %one]
        {layout = #pto.layout<nd>} : !pto.tensor_view<?x?xf16>
    %part = pto.partition_view %view, offsets = [%zero, %zero], sizes = [%sixteen, %sixteen]
        : !pto.tensor_view<?x?xf16>
    %tile = pto.alloc_tile addr = %base : !tile
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile : !tile)
    return
  }
  func.func private @relay(%a: !pto.ptr<f16, gm>) {
    func.call @pending(%a) : (!pto.ptr<f16, gm>) -> ()
    pto.barrier <PIPE_ALL>
    func.call @pending(%a) : (!pto.ptr<f16, gm>) -> ()
    return
  }
  func.func @parent(%a: !pto.ptr<f16, gm>) {
    func.call @relay(%a) : (!pto.ptr<f16, gm>) -> ()
    func.call @relay(%a) : (!pto.ptr<f16, gm>) -> ()
    return
  }
})mlir";
fs::CompletionSummary traceSummary(ArrayRef<uint8_t> trace)
{
    // Independent concrete trace: zero is an existing ALL, other tokens are
    // payload effects. A selective leaf's internally ordered payloads form one
    // token, followed by its checked existing completion.
    uint8_t all = 0, exposed = 0, pending = 0;
    bool completed = false;
    for (auto mode : trace) {
        if (!mode) { pending = 0; completed = true; continue; }
        all |= mode;
        if (!completed) { exposed |= mode; }
        pending |= mode;
    }
    return {{all}, {exposed}, {pending}, !completed};
}
bool sameSummary(const fs::CompletionSummary& a, const fs::CompletionSummary& b)
{
    return a.all == b.all && a.exposed == b.exposed && a.suffix == b.suffix && a.transparent == b.transparent;
}
bool pendingDescendants(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(eventful, &context);
    auto additions = parseSourceString<ModuleOp>(descendants, &context);
    if (!module || !additions || !compile(*module, false)) { return failedAt(__LINE__); }
    auto leaf = module->lookupSymbol<func::FuncOp>("child");
    for (auto name : {"pending", "relay", "parent"}) {
        auto function = additions->lookupSymbol<func::FuncOp>(name);
        function->remove();
        module->getBody()->push_back(function);
    }
    auto pending = module->lookupSymbol<func::FuncOp>("pending");
    auto relay = module->lookupSymbol<func::FuncOp>("relay");
    auto parent = module->lookupSymbol<func::FuncOp>("parent");
    auto first = *parent.front().getOps<func::CallOp>().begin();
    std::string error;
    if (fs::ClosedCompletionChild::import(first, error) ||
        error != "completion child requires physical allocation before closed import" ||
        failed(fs::allocatePhysicalEventIds(leaf, {0, 1, 2, 3, 4, 5}))) {
        return failedAt(__LINE__);
    }
    fs::ClosedCompletionImportContext imports;
    auto qualified = fs::ClosedCompletionChild::import(first, error, imports);
    auto reference = traceSummary({3, 0, 3, 0, 3, 0, 3});
    if (!qualified || !sameSummary(qualified->summary(), reference) || imports.retained.size() != 3 ||
        qualified->resourceClosure() != fs::ClosedCompletionChild::ResourceClosure::ClosedDescendants) {
        llvm::errs() << error << "\n";
        return failedAt(__LINE__);
    }
    auto again = fs::ClosedCompletionChild::import(first, error, imports);
    if (again != qualified) { return failedAt(__LINE__); }
    // A wrapper's unchanged local text cannot hide a modified grandchild.
    const auto wrapperText = render(pending);
    leaf->setAttr("test.changed_descendant", UnitAttr::get(&context));
    if (qualified->unchanged() || fs::ClosedCompletionChild::import(first, error) ||
        wrapperText != render(pending)) { return failedAt(__LINE__); }
    leaf->removeAttr("test.changed_descendant");
    if (!qualified->unchanged()) { return failedAt(__LINE__); }
    auto certificate = leaf->getAttr("pto.closed_completion");
    leaf->removeAttr("pto.closed_completion");
    if (fs::ClosedCompletionChild::import(first, error)) { return failedAt(__LINE__); }
    leaf->setAttr("pto.closed_completion", certificate);
    // A saved recipe still owns logical notifications. It cannot acquire
    // resource-closed status merely by appearing in an otherwise qualified child.
    {
        OpBuilder builder(pending.front().getTerminator());
        OperationState state(pending.getLoc(), pto::LogicalFamilyOp::getOperationName());
        state.addAttribute("plan_id", builder.getI64IntegerAttr(0));
        state.addAttribute("cut_id", builder.getI64IntegerAttr(0));
        state.addAttribute("axis_ids", builder.getDenseI64ArrayAttr({}));
        state.addAttribute("pieces", builder.getArrayAttr({builder.getDictionaryAttr({})}));
        auto* marker = builder.create(state);
        const auto original = render(module->getOperation());
        if (failed(verify(pending)) || fs::ClosedCompletionChild::import(first, error) ||
            original != render(module->getOperation())) { return failedAt(__LINE__); }
        marker->erase();
    }
    // Two pending tails require an internal completion; qualification refuses
    // instead of silently inserting it into the child's implementation.
    auto* payload = pending.front().getTerminator()->getPrevNode();
    auto* repeated = payload->clone();
    pending.front().getOperations().insert(pending.front().getTerminator()->getIterator(), repeated);
    const auto invalidTail = render(module->getOperation());
    if (fs::ClosedCompletionChild::import(first, error) ||
        error != "closed-descendant child requires additional internal completion" ||
        invalidTail != render(module->getOperation())) { return failedAt(__LINE__); }
    repeated->erase();
    // A fresh active-import stack detects an indirect cycle, without accepting
    // an unfinished memo entry or recursing through validation indefinitely.
    OpBuilder cycleBuilder(pending.front().getTerminator());
    auto cycle = cycleBuilder.create<func::CallOp>(pending.getLoc(), relay.getSymName(), TypeRange{},
                                                  pending.getArguments());
    if (fs::ClosedCompletionChild::import(first, error) ||
        error != "completion child has recursive call ownership") { return failedAt(__LINE__); }
    cycle.erase();
    Operation* localFlag = nullptr;
    leaf.walk([&](Operation* op) {
        if (!localFlag && isa<pto::SetFlagOp, pto::WaitFlagOp>(op)) { localFlag = op; }
    });
    if (!localFlag) { return failedAt(__LINE__); }
    auto* copiedFlag = localFlag->clone();
    pending.front().getOperations().insert(pending.front().getTerminator()->getIterator(), copiedFlag);
    if (fs::ClosedCompletionChild::import(first, error) ||
        error != "completion child has local event resources or hidden protocols") {
        return failedAt(__LINE__);
    }
    copiedFlag->erase();
    const auto leafText = render(leaf), pendingText = render(pending), relayText = render(relay);
    // The same retained pending child is admitted under a guard and nonunit
    // repeated visits. Zero visits execute no loop-local completion commands.
    OwningOpRef<ModuleOp> guarded(cast<ModuleOp>(module->clone()));
    std::string guardedCaller(caller);
    for (std::size_t at = 0; (at = guardedCaller.find("@child", at)) != std::string::npos; at += 6) {
        guardedCaller.replace(at, 6, "@relay");
    }
    const auto step = guardedCaller.find("%one = arith.constant 1 : index");
    if (step == std::string::npos) { return failedAt(__LINE__); }
    guardedCaller.replace(step, std::string("%one = arith.constant 1 : index").size(),
                          "%one = arith.constant 2 : index");
    auto guardedSource = parseSourceString<ModuleOp>(
        std::string("module { func.func private @relay(!pto.ptr<f16, gm>)\n") + guardedCaller + "}", &context);
    if (!guardedSource) { return failedAt(__LINE__); }
    guarded->lookupSymbol<func::FuncOp>("parent").erase();
    auto guardedParent = guardedSource->lookupSymbol<func::FuncOp>("parent");
    guardedParent->remove();
    guarded->getBody()->push_back(guardedParent);
    if (!compile(*guarded, true)) { return failedAt(__LINE__); }
    guardedParent = guarded->lookupSymbol<func::FuncOp>("parent");
    unsigned loopCalls = 0, guardedCalls = 0, barriers = 0;
    bool ordered = true;
    guardedParent.walk([&](Operation* op) {
        barriers += isa<pto::BarrierOp>(op);
        if (!isa<func::CallOp>(op)) { return; }
        if (op->getParentOfType<scf::ForOp>()) {
            ++loopCalls;
            auto barrier = dyn_cast_or_null<pto::BarrierOp>(op->getPrevNode());
            ordered &= barrier && barrier.getPipe().getPipe() == pto::PIPE::PIPE_ALL;
        } else {
            ++guardedCalls;
            ordered &= !isa_and_nonnull<pto::BarrierOp>(op->getPrevNode());
        }
    });
    if (!ordered || loopCalls != 2 || guardedCalls != 1 || barriers != 3 ||
        leafText != render(guarded->lookupSymbol<func::FuncOp>("child")) ||
        pendingText != render(guarded->lookupSymbol<func::FuncOp>("pending")) ||
        relayText != render(guarded->lookupSymbol<func::FuncOp>("relay"))) {
        return failedAt(__LINE__);
    }
    if (!compile(*module, true)) { return failedAt(__LINE__); }
    parent = module->lookupSymbol<func::FuncOp>("parent");
    std::vector<unsigned> commands;
    for (auto& operation : parent.front()) {
        if (isa<func::CallOp>(operation)) { commands.push_back(1); }
        else if (auto barrier = dyn_cast<pto::BarrierOp>(operation)) {
            if (barrier.getPipe().getPipe() != pto::PIPE::PIPE_ALL) { return failedAt(__LINE__); }
            commands.push_back(2);
        } else if (isa<func::ReturnOp>(operation)) { commands.push_back(3); }
    }
    // Independent command-word check: pending conflicts acquire between calls,
    // and the complete parent drains at exit. No child boundary adds a drain.
    if (commands != std::vector<unsigned>{1, 2, 1, 2, 3} ||
        leafText != render(module->lookupSymbol<func::FuncOp>("child")) ||
        pendingText != render(module->lookupSymbol<func::FuncOp>("pending")) ||
        relayText != render(module->lookupSymbol<func::FuncOp>("relay"))) {
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
bool pendingFragments(MLIRContext& context)
{
    const std::string write = "pto.texpands ins(%one : f32) outs(%tile : !tile)";
    struct Case { std::string body; std::vector<std::vector<uint8_t>> traces; };
    const Case cases[] = {
        {write + "\npto.barrier <PIPE_ALL>\n" + write, {{2, 0, 2}}},
        {"pto.barrier <PIPE_ALL>\n" + write, {{0, 2}}},
        {write + "\nscf.if %take { pto.barrier <PIPE_ALL> }", {{2}, {2, 0}}},
        {write + "\n" + write, {{2, 2}}},
        {write + "\npto.barrier <PIPE_V>\n" + write, {{2, 2}}},
        {write + "\nscf.if %take { pto.barrier <PIPE_ALL> }\n" + write,
            {{2, 2}, {2, 0, 2}}},
        {"%begin = arith.constant 0 : index\n%step = arith.constant 1 : index\n"
            "scf.for %i = %begin to %n step %step { pto.barrier <PIPE_ALL>\n" + write + "\n}",
            {{}, {0, 2}, {0, 2, 0, 2, 0, 2}}},
    };
    for (const auto& test : cases) {
        // Independent finite pending-state simulation. Zero is an existing
        // ALL command; each other token is a payload's coarse access modes.
        uint8_t all = 0, exposed = 0, suffix = 0;
        bool accepted = true, transparent = false;
        for (const auto& trace : test.traces) {
            uint8_t pending = 0;
            bool completed = false;
            for (auto mode : trace) {
                if (!mode) { pending = 0; completed = true; continue; }
                if ((pending & 2) || ((pending & 1) && (mode & 2))) { accepted = false; }
                all |= mode;
                if (!completed) { exposed |= mode; }
                pending |= mode;
            }
            suffix |= pending;
            transparent |= !completed;
        }
        std::string source(native);
        for (const auto& replacement : {
                 std::make_pair("func.func private @writer()", "func.func private @writer(%take: i1, %n: index)"),
                 std::make_pair("func.func @parent()", "func.func @parent(%take: i1, %n: index)"),
                 std::make_pair("func.call @writer() : () -> ()",
                                "func.call @writer(%take, %n) : (i1, index) -> ()")}) {
            const auto found = source.find(replacement.first);
            if (found == std::string::npos) { return failedAt(__LINE__); }
            source.replace(found, std::string(replacement.first).size(), replacement.second);
        }
        const auto position = source.find(write);
        if (position == std::string::npos) { return failedAt(__LINE__); }
        source.replace(position, write.size(), test.body);
        auto module = parseSourceString<ModuleOp>(source, &context);
        if (!module) { return failedAt(__LINE__); }
        auto parent = module->lookupSymbol<func::FuncOp>("parent");
        auto writer = module->lookupSymbol<func::FuncOp>("writer");
        const auto before = render(module->getOperation());
        const auto writerText = render(writer);
        auto call = *parent.front().getOps<func::CallOp>().begin();
        std::string error;
        auto child = fs::ClosedCompletionChild::import(call, error);
        if (bool(child) != accepted || before != render(module->getOperation())) {
            llvm::errs() << error << "\n";
            return failedAt(__LINE__);
        }
        if (!accepted) {
            if (error != "event-free child requires additional internal completion") { return failedAt(__LINE__); }
            continue;
        }
        if (child->summary().all != fs::CompletionFacts{all} ||
            child->summary().exposed != fs::CompletionFacts{exposed} ||
            child->summary().suffix != fs::CompletionFacts{suffix} ||
            child->summary().transparent != transparent ||
            child->resourceClosure() != fs::ClosedCompletionChild::ResourceClosure::Empty) {
            return failedAt(__LINE__);
        }
        OpBuilder builder(call);
        builder.create<func::CallOp>(call.getLoc(), writer.getSymName(), TypeRange{}, call.getOperands());
        if (!compile(*module, true)) { return failedAt(__LINE__); }
        writer = module->lookupSymbol<func::FuncOp>("writer");
        parent = module->lookupSymbol<func::FuncOp>("parent");
        std::vector<unsigned> commands;
        for (auto& operation : parent.front()) {
            if (auto invocation = dyn_cast<func::CallOp>(operation)) {
                commands.push_back(invocation.getCallee() == "writer" ? 1 : 3);
            } else if (auto barrier = dyn_cast<pto::BarrierOp>(operation)) {
                if (barrier.getPipe().getPipe() != pto::PIPE::PIPE_ALL) { return failedAt(__LINE__); }
                commands.push_back(2);
            }
            else if (isa<func::ReturnOp>(operation)) { commands.push_back(4); }
        }
        // The pending write acquires before the first read; no import boundary
        // completion is added. The whole parent still completes before return.
        std::vector<unsigned> expected{1};
        if (exposed) { expected.push_back(2); }
        expected.insert(expected.end(), {1, 2, 3, 3, 2, 4});
        if (commands != expected || writerText != render(writer)) { return failedAt(__LINE__); }
    }
    return true;
}
} // namespace
int runClosedCompletionChecks()
{
    MLIRContext context;
    context.disableMultithreading();
    context.loadDialect<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
    if (!eventfulImport(context) || !pendingDescendants(context) ||
        !pendingPrimitive(context) || !pendingFragments(context)) {
        return 1;
    }
    llvm::outs()
        << "closed completion checks: selective children, pending primitives, seals and coarse requests passed\n";
    return 0;
}
