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
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionInlineChild.h"
#include "../../lib/PTO/Transforms/FrontierSynch/AnalysisSessionInternal.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = pto::frontiersynch;
LogicalResult runCompletionInputChecks(func::FuncOp function);
namespace {
std::string render(Operation* operation)
{
    std::string text; llvm::raw_string_ostream stream(text); operation->print(stream); return text;
}
LogicalResult manyScopesChecks(func::FuncOp function)
{
    auto input = std::make_shared<pto::SyncInput>();
    fs::PhaseIndex index;
    if (failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, *input))) { return failure(); }
    std::string error;
    auto source = fs::CompletionRegion::captureSource(function, input, index, error);
    SmallVector<std::shared_ptr<const fs::CompletionInlineChild>> children;
    uint64_t visits = 0, phases = 0, vertices = 0;
    for (auto& op : function.front()) {
        if (!op.hasAttr("test.region_root")) { continue; }
        auto owner = fs::CompletionRegion::buildRegion(function, input, index, {&op}, {}, error, source);
        if (!owner || owner->sourceOwner() != source || owner->transfer().summary.suffix != std::vector<uint8_t>{2} ||
            !owner->transfer().summary.transparent) { return failure(); }
        if (owner->constructionWork().readinessOperationVisits != 3 ||
            owner->constructionWork().readinessUseVisits != 0) {
            return function.emitError("readiness discovery exceeded its scoped traversal contract");
        }
        visits += owner->constructionWork().scopedOperations;
        phases += owner->constructionWork().scopedPhases;
        vertices += owner->constructionWork().graphVertices;
        children.push_back(fs::CompletionInlineChild::qualify(owner));
    }
    const auto count = children.size();
    if (!count || visits != count || phases != count || vertices != 2 * count) {
        return function.emitError("regional construction revisited unrelated scope work");
    }
    auto whole = fs::CompletionRegion::buildWithChildren(function, input, index, children, error, source);
    if (!whole || whole->inlineChildren() != count) { return failure(); }
    children.clear(); source.reset(); input.reset();
    fs::PreparedLogicalPlan plan(0); plan.completeInvocation = true; plan.completionOwner = whole;
    if (failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) { return failure(); }
    std::size_t barriers = 0;
    function.walk([&](pto::BarrierOp) { ++barriers; });
    if (barriers != count) { return function.emitError("proper children acquired implicit exit drains"); }
    llvm::outs() << "regional scaling: children=" << count << " scoped-visits=" << visits
                 << " phase-visits=" << phases << " vertices=" << vertices << "\n";
    return success();
}
LogicalResult productionChecks(func::FuncOp function)
{
    std::shared_ptr<const fs::CompletionRegion> selected;
    std::unique_ptr<fs::PreparedLogicalPlan> prepared;
    for (bool prequery : {false, true}) {
        fs::FrontierAnalysis analysis(function);
        if (failed(analysis.initialize())) { return failure(); }
        if (prequery) {
            for (auto [id, node] : llvm::enumerate(analysis.result()->nodes)) {
                if (node.kind != fs::StructureKind::ExplicitRun) { continue; }
                fs::AnalysisRequest child; child.region = id; child.needs.synchronization = true;
                (void)analysis.analyze(child);
            }
        }
        fs::AnalysisRequest childRequest; childRequest.mode = fs::AnalysisMode::Fallback;
        childRequest.needs.synchronization = true;
        for (auto [id, node] : llvm::enumerate(analysis.result()->nodes)) {
            if (isa_and_nonnull<scf::WhileOp>(node.anchor)) { childRequest.region = id; break; }
        }
        if (!childRequest.region) { return function.emitError("proper while scope not found"); }
        auto child = analysis.analyze(childRequest);
        if (!child.completion) { return function.emitError("proper while completion unavailable"); }
        fs::AnalysisRequest request; request.mode = fs::AnalysisMode::Fallback;
        request.needs.synchronization = true;
        auto outcome = analysis.analyze(request);
        if (outcome.status != fs::AnalysisStatus::Ready || !outcome.completion || outcome.mathematical ||
            !outcome.completion->completesInvocation() || !outcome.completion->inlineChildren()) {
            return function.emitError("parent fallback did not retain qualified inline children: status=")
                << static_cast<int>(outcome.status) << " mathematical=" << bool(outcome.mathematical)
                << " completion=" << bool(outcome.completion)
                << " children=" << (outcome.completion ? outcome.completion->inlineChildren() : 0);
        }
        // Warm both scopes after their first evidence publication. Repeated
        // refreshes must not continually invalidate each other's generation.
        child = analysis.analyze(childRequest);
        outcome = analysis.analyze(request);
        child = analysis.analyze(childRequest);
        for (unsigned repeat = 0; repeat < 4; ++repeat) {
            if (analysis.analyze(request).completion != outcome.completion ||
                analysis.analyze(childRequest).completion != child.completion) {
                return function.emitError("completion generation caused parent/child rebuild feedback");
            }
        }
        if (selected && (selected->transfer().protectEntry != outcome.completion->transfer().protectEntry ||
            selected->inlineChildren() != outcome.completion->inlineChildren())) {
            return function.emitError("child request order changed parent completion selection");
        }
        if (!prequery) {
            bool published = false;
            for (auto [id, node] : llvm::enumerate(analysis.result()->nodes)) {
                if (node.kind != fs::StructureKind::ExplicitRun || node.payloads.empty()) { continue; }
                fs::AnalysisRequest childRequest; childRequest.region = id;
                auto child = analysis.analyzeFiniteExpansion(childRequest);
                published |= child.status == fs::AnalysisStatus::Ready && bool(child.mathematical);
            }
            auto refreshed = analysis.analyze(request);
            if (!published || !refreshed.completion || refreshed.completion == outcome.completion ||
                refreshed.completion->transfer().protectEntry != outcome.completion->transfer().protectEntry ||
                refreshed.completion->inlineChildren() != outcome.completion->inlineChildren()) {
                return function.emitError("new child evidence did not refresh the cached parent snapshot");
            }
            auto oldPlan = analysis.prepareLogical(outcome);
            if (failed(oldPlan) || failed(analysis.attachAllocation(outcome, **oldPlan))) {
                return function.emitError("parent refresh invalidated its previously issued plan");
            }
            outcome = std::move(refreshed);
        }
        selected = outcome.completion;
        auto plan = analysis.prepareLogical(outcome);
        if (failed(plan) || failed(analysis.attachAllocation(outcome, **plan))) { return failure(); }
        prepared = std::move(*plan);
    }
    // Both sessions are gone: retained recipes and input must own their lifetime.
    if (failed(fs::insertLogicalSynchronization(function, *prepared)) || failed(verify(function))) { return failure(); }
    int local = 0, returns = 0;
    function.walk([&](pto::BarrierOp barrier) {
        local += !barrier->hasAttr("pto.completion_boundary");
        returns += isa_and_nonnull<func::ReturnOp>(barrier->getNextNode());
    });
    if (local != 1 || returns != 1) { return function.emitError("parent lost child barrier or invocation drain"); }
    llvm::outs() << "regional production: " << function.getSymName() << " retained-local=1 invocation-drain=1\n";
    return success();
}
LogicalResult scopedChecks(func::FuncOp function)
{
    const auto original = render(function);
    auto input = std::make_shared<pto::SyncInput>();
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index->build(function, *input))) { return failure(); }
    SmallVector<Operation*> roots, childRoots;
    function.walk([&](Operation* op) {
        if (op->hasAttr("test.region_root")) { roots.push_back(op); }
        if (op->hasAttr("test.exact_child")) { childRoots.push_back(op); }
    });
    SmallVector<std::shared_ptr<const fs::CompletionInlineChild>> children;
    if (!childRoots.empty()) {
        auto source = fs::finiteFunctionInput(function, input, index);
        auto slice = fs::finiteRegionInput(source, childRoots);
        auto finite = fs::analyzeFinite(slice, fs::FiniteStrategy::Concrete);
        auto capability = fs::completionChildExport(finite);
        auto child = capability();
        const auto prepared = finite->work.recipeBuilds + finite->work.recipeHits;
        if (capability() != child || finite->work.recipeBuilds + finite->work.recipeHits != prepared) {
            return function.emitError("completion child capability repeated recipe qualification");
        }
        if (function->hasAttr("test.reject_exact_child")) {
            if (child) { return function.emitError("eventful inline child was called resource-empty"); }
        } else {
            if (!child) { return function.emitError("exact child did not qualify as event-free"); }
            children.push_back(child);
        }
    }
    std::string error;
    auto source = fs::CompletionRegion::captureSource(function, input, *index, error);
    auto owner = fs::CompletionRegion::buildRegion(function, input, *index, roots, children, error, source);
    if (!owner || owner->completesInvocation() || render(function) != original) {
        return function.emitError("proper-region qualification failed: ") << error;
    }
    // Publication belongs to this snapshot, not to the current candidate cache.
    // New exact evidence may invalidate a selection without invalidating an
    // already-issued proper-region owner or its executable fragment.
    fs::AnalysisSessionState snapshots;
    snapshots.completions[1].owner = owner;
    snapshots.issuedCompletions.insert(owner);
    snapshots.retainedEvidencePublished();
    snapshots.completions[1].owner.reset();
    snapshots.attempts[1][fs::AnalysisBackend::Sequence].mathematical =
        std::make_shared<fs::MathematicalResult>();
    if (!snapshots.ownsCompletion(owner)) { return function.emitError("issued completion owner was invalidated"); }
    const auto& summary = owner->transfer().summary;
    const auto expected = function->getAttrOfType<DenseI64ArrayAttr>("test.scoped_completion");
    SmallVector<int64_t> observed{summary.all[0], summary.exposed[0], summary.suffix[0], summary.transparent};
    if (!expected || expected.asArrayRef() != ArrayRef<int64_t>(observed)) {
        return function.emitError("proper-region pending summary differs from oracle");
    }
    if (roots.size() > 1) {
        SmallVector<Operation*> reversed(roots.rbegin(), roots.rend());
        if (fs::CompletionRegion::buildRegion(function, input, *index, reversed, {}, error, source)) {
            return function.emitError("noncontiguous original roots were accepted");
        }
    }
    if (!children.empty()) {
        SmallVector<std::shared_ptr<const fs::CompletionInlineChild>> duplicate{children[0], children[0]};
        if (fs::CompletionRegion::buildRegion(function, input, *index, roots, duplicate, error, source)) {
            return function.emitError("overlapping exact children were accepted");
        }
    }
    if (function->hasAttr("test.production_scope")) {
        fs::FrontierAnalysis analysis(function);
        if (failed(analysis.initialize())) { return failure(); }
        auto* program = analysis.result();
        std::optional<std::size_t> id;
        for (auto [i, node] : llvm::enumerate(program->nodes)) {
            if (roots.size() == 1 && node.anchor == roots.front()) { id = i; break; }
        }
        if (!id || !*id) { return function.emitError("proper structural scope not found"); }
        fs::AnalysisRequest request; request.region = *id; request.mode = fs::AnalysisMode::Fallback;
        request.needs.synchronization = true;
        auto result = analysis.analyze(request);
        if (!result.completion || result.mathematical || result.status != fs::AnalysisStatus::Ready ||
            result.completion->completesInvocation()) { return function.emitError("regional fallback request failed"); }
        auto repeated = analysis.analyze(request);
        if (repeated.completion != result.completion) { return function.emitError("regional fallback was rebuilt"); }
        auto prepared = analysis.prepareLogical(result);
        if (failed(prepared) || (*prepared)->completeInvocation ||
            failed(analysis.attachAllocation(result, **prepared))) {
            return function.emitError("regional completion preparation lost ownership");
        }
        request.needs.queries = true;
        auto query = analysis.analyze(request);
        if (!query.completion || query.available.queries || query.status != fs::AnalysisStatus::UnmetObligation) {
            return function.emitError("conservative summary impersonates exact queries");
        }
    }
    fs::PreparedLogicalPlan plan(0);
    plan.completionOwner = owner;
    {
        ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
        if (succeeded(fs::insertLogicalSynchronization(function, plan)) || render(function) != original) {
            return function.emitError("proper-region fragment independently changed whole-function metadata");
        }
    }
    auto expectedBarriers = function->getAttrOfType<IntegerAttr>("test.local_barriers");
    if (expectedBarriers) {
        auto fragment = fs::CompletionInlineChild::qualify(owner);
        auto whole = fs::CompletionRegion::buildWithChildren(function, input, *index, {fragment}, error, source);
        if (!whole || !whole->completesInvocation()) { return failure(); }
        fs::PreparedLogicalPlan wholePlan(0); wholePlan.completeInvocation = true; wholePlan.completionOwner = whole;
        // Equal module attributes do not preserve symbol resolution after relocation.
        auto* block = function->getBlock();
        auto position = std::next(function->getIterator());
        OwningOpRef<ModuleOp> other = ModuleOp::create(function.getLoc());
        other->getOperation()->setAttrs(function->getParentOp()->getAttrs());
        function->moveBefore(other->getBody(), other->getBody()->end());
        bool rejected = false;
        {
            ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
            rejected = failed(fs::insertLogicalSynchronization(function, wholePlan));
        }
        function->moveBefore(block, position);
        if (!rejected || render(function) != original) {
            return function.emitError("source snapshot accepted a different symbol context");
        }
        // Accepted fragments cannot silently fall back to primitive descent
        // after their original implementation changes.
        childRoots.back()->setAttr("test.stale_child", UnitAttr::get(function.getContext()));
        const auto stale = render(function);
        {
            ScopedDiagnosticHandler silence(function.getContext(), [](Diagnostic&) { return success(); });
            if (succeeded(fs::insertLogicalSynchronization(function, wholePlan)) || render(function) != stale) {
                return function.emitError("stale accepted child mutated its parent transaction");
            }
        }
        childRoots.back()->removeAttr("test.stale_child");
        if (failed(fs::insertLogicalSynchronization(function, wholePlan)) || failed(verify(function))) {
            return failure();
        }
        int64_t barriers = 0;
        function.walk([&](pto::BarrierOp barrier) { barriers += !barrier->hasAttr("pto.completion_boundary"); });
        if (barriers != expectedBarriers.getInt()) { return function.emitError("retained child barriers changed"); }
    }
    if (fs::CompletionInlineChild::qualify(std::make_shared<fs::FiniteAnalysis>(nullptr, fs::ExplicitAnalysis{}))) {
        return function.emitError("malformed finite ownership was accepted");
    }
    llvm::outs() << "scoped completion: " << function.getSymName() << " pending=" << unsigned(summary.suffix[0])
                 << " transparent=" << summary.transparent << " children=" << owner->inlineChildren() << "\n";
    return success();
}
}
LogicalResult runCompletionRegionChecks(func::FuncOp function)
{
    if (function.isDeclaration()) { return success(); }
    if (function->hasAttr("test.storage_classes") || function->hasAttr("test.readiness")) {
        return runCompletionInputChecks(function);
    }
    if (function->hasAttr("test.many_completion")) { return manyScopesChecks(function); }
    if (function->hasAttr("test.production_parent")) { return productionChecks(function); }
    if (function->hasAttr("test.scoped_completion")) { return scopedChecks(function); }
    const auto original = render(function);
    auto input = std::make_shared<pto::SyncInput>();
    fs::PhaseIndex index;
    if (failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index.build(function, *input))) { return failure(); }
    std::string error;
    auto completion = fs::CompletionRegion::build(function, input, index, error);
    std::string coarseError;
    auto coarse = fs::CompletionRegion::buildCoarse(function, pto::GMAliasPolicy::MayNotAlias, coarseError);
    if (bool(completion) != bool(coarse)) {
        return function.emitError("fine/coarse completion qualification differs");
    }
    if (completion) {
        const auto& fine = completion->transfer();
        const auto& broad = coarse->transfer();
        if (coarse->inputOwner()->hasFineStorage() || coarse->inputOwner()->descriptorState() ||
            fine.protectEntry != broad.protectEntry || fine.summary.all != broad.summary.all ||
            fine.summary.exposed != broad.summary.exposed || fine.summary.suffix != broad.summary.suffix ||
            fine.summary.transparent != broad.summary.transparent) {
            return function.emitError("coarse request changed completion transfer");
        }
    }
    if (function->hasAttr("test.completion_refuse")) {
        if (completion || error.empty() || render(function) != original) { return failure(); }
        llvm::outs() << "completion refused: " << function.getSymName() << "\n";
        return success();
    }
    if (!completion || render(function) != original) {
        return function.emitError("completion construction failed: ") << error;
    }
    if (function->hasAttr("test.class_scope")) {
        auto* first = input->instructions().front()->elementOp;
        auto proper = fs::CompletionRegion::buildRegion(function, input, index, {first}, {}, error);
        auto child = fs::CompletionInlineChild::qualify(proper);
        // Separate snapshots over the same input must agree on full-input class IDs.
        auto parent = fs::CompletionRegion::buildWithChildren(function, input, index, {child}, error);
        if (!parent || parent->sourceOwner() == proper->sourceOwner() ||
            parent->transfer().summary.all != completion->transfer().summary.all ||
            parent->transfer().protectEntry != completion->transfer().protectEntry) {
            return function.emitError("proper child changed the shared storage class interpretation");
        }
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
