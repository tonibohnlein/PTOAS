// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "RecognitionInternal.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
namespace {
std::string text(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result); operation->print(stream); return result;
}
std::vector<Operation*> operations(func::FuncOp function)
{
    std::vector<Operation*> result;
    function.walk([&](Operation* operation) { result.push_back(operation); });
    return result;
}
// Flat compact CFG of original cuts: structured loops contribute backedges,
// never repeated visits. Payloads are individually closed children. Importing
// larger selectively synchronized children requires a separate closed contract.
class GraphBuilder {
public:
    explicit GraphBuilder(const PhaseIndex& index) : index(index) {}
    std::vector<CompletionVertex> graph;
    std::vector<Operation*> anchors;
    std::size_t vertex(Operation* anchor)
    {
        auto id = graph.size();
        graph.push_back({{{0}, {0}, {0}, true}, {}, false});
        anchors.push_back(anchor);
        if (!anchor) { return id; }
        uint8_t modes = 0;
        for (const auto* phase : index.phasesFor(anchor)) {
            if (!phase->useVec.empty()) { modes |= 1; }
            if (!phase->defVec.empty()) { modes |= 2; }
        }
        graph[id].summary = {{modes}, {modes}, {modes}, true};
        if (auto barrier = dyn_cast<BarrierOp>(anchor); barrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
            graph[id].summary.transparent = false;
        }
        return id;
    }
    std::size_t block(Block& block, std::size_t continuation, bool skipTerminator = false)
    {
        auto next = continuation;
        for (Operation& operation : llvm::reverse(block)) {
            auto* op = &operation;
            if (skipTerminator && op == block.getTerminator()) { continue; }
            auto id = vertex(op);
            if (auto branch = dyn_cast<scf::IfOp>(op)) {
                auto yes = this->block(branch.getThenRegion().front(), next);
                auto no = branch.getElseRegion().empty() ? next : this->block(branch.getElseRegion().front(), next);
                graph[id].successors = {yes, no};
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                auto header = vertex(nullptr);
                auto body = this->block(*loop.getBody(), header);
                graph[header].successors = {body, next};
                graph[id].successors = {header};
            } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
                auto condition = vertex(loop.getBefore().front().getTerminator());
                auto before = this->block(loop.getBefore().front(), condition, true);
                auto after = this->block(loop.getAfter().front(), before);
                graph[condition].successors = {after, next};
                graph[id].successors = {before};
            } else {
                graph[id].successors = {next};
            }
            next = id;
        }
        return next;
    }
private:
    const PhaseIndex& index;
};
} // namespace
std::shared_ptr<const CompletionRegion> CompletionRegion::build(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error)
{
    error.clear();
    const bool valid = function && input && !function.isDeclaration() &&
        function.getBody().hasOneBlock() && isa<ModuleOp>(function->getParentOp());
    if (!valid || failed(verify(function))) {
        error = "completion fallback requires one verified original structured function"; return {};
    }
    DenseSet<const CompoundInstanceElement*> remaining;
    for (const auto* phase : input->instructions()) {
        if (!phase || !phase->elementOp || !function->isProperAncestor(phase->elementOp) ||
            !remaining.insert(phase).second) {
            error = "completion input has mismatched original phases"; return {};
        }
    }
    function.walk([&](Operation* op) {
        for (const auto* phase : index.phasesFor(op)) {
            if (!remaining.erase(phase)) { error = "completion index belongs to different source facts"; }
        }
    });
    if (!remaining.empty() || !error.empty()) {
        error = "completion index does not cover the exact shared input"; return {};
    }
    auto core = recoverSyncPhysicalCore(function);
    // A2/A3 PipeBarrier contract: ALL completes prior interfaces on every
    // pipeline of this core. Scalar-to-scalar ordering is hardware-provided.
    // https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html
    if (getTargetArch(function) != PTOArch::A3 ||
        (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV)) {
        error = "completion fallback lacks a qualified single-core A3 protocol"; return {};
    }
    RecognitionResult leaves;
    function.walk([&](Operation* op) {
        if (op == function.getOperation() || !error.empty()) { return; }
        if (recoverSyncPhysicalCore(op) != core || isa<func::CallOp, LogicalSetOp, LogicalWaitOp>(op) ||
            (isPreservedSyncProtocol(op) && !isa<BarrierOp>(op))) {
            error = "completion child lacks closed same-core execution"; return;
        }
        if (index.needsValuePrerequisite(op) ||
            llvm::any_of(index.prerequisitesFor(op), [](const auto& prerequisite) { return !prerequisite.native; })) {
            error = "completion fallback lacks a forward scalar prerequisite certificate"; return;
        }
        if (op->getNumRegions()) {
            if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op) ||
                llvm::any_of(op->getRegions(), [](Region& region) {
                    return !region.empty() && !region.hasOneBlock();
                })) { error = "completion fallback has unsupported control"; }
            return;
        }
        detail::inspectLeaf(*op, index, leaves);
        for (const auto* phase : index.phasesFor(op)) {
            auto present = getSyncPipelinePresence("a3", core, static_cast<PIPE>(phase->kPipeValue));
            if (present != std::optional<bool>(true) ||
                llvm::any_of(op->getResults(), [&](Value value) {
                    return resultAvailability(*phase, value) == SyncResultAvailability::RequiresCompletion;
                })) { error = "completion child requires an unavailable pipe or scalar readiness protocol"; }
        }
        if (llvm::any_of(index.prerequisitesFor(op), [](const auto& prerequisite) { return !prerequisite.native; })) {
            error = "completion fallback lacks a forward scalar prerequisite certificate";
        }
    });
    if (!error.empty()) { return {}; }
    if (leaves.state != RecognitionState::Applicable) {
        error = "completion fallback has unmodeled effects, macro matching or prerequisites"; return {};
    }
    auto owner = std::shared_ptr<CompletionRegion>(new CompletionRegion);
    owner->original = function; owner->input = std::move(input);
    owner->originalText = text(function); owner->originalOperations = operations(function);
    owner->moduleAttributes = function->getParentOp()->getAttrDictionary();
    GraphBuilder builder(index);
    auto exit = builder.vertex(nullptr);
    auto entry = builder.block(function.front(), exit);
    owner->result = buildCompletionTransfer(builder.graph, entry, exit, 1);
    if (!owner->result.error.empty()) { error = owner->result.error; return {}; }
    for (std::size_t v = 0; v < builder.graph.size(); ++v) {
        if (!owner->result.protectEntry[v]) { continue; }
        if (!builder.anchors[v]) { error = "completion placement lacks an original command cut"; return {}; }
        owner->cuts.push_back(builder.anchors[v]);
    }
    // The complete invocation drains after every child's work and before return.
    owner->cuts.push_back(function.front().getTerminator());
    owner->result.summary.suffix.assign(1, 0);
    owner->result.summary.transparent = false;
    return owner;
}
LogicalResult CompletionRegion::insert(func::FuncOp function, PreparedLogicalPlan& plan) const
{
    const bool clean = function == original && plan.completionOwner.get() == this && plan.completeInvocation &&
                       !plan.mathematicalOwner && !plan.privateState && !plan.compactBoundingOwner &&
                       !plan.regionalAllocation && !plan.allocationCertificate && !plan.allocationPreparation &&
                       !plan.familyRecipe && !plan.familyProtocol && !plan.physicalOutput && plan.preparation.empty() &&
                       plan.endpoints.empty() && plan.families.empty() && !plan.groupedFamilies &&
                       !plan.nestedIdentities && !plan.independentPieces;
    if (!clean || text(function) != originalText || operations(function) != originalOperations ||
        function->getParentOp()->getAttrDictionary() != moduleAttributes) {
        return function.emitError("completion plan no longer owns its unchanged original invocation");
    }
    IRMapping mapping;
    OwningOpRef<ModuleOp> staged = cast<ModuleOp>(function->getParentOp()->clone(mapping));
    auto clone = cast<func::FuncOp>(mapping.lookup(function.getOperation()));
    auto all = PipeAttr::get(function.getContext(), PIPE::PIPE_ALL);
    for (auto* cut : cuts) {
        auto* mapped = mapping.lookupOrNull(cut);
        if (!mapped) { return function.emitError("completion plan lost an original command cut"); }
        auto previous = dyn_cast_or_null<BarrierOp>(mapped->getPrevNode());
        if (previous && previous.getPipe() == all) { continue; }
        OpBuilder builder(mapped);
        auto barrier = builder.create<BarrierOp>(mapped->getLoc(), all);
        barrier->setAttr("pto.completion_boundary", builder.getUnitAttr());
    }
    Builder attributes(function.getContext());
    clone->setAttr("pto.completion_protocol", attributes.getStringAttr("a3-single-core-all; coarse-memory-class"));
    // Preserve the ordinary empty logical-plan contract for the physical pass.
    // No notification exists, so it has no event-ID assignment to construct.
    clone->setAttr("pto.endpoint_families", attributes.getDictionaryAttr({
        attributes.getNamedAttr("version", attributes.getI64IntegerAttr(1)),
        attributes.getNamedAttr("plan", attributes.getI64IntegerAttr(plan.planId)),
        attributes.getNamedAttr("families", attributes.getArrayAttr({})),
        attributes.getNamedAttr("pieces", attributes.getArrayAttr({}))}));
    if (failed(verify(*staged))) { return function.emitError("completion plan failed staged verification"); }
    function.getBody().takeBody(clone.getBody());
    function->setAttr("pto.endpoint_families", clone->getAttr("pto.endpoint_families"));
    function->setAttr("pto.completion_protocol", clone->getAttr("pto.completion_protocol"));
    if (plan.recognitionReport) { function->setAttr("pto.frontier_recognition", plan.recognitionReport); }
    return success();
}
} // namespace mlir::pto::frontiersynch
