// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Fuse retention with direct logical placement. Physical IDs are deliberately deferred.
#include "DirectEmissionInternal.h"
#include "GeneralQueries.h"
#include "SingleStreamLoop.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Dominance.h"
#include <limits>
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::frontiersynch {
class PendingPlanProvenance {
    friend DirectEmissionResult emitDirectDemands(
        func::FuncOp, const SyncInput&, const TraceDemandAnalysis&, CostLedger&);
    friend bool pendingPlanUnchanged(const DirectEmissionResult&);
    func::FuncOp source, pending;
    std::string sourceText, pendingText;
    SelectedAnalysisHandle selected;
    SmallVector<Operation*> sourceOperations;
    SmallVector<Attribute> sourceProperties, pendingProperties, contextProperties;
    SmallVector<std::pair<Operation*, DictionaryAttr>> sourceContexts;
    Operation* pendingParent = nullptr;
    static std::string render(func::FuncOp function)
    {
        std::string text;
        llvm::raw_string_ostream stream(text);
        function->print(stream);
        return text;
    }
    PendingPlanProvenance(func::FuncOp original, const DirectEmissionResult& result)
        : source(original), pending(result.pending.get()), sourceText(render(original)),
          pendingText(render(pending)), selected(result.selected), pendingParent(pending->getParentOp())
    {
        source.walk<WalkOrder::PreOrder>([&](Operation* operation) {
            sourceOperations.push_back(operation);
            sourceProperties.push_back(operation->getPropertiesAsAttribute());
        });
        pending->walk<WalkOrder::PreOrder>([&](Operation* operation) {
            pendingProperties.push_back(operation->getPropertiesAsAttribute());
        });
        for (auto* parent = source->getParentOp(); parent; parent = parent->getParentOp()) {
            sourceContexts.push_back({parent, parent->getAttrDictionary()});
            contextProperties.push_back(parent->getPropertiesAsAttribute());
        }
    }
};
llvm::StringRef realizedOrderStatement(const DirectEmissionResult& result)
{
    if (!result.pending || !result.pending.get()->hasAttr("pto.frontier.physical_status")) {
        return "not-certified";
    }
    auto physical = result.pending.get()->getAttrOfType<DictionaryAttr>("pto.frontier.physical");
    if (physical && physical.get("repair_excess")) {
        return "covers selected closure; physical repair adds prerequisites";
    }
    return "equal to selected closure under recorded modeled requirements";
}
bool pendingPlanUnchanged(const DirectEmissionResult& result)
{
    if (!result.pending || !result.pendingProvenance) { return false; }
    const auto& seal = *result.pendingProvenance;
    if (result.pending.get() != seal.pending || result.selected != seal.selected ||
        seal.pending->getParentOp() != seal.pendingParent) { return false; }
    // The owning root is borrowed and must remain alive through realization.
    // Find the original function by pointer identity before dereferencing it:
    // an erased function or payload is a rejection, not a stale-handle read.
    if (seal.sourceContexts.empty()) { return false; }
    Operation* root = seal.sourceContexts.back().first;
    bool sourcePresent = false;
    root->walk([&](Operation* operation) {
        sourcePresent |= operation == seal.source.operator->();
    });
    if (!sourcePresent) { return false; }
    auto* parent = seal.source->getParentOp();
    for (auto [index, context] : llvm::enumerate(seal.sourceContexts)) {
        if (parent != context.first || parent->getAttrDictionary() != context.second ||
            parent->getPropertiesAsAttribute() != seal.contextProperties[index]) { return false; }
        parent = parent->getParentOp();
    }
    if (parent) { return false; }
    SmallVector<Operation*> operations;
    SmallVector<Attribute> sourceProperties, pendingProperties;
    seal.source->walk<WalkOrder::PreOrder>([&](Operation* operation) {
        operations.push_back(operation);
        sourceProperties.push_back(operation->getPropertiesAsAttribute());
    });
    seal.pending->walk<WalkOrder::PreOrder>([&](Operation* operation) {
        pendingProperties.push_back(operation->getPropertiesAsAttribute());
    });
    return operations == seal.sourceOperations && sourceProperties == seal.sourceProperties &&
           pendingProperties == seal.pendingProperties &&
           PendingPlanProvenance::render(seal.source) == seal.sourceText &&
           PendingPlanProvenance::render(seal.pending) == seal.pendingText;
}
LogicalResult emitExplicitDemands(
    const TraceDemandAnalysis& trace, const SelectedAnalysis& selected, IRMapping& mapping,
    DirectEmissionResult& result)
{
    const auto& sequence = selected.sites;
    DenseMap<Operation*, Operation*> firstIncoming;
    auto retain = [&](const Demand& edge) {
        auto a = sequence[edge.source], b = sequence[edge.consumer];
        auto source = trace.sites()[a].phase->kPipeValue, target = trace.sites()[b].phase->kPipeValue;
        Operation* src = mapping.lookup(trace.sites()[a].anchor);
        Operation* dst = mapping.lookup(trace.sites()[b].anchor);
        OpBuilder builder(dst);
        if (edge.originalBarrier) {
            auto barrier = dyn_cast_or_null<pto::BarrierOp>(mapping.lookupOrNull(edge.originalBarrier));
            if (source != target || !barrier || barrier->getBlock() != dst->getBlock() ||
                barrier.getPipe().getPipe() != static_cast<pto::PIPE>(target) ||
                !src->isBeforeInBlock(barrier) || !barrier->isBeforeInBlock(dst)) {
                result.reason = "original mandatory drain lost its legal source cut";
                return failure();
            }
            return success();
        }
        if (source == target) {
            // The exact direct constructor's input class requires consecutive
            // executed occurrences on this pipe. A barrier before a skipped
            // local predecessor would add prerequisites, not establish equality.
            for (std::size_t between = edge.source + 1; between < edge.consumer; ++between) {
                if (trace.sites()[sequence[between]].phase->kPipeValue == target) {
                    result.reason = "unmet local-adjacency premise for exact direct construction";
                    return failure();
                }
            }
            if (firstIncoming.count(dst)) {
                builder.setInsertionPoint(firstIncoming.lookup(dst));
            }
            auto barrier = builder.create<pto::BarrierOp>(
                dst->getLoc(), pto::PipeAttr::get(dst->getContext(), static_cast<pto::PIPE>(target)));
            firstIncoming[dst] = barrier;
            ++result.barriers;
            return success();
        }
        auto srcPipe = pto::PipeAttr::get(src->getContext(), static_cast<pto::PIPE>(source));
        auto dstPipe = pto::PipeAttr::get(dst->getContext(), static_cast<pto::PIPE>(target));
        auto key = builder.getI64IntegerAttr(a * trace.sites().size() + b);
        auto wait = builder.create<pto::LogicalWaitOp>(dst->getLoc(), srcPipe, dstPipe, key, ValueRange{});
        firstIncoming.try_emplace(dst, wait);
        builder.setInsertionPointAfter(src);
        builder.create<pto::LogicalSetOp>(src->getLoc(), srcPipe, dstPipe, key, ValueRange{});
        ++result.sets;
        ++result.waits;
        return success();
    };
    for (auto id : selected.explicitReduction.retained()) {
        if (failed(retain(selected.generators[id]))) {
            return failure();
        }
    }
    return success();
}
DirectEmissionResult emitDirectDemands(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs)
{
    DirectEmissionResult result;
    const auto count = trace.sites().size();
    if (count && count > std::size_t(std::numeric_limits<int64_t>::max()) / count) {
        result.reason = "logical key identity is not representable";
        return result;
    }
    DenseMap<Operation*, unsigned> anchorPhases;
    for (const auto& site : trace.sites()) {
        if (++anchorPhases[site.anchor] > 1) {
            result.reason = "shared macro phases do not supply legal internal publication/acquisition cuts";
            return result;
        }
    }
    result.selected = selectAnalysis(function, input, trace, result.attempts, result.reason, costs);
    if (!result.selected) {
        return result;
    }
    result.route = result.selected->route;
    CostScope insertion(costs, CostStage::Insertion);
    IRMapping mapping;
    result.pending = cast<func::FuncOp>(function->clone(mapping));
    if (failed(input.target().retainSourceIds(result.pending.get(), mapping))) {
        result.reason = "original source identity has no cloned anchor";
        result.pending = nullptr;
        return result;
    }
    // Retain identities on borrowed source anchors so the durable relations can
    // still be interpreted after the original body is replaced. Original MLIR
    // remains the control tree; these attributes carry no new control structure.
    // Existing M0 loop/value IDs retain their separate historical postorder
    // numbering; phase site IDs remain the trace's contiguous site ordinals.
    DenseMap<Operation*, int64_t> originalIds;
    int64_t next = 0;
    function.walk<WalkOrder::PostOrder>([&](Operation* operation) { originalIds[operation] = next++; });
    for (auto [id, site] : llvm::enumerate(trace.sites())) {
        auto* anchor = mapping.lookup(site.anchor);
        anchor->setAttr("pto.frontier.site", IntegerAttr::get(IntegerType::get(function.getContext(), 64), id));
        for (Operation* loop : site.loops) {
            mapping.lookup(loop)->setAttr(
                "pto.frontier.loop",
                IntegerAttr::get(IntegerType::get(function.getContext(), 64), originalIds.lookup(loop)));
        }
    }
    if (result.selected->kind == SelectedAnalysis::Kind::Guarded) {
        for (const auto& node : result.selected->guarded.predicates().nodes()) {
            if (node.kind == PredicateKind::Atom && node.condition.getDefiningOp()) {
                auto* definition = node.condition.getDefiningOp();
                mapping.lookup(definition)
                    ->setAttr(
                        "pto.frontier.value_anchor",
                        IntegerAttr::get(IntegerType::get(function.getContext(), 64), originalIds.lookup(definition)));
            }
        }
    }
    switch (result.selected->kind) {
        case SelectedAnalysis::Kind::BoundaryLoop:
            result.emitted = succeeded(
                emitSingleStreamLoop(*result.selected->boundaryLoop, input, trace, mapping, result));
            break;
        case SelectedAnalysis::Kind::Explicit:
            result.emitted = succeeded(emitExplicitDemands(trace, *result.selected, mapping, result));
            break;
        case SelectedAnalysis::Kind::Periodic:
            result.emitted = succeeded(emitPeriodicDemands(trace, *result.selected, mapping, result));
            break;
        case SelectedAnalysis::Kind::Guarded:
            result.emitted = succeeded(emitGuardedDemands(trace, *result.selected, mapping, result));
            break;
        case SelectedAnalysis::Kind::General:
            result.emitted = succeeded(emitGeneralEndpoints(mapping, *result.selected, result));
            break;
        case SelectedAnalysis::Kind::Signed:
            result.emitted = succeeded(emitPreparedEndpoints(mapping, *result.selected, result));
            break;
    }
    // Validate the whole internal command plan before retaining/publishing it.
    // Invocation cleanup is a separate boundary contract, added below.
    // The private boundary builder qualified its internal plan before adding
    // and sealing the invocation drain; all other emitters do so here.
    if (result.emitted && result.selected->kind != SelectedAnalysis::Kind::BoundaryLoop &&
        failed(input.target().preflightMechanisms(result.pending.get(), mapping, result.reason))) {
        result.emitted = false;
    }
    if (result.emitted) {
        result.logicalContract = result.selected->contract;
        result.logicalContract.interfaces |= interfaceBit(DemandInterface::ExecutableEndpoints);
        auto logicalNeeds = AnalysisNeeds::defaultPolicy();
        logicalNeeds.interfaces |= interfaceBit(DemandInterface::ExecutableEndpoints);
        result.emitted = result.logicalContract.accepts(logicalNeeds, result.reason);
    }
    if (!result.emitted) {
        if (result.reason.empty()) {
            result.reason = "selected demand relation has no executable endpoint implementation";
        }
        for (auto& attempt : result.attempts) {
            if (attempt.outcome == "ready") {
                attempt.outcome = "unmet-obligation";
                attempt.obligation = result.reason;
            }
        }
        result.pending = nullptr;
        result.sets = result.waits = result.barriers = 0;
        result.privateSelectors = false;
        return result;
    }
    // The fixed invocation epilogue is outside all internal payloads. It is not
    // an internal PIPE_ALL repair and introduces no internal payload ordering.
    if (count == 0 || result.selected->kind == SelectedAnalysis::Kind::BoundaryLoop) {
        result.pendingProvenance = std::shared_ptr<const PendingPlanProvenance>(
            new PendingPlanProvenance(function, result));
        return result;
    }
    result.pending->walk([&](func::ReturnOp terminal) {
        OpBuilder builder(terminal);
        builder.create<pto::BarrierOp>(
            terminal.getLoc(),
            pto::PipeAttr::get(function.getContext(), static_cast<pto::PIPE>(PipelineType::PIPE_ALL)));
    });
    result.pendingProvenance = std::shared_ptr<const PendingPlanProvenance>(
        new PendingPlanProvenance(function, result));
    return result;
}
} // namespace mlir::pto::frontiersynch
