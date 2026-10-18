// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve a qualified exact child's barriers through the common materializer.
#include "CompletionInlineChild.h"
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include <map>
namespace mlir::pto::frontiersynch {
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<FiniteAnalysis> finite)
{
    if (!finite) { return {}; }
    using Result = std::shared_ptr<const CompletionInlineChild>;
    auto retained = std::make_shared<std::optional<Result>>();
    return [finite = std::move(finite), retained]() -> Result {
        if (!*retained) { *retained = CompletionInlineChild::qualify(finite); }
        return **retained;
    };
}
namespace {
bool contains(const DenseSet<Operation*>& roots, Operation* operation)
{
    for (auto* op = operation; op; op = op->getParentOp()) {
        if (roots.contains(op)) { return true; }
    }
    return false;
}
bool eventFree(const PreparedLogicalPlan& plan)
{
    if (plan.completeInvocation || plan.privateState || plan.completionOwner || plan.compactBoundingOwner ||
        plan.physicalOutput || plan.executedOutput || plan.deferredOutput) { return false; }
    if (llvm::any_of(plan.families, [](const auto& family) {
        return !family.local || family.sourceChoices || family.targetChoices;
    })) { return false; }
    if (llvm::any_of(plan.endpoints, [](const auto& endpoint) {
        return endpoint.kind != LogicalCommandKind::Barrier;
    })) { return false; }
    if (plan.familyRecipe) {
        auto shapes = familyMatchingShapes(*plan.familyRecipe);
        if (failed(shapes) || llvm::any_of(*shapes, [](const auto& entry) {
            return entry.second.hasSet || entry.second.hasWait;
        })) { return false; }
    }
    return true;
}
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(std::shared_ptr<FiniteAnalysis> finite)
{
    if (!finite || !finite->originalInput() || !finite->error().empty() || finite->originalInput()->wholeFunction() ||
        finite->originalInput()->roots().empty()) { return {}; }
    auto plan = finite->prepare();
    if (failed(plan) || !(*plan)->endpoints.empty() ||
        !(*plan)->preparation.empty() || (*plan)->familyProtocol) { return {}; }
    std::optional<PhysicalAllocationPlan> assignment;
    SmallVector<scf::ForOp> loops;
    if (!eventFree(**plan)) {
        // Concrete singleton demands have exactly one unconditional SET/WAIT
        // per selected record and visit. No early publication or cleanup.
        auto* concrete = finite->concrete();
        if (!concrete || (*plan)->completeInvocation || (*plan)->privateState ||
            llvm::any_of((*plan)->families, [](const auto& family) {
                return family.sourceChoices || family.targetChoices || family.members.size() != 1;
            })) { return {}; }
        for (auto* parent = finite->originalInput()->roots().front()->getParentOp();
             parent != finite->originalInput()->function(); parent = parent->getParentOp()) {
            if (auto loop = dyn_cast<scf::ForOp>(parent)) { loops.push_back(loop); }
            else if (!isa<scf::IfOp>(parent)) { return {}; }
        }
        auto certificate = explicitAllocationCertificate(*concrete, (*plan)->planId,
                                                        finite->originalInput()->function().getContext());
        if (!certificate) { return {}; }
        ScopedDiagnosticHandler silence(finite->originalInput()->function().getContext(),
                                        [](Diagnostic&) { return success(); });
        auto allocated = decodeFiniteAllocation(finite->originalInput()->function(), certificate, {0,1,2,3,4,5});
        if (failed(allocated) || allocated->records.empty()) { return {}; }
        assignment = std::move(*allocated);
        auto arena = std::make_shared<RegionExpressions>();
        for (auto loop : loops) {
            auto prefixed = prefixFamilyInvocation((*plan)->familyRecipe, arena, arena->input(loop.getInductionVar()));
            if (failed(prefixed)) { return {}; }
            (*plan)->familyRecipe = std::move(*prefixed);
        }
        (*plan)->nestedIdentities = true;
        (*plan)->independentPieces = true;
    }
    auto fragment = LogicalFamilyFragment::capture(**plan);
    if (!fragment) { return {}; }
    auto child = std::shared_ptr<CompletionInlineChild>(new CompletionInlineChild);
    child->finite = std::move(finite);
    child->fragment = std::move(fragment);
    child->eventAssignment = std::move(assignment);
    uint8_t modes = 0;
    const auto& input = *child->finite->originalInput();
    for (auto* root : input.roots()) {
        root->walk([&](Operation* op) {
            for (const auto* phase : input.index().phasesFor(op)) {
                if (!phase->useVec.empty()) { modes |= 1; }
                if (!phase->defVec.empty()) { modes |= 2; }
            }
        });
    }
    // Internal barriers remain in the implementation. The coarse import need
    // not reconstruct which effects they complete; it conservatively carries all.
    child->effects = {{modes}, {modes}, {modes}, true};
    return child;
}
std::optional<uint64_t> CompletionInlineChild::payloadCount() const
{
    if (finite && finite->concrete()) { return finite->concrete()->occurrences.size(); }
    return std::nullopt;
}
bool CompletionInlineChild::exactRealizedOrder() const
{
    return finite && finite->concrete() &&
        explicitRealizationOrder(*finite->concrete()) == RealizationOrder::Equivalent;
}
bool CompletionInlineChild::usesEvents() const
{
    return completion ? completion->usesEventResources() : eventAssignment.has_value();
}
bool CompletionInlineChild::returnsPendingEvents() const
{
    return completion ? completion->returnsPendingEvents() : eventAssignment.has_value();
}
CompletionSummary CompletionInlineChild::classifiedSummary(
    const CompletionStorage& storage, const PhaseIndex& index) const
{
    // Proper completion children use the same full-input class interpretation.
    // Finite children retain their exact code but import conservative effects.
    return completion ? effects : storage.summarize(roots(), index);
}
ArrayRef<Operation*> CompletionInlineChild::roots() const
{
    return completion ? completion->originalRoots() : finite->originalInput()->roots();
}
const SyncInput* CompletionInlineChild::input() const
{
    return completion ? completion->inputOwner().get() : &finite->originalInput()->storage();
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(
    std::shared_ptr<const CompletionRegion> completion)
{
    if (!completion || completion->completesInvocation() || completion->summaryOnly ||
        completion->originalRoots().empty()) { return {}; }
    auto child = std::shared_ptr<CompletionInlineChild>(new CompletionInlineChild);
    child->effects = completion->transfer().summary;
    child->completion = std::move(completion);
    return child;
}
namespace {
LogicalResult appendChildPalettes(const PhysicalAllocationPlan& assignment, MLIRContext* context,
    uint64_t recordBase, CompletionChildAssembly& assembly)
{
    Builder b(context);
    auto& groups = assembly.allocationGroups;
    // The ordinary finite allocator supplied constant lanes within this
    // child. The parent proves all eventful visits completion-separated;
    // no conflicts are introduced against another child's palette.
    using Lane = std::tuple<uint32_t, uint32_t, int64_t>;
    std::map<Lane, SmallVector<int64_t>> lanes;
    for (const auto& record : assignment.records) {
        if (record.ids.size() != 1) { return failure(); }
        lanes[{record.sourcePipe, record.targetPipe, record.ids.front()}].push_back(record.record + recordBase);
    }
    std::map<std::pair<uint32_t, uint32_t>, SmallVector<int64_t>> previous;
    for (const auto& [lane, records] : lanes) {
        const auto [source, target, slot] = lane;
        auto& conflicts = previous[{source, target}];
        SmallVector<int64_t> zeros(records.size(), 0);
        auto group = b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(source)),
            b.getNamedAttr("target", b.getI64IntegerAttr(target)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(1)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr(conflicts)),
            b.getNamedAttr("forbidden_ids", b.getDenseI64ArrayAttr({}))});
        conflicts.push_back(groups.size());
        groups.push_back(group);
    }
    return success();
}
} // namespace
LogicalResult CompletionInlineChild::append(PreparedLogicalPlan& destination, IRMapping& mapping,
    SmallVectorImpl<Operation*>& completionCuts, DenseSet<const CompletionSourceSnapshot*>& checkedSources,
    DenseSet<const ClosedCompletionChild*>& checkedCalls, CompletionChildAssembly& assembly) const
{
    if (completion) {
        return completion->append(destination, mapping, completionCuts, checkedSources, checkedCalls, assembly);
    }
    auto fresh = fragment->instantiate();
    auto& plan = *fresh;
    if (failed(materializeLogicalFamilies(plan)) || (!eventAssignment && !eventFree(plan))) { return failure(); }
    const uint64_t recordBase = assembly.nextRecord;
    for (const auto& family : plan.families) {
        for (const auto& member : family.members) {
            const auto record = recordBase + member.record;
            if (record > UINT32_MAX) { return failure(); }
            assembly.nextRecord = std::max(assembly.nextRecord, record + 1);
        }
    }
    // Preserve singleton family provenance, including barrier-only siblings.
    // The shared insertion path performs final namespace normalization.
    for (auto family : plan.families) {
        family.id += recordBase;
        for (auto* cut : {&family.sourceCut, &family.targetCut}) {
            cut->before = mapping.lookupOrNull(cut->before);
            if (!cut->before) { return failure(); }
            cut->block = cut->before->getBlock();
        }
        for (auto& member : family.members) {
            member.record += recordBase;
            for (auto* coordinates : {&member.sourceCoordinates, &member.targetCoordinates}) {
                for (auto& coordinate : *coordinates) {
                    coordinate.loop = cast<scf::ForOp>(mapping.lookup(coordinate.loop.getOperation()));
                }
            }
        }
        destination.families.push_back(std::move(family));
    }
    destination.independentPieces = true;
    destination.nestedIdentities = true;
    DenseSet<Operation*> rootSet(roots().begin(), roots().end());
    for (const auto& stage : plan.preparation) {
        // The common materializer can hoist safe scalar work outside the
        // child interval. Its certified original placement must still belong
        // to this invocation; only command cuts are confined to child roots.
        if (!stage.before || stage.before->getParentOfType<func::FuncOp>() !=
            finite->originalInput()->function()) { return failure(); }
        auto* cut = mapping.lookupOrNull(stage.before);
        if (!cut) { return failure(); }
        auto& block = destination.addPreparation(cut);
        for (auto& operation : *stage.code) { block.push_back(operation.clone(mapping)); }
    }
    for (auto endpoint : plan.endpoints) {
        if (!contains(rootSet, endpoint.before)) { return failure(); }
        endpoint.before = mapping.lookupOrNull(endpoint.before);
        endpoint.guard = mapping.lookupOrDefault(endpoint.guard);
        if (!endpoint.before) { return failure(); }
        endpoint.identity = mapping.lookupOrDefault(endpoint.identity);
        for (auto& coordinate : endpoint.memberCoordinates) { coordinate = mapping.lookupOrDefault(coordinate); }
        // Concrete/contextual singleton fragments retain one provenance label.
        if (endpoint.records.size() != 1) { return failure(); }
        endpoint.records.front() += recordBase;
        endpoint.domain.clear();
        endpoint.piece = destination.endpoints.size();
        if (endpoint.kind != LogicalCommandKind::Barrier) {
            if (endpoint.memberCoordinates.empty()) { return failure(); }
            auto& code = destination.addPreparation(endpoint.before);
            OpBuilder builder(endpoint.before->getContext());
            builder.setInsertionPointToEnd(&code);
            endpoint.memberCoordinates.front() = builder.create<arith::ConstantIndexOp>(
                endpoint.before->getLoc(), endpoint.records.front());
        }
        destination.endpoints.push_back(std::move(endpoint));
    }
    if (eventAssignment && failed(appendChildPalettes(*eventAssignment,
        finite->originalInput()->function().getContext(), recordBase, assembly))) { return failure(); }
    return success();
}
} // namespace mlir::pto::frontiersynch
