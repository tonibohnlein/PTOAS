// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve a qualified exact child's barriers through the common materializer.
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "CompletionInlineChild.h"
#include "CompletionVisitMapping.h"
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Diagnostics.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
std::optional<PhysicalRecordAllocation> completionVisitMapping(
    PhysicalRecordAllocation record, uint64_t originalArity, uint64_t enclosingArity)
{
    if (record.ids.empty() || !originalArity || originalArity > INT64_MAX ||
        enclosingArity > uint64_t(INT64_MAX) - originalArity ||
        llvm::any_of(record.ids, [](int64_t id) { return !isNativeEventId(id); })) { return {}; }
    // Validate before normalizing: a constant table cannot make an invalid
    // original offset rule or a foreign coordinate frame applicable.
    if (record.tupleRule && (record.tupleRule->coordinateCount != originalArity ||
        !validPhysicalTupleRule(*record.tupleRule, record.ids.size()))) { return {}; }
    const auto first = record.ids.front();
    if (llvm::all_of(record.ids, [first](int64_t id) { return id == first; })) {
        record.ids.assign(1, first);
        record.stride = record.phase = 0;
        if (record.tupleRule) { record.tupleRule = PhysicalTupleRule{originalArity, 0, {}}; }
    }
    for (auto [i, id] : llvm::enumerate(record.ids)) {
        if (id != first + static_cast<int64_t>(i)) { return {}; }
    }
    PhysicalTupleRule rule = record.tupleRule.value_or(PhysicalTupleRule{});
    if (!record.tupleRule && record.ids.size() > 1) {
        rule.terms.push_back({0, record.stride, record.phase, record.ids.size(), 1});
    }
    rule.coordinateCount = originalArity + enclosingArity;
    rule.base += first;
    record.tupleRule = std::move(rule);
    return record;
}
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
    // Automatic import retains the established default-pool qualification.
    return qualify(std::move(finite), defaultEventIds());
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(
    std::shared_ptr<FiniteAnalysis> finite, ArrayRef<int64_t> eligibleIds)
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
        auto allocated = decodeFiniteAllocation(finite->originalInput()->function(), certificate, eligibleIds);
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
std::function<std::shared_ptr<const CompletionInlineChild>()> completionChildExport(
    std::shared_ptr<BoundedLifetimeDemandResult> bounded, std::shared_ptr<const SyncInput> input)
{
    using Result = std::shared_ptr<const CompletionInlineChild>;
    auto retained = std::make_shared<std::optional<Result>>();
    return [bounded = std::move(bounded), input = std::move(input), retained]() -> Result {
        if (!*retained) { *retained = CompletionInlineChild::qualify(bounded, input); }
        return **retained;
    };
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualify(
    std::shared_ptr<BoundedLifetimeDemandResult> bounded, std::shared_ptr<const SyncInput> input)
{
    if (!bounded || !input || bounded->input != input.get()) { return {}; }
    ScopedDiagnosticHandler silence(bounded->function.getContext(), [](Diagnostic&) { return success(); });
    return qualifyVisit(bounded, std::move(input), bounded->loop, [bounded]() {
        std::string error;
        return prepareBoundedLifetimeClosedVisit(bounded, error);
    });
}
std::shared_ptr<const CompletionInlineChild> CompletionInlineChild::qualifyVisit(
    std::shared_ptr<const void> owner, std::shared_ptr<const SyncInput> input,
    scf::ForOp loop, VisitPreparation prepare)
{
    if (!prepare) { return {}; }
    auto prepared = prepare();
    if (failed(prepared)) { return {}; }
    auto plan = std::move(*prepared);
    if (!owner || !input || !loop || !plan || plan->completeInvocation || plan->privateState ||
        plan->completionOwner || plan->physicalOutput || plan->executedOutput || plan->deferredOutput ||
        !plan->allocationCertificate) { return {}; }
    auto function = loop->getParentOfType<func::FuncOp>();
    if (failed(materializeLogicalFamilies(*plan))) { return {}; }
    for (const auto& family : plan->families) {
        if (family.sourceChoices || family.targetChoices) { return {}; }
    }
    for (const auto& endpoint : plan->endpoints) {
        if (!loop->isProperAncestor(endpoint.before)) { return {}; }
    }
    auto assignment = decodePhysicalAllocationCertificate(function,
        plan->allocationCertificate, defaultEventIds());
    if (failed(assignment)) { return {}; }
    // The child is the whole loop, not one iteration. Outer visits are distinct
    // logical invocations; physical separation remains the parent's obligation.
    SmallVector<Value> enclosing;
    for (auto* parent = loop->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) { enclosing.push_back(loop.getInductionVar()); }
        else if (!isa<scf::IfOp>(parent)) { return {}; }
    }
    std::map<int64_t, uint64_t> arities;
    for (auto& endpoint : plan->endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        if (endpoint.records.size() != 1 || endpoint.memberCoordinates.empty()) { return {}; }
        endpoint.memberCoordinates.append(enclosing.begin(), enclosing.end());
        auto [entry, inserted] = arities.emplace(endpoint.records.front(), endpoint.memberCoordinates.size());
        if (!inserted && entry->second != endpoint.memberCoordinates.size()) { return {}; }
    }
    // Normalize certified constant tables, then preserve the exact assignment
    // as an offset into the existing directional palette. No record is removed.
    for (auto& record : assignment->records) {
        auto arity = arities.find(record.record);
        if (arity == arities.end()) { return {}; }
        auto converted = completionVisitMapping(record, arity->second - enclosing.size(), enclosing.size());
        if (!converted) { return {}; }
        record = std::move(*converted);
    }
    auto child = std::shared_ptr<CompletionInlineChild>(new CompletionInlineChild);
    child->visitRoots.push_back(loop);
    child->visitFunction = function;
    child->visitOwner = std::move(owner);
    child->visitInput = std::move(input);
    child->visitPreparation = std::move(prepare);
    child->enclosingVisits = std::move(enclosing);
    if (!assignment->records.empty()) { child->eventAssignment = std::move(*assignment); }
    uint8_t modes = 0;
    for (const auto* phase : child->visitInput->instructions()) {
        if (!loop->isProperAncestor(phase->elementOp)) { continue; }
        if (!phase->useVec.empty()) { modes |= 1; }
        if (!phase->defVec.empty()) { modes |= 2; }
    }
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
std::optional<CompletionInlineChild::ResourceFootprint> CompletionInlineChild::resourceFootprint() const
{
    if (completion) {
        return completion->usesEventResources() ? std::nullopt : std::optional<ResourceFootprint>(ResourceFootprint{});
    }
    ResourceFootprint result;
    if (eventAssignment) {
        for (const auto& record : eventAssignment->records) {
            for (auto id : record.ids) {
                if (!isNativeEventId(id)) { return std::nullopt; }
                // Final palette renaming retains these disjointness obligations.
                const auto mask = 1u << id;
                result[{record.sourcePipe, record.targetPipe}] |= uint8_t(mask);
            }
        }
    }
    return result;
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
    if (visitPreparation) { return visitRoots; }
    return completion ? completion->originalRoots() : finite->originalInput()->roots();
}
const SyncInput* CompletionInlineChild::input() const
{
    if (visitPreparation) { return visitInput.get(); }
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
LogicalResult appendChildPalettes(const PhysicalAllocationPlan& assignment, const PreparedLogicalPlan& plan,
    MLIRContext* context, uint64_t recordBase, CompletionChildAssembly& assembly)
{
    Builder b(context);
    std::map<uint32_t, uint64_t> arities;
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        if (endpoint.records.size() != 1) { return failure(); }
        auto [found, added] = arities.emplace(endpoint.records.front(), endpoint.memberCoordinates.size());
        if (!added && found->second != endpoint.memberCoordinates.size()) { return failure(); }
    }
    std::map<std::pair<uint32_t, uint32_t>, SmallVector<const PhysicalRecordAllocation*>> directions;
    for (const auto& record : assignment.records) {
        directions[{record.sourcePipe, record.targetPipe}].push_back(&record);
    }
    for (const auto& [direction, members] : directions) {
        std::set<int64_t> slots;
        for (const auto* member : members) {
            for (auto id : member->ids) {
                if (!isNativeEventId(id)) { return failure(); }
                slots.insert(id);
            }
        }
        const uint64_t budget = slots.size();
        SmallVector<int64_t> records, zeros;
        SmallVector<Attribute> rules;
        for (const auto* member : members) {
            auto rule = member->tupleRule.value_or(PhysicalTupleRule{});
            if (!member->tupleRule) {
                if (member->ids.size() != 1) { return failure(); }
                rule.base = member->ids.front();
                auto arity = arities.find(member->record);
                if (arity == arities.end()) { return failure(); }
                rule.coordinateCount = arity->second;
            }
            // Every admitted record table is contiguous. Compressing holes
            // between tables therefore translates its offset, without changing
            // the cyclic/tuple function within that table.
            const auto first = member->ids.front();
            if (rule.base < static_cast<uint64_t>(first)) { return failure(); }
            rule.base = rule.base - first + std::distance(slots.begin(), slots.find(first));
            if (!validPhysicalTupleRule(rule, budget)) { return failure(); }
            SmallVector<Attribute> terms;
            for (const auto& term : rule.terms) {
                terms.push_back(b.getDenseI64ArrayAttr({static_cast<int64_t>(term.coordinate),
                    static_cast<int64_t>(term.stride), static_cast<int64_t>(term.phase),
                    static_cast<int64_t>(term.modulus), static_cast<int64_t>(term.scale)}));
            }
            rules.push_back(b.getDictionaryAttr({
                b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(rule.coordinateCount)),
                b.getNamedAttr("base", b.getI64IntegerAttr(rule.base)),
                b.getNamedAttr("terms", b.getArrayAttr(terms))}));
            records.push_back(member->record + recordBase);
            zeros.push_back(0);
        }
        assembly.allocationGroups.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(direction.first)),
            b.getNamedAttr("target", b.getI64IntegerAttr(direction.second)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(budget)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("tuple_rules", b.getArrayAttr(rules)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
            b.getNamedAttr("forbidden_ids", b.getDenseI64ArrayAttr({}))}));
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
    auto fresh = fragment ? fragment->instantiate() : nullptr;
    if (visitPreparation) {
        auto prepared = visitPreparation();
        if (failed(prepared)) { return failure(); }
        fresh = std::move(*prepared);
    }
    if (!fresh || failed(materializeLogicalFamilies(*fresh))) { return failure(); }
    const auto& plan = *fresh;
    if (!eventAssignment && !eventFree(plan)) { return failure(); }
    auto function = visitPreparation ? visitFunction : finite->originalInput()->function();
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
            function) { return failure(); }
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
        if (visitPreparation && endpoint.kind != LogicalCommandKind::Barrier) {
            endpoint.memberCoordinates.append(enclosingVisits.begin(), enclosingVisits.end());
        }
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
    if (eventAssignment) {
        auto result = appendChildPalettes(*eventAssignment, plan, function.getContext(), recordBase, assembly);
        if (failed(result)) { return failure(); }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
