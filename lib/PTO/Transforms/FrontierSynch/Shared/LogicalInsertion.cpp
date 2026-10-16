// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate prepared endpoints, compact provably exclusive finite alternatives,
// then insert logical commands at actual cuts. Compaction factors only explicit
// coordinate equalities or producer-owned disjoint residues; opaque guards survive.
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PrivateStatePlan.h"
#include "PreparedControlLiveness.h"
#include "ScalarConstructionTrace.h"
#include "ScalarConstruction.h"
#include "FamilyEndpointNormalization.h"
#include "OwnedSelectorSinking.h"
#include "PhysicalMapInputs.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
PreparedLogicalPlan::PreparedLogicalPlan(int64_t planId) : planId(planId) {}
void retainEndpointCoordinates(PreparedLogicalEndpoint& endpoint, ArrayRef<TemplateCoordinate> coordinates)
{
    if (coordinates.empty()) { return; }
    EndpointDomainBox box;
    for (auto coordinate : coordinates) { box.push_back({coordinate.loop.getInductionVar(), 0, coordinate.induction}); }
    endpoint.domain.push_back(std::move(box));
}
void prepareAllocationSupport(PreparedLogicalPlan& plan, AllocationProfile* profile)
{
    if (plan.privateState || plan.completionOwner || failed(normalizeLogicalFamilies(plan))) {
        return;
    }
    auto prepare = std::move(plan.allocationPreparation);
    plan.allocationPreparation = AllocationPreparation{};
    llvm::SaveAndRestore<AllocationProfile*> profiling(plan.allocationProfile, profile);
    prepare(plan);
    prepare.preserveCertificates(plan);
}
LogicalResult normalizeLogicalFamilies(PreparedLogicalPlan& plan)
{
    if (!plan.familyRecipe) {
        return success();
    }
    if (plan.privateState || plan.completionOwner) {
        return failure();
    }
    return normalizeFamilyEndpointPlan(plan);
}
LogicalResult materializeLogicalFamilies(PreparedLogicalPlan& plan)
{
    if (!plan.familyRecipe) { return success(); }
    if (failed(normalizeLogicalFamilies(plan))) {
        return failure();
    }
    if (failed(materializeFamilyEndpointCode(*plan.familyRecipe, plan))) { return failure(); }
    plan.familyRecipe.reset();
    return success();
}
std::shared_ptr<PreparedLogicalPlan> takeRegionalAllocationView(PreparedLogicalPlan& plan, bool retainFamilyRecipe)
{
    if (plan.privateState || plan.completionOwner || plan.physicalOutput || plan.executedOutput ||
        (!retainFamilyRecipe && failed(materializeLogicalFamilies(plan)))) {
        return {};
    }
    auto view = std::make_shared<PreparedLogicalPlan>(plan.planId);
    view->completeInvocation = plan.completeInvocation;
    view->nestedIdentities = plan.nestedIdentities;
    view->independentPieces = plan.independentPieces;
    view->groupedFamilies = plan.groupedFamilies;
    view->familyRecipe = familyRecipeForComposition(plan.familyRecipe);
    view->endpoints = plan.endpoints;
    view->families = plan.families;
    view->mathematicalOwner = plan.mathematicalOwner;
    view->compactBoundingOwner = plan.compactBoundingOwner;
    view->regionalAllocation = std::move(plan.regionalAllocation);
    view->allocationPreparation = std::move(plan.allocationPreparation);
    view->allocationProofs = std::move(plan.allocationProofs);
    plan.allocationPreparation = AllocationPreparation{};
    // Certificates describe the child graph and cannot survive composition.
    plan.allocationCertificate = {};
    return view;
}
PreparedLogicalPlan::~PreparedLogicalPlan()
{
    // Detached blocks can reference each other. Drop uses before destroying any.
    for (auto& stage : preparation) {
        if (stage.code) {
            stage.code->dropAllReferences();
        }
    }
}
Block& PreparedLogicalPlan::addPreparation(Operation* before)
{
    preparation.push_back({before, std::make_unique<Block>()});
    return *preparation.back().code;
}
namespace {
// Called only after preflight. Independent producers can prepare the same
// arithmetic at one original cut. Reuse the first identical operation there;
// neither original payloads nor arithmetic at other cuts is changed.
LogicalResult deduplicatePreparation(PreparedLogicalPlan& plan)
{
    DenseMap<Value, uint64_t> identities;
    SmallVector<Value> values(1);
    auto identify = [&](Value value) {
        auto [entry, added] = identities.try_emplace(value, values.size());
        if (added) { values.push_back(value); }
        return entry->second;
    };
    SmallVector<detail::ScalarInstruction> instructions;
    SmallVector<Operation*> operations, redundant;
    for (auto& stage : plan.preparation) {
        for (auto& operation : *stage.code) {
            auto description = detail::describeScalarInstruction(operation, stage.before, identify);
            if (!description) { continue; }
            instructions.push_back(std::move(*description)); operations.push_back(&operation);
        }
    }
    const auto normalized = detail::deduplicateScalarInstructions(instructions);
    DenseMap<Value, Value> replacements;
    for (auto [old, shared] : normalized.replacements) {
        replacements[values[old]] = values[shared];
        values[old].replaceAllUsesWith(values[shared]);
    }
    for (auto index : normalized.redundant) { redundant.push_back(operations[index]); }
    auto remap = [&](Value& value) {
        if (auto found = replacements.find(value); found != replacements.end()) { value = found->second; }
    };
    // Endpoints hold Values directly, outside the SSA use lists.
    for (auto& endpoint : plan.endpoints) {
        remap(endpoint.guard); remap(endpoint.identity);
        for (auto& box : endpoint.domain) {
            for (auto& constraint : box) { remap(constraint.coordinate); }
        }
        for (auto& member : endpoint.memberCoordinates) { remap(member); }
    }
    // Cut-choice predicates also live outside SSA use lists. Rebuild their
    // immutable evidence before deleting any referenced preparation values.
    for (auto& family : plan.families) {
        for (auto* choices : {&family.sourceChoices, &family.targetChoices}) {
            if (!*choices || !(*choices)->consumerPredicate()) { continue; }
            auto predicate = (*choices)->consumerPredicate();
            remap(predicate);
            if (predicate == (*choices)->consumerPredicate()) { continue; }
            auto updated = qualifyTerminalCleanupCuts((*choices)->cuts(), predicate);
            if (!updated) { return failure(); }
            *choices = std::move(updated);
        }
    }
    for (auto* operation : redundant) { operation->erase(); }
    return success();
}
// This only prunes detached generated arithmetic. Required evidence and opaque
// allocation captures are roots alongside ordinary SSA command operands.
void discardUnusedPreparation(PreparedLogicalPlan& plan)
{
    auto live = livePreparedControl(plan);
    SmallVector<Operation*> dead;
    for (const auto& stage : plan.preparation) {
        for (auto& operation : *stage.code) {
            if (!live.contains(&operation)) { dead.push_back(&operation); }
        }
    }
    // A dead user may precede its operand's deletion across detached blocks.
    for (auto* operation : dead) { operation->dropAllReferences(); }
    for (auto* operation : dead) { operation->erase(); }
}
class FamilyNormalizationEmitter {
public:
    using Scalar = Value;
    explicit FamilyNormalizationEmitter(PreparedLogicalPlan& plan)
        : plan(plan), builder(plan.endpoints.front().before->getContext()),
          construction(builder, plan.endpoints.front().before->getLoc()) {}
    template<class Action> LogicalResult at(Operation* cut, Action action)
    {
        builder.setInsertionPointToEnd(&plan.addPreparation(cut));
        construction.setLocation(cut->getLoc());
        return action();
    }
    std::optional<uint64_t> operations() const { return construction.operations(); }
    Value original(Value value) const { return value; }
    bool less(Value a, Value b) const
    {
        return std::less<const void*>{}(a.getAsOpaquePointer(), b.getAsOpaquePointer());
    }
    detail::ScalarGuardPattern<Value> pattern(Value value) const
    {
        auto* operation = value.getDefiningOp();
        if (!operation) { return {}; }
        SmallVector<Value> operands(operation->getOperands());
        return detail::scalarGuardPattern<Value>(operation->getName().getStringRef(),
            operation->getPropertiesAsAttribute(), operands);
    }
    Value select(Value condition, Value yes, Value no)
    {
        return construction.select(
            construction.input(condition), construction.input(yes), construction.input(no)).value;
    }
    Value lor(Value a, Value b)
    {
        return construction.binary<arith::OrIOp>(construction.input(a), construction.input(b)).value;
    }
    Value land(Value a, Value b)
    {
        return construction.binary<arith::AndIOp>(construction.input(a), construction.input(b)).value;
    }
    std::pair<Value, Value> selection(const FamilyExpressionRecipe& recipe, bool member)
    {
        if (!member) { return {detail::constructFamilyPresence(construction, recipe).value, {}}; }
        auto result = detail::constructFamilyExpressions(construction, recipe);
        return {result.present.value, result.member.value};
    }
private:
    PreparedLogicalPlan& plan;
    OpBuilder builder;
    detail::ScalarConstruction construction;
};
LogicalResult normalizePreparedEndpoints(PreparedLogicalPlan& plan)
{
    if (plan.endpoints.empty()) { return success(); }
    FamilyNormalizationEmitter consumer(plan);
    // Preserve a supplied finite map before merging residual guard intervals.
    if (failed(detail::coalesceEndpointAlternatives(plan, consumer))) { return failure(); }
    auto result = detail::coalesceDomainAlternatives(plan, consumer, bool(plan.familyProtocol));
    if (auto constructed = consumer.operations()) { plan.familyWork.compactionScalars = *constructed; }
    return result;
}
bool concretePipe(uint32_t pipe)
{
    return pipe <= static_cast<uint32_t>(PIPE::PIPE_FIX) && pipe != static_cast<uint32_t>(PIPE::PIPE_ALL);
}
class Availability {
public:
    explicit Availability(func::FuncOp function) : function(function), dominance(function) {}
    bool cut(Operation* before)
    {
        if (!before || !before->getBlock()) {
            return false;
        }
        for (auto* parent = before->getParentOp(); parent; parent = parent->getParentOp()) {
            if (parent == function) {
                return true;
            }
            if (parent->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
                return false;
            }
        }
        return false;
    }
    bool value(Value value, Operation* before)
    {
        if (!value) {
            return false;
        }
        auto found = prepared.find(value);
        if (found != prepared.end()) {
            return found->second == before || dominance.properlyDominates(found->second, before);
        }
        // Never ask DominanceInfo about an unregistered detached definition.
        auto* region = value.getParentRegion();
        return region && region->getParentOp() &&
            (region->getParentOp() == function || function->isProperAncestor(region->getParentOp())) &&
            dominance.properlyDominates(value, before);
    }
    void define(Value value, Operation* before) { prepared[value] = before; }
private:
    func::FuncOp function;
    DominanceInfo dominance;
    llvm::DenseMap<Value, Operation*> prepared;
};
LogicalResult validatePieces(func::FuncOp function, const PreparedLogicalPlan& plan,
                             const std::map<uint32_t, const EndpointFamily*>& owners)
{
    using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::map<Namespace, uint32_t> namespaces;
    for (const auto& [record, family] : owners) {
        namespaces.try_emplace({family->sourcePipe, family->targetPipe, family->displacement}, record);
    }
    std::map<std::pair<uint32_t, unsigned>, std::set<Operation*>> coverage;
    std::map<uint32_t, std::size_t> arities;
    std::set<int64_t> pieces;
    for (const auto& endpoint : plan.endpoints) {
        const bool local = endpoint.kind == LogicalCommandKind::Barrier;
        const bool publish = endpoint.kind == LogicalCommandKind::Set;
        if (endpoint.records.empty() || endpoint.piece < 0 || !pieces.insert(endpoint.piece).second ||
            (local ? !endpoint.memberCoordinates.empty() :
             (plan.nestedIdentities ? endpoint.memberCoordinates.empty() : endpoint.memberCoordinates.size() != 1))) {
            return function.emitError("invalid endpoint-piece identity or member selector");
        }
        for (auto record : endpoint.records) {
            auto found = owners.find(record);
            if (found == owners.end()) {
                return function.emitError("endpoint piece has an unknown original record");
            }
            auto [arity, fresh] = arities.emplace(record, endpoint.memberCoordinates.size());
            if (!fresh && arity->second != endpoint.memberCoordinates.size()) {
                return function.emitError("matching endpoint pieces have different coordinate arities");
            }
            const auto& family = *found->second;
            if (publish && family.targetChoices && !family.targetChoices->loop() &&
                !qualifyTerminalCleanupCuts({family.sourceCut, family.targetChoices->cuts().back()}, endpoint.guard)) {
                return function.emitError("terminal cleanup requires one selected publication per invocation");
            }
            const auto key = Namespace{family.sourcePipe, family.targetPipe, family.displacement};
            const auto& choices = publish ? family.sourceChoices : family.targetChoices;
            const auto primary = publish ? family.sourceCut.before : family.targetCut.before;
            const bool validCut = choices ? llvm::any_of(choices->cuts(), [&](const auto& cut) {
                return cut.before == endpoint.before;
            }) : endpoint.before == primary;
            auto& covered = coverage[{record, static_cast<unsigned>(endpoint.kind)}];
            if (!covered.insert(endpoint.before).second || family.local != local ||
                endpoint.sourcePipe != family.sourcePipe || endpoint.targetPipe != family.targetPipe ||
                !validCut || endpoint.record != namespaces.at(key)) {
                return function.emitError("endpoint piece does not match its original record");
            }
        }
    }
    for (const auto& [record, family] : owners) {
        for (unsigned kind = 0; kind < 3; ++kind) {
            const auto& choices = kind == 0 ? family->sourceChoices : family->targetChoices;
            const bool required = family->local ? kind == 1 : kind != 1;
            const auto expected = required ? (choices ? choices->cuts().size() : 1U) : 0U;
            if (coverage[{record, kind}].size() != expected) {
                return function.emitError("endpoint-piece partition is missing a record side");
            }
        }
    }
    return success();
}
LogicalResult preflight(func::FuncOp function, const PreparedLogicalPlan& plan)
{
    if (plan.planId < 0 || !RegisteredOperationName::lookup("pto.logical_set", function.getContext()) ||
        !RegisteredOperationName::lookup("pto.logical_wait", function.getContext())) {
        return function.emitError("logical insertion requires a nonnegative namespace and registered logical commands");
    }
    if (plan.nestedIdentities && !plan.independentPieces) {
        return function.emitError("nested matching requires independent endpoint pieces");
    }
    bool collision = false;
    function.walk([&](Operation* operation) {
        if (isa<LogicalSetOp, LogicalWaitOp>(operation)) {
            auto id = operation->getAttrOfType<IntegerAttr>("plan_id");
            collision |= id && id.getInt() == plan.planId;
        }
    });
    if (collision) {
        return function.emitError("logical insertion namespace is already in use");
    }
    if (function->hasAttr("pto.deferred_family") || function.walk([](LogicalFamilyOp) {
            return WalkResult::interrupt();
        }).wasInterrupted()) {
        return function.emitError("existing deferred logical families require allocation before new insertion");
    }
    if (function->hasAttr("pto.endpoint_families")) {
        return function.emitError("logical insertion cannot replace existing endpoint-family provenance");
    }
    if (!plan.physicalOutput &&
        llvm::any_of(plan.endpoints, [](const auto& endpoint) { return bool(endpoint.physicalMapping); })) {
        return function.emitError("endpoint physical maps require their owning physical-output request");
    }
    Availability available(function);
    for (const auto& stage : plan.preparation) {
        if (!available.cut(stage.before) || !stage.code || stage.code->getParent() || stage.code->getNumArguments()) {
            return function.emitError("invalid detached logical preparation block or cut");
        }
        for (auto& operation : *stage.code) {
            if (!operation.getName().isRegistered() || operation.getName().getDialectNamespace() != "arith" ||
                operation.getNumRegions() || operation.getNumSuccessors() || failed(verify(&operation, false))) {
                return function.emitError("logical preparation requires valid region-free arithmetic");
            }
            for (auto operand : operation.getOperands()) {
                if (!available.value(operand, stage.before)) {
                    return function.emitError("logical preparation operand is unavailable at its cut");
                }
            }
            for (auto result : operation.getResults()) {
                available.define(result, stage.before);
            }
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        const bool barrier = endpoint.kind == LogicalCommandKind::Barrier;
        if (barrier && endpoint.sourcePipe == static_cast<uint32_t>(PIPE::PIPE_S)) {
            return function.emitError(
                "invalid logical plan: PIPE_S barrier is forbidden; "
                "same-scalar storage hazards must be hardware protected before reduction");
        }
        const bool needsIdentity = !plan.executedOutput &&
            (!endpoint.physicalMapping || physicalMapNeedsCoordinate(*endpoint.physicalMapping, 0));
        const bool kind =
            barrier || endpoint.kind == LogicalCommandKind::Set || endpoint.kind == LogicalCommandKind::Wait;
        if (!kind || !available.cut(endpoint.before) || endpoint.record < 0 || !concretePipe(endpoint.sourcePipe) ||
            !concretePipe(endpoint.targetPipe) || (barrier != (endpoint.sourcePipe == endpoint.targetPipe)) ||
            !available.value(endpoint.guard, endpoint.before) || !endpoint.guard.getType().isInteger(1) ||
            (!barrier && (needsIdentity || endpoint.identity) &&
             (!available.value(endpoint.identity, endpoint.before) || !endpoint.identity.getType().isIndex())) ||
            (barrier && endpoint.identity)) {
            return function.emitError("invalid or unavailable prepared logical endpoint");
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        for (const auto& box : endpoint.domain) {
            for (const auto& constraint : box) {
                if (!available.value(constraint.coordinate, endpoint.before) ||
                    !constraint.coordinate.getType().isIntOrIndex() || constraint.modulus > INT64_MAX ||
                    (constraint.modulus &&
                     (constraint.value < 0 || static_cast<uint64_t>(constraint.value) >= constraint.modulus))) {
                    return function.emitError("invalid or unavailable endpoint domain constraint");
                }
            }
        }
        if (!validPhysicalFamilyEndpoint(plan, endpoint) ||
            (plan.executedOutput && !validExecutedFamilyEndpoint(plan, endpoint))) {
            return function.emitError("physical endpoint mapping does not belong to its certified family output");
        }
        for (auto [index, member] : llvm::enumerate(endpoint.memberCoordinates)) {
            if (!member && plan.executedOutput) { continue; }
            if (!member && plan.physicalOutput && endpoint.physicalMapping &&
                (index ? !physicalMapNeedsCoordinate(*endpoint.physicalMapping, index) :
                         !physicalMapRequiresMember(*endpoint.physicalMapping))) {
                continue;
            }
            if (endpoint.kind == LogicalCommandKind::Barrier || !member || !member.getType().isIndex() ||
                !available.value(member, endpoint.before)) {
                return function.emitError("invalid or unavailable endpoint family member");
            }
        }
    }
    if (plan.groupedFamilies && plan.families.empty() && !plan.endpoints.empty()) {
        return function.emitError("grouped logical endpoints require family provenance");
    }
    if (!plan.families.empty()) {
        std::map<int64_t, const EndpointFamily*> owners;
        std::map<uint32_t, const EndpointFamily*> recordOwners;
        std::set<uint32_t> originalRecords;
        std::set<uint32_t> familyIds;
        for (const auto& family : plan.families) {
            if (family.members.empty() || !familyIds.insert(family.id).second ||
                !available.cut(family.sourceCut.before) || !available.cut(family.targetCut.before) ||
                family.sourceCut.block != family.sourceCut.before->getBlock() ||
                family.targetCut.block != family.targetCut.before->getBlock() ||
                family.local != (family.sourcePipe == family.targetPipe)) {
                return function.emitError("invalid endpoint-family cuts or identity");
            }
            for (bool target : {false, true}) {
                const auto& choices = target ? family.targetChoices : family.sourceChoices;
                if (!choices) { continue; }
                const auto primary = target ? family.targetCut : family.sourceCut;
                if (!plan.independentPieces || (!choices->loop() && (!target || family.local)) ||
                    choices->cuts().size() < 2 ||
                    choices->cuts().front().before != primary.before ||
                    !(choices->loop() ? qualifyEndpointCutChoices(choices->loop(), choices->cuts()) :
                      qualifyTerminalCleanupCuts(choices->cuts(), choices->consumerPredicate()))) {
                    return function.emitError("invalid structural endpoint alternatives");
                }
                for (const auto& cut : choices->cuts()) {
                    if (!available.cut(cut.before) || cut.block != cut.before->getBlock()) {
                        return function.emitError("unavailable structural endpoint alternative");
                    }
                }
            }
            if (plan.groupedFamilies) {
                owners.emplace(family.id, &family);
            }
            for (const auto& member : family.members) {
                recordOwners.emplace(member.record, &family);
                if (!originalRecords.insert(member.record).second) {
                    return function.emitError("duplicate endpoint-family member");
                }
                if (!plan.groupedFamilies) {
                    owners.emplace(member.record, &family);
                }
                for (auto side : {false, true}) {
                    const auto& tuple = side ? member.targetCoordinates : member.sourceCoordinates;
                    const auto& choices = side ? family.targetChoices : family.sourceChoices;
                    SmallVector<TemplateEndpointCut> cuts;
                    if (choices) { cuts.append(choices->cuts().begin(), choices->cuts().end()); }
                    else { cuts.push_back(side ? family.targetCut : family.sourceCut); }
                    for (const auto& cut : cuts) {
                        for (auto coordinate : tuple) {
                            if (!coordinate.loop || !coordinate.loop->isProperAncestor(cut.before) ||
                                !available.value(coordinate.loop.getInductionVar(), cut.before)) {
                                return function.emitError("unavailable endpoint-family coordinate");
                            }
                        }
                    }
                }
            }
        }
        if (plan.independentPieces) {
            return validatePieces(function, plan, recordOwners);
        }
        std::map<int64_t, unsigned> kinds;
        for (const auto& endpoint : plan.endpoints) {
            auto found = owners.find(endpoint.record);
            if (found == owners.end()) {
                return function.emitError("logical endpoint has no family member");
            }
            const auto& family = *found->second;
            const bool source = endpoint.kind == LogicalCommandKind::Set;
            unsigned kind = 1U << static_cast<unsigned>(endpoint.kind);
            if ((kinds[endpoint.record] & kind) || endpoint.sourcePipe != family.sourcePipe ||
                endpoint.targetPipe != family.targetPipe ||
                endpoint.before != (source ? family.sourceCut.before : family.targetCut.before) ||
                family.local != (endpoint.kind == LogicalCommandKind::Barrier) ||
                endpoint.memberCoordinates.size() !=
                    (plan.groupedFamilies && !family.local && family.members.size() > 1 ? 1U : 0U)) {
                return function.emitError("endpoint-family member does not match its endpoint");
            }
            kinds[endpoint.record] |= kind;
        }
        for (const auto& [record, family] : owners) {
            if (kinds[record] != (family->local ? 2U : 5U)) {
                return function.emitError("endpoint-family member is missing an endpoint");
            }
        }
    }
    return success();
}
// Persist coordinate provenance before materializing guards. Stable loop and cut
// numbers are local to this plan; allocation never needs compiler pointer values.
void serializeFamilies(func::FuncOp function, const PreparedLogicalPlan& plan,
                       const llvm::MapVector<Operation*, SmallVector<const PreparedLogicalEndpoint*>>& cuts)
{
    if (plan.families.empty()) {
        return;
    }
    Builder builder(function.getContext());
    llvm::DenseMap<Operation*, int64_t> loopIds, cutIds;
    int64_t nextCut = 0;
    for (const auto& entry : cuts) {
        cutIds[entry.first] = nextCut++;
    }
    auto loopId = [&](scf::ForOp loop) {
        auto [entry, added] = loopIds.try_emplace(loop.getOperation(), loopIds.size());
        if (added) { loop->setAttr("pto.family_loop", builder.getI64IntegerAttr(entry->second)); }
        return entry->second;
    };
    auto coordinates = [&](ArrayRef<TemplateCoordinate> tuple) {
        SmallVector<Attribute> values;
        for (auto coordinate : tuple) {
            values.push_back(builder.getDenseI64ArrayAttr({loopId(coordinate.loop), coordinate.induction}));
        }
        return builder.getArrayAttr(values);
    };
    auto choices = [&](const std::shared_ptr<const EndpointCutChoices>& proof) {
        SmallVector<int64_t> ids;
        if (proof) {
            ids.push_back(proof->loop() ? loopId(proof->loop()) : -1);
            for (const auto& cut : proof->cuts()) { ids.push_back(cutIds.lookup(cut.before)); }
        }
        return builder.getDenseI64ArrayAttr(ids);
    };
    SmallVector<Attribute> families;
    for (const auto& family : plan.families) {
        SmallVector<Attribute> members;
        for (const auto& member : family.members) {
            members.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("record", builder.getI64IntegerAttr(member.record)),
                builder.getNamedAttr("source", coordinates(member.sourceCoordinates)),
                builder.getNamedAttr("target", coordinates(member.targetCoordinates))}));
        }
        families.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("id", builder.getI64IntegerAttr(family.id)),
            builder.getNamedAttr("local", builder.getBoolAttr(family.local)),
            builder.getNamedAttr("source_pipe", builder.getI64IntegerAttr(family.sourcePipe)),
            builder.getNamedAttr("target_pipe", builder.getI64IntegerAttr(family.targetPipe)),
            builder.getNamedAttr("displacement", builder.getI64IntegerAttr(family.displacement)),
            builder.getNamedAttr("source_order", builder.getI64IntegerAttr(family.sourceOrder)),
            builder.getNamedAttr("target_order", builder.getI64IntegerAttr(family.targetOrder)),
            builder.getNamedAttr("source_cut", builder.getI64IntegerAttr(
                family.local ? -1 : cutIds.lookup(family.sourceCut.before))),
            builder.getNamedAttr("target_cut", builder.getI64IntegerAttr(cutIds.lookup(family.targetCut.before))),
            builder.getNamedAttr("members", builder.getArrayAttr(members)),
            builder.getNamedAttr("source_choices", choices(family.sourceChoices)),
            builder.getNamedAttr("target_choices", choices(family.targetChoices))}));
    }
    SmallVector<Attribute> pieces;
    if (plan.independentPieces) {
        std::map<uint32_t, uint64_t> displacements;
        for (const auto& family : plan.families) {
            for (const auto& member : family.members) {
                displacements.emplace(member.record, family.displacement);
            }
        }
        for (const auto& endpoint : plan.endpoints) {
            SmallVector<int64_t> records(endpoint.records.begin(), endpoint.records.end());
            pieces.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("id", builder.getI64IntegerAttr(endpoint.piece)),
                builder.getNamedAttr("family", builder.getI64IntegerAttr(endpoint.record)),
                builder.getNamedAttr("kind", builder.getI64IntegerAttr(static_cast<unsigned>(endpoint.kind))),
                builder.getNamedAttr("cut", builder.getI64IntegerAttr(cutIds.lookup(endpoint.before))),
                builder.getNamedAttr("source_pipe", builder.getI64IntegerAttr(endpoint.sourcePipe)),
                builder.getNamedAttr("target_pipe", builder.getI64IntegerAttr(endpoint.targetPipe)),
                builder.getNamedAttr("displacement",
                                     builder.getI64IntegerAttr(displacements.at(endpoint.records.front()))),
                builder.getNamedAttr("records", builder.getDenseI64ArrayAttr(records))}));
        }
    }
    function->setAttr(
        "pto.endpoint_families",
        builder.getDictionaryAttr(
            {builder.getNamedAttr(
                 "version",
                 builder.getI64IntegerAttr(
                     plan.nestedIdentities ? 4 : (plan.independentPieces ? 3 : (plan.groupedFamilies ? 2 : 1)))),
             builder.getNamedAttr("plan", builder.getI64IntegerAttr(plan.planId)),
             builder.getNamedAttr("protocol_version", builder.getI64IntegerAttr(plan.familyProtocol ? 1 : 0)),
             builder.getNamedAttr("families", builder.getArrayAttr(families)),
             builder.getNamedAttr("pieces", builder.getArrayAttr(pieces))}));
}
void command(OpBuilder& builder, const PreparedLogicalEndpoint& endpoint, int64_t planId, uint64_t cutId,
             FamilyMaterializationWork& work, detail::OwnedSelectorSinking& selectors)
{
    auto location = endpoint.before->getLoc();
    auto branch = builder.create<scf::IfOp>(location, endpoint.guard, false);
    selectors.guard(branch, endpoint.before);
    branch->setAttr("pto.endpoint_cut", builder.getI64IntegerAttr(cutId));
    OpBuilder::InsertionGuard restore(builder);
    builder.setInsertionPointToStart(branch.thenBlock());
    if (endpoint.physicalMapping) {
        SmallVector<Value> tuple{endpoint.identity};
        tuple.append(endpoint.memberCoordinates.begin() + 1, endpoint.memberCoordinates.end());
        const auto before = branch.thenBlock()->getOperations().size();
        auto id = emitPhysicalEndpointMap(
            builder, location, *endpoint.physicalMapping, tuple, endpoint.memberCoordinates.front());
        accumulateCost(work.mappingScalars, branch.thenBlock()->getOperations().size() - before);
        assert(id && "validated immutable physical map must produce an event ID");
        emitPhysicalCommand(
            builder, location, endpoint.kind == LogicalCommandKind::Set, endpoint.sourcePipe, endpoint.targetPipe, id);
        return;
    }
    OperationState state(location, endpoint.kind == LogicalCommandKind::Set ? "pto.logical_set" : "pto.logical_wait");
    state.addOperands(endpoint.identity);
    state.addOperands(endpoint.memberCoordinates);
    state.addAttribute("src_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(endpoint.sourcePipe)));
    state.addAttribute("dst_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(endpoint.targetPipe)));
    state.addAttribute("plan_id", builder.getI64IntegerAttr(planId));
    state.addAttribute("record_id", builder.getI64IntegerAttr(endpoint.record));
    if (endpoint.piece >= 0) {
        state.addAttribute("pto.endpoint_piece", builder.getI64IntegerAttr(endpoint.piece));
    }
    builder.create(state);
}
void emitCut(OpBuilder& builder, ArrayRef<const PreparedLogicalEndpoint*> endpoints, int64_t planId, uint64_t cutId,
             FamilyMaterializationWork& work, detail::OwnedSelectorSinking& selectors)
{
    builder.setInsertionPoint(endpoints.front()->before);
    auto location = endpoints.front()->before->getLoc();
    const auto barriers = detail::combineBarrierGuards(endpoints, [&](Value a, Value b) {
        accumulateCost(work.barrierScalars, 1);
        return Value(builder.create<arith::OrIOp>(location, a, b));
    });
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind == LogicalCommandKind::Set) {
            command(builder, *endpoint, planId, cutId, work, selectors);
        }
    }
    for (auto [pipe, guard] : barriers) {
        auto branch = builder.create<scf::IfOp>(location, guard, false);
        branch->setAttr("pto.endpoint_cut", builder.getI64IntegerAttr(cutId));
        OpBuilder::InsertionGuard restore(builder);
        builder.setInsertionPointToStart(branch.thenBlock());
        builder.create<BarrierOp>(location, PipeAttr::get(builder.getContext(), static_cast<PIPE>(pipe)));
    }
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind == LogicalCommandKind::Wait) {
            command(builder, *endpoint, planId, cutId, work, selectors);
        }
    }
}
} // namespace
LogicalResult insertLogicalSynchronization(func::FuncOp function, PreparedLogicalPlan& plan)
{
    if (!function) {
        return failure();
    }
    if (plan.deferredOutput && !isCertifiedEmptyFamilyPlan(function, plan)) {
        return insertDeferredFamilyOutput(function, plan);
    }
    // The complete count service owns this same recipe-only input boundary.
    // Concrete preparation and opaque allocation captures retain their current
    // positions rather than acquire uncounted result-bearing control.
    const bool sharedSelectors = plan.familyRecipe && plan.endpoints.empty() && plan.preparation.empty() &&
        !plan.allocationPreparation;
    if (!validFamilyEndpointProtocol(plan) || !validPhysicalFamilyOutput(plan) || !validExecutedFamilyOutput(plan) ||
        failed(materializeLogicalFamilies(plan))) {
        return failure();
    }
    if (plan.completionOwner || plan.privateState) {
        auto inserted = plan.completionOwner ? plan.completionOwner->insert(function, plan) :
            plan.privateState->insert(function, plan);
        if (succeeded(inserted)) { profileLogicalControl(function, plan, "inserted"); }
        return inserted;
    }
    function.getContext()->getOrLoadDialect<PTODialect>();
    function.getContext()->getOrLoadDialect<arith::ArithDialect>();
    function.getContext()->getOrLoadDialect<scf::SCFDialect>();
    if (failed(preflight(function, plan))) {
        return failure();
    }
    if (plan.allocationCertificate &&
        (function->hasAttr(CyclicAllocationAttr) || function->hasAttr(FiniteAllocationAttr))) {
        return function.emitError("logical insertion cannot replace an existing allocation certificate");
    }
    // Distinguish deferred construction from code eliminated by normalization.
    profileLogicalControl(function, plan, "materialized");
    plan.familyWork.preparationScalars = plan.familyWork.compactionScalars = 0;
    plan.familyWork.mappingScalars = plan.familyWork.barrierScalars = 0;
    for (const auto& stage : plan.preparation) {
        accumulateCost(plan.familyWork.preparationScalars, stage.code->getOperations().size());
    }
    if (failed(deduplicatePreparation(plan))) {
        return function.emitError("preparation deduplication invalidated terminal cleanup evidence");
    }
    // Retained numerical families were grouped before allocation. Concrete and
    // contextual compatibility contributions still use the trace-preserving
    // compactor. Physical contributions merge only with equal complete maps.
    if (!plan.executedOutput) {
        if (failed(normalizePreparedEndpoints(plan))) { return failure(); }
    }
    if (failed(preflight(function, plan))) { return failure(); }
    discardUnusedPreparation(plan);
    profileLogicalControl(function, plan, "normalized");
    detail::OwnedSelectorSinking selectors;
    for (const auto& family : plan.families) {
        for (const auto& choices : {family.sourceChoices, family.targetChoices}) {
            if (choices) { selectors.preserve(choices->consumerPredicate()); }
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        for (const auto& box : endpoint.domain) {
            for (const auto& constraint : box) { selectors.preserve(constraint.coordinate); }
        }
    }
    for (auto& stage : plan.preparation) {
        if (!plan.executedOutput) { selectors.retain(*stage.code, stage.before); }
        auto* target = stage.before->getBlock();
        target->getOperations().splice(stage.before->getIterator(), stage.code->getOperations());
    }
    llvm::MapVector<Operation*, SmallVector<const PreparedLogicalEndpoint*>> cuts;
    for (const auto& endpoint : plan.endpoints) {
        cuts[endpoint.before].push_back(&endpoint);
    }
    if (!plan.physicalOutput && !plan.executedOutput) {
        serializeFamilies(function, plan, cuts);
    }
    OpBuilder builder(function.getContext());
    uint64_t cutId = 0;
    if (plan.executedOutput) {
        emitExecutedFamilyOutput(function, plan);
    } else {
        for (const auto& cut : cuts) {
            emitCut(builder, cut.second, plan.planId, cutId++, plan.familyWork, selectors);
        }
        selectors.run(sharedSelectors);
        accumulateCost(plan.familyWork.compactionScalars, selectors.constructedScalars());
    }
    if (plan.completeInvocation) {
        function.walk([&](func::ReturnOp ret) {
            auto existing = dyn_cast_or_null<BarrierOp>(ret->getPrevNode());
            if (existing && existing.getPipe() == PipeAttr::get(function.getContext(), PIPE::PIPE_ALL)) { return; }
            builder.setInsertionPoint(ret);
            builder.create<BarrierOp>(ret.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_ALL));
        });
    }
    if (plan.recognitionReport) {
        function->setAttr("pto.frontier_recognition", plan.recognitionReport);
    }
    if (plan.allocationCertificate && !plan.physicalOutput && !plan.executedOutput) {
        auto kind = plan.allocationCertificate.getAs<StringAttr>("kind");
        function->setAttr(kind && kind.getValue() == "finite" ? FiniteAllocationAttr : CyclicAllocationAttr,
                          plan.allocationCertificate);
    }
    profileLogicalControl(function, plan, "inserted");
    return success();
}
} // namespace mlir::pto::frontiersynch
