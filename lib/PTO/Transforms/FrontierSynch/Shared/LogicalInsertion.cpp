// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate prepared endpoints, compact provably exclusive finite alternatives,
// then insert logical commands at actual cuts. Compaction factors only explicit
// coordinate equalities; opaque predicates and coincident command order survive.
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PrivateStatePlan.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
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
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
PreparedLogicalPlan::PreparedLogicalPlan(int64_t planId) : planId(planId) {}
void prepareAllocationSupport(PreparedLogicalPlan& plan)
{
    if (plan.privateState || plan.completionOwner) { return; }
    auto prepare = std::move(plan.allocationPreparation);
    plan.allocationPreparation = {};
    if (prepare) { prepare(plan); }
}
std::shared_ptr<PreparedLogicalPlan> takeRegionalAllocationView(PreparedLogicalPlan& plan)
{
    if (plan.privateState || plan.completionOwner) { return {}; }
    auto view = std::make_shared<PreparedLogicalPlan>(plan.planId);
    view->completeInvocation = plan.completeInvocation;
    view->nestedIdentities = plan.nestedIdentities;
    view->independentPieces = plan.independentPieces;
    view->groupedFamilies = plan.groupedFamilies;
    view->endpoints = plan.endpoints;
    view->families = plan.families;
    view->mathematicalOwner = plan.mathematicalOwner;
    view->compactBoundingOwner = plan.compactBoundingOwner;
    view->regionalAllocation = std::move(plan.regionalAllocation);
    view->allocationPreparation = std::move(plan.allocationPreparation);
    plan.allocationPreparation = {};
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
void deduplicatePreparation(PreparedLogicalPlan& plan)
{
    using Buckets = DenseMap<std::size_t, SmallVector<Operation*>>;
    DenseMap<Operation*, Buckets> cuts;
    DenseMap<Value, Value> replacements;
    for (auto& stage : plan.preparation) {
        auto& buckets = cuts[stage.before];
        for (auto& operation : llvm::make_early_inc_range(*stage.code)) {
            if (!isMemoryEffectFree(&operation)) { continue; }
            auto hash = static_cast<std::size_t>(OperationEquivalence::computeHash(&operation,
                OperationEquivalence::directHashValue, OperationEquivalence::ignoreHashValue,
                OperationEquivalence::IgnoreLocations));
            auto& candidates = buckets[hash];
            auto found = llvm::find_if(candidates, [&](Operation* candidate) {
                return OperationEquivalence::isEquivalentTo(candidate, &operation,
                    OperationEquivalence::IgnoreLocations);
            });
            if (found == candidates.end()) { candidates.push_back(&operation); continue; }
            for (auto [old, shared] : llvm::zip(operation.getResults(), (*found)->getResults())) {
                replacements[old] = shared;
                old.replaceAllUsesWith(shared);
            }
            operation.erase();
        }
    }
    auto remap = [&](Value& value) {
        if (auto found = replacements.find(value); found != replacements.end()) { value = found->second; }
    };
    // Endpoints hold Values directly, outside the SSA use lists.
    for (auto& endpoint : plan.endpoints) {
        remap(endpoint.guard); remap(endpoint.identity);
        for (auto& member : endpoint.memberCoordinates) { remap(member); }
    }
}
// Factor only explicit conjunctions and equality tests on certified induction
// coordinates. Other predicates remain opaque, evaluated at their original cut.
std::optional<SmallVector<Value>> residualGuard(Value guard, ArrayRef<TemplateCoordinate> coordinates)
{
    SmallVector<Value> pending{guard}, residual;
    llvm::DenseSet<Value> seen, bound;
    while (!pending.empty()) {
        auto value = pending.pop_back_val();
        if (!seen.insert(value).second || matchPattern(value, m_One())) { continue; }
        if (auto both = value.getDefiningOp<arith::AndIOp>()) {
            pending.push_back(both.getLhs()); pending.push_back(both.getRhs()); continue;
        }
        if (auto choice = value.getDefiningOp<arith::SelectOp>();
            choice && matchPattern(choice.getFalseValue(), m_Zero())) {
            pending.push_back(choice.getCondition()); pending.push_back(choice.getTrueValue()); continue;
        }
        bool coordinateTest = false;
        if (auto equal = value.getDefiningOp<arith::CmpIOp>();
            equal && equal.getPredicate() == arith::CmpIPredicate::eq) {
            for (auto coordinate : coordinates) {
                Value induction = coordinate.loop.getInductionVar();
                Value other = equal.getLhs() == induction ? equal.getRhs() :
                    equal.getRhs() == induction ? equal.getLhs() : Value{};
                APInt constant;
                if (other && matchPattern(other, m_ConstantInt(&constant)) && constant.isSignedIntN(64) &&
                    constant.getSExtValue() == coordinate.induction) {
                    bound.insert(induction); coordinateTest = true; break;
                }
            }
        }
        if (!coordinateTest) { residual.push_back(value); }
    }
    if (bound.size() != coordinates.size()) { return std::nullopt; }
    llvm::sort(residual, [](Value a, Value b) {
        return std::less<const void*>{}(a.getAsOpaquePointer(), b.getAsOpaquePointer());
    });
    return residual;
}
// Combine consecutive command alternatives only when exact finite coordinates
// prove mutual exclusion. Certificates retain every record; labels remain
// provenance, and simultaneously active commands are never collapsed.
LogicalResult coalesceEndpointAlternatives(PreparedLogicalPlan& plan)
{
    if (!plan.independentPieces || plan.endpoints.empty()) { return success(); }
    auto* cut = plan.endpoints.front().before;
    const auto width = DataLayout::closest(cut).getTypeSizeInBits(IndexType::get(cut->getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) { return success(); }
    struct Domain {
        SmallVector<Operation*> axes;
        std::set<std::vector<int64_t>> points;
        SmallVector<Value> residual;
    };
    std::map<uint32_t, std::pair<const EndpointFamily*, const EndpointFamilyMember*>> members;
    for (const auto& family : plan.families) {
        for (const auto& member : family.members) { members.emplace(member.record, std::make_pair(&family, &member)); }
    }
    auto domain = [&](const PreparedLogicalEndpoint& endpoint) -> std::optional<Domain> {
        Domain result;
        // Already grouped domains keep their existing compact representation.
        if (endpoint.records.size() != 1) { return std::nullopt; }
        for (auto record : endpoint.records) {
            const auto [family, member] = members.at(record);
            if (family->sourceChoices || family->targetChoices) { return std::nullopt; }
            const auto& coordinates = endpoint.kind == LogicalCommandKind::Set ?
                member->sourceCoordinates : member->targetCoordinates;
            if (coordinates.empty()) { return std::nullopt; }
            auto remaining = residualGuard(endpoint.guard, coordinates);
            if (!remaining) { return std::nullopt; }
            result.residual = std::move(*remaining);
            SmallVector<Operation*> axes;
            std::vector<int64_t> point;
            for (auto coordinate : coordinates) {
                axes.push_back(coordinate.loop.getOperation()); point.push_back(coordinate.induction);
            }
            if (result.points.empty()) { result.axes = axes; }
            if (result.axes != axes || !result.points.insert(std::move(point)).second) { return std::nullopt; }
        }
        return result.points.empty() ? std::nullopt : std::optional<Domain>(std::move(result));
    };
    using CutKind = std::pair<Operation*, LogicalCommandKind>;
    std::map<CutKind, std::size_t> previous;
    std::map<std::size_t, Domain> domains;
    std::vector<bool> removed(plan.endpoints.size(), false), combined(plan.endpoints.size(), false);
    for (std::size_t i = 0; i < plan.endpoints.size(); ++i) {
        auto& endpoint = plan.endpoints[i];
        if (endpoint.kind == LogicalCommandKind::Barrier || endpoint.memberCoordinates.empty()) { continue; }
        auto current = domain(endpoint);
        auto [entry, fresh] = previous.emplace(CutKind{endpoint.before, endpoint.kind}, i);
        const auto old = entry->second;
        auto known = domains.find(old);
        auto& preceding = plan.endpoints[old];
        const bool compatible = !fresh && current && known != domains.end() &&
            preceding.record == endpoint.record && preceding.identity == endpoint.identity &&
            preceding.sourcePipe == endpoint.sourcePipe && preceding.targetPipe == endpoint.targetPipe &&
            ArrayRef<Value>(preceding.memberCoordinates).drop_front() ==
                ArrayRef<Value>(endpoint.memberCoordinates).drop_front() &&
            known->second.axes == current->axes && known->second.residual == current->residual &&
            llvm::none_of(current->points, [&](const auto& point) { return known->second.points.count(point); });
        if (!compatible) {
            entry->second = i;
            if (current) { domains.emplace(i, std::move(*current)); }
            continue;
        }
        preceding.records.append(endpoint.records.begin(), endpoint.records.end());
        known->second.points.insert(current->points.begin(), current->points.end());
        removed[i] = true; combined[old] = true;
    }
    std::vector<PreparedLogicalEndpoint> retained;
    for (std::size_t i = 0; i < plan.endpoints.size(); ++i) {
        if (removed[i]) { continue; }
        auto& endpoint = plan.endpoints[i];
        if (combined[i]) {
            SmallVector<SmallVector<TemplateCoordinate>> points;
            SmallVector<int64_t> labels;
            for (auto record : endpoint.records) {
                const auto* member = members.at(record).second;
                points.push_back(endpoint.kind == LogicalCommandKind::Set ?
                    member->sourceCoordinates : member->targetCoordinates);
                labels.push_back(record);
            }
            OpBuilder builder(endpoint.before->getContext());
            builder.setInsertionPointToEnd(&plan.addPreparation(endpoint.before));
            auto selection = emitFamilyExpressions(builder, endpoint.before->getLoc(), points, labels);
            if (!selection.error.empty()) { return endpoint.before->emitError(selection.error); }
            endpoint.memberCoordinates.front() = selection.member;
            endpoint.guard = selection.present;
            for (auto factor : domains.at(i).residual) {
                endpoint.guard = builder.create<arith::AndIOp>(endpoint.before->getLoc(), endpoint.guard, factor);
            }
        }
        retained.push_back(std::move(endpoint));
    }
    plan.endpoints = std::move(retained);
    return success();
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
    if (function->hasAttr("pto.endpoint_families")) {
        return function.emitError("logical insertion cannot replace existing endpoint-family provenance");
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
            return function.emitError("invalid logical plan: PIPE_S barrier is forbidden; "
                                      "same-scalar storage hazards must be hardware protected before reduction");
        }
        const bool kind = barrier || endpoint.kind == LogicalCommandKind::Set ||
            endpoint.kind == LogicalCommandKind::Wait;
        if (!kind || !available.cut(endpoint.before) || endpoint.record < 0 ||
            !concretePipe(endpoint.sourcePipe) || !concretePipe(endpoint.targetPipe) ||
            (barrier != (endpoint.sourcePipe == endpoint.targetPipe)) ||
            !available.value(endpoint.guard, endpoint.before) || !endpoint.guard.getType().isInteger(1) ||
            (!barrier && (!available.value(endpoint.identity, endpoint.before) ||
                         !endpoint.identity.getType().isIndex())) ||
            (barrier && endpoint.identity)) {
            return function.emitError("invalid or unavailable prepared logical endpoint");
        }
    }
    for (const auto& endpoint : plan.endpoints) {
        for (auto member : endpoint.memberCoordinates) {
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
    function->setAttr("pto.endpoint_families", builder.getDictionaryAttr({
        builder.getNamedAttr("version",
                             builder.getI64IntegerAttr(plan.nestedIdentities ? 4 :
                                 (plan.independentPieces ? 3 : (plan.groupedFamilies ? 2 : 1)))),
        builder.getNamedAttr("plan", builder.getI64IntegerAttr(plan.planId)),
        builder.getNamedAttr("families", builder.getArrayAttr(families)),
        builder.getNamedAttr("pieces", builder.getArrayAttr(pieces))}));
}
void command(OpBuilder& builder, const PreparedLogicalEndpoint& endpoint, int64_t planId, uint64_t cutId)
{
    auto location = endpoint.before->getLoc();
    auto branch = builder.create<scf::IfOp>(location, endpoint.guard, false);
    branch->setAttr("pto.endpoint_cut", builder.getI64IntegerAttr(cutId));
    OpBuilder::InsertionGuard restore(builder);
    builder.setInsertionPointToStart(branch.thenBlock());
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
void emitCut(OpBuilder& builder, ArrayRef<const PreparedLogicalEndpoint*> endpoints, int64_t planId, uint64_t cutId)
{
    builder.setInsertionPoint(endpoints.front()->before);
    auto location = endpoints.front()->before->getLoc();
    std::map<uint32_t, Value> barriers;
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind == LogicalCommandKind::Barrier) {
            auto [entry, inserted] = barriers.emplace(endpoint->sourcePipe, endpoint->guard);
            if (!inserted) {
                entry->second = builder.create<arith::OrIOp>(location, entry->second, endpoint->guard);
            }
        }
    }
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind == LogicalCommandKind::Set) {
            command(builder, *endpoint, planId, cutId);
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
            command(builder, *endpoint, planId, cutId);
        }
    }
}
} // namespace
LogicalResult insertLogicalSynchronization(func::FuncOp function, PreparedLogicalPlan& plan)
{
    if (!function) {
        return failure();
    }
    if (plan.completionOwner) { return plan.completionOwner->insert(function, plan); }
    if (plan.privateState) { return plan.privateState->insert(function, plan); }
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
    deduplicatePreparation(plan);
    if (failed(coalesceEndpointAlternatives(plan))) { return failure(); }
    if (failed(preflight(function, plan))) { return failure(); }
    for (auto& stage : plan.preparation) {
        auto* target = stage.before->getBlock();
        target->getOperations().splice(stage.before->getIterator(), stage.code->getOperations());
    }
    llvm::MapVector<Operation*, SmallVector<const PreparedLogicalEndpoint*>> cuts;
    for (const auto& endpoint : plan.endpoints) {
        cuts[endpoint.before].push_back(&endpoint);
    }
    serializeFamilies(function, plan, cuts);
    OpBuilder builder(function.getContext());
    uint64_t cutId = 0;
    for (const auto& cut : cuts) {
        emitCut(builder, cut.second, plan.planId, cutId++);
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
    if (plan.allocationCertificate) {
        auto kind = plan.allocationCertificate.getAs<StringAttr>("kind");
        function->setAttr(kind && kind.getValue() == "finite" ? FiniteAllocationAttr : CyclicAllocationAttr,
                          plan.allocationCertificate);
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
