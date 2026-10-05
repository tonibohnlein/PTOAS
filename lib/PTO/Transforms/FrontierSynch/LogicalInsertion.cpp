// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate prepared endpoints, then insert logical commands at actual cuts.
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
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
LogicalResult preflight(func::FuncOp function, const PreparedLogicalPlan& plan)
{
    if (plan.planId < 0 || !RegisteredOperationName::lookup("pto.logical_set", function.getContext()) ||
        !RegisteredOperationName::lookup("pto.logical_wait", function.getContext())) {
        return function.emitError("logical insertion requires a nonnegative namespace and registered logical commands");
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
            if (plan.groupedFamilies) {
                owners.emplace(family.id, &family);
            }
            for (const auto& member : family.members) {
                if (!originalRecords.insert(member.record).second) {
                    return function.emitError("duplicate endpoint-family member");
                }
                if (!plan.groupedFamilies) {
                    owners.emplace(member.record, &family);
                }
                for (auto side : {false, true}) {
                    const auto& tuple = side ? member.targetCoordinates : member.sourceCoordinates;
                    auto* before = side ? family.targetCut.before : family.sourceCut.before;
                    for (auto coordinate : tuple) {
                        if (!coordinate.loop || !coordinate.loop->isProperAncestor(before) ||
                            !available.value(coordinate.loop.getInductionVar(), before)) {
                            return function.emitError("unavailable endpoint-family coordinate");
                        }
                    }
                }
            }
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
    auto coordinates = [&](ArrayRef<TemplateCoordinate> tuple) {
        SmallVector<Attribute> values;
        for (auto coordinate : tuple) {
            auto* loop = coordinate.loop.getOperation();
            auto [entry, added] = loopIds.try_emplace(loop, loopIds.size());
            if (added) {
                loop->setAttr("pto.family_loop", builder.getI64IntegerAttr(entry->second));
            }
            values.push_back(builder.getDenseI64ArrayAttr({entry->second, coordinate.induction}));
        }
        return builder.getArrayAttr(values);
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
            builder.getNamedAttr("members", builder.getArrayAttr(members))}));
    }
    function->setAttr("pto.endpoint_families", builder.getDictionaryAttr({
        builder.getNamedAttr("version", builder.getI64IntegerAttr(plan.groupedFamilies ? 2 : 1)),
        builder.getNamedAttr("plan", builder.getI64IntegerAttr(plan.planId)),
        builder.getNamedAttr("families", builder.getArrayAttr(families))}));
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
    function.getContext()->getOrLoadDialect<PTODialect>();
    function.getContext()->getOrLoadDialect<arith::ArithDialect>();
    function.getContext()->getOrLoadDialect<scf::SCFDialect>();
    if (failed(preflight(function, plan))) {
        return failure();
    }
    if (plan.allocationCertificate && function->hasAttr(CyclicAllocationAttr)) {
        return function.emitError("logical insertion cannot replace an existing allocation certificate");
    }
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
            builder.setInsertionPoint(ret);
            builder.create<BarrierOp>(ret.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_ALL));
        });
    }
    if (plan.allocationCertificate) {
        function->setAttr(CyclicAllocationAttr, plan.allocationCertificate);
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
