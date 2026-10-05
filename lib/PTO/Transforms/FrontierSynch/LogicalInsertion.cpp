// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate prepared endpoints, then insert logical commands at actual cuts.
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include <map>
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
    return success();
}
void command(OpBuilder& builder, const PreparedLogicalEndpoint& endpoint, int64_t planId)
{
    auto location = endpoint.before->getLoc();
    auto branch = builder.create<scf::IfOp>(location, endpoint.guard, false);
    OpBuilder::InsertionGuard restore(builder);
    builder.setInsertionPointToStart(branch.thenBlock());
    OperationState state(location, endpoint.kind == LogicalCommandKind::Set ? "pto.logical_set" : "pto.logical_wait");
    state.addOperands(endpoint.identity);
    state.addAttribute("src_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(endpoint.sourcePipe)));
    state.addAttribute("dst_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(endpoint.targetPipe)));
    state.addAttribute("plan_id", builder.getI64IntegerAttr(planId));
    state.addAttribute("record_id", builder.getI64IntegerAttr(endpoint.record));
    builder.create(state);
}
void emitCut(OpBuilder& builder, ArrayRef<const PreparedLogicalEndpoint*> endpoints, int64_t planId)
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
            command(builder, *endpoint, planId);
        }
    }
    for (auto [pipe, guard] : barriers) {
        auto branch = builder.create<scf::IfOp>(location, guard, false);
        OpBuilder::InsertionGuard restore(builder);
        builder.setInsertionPointToStart(branch.thenBlock());
        builder.create<BarrierOp>(location, PipeAttr::get(builder.getContext(), static_cast<PIPE>(pipe)));
    }
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind == LogicalCommandKind::Wait) {
            command(builder, *endpoint, planId);
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
    for (auto& stage : plan.preparation) {
        auto* target = stage.before->getBlock();
        target->getOperations().splice(stage.before->getIterator(), stage.code->getOperations());
    }
    llvm::MapVector<Operation*, SmallVector<const PreparedLogicalEndpoint*>> cuts;
    for (const auto& endpoint : plan.endpoints) {
        cuts[endpoint.before].push_back(&endpoint);
    }
    OpBuilder builder(function.getContext());
    for (const auto& cut : cuts) {
        emitCut(builder, cut.second, plan.planId);
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
