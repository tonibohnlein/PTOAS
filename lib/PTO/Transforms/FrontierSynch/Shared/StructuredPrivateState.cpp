// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Append private scalar state to structured control in an owned staging module.
#include "StructuredPrivateState.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::frontiersynch {
namespace {
class Materializer {
public:
    Materializer(func::FuncOp original, ArrayRef<PrivateStateSlot> slots,
                 ArrayRef<PrivateStateAction> actions, PrivateStateError& error)
        : original(original), slots(slots), actions(actions), error(error) {}
    FailureOr<std::unique_ptr<StructuredPrivateState>> run()
    {
        if (failed(validate())) { return failure(); }
        result = std::make_unique<StructuredPrivateState>();
        auto module = cast<ModuleOp>(original->getParentOp());
        result->module = cast<ModuleOp>(module->clone());
        result->function = result->module->lookupSymbol<func::FuncOp>(original.getSymName());
        if (!result->function) {
            (void)fail(PrivateStateFailure::InvalidInput, original, "staged function symbol is unavailable");
            return failure();
        }
        result->function.getBody().getBlocks().clear();
        auto* entry = result->function.addEntryBlock();
        result->mapping.map(original.getOperation(), result->function.getOperation());
        result->mapping.map(&original.front(), entry);
        result->mapping.map(original.getArguments(), entry->getArguments());
        OpBuilder builder(original.getContext());
        builder.setInsertionPointToStart(entry);
        SmallVector<Value> state;
        for (const auto& slot : slots) {
            state.push_back(builder.create<arith::ConstantOp>(original.getLoc(), slot.type, slot.initial));
        }
        if (failed(block(original.front(), builder, state, false))) { return failure(); }
        result->finalState.assign(state.begin(), state.end());
        if (failed(verify(*result->module))) {
            (void)fail(PrivateStateFailure::Verification, original, "staged private state violates IR verification");
            return failure();
        }
        DominanceInfo dominance(result->function);
        for (auto value : state) {
            const bool available = dominance.properlyDominates(value, entry->getTerminator());
            if (!available) {
                (void)fail(PrivateStateFailure::Action, original, "final state does not dominate the return");
                return failure();
            }
        }
        return std::move(result);
    }
private:
    LogicalResult fail(PrivateStateFailure kind, Operation* operation, StringRef message)
    {
        error = {kind, operation, message.str()};
        return failure();
    }
    LogicalResult validate()
    {
        const bool validFunction = original && isa<ModuleOp>(original->getParentOp()) &&
            original.getBody().hasOneBlock() && succeeded(verify(original));
        if (!validFunction) {
            return fail(PrivateStateFailure::InvalidInput, original, "expected a verified single-block function");
        }
        for (const auto& slot : slots) {
            const bool validSlot = slot.type && isa<IntegerType, IndexType>(slot.type) &&
                slot.initial && slot.initial.getType() == slot.type && isa<IntegerAttr>(slot.initial);
            if (!validSlot) {
                return fail(PrivateStateFailure::InvalidSlot, original, "slot requires a typed integer/index constant");
            }
        }
        auto unsupported = original.walk([&](Operation* op) {
            if (op == original.getOperation()) { return WalkResult::advance(); }
            const bool region = op->getNumRegions() && !isa<scf::ForOp, scf::IfOp>(op);
            if (region || op->getNumSuccessors()) { return WalkResult::interrupt(); }
            return WalkResult::advance();
        });
        if (unsupported.wasInterrupted()) {
            return fail(PrivateStateFailure::UnsupportedControl, original, "only scf.for/scf.if regions are supported");
        }
        for (const auto& action : actions) {
            const bool validAnchor = action.anchor && original->isProperAncestor(action.anchor) && action.apply;
            const bool validPosition = action.cut == PrivateStateCut::Before || action.cut == PrivateStateCut::After;
            const bool afterTerminator = validAnchor && action.cut == PrivateStateCut::After &&
                action.anchor->hasTrait<OpTrait::IsTerminator>();
            if (!validAnchor || !validPosition || afterTerminator) {
                return fail(PrivateStateFailure::InvalidCut, action.anchor, "invalid original private-state cut");
            }
            auto& table = action.cut == PrivateStateCut::Before ? before : after;
            table[action.anchor].push_back(&action);
        }
        return success();
    }
    LogicalResult apply(Operation* anchor, bool preceding, OpBuilder& builder, SmallVectorImpl<Value>& state)
    {
        auto& table = preceding ? before : after;
        auto found = table.find(anchor);
        if (found == table.end()) { return success(); }
        for (const auto* action : found->second) {
            auto* target = builder.getInsertionBlock();
            std::string reason;
            const bool applied = succeeded(action->apply(builder, result->mapping, state, reason));
            if (!applied) {
                return fail(PrivateStateFailure::Action, anchor, reason.empty() ? "state action failed" : reason);
            }
            const bool sameCut = builder.getInsertionBlock() == target && builder.getInsertionPoint() == target->end();
            if (!sameCut) {
                return fail(PrivateStateFailure::Action, anchor, "state action changed its insertion block");
            }
            for (std::size_t i = 0; i < slots.size(); ++i) {
                auto* region = state[i] ? state[i].getParentRegion() : nullptr;
                auto* owner = region ? region->getParentOp() : nullptr;
                const bool owned = owner && (owner == result->function || result->function->isProperAncestor(owner));
                const bool typed = state[i] && state[i].getType() == slots[i].type;
                if (!owned || !typed) {
                    return fail(PrivateStateFailure::Action, anchor, "state action changed a slot type");
                }
            }
        }
        return success();
    }
    SmallVector<Value> mapped(ValueRange values)
    {
        SmallVector<Value> output;
        for (auto value : values) { output.push_back(result->mapping.lookup(value)); }
        return output;
    }
    void mapResults(Operation* source, Operation* target)
    {
        result->mapping.map(source, target);
        result->mapping.map(source->getResults(), target->getResults().take_front(source->getNumResults()));
    }
    LogicalResult forLoop(scf::ForOp source, OpBuilder& builder, SmallVectorImpl<Value>& state)
    {
        auto initial = mapped(source.getInitArgs());
        initial.append(state.begin(), state.end());
        auto& mapping = result->mapping;
        auto target = builder.create<scf::ForOp>(source.getLoc(), mapping.lookup(source.getLowerBound()),
            mapping.lookup(source.getUpperBound()), mapping.lookup(source.getStep()), initial);
        target->setAttrs(source->getAttrs());
        mapResults(source, target);
        auto* body = target.getBody();
        body->getOperations().clear();
        mapping.map(source.getBody(), body);
        auto originalArguments = source.getBody()->getArguments();
        mapping.map(originalArguments, body->getArguments().take_front(originalArguments.size()));
        SmallVector<Value> inner(body->getArguments().take_back(slots.size()));
        {
            OpBuilder::InsertionGuard restore(builder);
            builder.setInsertionPointToEnd(body);
            if (failed(block(*source.getBody(), builder, inner, true))) { return failure(); }
        }
        auto outputs = target.getResults().take_back(slots.size());
        state.assign(outputs.begin(), outputs.end());
        return success();
    }
    LogicalResult branch(scf::IfOp source, OpBuilder& builder, SmallVectorImpl<Value>& state)
    {
        SmallVector<Type> types(source.getResultTypes());
        for (auto value : state) { types.push_back(value.getType()); }
        auto target = builder.create<scf::IfOp>(source.getLoc(), types,
            result->mapping.lookup(source.getCondition()), true, true);
        target->setAttrs(source->getAttrs());
        mapResults(source, target);
        for (bool thenArm : {true, false}) {
            auto& oldRegion = thenArm ? source.getThenRegion() : source.getElseRegion();
            auto* newBlock = thenArm ? target.thenBlock() : target.elseBlock();
            newBlock->getOperations().clear();
            SmallVector<Value> arm(state.begin(), state.end());
            OpBuilder::InsertionGuard restore(builder);
            builder.setInsertionPointToEnd(newBlock);
            if (oldRegion.empty()) {
                builder.create<scf::YieldOp>(source.getLoc(), arm);
            } else {
                result->mapping.map(&oldRegion.front(), newBlock);
                if (failed(block(oldRegion.front(), builder, arm, true))) { return failure(); }
            }
        }
        auto outputs = target.getResults().take_back(slots.size());
        state.assign(outputs.begin(), outputs.end());
        return success();
    }
    LogicalResult block(Block& source, OpBuilder& builder, SmallVectorImpl<Value>& state, bool appendState)
    {
        for (auto& op : source) {
            if (failed(apply(&op, true, builder, state))) { return failure(); }
            if (auto loop = dyn_cast<scf::ForOp>(op)) {
                if (failed(forLoop(loop, builder, state))) { return failure(); }
            } else if (auto conditional = dyn_cast<scf::IfOp>(op)) {
                if (failed(branch(conditional, builder, state))) { return failure(); }
            } else if (appendState && isa<scf::YieldOp>(op)) {
                auto values = mapped(op.getOperands());
                values.append(state.begin(), state.end());
                auto yield = builder.create<scf::YieldOp>(op.getLoc(), values);
                yield->setAttrs(op.getAttrs());
                result->mapping.map(&op, yield.getOperation());
            } else {
                builder.clone(op, result->mapping);
            }
            if (failed(apply(&op, false, builder, state))) { return failure(); }
        }
        return success();
    }
    func::FuncOp original;
    ArrayRef<PrivateStateSlot> slots;
    ArrayRef<PrivateStateAction> actions;
    PrivateStateError& error;
    std::unique_ptr<StructuredPrivateState> result;
    llvm::DenseMap<Operation*, SmallVector<const PrivateStateAction*>> before, after;
};
} // namespace
FailureOr<std::unique_ptr<StructuredPrivateState>> materializeStructuredPrivateState(
    func::FuncOp original, ArrayRef<PrivateStateSlot> slots,
    ArrayRef<PrivateStateAction> actions, PrivateStateError& error)
{
    error = {};
    return Materializer(original, slots, actions, error).run();
}
} // namespace mlir::pto::frontiersynch
