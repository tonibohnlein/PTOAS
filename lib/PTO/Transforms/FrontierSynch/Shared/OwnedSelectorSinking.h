// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute owned selector DAGs under their exact consumer guard or guard union.
#ifndef PTO_FRONTIER_OWNEDSELECTORSINKING_H
#define PTO_FRONTIER_OWNEDSELECTORSINKING_H
#include "SelectorGuardPlan.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch::detail {
class OwnedSelectorSinking {
public:
    void retain(Block& code, Operation* cut)
    {
        for (auto& operation : code) {
            if (operation.getName().getDialectNamespace() != "arith" || operation.getNumRegions() ||
                operation.getNumSuccessors() || !isMemoryEffectFree(&operation) ||
                llvm::any_of(operation.getResultTypes(), [](Type type) { return !type.isIntOrIndex(); })) {
                continue;
            }
            owners.try_emplace(&operation, cut);
            pending.push_back(&operation);
        }
    }
    void guard(scf::IfOp branch, Operation* cut)
    {
        if (guards.try_emplace(branch.thenBlock(), cut).second) { order.push_back(branch.thenBlock()); }
    }
    // Evidence retains original-cut Values outside SSA use lists. Such roots
    // and their producers must not become then-only values or fallback results.
    void preserve(Value value) { if (value) { evidence.insert(value); } }
    uint64_t mergedGuards() const { return merged; }
    uint64_t constructedScalars() const { return constructed; }
    uint64_t run(bool shared = true)
    {
        consolidateGuards();
        uint64_t moved = shared ? conditionShared() : 0;
        // This remaining exclusive-use movement creates no scalars. Integer
        // division may be partial: moving it inside its existing use guard only
        // removes evaluation on paths where its result is not consumed.
        while (!pending.empty()) {
            auto* operation = pending.pop_back_val();
            auto owner = owners.find(operation);
            if (owner == owners.end()) { continue; }
            auto* target = soleGuard(operation, owner->second);
            if (!target) { continue; }
            // Consumers move first. All operands were available outside this
            // guard, and a producer can move only after every remaining user.
            // Inserting at the front consequently preserves operand dominance.
            operation->moveBefore(target, target->begin());
            owners.erase(owner);
            ++moved;
            for (auto operand : operation->getOperands()) {
                if (auto* producer = operand.getDefiningOp(); owners.count(producer)) { pending.push_back(producer); }
            }
        }
        return moved;
    }
private:
    llvm::DenseMap<Operation*, Operation*> owners;
    llvm::DenseMap<Block*, Operation*> guards;
    llvm::DenseSet<Value> evidence;
    SmallVector<Operation*> pending;
    SmallVector<Block*> order;
    uint64_t merged = 0, constructed = 0;
    class ConditionalEmitter {
    public:
        ConditionalEmitter(OwnedSelectorSinking& owner, ArrayRef<Operation*> operations,
            llvm::DenseMap<uint64_t, Value>& values, const std::map<Operation*, Operation*>& first)
            : owner(owner), operations(operations), values(values), first(first),
              builder(first.begin()->first->getContext()), construction(builder, first.begin()->first->getLoc()) {}
        ScalarToken lookup(uint64_t id) { return construction.input(values.lookup(id)); }
        ScalarToken lor(ScalarToken a, ScalarToken b) { return construction.binary<arith::OrIOp>(a, b); }
        void at(Operation* cut) { builder.setInsertionPoint(first.at(cut)); construction.setLocation(cut->getLoc()); }
        void begin(const SelectorGuardGroup& group, ScalarToken guard)
        {
            branch = builder.create<scf::IfOp>(group.cut->getLoc(), group.types, guard.value, true);
            if (!branch.thenBlock()->empty()) { branch.thenBlock()->getTerminator()->erase(); }
            if (!branch.elseBlock()->empty()) { branch.elseBlock()->getTerminator()->erase(); }
            builder.setInsertionPointToEnd(branch.elseBlock());
            fallback.clear();
        }
        void zero(Type type)
        {
            fallback.push_back(type.isIndex() ? construction.number(0).value :
                construction.integer(APInt(cast<IntegerType>(type).getWidth(), 0)).value);
        }
        void finish(const SelectorGuardGroup& group)
        {
            builder.create<scf::YieldOp>(group.cut->getLoc(), fallback);
            for (auto index : group.instructions) {
                auto* operation = operations[index];
                operation->moveBefore(branch.thenBlock(), branch.thenBlock()->end());
                owner.owners.erase(operation);
                ++moved;
            }
            SmallVector<Value> outputs;
            for (auto id : group.outputs) { outputs.push_back(values.lookup(id)); }
            builder.setInsertionPointToEnd(branch.thenBlock());
            builder.create<scf::YieldOp>(group.cut->getLoc(), outputs);
            for (auto entry : llvm::enumerate(outputs)) {
                const auto replacement = branch.getResult(entry.index());
                entry.value().replaceUsesWithIf(replacement, [&](OpOperand& use) {
                    return !branch->isProperAncestor(use.getOwner());
                });
                values[group.outputs[entry.index()]] = replacement;
            }
        }
        uint64_t moved = 0;
        uint64_t operationsConstructed() const { return construction.operations().value_or(0); }
    private:
        OwnedSelectorSinking& owner;
        ArrayRef<Operation*> operations;
        llvm::DenseMap<uint64_t, Value>& values;
        const std::map<Operation*, Operation*>& first;
        OpBuilder builder;
        ScalarConstruction construction;
        scf::IfOp branch;
        SmallVector<Value> fallback;
    };
    uint64_t conditionShared()
    {
        if (owners.empty() || guards.empty()) { return 0; }
        llvm::DenseMap<Value, uint64_t> identities;
        llvm::DenseMap<uint64_t, Value> values;
        auto identify = [&](Value value) {
            auto [found, added] = identities.try_emplace(value, identities.size() + 1);
            if (added) { values[found->second] = value; }
            return found->second;
        };
        std::map<Operation*, Operation*> first;
        for (auto* block : order) {
            auto found = guards.find(block);
            if (found == guards.end()) { continue; }
            first.try_emplace(found->second, block->getParentOp());
        }
        SmallVector<ScalarInstruction, 0> instructions;
        SmallVector<Operation*> operations;
        SmallVector<SelectorUse> uses;
        for (auto value : evidence) { uses.push_back({identify(value), 0, nullptr}); }
        for (auto* operation : pending) {
            auto found = owners.find(operation);
            if (found == owners.end()) { continue; }
            auto described = describeScalarInstruction(*operation, found->second, identify);
            if (!described) { return 0; }
            instructions.push_back(std::move(*described)); operations.push_back(operation);
            for (auto result : operation->getResults()) {
                for (auto& use : result.getUses()) {
                    if (owners.count(use.getOwner())) { continue; }
                    auto* block = use.getOwner()->getBlock();
                    while (block && !guards.count(block)) {
                        auto* parent = block->getParentOp(); block = parent ? parent->getBlock() : nullptr;
                    }
                    auto guard = guards.find(block);
                    if (guard == guards.end()) { uses.push_back({identify(result), 0, nullptr}); }
                    else {
                        auto branch = cast<scf::IfOp>(block->getParentOp());
                        uses.push_back({identify(result), identify(branch.getCondition()), guard->second});
                    }
                }
            }
        }
        auto groups = planSelectorGuards(instructions, uses);
        if (groups.empty()) { return 0; }
        DominanceInfo dominance;
        for (const auto& group : groups) {
            auto cut = first.find(group.cut);
            if (cut == first.end()) { return 0; }
            for (auto guard : group.guards) {
                if (!dominance.properlyDominates(values.lookup(guard), cut->second)) { return 0; }
            }
        }
        ConditionalEmitter emitter(*this, operations, values, first);
        for (const auto& group : groups) { emitter.at(group.cut); constructSelectorGuardGroup(group, emitter); }
        constructed += emitter.operationsConstructed();
        return emitter.moved;
    }
    void consolidateGuards()
    {
        for (auto* block : order) {
            const auto owner = guards.find(block);
            if (owner == guards.end()) { continue; }
            auto branch = dyn_cast<scf::IfOp>(block->getParentOp());
            if (!branch || branch.getNumResults() || !branch.getElseRegion().empty()) { continue; }
            while (auto next = dyn_cast_or_null<scf::IfOp>(branch->getNextNode())) {
                const auto following = guards.find(next.thenBlock());
                if (following == guards.end() || following->second != owner->second ||
                    next.getNumResults() || !next.getElseRegion().empty() ||
                    branch.getCondition() != next.getCondition() ||
                    branch->getAttrDictionary() != next->getAttrDictionary()) { break; }
                // Only adjacent registered command guards combine. Barriers
                // and any unregistered operation remain ordering boundaries.
                auto* source = next.thenBlock();
                block->getOperations().splice(block->getTerminator()->getIterator(), source->getOperations(),
                    source->begin(), source->getTerminator()->getIterator());
                guards.erase(following);
                next->erase();
                ++merged;
            }
        }
    }
    Block* soleGuard(Operation* operation, Operation* cut) const
    {
        Block* selected = nullptr;
        bool used = false;
        for (auto result : operation->getResults()) {
            if (evidence.count(result)) { return nullptr; }
            for (auto& use : result.getUses()) {
                used = true;
                auto* block = use.getOwner()->getBlock();
                while (block && !guards.count(block)) {
                    auto* parent = block->getParentOp();
                    block = parent ? parent->getBlock() : nullptr;
                }
                const auto found = guards.find(block);
                if (found == guards.end() || found->second != cut || (selected && selected != block)) {
                    return nullptr;
                }
                selected = block;
            }
        }
        return used ? selected : nullptr;
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
