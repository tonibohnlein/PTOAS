// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Intern expressions independently of source cuts; emit shared arithmetic only
// after checking original-IR availability at each detached preparation cut.
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
RegionExpressions::Id RegionExpressions::cofactorAtCut(Id expression, Operation* cut,
                                                       llvm::DenseMap<Id, Id>& memo)
{
    llvm::DenseMap<Value, bool> assumptions;
    for (Operation* nested = cut; nested && nested->getParentOp(); nested = nested->getParentOp()) {
        if (auto branch = dyn_cast<scf::IfOp>(nested->getParentOp())) {
            assumptions[branch.getCondition()] = nested->getParentRegion() == &branch.getThenRegion();
        }
    }
    SmallVector<Id> pending{expression}, order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (memo.count(next) || !seen.insert(next).second) { continue; }
        order.push_back(next);
        appendOperands(nodes[next], pending);
    }
    llvm::sort(order);
    for (Id id : order) {
        // Interning may reallocate nodes, so keep a value copy.
        const Node node = nodes[id];
        Id result = id;
        if (node.kind == Kind::Integer) {
            result = rebuildInteger(node, memo);
        } else if (node.kind == Kind::Input && node.boolean && assumptions.count(node.value)) {
            result = boolean(assumptions.lookup(node.value));
        } else if (node.kind == Kind::Not) {
            result = lnot(memo.lookup(node.a));
        } else if (node.kind == Kind::Select) {
            result = select(memo.lookup(node.a), memo.lookup(node.b), memo.lookup(node.c));
        } else if (node.a != invalid && node.b != invalid) {
            result = binary(node.kind, memo.lookup(node.a), memo.lookup(node.b));
        }
        memo[id] = result;
    }
    return memo.lookup(expression);
}
FailureOr<Value> RegionExpressions::emitContextual(Id expression, OpBuilder& builder,
                                                  Operation* cut, CutEmission& context)
{
    emissionMessage.clear();
    auto function = cut ? cut->getParentOfType<func::FuncOp>() : func::FuncOp();
    if (!constructionMessage.empty() || !valid(expression) || !function ||
        !builder.getInsertionBlock() || builder.getInsertionBlock()->getParent()) {
        emissionMessage = "guarded expression requires a valid root, original cut and detached block";
        return failure();
    }
    if (dominanceRoot != function.getOperation()) {
        dominanceRoot = function.getOperation();
        dominance = std::make_unique<DominanceInfo>(dominanceRoot);
    }
    Id selected = cofactorAtCut(expression, cut, context.cofactors);
    SmallVector<Id> pending{selected}, inputNodes;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (context.values.count(next) || !seen.insert(next).second) { continue; }
        const Node& node = nodes[next];
        if (node.kind == Kind::Input) { inputNodes.push_back(next); }
        appendOperands(node, pending);
    }
    // Preflight the complete SSA operand DAG before creating any cloned code.
    SmallVector<std::pair<Value, bool>> work;
    for (auto id : inputNodes) { work.push_back({nodes[id].value, false}); }
    llvm::DenseSet<Value> complete;
    llvm::DenseSet<Operation*> scheduled;
    SmallVector<Operation*> recipe;
    while (!work.empty()) {
        auto [value, finish] = work.pop_back_val();
        if (complete.contains(value) || context.inputs.count(value)) { continue; }
        auto* region = value.getParentRegion();
        auto* owner = region ? region->getParentOp() : nullptr;
        if (!owner || (owner != function && !function->isProperAncestor(owner))) {
            emissionMessage = "guarded expression input belongs to another invocation";
            return failure();
        }
        if (dominance->properlyDominates(value, cut)) {
            complete.insert(value); continue;
        }
        Operation* op = value.getDefiningOp();
        // Pure/speculatable is not a determinism contract (e.g. LLVM freeze).
        // Replay is restricted to the deterministic scalar arithmetic language;
        // values from every other dialect are usable when they dominate the cut.
        auto dialect = op ? op->getName().getDialectNamespace() : StringRef();
        bool deterministic = dialect == "arith" || dialect == "index";
        if (!op || !deterministic || op->getNumRegions() || op->getNumSuccessors() ||
            forbiddenRecomputation.contains(op) || !function->isProperAncestor(op) ||
            !isMemoryEffectFree(op) || !isSpeculatable(op)) {
            emissionMessage = "guarded endpoint needs an unavailable value outside deterministic scalar replay";
            return failure();
        }
        if (!finish) {
            work.push_back({value, true});
            for (Value operand : op->getOperands()) { work.push_back({operand, false}); }
        } else {
            complete.insert(value);
            if (scheduled.insert(op).second) { recipe.push_back(op); }
        }
    }
    IRMapping mapping;
    for (const auto& item : context.inputs) { mapping.map(item.first,item.second); }
    for (Value value : complete) {
        if (dominance->properlyDominates(value,cut)) { mapping.map(value,value); }
    }
    for (Operation* op : recipe) {
        Operation* clone = builder.clone(*op,mapping);
        for (auto pair : llvm::zip(op->getResults(),clone->getResults())) {
            context.inputs[std::get<0>(pair)] = std::get<1>(pair);
        }
    }
    for (auto id : inputNodes) {
        Node node = nodes[id];
        node.value = mapping.lookupOrDefault(node.value);
        context.values[id] = emitNode(node,builder,cut->getLoc(),context.values);
    }
    return emit(selected,builder,cut,context.values);
}
} // namespace mlir::pto::frontiersynch
