// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Plan original-cut availability and deterministic scalar replay without
// generated SSA, then lower the immutable recipe into detached preparation.
// Both stages borrow an append-only arena and unchanged original invocation.
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "OriginalCutFacts.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
// No generated values are retained. These dependencies are interpreted once by
// planning, then consumed by lowering with the original cut and binding owner.
class CutExpressionRecipe {
public:
    const RegionExpressions* owner = nullptr;
    Operation* cut = nullptr;
    RegionExpressions::Id selected = RegionExpressions::invalid;
    SmallVector<RegionExpressions::Id> inputs, boundExpressions;
    SmallVector<Value> directInputs, boundInputs;
    SmallVector<Operation*> replay;
};
RegionExpressions::Id RegionExpressions::cofactorAtCut(Id expression, Operation* cut,
                                                       llvm::DenseMap<Id, Id>& memo)
{
    const auto assumptions = detail::originalCutFacts(cut);
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
FailureOr<std::shared_ptr<const CutExpressionRecipe>> RegionExpressions::prepareContextual(
    Id expression, Operation* cut, CutEmission& context)
{
    emissionMessage.clear();
    auto function = cut ? cut->getParentOfType<func::FuncOp>() : func::FuncOp();
    if (!constructionMessage.empty() || !valid(expression) || !function) {
        emissionMessage = "guarded expression requires a valid root and original cut";
        return failure();
    }
    if (dominanceRoot != function.getOperation()) {
        dominanceRoot = function.getOperation();
        dominance = std::make_unique<DominanceInfo>(dominanceRoot);
    }
    auto recipe = std::make_shared<CutExpressionRecipe>();
    recipe->owner = this;
    recipe->cut = cut;
    recipe->selected = cofactorAtCut(expression, cut, context.cofactors);
    if (!constructionMessage.empty() || !valid(recipe->selected)) {
        emissionMessage = "guarded expression specialization is unavailable"; return failure();
    }
    SmallVector<Id> pending{recipe->selected};
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (!seen.insert(next).second) { continue; }
        if (context.values.count(next)) { recipe->boundExpressions.push_back(next); continue; }
        const Node& node = nodes[next];
        if (node.kind == Kind::Input) { recipe->inputs.push_back(next); }
        appendOperands(node, pending);
    }
    // Preflight the complete original operand DAG. Missing bindings and
    // forbidden payload/scalar producers fail before any cloned code exists.
    SmallVector<std::pair<Value, bool>> work;
    for (auto id : recipe->inputs) { work.push_back({nodes[id].value, false}); }
    llvm::DenseSet<Value> complete;
    llvm::DenseSet<Operation*> scheduled;
    while (!work.empty()) {
        auto [value, finish] = work.pop_back_val();
        if (complete.contains(value)) { continue; }
        if (context.inputs.count(value)) {
            recipe->boundInputs.push_back(value); complete.insert(value); continue;
        }
        auto* region = value.getParentRegion();
        auto* owner = region ? region->getParentOp() : nullptr;
        if (!owner || (owner != function && !function->isProperAncestor(owner))) {
            emissionMessage = "guarded expression input belongs to another invocation";
            return failure();
        }
        if (dominance->properlyDominates(value, cut)) {
            recipe->directInputs.push_back(value); complete.insert(value); continue;
        }
        Operation* op = value.getDefiningOp();
        if (!mlir::pto::detail::canReplayScalar(op) ||
            forbiddenRecomputation.contains(op) || !function->isProperAncestor(op)) {
            emissionMessage = "guarded endpoint needs an unavailable value outside deterministic scalar replay";
            return failure();
        }
        if (!finish) {
            work.push_back({value, true});
            for (Value operand : op->getOperands()) { work.push_back({operand, false}); }
        } else {
            complete.insert(value);
            if (scheduled.insert(op).second) { recipe->replay.push_back(op); }
        }
    }
    return std::shared_ptr<const CutExpressionRecipe>(std::move(recipe));
}
FailureOr<Value> RegionExpressions::emitPreparedContextual(const CutExpressionRecipe& recipe,
    OpBuilder& builder, Operation* cut, CutEmission& context)
{
    emissionMessage.clear();
    if (recipe.owner != this || recipe.cut != cut || !valid(recipe.selected) ||
        !constructionMessage.empty() || !builder.getInsertionBlock() || builder.getInsertionBlock()->getParent()) {
        emissionMessage = "prepared guarded expression requires its original arena, cut and detached block";
        return failure();
    }
    // A recipe may depend on earlier cut-local bindings, but never retains
    // their generated Values. Reject missing dependencies before any mutation.
    for (Id id : recipe.boundExpressions) {
        if (!context.values.lookup(id)) {
            emissionMessage = "prepared guarded expression lost an expression binding"; return failure();
        }
    }
    for (Value value : recipe.boundInputs) {
        if (!context.inputs.lookup(value)) {
            emissionMessage = "prepared guarded expression lost an input binding"; return failure();
        }
    }
    for (Operation* op : recipe.replay) {
        if (forbiddenRecomputation.contains(op)) {
            emissionMessage = "prepared guarded expression producer is no longer replayable"; return failure();
        }
    }
    IRMapping mapping;
    auto bind = [&](Value original, Value replacement) {
        mapping.map(original, replacement);
        if (context.replayBindings != UINT64_MAX) { ++context.replayBindings; }
    };
    for (Value value : recipe.directInputs) { bind(value, value); }
    for (Value value : recipe.boundInputs) { bind(value, context.inputs.lookup(value)); }
    for (Operation* op : recipe.replay) {
        Operation* clone = builder.clone(*op, mapping);
        for (auto pair : llvm::zip(op->getResults(), clone->getResults())) {
            context.inputs[std::get<0>(pair)] = std::get<1>(pair);
        }
    }
    for (auto id : recipe.inputs) {
        Node node = nodes[id];
        node.value = mapping.lookupOrDefault(node.value);
        context.values[id] = emitNode(node, builder, cut->getLoc(), context.values);
    }
    return emit(recipe.selected, builder, cut, context.values);
}
FailureOr<Value> RegionExpressions::emitContextual(Id expression, OpBuilder& builder,
    Operation* cut, CutEmission& context)
{
    if (!builder.getInsertionBlock() || builder.getInsertionBlock()->getParent()) {
        emissionMessage = "guarded expression requires a valid root, original cut and detached block";
        return failure();
    }
    auto recipe = prepareContextual(expression, cut, context);
    if (failed(recipe)) { return failure(); }
    return emitPreparedContextual(**recipe, builder, cut, context);
}
} // namespace mlir::pto::frontiersynch
