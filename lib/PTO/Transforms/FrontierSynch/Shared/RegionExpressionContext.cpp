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
#include "RegionConstruction.h"
#include <type_traits>
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
uint32_t cutExpressionRoot(const CutExpressionRecipe& recipe) { return recipe.selected; }
ArrayRef<uint32_t> cutExpressionSharedExpressions(const CutExpressionRecipe& recipe)
{
    return recipe.boundExpressions;
}
void appendCutExpressionKey(const CutExpressionRecipe& recipe, SmallVectorImpl<uint64_t>& key)
{
    key.push_back(reinterpret_cast<uintptr_t>(recipe.owner));
    key.push_back(reinterpret_cast<uintptr_t>(recipe.cut));
    key.push_back(recipe.selected);
    for (const auto* values : {&recipe.inputs, &recipe.boundExpressions}) {
        key.push_back(values->size());
        for (auto value : *values) {
            key.push_back(value);
        }
    }
    for (const auto* values : {&recipe.directInputs, &recipe.boundInputs}) {
        key.push_back(values->size());
        for (auto value : *values) {
            key.push_back(reinterpret_cast<uintptr_t>(value.getAsOpaquePointer()));
        }
    }
    key.push_back(recipe.replay.size());
    for (auto* operation : recipe.replay) {
        key.push_back(reinterpret_cast<uintptr_t>(operation));
    }
}
RegionExpressions::Id RegionExpressions::cofactorAtCut(Id expression, Operation* cut,
    llvm::DenseMap<Id, Id>& memo, llvm::function_ref<bool(Id)> shared)
{
    const auto assumptions = detail::originalCutFacts(cut);
    SmallVector<Id> pending{expression}, order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (memo.count(next) || !seen.insert(next).second) { continue; }
        if (shared && shared(next)) { memo[next] = next; continue; }
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
    Id expression, Operation* cut, CutEmission& context, const SharedCutPlanning* shared)
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
    // The binding owner clears this cache when adding shared roots. Seed once
    // per cut, not once per endpoint request over the same invariant inventory.
    if (context.cofactors.empty()) {
        for (Id bound : context.plannedValues) {
            if (!valid(bound)) { emissionMessage = "invalid shared expression binding"; return failure(); }
            context.cofactors[bound] = bound;
        }
    }
    auto binding = [&](Id id) {
        if (!shared || !shared->values) { return false; }
        if (context.planningSharedRequests != UINT64_MAX) { ++context.planningSharedRequests; }
        if (!shared->values(id)) { return false; }
        context.plannedValues.insert(id);
        return true;
    };
    recipe->selected = cofactorAtCut(expression, cut, context.cofactors, binding);
    if (!constructionMessage.empty() || !valid(recipe->selected)) {
        emissionMessage = "guarded expression specialization is unavailable"; return failure();
    }
    SmallVector<Id> pending{recipe->selected};
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        Id next = pending.pop_back_val();
        if (!seen.insert(next).second) { continue; }
        if (context.values.count(next) || context.plannedValues.count(next)) {
            recipe->boundExpressions.push_back(next); continue;
        }
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
template<class Memo>
FailureOr<detail::ScalarToken> detail::RegionConstructionAccess::schedule(RegionExpressions& arena,
    const CutExpressionRecipe& recipe, ScalarConstruction& construction, Operation* cut,
    Memo& memo, const RegionConstructionBindings* shared)
{
    using Id = RegionExpressions::Id;
    auto reject = [&](const char* message) -> FailureOr<ScalarToken> {
        arena.emissionMessage = message; return failure();
    };
    if (recipe.owner != &arena || recipe.cut != cut || !cut || !arena.valid(recipe.selected) ||
        !arena.constructionMessage.empty()) {
        return reject("prepared guarded expression requires its original arena and cut");
    }
    bool foreignToken = false;
    auto token = [&](auto value) -> ScalarToken {
        if constexpr (std::is_same_v<decltype(value), Value>) { return construction.input(value); }
        else {
            if (!construction.owns(value)) { foreignToken = true; return {}; }
            return value;
        }
    };
    auto put = [](auto& map, auto key, ScalarToken value) {
        if constexpr (std::is_same_v<decltype(map.lookup(key)), Value>) { map[key] = value.value; }
        else { map[key] = value; }
    };
    auto increment = [](uint64_t& value) { if (value != UINT64_MAX) { ++value; } };
    auto import = [&](auto key, auto& map, auto lookup) {
        if (map.count(key) || !lookup) { return; }
        increment(memo.sharedRequests);
        if (auto value = lookup(key)) {
            if (!construction.owns(value)) { foreignToken = true; return; }
            put(map, key, value); increment(memo.sharedImports);
        }
    };
    if (shared) {
        for (Id id : recipe.boundExpressions) { import(id, memo.values, shared->values); }
        SmallVector<Id> pending{recipe.selected};
        llvm::DenseSet<Id> seen;
        while (!pending.empty()) {
            const auto id = pending.pop_back_val();
            if (memo.values.count(id) || !seen.insert(id).second) { continue; }
            import(id, memo.values, shared->values);
            if (!memo.values.count(id)) { arena.appendOperands(arena.nodes[id], pending); }
        }
        for (Value value : recipe.boundInputs) { import(value, memo.inputs, shared->inputs); }
        for (Operation* op : recipe.replay) {
            for (Value value : op->getResults()) { import(value, memo.inputs, shared->inputs); }
        }
    }
    for (Id id : recipe.boundExpressions) {
        if (!token(memo.values.lookup(id))) { return reject("prepared guarded expression lost an expression binding"); }
    }
    for (Value value : recipe.boundInputs) {
        if (!token(memo.inputs.lookup(value))) { return reject("prepared guarded expression lost an input binding"); }
    }
    for (Operation* op : recipe.replay) {
        if (arena.forbiddenRecomputation.contains(op)) {
            return reject("prepared guarded expression producer is no longer replayable");
        }
    }
    if (foreignToken) { return reject("prepared guarded expression mixes scalar construction owners"); }
    const auto cached = token(memo.values.lookup(recipe.selected));
    if (foreignToken) { return reject("prepared guarded expression mixes scalar construction owners"); }
    if (cached) { return cached; }
    llvm::DenseMap<Value, ScalarToken> mapping;
    auto bind = [&](Value original, ScalarToken replacement) {
        mapping[original] = replacement; increment(memo.replayBindings);
    };
    for (Value value : recipe.directInputs) { bind(value, construction.input(value)); }
    for (Value value : recipe.boundInputs) { bind(value, token(memo.inputs.lookup(value))); }
    for (Operation* op : recipe.replay) {
        if (llvm::all_of(op->getResults(), [&](Value value) { return memo.inputs.count(value); })) {
            for (Value value : op->getResults()) { mapping[value] = token(memo.inputs.lookup(value)); }
            continue;
        }
        SmallVector<ScalarToken> operands;
        for (auto value : op->getOperands()) {
            auto found = mapping.find(value);
            operands.push_back(found == mapping.end() ? construction.input(value) : found->second);
        }
        const auto results = construction.replay(*op, operands);
        if (results.size() != op->getNumResults()) { return reject("prepared scalar replay lost result bindings"); }
        for (auto [original, result] : llvm::zip(op->getResults(), results)) {
            mapping[original] = result; put(memo.inputs, original, result);
        }
    }
    auto lookup = [&](Id id) { return token(memo.values.lookup(id)); };
    for (Id id : recipe.inputs) {
        if (memo.values.count(id)) { continue; }
        const auto& node = arena.nodes[id];
        auto found = mapping.find(node.value);
        const auto input = found == mapping.end() ? construction.input(node.value) : found->second;
        put(memo.values, id, arena.constructNode(node, construction, cut->getContext(), lookup, &input));
    }
    auto function = cut->getParentOfType<func::FuncOp>();
    if (!function) { return reject("regional expression emission requires an original function"); }
    if (arena.dominanceRoot != function.getOperation()) {
        arena.dominanceRoot = function.getOperation();
        arena.dominance = std::make_unique<DominanceInfo>(arena.dominanceRoot);
    }
    SmallVector<Id> pending{recipe.selected}, order;
    llvm::DenseSet<Id> seen;
    while (!pending.empty()) {
        const auto id = pending.pop_back_val();
        if (memo.values.count(id) || !seen.insert(id).second) { continue; }
        const auto& node = arena.nodes[id];
        if (node.kind == RegionExpressions::Kind::Input &&
            !detail::originalValueAvailable(node.value, cut, *arena.dominance)) {
            return reject("regional expression input is unavailable at endpoint cut");
        }
        arena.appendOperands(node, pending); order.push_back(id);
    }
    llvm::sort(order);
    for (Id id : order) {
        put(memo.values, id, arena.constructNode(arena.nodes[id], construction, cut->getContext(), lookup));
    }
    auto result = lookup(recipe.selected);
    if (foreignToken || !result) { return reject("prepared guarded expression lost its construction owner"); }
    return result;
}
FailureOr<detail::ScalarToken> detail::RegionConstructionAccess::construct(RegionExpressions& arena,
    const CutExpressionRecipe& recipe, ScalarConstruction& construction, Operation* cut,
    RegionConstructionContext& context, const RegionConstructionBindings* shared)
{
    const auto message = arena.emissionMessage;
    arena.emissionMessage.clear();
    auto result = schedule(arena, recipe, construction, cut, context, shared);
    arena.emissionMessage = message;
    return result;
}
FailureOr<Value> RegionExpressions::emitPreparedContextual(const CutExpressionRecipe& recipe,
    OpBuilder& builder, Operation* cut, CutEmission& context, const SharedCutBindings* shared)
{
    emissionMessage.clear();
    if (!cut || !builder.getInsertionBlock() || builder.getInsertionBlock()->getParent()) {
        emissionMessage = "prepared guarded expression requires a detached block"; return failure();
    }
    detail::ScalarConstruction construction(builder, cut->getLoc());
    auto value = [&](Id id) {
        return shared && shared->values ? construction.input(shared->values(id)) : detail::ScalarToken{};
    };
    auto input = [&](Value original) {
        return shared && shared->inputs ? construction.input(shared->inputs(original)) : detail::ScalarToken{};
    };
    detail::RegionConstructionBindings bindings{value, input};
    auto result = detail::RegionConstructionAccess::schedule(
        *this, recipe, construction, cut, context, shared ? &bindings : nullptr);
    return succeeded(result) ? FailureOr<Value>(result->value) : FailureOr<Value>(failure());
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
