// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Realize retained Boolean evaluations without evaluating opaque producers again.
#include "PrefixExpression.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
struct Binding {
    Expr input;
    scf::ForOp loop;
    Operation* definition;
    int64_t lower, induction;
    std::size_t valueSlot, validSlot;
};
std::optional<int64_t> literal(Value value)
{
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) { return {}; }
    return integer.getSExtValue();
}
bool sameFrame(scf::ForOp loop, Operation* cut, DominanceInfo& dominance)
{
    if (!loop->isProperAncestor(cut) && !dominance.properlyDominates(loop.getOperation(), cut)) { return false; }
    for (auto* parent = loop->getParentOp(); !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
        if (isa<scf::ForOp, scf::WhileOp>(parent) && !parent->isProperAncestor(cut)) { return false; }
    }
    return true;
}
std::vector<Binding> collectBindings(RegionExpressions& arena, Expr root, Operation* cut,
                                    InvocationPrefixCollection& collection)
{
    std::vector<Binding> bindings;
    DominanceInfo dominance(collection.function());
    for (auto [id, unused] : arena.referencedInputs(root)) {
        auto evaluation = arena.evaluation(id);
        if (!evaluation) { continue; }
        auto loop = dyn_cast_or_null<scf::ForOp>(evaluation->loop);
        auto* definition = evaluation->original.getDefiningOp();
        if (!loop || !definition || definition->getBlock() != loop.getBody() ||
            !collection.function()->isProperAncestor(loop) || !sameFrame(loop, cut, dominance)) { continue; }
        const auto ordinal = arena.constantValue(evaluation->ordinal);
        const auto lower = literal(loop.getLowerBound()), step = literal(loop.getStep());
        // Unit-step signed loops cannot wrap on the final increment: IV<upper
        // implies IV+1<=upper<=INT64_MAX. Other strides need a retained proof.
        if (!ordinal || !lower || step != 1 || !loop.getInductionVar().getType().isIndex()) { continue; }
        const APInt actual = APInt(128, static_cast<uint64_t>(*lower), true) + APInt(128, *ordinal);
        if (!actual.isSignedIntN(64)) { continue; }
        const auto induction = actual.getSExtValue();
        std::string error;
        auto slots = collection.snapshotPredicate(evaluation->original, loop, induction, error);
        if (failed(slots)) { continue; }
        bindings.push_back({id, loop, definition, *lower, induction, slots->first, slots->second});
    }
    return bindings;
}
// Each proof leaves every other historical input unconstrained. The premise
// contains only numeric control bounds, so the certificates compose when
// several unavailable evaluations are replaced simultaneously.
bool independentWhenUnavailable(RegionExpressions& arena, Expr root, Operation* cut, Binding binding)
{
    RegionExpressions::Transaction scratch(arena);
    auto available = arena.boolean(false), domain = arena.boolean(true);
    const auto selected = arena.constant(static_cast<uint64_t>(binding.induction));
    if (binding.loop->isProperAncestor(cut)) {
        const auto current = arena.input(binding.loop.getInductionVar());
        available = arena.slt(selected, current);
        DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
        if (dominance.properlyDominates(binding.definition, cut)) { available = arena.sle(selected, current); }
        domain = arena.land(arena.sle(arena.constant(static_cast<uint64_t>(binding.lower)), current),
                            arena.slt(current, arena.input(binding.loop.getUpperBound())));
    } else {
        available = arena.slt(selected, arena.input(binding.loop.getUpperBound()));
    }
    if (arena.constantUnder(domain, available) == 1) { return true; }
    RegionExpressions::Substitution no({{binding.input, arena.boolean(false)}});
    RegionExpressions::Substitution yes({{binding.input, arena.boolean(true)}});
    const auto a = arena.substitute(root, no), b = arena.substitute(root, yes);
    const auto differs = arena.land(domain, arena.land(arena.lnot(available), arena.lnot(arena.eq(a, b))));
    return arena.constantValue(differs) == 0 || arena.inactiveByThresholds(differs);
}
RegionExpressions::CutEmission contextFor(RegionExpressions& arena, Expr root, Operation* cut,
                                          const IRMapping& mapping)
{
    RegionExpressions::CutEmission context;
    DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
    SmallVector<Value> pending;
    for (auto [id, original] : arena.referencedInputs(root)) {
        pending.push_back(original);
        auto value = mapping.lookupOrNull(original);
        if (value && dominance.properlyDominates(original, cut)) {
            context.values[id] = value; context.cofactors[id] = id;
        }
    }
    // Bind only reached values and deterministic replay dependencies. Copying
    // the complete module-clone mapping per expression grows with unrelated IR.
    DenseSet<Value> seen;
    while (!pending.empty()) {
        auto value = pending.pop_back_val();
        if (!seen.insert(value).second) { continue; }
        if (dominance.properlyDominates(value, cut)) {
            if (auto replacement = mapping.lookupOrNull(value)) { context.inputs[value] = replacement; }
        } else if (auto* definition = value.getDefiningOp(); mlir::pto::detail::canReplayScalar(definition)) {
            llvm::append_range(pending, definition->getOperands());
        }
    }
    return context;
}
struct ExpressionRecipe {
    std::shared_ptr<RegionExpressions> arena;
    Expr root;
    Operation* cut;
    std::vector<Binding> bindings;
    std::optional<bool> historyProved;
    FailureOr<Value> emit(OpBuilder& builder, const IRMapping& mapping,
                          ArrayRef<Value> state, std::string& error)
    {
        auto& e = *arena;
        Block scratch;
        OpBuilder detached(builder.getContext()); detached.setInsertionPointToEnd(&scratch);
        auto context = contextFor(e, root, cut, mapping);
        auto value = e.emitContextual(root, detached, cut, context);
        if (failed(value)) {
            const std::string directError = e.lastEmissionError();
            scratch.clear(); detached.setInsertionPointToEnd(&scratch);
            if (e.inactiveByThresholds(root)) {
                RegionExpressions::CutEmission inactive;
                value = e.emitContextual(e.boolean(false), detached, cut, inactive);
            } else {
                if (!historyProved) {
                    historyProved = !bindings.empty() && llvm::all_of(bindings, [&](const Binding& binding) {
                        return independentWhenUnavailable(e, root, cut, binding);
                    });
                }
                if (!*historyProved) { error = directError; return failure(); }
                context = contextFor(e, root, cut, mapping);
                for (const auto& binding : bindings) {
                    if (binding.validSlot >= state.size() || !state[binding.valueSlot].getType().isInteger(1) ||
                        !state[binding.validSlot].getType().isInteger(1)) {
                        error = "historical expression lost its invocation state"; return failure();
                    }
                    // The independent cofactor proof justifies any defined bit
                    // while invalid. Validity is retained, never assumed true.
                    context.values[binding.input] = state[binding.valueSlot];
                    context.cofactors[binding.input] = binding.input;
                }
                value = e.emitContextual(root, detached, cut, context);
            }
            if (failed(value)) { error = e.lastEmissionError(); return failure(); }
        }
        builder.getInsertionBlock()->getOperations().splice(builder.getInsertionPoint(), scratch.getOperations());
        return *value;
    }
};
} // namespace
PrefixIncomingSelector preparePrefixExpression(std::shared_ptr<RegionExpressions> arena,
    Expr root, Operation* cut, InvocationPrefixCollection& collection)
{
    auto recipe = std::make_shared<ExpressionRecipe>();
    recipe->arena = std::move(arena); recipe->root = root; recipe->cut = cut;
    recipe->bindings = collectBindings(*recipe->arena, root, cut, collection);
    return [recipe](OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) {
        return recipe->emit(b, mapping, state, error);
    };
}
} // namespace mlir::pto::frontiersynch
