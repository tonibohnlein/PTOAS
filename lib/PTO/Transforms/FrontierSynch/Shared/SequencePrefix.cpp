// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "InvocationPrefixCollection.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Dominance.h"
namespace mlir::pto::frontiersynch {
namespace {
FailureOr<Value> emitPrefixCrossing(
    RegionExpressions& e, Expr root, Operation* cut, OpBuilder& builder, const IRMapping& mapping, std::string& error)
{
    RegionExpressions::CutEmission context;
    DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
    for (const auto& item : mapping.getValueMap()) {
        if (dominance.properlyDominates(item.first, cut)) {
            context.inputs[item.first] = item.second;
        }
    }
    for (auto [id, original] : e.referencedInputs(root)) {
        auto value = mapping.lookupOrNull(original);
        if (value && dominance.properlyDominates(original, cut)) {
            context.values[id] = value;
            context.cofactors[id] = id;
        }
    }
    Block scratch;
    OpBuilder detached(builder.getContext());
    detached.setInsertionPointToEnd(&scratch);
    auto result = e.emitContextual(root, detached, cut, context);
    if (failed(result)) {
        error = e.lastEmissionError();
        return failure();
    }
    builder.getInsertionBlock()->getOperations().splice(builder.getInsertionPoint(), scratch.getOperations());
    return *result;
}
FailureOr<Expr> prefixPortGuard(SequenceAnalysisState& state, const Port& point, Expr guard)
{
    auto& e = state.expressions;
    const auto& child = state.children[point.child];
    const auto no = e.boolean(false);
    if (child.regional.endpointSiteGuard) {
        guard = e.select(guard, *child.regional.endpointSiteGuard, no);
    }
    if (child.regional.endpointInvocationGuard) {
        guard = e.select(guard, *child.regional.endpointInvocationGuard, no);
    }
    if (child.regional.endpointEventGuard) {
        auto value = child.regional.endpointEventGuard(point.event());
        if (!value) {
            return failure();
        }
        guard = e.select(guard, *value, no);
    }
    auto loop = child.regional.occurrenceLoops.empty() ? scf::ForOp{} : child.regional.occurrenceLoops[point.type];
    auto coordinate = [&](scf::ForOp loop, Expr selected, uint64_t divisor) -> bool {
        auto step = sequenceInteger(loop.getStep());
        if (!step || *step <= 0 || !divisor) {
            return false;
        }
        auto ordinal = e.div(e.sub(e.input(loop.getInductionVar()), e.input(loop.getLowerBound())), e.constant(*step));
        ordinal = e.div(ordinal, e.constant(divisor));
        guard = e.select(guard, e.eq(ordinal, selected), no);
        return true;
    };
    if (loop && !coordinate(loop, point.ordinal, 1)) {
        return failure();
    }
    if (!child.regional.outerLoops.empty()) {
        auto& loops = child.regional.outerLoops[point.type];
        if (loops.size() != point.visits.size()) {
            return failure();
        }
        for (std::size_t i = 0; i < loops.size(); ++i) {
            auto divisor = child.regional.outerDivisors.empty() ? 1 : child.regional.outerDivisors[point.type][i];
            if (!coordinate(loops[i], point.visits[i], divisor)) {
                return failure();
            }
        }
    }
    for (auto fixed : child.anchors[point.type].coordinates) {
        guard = e.select(guard, e.eq(e.input(fixed.loop.getInductionVar()), e.constant(fixed.induction)), no);
    }
    return guard;
}
bool originalPrefixAnchor(const TemplateEndpointAnchor& anchor)
{
    auto* op = anchor.phase ? anchor.phase->elementOp : nullptr;
    return op && anchor.before.block == op->getBlock() && anchor.before.before == op &&
           anchor.after.block == op->getBlock() && anchor.after.before == op->getNextNode();
}
LogicalResult appendPrefixCrossing(
    const std::shared_ptr<SequenceAnalysisState>& owner, const Crossing& edge, InvocationPrefixCollection& collection,
    std::string& error)
{
    auto& e = owner->expressions;
    if (edge.source >= owner->ports.size() || edge.target >= owner->ports.size()) {
        error = "prefix crossing refers to an invalid retained port";
        return failure();
    }
    const auto& a = owner->ports[edge.source];
    const auto& b = owner->ports[edge.target];
    const bool valid = a.child < owner->children.size() && b.child < owner->children.size() &&
                       a.type < owner->children[a.child].anchors.size() &&
                       b.type < owner->children[b.child].anchors.size();
    if (!valid) {
        error = "prefix crossing refers to an invalid original phase";
        return failure();
    }
    const auto& aa = owner->children[a.child].anchors[a.type];
    const auto& ab = owner->children[b.child].anchors[b.type];
    if (!originalPrefixAnchor(aa) || !originalPrefixAnchor(ab)) {
        error = "prefix crossing requires its original phase cut adapter";
        return failure();
    }
    auto presence = regionalPresence(owner->children[a.child].regional, a.event(PeriodicEventKind::Completion));
    if (!presence || !e.implies(edge.guard, *presence)) {
        error = "prefix crossing lacks current-context source presence";
        return failure();
    }
    auto p = static_cast<uint32_t>(aa.phase->kPipeValue), q = static_cast<uint32_t>(ab.phase->kPipeValue);
    auto target = prefixPortGuard(*owner, b, edge.guard);
    auto source = prefixPortGuard(*owner, a, e.boolean(true));
    if (failed(target) || failed(source)) {
        error = "prefix crossing has unavailable original coordinates";
        return failure();
    }
    auto* sourceCut = aa.phase->elementOp;
    auto* targetCut = ab.phase->elementOp;
    if (p == q) {
        if (p == static_cast<uint32_t>(PIPE::PIPE_S)) {
            error = "scalar crossing requires its native prerequisite contract";
            return failure();
        }
        collection.recipe().actions.push_back(
            {targetCut, PrivateStateCut::Before,
             [owner, root = *target, targetCut, p](
                 OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value>, std::string& error) {
                 auto active = emitPrefixCrossing(owner->expressions, root, targetCut, b, mapping, error);
                 if (failed(active)) {
                     return failure();
                 }
                 auto branch = b.create<scf::IfOp>(targetCut->getLoc(), *active, false);
                 OpBuilder::InsertionGuard restore(b);
                 b.setInsertionPointToStart(branch.thenBlock());
                 b.create<BarrierOp>(targetCut->getLoc(), PipeAttr::get(b.getContext(), static_cast<PIPE>(p)));
                 return success();
             }});
        return success();
    }
    auto& recipe = collection.recipe();
    const auto slot = recipe.slots.size();
    auto type = IndexType::get(collection.function().getContext());
    recipe.slots.push_back({type, IntegerAttr::get(type, 0)});
    // A shared enclosing visit is a fresh sequence context. Adapters spanning
    // different visits require their own rank transport and are not inferred.
    scf::ForOp common;
    for (auto* parent = sourceCut->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (loop && loop->isProperAncestor(targetCut)) {
            common = loop;
            break;
        }
    }
    if (common) {
        if (!llvm::is_contained(owner->requiredOuterLoops, common)) {
            error = "prefix crossing needs a certified enclosing-visit rank adapter";
            return failure();
        }
        recipe.actions.push_back(
            {&common.getBody()->front(), PrivateStateCut::Before,
             [slot](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
                 state[slot] = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);
                 return success();
             }});
    }
    if (failed(collection.source(
            aa.phase,
            [owner, root = *source, sourceCut, slot](
                OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value> state, Value ordinal,
                std::string& error) {
                auto selected = emitPrefixCrossing(owner->expressions, root, sourceCut, b, mapping, error);
                if (failed(selected)) {
                    return failure();
                }
                state[slot] = b.create<arith::SelectOp>(sourceCut->getLoc(), *selected, ordinal, state[slot]);
                return success();
            },
            error))) {
        return failure();
    }
    return collection.consumer(
        targetCut, {p, q},
        [owner, root = *target, targetCut, slot](
            OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) -> FailureOr<Value> {
            auto selected = emitPrefixCrossing(owner->expressions, root, targetCut, b, mapping, error);
            if (failed(selected)) {
                return failure();
            }
            auto zero = b.create<arith::ConstantIndexOp>(targetCut->getLoc(), 0);
            return b.create<arith::SelectOp>(targetCut->getLoc(), *selected, state[slot], zero).getResult();
        },
        error);
}
} // namespace
LogicalResult SequenceAnalysisState::collectPrefix(InvocationPrefixCollection& collection, std::string& diagnostic)
{
    if (function != collection.function()) {
        diagnostic = "sequence prefix collection belongs to a different invocation";
        return failure();
    }
    if (!error.empty()) {
        diagnostic = "sequence prefix collection retains an analysis failure: " + error;
        return failure();
    }
    if (relationalResult) {
        diagnostic = "sequence prefix collection requires a retained relational crossing adapter";
        return failure();
    }
    for (const auto& child : children) {
        if (child.regional.invocationPrefix) {
            if (failed(collectRegionalInvocationPrefix(child.regional, collection, diagnostic))) { return failure(); }
        } else if (child.prefixMathematics) {
            if (failed(collectInvocationPrefix(child.prefixMathematics, collection, diagnostic))) {
                return failure();
            }
        } else if (!child.explicitAnalysis.phases.empty()) {
            if (failed(appendExplicitPrefixRealization(child.explicitAnalysis, collection, diagnostic))) {
                return failure();
            }
        } else if (!child.anchors.empty()) {
            diagnostic = "sequence child lacks its exact retained prefix owner";
            return failure();
        }
    }
    for (const auto& edge : crossings) {
        if (failed(appendPrefixCrossing(shared_from_this(), edge, collection, diagnostic))) {
            return failure();
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
