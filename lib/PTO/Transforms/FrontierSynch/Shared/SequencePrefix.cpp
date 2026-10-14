// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "InvocationPrefixCollection.h"
#include "PrefixExpression.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch {
// Construction evidence, not a general implication result. Imported selectors
// certify endpoint existence; bridges intersect their guards. Folding chooses
// only an active alternative, and both reducers only narrow the resulting
// guards. Thus every retained guard implies the selected endpoints exist.
// Keep this evidence with the finalized producer instead of reconstructing a
// child presence predicate at a folded coordinate and attempting a fresh proof.
struct SequenceCrossingPresence {
    using Edge = std::tuple<uint32_t, uint32_t, Expr>;
    using Point = std::tuple<uint32_t, uint32_t, Expr, std::vector<Expr>>;
    using Choice = std::tuple<uint32_t, Expr, uint32_t, uint32_t>;
    using Origin = std::tuple<const CompoundInstanceElement*, Operation*, uint32_t,
        Block*, Operation*, Block*, Operation*, std::vector<std::pair<scf::ForOp, int64_t>>,
        const RegionExpressions*, const SyncStorageEffects*, GMAliasPolicy, scf::ForOp,
        std::vector<scf::ForOp>, std::vector<uint64_t>, std::optional<Expr>, std::optional<Expr>>;
    static std::optional<Origin> origin(const SequenceAnalysisState& state, const Port& point)
    {
        if (point.child >= state.children.size()) { return std::nullopt; }
        const auto& child = state.children[point.child];
        const auto& region = child.regional;
        if (point.type >= child.anchors.size() ||
            (!region.occurrenceLoops.empty() && point.type >= region.occurrenceLoops.size()) ||
            (!region.outerLoops.empty() && point.type >= region.outerLoops.size()) ||
            (!region.outerDivisors.empty() && point.type >= region.outerDivisors.size())) { return std::nullopt; }
        const auto& anchor = child.anchors[point.type];
        if (!anchor.phase) { return std::nullopt; }
        std::vector<std::pair<scf::ForOp, int64_t>> fixed;
        for (const auto& coordinate : anchor.coordinates) { fixed.emplace_back(coordinate.loop, coordinate.induction); }
        return Origin(anchor.phase, anchor.phase->elementOp, static_cast<uint32_t>(anchor.phase->kPipeValue),
            anchor.before.block, anchor.before.before, anchor.after.block, anchor.after.before, std::move(fixed),
            region.expressions.get(), region.accessModel, region.gmAliasPolicy,
            region.occurrenceLoops.empty() ? scf::ForOp{} : region.occurrenceLoops[point.type],
            region.outerLoops.empty() ? std::vector<scf::ForOp>{} : region.outerLoops[point.type],
            region.outerDivisors.empty() ? std::vector<uint64_t>{} : region.outerDivisors[point.type],
            region.endpointSiteGuard, region.endpointInvocationGuard);
    }
    const SequenceAnalysisState* owner = nullptr;
    const RegionExpressions* arena = nullptr;
    const SyncInput* input = nullptr;
    func::FuncOp function;
    SmallVector<scf::ForOp> enclosing;
    std::vector<Edge> edges;
    std::vector<Point> points;
    std::vector<Origin> origins;
    std::vector<Choice> choices;
};
void SequenceAnalysisState::sealCrossingPresence()
{
    auto evidence = std::make_shared<SequenceCrossingPresence>();
    evidence->owner = this; evidence->arena = arena.get(); evidence->input = input;
    evidence->function = function; evidence->enclosing = requiredOuterLoops;
    for (const auto& edge : crossings) {
        if (edge.source >= ports.size() || edge.target >= ports.size()) { return; }
        evidence->edges.emplace_back(edge.source, edge.target, edge.guard);
    }
    for (const auto& point : ports) {
        auto origin = SequenceCrossingPresence::origin(*this, point);
        if (!origin) { return; }
        evidence->origins.push_back(std::move(*origin));
        evidence->points.emplace_back(point.child, point.type, point.ordinal, point.visits);
    }
    for (const auto& [id, choice] : portChoices) {
        evidence->choices.emplace_back(id, choice.choose, choice.yes, choice.no);
    }
    crossingPresence = std::move(evidence);
}
bool SequenceAnalysisState::hasCertifiedCrossingPresence() const
{
    const auto* proof = crossingPresence.get();
    if (!proof || proof->owner != this || proof->arena != arena.get() || proof->input != input ||
        proof->function != function || proof->enclosing != requiredOuterLoops || !error.empty() ||
        relationalResult || !expressions.constructionError().empty() || proof->edges.size() != crossings.size() ||
        proof->points.size() != ports.size() || proof->choices.size() != portChoices.size()) { return false; }
    for (std::size_t i = 0; i < crossings.size(); ++i) {
        const auto& edge = crossings[i];
        if (proof->edges[i] != SequenceCrossingPresence::Edge(edge.source, edge.target, edge.guard)) { return false; }
    }
    for (std::size_t i = 0; i < ports.size(); ++i) {
        const auto& point = ports[i];
        auto origin = SequenceCrossingPresence::origin(*this, point);
        if (!origin || *origin != proof->origins[i]) { return false; }
        if (proof->points[i] != SequenceCrossingPresence::Point(
                point.child, point.type, point.ordinal, point.visits)) { return false; }
    }
    std::size_t i = 0;
    for (const auto& [id, choice] : portChoices) {
        if (proof->choices[i++] != SequenceCrossingPresence::Choice(id, choice.choose, choice.yes, choice.no)) {
            return false;
        }
    }
    return true;
}
namespace {
bool originalPrefixAnchor(const TemplateEndpointAnchor& anchor)
{
    auto* op = anchor.phase ? anchor.phase->elementOp : nullptr;
    return op && anchor.before.block == op->getBlock() && anchor.before.before == op &&
           anchor.after.block == op->getBlock() && anchor.after.before == op->getNextNode();
}
// Save each primitive occurrence before evaluating any folded boundary choice.
// Actual execution establishes presence, but phase-view guards still identify
// which occurrence shares this original anchor. Preserve those guards and the
// primitive coordinates; defer only the folded choices to the consumer.
struct PrefixCapturedRank {
    std::optional<std::size_t> slot;
    PrefixIncomingSelector choose;
    uint32_t yes = 0, no = 0;
};
// Scoped to one sequence contribution and its append-only arena. Identical
// captures have the same initial value, exact source update and invocation
// reset, regardless of which consumer later selects that saved rank.
using PrimitiveCaptures = std::map<std::tuple<const CompoundInstanceElement*, Expr, Operation*>, std::size_t>;
LogicalResult capturePrimitivePrefixSource(
    const std::shared_ptr<SequenceAnalysisState>& owner, uint32_t id, const PrefixOccurrencePtr& occurrence,
    Operation* targetCut, InvocationPrefixCollection& collection,
    std::map<uint32_t, PrefixCapturedRank>& captured, PrimitiveCaptures& shared, std::string& error)
{
    if (!occurrence || occurrence->choose) { error = "prefix source has no primitive occurrence"; return failure(); }
    const auto& anchor = occurrence->anchor;
    if (!originalPrefixAnchor(anchor)) { error = "prefix source lacks its original cut"; return failure(); }
    const auto guard = occurrence->match;
    auto* sourceCut = anchor.phase->elementOp;
    auto& recipe = collection.recipe();
    scf::ForOp common;
    for (auto* parent = sourceCut->getParentOp(); parent; parent = parent->getParentOp()) {
        auto loop = dyn_cast<scf::ForOp>(parent);
        if (loop && loop->isProperAncestor(targetCut)) { common = loop; break; }
    }
    if (common) {
        if (!llvm::is_contained(owner->requiredOuterLoops, common)) {
            error = "prefix crossing needs a certified enclosing-visit rank adapter"; return failure();
        }
    }
    auto key = std::make_tuple(anchor.phase, guard, common ? common.getOperation() : nullptr);
    if (auto found = shared.find(key); found != shared.end()) {
        captured.emplace(id, PrefixCapturedRank{found->second});
        return success();
    }
    const auto slot = recipe.slots.size();
    auto type = IndexType::get(collection.function().getContext());
    recipe.slots.push_back({type, IntegerAttr::get(type, 0)});
    if (common) {
        recipe.actions.push_back({&common.getBody()->front(), PrivateStateCut::Before,
            [slot](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
                state[slot] = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 0);
                return success();
            }});
    }
    auto expression = preparePrefixExpression(owner->arena, guard, sourceCut, collection);
    if (failed(collection.source(anchor.phase,
        [expression, sourceCut, slot](OpBuilder& b, const IRMapping& mapping,
            MutableArrayRef<Value> state, Value ordinal, std::string& diagnostic) {
            auto selected = expression(b, mapping, state, diagnostic);
            if (failed(selected)) { return failure(); }
            state[slot] = b.create<arith::SelectOp>(sourceCut->getLoc(), *selected, ordinal, state[slot]);
            return success();
        }, error))) { return failure(); }
    shared.emplace(key, slot);
    captured.emplace(id, PrefixCapturedRank{slot});
    return success();
}
FailureOr<uint32_t> capturePrefixSource(
    const std::shared_ptr<SequenceAnalysisState>& owner, const PrefixOccurrencePtr& root, Operation* targetCut,
    InvocationPrefixCollection& collection, std::map<uint32_t, PrefixCapturedRank>& captured, PrimitiveCaptures& shared,
    std::string& error)
{
    // Preparation owns a separate topological directory. Imported child recipes
    // never mutate the retained ports, choices or sealed crossing evidence.
    std::map<const PrefixOccurrence*, uint32_t> ids;
    std::function<FailureOr<uint32_t>(const PrefixOccurrencePtr&)> visit =
        [&](const auto& node) -> FailureOr<uint32_t> {
        if (!node) {
            if (error.empty()) { error = "prefix source has no producer occurrence recipe"; }
            return failure();
        }
        if (auto found = ids.find(node.get()); found != ids.end()) { return found->second; }
        PrefixCapturedRank rank;
        if (node->choose) {
            auto yes = visit(node->yes), no = visit(node->no);
            if (failed(yes) || failed(no)) { return failure(); }
            rank.yes = *yes; rank.no = *no;
            rank.choose = preparePrefixExpression(owner->arena, *node->choose, targetCut, collection);
        }
        if (ids.size() >= UINT32_MAX) { error = "prefix occurrence identity overflow"; return failure(); }
        const auto id = static_cast<uint32_t>(ids.size());
        ids.emplace(node.get(), id);
        if (node->choose) { captured.emplace(id, std::move(rank)); }
        else if (failed(capturePrimitivePrefixSource(
                     owner, id, node, targetCut, collection, captured, shared, error))) {
            return failure();
        }
        return id;
    };
    return visit(root);
}
FailureOr<Value> selectedPrefixRank(uint32_t root,
    const std::map<uint32_t, PrefixCapturedRank>& captured, Operation* cut, OpBuilder& b,
    const IRMapping& mapping, ArrayRef<Value> state, std::string& error)
{
    // Port choices are topologically ordered by construction. Shared primitive
    // and choice nodes are evaluated once per consumer, without copying a tree.
    std::map<uint32_t, Value> values;
    for (const auto& [port, rank] : captured) {
        if (rank.slot) { values.emplace(port, state[*rank.slot]); continue; }
        auto condition = rank.choose(b, mapping, state, error);
        if (failed(condition)) { return failure(); }
        values.emplace(port, b.create<arith::SelectOp>(cut->getLoc(), *condition,
            values.at(rank.yes), values.at(rank.no)).getResult());
    }
    return values.at(root);
}
LogicalResult appendPrefixCrossing(
    const std::shared_ptr<SequenceAnalysisState>& owner, const Crossing& edge, InvocationPrefixCollection& collection,
    std::string& error, bool certifiedPresence, PrimitiveCaptures& shared)
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
    if (!certifiedPresence) {
        auto presence = regionalPresence(owner->children[a.child].regional, a.event(PeriodicEventKind::Completion));
        if (!presence || !e.implies(edge.guard, *presence)) {
            error = "prefix crossing lacks current-context source presence";
            return failure();
        }
    }
    auto p = static_cast<uint32_t>(aa.phase->kPipeValue), q = static_cast<uint32_t>(ab.phase->kPipeValue);
    PrefixOccurrenceCache occurrences;
    auto targetOccurrence = sequencePrefixOccurrence(owner, edge.target, occurrences, error);
    auto target = prefixOccurrenceMatch(e, targetOccurrence);
    if (!target) {
        if (error.empty()) { error = "prefix crossing has unavailable original coordinates"; }
        return failure();
    }
    auto* targetCut = ab.phase->elementOp;
    auto expression = preparePrefixExpression(
        owner->arena, e.select(edge.guard, *target, e.boolean(false)), targetCut, collection);
    if (p == q) {
        if (p == static_cast<uint32_t>(PIPE::PIPE_S)) {
            error = "scalar crossing requires its native prerequisite contract";
            return failure();
        }
        collection.recipe().actions.push_back(
            {targetCut, PrivateStateCut::Before,
             [expression, targetCut, p](
                 OpBuilder& b, const IRMapping& mapping, MutableArrayRef<Value> state, std::string& error) {
                 auto active = expression(b, mapping, state, error);
                 if (failed(active)) {
                     return failure();
                 }
                 if (matchPattern(*active, m_Zero())) { return success(); }
                 auto branch = b.create<scf::IfOp>(targetCut->getLoc(), *active, false);
                 OpBuilder::InsertionGuard restore(b);
                 b.setInsertionPointToStart(branch.thenBlock());
                 b.create<BarrierOp>(targetCut->getLoc(), PipeAttr::get(b.getContext(), static_cast<PIPE>(p)));
                 return success();
             }});
        return success();
    }
    std::map<uint32_t, PrefixCapturedRank> captured;
    auto sourceOccurrence = sequencePrefixOccurrence(owner, edge.source, occurrences, error);
    auto sourceId = capturePrefixSource(owner, sourceOccurrence, targetCut, collection, captured, shared, error);
    if (failed(sourceId)) { return failure(); }
    return collection.consumer(
        targetCut, {p, q},
        [expression, targetCut, source = *sourceId, captured = std::move(captured)](
            OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) -> FailureOr<Value> {
            auto selected = expression(b, mapping, state, error);
            if (failed(selected)) {
                return failure();
            }
            if (matchPattern(*selected, m_Zero())) {
                return b.create<arith::ConstantIndexOp>(targetCut->getLoc(), 0).getResult();
            }
            auto rank = selectedPrefixRank(source, captured, targetCut, b, mapping, state, error);
            if (failed(rank)) { return failure(); }
            auto zero = b.create<arith::ConstantIndexOp>(targetCut->getLoc(), 0);
            return b.create<arith::SelectOp>(targetCut->getLoc(), *selected, *rank, zero).getResult();
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
    // Check the sealed context once. Uncertified or changed graphs retain the
    // ordinary checked path. This evidence does not certify executable guards,
    // source ordinals, original cuts, or rank transport across invocations.
    const bool certifiedPresence = hasCertifiedCrossingPresence();
    PrimitiveCaptures captures;
    for (const auto& edge : crossings) {
        if (failed(appendPrefixCrossing(
                shared_from_this(), edge, collection, diagnostic, certifiedPresence, captures))) {
            return failure();
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
