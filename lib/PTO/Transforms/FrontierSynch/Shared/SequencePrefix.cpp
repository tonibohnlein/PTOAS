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
#include "PrefixExpressionProfile.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PrefixDemand.h"
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
    auto expression = preparePrefixExpression(owner->arena, guard, sourceCut, collection, "sequence-source-match");
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
            rank.choose =
                preparePrefixExpression(owner->arena, *node->choose, targetCut, collection, "sequence-source-choice");
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
FailureOr<Expr> prefixCrossingRoot(
    const std::shared_ptr<SequenceAnalysisState>& owner, const Crossing& edge, bool certifiedPresence,
    PrefixOccurrenceCache& occurrences, std::string& error)
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
    auto target = prefixOccurrenceMatch(e, sequencePrefixOccurrence(owner, edge.target, occurrences, error));
    if (!target) {
        if (error.empty()) { error = "prefix crossing has unavailable original coordinates"; }
        return failure();
    }
    return e.select(edge.guard, *target, e.boolean(false));
}
LogicalResult appendPrefixCrossing(
    const std::shared_ptr<SequenceAnalysisState>& owner, const Crossing& edge, InvocationPrefixCollection& collection,
    PrefixIncomingSelector expression, const PrefixOccurrencePtr& originalSource,
    const PrefixOccurrencePtr& selectedSource, PrimitiveCaptures& shared, std::string& error)
{
    // The complete batch has already validated ports, cuts and presence before
    // any root is prepared. Share its producer-owned occurrence directory.
    const auto& a = owner->ports[edge.source];
    const auto& b = owner->ports[edge.target];
    const auto& aa = owner->children[a.child].anchors[a.type];
    const auto& ab = owner->children[b.child].anchors[b.type];
    auto p = static_cast<uint32_t>(aa.phase->kPipeValue), q = static_cast<uint32_t>(ab.phase->kPipeValue);
    auto* targetCut = ab.phase->elementOp;
    if (p == q) {
        return appendPrefixIncoming(collection, targetCut, {p, q}, std::move(expression), {}, error);
    }
    // Publications define the executed rank namespace. Even an unused capture
    // leaf must retain its original publication and counter increment.
    std::set<const PrefixOccurrence*> seen;
    std::vector<PrefixOccurrencePtr> pending{originalSource};
    while (!pending.empty()) {
        auto node = pending.back(); pending.pop_back();
        if (!node) { error = "prefix publication has an incomplete source choice"; return failure(); }
        if (!seen.insert(node.get()).second) { continue; }
        if (node->choose) {
            if (*node->choose >= owner->arena->size() || !owner->arena->isBoolean(*node->choose)) {
                error = "prefix publication has an invalid source choice"; return failure();
            }
            pending.push_back(node->no); pending.push_back(node->yes); continue;
        }
        if (!originalPrefixAnchor(node->anchor) || node->match >= owner->arena->size() ||
            !owner->arena->isBoolean(node->match) || failed(collection.source(node->anchor.phase, {}, error))) {
            if (error.empty()) { error = "prefix publication lacks its original source cut"; }
            return failure();
        }
    }
    if (!selectedSource) {
        return appendPrefixIncoming(collection, targetCut, {p, q}, std::move(expression),
            [targetCut](OpBuilder& builder, const IRMapping&, ArrayRef<Value>, std::string&) -> FailureOr<Value> {
                return builder.create<arith::ConstantIndexOp>(targetCut->getLoc(), 0).getResult();
            }, error);
    }
    std::map<uint32_t, PrefixCapturedRank> captured;
    auto sourceId = capturePrefixSource(owner, selectedSource, targetCut, collection, captured, shared, error);
    if (failed(sourceId)) { return failure(); }
    return appendPrefixIncoming(
        collection, targetCut, {p, q}, std::move(expression),
        [targetCut, source = *sourceId, captured = std::move(captured)](
            OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) {
            return selectedPrefixRank(source, captured, targetCut, b, mapping, state, error);
        },
        error);
}
std::string originalLocation(Operation* operation)
{
    if (!operation) { return {}; }
    std::string result;
    llvm::raw_string_ostream out(result);
    out << operation->getName() << " ";
    operation->getLoc().print(out);
    return result;
}
std::string originalValue(Value value)
{
    if (auto result = dyn_cast<OpResult>(value)) {
        return originalLocation(result.getOwner()) + " result " + std::to_string(result.getResultNumber());
    }
    auto argument = dyn_cast<BlockArgument>(value);
    return argument ? originalLocation(argument.getOwner()->getParentOp()) +
        " argument " + std::to_string(argument.getArgNumber()) : std::string{};
}
class QueryLineageSnapshots {
public:
    explicit QueryLineageSnapshots(const SequenceAnalysisState& owner) : owner(owner) {}
    PrefixExpressionProfile::QueryEvent event(const SequenceAnalysisState::EventKey& key)
    {
        PrefixExpressionProfile::QueryEvent result;
        const auto& [child, type, ordinal, kind, visits] = key;
        result.type = type; result.kind = kind == PeriodicEventKind::Start ? "start" : "completion";
        result.ordinal = coordinate(ordinal); result.visitCount = visits.size();
        for (auto visit : ArrayRef<Expr>(visits).take_front(PrefixExpressionProfile::maxFrameCoordinates)) {
            result.visits.push_back(coordinate(visit));
        }
        const auto& source = owner.children[child];
        if (type < source.anchors.size() && source.anchors[type].phase) {
            result.anchor = originalLocation(source.anchors[type].phase->elementOp);
        }
        const auto& region = source.regional;
        if (type < region.occurrenceLoops.size()) {
            auto loop = region.occurrenceLoops[type];
            result.occurrenceLoop = originalLocation(loop.getOperation());
        }
        if (type < region.outerLoops.size()) {
            result.outerLoopCount = region.outerLoops[type].size();
            for (auto loop : ArrayRef<scf::ForOp>(region.outerLoops[type]).take_front(
                     PrefixExpressionProfile::maxFrameCoordinates)) {
                result.outerLoops.push_back(originalLocation(loop.getOperation()));
            }
        }
        return result;
    }
private:
    const SequenceAnalysisState& owner;
    std::map<Expr, PrefixExpressionProfile::Coordinate> coordinates;
    const PrefixExpressionProfile::Coordinate& coordinate(Expr root)
    {
        auto [position, fresh] = coordinates.try_emplace(root);
        auto& result = position->second;
        if (!fresh) { return result; }
        result.expression = root; result.constant = owner.expressions.constantValue(root);
        auto inputs = owner.expressions.referencedInputs(ArrayRef<Expr>(root), &result.nodes);
        result.inputCount = inputs.size();
        for (auto [id, value] : ArrayRef<std::pair<Expr, Value>>(inputs).take_front(
                 PrefixExpressionProfile::maxCoordinateInputs)) {
            PrefixExpressionProfile::CoordinateInput input;
            input.expression = id; input.original = originalValue(value);
            if (auto evaluation = owner.expressions.evaluation(id)) {
                input.original = originalValue(evaluation->original);
                input.evaluationLoop = originalLocation(evaluation->loop);
                input.evaluationOrdinal = evaluation->ordinal;
            }
            result.inputs.push_back(std::move(input));
        }
        return result;
    }
};
void annotateIncomingLineage(const SequenceAnalysisState& owner,
    ArrayRef<std::pair<Operation*, std::size_t>> requests,
    const std::map<Operation*, SmallVector<Expr>>& roots, InvocationPrefixCollection& collection)
{
    if (!collection.hasDetailedProfile() || !collection.recipe().expressionProfile) { return; }
    using Label = std::tuple<uint32_t, std::string, std::string>;
    std::map<Expr, std::map<Label, uint64_t>> queryOutputs;
    using CacheEntry = decltype(owner.reachabilityCache)::value_type;
    std::map<Expr, std::vector<const CacheEntry*>> queryPairs;
    QueryLineageSnapshots snapshots(owner);
    // Reuse the retained cache, including directly composed singleton bodies.
    // Count same-child cached query pairs with nonconstant outputs. Interned
    // aliases may share roots: these counts are not an exclusive work partition.
    for (const auto& entry : owner.reachabilityCache) {
        const auto& [pair, root] = entry;
        const auto child = std::get<0>(pair.first);
        if (child != std::get<0>(pair.second) || child >= owner.children.size() ||
            owner.expressions.constantValue(root)) { continue; }
        const auto& source = owner.children[child];
        auto mathematics = source.prefixMathematics ? source.prefixMathematics :
            source.regional.retainedMathematics.lock();
        const auto producer = mathematics ? mathematics->backend : "unlabelled-regional";
        const std::string kinds =
            (std::get<3>(pair.first) == PeriodicEventKind::Start ? "start" : "completion") +
            std::string("->") +
            (std::get<3>(pair.second) == PeriodicEventKind::Start ? "start" : "completion");
        accumulateCost(queryOutputs[root][{child, producer, kinds}], 1);
        queryPairs[root].push_back(&entry);
    }
    std::map<Expr, PrefixExpressionProfile::SequenceLineage> byRoot;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto [cut, output] = requests[i];
        const auto root = roots.at(cut)[output];
        auto [position, fresh] = byRoot.try_emplace(root);
        auto& lineage = position->second;
        if (fresh) {
            lineage.origin = owner.constructionOrigin;
            lineage.anchor = originalLocation(owner.constructionAnchor);
            if (owner.diagnosticRegion) { lineage.region = *owner.diagnosticRegion; }
            for (auto loop : owner.requiredOuterLoops) {
                lineage.frame.push_back(originalLocation(loop.getOperation()));
            }
            lineage.cachedQueries = owner.reachabilityCache.size();
            lineage.entries = owner.crossingQueryWork.entries;
            lineage.prefixQueries = owner.crossingQueryWork.prefixQueries;
            lineage.suffixQueries = owner.crossingQueryWork.suffixQueries;
            std::map<Label, uint64_t> groups;
            lineage.sourceNodes = owner.expressions.visitReferenced(ArrayRef<Expr>(root), [&](Expr id) {
                auto found = queryOutputs.find(id);
                if (found == queryOutputs.end()) { return; }
                for (const auto& [label, count] : found->second) {
                    accumulateCost(groups[label], count);
                    accumulateCost(lineage.reachableQueries, count);
                }
                for (const auto* entry : queryPairs.at(id)) {
                    if (lineage.queries.size() == PrefixExpressionProfile::maxQueryDetails) {
                        accumulateCost(lineage.omittedQueryDetails, 1); continue;
                    }
                    const auto child = std::get<0>(entry->first.first);
                    const auto& source = owner.children[child];
                    auto mathematics = source.prefixMathematics ? source.prefixMathematics :
                        source.regional.retainedMathematics.lock();
                    lineage.queries.push_back({child, id, mathematics ? mathematics->backend : "unlabelled-regional",
                        snapshots.event(entry->first.first), snapshots.event(entry->first.second)});
                }
            });
            lineage.queryGroups = groups.size();
            std::vector<std::pair<Label, uint64_t>> ranked(groups.begin(), groups.end());
            llvm::stable_sort(ranked, [](const auto& a, const auto& b) { return a.second > b.second; });
            for (const auto& [label, count] : ranked) {
                if (lineage.groups.size() == 16) { accumulateCost(lineage.omittedQueryPairs, count); continue; }
                const auto& [child, producer, kinds] = label;
                lineage.groups.push_back({child, count, producer, kinds});
            }
        }
        const auto& port = owner.ports[owner.crossings[i].source];
        lineage.source = originalLocation(owner.children[port.child].anchors[port.type].phase->elementOp);
        collection.recipe().expressionProfile->annotateSequence(owner.arena.get(), root, cut, lineage);
    }
}
} // namespace
LogicalResult collectSequencePrefixChildren(
    const SequenceAnalysisState& state, InvocationPrefixCollection& collection, std::string& diagnostic)
{
    for (const auto& child : state.children) {
        if (child.regional.invocationPrefix) {
            if (failed(collectRegionalInvocationPrefix(child.regional, collection, diagnostic))) { return failure(); }
        } else if (child.prefixMathematics) {
            if (failed(collectInvocationPrefix(child.prefixMathematics, collection, diagnostic))) {
                return failure();
            }
        } else if (child.finite && child.finite->concrete()) {
            if (failed(appendExplicitPrefixRealization(*child.finite->concrete(), collection, diagnostic))) {
                return failure();
            }
        } else if (!child.anchors.empty()) {
            diagnostic = "sequence child lacks its exact retained prefix owner";
            return failure();
        }
    }
    return success();
}
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
        auto contribution = sequencePrefixContribution(*relationalResult, shared_from_this());
        if (!contribution) {
            diagnostic = "sequence prefix collection requires a supported relational crossing adapter";
            return failure();
        }
        if (failed(collection.bindInput(contribution->input, diagnostic))) {
            return failure();
        }
        return contribution->collect(collection, diagnostic);
    }
    if (failed(collectSequencePrefixChildren(*this, collection, diagnostic))) {
        return failure();
    }
    // Check the sealed context once. Uncertified or changed graphs retain the
    // ordinary checked path. This evidence does not certify executable guards,
    // source ordinals, original cuts, or rank transport across invocations.
    const bool certifiedPresence = hasCertifiedCrossingPresence();
    PrefixOccurrenceCache occurrences;
    std::map<Operation*, SmallVector<Expr>> roots;
    std::vector<std::pair<Operation*, std::size_t>> requests;
    struct SourceRequest { PrefixOccurrencePtr original, selected; };
    std::vector<SourceRequest> sources;
    PrefixOccurrenceRestrictionWork restrictionWork;
    for (const auto& edge : crossings) {
        auto root = prefixCrossingRoot(shared_from_this(), edge, certifiedPresence, occurrences, diagnostic);
        if (failed(root)) {
            return failure();
        }
        const auto& target = ports[edge.target];
        auto* cut = children[target.child].anchors[target.type].phase->elementOp;
        auto& batch = roots[cut];
        requests.emplace_back(cut, batch.size());
        batch.push_back(*root);
        SourceRequest source;
        if (pipe(edge.source) != pipe(edge.target)) {
            source.original = sequencePrefixOccurrence(shared_from_this(), edge.source, occurrences, diagnostic);
            if (!source.original) { return failure(); }
            source.selected = restrictPrefixOccurrence(expressions, source.original, *root, restrictionWork);
        }
        sources.push_back(std::move(source));
    }
    std::map<Operation*, std::vector<PrefixIncomingSelector>> expressionsAtCuts;
    for (const auto& [cut, outputs] : roots) {
        expressionsAtCuts.emplace(cut, preparePrefixExpressions(arena, outputs, cut, collection, "sequence-incoming"));
    }
    annotateIncomingLineage(*this, requests, roots, collection);
    if (collection.recipe().expressionProfile) {
        collection.recipe().expressionProfile->recordOccurrenceRestriction(restrictionWork);
    }
    PrimitiveCaptures captures;
    for (std::size_t i = 0; i < crossings.size(); ++i) {
        const auto [cut, output] = requests[i];
        if (failed(appendPrefixCrossing(
                shared_from_this(), crossings[i], collection, expressionsAtCuts.at(cut)[output],
                sources[i].original, sources[i].selected, captures, diagnostic))) {
            return failure();
        }
    }
    return success();
}
} // namespace mlir::pto::frontiersynch
