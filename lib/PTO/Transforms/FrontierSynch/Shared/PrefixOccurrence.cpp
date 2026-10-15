// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PrefixOccurrence.h"
#include "OccurrenceBinding.h"
#include "InvocationPrefixCollection.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
namespace mlir::pto::frontiersynch {
namespace {
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
    std::string error;
    auto conditions = regionalOccurrenceConstraints(child.regional, point.event(), error);
    if (failed(conditions)) { return failure(); }
    for (auto condition : *conditions) { guard = e.select(guard, condition, no); }
    return guard;
}
// Callback semantics are immutable under the regional export contract. Adapters
// changing them must invalidate or explicitly transport the whole capability.
} // namespace
bool samePrefixOccurrenceFrame(const RegionalAnalysis& a, const RegionalAnalysis& b)
{
    if (a.relations != b.relations || a.arithmeticRelations != b.arithmeticRelations ||
        a.expressions != b.expressions || a.accessModel != b.accessModel || a.gmAliasPolicy != b.gmAliasPolicy ||
        a.occurrenceLoops != b.occurrenceLoops || a.outerLoops != b.outerLoops || a.outerDivisors != b.outerDivisors ||
        a.endpointSiteGuard != b.endpointSiteGuard || a.endpointInvocationGuard != b.endpointInvocationGuard ||
        bool(a.endpointEventGuard) != bool(b.endpointEventGuard) || a.anchors.size() != b.anchors.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.anchors.size(); ++i) {
        const auto& x = a.anchors[i]; const auto& y = b.anchors[i];
        if (x.phase != y.phase || x.before.block != y.before.block || x.before.before != y.before.before ||
            x.after.block != y.after.block || x.after.before != y.after.before ||
            x.coordinates.size() != y.coordinates.size()) { return false; }
        for (std::size_t j = 0; j < x.coordinates.size(); ++j) {
            if (x.coordinates[j].loop != y.coordinates[j].loop ||
                x.coordinates[j].induction != y.coordinates[j].induction) { return false; }
        }
    }
    return true;
}
namespace {
using EventKey = std::tuple<uint32_t, Expr, std::vector<Expr>>;
EventKey eventKey(const RegionalEvent& event) { return {event.type, event.ordinal, event.visits}; }
} // namespace
PrefixOccurrencePtr sequencePrefixOccurrence(const std::shared_ptr<SequenceAnalysisState>& owner,
    uint32_t port, PrefixOccurrenceCache& cache, std::string& error)
{
    if (auto found = cache.find(port); found != cache.end()) { return found->second; }
    if (port >= owner->ports.size()) { error = "prefix occurrence refers to an invalid port"; return {}; }
    const auto& point = owner->ports[port];
    if (point.child >= owner->children.size() || point.type >= owner->children[point.child].anchors.size()) {
        error = "prefix occurrence refers to an invalid original phase"; return {};
    }
    auto node = std::make_shared<PrefixOccurrence>();
    auto choice = owner->portChoices.find(port);
    if (choice != owner->portChoices.end()) {
        if (choice->second.yes >= port || choice->second.no >= port) {
            error = "prefix occurrence choice is not an acyclic original recipe"; return {};
        }
        node->choose = choice->second.choose;
        node->yes = sequencePrefixOccurrence(owner, choice->second.yes, cache, error);
        node->no = sequencePrefixOccurrence(owner, choice->second.no, cache, error);
        if (!node->yes || !node->no) { return {}; }
    } else {
        const auto& child = owner->children[point.child];
        const auto& region = child.regional;
        if (const auto& capability = region.invocationPrefix) {
            if (!capability->occurrence || !capability->matchesOccurrenceFrame ||
                !capability->matchesOccurrenceFrame(region)) {
                error = "child occurrence recipe has unavailable or stale export context"; return {};
            }
            auto nested = capability->occurrence(point.event(), error);
            if (nested) { cache.emplace(port, nested); }
            return nested;
        }
        if (child.prefixMathematics && child.prefixMathematics->sequenceDemands) {
            const auto& retained = *child.prefixMathematics->sequenceDemands;
            auto canonical = sequenceRegionalResult(retained);
            if (!retained.state || !samePrefixOccurrenceFrame(canonical, region)) {
                error = "retained sequence occurrence has a transformed export frame"; return {};
            }
            // Borrowed leaf analyses can lack a detached prefix capability.
            // Their retained sequence owner still supplies the original choice
            // construction; never reinterpret its folded ordinal as a leaf.
            RegionalPrefixContribution recipe;
            attachSequenceOccurrenceRecipes(canonical, retained.state, recipe);
            auto nested = recipe.occurrence(point.event(), error);
            if (nested) { cache.emplace(port, nested); }
            return nested;
        }
        // This is an original leaf adapter, not a flattened composed export.
        // Composed mathematics must supply its producer's occurrence recipe.
        if (child.prefixMathematics && (child.prefixMathematics->sequenceDemands ||
            child.prefixMathematics->identityComposition || child.prefixMathematics->regionalDemands)) {
            error = "composed child has no executable occurrence recipe: " + child.prefixMathematics->backend;
            return {};
        }
        auto guard = prefixPortGuard(*owner, point, owner->yes());
        if (failed(guard)) { error = "primitive occurrence has unavailable original coordinates"; return {}; }
        node->anchor = child.anchors[point.type]; node->match = *guard;
    }
    cache.emplace(port, node);
    return node;
}
std::optional<Expr> prefixOccurrenceMatch(RegionExpressions& e, const PrefixOccurrencePtr& root)
{
    if (!root) { return {}; }
    std::map<const PrefixOccurrence*, Expr> values;
    std::function<std::optional<Expr>(const PrefixOccurrencePtr&)> visit =
        [&](const auto& node) -> std::optional<Expr> {
        if (!node) { return {}; }
        if (auto found = values.find(node.get()); found != values.end()) { return found->second; }
        auto result = node->match;
        if (node->choose) {
            auto yes = visit(node->yes), no = visit(node->no);
            if (!yes || !no) { return {}; }
            result = e.select(*node->choose, *yes, *no);
        }
        if (result >= e.size() || !e.isBoolean(result)) { return {}; }
        values.emplace(node.get(), result); return result;
    };
    return visit(root);
}
PrefixOccurrencePtr restrictPrefixOccurrence(RegionExpressions& e,
    const PrefixOccurrencePtr& root, Expr activity, PrefixOccurrenceRestrictionWork& work)
{
    accumulateCost(work.requests, 1);
    if (!root || activity >= e.size() || !e.isBoolean(activity)) { return root; }
    RegionExpressions::Transaction scratch(e);
    if (e.constantValue(activity) == 0) { return {}; }
    // One immutable activity premise per request keeps shared occurrence DAGs
    // shared. Do not enumerate path-dependent premises through a choice DAG.
    std::map<const PrefixOccurrence*, PrefixOccurrencePtr> retained;
    std::map<Expr, std::optional<uint64_t>> answers;
    std::function<PrefixOccurrencePtr(const PrefixOccurrencePtr&)> visit =
        [&](const auto& node) -> PrefixOccurrencePtr {
        if (!node || !node->choose) { return node; }
        if (auto found = retained.find(node.get()); found != retained.end()) { return found->second; }
        accumulateCost(work.choices, 1);
        auto [answer, fresh] = answers.try_emplace(*node->choose);
        if (fresh) { answer->second = e.constantUnder(activity, *node->choose, &work.proofVisits); }
        PrefixOccurrencePtr result;
        if (answer->second && e.constructionError().empty()) {
            accumulateCost(work.pruned, 1);
            result = visit(*answer->second ? node->yes : node->no);
        } else {
            accumulateCost(work.residual, 1);
            auto yes = visit(node->yes), no = visit(node->no);
            if (yes == node->yes && no == node->no) { result = node; }
            else {
                auto copy = std::make_shared<PrefixOccurrence>(*node);
                copy->yes = std::move(yes); copy->no = std::move(no);
                result = std::move(copy);
            }
        }
        retained.emplace(node.get(), result);
        return result;
    };
    // Returned nodes refer exclusively to original selector/match IDs; proof
    // temporaries and diagnostics are discarded before capture preparation.
    return visit(root);
}
void attachSequenceOccurrenceRecipes(const RegionalAnalysis& region,
    const std::shared_ptr<SequenceAnalysisState>& state, RegionalPrefixContribution& result)
{
    auto snapshot = std::make_shared<RegionalAnalysis>(region);
    snapshot->invocationPrefix.reset();
    result.matchesOccurrenceFrame = [snapshot](const RegionalAnalysis& current) {
        return samePrefixOccurrenceFrame(*snapshot, current);
    };
    std::vector<uint32_t> starts;
    uint32_t start = 0;
    for (const auto& child : state->children) { starts.push_back(start); start += child.anchors.size(); }
    std::map<EventKey, uint32_t> directory;
    for (uint32_t id = 0; id < state->ports.size(); ++id) {
        auto event = state->ports[id].event(); event.type += starts[state->ports[id].child];
        directory.emplace(eventKey(event), id);
    }
    result.occurrence = [state, directory = std::move(directory)](RegionalEvent event, std::string& error) {
        auto found = directory.find(eventKey(event));
        if (found == directory.end()) { error = "occurrence is absent from the producer's frozen export directory";
            return PrefixOccurrencePtr{}; }
        // Resolution may run inside a preparation transaction. Only the frozen
        // port directory survives; newly constructed expression IDs must not.
        PrefixOccurrenceCache cache;
        return sequencePrefixOccurrence(state, found->second, cache, error);
    };
}
void liftSingletonPrefixContribution(const RegionalAnalysis& body, RegionalAnalysis& lifted)
{
    const auto original = body.invocationPrefix;
    if (!original || !original->occurrence || !original->matchesOccurrenceFrame ||
        !original->matchesOccurrenceFrame(body) || body.anchors.size() != lifted.anchors.size()) { return; }
    auto result = std::make_shared<RegionalPrefixContribution>(*original);
    auto snapshot = std::make_shared<RegionalAnalysis>(lifted); snapshot->invocationPrefix.reset();
    result->matchesOccurrenceFrame = [snapshot](const RegionalAnalysis& current) {
        return samePrefixOccurrenceFrame(*snapshot, current);
    };
    result->occurrence = [original, snapshot](RegionalEvent event, std::string& error) -> PrefixOccurrencePtr {
        if (!validRegionalEvent(*snapshot, event) || event.visits.empty()) {
            error = "singleton occurrence recipe has an invalid lifted coordinate"; return {};
        }
        const auto coordinate = event.visits.front(); event.visits.erase(event.visits.begin());
        const auto type = event.type;
        auto bodyRecipe = original->occurrence(event, error);
        if (!bodyRecipe) { return {}; }
        auto& e = *snapshot->expressions;
        if (snapshot->outerLoops.empty() || snapshot->outerLoops[type].empty()) { return {}; }
        auto loop = snapshot->outerLoops[type].front();
        const auto divisor = snapshot->outerDivisors.empty() ? 1 : snapshot->outerDivisors[type].front();
        auto actual = originalLoopOrdinal(e, loop, divisor);
        if (!actual) { error = "singleton occurrence recipe has an unsupported loop coordinate"; return {}; }
        const auto match = e.eq(*actual, coordinate);
        std::map<const PrefixOccurrence*, PrefixOccurrencePtr> copies;
        std::function<PrefixOccurrencePtr(const PrefixOccurrencePtr&)> lift =
            [&](const auto& node) -> PrefixOccurrencePtr {
            if (auto found = copies.find(node.get()); found != copies.end()) { return found->second; }
            auto copy = std::make_shared<PrefixOccurrence>(*node);
            if (node->choose) { copy->yes = lift(node->yes); copy->no = lift(node->no); }
            else { copy->match = e.land(match, node->match); }
            copies.emplace(node.get(), copy); return copy;
        };
        return lift(bodyRecipe);
    };
    lifted.invocationPrefix = std::move(result);
}
} // namespace mlir::pto::frontiersynch
