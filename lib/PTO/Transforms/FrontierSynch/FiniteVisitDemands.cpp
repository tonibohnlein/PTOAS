// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteVisitDemands.h"
#include "SequenceAnalysisInternal.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool addCost(RegionalCost& target, const RegionalCost& source)
{
    for (auto member : {&RegionalCost::repeatedRegions, &RegionalCost::phaseDescriptions,
        &RegionalCost::children, &RegionalCost::cells, &RegionalCost::ports, &RegionalCost::crossings,
        &RegionalCost::physicalFragments, &RegionalCost::rotatingResidues, &RegionalCost::numericVisits,
        &RegionalCost::arithmeticRegions, &RegionalCost::boundaryBytes, &RegionalCost::selectorComparisons,
        &RegionalCost::crossingCandidates, &RegionalCost::expressionNodes, &RegionalCost::implicationChecks,
        &RegionalCost::numericalLeafQueries, &RegionalCost::numericalIndexOperations,
        &RegionalCost::numericalMerges, &RegionalCost::numericalReusedChildren}) {
        if (source.*member > UINT64_MAX - target.*member) { return false; }
        target.*member += source.*member;
    }
    return true;
}
class Constructor {
public:
    Constructor(func::FuncOp function, FiniteVisitDemands& result)
        : function(function), result(result), input(*result.original), arena(input.types.front().body.expressions),
          e(*arena) {}
    bool run()
    {
        context = input.context == RegionExpressions::invalid ? e.boolean(true) : input.context;
        if (!e.isBoolean(context)) { return fail("finite visit context must be a common immutable predicate"); }
        if (!makeViews()) { return false; }
        SequenceAnalysisState common(function, arena);
        setChildren(common, views);
        if (!common.importSummaries(false)) { return fail(common.error); }
        if (!qualifyPipes(common) || !qualifyCells(common) || !validatePrerequisites()) { return false; }
        common.costs.children = views.size();
        common.costs.ports = common.ports.size();
        common.costs.cells = common.cells.size();
        if (!addCost(result.cost.regional, common.costs)) { return fail("finite visit qualification cost overflow"); }
        const auto h = views.size();
        std::vector<std::vector<const FiniteVisitPrerequisite*>> supplied(h * h);
        for (const auto& edge : input.prerequisites) {
            supplied[edge.sourceType * h + edge.targetType].push_back(&edge);
        }
        for (uint32_t a = 0; a < h; ++a) {
            for (uint32_t b = 0; b < h; ++b) {
                if (!buildPair(a, b, supplied[a * h + b])) { return false; }
            }
        }
        return e.constructionError().empty() || fail(e.constructionError());
    }
private:
    func::FuncOp function;
    FiniteVisitDemands& result;
    const FiniteVisitInput& input;
    std::shared_ptr<RegionExpressions> arena;
    RegionExpressions& e;
    RegionExpressions::Id context = RegionExpressions::invalid;
    std::vector<RegionalAnalysis> views;
    bool fail(StringRef message) { result.error = message.str(); return false; }
    bool implies(Expr premise, Expr conclusion)
    {
        if (result.cost.qualificationChecks == UINT64_MAX) {
            fail("finite visit qualification count overflow"); return false;
        }
        ++result.cost.qualificationChecks;
        return e.implies(premise, conclusion) || e.constantUnder(premise, conclusion) == 1;
    }
    void setChildren(SequenceAnalysisState& state, ArrayRef<RegionalAnalysis> regions)
    {
        state.requiredOuterLoops.assign(input.enclosing.begin(), input.enclosing.end());
        state.reconstructPrerequisites = false;
        state.requireEndpoints = false;
        for (const auto& region : regions) {
            Child child;
            child.regional = region;
            child.anchors = region.anchors;
            state.children.push_back(std::move(child));
        }
    }
    bool represented(const RegionalAnalysis& body)
    {
        if (body.storageSelectors || !body.symbolicStorageEffects.empty() || !body.deferredAccessBoundary.empty()) {
            return fail("finite visit storage qualification needs an exact finite bridge projection");
        }
        std::set<std::size_t> covered;
        for (const auto& access : body.accessBoundary) {
            if (!access.representedByCells) {
                return fail("finite visit residual effect needs a certified bridge projection");
            }
            covered.insert(access.effect);
        }
        if (!body.accessModel) { return true; }
        for (const auto& anchor : body.anchors) {
            for (auto id : body.accessModel->effectsFor(anchor.phase)) {
                const auto& effect = body.accessModel->effects()[id];
                if (effect.rangesMaterialized && effect.ranges.empty()) { continue; }
                if (!covered.count(id)) {
                    return fail("finite visit omitted effect lacks a certified ownership projection");
                }
            }
        }
        return true;
    }
    bool makeViews()
    {
        SmallVector<const CompoundInstanceElement*> phases;
        const SyncStorageEffects* model = nullptr;
        bool allProjected = true;
        for (const auto& type : input.types) {
            const auto& body = type.body;
            if (body.expressions != arena) { return fail("finite visit types require one expression arena"); }
            for (const auto& anchor : body.anchors) {
                if (!anchor.phase || !anchor.phase->elementOp ||
                    anchor.phase->elementOp->getParentOfType<func::FuncOp>() != function) {
                    return fail("finite visit type has no original payload anchor in this function");
                }
                phases.push_back(anchor.phase);
            }
            if (body.accessModel) {
                if (model && model != body.accessModel) { return fail("finite visit shared effect contexts differ"); }
                model = body.accessModel;
            }
            auto view = body;
            // Type-presence masking changes exported selector predicates; the
            // original relation/certificate has not undergone that transform.
            view.arithmeticRelations.reset(); view.relations.reset();
            view.symbolicStorage.reset();
            if (type.projectedStorage) {
                if (!type.projectedStorage->owner) {
                    return fail("finite visit storage projection has no proof owner");
                }
                view.storageBoundary = type.projectedStorage->storageBoundary;
                view.accessBoundary.clear(); view.deferredAccessBoundary.clear();
                view.storageSelectors = {}; view.symbolicStorageEffects.clear();
            } else {
                allProjected = false;
                if (!represented(body)) { return false; }
            }
            if (!maskSelectors(view)) { return false; }
            views.push_back(std::move(view));
        }
        if (!allProjected && model && model->hasUniformRelationships(phases)) {
            return fail("finite visit unresolved cross-type relationships need certified bridge projections");
        }
        return true;
    }
    bool maskSelectors(RegionalAnalysis& view)
    {
        auto mask = [&](std::vector<RegionalSelector>& selectors, std::optional<uint32_t> pipe = {}) {
            for (auto& selector : selectors) {
                if (selector.event.type >= view.anchors.size() || !e.isBoolean(selector.present) ||
                    !validRegionalEvent(view, selector.event)) {
                    return fail("finite visit selector has an invalid occurrence or predicate");
                }
                if (pipe && static_cast<uint32_t>(view.anchors[selector.event.type].phase->kPipeValue) != *pipe) {
                    return fail("finite visit selector has an inconsistent pipe");
                }
                auto present = regionalPresence(view, selector.event);
                if (!present) { return fail("finite visit selector presence unavailable"); }
                selector.present = e.land(context, e.land(selector.present, *present));
            }
            return true;
        };
        for (auto& cell : view.storageBoundary) {
            if (!mask(cell.firstWriters) || !mask(cell.lastWriters)) { return false; }
            for (auto* side : {&cell.firstReaders, &cell.lastReaders}) {
                for (auto& [pipe, selectors] : *side) { if (!mask(selectors, pipe)) { return false; } }
            }
        }
        for (auto* side : {&view.firstPayloads, &view.lastPayloads}) {
            for (auto& [pipe, selectors] : *side) { if (!mask(selectors, pipe)) { return false; } }
        }
        return true;
    }
    std::optional<Expr> nonempty(SequenceAnalysisState& state, ArrayRef<Selected> selectors,
                                 std::optional<uint32_t> pipe = {})
    {
        auto value = e.boolean(false);
        for (auto selector : selectors) {
            if (pipe && state.pipe(selector.port) != *pipe) {
                fail("finite visit native selector has an inconsistent pipe"); return {};
            }
            const auto& port = state.ports[selector.port];
            auto present = regionalPresence(state.children[port.child].regional, port.event());
            if (!present) { fail("finite visit boundary presence query unavailable"); return {}; }
            value = e.lor(value, e.land(selector.present, *present));
        }
        return value;
    }
    bool qualifyPipes(SequenceAnalysisState& state)
    {
        std::set<uint32_t> pipes;
        for (const auto& child : state.children) {
            for (const auto& anchor : child.anchors) { pipes.insert(static_cast<uint32_t>(anchor.phase->kPipeValue)); }
        }
        for (auto pipe : pipes) {
            std::vector<Expr> first, last;
            auto any = e.boolean(false);
            for (std::size_t t = 0; t < views.size(); ++t) {
                auto a = nonempty(state, state.nativeFirst[t][pipe], pipe);
                auto b = nonempty(state, state.nativeLast[t][pipe], pipe);
                if (!a || !b) { return false; }
                first.push_back(*a); last.push_back(*b);
                any = e.lor(any, e.lor(*a, *b));
            }
            if (implies(context, e.lnot(any))) { continue; }
            for (std::size_t t = 0; t < views.size(); ++t) {
                if (!implies(e.land(context, any), e.land(first[t], last[t]))) {
                    return fail("finite visit types do not certify the same mandatory present pipes");
                }
            }
            result.pipes.push_back(pipe);
        }
        return result.error.empty();
    }
    bool qualifyCells(SequenceAnalysisState& state)
    {
        for (std::size_t cell = 0; cell < state.cells.size(); ++cell) {
            auto any = e.boolean(false);
            std::vector<Expr> refresh;
            for (std::size_t t = 0; t < views.size(); ++t) {
                const auto& boundary = state.boundaries[t][cell];
                auto a = nonempty(state, boundary.firstWriters), b = nonempty(state, boundary.lastWriters);
                if (!a || !b) { return false; }
                any = e.lor(any, e.lor(*a, *b));
                refresh.push_back(e.land(*a, *b));
            }
            if (implies(context, e.lnot(any))) { continue; }
            for (auto refreshed : refresh) {
                if (!implies(e.land(context, any), refreshed)) {
                    return fail("finite visit types do not certify a common persistent refresh set");
                }
            }
            result.persistentCells.push_back(state.cells[cell]);
        }
        return result.error.empty();
    }
    bool validatePrerequisites()
    {
        for (const auto& edge : input.prerequisites) {
            if (edge.sourceType >= views.size() || edge.targetType >= views.size() || !e.isBoolean(edge.guard) ||
                edge.source.kind != PeriodicEventKind::Completion || edge.target.kind != PeriodicEventKind::Start ||
                !validRegionalEvent(views[edge.sourceType], edge.source) ||
                !validRegionalEvent(views[edge.targetType], edge.target)) {
                return fail("finite visit prerequisite has an invalid type, event or predicate");
            }
        }
        return true;
    }
    bool buildPair(uint32_t a, uint32_t b, ArrayRef<const FiniteVisitPrerequisite*> supplied)
    {
        SequenceAnalysisState pair(function, arena);
        const RegionalAnalysis selected[] = {views[a], views[b]};
        setChildren(pair, selected);
        if (!pair.importSummaries(false)) { return fail(pair.error); }
        pair.bridges();
        if (!pair.error.empty()) { return fail(pair.error); }
        FiniteVisitBoundary boundary;
        boundary.sourceType = a; boundary.targetType = b;
        if (!addPrerequisites(pair, supplied, boundary) || !addNative(pair, boundary)) { return false; }
        // Numerical closure may consume its native edge vector. The immutable
        // exported native records have already been captured above.
        if (!pair.closure()) { return fail(pair.error); }
        pair.costs.children = 2;
        pair.costs.ports = pair.ports.size();
        pair.costs.cells = pair.cells.size();
        pair.costs.crossings = pair.crossings.size();
        for (const auto& edge : pair.crossings) {
            if (e.constantValue(edge.guard) == 0) { continue; }
            const auto& source = pair.ports[edge.source];
            const auto& target = pair.ports[edge.target];
            if (source.child != 0 || target.child != 1) { return fail("finite visit crossing left its pair frame"); }
            boundary.demands.push_back({source.event(PeriodicEventKind::Completion), target.event(),
                                       e.land(context, edge.guard), false, 1});
        }
        if (result.cost.pairReductions == UINT64_MAX || !addCost(result.cost.regional, pair.costs)) {
            return fail("finite visit pair cost overflow");
        }
        ++result.cost.pairReductions;
        result.boundaries.push_back(std::move(boundary));
        return true;
    }
    bool addPrerequisites(SequenceAnalysisState& pair, ArrayRef<const FiniteVisitPrerequisite*> supplied,
                          FiniteVisitBoundary& boundary)
    {
        for (const auto* edge : supplied) {
            auto source = regionalPresence(views[edge->sourceType], edge->source);
            auto target = regionalPresence(views[edge->targetType], edge->target);
            if (!source || !target) { return fail("finite visit prerequisite presence unavailable"); }
            auto guard = e.land(context, e.land(edge->guard, e.land(*source, *target)));
            auto x = pair.port(0, edge->source), y = pair.port(1, edge->target);
            if (!pair.error.empty()) { return fail(pair.error); }
            if (edge->native) {
                pair.nativeValueCrossings.push_back({x, y, guard});
                boundary.native.push_back({edge->source, edge->target, guard, true, 1});
            } else {
                pair.crossing({x, guard}, {y, e.boolean(true)});
            }
        }
        return true;
    }
    bool addNative(SequenceAnalysisState& pair, FiniteVisitBoundary& boundary)
    {
        for (auto pipe : result.pipes) {
            for (auto last : pair.nativeLast[0][pipe]) {
                for (auto first : pair.nativeFirst[1][pipe]) {
                    auto guard = e.land(context, e.land(last.present, first.present));
                    for (auto kind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                        boundary.native.push_back({pair.ports[last.port].event(kind),
                            pair.ports[first.port].event(kind), guard, true, 1});
                    }
                }
            }
        }
        return e.constructionError().empty() || fail(e.constructionError());
    }
};
} // namespace
FiniteVisitDemands buildFiniteVisitDemands(func::FuncOp function, FiniteVisitInput input)
{
    FiniteVisitDemands result;
    result.original = std::make_shared<const FiniteVisitInput>(std::move(input));
    const auto& types = result.original->types;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || types.empty() ||
        types.size() > UINT32_MAX || types.size() > result.boundaries.max_size() / types.size() ||
        !types.front().body.expressions || !types.front().body.expressions->constructionError().empty()) {
        result.error = "finite visit construction requires a valid common function, arena and finite type list";
        return result;
    }
    Operation* parent = nullptr;
    for (auto loop : result.original->enclosing) {
        if (!loop || loop->getParentOfType<func::FuncOp>() != function ||
            (parent && !parent->isProperAncestor(loop))) {
            result.error = "finite visit fixed enclosing loops must belong to one original nested frame";
            return result;
        }
        parent = loop;
    }
    if (parent) {
        for (const auto& type : types) {
            for (const auto& anchor : type.body.anchors) {
                if (!anchor.phase || !anchor.phase->elementOp || !parent->isProperAncestor(anchor.phase->elementOp)) {
                    result.error = "finite visit fixed enclosing frame does not contain every original type";
                    return result;
                }
            }
        }
    }
    // Query providers can memoize newly constructed roots. Keep the arena
    // append-only even when qualification fails, rather than rolling them back.
    const auto oldSize = types.front().body.expressions->size();
    Constructor constructor(function, result);
    if (!constructor.run()) { result.boundaries.clear(); }
    result.cost.regional.expressionNodes = types.front().body.expressions->size() - oldSize;
    return result;
}
} // namespace mlir::pto::frontiersynch
