// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_DIRECTIONALCOVERING_H
#define PTO_FRONTIERSYNCH_DIRECTIONALCOVERING_H
#include "OriginalIntervals.h"
#include "PTO/Transforms/FrontierSynch/CoveringBoundary.h"
#include <functional>
#include <map>
#include <set>

namespace mlir::pto::frontiersynch::detail {
// NoHit is a proved exclusion from the complete imported may-effects. May and
// Unknown both block trimming; neither proves participation. Must additionally
// proves a matching access on every visit of this leaf, not full completion.
enum class CoveringMatch { NoHit, May, Must, Unknown };

// Production I.3 core, independent of MLIR. The borrowed original body, graph
// and callback inputs must remain immutable and outlive this index. The adapter supplies complete
// original effects and finite-repeat qualification. No selected plan, endpoint
// menu, observation valuation, dynamic unrolling or event resource is an input.
//
// A lane is a sequence in ONE original control visit. Sequence wrappers are
// flattened only in this view. A choice/repeat remains an atomic child with its
// own child lanes. Two cuts in one lane delimit the supplied SESE fragment;
// sibling-arm cuts and backedge-crossing intervals do not acquire such a proof
// merely because they are reachable in the conservative control graph.
class DirectionalCovering {
public:
    using Match = std::function<CoveringMatch(std::size_t, const OriginalAccessSelector&)>;
    using FiniteRepeat = std::function<bool(const Region&)>;
    DirectionalCovering(
        const Region& body, const ControlGraph& graph, OriginalProgramVersion version, Match match,
        FiniteRepeat finiteRepeat)
        : graph(graph), version(std::move(version)), match(std::move(match)), finiteRepeat(std::move(finiteRepeat))
    {
        lanes.emplace_back();
        lanes[0].entry = OriginalCut::scope(NoControlId, OriginalCut::Before);
        lanes[0].exit = OriginalCut::scope(NoControlId, OriginalCut::After);
        recordCut(lanes[0].entry, 0);
        // Same legacy flat-fixture convention as buildControlGraph.
        if (body.kind == Region::Sequence && body.children.empty() && !graph.operationOwners.empty()) {
            flat = body;
            for (std::size_t op = 0; op < graph.operationOwners.size(); ++op) {
                Region leaf;
                leaf.kind = Region::Operation;
                leaf.operation = op;
                flat.children.push_back(std::move(leaf));
            }
            append(flat, 0);
        } else {
            append(body, 0);
        }
        recordCut(lanes[0].exit, 0);
    }
    DirectionalCovering(const DirectionalCovering&) = delete;
    DirectionalCovering& operator=(const DirectionalCovering&) = delete;
    const CoveringBoundaryStats& stats() const { return work; }
    CoveringBoundary query(const OriginalInterval& interval, CoveringDirection direction) const
    {
        ++work.queries;
        CoveringBoundary result;
        result.interval = interval;
        result.direction = direction;
        const auto prepared = prepareOriginalInterval(graph, version, interval.query);
        if (!prepared.valid || prepared.interval != interval) {
            return fail(std::move(result), CoveringBoundary::Obstruction::InvalidInterval,
                        prepared.valid ? "inconsistent original interval owner" : prepared.reason);
        }
        const auto key = std::make_pair(interval, direction);
        if (const auto found = answers.find(key); found != answers.end()) {
            ++work.cacheHits;
            // Returned records are values: their copied output is charged too.
            chargeOutput(found->second);
            return found->second;
        }
        result = derive(interval, direction);
        chargeOutput(result);
        answers.emplace(key, result);
        return result;
    }

private:
    struct Position {
        std::size_t lane, gap, order;
    };
    struct Lane {
        OriginalCut entry, exit;
        std::vector<std::size_t> items;
        std::size_t nextCutOrder = 0;
    };
    struct Node {
        const Region* region;
        CoveringWorkItem item;
        std::vector<std::size_t> children;
    };
    struct Summary {
        bool may = false, must = false, uncertain = false, finite = true;
    };
    const ControlGraph& graph;
    OriginalProgramVersion version;
    Match match;
    FiniteRepeat finiteRepeat;
    Region flat;
    std::vector<Lane> lanes;
    std::vector<Node> nodes;
    std::map<OriginalCut, Position> positions;
    // Share the complete selector key once, rather than copy a potentially
    // long predicate-dependency list into every node's memo key.
    mutable std::map<OriginalAccessSelector, std::map<std::size_t, Summary>> summaries;
    mutable std::map<std::pair<OriginalInterval, CoveringDirection>, CoveringBoundary> answers;
    mutable CoveringBoundaryStats work;

    void recordCut(OriginalCut cut, std::size_t lane)
    {
        positions.emplace(cut, Position{lane, lanes[lane].items.size(), lanes[lane].nextCutOrder++});
    }
    void append(const Region& region, std::size_t lane)
    {
        if (region.kind == Region::Sequence) {
            if (region.originalOwner != NoControlId) {
                recordCut(OriginalCut::scope(region.originalOwner, OriginalCut::Before), lane);
            }
            for (const auto& child : region.children) {
                append(child, lane);
            }
            if (region.originalOwner != NoControlId) {
                recordCut(OriginalCut::scope(region.originalOwner, OriginalCut::After), lane);
            }
            return;
        }
        CoveringWorkItem item;
        item.kind = region.kind;
        item.owner = region.originalOwner;
        if (region.kind == Region::Operation) {
            item.operation = region.operation;
            item.before = {region.operation, OriginalCut::Before};
            item.after = {region.operation, OriginalCut::After};
        } else {
            item.before = OriginalCut::scope(region.originalOwner, OriginalCut::Before);
            item.after = OriginalCut::scope(region.originalOwner, OriginalCut::After);
        }
        const auto id = nodes.size();
        nodes.push_back({&region, item, {}});
        recordCut(item.before, lane);
        lanes[lane].items.push_back(id);
        recordCut(item.after, lane);
        for (std::size_t child = 0; child < region.children.size(); ++child) {
            const auto childLane = lanes.size();
            lanes.emplace_back();
            lanes[childLane].entry = OriginalCut::childBoundary(region.originalOwner, child, OriginalCut::Before);
            lanes[childLane].exit = OriginalCut::childBoundary(region.originalOwner, child, OriginalCut::After);
            nodes[id].children.push_back(childLane);
            recordCut(lanes[childLane].entry, childLane);
            append(region.children[child], childLane);
            recordCut(lanes[childLane].exit, childLane);
        }
    }
    CoveringMatch matchAt(std::size_t operation, const OriginalAccessSelector& selector) const
    {
        ++work.effectQueries;
        return match(operation, selector);
    }
    static Summary sequence(Summary a, const Summary& b)
    {
        a.may |= b.may;
        a.must |= b.must;
        a.uncertain |= b.uncertain;
        a.finite &= b.finite;
        return a;
    }
    Summary summarize(std::size_t id, const OriginalAccessSelector& selector,
                      std::map<std::size_t, Summary>& memo) const
    {
        if (const auto found = memo.find(id); found != memo.end()) {
            return found->second;
        }
        ++work.summaryEvaluations;
        const auto& node = nodes[id];
        const auto& region = *node.region;
        Summary result;
        if (region.kind == Region::Operation) {
            const auto effect = matchAt(region.operation, selector);
            result.may = effect != CoveringMatch::NoHit;
            result.must = effect == CoveringMatch::Must;
            result.uncertain = effect == CoveringMatch::May || effect == CoveringMatch::Unknown;
        } else {
            bool allChildrenMust = !node.children.empty();
            for (auto lane : node.children) {
                Summary child;
                for (auto item : lanes[lane].items) {
                    child = sequence(child, summarize(item, selector, memo));
                }
                result = sequence(result, child);
                allChildrenMust &= child.must;
            }
            if (region.kind == Region::Choice) {
                result.must = allChildrenMust;
            } else {
                // No guess of the last participating iteration. Finiteness is
                // independent of invariant participation (the latter is D3).
                result.finite &= finiteRepeat(region);
                // While-before executes at least once, but absent a separately
                // supplied participation proof retain the conservative empty case.
                result.must = region.kind == Region::For && !region.zeroTripPossible && result.must;
            }
        }
        memo.emplace(id, result);
        return result;
    }
    static CoveringBoundary fail(CoveringBoundary result, CoveringBoundary::Obstruction why, std::string reason)
    {
        result.status = CoveringBoundary::Status::Unknown;
        result.obstruction = why;
        result.reason = std::move(reason);
        result.cut.reset();
        result.referenceCoverage = false;
        return result;
    }
    void chargeOutput(const CoveringBoundary& result) const
    {
        work.materializedReferences += result.mayAccesses.size() + result.enclosedWork.items.size() +
                                       result.trimmedWork.items.size() + result.loopOwners.size() +
                                       result.finiteLoopOwners.size() +
                                       result.interval.query.selector.predicateDependencies.size();
    }
    void collect(std::size_t id, const OriginalAccessSelector& selector, std::size_t excluded,
                 CoveringBoundary& result) const
    {
        ++work.intervalNodes;
        const auto& node = nodes[id];
        const auto& region = *node.region;
        if (region.kind == Region::Operation) {
            if (region.operation != excluded && matchAt(region.operation, selector) != CoveringMatch::NoHit) {
                result.mayAccesses.push_back(region.operation);
            }
            return;
        }
        result.cases.childEntry = true;
        if (region.kind == Region::For || region.kind == Region::While) {
            result.loopOwners.push_back(region.originalOwner);
            if (finiteRepeat(region)) {
                result.finiteLoopOwners.push_back(region.originalOwner);
            }
            result.cases.backedge = true;
            result.cases.bypass |= region.kind == Region::While || region.zeroTripPossible;
        }
        for (auto lane : node.children) {
            for (auto item : lanes[lane].items) {
                collect(item, selector, excluded, result);
            }
        }
    }
    // Retain conservative evidence for an interval outside the SESE derivation.
    // This never promotes a control walk to a finite-visit boundary certificate.
    CoveringBoundary unresolved(CoveringBoundary result, CoveringBoundary::Obstruction why,
                                const std::string& reason) const
    {
        const auto& selector = result.interval.query.selector;
        const auto walk = walkOriginalInterval(graph, result.interval, [&](std::size_t operation) {
            return matchAt(operation, selector) != CoveringMatch::NoHit;
        });
        work.intervalNodes += walk.visitedSites;
        result.mayAccesses = walk.operations;
        result.cases = walk.cases;
        result.conservativeEffects = true;
        // The may evidence is still useful, but a missing visit/SESE premise
        // is not a qualified empty interval. All remains separately queryable.
        return fail(std::move(result), why, reason);
    }
    CoveringBoundary derive(const OriginalInterval& interval, CoveringDirection direction) const
    {
        CoveringBoundary result;
        result.interval = interval;
        result.direction = direction;
        const auto& q = interval.query;
        result.cases.incoming = q.occurrence.incomingInterface != NoControlId;
        const auto startNode = graph.cuts.at(q.start), stopNode = graph.cuts.at(q.stop);
        if (!graph.entryReachable[startNode]) {
            return fail(std::move(result), CoveringBoundary::Obstruction::InvalidInterval,
                        "unreachable original starting cut");
        }
        if (q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge ||
            (q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::Unqualified &&
             (graph.repeated[startNode] || graph.repeated[stopNode]))) {
            return unresolved(std::move(result), CoveringBoundary::Obstruction::OccurrenceUnqualified,
                              "covering interval lacks a qualified single-visit interpretation");
        }
        const auto beginPosition = positions.find(q.start), endPosition = positions.find(q.stop);
        if (beginPosition == positions.end() || endPosition == positions.end() ||
            beginPosition->second.lane != endPosition->second.lane) {
            return unresolved(std::move(result), CoveringBoundary::Obstruction::NotSingleEntrySingleExit,
                              "cuts do not delimit one structural single-entry/single-exit interval");
        }
        // A gap may have several distinct original cuts (and scalar work
        // between them). Preserve their order; stop-access inclusion must not
        // turn after(a)..before(a) into a zero-length same-visit interval.
        if (beginPosition->second.order > endPosition->second.order) {
            return unresolved(std::move(result), CoveringBoundary::Obstruction::NotSingleEntrySingleExit,
                              "stopping cut precedes the starting cut in the original visit");
        }
        const auto laneId = beginPosition->second.lane;
        const auto& lane = lanes[laneId];
        const auto begin = beginPosition->second.gap;
        auto end = endPosition->second.gap;
        auto endCut = q.stop;
        std::size_t excluded = NoControlId;
        if (q.stop.kind == OriginalCut::Kind::Payload) {
            if (q.includeStoppingAccess && q.stop.side == OriginalCut::Before) {
                // Include the stop access, not the following operation or visit.
                ++end;
                endCut = {q.stop.operation, OriginalCut::After};
            } else if (!q.includeStoppingAccess) {
                // An After stop still encloses its payload's original work, but
                // excludes that payload from this query's matching family.
                excluded = q.stop.operation;
            }
        }
        if (begin > end || end > lane.items.size()) {
            return unresolved(std::move(result), CoveringBoundary::Obstruction::NotSingleEntrySingleExit,
                              "original interval crosses a visit boundary or has reversed cuts");
        }
        result.visitEntry = lane.entry;
        result.visitExit = lane.exit;
        result.cases.reachedStop = true;
        auto& memo = summaries.try_emplace(q.selector).first->second;
        std::vector<Summary> parts;
        Summary whole;
        for (auto i = begin; i < end; ++i) {
            ++work.intervalNodes;
            const auto id = lane.items[i];
            const auto& node = nodes[id];
            const auto part = node.item.kind == Region::Operation && node.item.operation == excluded ?
                                  Summary{} : summarize(id, q.selector, memo);
            parts.push_back(part);
            whole = sequence(whole, part);
            collect(id, q.selector, excluded, result);
        }
        result.conservativeEffects = whole.uncertain;
        result.conservativeSelection = q.selector.qualification != NoControlId ||
                                       !q.selector.predicateDependencies.empty() ||
                                       q.occurrence.qualification != NoControlId;
        if (!whole.may) {
            result.status = CoveringBoundary::Status::NoHit;
            return result;
        }
        // NoHit above needs no endpoint or finite repetition. A positive answer
        // does: even a no-hit child retained in this interval can obstruct its
        // finite-exit contract. Do not simply skip an unqualified while loop.
        if (!whole.finite) {
            return fail(std::move(result), CoveringBoundary::Obstruction::RepetitionUnqualified,
                        "original interval contains repetition without a finite-exit qualification");
        }
        if (!q.selector.engine) {
            return fail(std::move(result), CoveringBoundary::Obstruction::EngineRequired,
                        "directional covering requires one original source or target engine");
        }
        auto selected = direction == CoveringDirection::Source ? end - 1 : begin;
        if (direction == CoveringDirection::Source) {
            while (!parts[selected - begin].may) {
                --selected;
            }
        } else {
            while (!parts[selected - begin].may) {
                ++selected;
            }
        }
        const auto& item = nodes[lane.items[selected]].item;
        const auto boundary = direction == CoveringDirection::Source ? item.after : item.before;
        const auto at = graph.cuts.find(boundary);
        if (at == graph.cuts.end() || !graph.legalCuts[at->second]) {
            return fail(std::move(result), CoveringBoundary::Obstruction::CutUnavailable,
                        "prescribed covering cut is not executable in the unchanged original IR");
        }
        const bool source = direction == CoveringDirection::Source;
        result.enclosedWork.begin = source ? q.start : boundary;
        result.enclosedWork.end = source ? boundary : endCut;
        result.trimmedWork.begin = source ? boundary : q.start;
        result.trimmedWork.end = source ? endCut : boundary;
        for (auto i = begin; i < end; ++i) {
            const bool enclosed = source ? i <= selected : i >= selected;
            auto& scope = enclosed ? result.enclosedWork : result.trimmedWork;
            scope.items.push_back(nodes[lane.items[i]].item);
        }
        result.status = CoveringBoundary::Status::Covering;
        result.cut = boundary;
        result.referenceCoverage = true;
        result.guardIsTrue = true;
        result.exactlyOneVisitPerInterval = true;
        result.mayExecuteWithoutAccess = !whole.must || result.conservativeSelection;
        result.mayRepeatInInvocation = graph.repeated[at->second];
        return result;
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
