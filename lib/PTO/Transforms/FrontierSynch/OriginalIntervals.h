// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_ORIGINALINTERVALS_H
#define PTO_FRONTIERSYNCH_ORIGINALINTERVALS_H
#include "Control.h"

namespace mlir::pto::frontiersynch::detail {
inline std::size_t enclosingOwner(const ControlGraph& graph, std::size_t a, std::size_t b)
{
    std::set<std::size_t> ancestors;
    for (;;) {
        ancestors.insert(a);
        if (a == NoControlId) {
            break;
        }
        a = graph.scopeParents.at(a);
    }
    while (!ancestors.count(b)) {
        b = graph.scopeParents.at(b);
    }
    return b;
}
inline bool insideOwner(const ControlGraph& graph, std::size_t scope, std::size_t owner)
{
    while (scope != owner && scope != NoControlId) {
        scope = graph.scopeParents.at(scope);
    }
    return scope == owner;
}
inline OriginalIntervalResult prepareOriginalInterval(
    const ControlGraph& graph, OriginalProgramVersion version, OriginalIntervalRequest query)
{
    OriginalIntervalResult result;
    result.interval.query = std::move(query);
    auto& q = result.interval.query;
    if (!q.version) {
        q.version = version;
    }
    if (!graph.valid || !version || q.version != version || (graph.version && graph.version != version)) {
        result.reason = graph.valid ? "original-program version mismatch" : graph.reason;
        return result;
    }
    const auto start = graph.cuts.find(q.start), stop = graph.cuts.find(q.stop);
    if (start == graph.cuts.end() || stop == graph.cuts.end()) {
        result.reason = "original interval has an unrepresented cut";
        return result;
    }
    if (!graph.legalCuts[start->second] || !graph.legalCuts[stop->second]) {
        result.reason = "original interval cut is not executable in the unchanged IR";
        return result;
    }
    if (q.includeStoppingAccess && q.stop.kind != OriginalCut::Kind::Payload) {
        result.reason = "structured stopping cut has no access to include";
        return result;
    }
    if (!q.selector.read && !q.selector.write) {
        result.reason = "original interval has no access selector";
        return result;
    }
    auto owner = enclosingOwner(graph, graph.nodeOwners[start->second], graph.nodeOwners[stop->second]);
    for (auto role : {q.occurrence.source, q.occurrence.target}) {
        if (role == NoControlId) {
            continue;
        }
        if (role >= graph.operationOwners.size()) {
            result.reason = "original occurrence role is not a payload identity";
            return result;
        }
        owner = enclosingOwner(graph, owner, graph.operationOwners[role]);
    }
    if (q.continuationOwner) {
        if (!graph.scopeParents.count(*q.continuationOwner)) {
            result.reason = "unrepresented declared continuation owner";
            return result;
        }
        owner = enclosingOwner(graph, owner, *q.continuationOwner);
    }
    if (q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge) {
        if (!graph.loopEntries.count(q.occurrence.backedgeOwner)) {
            result.reason = "occurrence interpretation has no represented original backedge";
            return result;
        }
        owner = enclosingOwner(graph, owner, q.occurrence.backedgeOwner);
    } else if (
        q.occurrence.stopVisit != OriginalOccurrenceContext::StopVisit::FirstReach &&
        q.occurrence.stopVisit != OriginalOccurrenceContext::StopVisit::Unqualified) {
        result.reason = "unknown original stop-visit interpretation";
        return result;
    }
    result.interval.owner = owner;
    result.valid = true;
    return result;
}

enum class IntervalSelection { All, First, Last };
struct OriginalIntervalWalk {
    bool complete = false, unresolvedOccurrence = false, noHitPath = false;
    std::vector<std::size_t> operations;
    OriginalContinuationCases cases;
    std::size_t visitedSites = 0;
    std::string reason;
};
// May-access query on the finite original-control language. The callback tests
// original effects only. Reachability is not predicate feasibility, exact use
// pairing, asynchronous completion, or a proof that a counted loop terminates.
// With AfterBackedge the one extra bit records whether its NAMED edge was seen;
// unrelated selectors or branch valuations are never multiplied into this state.
template <typename Matches>
OriginalIntervalWalk walkOriginalInterval(
    const ControlGraph& graph, const OriginalInterval& interval, Matches matches,
    IntervalSelection selection = IntervalSelection::All)
{
    OriginalIntervalWalk result;
    const auto& q = interval.query;
    const auto prepared = prepareOriginalInterval(graph, q.version, q);
    if (!prepared.valid || prepared.interval != interval) {
        result.reason = prepared.valid ? "inconsistent original interval owner" : prepared.reason;
        return result;
    }
    const auto& reachable = graph.entryReachable;
    if (reachable.size() != graph.sites.size()) {
        result.reason = "missing original entry-reachability index";
        return result;
    }
    const auto start = graph.cuts.at(q.start), stop = graph.cuts.at(q.stop);
    if (!reachable[start]) {
        result.reason = "unreachable original starting cut";
        return result;
    }
    const auto ownerExit = graph.cuts.at(OriginalCut::scope(interval.owner, OriginalCut::After));
    const bool delayed = q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge;
    const auto wantedBackedge = delayed ? graph.loopEntries.at(q.occurrence.backedgeOwner) : NoControlId;
    struct Visit {
        std::size_t site;
        bool armed;
    };
    std::vector<Visit> pending{{start, !delayed}};
    std::vector<bool> seen(graph.sites.size() * (delayed ? 2 : 1));
    std::set<std::size_t> matching;
    const bool frontier = selection != IntervalSelection::All;
    std::vector<std::vector<std::size_t>> links(frontier ? seen.size() : 0);
    std::vector<std::size_t> matchAt(frontier ? seen.size() : 0, NoControlId), terminals;
    const auto initial = start;
    result.cases.incoming = q.occurrence.incomingInterface != NoControlId;
    while (!pending.empty()) {
        const auto visit = pending.back();
        pending.pop_back();
        const auto site = visit.site;
        const auto state = site + (delayed && visit.armed ? graph.sites.size() : 0);
        if (seen[state]) {
            continue;
        }
        seen[state] = true;
        ++result.visitedSites;
        const bool atStop = visit.armed && site == stop;
        const auto operation = graph.operations[site];
        const bool stoppingPayload = visit.armed && q.stop.kind == OriginalCut::Kind::Payload &&
                                     operation != NoControlId && operation == q.stop.operation;
        // A Before stop includes its local access only if requested. For an After
        // stop the access is encountered at its payload node, before the exit port.
        const bool include = !stoppingPayload || q.includeStoppingAccess;
        if (operation != NoControlId && include && matches(operation)) {
            matching.insert(operation);
            if (frontier) {
                matchAt[state] = operation;
            }
        }
        if (atStop) {
            result.cases.reachedStop = true;
            terminals.push_back(state);
            continue;
        }
        if (site == ownerExit) {
            result.cases.reachedOwnerExit = true;
            result.cases.bypass = true;
            terminals.push_back(state);
            continue;
        }
        if (!insideOwner(graph, graph.nodeOwners[site], interval.owner)) {
            result.reason = "original continuation escaped its selected owner";
            return result;
        }
        const auto& edges = graph.sites[site];
        for (std::size_t i = 0; i < edges.successors.size(); ++i) {
            const auto backedge = edges.backedgeOwners[i];
            result.cases.backedge |= backedge != NoControlId;
            result.cases.childEntry |= edges.childEntries[i];
            result.cases.bypass |= edges.bypasses[i];
            const bool armed = visit.armed || (delayed && backedge == wantedBackedge);
            pending.push_back({edges.successors[i], armed});
            if (frontier) {
                const auto next = edges.successors[i] + (delayed && armed ? graph.sites.size() : 0);
                if (selection == IntervalSelection::First) {
                    links[state].push_back(next);
                } else {
                    links[next].push_back(state);
                }
            }
        }
    }
    if (frontier) {
        matching.clear();
        std::fill(seen.begin(), seen.end(), false);
        std::vector<std::size_t> todo =
            selection == IntervalSelection::First ? std::vector<std::size_t>{initial} : terminals;
        while (!todo.empty()) {
            const auto state = todo.back();
            todo.pop_back();
            if (seen[state]) {
                continue;
            }
            seen[state] = true;
            if (matchAt[state] != NoControlId) {
                matching.insert(matchAt[state]);
                continue;
            }
            if (selection == IntervalSelection::Last && state == initial) {
                // The initial static state can also be revisited through a cycle.
                // Retain its reached predecessors for last-use queries; only the
                // zero-length incoming path ends here.
                result.noHitPath = true;
            }
            if (selection == IntervalSelection::First && links[state].empty()) {
                result.noHitPath = true;
            }
            todo.insert(todo.end(), links[state].begin(), links[state].end());
        }
    }
    result.operations.assign(matching.begin(), matching.end());
    result.unresolvedOccurrence = q.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::Unqualified &&
                                  (result.cases.backedge || graph.repeated[start] || graph.repeated[stop]);
    result.complete = true;
    return result;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
