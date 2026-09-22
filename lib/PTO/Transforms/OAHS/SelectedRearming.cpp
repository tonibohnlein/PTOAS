// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Deadline-driven ordinary rearming. These records never modify causal state.
// Acyclic continuations may merge shared receipts and split before key reuse.
// Immutable balanced boundaries and actual source-time consumption checks
// qualify the later F7 helper without allocating it speculatively.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
bool Constructor::deferCommonRearming(Id key, Id acquisition, const SelectedDecision& decision)
{
    if (!options.deferredAcyclicAcknowledgments || !control.acyclic) {
        return false;
    }
    const auto& correspondence = control.correspondence(current, control.graph.exit);
    const auto boundary = control.rearmingBoundary[current];
    if (boundary == NoAnalysisId || !legalCommandCut(program, boundary)) {
        return false;
    }
    if (!correspondence.qualified ||
        std::any_of(correspondence.pairs.begin(), correspondence.pairs.end(), [&](const auto& pair) {
            const auto next = control.rearmingBoundary[pair.first];
            return pair.second != control.graph.exit || next == NoAnalysisId ||
                control.canonicalCut[next] != control.canonicalCut[boundary];
        })) {
        return false;
    }
    const auto& identity = frontier.keys()[key];
    // An already selected later publication cannot borrow a future return.
    // Leave its existing closed policy in place; defer only new ordinary uses.
    for (const auto& endpoint : ledger.records()) {
        const auto& command = endpoint.command;
        if (ledger.active(endpoint.id) && endpoint.cut != control.canonicalCut[current] &&
            command.kind == Command::Publish && command.source == identity.source &&
            command.observer == identity.observer && command.key == identity.key) {
            return false;
        }
    }
    SelectedRearmingObligation obligation;
    obligation.key = identity;
    obligation.acquisition = acquisition;
    obligation.fallback = control.canonicalCut[boundary];
    for (const auto& demand : decision.lifecycles) {
        obligation.returnDeadlines.insert(obligation.returnDeadlines.end(),
            demand.returnDeadlines.begin(), demand.returnDeadlines.end());
    }
    std::sort(obligation.returnDeadlines.begin(), obligation.returnDeadlines.end());
    obligation.returnDeadlines.erase(
        std::unique(obligation.returnDeadlines.begin(), obligation.returnDeadlines.end()),
        obligation.returnDeadlines.end());
    deferredByAcquisition.emplace(acquisition, result.rearming.size());
    result.rearming.push_back(std::move(obligation));
    if (control.lookahead.mayIssueAfter(current)) {
        ++result.work.deferredAcknowledgments;
    }
    return true;
}

Cut Constructor::deferredReturnCut(Id acquisition, Cut publication, Id key) const
{
    const auto found = deferredByAcquisition.find(acquisition);
    if (found == deferredByAcquisition.end()) {
        return NoAnalysisId;
    }
    // The consumer at the traversal cursor is not evidence at an earlier
    // publication. Recheck the actual generation at every source occurrence.
    for (auto at : control.wordOccurrences[control.canonicalCut[publication]]) {
        const auto& state = cache.cuts[at].before;
        if (control.reachable[at] && (!state.causal.reachable() ||
            state.causal.facts()->events[key].occupancy != 1 || state.consumptions[key].size() != 1 ||
            state.consumptions[key].front() != acquisition)) {
            return NoAnalysisId;
        }
    }
    const auto source = ledger.endpoint(acquisition).cut;
    if (control.correspondence(source, publication).qualified) {
        return publication;
    }
    const auto boundary = result.rearming[found->second].fallback;
    const auto lastBoundary = control.wordSpan[boundary].second;
    const auto firstPublication = control.wordSpan[control.canonicalCut[publication]].first;
    // Every entry-to-exit path contains the receipt and its first split. A
    // publication after that split can be conditional; the return cannot be.
    // Topological components distinguish earlier saved source positions.
    return lastBoundary != NoAnalysisId && firstPublication != NoAnalysisId &&
        lastBoundary <= firstPublication ? boundary : NoAnalysisId;
}

void Constructor::observeDeferredRearming(const SelectedDecision& decision)
{
    if (deferredByAcquisition.empty()) {
        return;
    }
    // Reuse the exact snapshots already produced by mandatory command replay.
    // No new solve, graph walk, speculative return, or helper-removal trial.
    for (auto id : decision.endpoints) {
        const auto& endpoint = ledger.endpoint(id);
        if (!ledger.active(id) || endpoint.command.kind != Command::Publish) {
            continue;
        }
        const auto snapshot = cache.afterEndpoint.find(id);
        if (snapshot == cache.afterEndpoint.end()) {
            continue;
        }
        const auto& command = endpoint.command;
        const auto& keys = frontier.keys();
        const auto key = std::find_if(keys.begin(), keys.end(), [&](const auto& identity) {
            return identity.source == command.source && identity.observer == command.observer &&
                identity.key == command.key;
        });
        if (key == keys.end()) {
            continue;
        }
        for (auto acquisition : snapshot->second.consumptions[std::size_t(key - keys.begin())]) {
            const auto found = deferredByAcquisition.find(acquisition);
            if (found == deferredByAcquisition.end()) {
                continue;
            }
            auto& obligation = result.rearming[found->second];
            if (std::find(obligation.reusePublications.begin(), obligation.reusePublications.end(), id) ==
                obligation.reusePublications.end()) {
                obligation.reusePublications.push_back(id);
            }
        }
    }
}
} // namespace mlir::pto::oahs::selected
