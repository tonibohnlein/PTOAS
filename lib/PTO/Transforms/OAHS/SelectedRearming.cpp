// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Deadline-driven ordinary rearming. These records never modify causal state.
// Acyclic continuations may merge shared receipt occurrences. Correspondence
// and actual source-time consumption checks qualify the later F7 helper.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
bool Constructor::deferCommonRearming(Id key, Id acquisition, const SelectedDecision& decision)
{
    if (!options.deferredAcyclicAcknowledgments ||
        std::any_of(control.components.begin(), control.components.end(),
                    [](const auto& component) { return component.cyclic; }) ||
        !control.mergesToExit[current]) {
        return false;
    }
    const auto& correspondence = control.correspondence(current, control.graph.exit);
    if (!correspondence.qualified ||
        std::any_of(correspondence.pairs.begin(), correspondence.pairs.end(), [&](const auto& pair) {
            return pair.second != control.graph.exit || !control.mergesToExit[pair.first];
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
