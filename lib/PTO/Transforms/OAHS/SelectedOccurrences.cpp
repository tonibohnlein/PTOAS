// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Immutable occurrence correspondence over original control. Canonical words
// identify shared commands, not interchangeable execution histories. Clients
// must check coverage, placement and physical-key evidence at every pair.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
const OccurrenceCorrespondence& Control::correspondence(Cut publication, Cut acquisition) const
{
    static const OccurrenceCorrespondence unknown;
    if (!complete || publication >= canonicalCut.size() || acquisition >= canonicalCut.size()) {
        return unknown;
    }
    const auto key = std::make_pair(canonicalCut[publication], canonicalCut[acquisition]);
    const auto found = correspondences.find(key);
    if (found != correspondences.end()) {
        return found->second;
    }
    return correspondences.emplace(key, pairOccurrences(key.first, key.second)).first->second;
}

OccurrenceCorrespondence Control::pairOccurrences(Cut publication, Cut acquisition) const
{
    OccurrenceCorrespondence result;
    if (publication == acquisition) {
        for (auto cut : wordOccurrences[publication]) {
            if (reachable[cut]) {
                result.pairs.emplace_back(cut, cut);
            }
        }
        result.qualified = !result.pairs.empty();
        return result;
    }
    std::vector<std::pair<Cut, Cut>> todo{{graph.entry, NoAnalysisId}};
    std::set<std::pair<Cut, Cut>> seen, pairs;
    while (!todo.empty()) {
        auto [at, source] = todo.back();
        todo.pop_back();
        if (!seen.insert({at, source}).second) {
            continue;
        }
        if (canonicalCut[at] == publication) {
            if (source != NoAnalysisId) {
                return {};
            }
            source = at;
        }
        if (canonicalCut[at] == acquisition) {
            if (source == NoAnalysisId) {
                return {};
            }
            pairs.emplace(source, at);
            source = NoAnalysisId;
        }
        const bool unconsumedExit = graph.sites[at].successors.empty() && source != NoAnalysisId;
        if (unconsumedExit) {
            return {};
        }
        for (auto next : graph.sites[at].successors) {
            todo.emplace_back(next, source);
        }
    }
    result.pairs.assign(pairs.begin(), pairs.end());
    result.qualified = !result.pairs.empty();
    return result;
}

bool Control::balancedWords(Cut publication, Cut acquisition) const
{
    return correspondence(publication, acquisition).qualified;
}

bool Control::appendChoiceOccurrence(const Program& program, Cut entry, ChoiceFrontier& facts)
{
    const bool choice = graph.legalCuts[entry] && graph.sites[entry].successors.size() == 2;
    if (!choice) {
        return false;
    }
    for (auto at : graph.sites[entry].successors) {
        bool found = false;
        std::set<Cut> visited;
        while (visited.insert(at).second) {
            ++choicePreparationSites;
            const auto& context = graph.contexts[graph.cutContexts[at]];
            // Refined copies retain the scope's original owner identity.
            const bool originalArm = context.ownerSite < canonicalCut.size() &&
                canonicalCut[context.ownerSite] == facts.entry &&
                (context.kind == AnalysisContext::ThenArm || context.kind == AnalysisContext::ElseArm);
            if (!originalArm) {
                break;
            }
            facts.crossedWords.push_back(at);
            const auto operation = graph.operations[at];
            if (operation != NoAnalysisId) {
                found = program.operations[operation].pipe == facts.observer;
                break;
            }
            const bool straightEdge = graph.sites[at].successors.size() == 1 &&
                graph.sites[at].backedgeOwners.front() == NoAnalysisId;
            if (!straightEdge) {
                break;
            }
            at = graph.sites[at].successors.front();
        }
        if (!found) {
            return false;
        }
        facts.consumers.push_back(at);
    }
    return true;
}

void Control::prepareChoiceFrontiers(const Program& program)
{
    choicesAtConsumer.resize(graph.sites.size());
    std::set<Cut> choices;
    for (const auto& context : graph.contexts) {
        const bool arm = context.kind == AnalysisContext::ThenArm || context.kind == AnalysisContext::ElseArm;
        if (arm && context.ownerSite < canonicalCut.size()) {
            choices.insert(canonicalCut[context.ownerSite]);
        }
    }
    for (auto entry : choices) {
        for (unsigned pipe = 0; pipe < PipeCount; ++pipe) {
            ChoiceFrontier facts{entry, Pipe(pipe), {}, {}, {}};
            bool qualified = true;
            for (auto occurrence : wordOccurrences[entry]) {
                if (reachable[occurrence] && !appendChoiceOccurrence(program, occurrence, facts)) {
                    qualified = false;
                    break;
                }
            }
            if (!qualified || facts.consumers.empty()) {
                continue;
            }
            for (auto consumer : facts.consumers) {
                choicesAtConsumer[consumer].push_back(choiceFrontiers.size());
            }
            choiceFrontiers.push_back(std::move(facts));
        }
    }
}
} // namespace mlir::pto::oahs::selected
