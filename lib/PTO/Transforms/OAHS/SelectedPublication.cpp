// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Construction-time publication motion inside one word. The structural
// certificate removes incoming WAIT prerequisites from a new SET without
// crossing another source-prefix observer. Coverage and physical-key legality
// are separate checks. This is not a certificate for motion across payloads,
// coalescing recurring roles, or changes to an earlier word's exported state.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
namespace {
Pipe commandPipe(const Command& command)
{
    return command.kind == Command::Acquire ? command.observer : command.source;
}

bool crossesPublication(const Command& publication, const Command& crossed)
{
    if (crossed.kind == Command::BarrierAll) {
        return false;
    }
    if (crossed.source == publication.source && crossed.observer == publication.observer &&
        crossed.key == publication.key &&
        (crossed.kind == Command::Publish || crossed.kind == Command::Acquire)) {
        return false;
    }
    return commandPipe(crossed) != publication.source || crossed.kind == Command::Acquire;
}

using Missing = std::tuple<Cut, Id, Id, unsigned, unsigned, unsigned>;
std::set<Missing> missingRequirements(const AnalysisResult& analysis)
{
    std::set<Missing> missing;
    for (const auto& requirement : analysis.residuals) {
        missing.emplace(requirement.consumerCut, requirement.demand.producer, requirement.demand.consumer,
                        requirement.demand.cell, unsigned(requirement.kind), unsigned(requirement.demand.property));
    }
    return missing;
}

bool protocolValid(const AnalysisResult& analysis)
{
    return analysis.complete && analysis.diagnostics.empty() && analysis.protocol.empty() &&
           analysis.phaseResources.empty();
}
} // namespace

bool Ledger::movePublicationBefore(Id publication, Id before)
{
    const bool available = publication < endpoints.size() && before < endpoints.size() &&
                           active(publication) && active(before);
    if (!available) {
        return false;
    }
    const auto& endpoint = endpoints[publication];
    if (endpoint.command.kind != Command::Publish || endpoint.cut != endpoints[before].cut) {
        return false;
    }
    auto& ids = words[endpoint.cut];
    const auto first = std::find(ids.begin(), ids.end(), before);
    const auto last = std::find(ids.begin(), ids.end(), publication);
    if (first >= last || !std::all_of(first, last, [&](Id id) {
            return crossesPublication(endpoint.command, endpoints[id].command);
        })) {
        return false;
    }
    const auto protectedEnd = protectedPrefixes.find(endpoint.cut);
    const bool crossesProtected = protectedEnd != protectedPrefixes.end() &&
        std::find(ids.begin(), ids.end(), protectedEnd->second) >= first;
    if (crossesProtected) {
        return false;
    }
    // Moving SET across a WAIT on its source removes WAIT-finish -> SET-finish.
    // SET does not gate subsequent launches. Other engines' words are unchanged;
    // no crossed source SET/fence can gain a new prefix-finish dependency.
    // Matching-key endpoints cannot cross, so generation order is retained.
    const auto offset = Id(first - ids.begin());
    std::vector<Id> prefix;
    for (auto at = ids.begin(); at != first; ++at) {
        const bool onSource = commandPipe(endpoints[*at].command) == endpoint.command.source;
        if (onSource) {
            prefix.push_back(*at);
        }
    }
    ids.erase(last);
    ids.insert(ids.begin() + offset, publication);
    publicationPrefixes[publication] = std::move(prefix);
    protectPublicationPrefix(publication);
    changed.push_back(endpoint.cut);
    ++revision;
    return true;
}

bool Ledger::publicationPrefixesValid() const
{
    for (const auto& [publication, prefix] : publicationPrefixes) {
        if (!active(publication)) {
            continue;
        }
        const auto& endpoint = endpoints[publication];
        for (auto id : words[endpoint.cut]) {
            if (id == publication) {
                break;
            }
            const bool onSource = commandPipe(endpoints[id].command) == endpoint.command.source;
            const bool recorded = std::find(prefix.begin(), prefix.end(), id) != prefix.end();
            if (onSource && !recorded) {
                return false;
            }
        }
    }
    return true;
}

bool Constructor::publicationGapCovers(
    Id publication, Id before, const std::vector<FrontierRequirement>& required) const
{
    const auto& endpoint = ledger.endpoint(publication);
    const auto& command = endpoint.command;
    const auto key = std::find_if(frontier.keys().begin(), frontier.keys().end(), [&](const auto& identity) {
        return identity.source == command.source && identity.observer == command.observer &&
               identity.key == command.key;
    });
    if (key == frontier.keys().end()) {
        return false;
    }
    bool reached = false;
    for (auto cut : control.wordOccurrences[endpoint.cut]) {
        if (!control.reachable[cut]) {
            continue;
        }
        auto state = cache.cuts[cut].incoming;
        Id offset = 0;
        for (auto id : ledger.word(cut)) {
            if (id == before) {
                break;
            }
            const auto step = frontier.command(state.causal, ledger.endpoint(id).command, {cut, offset++});
            if (!step.applied) {
                return false;
            }
            state.causal = step.state;
        }
        if (!canPublish(state, Id(key - frontier.keys().begin()))) {
            return false;
        }
        for (const auto& requirement : required) {
            const auto* history = state.causal.facts()->history.find(accessClass(requirement));
            if (!history || !frontierContains(*history, PipeCount + unsigned(command.source))) {
                return false;
            }
        }
        reached = true;
    }
    return reached;
}

bool Constructor::preservePublicationPrefixes(SelectedDecision& decision)
{
    if (!options.publicationPrefixes || decision.required.empty()) {
        return true;
    }
    for (auto publication : decision.endpoints) {
        const auto& endpoint = ledger.endpoint(publication);
        if (endpoint.purpose != EndpointPurpose::Completion || endpoint.command.kind != Command::Publish) {
            continue;
        }
        const auto& ids = ledger.word(endpoint.cut);
        auto first = std::find(ids.begin(), ids.end(), publication);
        Id before = NoAnalysisId;
        while (first != ids.begin()) {
            const auto previous = std::prev(first);
            const bool canCross = crossesPublication(endpoint.command, ledger.endpoint(*previous).command);
            if (!canCross) {
                break;
            }
            --first;
            const auto& crossed = ledger.endpoint(*first).command;
            if (crossed.kind == Command::Acquire && crossed.observer == endpoint.command.source) {
                before = *first;
            }
        }
        // One structurally selected gap, not a search over candidate programs.
        if (before == NoAnalysisId || !publicationGapCovers(publication, before, decision.required)) {
            continue;
        }
        Ledger proposed = ledger;
        if (!proposed.movePublicationBefore(publication, before)) {
            continue;
        }
        ++result.work.prefixChecks;
        const auto original = analyze(program, ledger.commands(), {false});
        const auto candidate = analyze(program, proposed.commands(), {false});
        result.work.prefixAnalysisSites += original.stats.siteEvaluations + candidate.stats.siteEvaluations;
        const auto oldMissing = missingRequirements(original);
        const auto newMissing = missingRequirements(candidate);
        const bool preservesCoverage =
            std::includes(oldMissing.begin(), oldMissing.end(), newMissing.begin(), newMissing.end());
        const bool preservesRetirement =
            std::all_of(candidate.retirement.begin(), candidate.retirement.end(), [&](const auto& requirement) {
                return std::any_of(original.retirement.begin(), original.retirement.end(), [&](const auto& old) {
                    return requirement.operation == old.operation && requirement.observer == old.observer;
                });
            });
        const bool validProtocol = protocolValid(original) && protocolValid(candidate);
        if (!validProtocol || !preservesCoverage || !preservesRetirement) {
            continue;
        }
        const bool moved = ledger.movePublicationBefore(publication, before);
        if (!moved || !update()) {
            return false;
        }
        ++result.work.prefixPublications;
    }
    return true;
}
} // namespace mlir::pto::oahs::selected
