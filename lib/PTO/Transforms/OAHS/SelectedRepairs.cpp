// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
bool Constructor::fixedBoundaryPacket(const SourceGapQualification& facts, Group& group)
{
    // One designated source-local acknowledgment, followed by the ordinary
    // matched transfer. No candidate-ledger solve or provisional reservation.
    const bool actualBoundary = facts.deadline == current && facts.gap.left == NoAnalysisId;
    if (!actualBoundary) { return false; }
    const auto& keys = frontier.keys();
    for (Id forward = 0; forward < keys.size(); ++forward) {
        if (!sourceKeyNeighbors(facts, forward)) { continue; }
        // A unique old WAIT has its own earlier prescribed helper source.
        // Do not move that publication here merely to fit a source-local proof.
        bool joined = true;
        for (const auto& pair : control.correspondence(facts.gap.cut, current).pairs) {
            if (!control.reachable[pair.first]) { continue; }
            joined &= cache.cuts[pair.first].incoming.consumptions[forward].size() > 1;
        }
        if (!joined) { continue; }
        const auto& a = keys[forward];
        bool empty = true, missingConsumption = false;
        for (const auto& prefix : facts.prefixes) {
            empty &= prefix.facts()->events[forward].occupancy == 1;
            State before; before.causal = prefix;
            missingConsumption |= !canPublish(before, forward);
        }
        if (!empty || !missingConsumption) { continue; }
        for (Id reverse = 0; reverse < keys.size(); ++reverse) {
            const auto& b = keys[reverse];
            const bool eligible = b.source == a.observer && b.observer == a.source &&
                helperFreeKey(reverse) && ledger.eventUses(b).empty();
            if (!eligible) { continue; }
            const Command publish{Command::Publish, b.source, b.observer, b.key};
            const Command acquire{Command::Acquire, b.source, b.observer, b.key};
            const Command send{Command::Publish, a.source, a.observer, a.key};
            bool executable = true;
            for (const auto& prefix : facts.prefixes) {
                auto state = prefix;
                Id offset = 0;
                for (const auto& command : {publish, acquire, send}) {
                    ++result.work.repairSourceCommands;
                    const auto step = frontier.command(state, command, {facts.gap.cut, offset++});
                    if (!step.applied) { executable = false; break; }
                    state = step.state;
                }
                if (!executable) { break; }
            }
            if (!executable) { continue; }
            const auto request = result.decisions.size();
            const OrderedPacket endpoints{
                {facts.gap.cut, publish, EndpointPurpose::ConsumptionAcknowledgment,
                    request, NoAnalysisId, facts.gap},
                {facts.gap.cut, acquire, EndpointPurpose::ConsumptionAcknowledgment,
                    request, NoAnalysisId, facts.gap},
                {facts.gap.cut, send, EndpointPurpose::Completion, request, NoAnalysisId, facts.gap},
                {current, {Command::Acquire, a.source, a.observer, a.key}, EndpointPurpose::Completion, request}};
            auto packet = prepareOwnedPacket(endpoints);
            const bool supported = packet && !packet->restoredEndpoints && preservePublications(*packet);
            if (!supported) { continue; }
            packet->qualified = true;
            group.packet = std::move(*packet);
            group.forwardKey = forward;
            group.repairKey = reverse;
            return true;
        }
    }
    return false;
}
} // namespace mlir::pto::oahs::selected

namespace mlir::pto::oahs::selected {
std::optional<ConsumptionFrontier> Constructor::singletonConsumptionFrontier(
    const SourceGapQualification& facts, Id forward) const
{
    const bool valid = facts.proved() && facts.version == ledger.version() &&
        facts.gap.left == NoAnalysisId && forward < frontier.keys().size() &&
        facts.prefixes.size() == 1;
    if (!valid) { return {}; }
    const auto source = facts.gap.cut;
    const auto& toDeadline = control.correspondence(source, facts.deadline);
    const bool matchedDeadline = toDeadline.proved() && toDeadline.pairs.size() == 1 &&
        toDeadline.pairs.front().first == source;
    if (!matchedDeadline) { return {}; }
    const auto unique = [&](Cut cut) {
        const auto component = control.component[cut];
        if (component == NoAnalysisId || control.components[component].cyclic ||
            control.canonicalCut[cut] != cut) { return false; }
        return std::count_if(control.wordOccurrences[cut].begin(),
            control.wordOccurrences[cut].end(), [&](Cut site) {
                return control.reachable[site];
            }) == 1;
    };
    const bool uniqueWords = unique(source) && unique(facts.deadline);
    if (!uniqueWords) { return {}; }
    const auto& consumers = cache.cuts[source].incoming.consumptions[forward];
    const bool singleConsumer = consumers.size() == 1;
    if (!singleConsumer) { return {}; }
    const auto wait = consumers.front();
    const bool activeWait = wait < ledger.records().size() && ledger.active(wait);
    if (!activeWait) { return {}; }
    const auto& old = ledger.endpoint(wait);
    const auto& identity = frontier.keys()[forward];
    const bool matching = old.command.kind == Command::Acquire &&
        old.command.source == identity.source && old.command.observer == identity.observer &&
        old.command.key == identity.key && unique(old.cut) &&
        control.straight(old.cut, source) && old.cut != source;
    if (!matching) { return {}; }
    const auto& fromWait = control.correspondence(old.cut, source);
    const bool matchedSource = fromWait.proved() && fromWait.pairs.size() == 1 &&
        fromWait.pairs.front() == std::make_pair(old.cut, source) &&
        clearInterval(forward, old.cut, source);
    if (!matchedSource) { return {}; }
    const auto sourceGap = ledger.gapAfter(wait);
    const auto after = cache.afterEndpoint.find(wait);
    const bool actualSource = sourceGap && after != cache.afterEndpoint.end() &&
        after->second.consumptions[forward] == std::vector<Id>{wait} &&
        after->second.causal.facts()->events[forward].occupancy == 1 &&
        facts.prefixes.front().facts()->events[forward].occupancy == 1;
    if (!actualSource) { return {}; }
    // clearInterval excludes its starting word; preserve the exact suffix
    // after the old WAIT before inserting the reverse publication.
    const auto& word = ledger.word(old.cut);
    const auto at = std::find(word.begin(), word.end(), wait);
    const bool foundWait = at != word.end();
    if (!foundWait) { return {}; }
    for (auto next = at + 1; next != word.end(); ++next) {
        const auto& command = ledger.endpoint(*next).command;
        const bool sameForwardUse =
            (command.kind == Command::Publish || command.kind == Command::Acquire) &&
            keyIndex(frontier, command) == forward;
        if (sameForwardUse) { return {}; }
    }
    return ConsumptionFrontier{wait, *sourceGap, facts.gap, ledger.version()};
}

bool Constructor::separatedBoundaryPacket(const SourceGapQualification& facts, Group& group)
{
    if (facts.deadline != current || facts.gap.left != NoAnalysisId) { return false; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    for (Id forward = 0; forward < frontier.keys().size(); ++forward) {
        const auto& a = frontier.keys()[forward];
        if (a.source != group.source || a.observer != observer ||
            !sourceKeyNeighbors(facts, forward)) { continue; }
        const bool missingRearming = std::all_of(
            facts.prefixes.begin(), facts.prefixes.end(), [&](const FrontierState& prefix) {
                State atSource;
                atSource.causal = prefix;
                return !canPublish(atSource, forward);
            });
        if (!missingRearming) { continue; }
        const auto consumed = singletonConsumptionFrontier(facts, forward);
        if (!consumed) { continue; }
        // The reverse key may have completed an earlier exchange. Only a
        // historical candidate needs the original continuation mask; share
        // its word queries across all reverse-key alternatives.
        std::vector<bool> afterConsumption;
        std::map<Cut, bool> wordIntersects;
        const auto after = cache.afterEndpoint.find(consumed->wait);
        for (Id reverse = 0; reverse < frontier.keys().size(); ++reverse) {
            const auto& b = frontier.keys()[reverse];
            const bool eligibleReverse = b.source == observer && b.observer == group.source &&
                helperFreeKey(reverse);
            const bool available = after != cache.afterEndpoint.end() &&
                canPublish(after->second, reverse);
            if (!eligibleReverse || !available) { continue; }
            if (!ledger.eventUses(b).empty()) {
                if (afterConsumption.empty()) {
                    afterConsumption.resize(control.graph.sites.size());
                    result.work.normalKeySites += afterConsumption.size();
                    std::vector<Cut> pending{ledger.endpoint(consumed->wait).cut};
                    while (!pending.empty()) {
                        const auto site = pending.back(); pending.pop_back();
                        if (!control.reachable[site] || afterConsumption[site]) { continue; }
                        afterConsumption[site] = true;
                        ++result.work.normalKeySites;
                        const auto& next = control.graph.sites[site].successors;
                        pending.insert(pending.end(), next.begin(), next.end());
                    }
                }
                if (!selectedKeyUsesOutside(afterConsumption, reverse, wordIntersects)) { continue; }
            }
            const Command returnSet{Command::Publish, b.source, b.observer, b.key};
            const Command returnWait{Command::Acquire, b.source, b.observer, b.key};
            const Command forwardSet{Command::Publish, a.source, a.observer, a.key};
            const Command forwardWait{Command::Acquire, a.source, a.observer, a.key};
            // This contracted chain proves only event rearming. Payload
            // coverage remains the real sourceGapFacts certificate.
            const std::vector<std::pair<Command, FrontierBinding>> chain{
                {returnSet, {consumed->returnSource.cut, 1}},
                {returnWait, {facts.gap.cut, 0}},
                {forwardSet, {facts.gap.cut, 1}},
                {forwardWait, {current, 0}}};
            result.work.repairSourceCommands += chain.size();
            if (!frontier.eventChain(after->second.causal, chain)) { continue; }
            const auto request = result.decisions.size();
            const OrderedPacket endpoints{
                {consumed->returnSource.cut, returnSet,
                    EndpointPurpose::ConsumptionAcknowledgment, request, consumed->wait,
                    consumed->returnSource},
                {facts.gap.cut, returnWait,
                    EndpointPurpose::ConsumptionAcknowledgment, request, consumed->wait, facts.gap},
                {facts.gap.cut, forwardSet, EndpointPurpose::Completion, request,
                    NoAnalysisId, facts.gap},
                {current, forwardWait, EndpointPurpose::Completion, request}};
            auto packet = prepareOwnedPacket(endpoints);
            if (!packet || packet->restoredEndpoints || !preservePublications(*packet)) {
                continue;
            }
            packet->qualified = true;
            group.packet = std::move(*packet);
            group.forwardKey = forward;
            group.repairKey = reverse;
            group.repairedAcquisition = consumed->wait;
            return true;
        }
    }
    return false;
}
} // namespace mlir::pto::oahs::selected
