// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
namespace {
bool uniqueAcyclicWord(const Control& control, Cut cut)
{
    const bool validCut = cut < control.canonicalCut.size() && control.canonicalCut[cut] == cut;
    if (!validCut) {
        return false;
    }
    const auto component = control.component[cut];
    if (component == NoAnalysisId || control.components[component].cyclic) { return false; }
    return std::count_if(control.wordOccurrences[cut].begin(),
        control.wordOccurrences[cut].end(), [&](Cut site) {
            return control.reachable[site];
        }) == 1;
}
} // namespace

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
    const bool uniqueWords = uniqueAcyclicWord(control, source) &&
        uniqueAcyclicWord(control, facts.deadline);
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
        old.command.key == identity.key && uniqueAcyclicWord(control, old.cut) &&
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

std::optional<JoinedConsumptionFrontier> Constructor::joinedConsumptionFrontier(
    const SourceGapQualification& facts, Id forward)
{
    const bool valid = facts.proved() && facts.version == ledger.version() &&
        facts.gap.left == NoAnalysisId && forward < frontier.keys().size() &&
        facts.prefixes.size() == 1 && uniqueAcyclicWord(control, facts.gap.cut) &&
        uniqueAcyclicWord(control, facts.deadline);
    if (!valid) { return {}; }
    const auto source = facts.gap.cut;
    const auto& toDeadline = control.correspondence(source, facts.deadline);
    const bool matchedDeadline = toDeadline.proved() && toDeadline.pairs.size() == 1 &&
        toDeadline.pairs.front().first == source;
    if (!matchedDeadline) {
        return {};
    }
    const auto& waits = cache.cuts[source].incoming.consumptions[forward];
    const bool alternatives = waits.size() >= 2;
    if (!alternatives) {
        return {};
    }
    const auto& identity = frontier.keys()[forward];
    JoinedConsumptionFrontier out;
    out.version = ledger.version();
    std::vector<Cut> waitCuts;
    std::map<Cut, Id> waitByCut;
    std::set<Id> oldWaits;
    for (auto wait : waits) {
        const bool activeWait = wait < ledger.records().size() && ledger.active(wait);
        if (!activeWait) {
            return {};
        }
        const auto& old = ledger.endpoint(wait);
        const bool matching = old.command.kind == Command::Acquire &&
            old.command.source == identity.source && old.command.observer == identity.observer &&
            old.command.key == identity.key && uniqueAcyclicWord(control, old.cut) &&
            waitByCut.emplace(old.cut, wait).second;
        if (!matching) { return {}; }
        const auto gap = ledger.gapAfter(wait);
        const auto after = cache.afterEndpoint.find(wait);
        const bool consumed = gap && after != cache.afterEndpoint.end() &&
            after->second.consumptions[forward] == std::vector<Id>{wait} &&
            after->second.causal.facts()->events[forward].occupancy == 1;
        if (!consumed) { return {}; }
        waitCuts.push_back(old.cut);
        oldWaits.insert(wait);
        out.alternatives.push_back({wait, *gap, facts.gap, ledger.version()});
    }
    const auto& matching = control.correspondence(waitCuts, {source});
    const bool matchedSources = matching.proved() && matching.pairs.size() == waits.size();
    if (!matchedSources) {
        return {};
    }
    for (const auto& [old, next] : matching.pairs) {
        if (next != source || !waitByCut.count(old)) { return {}; }
    }

    // Original correspondence proves one old WAIT on each participating path.
    // Exclude another selected use of its event generation before republication.
    std::vector<bool> interior(control.graph.sites.size(), false);
    result.work.normalKeySites += interior.size();
    std::vector<Cut> pending;
    for (auto cut : waitCuts) {
        const auto& next = control.graph.sites[cut].successors;
        pending.insert(pending.end(), next.begin(), next.end());
    }
    while (!pending.empty()) {
        const auto at = pending.back();
        pending.pop_back();
        if (at == source || !control.reachable[at] || interior[at]) { continue; }
        interior[at] = true;
        ++result.work.normalKeySites;
        const auto& next = control.graph.sites[at].successors;
        pending.insert(pending.end(), next.begin(), next.end());
    }
    std::map<Cut, bool> wordIntersects;
    for (auto id : ledger.eventUses(identity)) {
        ++result.work.repairNeighborUses;
        const bool relevantUse = ledger.active(id) && !oldWaits.count(id);
        if (!relevantUse) {
            continue;
        }
        const auto& endpoint = ledger.endpoint(id);
        if (endpoint.cut == source) { continue; }
        const auto wordCut = control.canonicalCut[endpoint.cut];
        if (const auto found = waitByCut.find(wordCut); found != waitByCut.end()) {
            const auto [cached, inserted] = wordIntersects.try_emplace(wordCut, false);
            if (inserted) {
                const auto& word = ledger.word(wordCut);
                const auto oldAt = std::find(word.begin(), word.end(), found->second);
                cached->second = oldAt == word.end();
                if (!cached->second) {
                    for (auto next = oldAt + 1; next != word.end(); ++next) {
                        ++result.work.normalKeySites;
                        const auto& command = ledger.endpoint(*next).command;
                        const bool eventUse = command.kind == Command::Publish ||
                            command.kind == Command::Acquire;
                        cached->second |= eventUse && keyIndex(frontier, command) == forward;
                    }
                }
            }
            if (cached->second) { return {}; }
            continue;
        }
        const auto [cached, inserted] = wordIntersects.try_emplace(wordCut, false);
        if (inserted) {
            for (auto site : control.wordOccurrences[wordCut]) {
                ++result.work.normalKeySites;
                cached->second |= interior[site];
            }
        }
        if (cached->second) { return {}; }
    }
    if (facts.prefixes.front().facts()->events[forward].occupancy != 1) { return {}; }
    return out;
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
        std::vector<ConsumptionFrontier> arms;
        if (const auto single = singletonConsumptionFrontier(facts, forward)) {
            arms.push_back(*single);
        } else if (const auto joined = joinedConsumptionFrontier(facts, forward)) {
            arms = joined->alternatives;
        }
        if (arms.empty()) { continue; }
        // Historical reverse reuse is currently certified only for one old
        // consumption. Joined sources require a virgin reverse key until a
        // neighboring-use proof spans every alternative continuation.
        std::vector<bool> afterConsumption;
        std::map<Cut, bool> wordIntersects;
        for (Id reverse = 0; reverse < frontier.keys().size(); ++reverse) {
            const auto& b = frontier.keys()[reverse];
            const bool eligibleReverse = b.source == observer && b.observer == group.source &&
                helperFreeKey(reverse);
            if (!eligibleReverse) { continue; }
            const auto& reverseUses = ledger.eventUses(b);
            const bool unsupportedHistoricalJoin = arms.size() != 1 && !reverseUses.empty();
            if (unsupportedHistoricalJoin) { continue; }
            const auto firstAfter = cache.afterEndpoint.find(arms.front().wait);
            const bool initiallyAvailable = firstAfter != cache.afterEndpoint.end() &&
                canPublish(firstAfter->second, reverse);
            if (!initiallyAvailable) { continue; }
            if (!reverseUses.empty()) {
                if (afterConsumption.empty()) {
                    afterConsumption.resize(control.graph.sites.size());
                    result.work.normalKeySites += afterConsumption.size();
                    std::vector<Cut> pending{ledger.endpoint(arms.front().wait).cut};
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
            bool legal = true;
            for (const auto& old : arms) {
                const auto after = cache.afterEndpoint.find(old.wait);
                const bool available = after != cache.afterEndpoint.end() &&
                    canPublish(after->second, reverse);
                if (!available) { legal = false; break; }
                // The contracted chain establishes event rearming only.
                // sourceGapFacts supplies payload coverage independently.
                const std::vector<std::pair<Command, FrontierBinding>> chain{
                    {returnSet, {old.returnSource.cut, 1}},
                    {returnWait, {facts.gap.cut, 0}},
                    {forwardSet, {facts.gap.cut, 1}},
                    {forwardWait, {current, 0}}};
                result.work.repairSourceCommands += chain.size();
                if (!frontier.eventChain(after->second.causal, chain)) {
                    legal = false;
                    break;
                }
            }
            if (!legal) { continue; }
            const auto request = result.decisions.size();
            OrderedPacket endpoints;
            std::vector<Cut> returnSources;
            for (const auto& old : arms) {
                endpoints.push_back({old.returnSource.cut, returnSet,
                    EndpointPurpose::ConsumptionAcknowledgment, request, old.wait,
                    old.returnSource});
                returnSources.push_back(control.canonicalCut[old.returnSource.cut]);
            }
            const auto acknowledged = arms.size() == 1 ? arms.front().wait : NoAnalysisId;
            endpoints.push_back({facts.gap.cut, returnWait,
                EndpointPurpose::ConsumptionAcknowledgment, request, acknowledged, facts.gap});
            endpoints.push_back({facts.gap.cut, forwardSet,
                EndpointPurpose::Completion, request, NoAnalysisId, facts.gap});
            endpoints.push_back({current, forwardWait, EndpointPurpose::Completion, request});
            auto packet = prepareOwnedPacket(endpoints);
            if (!packet || packet->restoredEndpoints || !preservePublications(*packet)) { continue; }
            packet->qualified = true;
            group.packet = std::move(*packet);
            group.forwardKey = forward;
            group.repairKey = reverse;
            group.repairedAcquisition = acknowledged;
            std::sort(returnSources.begin(), returnSources.end());
            group.repairPublications = std::move(returnSources);
            return true;
        }
    }
    return false;
}
} // namespace mlir::pto::oahs::selected
