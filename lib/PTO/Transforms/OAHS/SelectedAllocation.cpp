// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <deque>

namespace mlir::pto::oahs::selected {
bool Constructor::canPublish(const State& state, Id key) const
{
    if (!state.causal.reachable() || key >= frontier.keys().size()) {
        return false;
    }
    const auto& facts = *state.causal.facts();
    const auto source = unsigned(frontier.keys()[key].source);
    const auto consumption = 2 * PipeCount + frontier.keys().size() + key;
    return facts.events[key].occupancy == 1 &&
           (frontierContains(facts.reach[consumption], source) ||
            frontierContains(facts.reach[consumption], PipeCount + source));
}
Id Constructor::reusable(Pipe source, Pipe observer, const State& state)
{
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != source || identity.observer != observer || closedKeys.count(key)) {
            continue;
        }
        ++result.work.keyQueries;
        if (canPublish(state, key)) {
            return key;
        }
    }
    return NoAnalysisId;
}
bool Constructor::clearInterval(Id key, Cut source, Cut target) const
{
    if (source == target) {
        return true; // appended after all existing words at this cut
    }
    if (!control.straight(source, target)) {
        return false;
    }
    const auto& identity = frontier.keys()[key];
    for (const auto& endpoint : ledger.records()) {
        if (!ledger.active(endpoint.id)) continue;
        const auto& c = endpoint.command;
        if ((c.kind != Command::Publish && c.kind != Command::Acquire) || c.source != identity.source ||
            c.observer != identity.observer || c.key != identity.key) {
            continue;
        }
        if (control.straight(source, endpoint.cut) && control.straight(endpoint.cut, target) &&
            endpoint.cut != source) {
            return false;
        }
    }
    return true;
}
std::vector<Pipe> Constructor::route(Pipe source, Pipe observer) const
{
    std::vector<Id> parent(PipeCount, NoAnalysisId);
    std::deque<Pipe> queue{source};
    parent[unsigned(source)] = unsigned(source);
    while (!queue.empty() && parent[unsigned(observer)] == NoAnalysisId) {
        const auto at = queue.front();
        queue.pop_front();
        for (unsigned next = 0; next < PipeCount; ++next) {
            const bool eligible = std::any_of(frontier.keys().begin(), frontier.keys().end(), [&](const auto& key) {
                return key.source == at && unsigned(key.observer) == next;
            });
            if (eligible && parent[next] == NoAnalysisId) {
                parent[next] = unsigned(at);
                queue.push_back(Pipe(next));
            }
        }
    }
    if (parent[unsigned(observer)] == NoAnalysisId) {
        return {};
    }
    std::vector<Pipe> path{observer};
    while (path.back() != source) {
        path.push_back(Pipe(parent[unsigned(path.back())]));
    }
    std::reverse(path.begin(), path.end());
    return path;
}
bool Constructor::acknowledgment(Pipe source, Pipe observer, Cut& publication, Id& key, SelectedDecision& decision)
{
    Id oldWait = NoAnalysisId, reverse = NoAnalysisId;
    Cut moved = publication;
    for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
        const auto& identity = frontier.keys()[candidate];
        if (identity.source != source || identity.observer != observer || closedKeys.count(candidate) ||
            currentState().causal.facts()->events[candidate].occupancy != 1 ||
            currentState().consumptions[candidate].size() != 1) {
            continue;
        }
        const auto wait = currentState().consumptions[candidate].front();
        const auto cut = ledger.endpoint(wait).cut;
        if (!control.straight(cut, current) || !control.straight(publication, current)) {
            continue;
        }
        const auto after = cache.afterEndpoint.find(wait);
        if (after == cache.afterEndpoint.end()) {
            continue;
        }
        const auto newCut = control.position[cut] > control.position[publication] ? cut : publication;
        if (!clearInterval(candidate, newCut, current)) {
            continue;
        }
        // F7 chooses the lowest reverse key whose COMPLETE certificate passes.
        // A lower reusable key with an intervening use must not hide a later
        // eligible key for this same stable forward repair target.
        for (Id reverseKey = 0; reverseKey < frontier.keys().size(); ++reverseKey) {
            const auto& reverseIdentity = frontier.keys()[reverseKey];
            if (reverseIdentity.source != observer || reverseIdentity.observer != source ||
                closedKeys.count(reverseKey)) continue;
            ++result.work.keyQueries;
            if (!canPublish(after->second, reverseKey) || !clearInterval(reverseKey, cut, newCut)) continue;
            key = candidate;
            oldWait = wait;
            reverse = reverseKey;
            moved = newCut;
            break;
        }
        if (oldWait != NoAnalysisId) break;
    }
    if (oldWait == NoAnalysisId) {
        return fail(SelectedFailure::EventResource,
            "no reusable key or nonrecursive consumption acknowledgment", publication);
    }
    const auto& identity = frontier.keys()[reverse];
    const auto request = result.decisions.size();
    decision.repairedAcquisition = oldWait;
    decision.repairedForwardKey = frontier.keys()[key].key;
    decision.repairReverseKey = identity.key;
    decision.repairInputVersion = ledger.version();
    decision.endpoints.push_back(ledger.after(oldWait,
        {Command::Publish, observer, source, identity.key}, EndpointPurpose::ConsumptionAcknowledgment, request, oldWait));
    decision.endpoints.push_back(ledger.append(moved,
        {Command::Acquire, observer, source, identity.key}, EndpointPurpose::ConsumptionAcknowledgment, request, oldWait));
    ++result.work.acknowledgments;
    decision.enlargedPrefix |= moved != publication;
    publication = moved;
    if (!update()) {
        return false;
    }
    decision.repairOutputVersion = ledger.version();
    if (!canPublish(cache.cuts[publication].before, key)) {
        return fail(SelectedFailure::SelectedUpdate,
            "selected acknowledgment does not rearm its new publication", publication);
    }
    return true;
}
Id Constructor::virginAtStart(Cut cut, Pipe source, Pipe observer) const
{
    if (cut == current || !control.straight(cut, current) ||
        control.components[control.component[cut]].cyclic ||
        control.components[activeComponent].cyclic ||
        control.wordOccurrences[control.canonicalCut[cut]].size() != 1 ||
        control.wordOccurrences[control.canonicalCut[current]].size() != 1) return NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& e = frontier.keys()[key];
        if (e.source != source || e.observer != observer || closedKeys.count(key) ||
            recurringKeys.count(key) || !canPublish(cache.cuts[cut].incoming, key)) continue;
        // No neighboring selected generation exists. Reuse/recurring gaps need
        // their own certificates and deliberately stay on the existing path.
        if (std::none_of(ledger.records().begin(), ledger.records().end(), [&](const auto& record) {
                const auto& command = record.command;
                return ledger.active(record.id) && command.kind != Command::Barrier &&
                    command.kind != Command::BarrierAll && command.source == source &&
                    command.observer == observer && command.key == e.key;
            })) return key;
    }
    return NoAnalysisId;
}
Id Constructor::helperFreeBinding(const Group& group, Pipe observer) const
{
    if (group.atWordStart) return group.version == ledger.version()
        ? virginAtStart(group.publication, group.source, observer) : NoAnalysisId;
    // Structured F3 queries already supply a complete candidate certificate.
    // Reuse it without another solve, only if it needs no reverse helper.
    if (!group.publications.empty()) return group.version == ledger.version() &&
        group.entryReturnKey == NoAnalysisId ? group.forwardKey : NoAnalysisId;
    if (group.common || !control.straight(group.publication, current) ||
        control.components[activeComponent].cyclic ||
        control.components[control.component[group.publication]].cyclic ||
        control.wordOccurrences[control.canonicalCut[group.publication]].size() != 1 ||
        control.wordOccurrences[control.canonicalCut[current]].size() != 1) return NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& e = frontier.keys()[key];
        if (e.source != group.source || e.observer != observer || closedKeys.count(key) ||
            recurringKeys.count(key) || !canPublish(cache.cuts[group.publication].before, key)) continue;
        // Conservative positive certificate: a virgin key has no old/next
        // selected generation. Unsupported reused-key queries return Unknown.
        if (std::none_of(ledger.records().begin(),ledger.records().end(),[&](const auto& endpoint) {
                const auto& c = endpoint.command;
                return ledger.active(endpoint.id) && (c.kind == Command::Publish || c.kind == Command::Acquire) &&
                    c.source == e.source && c.observer == e.observer && c.key == e.key;
            })) return key;
    }
    return NoAnalysisId;
}
bool Constructor::needsCommonAcknowledgment(const State& afterForward, Id key) const
{
    // A syntactically last body operation is not a last dynamic operation. The
    // backward summary retains original backedges and all branch alternatives.
    if (options.deferredAcyclicAcknowledgments &&
        std::none_of(control.components.begin(), control.components.end(),
                     [](const auto& component) { return component.cyclic; }) &&
        control.wordOccurrences[control.canonicalCut[current]].size() == 1 &&
        control.lookahead.balancedTransfer({control.graph.entry}, current,
                                          control.graph.entry, control.graph.exit)) {
        const auto& e = frontier.keys()[key];
        const bool otherPublication = std::any_of(ledger.records().begin(), ledger.records().end(),
            [&](const auto& endpoint) {
                const auto& command = endpoint.command;
                return ledger.active(endpoint.id) && endpoint.cut != current &&
                    command.kind == Command::Publish && command.source == e.source &&
                    command.observer == e.observer && command.key == e.key;
            });
        if (!otherPublication) return false;
    }
    if (control.lookahead.mayIssueAfter(current)) return true;
    // A selected word may be shared by several original occurrences. Being
    // terminal at only this occurrence is not a terminal-channel certificate.
    const auto canonical = control.canonicalCut[current];
    const auto& occurrences = control.wordOccurrences[canonical];
    if (std::count_if(occurrences.begin(), occurrences.end(),
                     [&](Id site) { return control.reachable[site]; }) != 1) return true;
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId) return true;
    const auto checked = frontier.inspect(afterForward.causal, operation);
    if (!checked.applied && checked.failure != FrontierFailure::Payload) return true;
    const auto observer = program.operations[operation].pipe;
    if (std::any_of(checked.residuals.begin(), checked.residuals.end(),
                   [&](const auto& r) { return r.source != observer; })) return true;
    // Only the final cross-engine acquisition of this terminal payload can use
    // this rule. Later fixed/recurring words can have rearming obligations even
    // when there are no later payloads; inspect the CURRENT ledger, not the
    // immutable analysis. Terminal ALL neither consumes nor rearms an event.
    std::vector<bool> seen(control.graph.sites.size());
    auto todo = control.graph.sites[current].successors;
    while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (seen[site]) continue;
        seen[site] = true;
        for (auto endpoint : ledger.word(site)) {
            if (ledger.endpoint(endpoint).command.kind != Command::BarrierAll) return true;
        }
        const auto& next = control.graph.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    return false;
}
bool Constructor::edge(Pipe source, Pipe observer, Cut& publication, bool closed, SelectedDecision& decision, Id certifiedKey)
{
    const bool recurringClosed = closed && control.components[activeComponent].cyclic;
    const auto binding = closedBindings.find({source, observer});
    const bool retained = recurringClosed && binding != closedBindings.end();
    Id key = retained ? binding->second.first : NoAnalysisId;
    if (certifiedKey != NoAnalysisId) {
        if (closed || certifiedKey >= frontier.keys().size() || closedKeys.count(certifiedKey) ||
            !canPublish(cache.cuts[publication].before, certifiedKey) ||
            !clearInterval(certifiedKey, publication, current))
            return fail(SelectedFailure::SelectedUpdate, "binding certificate no longer applies", publication);
        key = certifiedKey;
    } else if (retained) {
        if (!canPublish(cache.cuts[publication].before, key) &&
            !restoreRearming(key, publication)) {
            return fail(SelectedFailure::EventResource,
                "recurring forward role lacks its consumption path", publication);
        }
    } else {
        for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
            const auto& identity = frontier.keys()[candidate];
            if (identity.source != source || identity.observer != observer || closedKeys.count(candidate)) {
                continue;
            }
            ++result.work.keyQueries;
            if (canPublish(cache.cuts[publication].before, candidate) &&
                clearInterval(candidate, publication, current)) {
                key = candidate;
                break;
            }
        }
        if (key == NoAnalysisId && !acknowledgment(source, observer, publication, key, decision)) {
            return false;
        }
    }
    const auto number = frontier.keys()[key].key;
    const auto request = result.decisions.size();
    decision.endpoints.push_back(ledger.append(publication,
        {Command::Publish, source, observer, number}, EndpointPurpose::Completion, request));
    const auto wait = ledger.append(current,
        {Command::Acquire, source, observer, number}, EndpointPurpose::Completion, request);
    decision.endpoints.push_back(wait);
    if (!closed) {
        return update();
    }
    // A recurring closed word is one selected edit. Replaying its forward half
    // over a backedge before adding its acknowledgment would reject a protocol
    // that is deliberately not complete yet. Select the reply from the actual
    // local forward transfers, then check the complete word on the full graph.
    if (publication != current) {
        return fail(SelectedFailure::MissingParticipation, "closed word requires one common cut", current);
    }
    auto afterForward = currentState();
    const auto offset = ledger.word(current).size() - 2;
    auto sent = frontier.command(afterForward.causal,
        {Command::Publish, source, observer, number}, {current, offset});
    if (!sent.applied) return fail(SelectedFailure::SelectedUpdate, sent.reason, current);
    auto acquired = frontier.command(sent.state,
        {Command::Acquire, source, observer, number}, {current, offset + 1});
    if (!acquired.applied) return fail(SelectedFailure::SelectedUpdate, acquired.reason, current);
    afterForward.causal = std::move(acquired.state);
    if (observer == program.operations[control.graph.operations[current]].pipe &&
        !needsCommonAcknowledgment(afterForward, key)) {
        if (options.deferredAcyclicAcknowledgments && control.lookahead.mayIssueAfter(current))
            ++result.work.deferredAcknowledgments;
        // The token is consumed. Either no future use needs publisher knowledge,
        // or the acyclic policy defers that obligation to actual key reuse.
        // Do not invent consumption knowledge. Subsequent F7 and validation
        // and the native reconstruction checker are unchanged.
        ++result.work.commonCutTransfers;
        return update();
    }
    const auto reverse = retained ? binding->second.second : reusable(observer, source, afterForward);
    if (reverse == NoAnalysisId || !canPublish(afterForward, reverse)) {
        return fail(SelectedFailure::EventResource,
            "common-cut acknowledgment has no independently reusable reverse key", current);
    }
    const auto reverseNumber = frontier.keys()[reverse].key;
    decision.endpoints.push_back(ledger.append(current,
        {Command::Publish, observer, source, reverseNumber}, EndpointPurpose::ConsumptionAcknowledgment, request, wait));
    decision.endpoints.push_back(ledger.append(current,
        {Command::Acquire, observer, source, reverseNumber}, EndpointPurpose::ConsumptionAcknowledgment, request, wait));
    rememberReturn(decision.endpoints[decision.endpoints.size() - 2], decision.endpoints.back());
    ++result.work.acknowledgments;
    ++result.work.commonCutTransfers;
    if (!update()) {
        return false;
    }
    if (recurringClosed && !retained) {
        closedBindings[{source, observer}] = {key, reverse};
        closedKeys.insert(key);
        closedKeys.insert(reverse);
    }
    return true;
}
bool Constructor::restoreReturns(Id key)
{
    const auto& identity = frontier.keys()[key];
    const auto found = pendingRearming.find({identity.observer, identity.source});
    if (found == pendingRearming.end()) return false;
    bool changed = false;
    for (const auto& helper : found->second) {
        if (ledger.active(helper.second)) continue;
        const auto wait = ledger.endpoint(helper.second).acknowledges;
        const auto& forward = ledger.endpoint(wait).command;
        if (forward.source != identity.source || forward.observer != identity.observer || forward.key != identity.key)
            continue;
        // A newly selected publication can introduce an EARLIER deadline than
        // the return which discharged this obligation. Restore its original
        // source prefix, before advancing that publication; never assume a
        // receipt from the still-later necessary transfer.
        ledger.restoreAfter(helper.first, wait);
        ledger.restoreAfter(helper.second, helper.first);
        requiredReturns.insert(helper.second);
        ++result.work.acknowledgments;
        ++result.work.rearmingRestored;
        --result.work.rearmingDischarged;
        changed = true;
    }
    return changed;
}
bool Constructor::restoreRearming(Id key, Cut publication)
{
    return restoreReturns(key) && update() && canPublish(cache.cuts[publication].before, key);
}

void Constructor::rememberReturn(Id publication, Id acquisition)
{
    const auto& c = ledger.endpoint(acquisition).command;
    pendingRearming[{c.source, c.observer}].push_back({publication, acquisition});
}

bool Constructor::returnBeforeUse(Id helperWait, Id necessaryWait)
{
    const auto& helper = ledger.endpoint(helperWait);
    const auto& necessary = ledger.endpoint(necessaryWait);
    const auto sameKey = [](const Command& a, const Command& b) {
        return a.source == b.source && a.observer == b.observer && a.key == b.key;
    };
    // Track whether the ACTUAL return publication follows this consumption.
    // Merely reaching an acquisition of an older source prefix is insufficient.
    using Point = std::tuple<Cut, Id, bool>;
    std::vector<Point> todo;
    for (auto cut : control.wordOccurrences[helper.cut]) {
        if (!control.reachable[cut]) continue;
        const auto& word = ledger.word(cut);
        const auto at = std::find(word.begin(), word.end(), helperWait);
        todo.emplace_back(cut, Id(at - word.begin()) + 1, false);
    }
    std::set<Point> seen;
    bool reached = false;
    while (!todo.empty()) {
        const auto point = todo.back(); todo.pop_back();
        if (!seen.insert(point).second) continue;
        ++result.work.rearmingQuerySites;
        auto [cut, offset, published] = point;
        const auto& word = ledger.word(cut);
        bool acquired = false;
        for (; offset < word.size(); ++offset) {
            const auto id = word[offset];
            const auto& c = ledger.endpoint(id).command;
            if (id == necessaryWait && published) { acquired = reached = true; break; }
            if (c.kind == Command::Publish && sameKey(c, necessary.command)) published = true;
            // Do not remove completion used by a payload or transmitted to a
            // third engine before the real return. This also stops at every
            // earlier selected republication deadline, including backedges.
            if ((c.kind == Command::BarrierAll && cut != control.graph.exit) ||
                (c.kind == Command::Publish && c.source == helper.command.observer) ||
                ((c.kind == Command::Publish || c.kind == Command::Acquire) && sameKey(c, helper.command)))
                return false;
        }
        if (acquired) continue;
        const auto operation = control.graph.operations[cut];
        if (operation != NoAnalysisId && program.operations[operation].pipe == helper.command.observer)
            return false;
        const auto& next = control.graph.sites[cut].successors;
        if (next.empty()) {
            if (cut != control.graph.exit) return false;
            continue; // consumed token, no subsequent republication deadline
        }
        for (auto successor : next) todo.emplace_back(successor, 0, published);
    }
    return reached;
}

bool Constructor::settleRearming(const SelectedDecision& decision)
{
    // This certificate refers to the selected original-graph continuation.
    // The construction-only loop hypothesis traversal does not retain that
    // continuation's token generations. Keep its closed fallback until it has
    // a contextual interface; do not enable extra global solves for this rule.
    if (!needsContextualReplay) return true;
    std::set<std::pair<Pipe, Pipe>> directions;
    for (auto id : decision.endpoints) {
        const auto& endpoint = ledger.endpoint(id);
        if (endpoint.command.kind != Command::Acquire) continue;
        const auto direction = std::make_pair(endpoint.command.source, endpoint.command.observer);
        if (endpoint.purpose == EndpointPurpose::Completion) necessaryReturns[direction].push_back(id);
        if (endpoint.purpose == EndpointPurpose::Completion ||
            endpoint.purpose == EndpointPurpose::ConsumptionAcknowledgment) directions.insert(direction);
    }
    bool changed = false;
    for (const auto& direction : directions) {
        const auto pending = pendingRearming.find(direction);
        const auto returns = necessaryReturns.find(direction);
        if (pending == pendingRearming.end() || returns == necessaryReturns.end()) continue;
        auto& paired = pairedReturns[direction];
        const auto helpers = pending->second.size(), actuals = returns->second.size();
        if (paired.first == helpers && paired.second == actuals) continue;
        const auto firstHelper = paired.second == actuals ? paired.first : 0;
        for (Id h = firstHelper; h < helpers; ++h) {
            const auto& helper = pending->second[h];
            if (!ledger.active(helper.second) || requiredReturns.count(helper.second)) continue;
            for (Id r = h < paired.first ? paired.second : 0; r < actuals; ++r) {
                ++result.work.rearmingPairVisits;
                if (!returnBeforeUse(helper.second, returns->second[r])) continue;
                ledger.erase(helper.first);
                ledger.erase(helper.second);
                --result.work.acknowledgments;
                ++result.work.rearmingDischarged;
                changed = true;
                break;
            }
        }
        paired = {helpers, actuals};
    }
    // One structural certificate per helper/actual-return pair, no speculative
    // command populations or cold-check deletion sweep. Replay the changed
    // selected ledger once and recheck every previously finalized payload.
    return !changed || update();
}

bool Constructor::bind(Group& group, RequirementStage stage)
{
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    if (group.atWordStart) {
        const auto key = virginAtStart(group.publication, group.source, observer);
        if (group.version != ledger.version() || key != group.forwardKey || key == NoAnalysisId)
            return fail(SelectedFailure::SelectedUpdate, "stale source-gap binding", current);
        const auto covered = coverage(group.publication, group.source, group.requirements, true);
        if (std::any_of(group.requirements.begin(), group.requirements.end(), [&](const auto& r) {
                return !covered.count(accessClass(r));
            })) return fail(SelectedFailure::SelectedUpdate, "source-gap coverage changed", current);
        SelectedDecision decision;
        decision.consumer = current; decision.publication = group.publication;
        decision.publicationAtWordStart = true; decision.stage = stage;
        decision.source = group.source; decision.observer = observer;
        decision.required = group.requirements;
        const auto number = frontier.keys()[key].key;
        decision.endpoints.push_back(ledger.prepend(group.publication,
            {Command::Publish, group.source, observer, number}, EndpointPurpose::Completion, result.decisions.size()));
        decision.endpoints.push_back(ledger.append(current,
            {Command::Acquire, group.source, observer, number}, EndpointPurpose::Completion, result.decisions.size()));
        ++result.work.gapPublications;
        if (!update() || !settleRearming(decision)) return false;
        result.decisions.push_back(std::move(decision));
        return true;
    }
    if (!group.publications.empty()) {
        if (group.version != ledger.version() || group.forwardKey >= frontier.keys().size()) {
            return fail(SelectedFailure::SelectedUpdate, "stale alternative source frontier", current);
        }
        const auto& key = frontier.keys()[group.forwardKey];
        if (key.source != group.source || key.observer != observer) {
            return fail(SelectedFailure::SelectedUpdate, "alternative source key changed direction", current);
        }
        SelectedDecision decision;
        decision.consumer = current;
        decision.publication = group.publication;
        decision.publicationFrontier = group.publications;
        decision.stage = stage;
        decision.source = group.source;
        decision.observer = observer;
        decision.required = group.requirements;
        const auto request = result.decisions.size();
        for (auto cut : group.publications) {
            decision.endpoints.push_back(ledger.append(cut,
                {Command::Publish, group.source, observer, key.key}, EndpointPurpose::Completion, request));
        }
        const auto acquisition = group.entryAcquisition == NoAnalysisId ? current : group.entryAcquisition;
        const auto acquired = ledger.append(acquisition,
            {Command::Acquire, group.source, observer, key.key}, EndpointPurpose::Completion, request);
        decision.endpoints.push_back(acquired);
        if (group.entryAcquisition != NoAnalysisId) {
            ++result.work.loopEntryTransfers;
            needsContextualReplay = true;
            if (group.entryRepeats) {
                entryProtocolKeys.insert(group.forwardKey);
                recurringKeys.insert(group.forwardKey);
                closedKeys.insert(group.forwardKey);
            }
            if (group.entryReturnKey != NoAnalysisId) {
                const auto& reply = frontier.keys()[group.entryReturnKey];
                if (group.entryRepeats) {
                    entryProtocolKeys.insert(group.entryReturnKey);
                    recurringKeys.insert(group.entryReturnKey);
                    closedKeys.insert(group.entryReturnKey);
                }
                decision.endpoints.push_back(ledger.append(acquisition,
                    {Command::Publish, observer, group.source, reply.key}, EndpointPurpose::ConsumptionAcknowledgment, request, acquired));
                decision.endpoints.push_back(ledger.append(acquisition,
                    {Command::Acquire, observer, group.source, reply.key}, EndpointPurpose::ConsumptionAcknowledgment, request, acquired));
                rememberReturn(decision.endpoints[decision.endpoints.size() - 2], decision.endpoints.back());
                ++result.work.acknowledgments;
            }
        }
        if (!update()) return false;
        if (!settleRearming(decision)) return false;
        result.decisions.push_back(std::move(decision));
        return true;
    }
    if (group.bindingCertified && (group.version != ledger.version() ||
        helperFreeBinding(group, observer) != group.forwardKey))
        return fail(SelectedFailure::SelectedUpdate, "stale equal-coverage certificate", current);
    const auto path = route(group.source, observer);
    if (path.empty()) {
        return fail(SelectedFailure::EventResource, "no eligible engine route for the required completion", current);
    }
    SelectedDecision decision;
    decision.consumer = current;
    decision.publication = group.publication;
    decision.stage = stage;
    decision.source = group.source;
    decision.observer = observer;
    decision.required = group.requirements;
    decision.commonCut = group.common;
    auto source = group.publication;
    for (Id hop = 1; hop < path.size(); ++hop) {
        if (!edge(path[hop - 1], path[hop], source, group.common, decision,
                  hop == 1 && group.bindingCertified ? group.forwardKey : NoAnalysisId)) {
            return false;
        }
        if (hop == 1) {
            decision.publication = source;
        }
        source = current;
    }
    if (!settleRearming(decision)) return false;
    result.decisions.push_back(std::move(decision));
    return true;
}
} // namespace mlir::pto::oahs::selected
