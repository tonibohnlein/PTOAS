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
#include <tuple>

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
bool Constructor::canPublishAt(Cut cut, Id key) const
{
    // One emitted word may execute in both the peeled first visit and later
    // visits. Publisher knowledge at only the later visit cannot rearm its
    // first publication after a preceding sibling.
    for (auto occurrence : control.wordOccurrences[control.canonicalCut[cut]])
        if (control.reachable[occurrence] && !canPublish(cache.cuts[occurrence].before, key))
            return false;
    return true;
}
Id Constructor::reusable(Pipe source, Pipe observer, const State& state)
{
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != source || identity.observer != observer || !availableKey(key)) {
            continue;
        }
        ++result.work.keyQueries;
        // This helper creates a new consumption. Protect already-selected
        // neighboring publications, including earlier words on the next visit.
        if (canPublish(state, key) &&
            consumptionBeforeNextPublication(current, current, key)) {
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
    const auto& identity = frontier.keys()[key];
    if (!control.straight(source, target)) return false;
    for (const auto& endpoint : ledger.records()) {
        if (!ledger.active(endpoint.id)) continue;
        const auto& c = endpoint.command;
        if ((c.kind != Command::Publish && c.kind != Command::Acquire) || c.source != identity.source ||
            c.observer != identity.observer || c.key != identity.key) {
            continue;
        }
        for (auto occurrence : control.wordOccurrences[endpoint.cut]) {
            if (control.straight(source, occurrence) && control.straight(occurrence, target) &&
                occurrence != source) {
                return false;
            }
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
bool Constructor::crossControlReturn(Id wait, Cut publication, Id forward, Id reverse)
{
    ++result.work.splitRearmingQueries;
    // Exact consumption identity, at every execution of the publication word.
    for (auto at : control.wordOccurrences[control.canonicalCut[publication]]) {
        if (!control.reachable[at]) continue;
        const auto& state = cache.cuts[at].before;
        if (!state.causal.reachable() || state.causal.facts()->events[forward].occupancy != 1 ||
            state.consumptions[forward].size() != 1 || state.consumptions[forward].front() != wait)
            return false;
    }
    const auto& back = frontier.keys()[reverse];
    const auto& front = frontier.keys()[forward];
    // Track the actual old consumption, not its representative site's numeric
    // position. Branch bypasses, repeated entries and shared words must all
    // balance the new return, with no intervening use of its physical key.
    std::vector<std::pair<Cut, bool>> todo{{control.graph.entry, false}};
    std::set<std::pair<Cut, bool>> seen;
    while (!todo.empty()) {
        auto [cut, live] = todo.back(); todo.pop_back();
        if (!seen.insert({cut, live}).second) continue;
        ++result.work.splitRearmingSites;
        for (auto id : ledger.word(cut)) {
            const auto& command = ledger.endpoint(id).command;
            if (live && (command.kind == Command::Publish || command.kind == Command::Acquire) &&
                command.source == back.source && command.observer == back.observer && command.key == back.key)
                return false;
            if (live && command.kind == Command::Publish && command.source == front.source &&
                command.observer == front.observer && command.key == front.key) return false;
            if (id == wait) {
                if (live) return false;
                live = true;
            }
        }
        if (control.canonicalCut[cut] == control.canonicalCut[publication]) {
            if (!live) return false;
            live = false;
        }
        if (control.graph.sites[cut].successors.empty() && live) return false;
        for (auto next : control.graph.sites[cut].successors) todo.push_back({next, live});
    }
    return true;
}
bool Constructor::acknowledgment(Pipe source, Pipe observer, Cut& publication, Id& key, SelectedDecision& decision)
{
    Id oldWait = NoAnalysisId, reverse = NoAnalysisId;
    Cut moved = publication;
    Cut returnCut = NoAnalysisId;
    bool stagedAcrossControl = false;
    for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
        const auto& identity = frontier.keys()[candidate];
        if (identity.source != source || identity.observer != observer || !availableKey(candidate) ||
            currentState().causal.facts()->events[candidate].occupancy != 1 ||
            currentState().consumptions[candidate].size() != 1) {
            continue;
        }
        const auto wait = currentState().consumptions[candidate].front();
        auto cut = ledger.endpoint(wait).cut;
        // A shared endpoint's recorded cut may be in the repeated body while
        // the current deadline is in its peeled visit. Match an actual original
        // occurrence before applying the existing straight-corridor repair.
        if (options.firstWriteConsumers && !control.straight(cut, current)) {
            for (auto occurrence : control.wordOccurrences[control.canonicalCut[cut]])
                if (control.reachable[occurrence] && control.straight(occurrence, current)) {
                    cut = occurrence;
                    break;
                }
        }
        // A deferred shared receipt can have several mutually exclusive
        // occurrences feeding one later publication. Match every occurrence;
        // its record identifies the debt but supplies no consumption credit.
        const bool deferredAcrossControl = deferredByAcquisition.count(wait) != 0;
        const bool acrossControl = deferredAcrossControl || (options.firstWriteConsumers && publication != current &&
            (cut != ledger.endpoint(wait).cut || !control.straight(cut, current)));
        if ((!control.straight(cut, current) && !acrossControl) || !control.straight(publication, current)) {
            continue;
        }
        const auto after = cache.afterEndpoint.find(wait);
        if (after == cache.afterEndpoint.end()) {
            continue;
        }
        const auto newCut = control.straight(cut, current) &&
            control.position[cut] > control.position[publication] ? cut : publication;
        if (!clearInterval(candidate, newCut, current)) {
            continue;
        }
        const auto receipt = deferredAcrossControl ? deferredReturnCut(wait, newCut, candidate) : newCut;
        if (receipt == NoAnalysisId) {
            continue;
        }
        // F7 chooses the lowest reverse key whose COMPLETE certificate passes.
        // A lower reusable key with an intervening use must not hide a later
        // eligible key for this same stable forward repair target.
        for (Id reverseKey = 0; reverseKey < frontier.keys().size(); ++reverseKey) {
            const auto& reverseIdentity = frontier.keys()[reverseKey];
            if (reverseIdentity.source != observer || reverseIdentity.observer != source ||
                !availableKey(reverseKey)) continue;
            ++result.work.keyQueries;
            if (!canPublish(after->second, reverseKey)) continue;
            if (acrossControl ? !crossControlReturn(wait, receipt, candidate, reverseKey) :
                !clearInterval(reverseKey, cut, newCut)) {
                continue;
            }
            // The acyclic return half must stand on its own during replay.
            // Do not presume the new forward receipt rearms an already selected
            // later reverse publication. Existing first-write staging is separate.
            if (deferredAcrossControl && !consumptionBeforeNextPublication(cut, receipt, reverseKey)) {
                continue;
            }
            key = candidate;
            oldWait = wait;
            reverse = reverseKey;
            moved = newCut;
            returnCut = receipt;
            stagedAcrossControl = acrossControl && !deferredAcrossControl;
            break;
        }
        if (oldWait != NoAnalysisId) break;
    }
    if (oldWait == NoAnalysisId) {
        if (joinedAcknowledgment(source, observer, publication, key, decision)) return true;
        return fail(SelectedFailure::EventResource,
            "no reusable key or nonrecursive consumption acknowledgment", publication);
    }
    const auto& identity = frontier.keys()[reverse];
    const auto request = result.decisions.size();
    decision.repairedAcquisition = oldWait;
    decision.repairedForwardKey = frontier.keys()[key].key;
    decision.repairReverseKey = identity.key;
    decision.repairInputVersion = ledger.version();
    const auto sent = ledger.after(oldWait, {Command::Publish, observer, source, identity.key},
        EndpointPurpose::ConsumptionAcknowledgment, request, oldWait);
    decision.endpoints.push_back(sent);
    // Append the wait after existing outward publications, including when the
    // balancing boundary is the old word itself. Only its SET needs the exact
    // original receipt prefix; earlier publications must not acquire a new gate.
    decision.endpoints.push_back(ledger.append(returnCut, {Command::Acquire, observer, source, identity.key},
        EndpointPurpose::ConsumptionAcknowledgment, request, oldWait));
    ++result.work.acknowledgments;
    decision.enlargedPrefix |= moved != publication;
    publication = moved;
    if (stagedAcrossControl) {
        // The new forward receipt carries consumption of this return back to
        // its publisher. Validate the complete exchange in edge(), not its
        // temporarily incomplete reverse half. No live causal credit is added.
        // The existing online helper-restoration interface is common-cut:
        // it restores the WAIT after the SET in that same word. This split
        // return remains explicit until an anchor-aware interface exists.
        return true;
    }
    if (!update()) {
        return false;
    }
    decision.repairOutputVersion = ledger.version();
    if (!canPublishAt(publication, key)) {
        return fail(SelectedFailure::SelectedUpdate,
            "selected acknowledgment does not rearm its new publication", publication);
    }
    return true;
}
bool Constructor::joinedAcknowledgment(
    Pipe source, Pipe observer, Cut publication, Id& key, SelectedDecision& decision)
{
    // Alternative paths may consume the same physical key at different WAITs.
    // The joined consumer gate still knows that actual consumption. Return it
    // at the existing publication deadline, without choosing one branch's WAIT
    // as representative or moving the forward source behind its payload.
    if (publication == current ||
        (!cache.contextualFixedPoint && std::any_of(control.components.begin(), control.components.end(),
            [](const Component& c) { return c.cyclic; })) || !cache.success ||
        !control.straight(publication, current)) return false;
    const auto& state = cache.cuts[publication].before;
    if (!state.causal.reachable()) return false;
    for (Id forward = 0; forward < frontier.keys().size(); ++forward) {
        const auto& a = frontier.keys()[forward];
        if (a.source != source || a.observer != observer || !availableKey(forward) || state.consumptions[forward].size() < 2 ||
            state.causal.facts()->events[forward].occupancy != 1 ||
            !clearInterval(forward, publication, current)) continue;
        for (Id reverse = 0; reverse < frontier.keys().size(); ++reverse) {
            const auto& b = frontier.keys()[reverse];
            if (b.source != observer || b.observer != source || !availableKey(reverse) || !canPublishAt(publication, reverse)) continue;
            const Command publish{Command::Publish, observer, source, b.key};
            const Command acquire{Command::Acquire, observer, source, b.key};
            bool supported = true;
            for (auto at : control.wordOccurrences[control.canonicalCut[publication]]) {
                if (!control.reachable[at]) continue;
                const auto offset = ledger.word(at).size();
                auto sent = frontier.command(cache.cuts[at].before.causal, publish, {at, offset});
                if (!sent.applied) { supported = false; break; }
                auto received = frontier.command(sent.state, acquire, {at, offset + 1});
                if (!received.applied) { supported = false; break; }
                auto local = cache.cuts[at].before;
                local.causal = received.state;
                if (!canPublish(local, forward)) { supported = false; break; }
            }
            if (!supported) continue;
            // Check the full exchange together: its forward receipt may be the
            // path that rearms the reverse key on the next original visit.
            Ledger trial = ledger;
            trial.append(publication, publish, EndpointPurpose::ConsumptionAcknowledgment);
            trial.append(publication, acquire, EndpointPurpose::ConsumptionAcknowledgment);
            trial.append(publication, {Command::Publish, source, observer, a.key}, EndpointPurpose::Completion);
            trial.append(current, {Command::Acquire, source, observer, a.key}, EndpointPurpose::Completion);
            const auto checked = analyze(program, trial.commands(), {false});
            ++result.work.acknowledgmentChecks;
            result.work.acknowledgmentCheckSites += checked.stats.siteEvaluations;
            if (!checked.complete || !checked.protocol.empty() || !checked.diagnostics.empty() ||
                !checked.phaseResources.empty()) continue;
            const auto request = result.decisions.size();
            decision.endpoints.push_back(ledger.append(publication, publish,
                EndpointPurpose::ConsumptionAcknowledgment, request));
            decision.endpoints.push_back(ledger.append(publication, acquire,
                EndpointPurpose::ConsumptionAcknowledgment, request));
            // There is no single acknowledged endpoint. Keep this return
            // explicit; the single-anchor online discharge does not apply.
            decision.repairedForwardKey = a.key;
            decision.repairReverseKey = b.key;
            ++result.work.acknowledgments;
            ++result.work.joinedAcknowledgments;
            key = forward;
            return true;
        }
    }
    return false;
}
Id Constructor::reusableAtStart(Cut cut, Pipe source, Pipe observer) const
{
    if (cut == current || !control.straight(cut, current) ||
        control.components[control.component[cut]].cyclic ||
        control.components[activeComponent].cyclic ||
        control.wordOccurrences[control.canonicalCut[cut]].size() != 1 ||
        control.wordOccurrences[control.canonicalCut[current]].size() != 1) return NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& e = frontier.keys()[key];
        if (e.source != source || e.observer != observer || !availableKey(key) || !canPublish(cache.cuts[cut].incoming, key)) continue;
        // Source-time emptiness is insufficient: canPublish above also proves
        // the preceding consumption. Bound this reuse certificate to a straight,
        // acyclic history with no selected use at or after the proposed gap.
        // In particular, an endpoint in this word follows the gap, even when
        // the post-word state would certify reuse.
        if (std::none_of(ledger.records().begin(), ledger.records().end(), [&](const auto& record) {
                const auto& command = record.command;
                return ledger.active(record.id) && command.kind != Command::Barrier &&
                    command.kind != Command::BarrierAll && command.source == source &&
                    command.observer == observer && command.key == e.key &&
                    (record.cut == cut || !control.straight(record.cut, cut) ||
                     control.components[control.component[record.cut]].cyclic ||
                     control.wordOccurrences[control.canonicalCut[record.cut]].size() != 1);
            })) return key;
    }
    return NoAnalysisId;
}
Id Constructor::helperFreeBinding(const Group& group, Pipe observer) const
{
    if (group.finalReadSource) return group.version == ledger.version() ? group.forwardKey : NoAnalysisId;
    if (group.atWordStart) return group.version == ledger.version()
        ? reusableAtStart(group.publication, group.source, observer) : NoAnalysisId;
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
        if (e.source != group.source || e.observer != observer || !availableKey(key) || !canPublishAt(group.publication, key)) continue;
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
bool Constructor::needsCommonAcknowledgment(const State& afterForward) const
{
    // A syntactically last body operation is not a last dynamic operation. The
    // backward summary retains original backedges and all branch alternatives.
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
bool Constructor::consumptionBeforeNextPublication(Cut publication, Cut acquisition, Id key, const Ledger* proposed)
{
    // Follow original occurrences, not just the representative command word.
    // The proposed consumption is appended after the existing acquisition word.
    // A later selected reverse publication followed by its acquisition carries
    // that consumption to the publisher. Unsupported relay paths remain unknown.
    const auto& selected = proposed ? *proposed : ledger;
    const auto& forward = frontier.keys()[key];
    ++result.work.splitRearmingQueries;
    const auto at = control.wordSpan[control.canonicalCut[acquisition]];
    const auto pub = control.wordSpan[control.canonicalCut[publication]];
    const auto last = selected.lastPublicationComponent(forward.source, forward.observer, forward.key);
    // SCCs are topologically ordered. One acyclic acquisition component cannot
    // return to itself or an earlier component. Include EVERY occurrence of
    // the proposed publication as well as every active selected publication.
    // This is a sufficient absence proof; ambiguous/later components retain
    // the complete continuation walk. No graph-sized cache per query/key.
    if (at.first != NoAnalysisId && at.first == at.second &&
        !control.components[at.first].cyclic && pub.first != NoAnalysisId &&
        pub.second <= at.first && (last == NoAnalysisId || last <= at.first)) {
        ++result.work.splitRearmingNoNextUse;
        return true;
    }
    using Point = std::pair<Cut, std::set<unsigned>>;
    std::vector<Point> todo;
    for (auto occurrence : control.wordOccurrences[control.canonicalCut[acquisition]])
        if (control.reachable[occurrence])
            for (auto next : control.graph.sites[occurrence].successors) todo.push_back({next, {}});
    // Merge pending reverse generations by intersection. Facts only shrink
    // at joins; do not enumerate a product of path histories. Losing a
    // path-specific proof is conservative and requests the direct helper.
    std::map<Cut, std::set<unsigned>> incoming;
    while (!todo.empty()) {
        auto point = std::move(todo.back()); todo.pop_back();
        auto [cut, published] = std::move(point);
        auto [entry, inserted] = incoming.emplace(cut, published);
        if (!inserted) {
            std::set<unsigned> common;
            std::set_intersection(entry->second.begin(), entry->second.end(), published.begin(), published.end(),
                                  std::inserter(common, common.end()));
            if (common == entry->second) continue;
            entry->second = common;
            published = std::move(common);
        }
        ++result.work.splitRearmingSites;
        bool established = false;
        for (auto id : selected.word(cut)) {
            const auto& command = selected.endpoint(id).command;
            if (command.kind == Command::Publish && command.source == forward.source &&
                command.observer == forward.observer && command.key == forward.key) return false;
            if (command.source != forward.observer || command.observer != forward.source) continue;
            if (command.kind == Command::Publish) published.insert(command.key);
            if (command.kind == Command::Acquire && published.count(command.key)) {
                established = true;
                break;
            }
        }
        if (established) continue;
        // Include a next execution of the NEW publication, not only old uses.
        if (control.canonicalCut[cut] == control.canonicalCut[publication]) return false;
        for (auto next : control.graph.sites[cut].successors) todo.push_back({next, published});
    }
    return true;
}
bool Constructor::inactiveReservation(Cut publication, Cut acquisition, Id key)
{
    // Borrow a fully materialized recurring role; never release its ownership.
    // Lazy closed exchanges and entry protocols can still promise future uses.
    if (!recurringKeys.count(key) || entryProtocolKeys.count(key) ||
        !canPublishAt(publication, key)) return false;
    for (const auto& binding : closedBindings)
        if (binding.second.first == key || binding.second.second == key) return false;
    const auto& identity = frontier.keys()[key];
    auto matches = [&](const Command& command) {
        return (command.kind == Command::Publish || command.kind == Command::Acquire) &&
            command.source == identity.source && command.observer == identity.observer &&
            command.key == identity.key;
    };
    std::set<std::pair<Cut, Command::Kind>> owned, materialized;
    for (const auto& channel : result.channels) {
        if (channel.source != identity.source || channel.observer != identity.observer ||
            channel.key != identity.key) continue;
        for (auto cut : channel.publications)
            owned.insert({control.canonicalCut[cut], Command::Publish});
        for (auto cut : channel.acquisitions)
            owned.insert({control.canonicalCut[cut], Command::Acquire});
    }
    if (owned.empty()) return false;
    for (const auto& endpoint : ledger.records()) {
        if (!matches(endpoint.command)) continue;
        // Removed endpoints could later be restored; fixed or lazy ownership
        // is not covered by this certificate. Previous ordinary borrows remain
        // explicit uses and are checked by the interval walk below.
        if (!ledger.active(endpoint.id) || endpoint.purpose == EndpointPurpose::Fixed) return false;
        if (endpoint.purpose == EndpointPurpose::RecurringCompletion) {
            if (endpoint.request >= result.channels.size()) return false;
            const auto& owner = result.channels[endpoint.request];
            if (owner.source != identity.source || owner.observer != identity.observer ||
                owner.key != identity.key) return false;
            materialized.insert({control.canonicalCut[endpoint.cut], endpoint.command.kind});
        }
    }
    if (owned != materialized) return false;
    return borrowedInterval(publication, acquisition, key);
}
bool Constructor::inactiveClosedReservation(Cut publication, Cut acquisition, Id key)
{
    if (!closedKeys.count(key) || recurringKeys.count(key) || entryProtocolKeys.count(key) ||
        !canPublishAt(publication, key)) return false;
    bool owned = false;
    for (const auto& binding : closedBindings)
        owned |= binding.second.first == key || binding.second.second == key;
    if (!owned) return false;
    const auto& identity = frontier.keys()[key];
    // Keep the role reserved. Suspended helpers might be restored later at an
    // old gap, so this certificate covers active, explicit uses only.
    for (const auto& endpoint : ledger.records()) {
        const auto& command = endpoint.command;
        if ((command.kind == Command::Publish || command.kind == Command::Acquire) &&
            command.source == identity.source && command.observer == identity.observer &&
            command.key == identity.key &&
            (!ledger.active(endpoint.id) || endpoint.purpose == EndpointPurpose::Fixed)) return false;
    }
    return borrowedInterval(publication, acquisition, key);
}
bool Constructor::availableKey(Id key) const
{
    // Occupancy and causal credit are separate from ownership. Only explicit
    // reservation-borrowing paths may bypass this ordinary binding condition.
    return key < frontier.keys().size() && !closedKeys.count(key) &&
        !recurringKeys.count(key) && !suspendedReturnKey(key);
}
Id Constructor::splitReturnKey(Cut publication, Id forward)
{
    const auto& a = frontier.keys()[forward];
    for (Id reverse = 0; reverse < frontier.keys().size(); ++reverse) {
        const auto& b = frontier.keys()[reverse];
        if (b.source != a.observer || b.observer != a.source ||
            !availableKey(reverse) || !canPublishAt(current, reverse)) continue;
        // Check the next use against the exact packet. On re-entry, the
        // forward receipt can carry this reverse consumption before the
        // reverse key is published again. It is a selected endpoint, not a
        // desired memory edge or an assumed future repair.
        Ledger proposed = ledger;
        proposed.append(publication, {Command::Publish, a.source, a.observer, a.key}, EndpointPurpose::Completion);
        proposed.append(current, {Command::Acquire, a.source, a.observer, a.key}, EndpointPurpose::Completion);
        proposed.append(current, {Command::Publish, b.source, b.observer, b.key}, EndpointPurpose::ConsumptionAcknowledgment);
        proposed.append(current, {Command::Acquire, b.source, b.observer, b.key}, EndpointPurpose::ConsumptionAcknowledgment);
        if (consumptionBeforeNextPublication(current, current, reverse, &proposed)) return reverse;
    }
    return NoAnalysisId;
}
bool Constructor::suspendedReturnKey(Id key) const
{
    const auto& identity = frontier.keys()[key];
    const auto found = pendingRearming.find({identity.source, identity.observer});
    if (found == pendingRearming.end()) return false;
    for (const auto& helper : found->second) {
        if (ledger.endpoint(helper.first).command.key == identity.key &&
            (!ledger.active(helper.first) || !ledger.active(helper.second))) return true;
    }
    return false;
}
bool Constructor::prepareClosedReservation(Cut publication, Cut acquisition, Id key)
{
    if (inactiveClosedReservation(publication, acquisition, key)) return true;
    if (!closedKeys.count(key) || recurringKeys.count(key) || entryProtocolKeys.count(key) ||
        !canPublishAt(publication, key) || !suspendedReturnKey(key)) return false;
    bool owned = false;
    for (const auto& binding : closedBindings)
        owned |= binding.second.first == key || binding.second.second == key;
    if (!owned) return false;
    return prepareDormantKey(publication, acquisition, key);
}
bool Constructor::prepareDormantKey(Cut publication, Cut acquisition, Id key)
{
    if (recurringKeys.count(key) || entryProtocolKeys.count(key) ||
        !suspendedReturnKey(key) || !canPublishAt(publication, key)) return false;
    // Closed roles enter only through their explicit ownership certificate.
    if (closedKeys.count(key)) {
        bool owned = false;
        for (const auto& binding : closedBindings)
            owned |= binding.second.first == key || binding.second.second == key;
        if (!owned || !borrowedInterval(publication, acquisition, key)) return false;
    } else if (!clearInterval(key, publication, acquisition)) return false;
    const auto& identity = frontier.keys()[key];
    std::vector<std::pair<Id, Id>> restore;
    std::set<Id> accounted;
    for (const auto& helper : pendingRearming.at({identity.source, identity.observer})) {
        if (ledger.endpoint(helper.first).command.key != identity.key || ledger.active(helper.second)) continue;
        restore.push_back(helper);
        accounted.insert(helper.first); accounted.insert(helper.second);
    }
    for (const auto& endpoint : ledger.records()) {
        const auto& command = endpoint.command;
        if ((command.kind == Command::Publish || command.kind == Command::Acquire) &&
            command.source == identity.source && command.observer == identity.observer &&
            command.key == identity.key && (endpoint.purpose == EndpointPurpose::Fixed ||
            (!ledger.active(endpoint.id) && !accounted.count(endpoint.id)))) return false;
    }
    auto materializeReturns = [&](Ledger& target) {
        for (const auto& helper : restore) {
            target.restoreAfter(helper.first, target.endpoint(helper.second).acknowledges);
            target.restoreAfter(helper.second, helper.first);
        }
    };
    // A dormant helper is a real ownership promise. Validate its original
    // endpoints together with the proposed borrow, not with invented credit.
    // This is one checked restoration, not a deletion/subset search.
    auto proposed = ledger;
    materializeReturns(proposed);
    proposed.append(publication, {Command::Publish, identity.source, identity.observer, identity.key},
                    EndpointPurpose::Completion);
    proposed.append(acquisition, {Command::Acquire, identity.source, identity.observer, identity.key},
                    EndpointPurpose::Completion);
    ++result.work.closedReservationChecks;
    const auto checked = analyze(program, proposed.commands(), {false});
    result.work.closedReservationCheckSites += checked.stats.siteEvaluations;
    if (!checked.complete || !checked.diagnostics.empty() || !checked.protocol.empty() ||
        !checked.phaseResources.empty()) return false;
    materializeReturns(ledger);
    for (const auto& helper : restore) {
        requiredReturns.insert(helper.second);
        ++result.work.acknowledgments;
        ++result.work.rearmingRestored;
        --result.work.rearmingDischarged;
    }
    // edge() installs the checked forward pair before replaying this edit.
    return true;
}
bool Constructor::borrowedInterval(Cut publication, Cut acquisition, Id key)
{
    const auto& identity = frontier.keys()[key];
    auto matches = [&](const Command& command) {
        return (command.kind == Command::Publish || command.kind == Command::Acquire) &&
            command.source == identity.source && command.observer == identity.observer &&
            command.key == identity.key;
    };
    // A two-state walk over original occurrences proves exactly one borrowed
    // acquisition per publication, with no owning/other key use in between.
    // Shared words and backedges are visited in both states, not flattened.
    std::vector<std::pair<Cut, bool>> todo{{control.graph.entry, false}};
    std::set<std::pair<Cut, bool>> seen;
    const auto sourceWord = control.canonicalCut[publication];
    const auto targetWord = control.canonicalCut[acquisition];
    while (!todo.empty()) {
        auto [cut, live] = todo.back(); todo.pop_back();
        if (!seen.insert({cut, live}).second) continue;
        ++result.work.splitRearmingSites;
        for (auto id : ledger.word(cut))
            if (live && matches(ledger.endpoint(id).command)) return false;
        if (control.canonicalCut[cut] == sourceWord) {
            if (live) return false;
            live = true;
        }
        if (control.canonicalCut[cut] == targetWord) {
            if (!live) return false;
            live = false;
        }
        if (control.graph.sites[cut].successors.empty() && live) return false;
        for (auto next : control.graph.sites[cut].successors) todo.push_back({next, live});
    }
    // The owner may run again after this interval. Actual selected reverse
    // receipts must carry the new consumption before ANY next publication.
    return consumptionBeforeNextPublication(publication, acquisition, key);
}
bool Constructor::edge(Pipe source, Pipe observer, Cut& publication, bool closed, SelectedDecision& decision, Id certifiedKey)
{
    const bool recurringClosed = closed && control.components[activeComponent].cyclic;
    const auto binding = closedBindings.find({source, observer});
    const bool retained = recurringClosed && binding != closedBindings.end();
    Id key = retained ? binding->second.first : NoAnalysisId;
    bool restoredTransfer = false;
    if (certifiedKey != NoAnalysisId) {
        if (closed || certifiedKey >= frontier.keys().size() || !availableKey(certifiedKey) ||
            !canPublishAt(publication, certifiedKey) ||
            !clearInterval(certifiedKey, publication, current))
            return fail(SelectedFailure::SelectedUpdate, "binding certificate no longer applies", publication);
        key = certifiedKey;
    } else if (retained) {
        if (!canPublishAt(publication, key) &&
            !restoreRearming(key, publication)) {
            // Restoring a return can fail replay at another endpoint. Preserve
            // that exact occurrence and cause instead of blaming this binding.
            if (result.failure != SelectedFailure::None) return false;
            return fail(SelectedFailure::EventResource,
                "recurring forward role lacks its consumption path", publication);
        }
    } else {
        for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
            const auto& identity = frontier.keys()[candidate];
            if (identity.source != source || identity.observer != observer || !availableKey(candidate)) {
                continue;
            }
            ++result.work.keyQueries;
            if (canPublishAt(publication, candidate) &&
                clearInterval(candidate, publication, current)) {
                key = candidate;
                break;
            }
        }
        if (key == NoAnalysisId && options.firstWriteConsumers && !closed) {
            for (auto candidate : recurringKeys) {
                const auto& identity = frontier.keys()[candidate];
                if (identity.source != source || identity.observer != observer) continue;
                ++result.work.keyQueries;
                if (inactiveReservation(publication, current, candidate)) {
                    key = candidate;
                    break;
                }
            }
        }
        if (key == NoAnalysisId && !closed) {
            for (auto candidate : closedKeys) {
                const auto& identity = frontier.keys()[candidate];
                if (identity.source != source || identity.observer != observer) continue;
                ++result.work.keyQueries;
                if (prepareClosedReservation(publication, current, candidate)) {
                    key = candidate;
                    ++result.work.closedReservationBorrows;
                    break;
                }
            }
        }
        if (key == NoAnalysisId) {
            // Dormant ordinary helpers retain ownership just like closed
            // roles. Exhaustion may use their keys only by checking restoration
            // and the new transfer together, never by ignoring the promise.
            for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
                const auto& identity = frontier.keys()[candidate];
                if (identity.source != source || identity.observer != observer ||
                    closedKeys.count(candidate) || !suspendedReturnKey(candidate)) continue;
                ++result.work.keyQueries;
                if (prepareDormantKey(publication, current, candidate)) {
                    key = candidate;
                    restoredTransfer = true;
                    break;
                }
            }
        }
        if (key == NoAnalysisId && !acknowledgment(source, observer, publication, key, decision)) {
            return false;
        }
    }
    const auto number = frontier.keys()[key].key;
    Id splitReverse = NoAnalysisId;
    const auto& publicationOccurrences = control.wordOccurrences[control.canonicalCut[publication]];
    const auto& acquisitionOccurrences = control.wordOccurrences[control.canonicalCut[current]];
    const bool repeated = std::any_of(acquisitionOccurrences.begin(), acquisitionOccurrences.end(), [&](Cut cut) {
        return control.reachable[cut] && control.components[control.component[cut]].cyclic;
    });
    if (!restoredTransfer && !closed && repeated &&
        (publicationOccurrences.size() > 1 || acquisitionOccurrences.size() > 1 ||
            (options.firstWriteConsumers && control.firstWriteWords.count(control.canonicalCut[publication]))) &&
        !consumptionBeforeNextPublication(publication, current, key)) {
        // Certify a direct return at the actual new consumption. Its source
        // position never moves the forward publication behind unrelated work.
        // Existing source-time knowledge must already rearm the reverse key;
        // no target-time emptiness or hypothetical future receipt is borrowed.
        splitReverse = splitReturnKey(publication, key);
        if (splitReverse == NoAnalysisId)
            return fail(SelectedFailure::EventResource, "split recurring receipt has no certified return key", current);
    }
    const auto request = result.decisions.size();
    decision.endpoints.push_back(ledger.append(publication,
        {Command::Publish, source, observer, number}, EndpointPurpose::Completion, request));
    const auto wait = ledger.append(current,
        {Command::Acquire, source, observer, number}, EndpointPurpose::Completion, request);
    decision.endpoints.push_back(wait);
    if (!closed || restoredTransfer) {
        // Restoration was checked with this exact forward pair on the whole
        // original graph, including matching and rearming on repeated visits.
        // Keep that certified packet; do not append an unchecked private return
        // merely because the fallback source and deadline share a cut.
        if (splitReverse != NoAnalysisId) {
            const auto reverseNumber = frontier.keys()[splitReverse].key;
            decision.endpoints.push_back(ledger.append(current,
                {Command::Publish, observer, source, reverseNumber}, EndpointPurpose::ConsumptionAcknowledgment, request, wait));
            decision.endpoints.push_back(ledger.append(current,
                {Command::Acquire, observer, source, reverseNumber}, EndpointPurpose::ConsumptionAcknowledgment, request, wait));
            rememberReturn(decision.endpoints[decision.endpoints.size() - 2], decision.endpoints.back());
            ++result.work.acknowledgments;
        }
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
        (deferCommonRearming(key, wait, decision) || !needsCommonAcknowledgment(afterForward))) {
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
    return restoreReturns(key) && update() && canPublishAt(publication, key);
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
    observeDeferredRearming(decision);
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

// A qualified FIFO cohort supplies only the physical send/pop correspondence.
// Select a relay prefix separately from the final receipt deadline. This is an
// optional staged construction, not a rewrite of a completed plan.
std::optional<bool> Constructor::splitRelay(const Group& group, Pipe observer, RequirementStage stage)
{
    if (!program.staticFifoSlots || group.common || group.publication == current ||
        !control.straight(group.publication, current) || route(group.source, observer).size() != 3)
        return std::nullopt;
    const auto& slots = *program.staticFifoSlots;
    if (group.requirements.empty() ||
        std::any_of(group.requirements.begin(), group.requirements.end(), [&](const auto& r) {
            return !r.sourceWrite || std::find(slots.cells.begin(), slots.cells.end(), r.cell) == slots.cells.end();
        }))
        return std::nullopt;
    auto unique = [&](Cut cut) { return control.wordOccurrences[control.canonicalCut[cut]].size() == 1; };
    if (!unique(group.publication) || !unique(current))
        return std::nullopt;
    // Do not gate intervening sends of another bank through an earlier bank's
    // receipt. The immutable shared-slot view identifies this lower bound; it
    // does not enlarge the publication or grant the other bank's completion.
    Cut lower = group.publication;
    for (Cut at = 0; at < control.graph.sites.size(); ++at) {
        ++result.work.relayPreparationSites;
        const auto op = control.graph.operations[at];
        if (op == NoAnalysisId || program.operations[op].pipe != group.source ||
            std::find(slots.writes.begin(), slots.writes.end(), op) == slots.writes.end() ||
            !control.straight(group.publication, at) || !control.straight(at, current))
            continue;
        const auto after = control.after(at);
        if (after != NoAnalysisId && control.straight(after, current) &&
            control.position[after] > control.position[lower])
            lower = after;
    }
    auto eligible = [&](Pipe a, Pipe b) {
        for (Id key = 0; key < frontier.keys().size(); ++key)
            if (frontier.keys()[key].source == a && frontier.keys()[key].observer == b && availableKey(key))
                return true;
        return false;
    };
    auto keyFor = [&](Pipe a, Pipe b, Cut pub, Cut wait, const State& state, bool atStart) {
        for (Id key = 0; key < frontier.keys().size(); ++key) {
            const auto& e = frontier.keys()[key];
            if (e.source != a || e.observer != b || !availableKey(key))
                continue;
            ++result.work.keyQueries;
            if (!canPublish(state, key) || !clearInterval(key, pub, wait))
                continue;
            if (atStart && std::any_of(ledger.word(pub).begin(), ledger.word(pub).end(), [&](Id id) {
                    const auto& c = ledger.endpoint(id).command;
                    return (c.kind == Command::Publish || c.kind == Command::Acquire) && c.source == a &&
                           c.observer == b && c.key == e.key;
                }))
                continue;
            return key;
        }
        return NoAnalysisId;
    };
    // Positive read-only certificates at the exact publication gaps. Unknown
    // binding leaves this candidate out of ranking; it does not prove that no
    // helper-augmented realization exists. Both legs still need current credit.
    Id first = NoAnalysisId, second = NoAnalysisId;
    Pipe middle = Pipe::Count;
    Cut relay = current;
    std::set<Id> incidental;
    // Distinguish payload deadlines from selected endpoint identities. A smaller
    // forwarded prefix must not win by imposing new prerequisites elsewhere.
    using Gate = std::tuple<bool, Id, Id>;
    std::set<Gate> gated;
    const auto* sourceFacts = cache.cuts[group.publication].before.causal.facts();
    if (!sourceFacts)
        return std::nullopt;
    const auto* receiverFacts = cache.cuts[current].before.causal.facts();
    if (!receiverFacts) return std::nullopt;
    std::set<Id> needed;
    for (const auto& r : residual()) needed.insert(accessClass(r));
    std::vector<Id> sourceHistory;
    for (const auto& [access, reach] : sourceFacts->history.present())
        if (frontierContains(reach, PipeCount + unsigned(group.source))) sourceHistory.push_back(access);
    auto subset = [](const auto& a, const auto& b) {
        return std::includes(b.begin(), b.end(), a.begin(), a.end());
    };
    // Share the immutable straight corridor between the bounded candidate
    // queries. No per-candidate whole-program propagation or order closure.
    std::vector<Cut> corridor;
    for (Cut at = 0; at < control.graph.sites.size(); ++at) {
        ++result.work.relayPreparationSites;
        if (control.straight(lower, at) && control.straight(at, current)) corridor.push_back(at);
    }
    std::sort(corridor.begin(), corridor.end(), [&](Cut a, Cut b) {
        return control.position[a] < control.position[b];
    });
    for (unsigned candidate = 0; candidate < PipeCount; ++candidate) {
        if (Pipe(candidate) == group.source || Pipe(candidate) == observer ||
            !eligible(group.source, Pipe(candidate)) || !eligible(Pipe(candidate), observer))
            continue;
        Cut gap = current;
        for (auto at : corridor) {
            ++result.work.relayPreparationSites;
            if (at == current) break;
            auto op = control.graph.operations[at];
            bool issues = op != NoAnalysisId && program.operations[op].pipe == Pipe(candidate);
            for (auto id : ledger.word(at)) {
                const auto& c = ledger.endpoint(id).command;
                issues |= c.kind == Command::BarrierAll ||
                          (c.kind == Command::Acquire ? c.observer : c.source) == Pipe(candidate);
            }
            if (issues && unique(at)) {
                gap = at;
                break;
            }
        }
        if (gap == group.publication || !legalCommandCut(program, gap))
            continue;
        // A later receipt must not be prepended ahead of an already selected
        // forwarding publication to this same receiver when that broadens the
        // earlier receipt. Retain the final deadline for this middle engine.
        // Other outward interfaces are compared below, not presumed harmless.
        const auto gapBegin = std::lower_bound(corridor.begin(), corridor.end(), gap, [&](Cut a, Cut b) {
            return control.position[a] < control.position[b];
        });
        bool widensReceipt = false;
        if (gap != current) for (auto atIt = gapBegin; atIt != corridor.end() && !widensReceipt; ++atIt) {
            const auto at = *atIt;
            ++result.work.relayPreparationSites;
            for (auto id : ledger.word(at)) {
                const auto& command = ledger.endpoint(id).command;
                if (command.kind != Command::Publish || command.source != Pipe(candidate) ||
                    command.observer != observer) continue;
                const auto found = cache.afterEndpoint.find(id);
                const auto* published = found == cache.afterEndpoint.end() ? nullptr : found->second.causal.facts();
                for (auto access : sourceHistory) {
                    const auto* reach = published ? published->history.find(access) : nullptr;
                    if (!freshBetween(group.publication, at, access) || !reach ||
                        !frontierContains(*reach, PipeCount + candidate)) {
                        widensReceipt = true;
                        break;
                    }
                }
            }
        }
        if (widensReceipt) {
            if (!legalCommandCut(program, current)) continue;
            gap = current;
        }
        const auto& state = gap == current ? cache.cuts[gap].before : cache.cuts[gap].incoming;
        const auto* facts = state.causal.facts();
        if (!facts)
            continue;
        const auto candidateFirst = keyFor(
            group.source, Pipe(candidate), group.publication, gap, cache.cuts[group.publication].before, false);
        if (candidateFirst == NoAnalysisId) continue;
        const auto candidateSecond = keyFor(Pipe(candidate), observer, gap, current, state, gap != current);
        if (candidateSecond == NoAnalysisId) continue;
        std::set<Id> extra;
        for (const auto& [access, reach] : facts->history.present()) {
            if (!frontierContains(reach, PipeCount + candidate))
                continue;
            const auto* received = receiverFacts->history.find(access);
            // Only the same occurrence can be treated as already acquired or
            // required at this deadline. This scoring view grants no receipt.
            if (freshBetween(gap, current, access) &&
                (needed.count(access) || (received && frontierContains(*received, unsigned(observer)))))
                continue;
            const auto* original = sourceFacts->history.find(access);
            const bool fresh = !control.lookahead.hasIssueBetween(
                control.frame[group.publication], access, control.position[group.publication], control.position[gap]);
            if (!fresh || !original || !frontierContains(*original, PipeCount + unsigned(group.source)))
                extra.insert(access);
        }
        std::set<Gate> newGates;
        auto inspectGate = [&](bool endpoint, Id identity, Cut at, const State* before, unsigned port) {
            const auto* known = before ? before->causal.facts() : nullptr;
            for (auto access : sourceHistory) {
                const auto* reach = known ? known->history.find(access) : nullptr;
                if (!freshBetween(group.publication, at, access) || !reach || !frontierContains(*reach, port))
                    newGates.emplace(endpoint, identity, access);
            }
        };
        if (gap != current) for (auto atIt = gapBegin; atIt != corridor.end(); ++atIt) {
            const auto at = *atIt;
            ++result.work.relayPreparationSites;
            // Include the final cut's existing word: the late alternative would
            // acquire after it. Actual endpoint states preserve intra-word order.
            for (auto id : ledger.word(at)) {
                const auto& command = ledger.endpoint(id).command;
                const auto pipe = command.kind == Command::Acquire ? command.observer : command.source;
                if (command.kind != Command::BarrierAll && pipe != Pipe(candidate)) continue;
                const auto found = cache.afterEndpoint.find(id);
                inspectGate(true, id, at, found == cache.afterEndpoint.end() ? nullptr : &found->second,
                    command.kind == Command::Publish ? PipeCount + candidate : candidate);
            }
            const auto op = control.graph.operations[at];
            if (at != current && op != NoAnalysisId && program.operations[op].pipe == Pipe(candidate))
                inspectGate(false, at, at, &cache.cuts[at].before, candidate);
        }
        // Compare both sets, never a weighted count. Keep the first candidate
        // when the choices trade forwarded history for newly gated work. Equal
        // views retain the latest-gap tie. This is a bounded placement heuristic,
        // not a certificate for arbitrary future edits or complete event interfaces.
        const bool noWorse = subset(extra, incidental) && subset(newGates, gated);
        if (middle == Pipe::Count ||
            (noWorse && (extra != incidental || newGates != gated ||
                        control.position[gap] > control.position[relay]))) {
            middle = Pipe(candidate);
            relay = gap;
            first = candidateFirst;
            second = candidateSecond;
            incidental = std::move(extra);
            gated = std::move(newGates);
        }
    }
    if (middle == Pipe::Count)
        return std::nullopt;
    const auto request = result.decisions.size();
    auto materialize = [&](Ledger& target) {
        std::vector<Id> ids;
        ids.push_back(target.append(
            group.publication, {Command::Publish, group.source, middle, frontier.keys()[first].key},
            EndpointPurpose::Completion, request));
        const Command acquire{Command::Acquire, group.source, middle, frontier.keys()[first].key};
        ids.push_back(
            relay == current ? target.append(relay, acquire, EndpointPurpose::Completion, request) :
                               target.prepend(relay, acquire, EndpointPurpose::Completion, request));
        ids.push_back(target.after(
            ids.back(), {Command::Publish, middle, observer, frontier.keys()[second].key}, EndpointPurpose::Completion,
            request, NoAnalysisId));
        ids.push_back(target.append(
            current, {Command::Acquire, middle, observer, frontier.keys()[second].key}, EndpointPurpose::Completion,
            request));
        return ids;
    };
    auto proposed = ledger;
    materialize(proposed);
    ++result.work.relayTrials;
    auto check = analyze(program, proposed.commands(), {false});
    result.work.relayTrialSites += check.stats.siteEvaluations;
    if (!check.complete || !check.diagnostics.empty() || !check.protocol.empty() || !check.phaseResources.empty())
        return std::nullopt;
    SelectedDecision decision;
    decision.consumer = current;
    decision.publication = group.publication;
    decision.source = group.source;
    decision.observer = observer;
    decision.stage = stage;
    decision.required = group.requirements;
    decision.lifecycles = requirements.demandsAt(current, group.requirements);
    decision.endpoints = materialize(ledger);
    ++result.work.splitRelays;
    if (!update() || !settleRearming(decision))
        return false;
    result.decisions.push_back(std::move(decision));
    return true;
}
bool Constructor::bind(Group& group, RequirementStage stage)
{
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    if (group.finalReadSource) {
        const auto key = group.forwardKey;
        if (group.version != ledger.version() || key >= frontier.keys().size() ||
            !availableKey(key) ||
            !finalReadGap(group.publication, group.source, group.requirements, key) ||
            !consumptionBeforeNextPublication(group.publication, current, key))
            return fail(SelectedFailure::SelectedUpdate, "stale final-read source certificate", current);
        SelectedDecision decision;
        decision.consumer = current; decision.publication = group.publication;
        decision.publicationAtWordStart = true; decision.stage = stage;
        decision.source = group.source; decision.observer = observer;
        decision.required = group.requirements;
        decision.lifecycles = requirements.demandsAt(current, group.requirements);
        const auto number = frontier.keys()[key].key;
        decision.endpoints.push_back(ledger.append(group.publication,
            {Command::Publish, group.source, observer, number}, EndpointPurpose::Completion, result.decisions.size()));
        ledger.protectPublicationPrefix(decision.endpoints.back());
        decision.endpoints.push_back(ledger.append(current,
            {Command::Acquire, group.source, observer, number}, EndpointPurpose::Completion, result.decisions.size()));
        ++result.work.finalReadPublications;
        // Keep all existing acknowledgments; this selection changes only the
        // completion source. Actual replay supplies the receipt's credit.
        if (!update()) return false;
        result.decisions.push_back(std::move(decision));
        return true;
    }
    if (group.atWordStart) {
        const auto key = reusableAtStart(group.publication, group.source, observer);
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
        decision.lifecycles = requirements.demandsAt(current, group.requirements);
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
        decision.lifecycles = requirements.demandsAt(current, group.requirements);
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
            if (group.choiceAcquisition) ++result.work.choiceTransfers;
            else ++result.work.loopEntryTransfers;
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
    if (auto split = splitRelay(group, observer, stage)) return *split;
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
    decision.lifecycles = requirements.demandsAt(current, group.requirements);
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
    if (!preservePublicationPrefixes(decision)) {
        return false;
    }
    if (!settleRearming(decision)) {
        return false;
    }
    result.decisions.push_back(std::move(decision));
    return true;
}
} // namespace mlir::pto::oahs::selected
