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
bool Constructor::unownedKey(Id key) const
{
    return !closedKeys.count(key) && !recurringKeys.count(key) &&
           !ledger.hasDormantUses(frontier.keys()[key]);
}
std::optional<OwnedPacket> Constructor::prepareOwnedPacket(const OrderedPacket& proposed)
{
    ++result.work.ownershipQueries;
    std::set<std::tuple<Pipe, Pipe, unsigned>> touched;
    std::set<std::pair<Id, Id>> owners;
    for (const auto& item : proposed) {
        const auto& c = item.command;
        if (c.kind != Command::Publish && c.kind != Command::Acquire) { continue; }
        const EventIdentity identity{c.source, c.observer, c.key};
        const bool newOwnerQuery = ledger.hasDormantUses(identity) &&
            touched.emplace(c.source, c.observer, c.key).second;
        if (!newOwnerQuery) { continue; }
        for (auto id : ledger.eventUses(identity)) {
            if (ledger.active(id)) { continue; }
            const auto owner = helperOwners.find(id);
            if (owner == helperOwners.end()) { return {}; }
            const auto pair = owner->second;
            if (id != pair.first && id != pair.second) { return {}; }
            const bool dormantPair = !ledger.active(pair.first) && !ledger.active(pair.second);
            if (!dormantPair) { return {}; }
            const auto& pub = ledger.endpoint(pair.first);
            const auto& wait = ledger.endpoint(pair.second);
            const auto anchor = wait.acknowledges;
            const bool matching = pub.purpose == EndpointPurpose::ConsumptionAcknowledgment &&
                wait.purpose == pub.purpose && pub.acknowledges == anchor &&
                pub.command.kind == Command::Publish && wait.command.kind == Command::Acquire &&
                pub.command.source == c.source && pub.command.observer == c.observer && pub.command.key == c.key &&
                wait.command.source == c.source && wait.command.observer == c.observer && wait.command.key == c.key &&
                anchor < ledger.records().size() && ledger.active(anchor);
            if (!matching) { return {}; }
            const auto& forward = ledger.endpoint(anchor).command;
            const bool actualReceipt = forward.kind == Command::Acquire &&
                forward.source == c.observer && forward.observer == c.source;
            if (!actualReceipt) { return {}; }
            owners.insert(pair);
        }
    }
    std::vector<Id> anchors;
    for (const auto& owner : owners) { anchors.push_back(ledger.endpoint(owner.second).acknowledges); }
    const auto gaps = ledger.gapsAfter(anchors);
    OwnedPacket out;
    OrderedPacket packet;
    for (const auto& owner : owners) {
        const auto gap = gaps.find(ledger.endpoint(owner.second).acknowledges);
        if (gap == gaps.end()) { return {}; }
        const auto pub = ledger.restoration(owner.first, gap->second);
        const auto wait = ledger.restoration(owner.second, gap->second);
        if (!pub || !wait) { return {}; }
        packet.push_back(*pub);
        packet.push_back(*wait);
        out.restoredWaits.push_back(owner.second);
    }
    out.restoredEndpoints = packet.size();
    for (Id index = 0; index < proposed.size(); ++index) {
        auto item = proposed[index];
        if (item.acknowledgesPacket != NoAnalysisId) {
            if (item.acknowledgesPacket >= index) { return {}; }
            item.acknowledgesPacket += out.restoredEndpoints;
        }
        packet.push_back(std::move(item));
    }
    out.prepared = ledger.preparePacket(packet);
    if (!out.prepared.valid()) { return {}; }
    return out;
}
std::optional<OwnedPacket> Constructor::qualifyOwnedPacket(const OrderedPacket& proposed, bool alwaysCheck)
{
    auto out = prepareOwnedPacket(proposed);
    if (!out) { return {}; }
    if (alwaysCheck || !out->restoredWaits.empty()) {
        const auto commands = ledger.withPacket(out->prepared);
        if (!commands) { return {}; }
        const auto checked = analyze(program, *commands, {false});
        ++result.work.ownershipChecks;
        result.work.ownershipCheckSites += checked.stats.siteEvaluations;
        if (!acceptOwnedPacket(*out, checked)) { return {}; }
    } else {
        out->qualified = true; // Existing source-time and neighboring-use certificates.
    }
    return out;
}
bool Constructor::acceptOwnedPacket(OwnedPacket& packet, const AnalysisResult& checked) const
{
    const bool protocol = checked.complete && checked.protocol.empty() &&
        checked.diagnostics.empty() && checked.phaseResources.empty();
    if (!protocol) { return false; }
    for (const auto& residual : checked.residuals) {
        const auto cut = residual.consumerCut;
        if (cut >= finalized.size()) { return false; }
        if (finalized[cut]) { return false; }
        const bool supported = cut < producerSupportConsumers.size() && producerSupportConsumers[cut];
        if (!supported) { continue; }
        const auto operation = control.graph.operations[cut];
        if (operation == NoAnalysisId || residual.demand.producer >= program.operations.size()) { return false; }
        const auto pipe = unsigned(program.operations[operation].pipe);
        const auto source = unsigned(program.operations[residual.demand.producer].pipe);
        const auto base = (Id(residual.producerAccess.cell) * PipeCount + source) * 2;
        // Preserve both RMW roles; this filter may conservatively retain more
        // than the native access exception. Selected causal replay is final.
        const bool protectedRead = residual.producerAccess.read && producerSupportClasses[pipe].count(base);
        const bool protectedWrite = residual.producerAccess.write && producerSupportClasses[pipe].count(base + 1);
        if (protectedRead || protectedWrite) { return false; }
    }
    packet.qualified = true;
    return true;
}
bool Constructor::commitOwnedPacket(const OwnedPacket& packet, SelectedDecision& decision)
{
    if (!packet.qualified) {
        return fail(SelectedFailure::SelectedUpdate, "owned packet was not qualified", current);
    }
    const auto ids = ledger.appendPacket(packet.prepared);
    const bool committed = ids.size() == packet.prepared.size();
    if (!committed) {
        return fail(SelectedFailure::SelectedUpdate, "owned packet changed after qualification", current);
    }
    decision.endpoints.insert(decision.endpoints.end(), ids.begin() + packet.restoredEndpoints, ids.end());
    if (!packet.restoredWaits.empty()) { ++result.work.ownershipBindings; }
    for (auto wait : packet.restoredWaits) {
        requiredReturns.insert(wait);
        ++result.work.acknowledgments;
        ++result.work.rearmingRestored;
        --result.work.rearmingDischarged;
    }
    return true;
}
bool Constructor::commitPacket(const OrderedPacket& packet, SelectedDecision& decision)
{
    const auto qualified = qualifyOwnedPacket(packet);
    if (!qualified) {
        return fail(SelectedFailure::SelectedUpdate, "packet has no complete ownership certificate", current);
    }
    return commitOwnedPacket(*qualified, decision);
}
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
        if (identity.source != source || identity.observer != observer || !unownedKey(key)) {
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
        return true; // appended after all existing commands in this word
    }
    const auto& correspondence = control.correspondence(source, target);
    if (!correspondence.proved()) {
        return false;
    }
    const auto& identity = frontier.keys()[key];
    for (const auto& pair : correspondence.pairs) {
        if (!control.straight(pair.first, pair.second)) {
            return false;
        }
        for (auto id : ledger.eventUses(identity)) {
            if (!ledger.active(id)) {
                continue;
            }
            for (auto occurrence : control.wordOccurrences[ledger.endpoint(id).cut]) {
                if (occurrence != pair.first && control.straight(pair.first, occurrence) &&
                    control.straight(occurrence, pair.second)) {
                    return false;
                }
            }
        }
    }
    return true;
}
SourceGapQualification Constructor::sourceGap(
    const WordGap& gap, Id key, const std::vector<FrontierRequirement>& required)
{
    ++result.work.sourceGapQueries;
    SourceGapQualification out;
    out.gap = gap;
    out.version = ledger.version();
    out.reason = "source gap lacks a current matched occurrence certificate";
    const bool invalidQuery = !cache.success || cache.version != ledger.version() ||
        key >= frontier.keys().size() || required.empty() || gap.cut >= control.graph.sites.size();
    if (invalidQuery) {
        return out;
    }
    const auto& ids = ledger.word(gap.cut);
    const auto right = gap.right == NoAnalysisId ? ids.end() : std::find(ids.begin(), ids.end(), gap.right);
    const auto offset = Id(right - ids.begin());
    const bool changedGap = (gap.right != NoAnalysisId && right == ids.end()) ||
        gap.left != (offset == 0 ? NoAnalysisId : ids[offset - 1]);
    if (changedGap) {
        out.reason = "source gap neighbors changed";
        return out;
    }
    const auto& identity = frontier.keys()[key];
    // The initial early-placement client uses a virgin physical key. This is a
    // sufficient neighboring-generation certificate, including dormant uses;
    // source-time publishability alone would not protect a later republication.
    if (!ledger.eventUses(identity).empty()) {
        out.reason = "early gap lacks neighboring-generation support for an existing key";
        return out;
    }
    const auto& correspondence = control.correspondence(gap.cut, current);
    if (!correspondence.proved()) {
        return out;
    }
    bool reached = false;
    for (const auto& pair : correspondence.pairs) {
        if (!control.reachable[pair.first]) {
            continue;
        }
        if (control.components[control.component[pair.first]].cyclic ||
            control.components[control.component[pair.second]].cyclic ||
            !control.straight(pair.first, pair.second)) {
            return out;
        }
        auto state = cache.cuts[pair.first].incoming.causal;
        if (!state.reachable()) {
            return out;
        }
        for (Id index = 0; index < offset; ++index) {
            ++result.work.sourceGapCommands;
            const auto step = frontier.command(state, ledger.endpoint(ids[index]).command, {pair.first, index});
            if (!step.applied) {
                out.reason = "source word prefix is not executable";
                return out;
            }
            state = step.state;
        }
        State atGap;
        atGap.causal = state;
        if (!canPublish(atGap, key)) {
            out.reason = "source gap has no established publication credit";
            return out;
        }
        for (const auto& requirement : required) {
            const auto access = accessClass(requirement);
            const auto* history = state.facts()->history.find(access);
            const bool covered = history && frontierContains(*history, PipeCount + unsigned(identity.source)) &&
                freshBetween(pair.first, pair.second, access);
            if (!covered) {
                out.reason = "source gap does not cover the required physical occurrence";
                return out;
            }
        }
        reached = true;
    }
    if (reached) {
        out.outcome = ProofOutcome::Proved;
        out.reason.clear();
    }
    return out;
}
std::optional<WordGap> Constructor::earlyPublicationGap(Cut cut, Id key, const SelectedDecision& decision)
{
    const auto& identity = frontier.keys()[key];
    const auto& ids = ledger.word(cut);
    auto offset = ids.size();
    bool incomingWait = false;
    while (offset != 0) {
        const auto& command = ledger.endpoint(ids[offset - 1]).command;
        const bool event = command.kind == Command::Publish || command.kind == Command::Acquire;
        const auto pipe = command.kind == Command::Acquire ? command.observer : command.source;
        if (command.kind == Command::BarrierAll ||
            (event && command.source == identity.source && command.observer == identity.observer &&
             command.key == identity.key) ||
            (pipe == identity.source && command.kind != Command::Acquire)) {
            break;
        }
        incomingWait |= command.kind == Command::Acquire && command.observer == identity.source;
        --offset;
    }
    if (!incomingWait) {
        return {};
    }
    const WordGap gap{control.canonicalCut[cut], offset == 0 ? NoAnalysisId : ids[offset - 1], ids[offset]};
    auto required = decision.required;
    required.insert(required.end(), decision.supporting.begin(), decision.supporting.end());
    const auto certificate = sourceGap(gap, key, required);
    const bool currentProof = certificate.proved() && certificate.version == ledger.version();
    if (!currentProof) {
        return {};
    }
    return certificate.gap;
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
bool Constructor::acknowledgment(Pipe source, Pipe observer, Cut& publication, Id& key,
                                 SelectedDecision& decision, bool& completed)
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
            const bool dormant = ledger.hasDormantUses(identity) || ledger.hasDormantUses(reverseIdentity);
            if (dormant) {
                const auto gap = ledger.gapAfter(wait);
                if (!gap) { continue; }
                const auto request = result.decisions.size();
                const OrderedPacket packet{
                    {gap->cut, {Command::Publish, observer, source, reverseIdentity.key},
                        EndpointPurpose::ConsumptionAcknowledgment, request, wait, gap},
                    {newCut, {Command::Acquire, observer, source, reverseIdentity.key},
                        EndpointPurpose::ConsumptionAcknowledgment, request, wait},
                    {newCut, {Command::Publish, source, observer, identity.key}, EndpointPurpose::Completion, request},
                    {current, {Command::Acquire, source, observer, identity.key},
                        EndpointPurpose::Completion, request}};
                const auto qualified = qualifyOwnedPacket(packet);
                if (!qualified) { continue; }
                const auto inputVersion = ledger.version();
                if (!commitOwnedPacket(*qualified, decision)) { return false; }
                decision.repairedAcquisition = wait;
                decision.repairedForwardKey = identity.key;
                decision.repairReverseKey = reverseIdentity.key;
                decision.repairInputVersion = inputVersion;
                decision.enlargedPrefix |= newCut != publication;
                publication = newCut;
                key = candidate;
                completed = true;
                ++result.work.acknowledgments;
                if (!update()) { return false; }
                decision.repairOutputVersion = ledger.version();
                return true;
            }
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
        if (auto packet = joinedAcknowledgment(source, observer, publication, decision)) {
            completed = true;
            return *packet;
        }
        std::string reason = "no reusable key or nonrecursive consumption acknowledgment: source=" +
            std::to_string(unsigned(source)) + " observer=" + std::to_string(unsigned(observer)) +
            " deadline=" + std::to_string(current);
        for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
            const auto &identity = frontier.keys()[candidate];
            if (identity.source == source && identity.observer == observer) {
                const auto &state = cache.cuts[publication].before;
                reason += " key=" + std::to_string(identity.key) +
                    "/closed=" + std::to_string(closedKeys.count(candidate)) +
                    "/occupancy=" + std::to_string(state.causal.facts()->events[candidate].occupancy) +
                    "/receipts=" + std::to_string(state.consumptions[candidate].size());
            }
        }
        return fail(SelectedFailure::EventResource, reason, publication);
    }
    const auto& identity = frontier.keys()[reverse];
    const auto request = result.decisions.size();
    decision.repairedAcquisition = oldWait;
    decision.repairedForwardKey = frontier.keys()[key].key;
    decision.repairReverseKey = identity.key;
    decision.repairInputVersion = ledger.version();
    const auto gap = ledger.gapAfter(oldWait);
    if (!gap) {
        return fail(SelectedFailure::SelectedUpdate, "consumption acknowledgment lost its source gap", moved);
    }
    const OrderedPacket packet{
        {gap->cut, {Command::Publish, observer, source, identity.key},
         EndpointPurpose::ConsumptionAcknowledgment, request, oldWait, gap},
        {moved, {Command::Acquire, observer, source, identity.key},
         EndpointPurpose::ConsumptionAcknowledgment, request, oldWait}};
    if (!commitPacket(packet, decision)) {
        return false;
    }
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
std::optional<bool> Constructor::joinedAcknowledgment(
    Pipe source, Pipe observer, Cut publication, SelectedDecision& decision)
{
    // Distinct original paths may consume one key at different WAITs. IDs are
    // not the certificate: require actual empty occupancy, execute the return,
    // and validate the complete reverse/forward exchange on the original graph.
    if (!cache.success || !control.straight(publication, current)) {
        return {};
    }
    const auto& state = cache.cuts[publication].before;
    if (!state.causal.reachable()) {
        return {};
    }
    for (Id forward = 0; forward < frontier.keys().size(); ++forward) {
        const auto& a = frontier.keys()[forward];
        if (a.source != source || a.observer != observer || closedKeys.count(forward) ||
            state.causal.facts()->events[forward].occupancy != 1 ||
            !clearInterval(forward, publication, current)) {
            continue;
        }
        for (Id reverse = 0; reverse < frontier.keys().size(); ++reverse) {
            const auto& b = frontier.keys()[reverse];
            if (b.source != observer || b.observer != source || closedKeys.count(reverse)) {
                continue;
            }
            const Command publish{Command::Publish, observer, source, b.key};
            const Command acquire{Command::Acquire, observer, source, b.key};
            const bool dormant = ledger.hasDormantUses(a) || ledger.hasDormantUses(b);
            bool supported = true;
            if (!dormant) {
                for (auto at : control.wordOccurrences[control.canonicalCut[publication]]) {
                    if (!control.reachable[at]) {
                        continue;
                    }
                    const auto& before = cache.cuts[at].before;
                    const auto offset = ledger.word(at).size();
                    auto sent = frontier.command(before.causal, publish, {at, offset});
                    if (!sent.applied) {
                        supported = false;
                        break;
                    }
                    auto received = frontier.command(sent.state, acquire, {at, offset + 1});
                    auto local = before;
                    local.causal = received.state;
                    if (!received.applied || !canPublish(local, forward)) {
                        supported = false;
                        break;
                    }
                }
            }
            if (!supported) {
                continue;
            }
            const auto request = result.decisions.size();
            const OrderedPacket packet{
                {publication, publish, EndpointPurpose::ConsumptionAcknowledgment, request},
                {publication, acquire, EndpointPurpose::ConsumptionAcknowledgment, request},
                {publication, {Command::Publish, source, observer, a.key}, EndpointPurpose::Completion, request},
                {current, {Command::Acquire, source, observer, a.key}, EndpointPurpose::Completion, request}};
            const auto beforeSites = result.work.ownershipCheckSites;
            const auto qualified = qualifyOwnedPacket(packet, true);
            ++result.work.acknowledgmentChecks;
            result.work.acknowledgmentCheckSites += result.work.ownershipCheckSites - beforeSites;
            if (!qualified) { continue; }
            // Commit exactly the packet checked above. In particular do not
            // replay an incomplete reverse half: the forward receipt may rearm
            // its reverse key on the next original visit. No promised credit.
            if (!commitOwnedPacket(*qualified, decision)) { return false; }
            decision.repairedForwardKey = a.key;
            decision.repairReverseKey = b.key;
            ++result.work.acknowledgments;
            ++result.work.joinedAcknowledgments;
            if (publication == current) {
                ++result.work.commonCutTransfers;
            }
            return update();
        }
    }
    return {};
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
bool Constructor::edge(Pipe source, Pipe observer, Cut& publication, bool closed, SelectedDecision& decision)
{
    const bool recurringClosed = closed && control.components[activeComponent].cyclic;
    const auto binding = closedBindings.find({source, observer});
    const bool retained = recurringClosed && binding != closedBindings.end();
    Id key = retained ? binding->second.first : NoAnalysisId;
    if (retained) {
        if (!canPublish(cache.cuts[publication].before, key) &&
            !restoreRearming(key, publication)) {
            return fail(SelectedFailure::EventResource,
                "recurring forward role lacks its consumption path", publication);
        }
    } else {
        for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
            const auto& identity = frontier.keys()[candidate];
            if (identity.source != source || identity.observer != observer || !unownedKey(candidate)) {
                continue;
            }
            ++result.work.keyQueries;
            if (canPublish(cache.cuts[publication].before, candidate) &&
                clearInterval(candidate, publication, current)) {
                key = candidate;
                break;
            }
        }
        if (key == NoAnalysisId) {
            if (auto owned = dormantTransfer(source, observer, publication, closed, decision)) {
                return *owned;
            }
            bool completed = false;
            if (!acknowledgment(source, observer, publication, key, decision, completed)) {
                return false;
            }
            if (completed) {
                return true;
            }
        }
    }
    const auto number = frontier.keys()[key].key;
    const auto request = result.decisions.size();
    OrderedPacket packet{
        {publication, {Command::Publish, source, observer, number}, EndpointPurpose::Completion, request},
        {current, {Command::Acquire, source, observer, number}, EndpointPurpose::Completion, request}};
    if (!closed) {
        // Relay and common-cut simulations still use their checked tail order.
        // Qualify only this new direct SET; no existing endpoint is moved.
        if (source == decision.source && observer == decision.observer) {
            if (auto gap = earlyPublicationGap(publication, key, decision)) {
                packet.front().gap = *gap;
                ++result.work.earlyPublications;
            }
        }
        return commitPacket(packet, decision) && update();
    }
    // A recurring closed word is one selected edit. Replaying its forward half
    // over a backedge before adding its acknowledgment would reject a protocol
    // that is deliberately not complete yet. Select the reply from the actual
    // local forward transfers, then check the complete word on the full graph.
    if (publication != current) {
        return fail(SelectedFailure::MissingParticipation, "closed word requires one common cut", current);
    }
    auto afterForward = currentState();
    const auto offset = ledger.word(current).size();
    auto sent = frontier.command(afterForward.causal,
        {Command::Publish, source, observer, number}, {current, offset});
    if (!sent.applied) return fail(SelectedFailure::SelectedUpdate, sent.reason, current);
    auto acquired = frontier.command(sent.state,
        {Command::Acquire, source, observer, number}, {current, offset + 1});
    if (!acquired.applied) return fail(SelectedFailure::SelectedUpdate, acquired.reason, current);
    afterForward.causal = std::move(acquired.state);
    if (observer == program.operations[control.graph.operations[current]].pipe &&
        !needsCommonAcknowledgment(afterForward)) {
        // The token is consumed, and no future selected/publication or payload
        // needs knowledge of that consumption at its publisher. Do not invent
        // such knowledge: simply omit its unused return transfer. Validation
        // and the native reconstruction checker are unchanged.
        ++result.work.commonCutTransfers;
        return commitPacket(packet, decision) && update();
    }
    const auto reverse = retained ? binding->second.second : reusable(observer, source, afterForward);
    if (reverse == NoAnalysisId || !canPublish(afterForward, reverse)) {
        if (!retained) {
            if (auto owned = dormantTransfer(source, observer, publication, true, decision)) { return *owned; }
        }
        return fail(SelectedFailure::EventResource,
            "common-cut acknowledgment has no independently reusable reverse key", current);
    }
    const auto reverseNumber = frontier.keys()[reverse].key;
    packet.push_back({current, {Command::Publish, observer, source, reverseNumber},
                      EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
    packet.push_back({current, {Command::Acquire, observer, source, reverseNumber},
                      EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
    if (!commitPacket(packet, decision)) {
        return false;
    }
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
std::optional<bool> Constructor::dormantTransfer(
    Pipe source, Pipe observer, Cut publication, bool closed, SelectedDecision& decision)
{
    // Fixed hardware-key alternatives; never subsets of an owning population.
    // Current publishability is not a premise: restoration may establish it.
    if (closed && publication != current) { return {}; }
    const auto request = result.decisions.size();
    for (Id forward = 0; forward < frontier.keys().size(); ++forward) {
        const auto& a = frontier.keys()[forward];
        if (a.source != source || a.observer != observer || closedKeys.count(forward)) { continue; }
        const auto count = closed ? frontier.keys().size() : 1;
        for (Id reverse = 0; reverse < count; ++reverse) {
            const auto& b = frontier.keys()[reverse];
            if (closed && (b.source != observer || b.observer != source || closedKeys.count(reverse))) { continue; }
            const bool dormant = ledger.hasDormantUses(a) || (closed && ledger.hasDormantUses(b));
            if (!dormant) { continue; }
            OrderedPacket packet{
                {publication, {Command::Publish, source, observer, a.key}, EndpointPurpose::Completion, request},
                {current, {Command::Acquire, source, observer, a.key}, EndpointPurpose::Completion, request}};
            if (closed) {
                packet.push_back({current, {Command::Publish, observer, source, b.key},
                    EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
                packet.push_back({current, {Command::Acquire, observer, source, b.key},
                    EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
            }
            const auto qualified = qualifyOwnedPacket(packet);
            if (!qualified) { continue; }
            if (!commitOwnedPacket(*qualified, decision)) { return false; }
            if (closed) {
                rememberReturn(decision.endpoints[decision.endpoints.size() - 2], decision.endpoints.back());
                ++result.work.acknowledgments;
                ++result.work.commonCutTransfers;
                if (control.components[activeComponent].cyclic) {
                    closedBindings[{source, observer}] = {forward, reverse};
                    closedKeys.insert(forward);
                    closedKeys.insert(reverse);
                }
            }
            return update();
        }
    }
    return {};
}
bool Constructor::restoreReturns(Id key)
{
    const auto& identity = frontier.keys()[key];
    const auto found = pendingRearming.find({identity.observer, identity.source});
    if (found == pendingRearming.end()) return false;
    OrderedPacket packet;
    std::vector<Id> restored;
    std::vector<Id> anchors;
    std::vector<std::pair<Id, Id>> selected;
    for (const auto& helper : found->second) {
        if (ledger.active(helper.second)) { continue; }
        const auto wait = ledger.endpoint(helper.second).acknowledges;
        const auto& forward = ledger.endpoint(wait).command;
        const bool matches = forward.source == identity.source && forward.observer == identity.observer &&
            forward.key == identity.key;
        if (!matches) { continue; }
        selected.push_back(helper);
        anchors.push_back(wait);
    }
    const auto gaps = ledger.gapsAfter(anchors);
    for (const auto& helper : selected) {
        const auto wait = ledger.endpoint(helper.second).acknowledges;
        const auto gap = gaps.find(wait);
        if (gap == gaps.end()) { return false; }
        const auto publication = ledger.restoration(helper.first, gap->second);
        const auto acquisition = ledger.restoration(helper.second, gap->second);
        if (!publication || !acquisition) { return false; }
        packet.push_back(*publication);
        packet.push_back(*acquisition);
        restored.push_back(helper.second);
    }
    if (packet.empty()) { return false; }
    const auto prepared = ledger.preparePacket(packet);
    const auto ids = ledger.appendPacket(prepared);
    const bool completePacket = ids.size() == packet.size();
    if (!completePacket) { return false; }
    for (auto wait : restored) {
        requiredReturns.insert(wait);
        ++result.work.acknowledgments;
        ++result.work.rearmingRestored;
        --result.work.rearmingDischarged;
    }
    return true;
}
bool Constructor::restoreRearming(Id key, Cut publication)
{
    return restoreReturns(key) && update() && canPublish(cache.cuts[publication].before, key);
}

void Constructor::rememberReturn(Id publication, Id acquisition)
{
    const auto& c = ledger.endpoint(acquisition).command;
    pendingRearming[{c.source, c.observer}].push_back({publication, acquisition});
    helperOwners[publication] = {publication, acquisition};
    helperOwners[acquisition] = {publication, acquisition};
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
        decision.supporting = group.supporting;
        decision.lifecycles = requirements.demandsAt(current, group.requirements);
        const auto request = result.decisions.size();
        OrderedPacket packet;
        for (auto cut : group.publications) {
            packet.push_back({cut, {Command::Publish, group.source, observer, key.key},
                              EndpointPurpose::Completion, request});
        }
        const auto acquisition = group.entryAcquisition == NoAnalysisId ? current : group.entryAcquisition;
        const auto receipt = packet.size();
        packet.push_back({acquisition, {Command::Acquire, group.source, observer, key.key},
                          EndpointPurpose::Completion, request});
        if (group.entryReturnKey != NoAnalysisId) {
            const auto& reply = frontier.keys()[group.entryReturnKey];
            packet.push_back({acquisition, {Command::Publish, observer, group.source, reply.key},
                              EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, receipt});
            packet.push_back({acquisition, {Command::Acquire, observer, group.source, reply.key},
                              EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, receipt});
        }
        if (group.packet) {
            if (!commitOwnedPacket(*group.packet, decision)) { return false; }
        } else if (!commitPacket(packet, decision)) {
            return false;
        }
        if (group.entryAcquisition != NoAnalysisId) {
            ++result.work.loopEntryTransfers;
            needsContextualReplay = true;
            if (group.entryRepeats) {
                recurringKeys.insert(group.forwardKey);
                closedKeys.insert(group.forwardKey);
            }
            if (group.entryReturnKey != NoAnalysisId) {
                if (group.entryRepeats) {
                    recurringKeys.insert(group.entryReturnKey);
                    closedKeys.insert(group.entryReturnKey);
                }
                rememberReturn(decision.endpoints[decision.endpoints.size() - 2], decision.endpoints.back());
                ++result.work.acknowledgments;
            }
        }
        if (!update()) return false;
        if (!settleRearming(decision)) return false;
        result.decisions.push_back(std::move(decision));
        return true;
    }
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
    decision.supporting = group.supporting;
    decision.lifecycles = requirements.demandsAt(current, group.requirements);
    decision.commonCut = group.common;
    auto source = group.publication;
    for (Id hop = 1; hop < path.size(); ++hop) {
        if (!edge(path[hop - 1], path[hop], source, group.common, decision)) {
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
