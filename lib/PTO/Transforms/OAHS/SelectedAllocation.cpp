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
bool Constructor::helperFreeKey(Id key) const
{
    return unownedKey(key) && !deferredByKey.count(key);
}
std::optional<WordGap> Constructor::restorationDeadline(
    const RearmingObligation& owner, const OrderedPacket& proposed, Id deadline,
    const std::vector<Id>& proposedUses, std::map<Cut, std::map<Id, Id>>& positions, std::string& reason)
{
    ++result.work.restorationDeadlineQueries;
    const auto& consumption = ledger.endpoint(owner.consumption);
    const auto& publication = proposed[deadline];
    reason = "reuse deadline has no valid selected word gap";
    if (publication.cut >= control.graph.sites.size()) { return {}; }
    const auto gap = publication.gap.value_or(ledger.tail(publication.cut));
    const bool invalidGap = gap.cut >= control.graph.sites.size() ||
        control.canonicalCut[gap.cut] != control.canonicalCut[publication.cut];
    if (invalidGap) { return {}; }
    // Packet-local index: each touched word is scanned once, even when several
    // owners and occurrence pairs query positions in that word.
    const auto position = [&](Cut cut, Id endpoint) -> std::optional<Id> {
        cut = control.canonicalCut[cut];
        auto found = positions.find(cut);
        if (found == positions.end()) {
            auto& index = positions[cut];
            const auto& word = ledger.word(cut);
            for (Id offset = 0; offset < word.size(); ++offset) {
                index.emplace(word[offset], offset);
                ++result.work.restorationPositionEntries;
            }
            found = positions.find(cut);
        }
        if (endpoint == NoAnalysisId) {
            return ledger.word(cut).size();
        }
        const auto value = found->second.find(endpoint);
        if (value == found->second.end()) { return {}; }
        return value->second;
    };
    const auto gapOffset = [&](const WordGap& at) -> std::optional<Id> {
        if (at.cut >= control.graph.sites.size()) { return {}; }
        const auto offset = position(at.cut, at.right);
        if (!offset) { return {}; }
        const auto& word = ledger.word(at.cut);
        if (at.left != (*offset ? word[*offset - 1] : NoAnalysisId)) { return {}; }
        return offset;
    };
    const auto target = gapOffset(gap);
    const auto consumed = position(consumption.cut, owner.consumption);
    if (!target || !consumed) { return {}; }
    const auto targetOffset = *target;
    const auto sourceOffset = *consumed + 1;
    reason = "reuse deadline lacks one acyclic matched occurrence interval";
    const auto& relation = control.correspondence(consumption.cut, publication.cut);
    if (!relation.proved()) { return {}; }
    const auto reverseKey = keyIndex(frontier, ledger.endpoint(owner.acquisition).command);
    auto within = [&](Cut cut, Id offset, const auto& pair) {
        const bool enclosed = control.straight(pair.first, cut) && control.straight(cut, pair.second);
        if (!enclosed) { return false; }
        if (cut == pair.first && offset < sourceOffset) { return false; }
        if (cut == pair.second && offset >= targetOffset) { return false; }
        return true;
    };
    for (const auto& pair : relation.pairs) {
        if (!control.reachable[pair.first]) { continue; }
        const bool cyclic = control.components[control.component[pair.first]].cyclic ||
            control.components[control.component[pair.second]].cyclic;
        if (cyclic || !control.straight(pair.first, pair.second)) { return {}; }
        if (pair.first == pair.second && sourceOffset > targetOffset) { return {}; }
        // The inactive return may remain live until this deadline only when
        // neither key has an earlier selected use. Event population is sparse.
        for (auto key : {owner.forwardKey, reverseKey}) {
            for (auto id : ledger.eventUses(frontier.keys()[key])) {
                if (!ledger.active(id)) { continue; }
                ++result.work.restorationUseChecks;
                const auto& endpoint = ledger.endpoint(id);
                const auto offset = position(endpoint.cut, id);
                if (!offset) { return {}; }
                for (auto site : control.wordOccurrences[endpoint.cut]) {
                    if (within(site, *offset, pair)) {
                        reason = "earlier selected forward or reverse key use"; return {};
                    }
                }
            }
        }
        for (auto index : proposedUses) {
            if (index == deadline) { continue; }
            ++result.work.restorationUseChecks;
            const auto& use = proposed[index];
            const auto useGap = use.gap.value_or(ledger.tail(use.cut));
            const auto offset = gapOffset(useGap);
            if (!offset || control.canonicalCut[useGap.cut] != control.canonicalCut[use.cut]) { return {}; }
            for (auto site : control.wordOccurrences[control.canonicalCut[use.cut]]) {
                const bool afterSource = site != pair.first || *offset >= sourceOffset;
                const bool beforeTarget = site != pair.second || *offset < targetOffset ||
                    (*offset == targetOffset && index < deadline);
                const bool earlier = control.straight(pair.first, site) && control.straight(site, pair.second) &&
                    afterSource && beforeTarget;
                if (earlier) { reason = "packet has an earlier use of a restored key"; return {}; }
            }
        }
    }
    reason.clear();
    return gap;
}
std::optional<OwnedPacket> Constructor::prepareOwnedPacket(
    const OrderedPacket& proposed, const std::vector<Id>& obligations, bool placeAtDeadline)
{
    ++result.work.ownershipQueries;
    std::set<std::tuple<Pipe, Pipe, unsigned>> touched;
    std::set<Id> owners;
    auto pending = obligations;
    // Both a proposed physical use and an explicit rearming deadline request
    // complete ownership, never a selected subset of the dormant population.
    const auto touch = [&](const Command& command) {
        if (command.kind != Command::Publish && command.kind != Command::Acquire) { return true; }
        const EventIdentity identity{command.source, command.observer, command.key};
        const bool newIdentity = touched.emplace(command.source, command.observer, command.key).second;
        if (!newIdentity) { return true; }
        const auto forward = deferredByKey.find(keyIndex(frontier, command));
        if (forward != deferredByKey.end()) {
            pending.insert(pending.end(), forward->second.begin(), forward->second.end());
        }
        if (!ledger.hasDormantUses(identity)) { return true; }
        for (auto endpoint : ledger.eventUses(identity)) {
            if (ledger.active(endpoint)) { continue; }
            const auto owner = helperOwners.find(endpoint);
            if (owner == helperOwners.end()) { return false; }
            const auto record = rearming.find(owner->second);
            if (record == rearming.end()) { return false; }
            if (endpoint != record->second.publication && endpoint != record->second.acquisition) { return false; }
            pending.push_back(owner->second);
        }
        return true;
    };
    for (const auto& item : proposed) {
        const bool supported = legalCommandCut(program, item.cut) && touch(item.command);
        if (!supported) { return {}; }
    }
    for (Id index = 0; index < pending.size(); ++index) {
        const auto id = pending[index];
        if (!owners.insert(id).second) { continue; }
        const auto found = rearming.find(id);
        if (found == rearming.end()) { return {}; }
        const auto& owner = found->second;
        const bool validIds = owner.publication < ledger.records().size() &&
            owner.acquisition < ledger.records().size() && owner.consumption < ledger.records().size();
        if (!validIds) { return {}; }
        const bool partiallyActive = ledger.active(owner.publication) || ledger.active(owner.acquisition);
        if (partiallyActive) { return {}; }
        const auto& pub = ledger.endpoint(owner.publication);
        const auto& wait = ledger.endpoint(owner.acquisition);
        const auto& forward = ledger.endpoint(owner.consumption).command;
        const bool matching = pub.purpose == EndpointPurpose::ConsumptionAcknowledgment &&
            wait.purpose == pub.purpose && pub.acknowledges == owner.consumption &&
            wait.acknowledges == owner.consumption && ledger.active(owner.consumption) &&
            pub.command.kind == Command::Publish && wait.command.kind == Command::Acquire &&
            pub.command.source == wait.command.source && pub.command.observer == wait.command.observer &&
            pub.command.key == wait.command.key && forward.kind == Command::Acquire &&
            forward.source == wait.command.observer && forward.observer == wait.command.source &&
            owner.forwardKey == keyIndex(frontier, forward);
        if (!matching || !touch(wait.command)) { return {}; }
    }
    std::vector<Id> anchors;
    for (auto id : owners) { anchors.push_back(rearming.at(id).consumption); }
    const auto gaps = ledger.gapsAfter(anchors);
    OwnedPacket out;
    OrderedPacket packet;
    std::map<Id, std::vector<Id>> publications, uses;
    std::map<Id, Id> reverseOwners;
    std::map<Cut, std::map<Id, Id>> positions;
    for (Id index = 0; index < proposed.size(); ++index) {
        const auto& command = proposed[index].command;
        if (command.kind != Command::Publish && command.kind != Command::Acquire) { continue; }
        const auto key = keyIndex(frontier, command);
        uses[key].push_back(index);
        if (command.kind == Command::Publish) { publications[key].push_back(index); }
    }
    for (auto id : owners) {
        ++reverseOwners[keyIndex(frontier, ledger.endpoint(rearming.at(id).acquisition).command)];
    }
    for (auto id : owners) {
        const auto& owner = rearming.at(id);
        const auto gap = gaps.find(owner.consumption);
        if (gap == gaps.end()) { return {}; }
        const auto pub = ledger.restoration(owner.publication, gap->second);
        auto wait = ledger.restoration(owner.acquisition, gap->second);
        auto& reason = out.fallbackReasons[id];
        reason = "no proposed forward republication deadline";
        const auto found = publications.find(owner.forwardKey);
        const auto reverse = keyIndex(frontier, ledger.endpoint(owner.acquisition).command);
        const bool single = found != publications.end() && found->second.size() == 1 && reverseOwners[reverse] == 1;
        const bool multiple = found != publications.end() && found->second.size() != 1;
        if (multiple) {
            reason = "multiple proposed forward republication deadlines";
        }
        if (reverseOwners[reverse] != 1) { reason = "multiple dormant reverse-key owners"; }
        if (!placeAtDeadline) { reason = "deadline realization failed complete packet qualification"; }
        if (single && placeAtDeadline) {
            auto relevant = uses[owner.forwardKey];
            relevant.insert(relevant.end(), uses[reverse].begin(), uses[reverse].end());
            const auto deadline = found->second.front();
            if (const auto late = restorationDeadline(owner, proposed, deadline, relevant, positions, reason)) {
                wait = ledger.relocateAcknowledgment(owner.acquisition, *late);
                out.deadlines.emplace(id, deadline);
            }
        }
        if (!pub || !wait) { return {}; }
        packet.push_back(*pub);
        packet.push_back(*wait);
        out.restoredWaits.push_back(id);
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
    publicationSupport.refresh(ledger);
    const auto view = ledger.packetView(out.prepared);
    if (view) {
        out.publicationProbe = publicationSupport.inspect(ledger, &*view, view->changedCuts());
    }
    if (!out.prepared.valid()) { return {}; }
    return out;
}
std::optional<OwnedPacket> Constructor::qualifyOwnedPacket(
    const OrderedPacket& proposed, bool alwaysCheck, const std::vector<Id>& obligations, AnalysisResult* refusal)
{
    auto out = prepareOwnedPacket(proposed, obligations);
    if (!out) { return {}; }
    if (alwaysCheck || !out->restoredWaits.empty()) {
        const auto qualify = [&](OwnedPacket& candidate) {
            const auto commands = ledger.withPacket(candidate.prepared);
            if (!commands) { return false; }
            auto checked = analyze(program, *commands, {false});
            ++result.work.ownershipChecks;
            result.work.ownershipCheckSites += checked.stats.siteEvaluations;
            const bool accepted = acceptOwnedPacket(candidate, checked);
            if (!accepted && refusal) { *refusal = std::move(checked); }
            return accepted;
        };
        if (!qualify(*out)) {
            // One prescribed fallback, with the same complete dormant-owner
            // closure and logical request. Never search owner subsets or shapes.
            if (out->deadlines.empty()) { return {}; }
            ++result.work.restorationDeadlineFallbacks;
            out = prepareOwnedPacket(proposed, obligations, false);
            if (!out || !qualify(*out)) { return {}; }
        }
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
    const bool currentPublicationProof = packet.publicationProbe &&
        packet.publicationProbe->version == ledger.version() + packet.prepared.size();
    if (!currentPublicationProof) {
        return fail(SelectedFailure::SelectedUpdate, "publication certificate changed after qualification", current);
    }
    const auto ids = ledger.appendPacket(packet.prepared);
    const bool committed = ids.size() == packet.prepared.size();
    if (!committed) {
        return fail(SelectedFailure::SelectedUpdate, "owned packet changed after qualification", current);
    }
    publicationSupport.accept(ledger, *packet.publicationProbe);
    decision.existingPublicationsPreserved &= packet.publicationProbe->preserved;
    decision.endpoints.insert(decision.endpoints.end(), ids.begin() + packet.restoredEndpoints, ids.end());
    if (!packet.restoredWaits.empty()) { ++result.work.ownershipBindings; }
    for (auto wait : packet.restoredWaits) {
        auto& obligation = rearming.at(wait);
        obligation.required = true;
        const auto deadline = packet.deadlines.find(wait);
        const auto& placed = ledger.endpoint(obligation.acquisition);
        const auto originalCut = placed.originalCut == NoAnalysisId ? placed.cut : placed.originalCut;
        const auto publication = deadline == packet.deadlines.end() ? NoAnalysisId :
            ids[packet.restoredEndpoints + deadline->second];
        result.restorations.push_back({obligation.consumption, obligation.publication, obligation.acquisition,
            publication, originalCut, placed.cut, ledger.version(), packet.fallbackReasons.at(wait)});
        result.work.deadlineRestorations += deadline != packet.deadlines.end();
        ++result.work.acknowledgments;
        ++result.work.rearmingRestored;
        if (obligation.deferred) {
            obligation.deferred = false;
            auto& forwardOwners = deferredByKey.at(obligation.forwardKey);
            forwardOwners.erase(wait);
            if (forwardOwners.empty()) {
                deferredByKey.erase(obligation.forwardKey);
            }
            auto& latent = latentReturns.at(obligation.consumption);
            latent.erase(wait);
            if (latent.empty()) {
                latentReturns.erase(obligation.consumption);
            }
            ++result.work.deferredMaterialized;
        } else {
            --result.work.rearmingDischarged;
        }
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
        std::map<Cut, bool> wordIntersects;
        for (auto id : ledger.eventUses(identity)) {
            if (!ledger.active(id)) { continue; }
            const auto word = control.canonicalCut[ledger.endpoint(id).cut];
            const auto [found, inserted] = wordIntersects.try_emplace(word, false);
            if (inserted) {
                for (auto occurrence : control.wordOccurrences[word]) {
                    found->second |= occurrence != pair.first &&
                        control.straight(pair.first, occurrence) &&
                        control.straight(occurrence, pair.second);
                }
            }
            if (found->second) { return false; }
        }
    }
    return true;
}
SourceGapQualification Constructor::sourceGapFacts(
    const WordGap& gap, Pipe source, Pipe observer, const std::vector<FrontierRequirement>& required,
    Cut deadline)
{
    ++result.work.sourceGapQueries;
    if (deadline == NoAnalysisId) { deadline = current; }
    SourceGapQualification out;
    out.gap = gap;
    out.source = source;
    out.observer = observer;
    out.deadline = deadline;
    out.version = ledger.version();
    out.reason = "source gap lacks a current matched occurrence certificate";
    const bool invalidQuery = !cache.success || cache.version != ledger.version() ||
        required.empty() || gap.cut >= control.graph.sites.size() || deadline >= control.graph.sites.size();
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
    const auto& correspondence = control.correspondence(gap.cut, deadline);
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
        for (const auto& requirement : required) {
            const auto access = accessClass(requirement);
            const auto* history = state.facts()->history.find(access);
            const bool covered = history && frontierContains(*history, PipeCount + unsigned(source)) &&
                freshBetween(pair.first, pair.second, access);
            if (!covered) {
                out.reason = "source gap does not cover the required physical occurrence";
                return out;
            }
        }
        out.prefixes.push_back(std::move(state));
        reached = true;
    }
    if (reached) {
        out.outcome = ProofOutcome::Proved;
        out.reason.clear();
    }
    return out;
}
bool Constructor::sourceKeyNeighbors(const SourceGapQualification& facts, Id key)
{
    const bool valid = facts.proved() && facts.version == ledger.version() && cache.version == facts.version &&
        facts.deadline < control.graph.sites.size() && key < frontier.keys().size() &&
        !facts.prefixes.empty();
    if (!valid || !helperFreeKey(key)) { return false; }
    const auto& identity = frontier.keys()[key];
    if (identity.source != facts.source || identity.observer != facts.observer) { return false; }
    const auto& uses = ledger.eventUses(identity);
    if (uses.empty()) { return true; }
    // Include every original continuation and backedge. Earlier uses and
    // disjoint alternatives are permitted only with actual source-time rearming.
    // Same-word uses remain Unknown: this certificate admits word-beginning gaps.
    if (facts.futureSites.empty()) {
        facts.futureSites.resize(control.graph.sites.size());
        result.work.normalKeySites += facts.futureSites.size();
        std::vector<Cut> pending;
        for (const auto& pair : control.correspondence(facts.gap.cut, facts.deadline).pairs) {
            pending.push_back(pair.first);
        }
        while (!pending.empty()) {
            const auto site = pending.back(); pending.pop_back();
            if (!control.reachable[site] || facts.futureSites[site]) { continue; }
            facts.futureSites[site] = true;
            ++result.work.normalKeySites;
            const auto& next = control.graph.sites[site].successors;
            pending.insert(pending.end(), next.begin(), next.end());
        }
    }
    std::map<Cut, bool> wordIntersects;
    return selectedKeyUsesOutside(facts.futureSites, key, wordIntersects);
}
bool Constructor::selectedKeyUsesOutside(const std::vector<bool>& futureSites, Id key,
    std::map<Cut, bool>& wordIntersects)
{
    const bool valid = futureSites.size() == control.graph.sites.size() &&
        key < frontier.keys().size();
    if (!valid) { return false; }
    for (auto id : ledger.eventUses(frontier.keys()[key])) {
        ++result.work.repairNeighborUses;
        const auto& endpoint = ledger.endpoint(id);
        const bool supported = ledger.active(id) &&
            (endpoint.purpose == EndpointPurpose::Completion || endpoint.purpose == EndpointPurpose::Fixed);
        if (!supported) { return false; }
        const auto [found, inserted] = wordIntersects.try_emplace(endpoint.cut, false);
        if (inserted) {
            for (auto site : control.wordOccurrences[endpoint.cut]) {
                ++result.work.normalKeySites;
                found->second |= futureSites[site];
            }
        }
        if (found->second) { return false; }
    }
    return true;
}
bool Constructor::sourceGapKey(const SourceGapQualification& facts, Id key)
{
    if (!sourceKeyNeighbors(facts, key)) { return false; }
    for (const auto& prefix : facts.prefixes) {
        State atGap;
        atGap.causal = prefix;
        if (!canPublish(atGap, key)) { return false; }
    }
    return true;
}
std::optional<WordGap> Constructor::earlyPublicationMilestone(Cut cut, Pipe source) const
{
    const auto& ids = ledger.word(cut);
    auto offset = ids.size();
    bool incomingWait = false;
    while (offset != 0) {
        const auto& command = ledger.endpoint(ids[offset - 1]).command;
        const auto pipe = command.kind == Command::Acquire ? command.observer : command.source;
        if (command.kind == Command::BarrierAll ||
            (pipe == source && command.kind != Command::Acquire)) {
            break;
        }
        incomingWait |= command.kind == Command::Acquire && command.observer == source;
        --offset;
    }
    if (!incomingWait) {
        return {};
    }
    return WordGap{control.canonicalCut[cut], offset == 0 ? NoAnalysisId : ids[offset - 1], ids[offset]};
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
                                 SelectedDecision& decision, bool& completed, OrderedPacket& prefix)
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
    prefix = {
        {gap->cut, {Command::Publish, observer, source, identity.key},
         EndpointPurpose::ConsumptionAcknowledgment, request, oldWait, gap},
        {moved, {Command::Acquire, observer, source, identity.key},
         EndpointPurpose::ConsumptionAcknowledgment, request, oldWait}};
    // This is a selected shape, not selected credit. edge() appends the forward
    // transfer and any closed return before qualifying and committing one edit.
    decision.enlargedPrefix |= moved != publication;
    publication = moved;
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

void Constructor::deferReturn(Id id)
{
    auto& obligation = rearming.at(id);
    obligation.deferred = true;
    deferredByKey[obligation.forwardKey].insert(id);
    latentReturns[obligation.consumption].insert(id);
    ledger.erase(obligation.publication);
    ledger.erase(obligation.acquisition);
    ++result.work.rearmingDeferred;
}
bool Constructor::canDeferCommonReturn() const
{
    const auto canonical = control.canonicalCut[current];
    for (auto site : control.wordOccurrences[canonical]) {
        if (control.reachable[site] && (site != current || control.components[control.component[site]].cyclic)) {
            return false;
        }
    }
    return control.reachable[current];
}
bool Constructor::latentPublicationAfter(Id anchor, Pipe observer) const
{
    const auto found = latentReturns.find(anchor);
    if (found == latentReturns.end()) { return false; }
    for (auto id : found->second) {
        const auto& obligation = rearming.at(id);
        if (obligation.deferred && ledger.endpoint(obligation.publication).command.source == observer) {
            return true;
        }
    }
    return false;
}
bool Constructor::terminalCommonCut()
{
    ++result.work.commonCutContinuationQueries;
    // A syntactically last body operation is not a last dynamic operation. The
    // backward summary retains original backedges and all branch alternatives.
    if (control.lookahead.mayIssueAfter(current)) { return false; }
    // A selected word may be shared by several original occurrences. Being
    // terminal at only this occurrence is not a terminal-channel certificate.
    const auto canonical = control.canonicalCut[current];
    const auto& occurrences = control.wordOccurrences[canonical];
    if (std::count_if(occurrences.begin(), occurrences.end(),
                     [&](Id site) { return control.reachable[site]; }) != 1) { return false; }
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId) { return false; }
    // Only the final cross-engine acquisition of this terminal payload can use
    // this rule. Later fixed/recurring words can have rearming obligations even
    // when there are no later payloads; inspect the CURRENT ledger, not the
    // immutable analysis. Terminal ALL neither consumes nor rearms an event.
    std::vector<bool> seen(control.graph.sites.size());
    auto todo = control.graph.sites[current].successors;
    while (!todo.empty()) {
        const auto site = todo.back();
        todo.pop_back();
        if (seen[site]) { continue; }
        seen[site] = true;
        ++result.work.commonCutContinuationSites;
        result.work.commonCutContinuationWords += ledger.word(site).size();
        for (auto endpoint : ledger.word(site)) {
            if (ledger.endpoint(endpoint).command.kind != Command::BarrierAll) { return false; }
        }
        const auto& next = control.graph.sites[site].successors;
        todo.insert(todo.end(), next.begin(), next.end());
    }
    return true;
}
bool Constructor::needsCommonAcknowledgment(const State& afterForward)
{
    if (!terminalCommonCut()) { return true; }
    const auto operation = control.graph.operations[current];
    const auto checked = frontier.inspect(afterForward.causal, operation);
    if (!checked.applied && checked.failure != FrontierFailure::Payload) { return true; }
    const auto observer = program.operations[operation].pipe;
    return std::any_of(checked.residuals.begin(), checked.residuals.end(),
        [&](const auto& r) { return r.source != observer; });
}
bool Constructor::edge(Pipe source, Pipe observer, Cut& publication, bool closed, SelectedDecision& decision)
{
    const bool recurringClosed = closed && control.components[activeComponent].cyclic;
    const auto binding = closedBindings.find({source, observer});
    const bool retained = recurringClosed && binding != closedBindings.end();
    Id key = retained ? binding->second.first : NoAnalysisId;
    std::optional<WordGap> selectedGap;
    OrderedPacket prefix;
    if (!retained && !closed && source == decision.source && observer == decision.observer) {
        const auto milestone = earlyPublicationMilestone(publication, source);
        if (milestone) {
            auto required = decision.required;
            required.insert(required.end(), decision.supporting.begin(), decision.supporting.end());
            const auto facts = sourceGapFacts(*milestone, source, observer, required);
            if (facts.proved()) {
                for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
                    const auto& identity = frontier.keys()[candidate];
                    if (identity.source != source || identity.observer != observer) { continue; }
                    ++result.work.keyQueries;
                    if (!sourceGapKey(facts, candidate)) { continue; }
                    key = candidate;
                    selectedGap = *milestone;
                    break;
                }
            }
        }
    }
    if (retained) {
        if (!canPublish(cache.cuts[publication].before, key) &&
            !restoreRearming(key, publication)) {
            return fail(SelectedFailure::EventResource,
                "recurring forward role lacks its consumption path", publication);
        }
    } else if (key == NoAnalysisId) {
        for (Id candidate = 0; candidate < frontier.keys().size(); ++candidate) {
            const auto& identity = frontier.keys()[candidate];
            if (identity.source != source || identity.observer != observer || !helperFreeKey(candidate)) {
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
            if (!acknowledgment(source, observer, publication, key, decision, completed, prefix)) {
                return false;
            }
            if (completed) {
                return true;
            }
        }
    }
    const auto number = frontier.keys()[key].key;
    const auto request = result.decisions.size();
    OrderedPacket packet = prefix;
    const auto forwardWait = packet.size() + 1;
    packet.push_back({publication, {Command::Publish, source, observer, number}, EndpointPurpose::Completion, request});
    packet.push_back({current, {Command::Acquire, source, observer, number}, EndpointPurpose::Completion, request});
    const auto commit = [&]() {
        if (prefix.empty()) {
            return commitPacket(packet, decision);
        }
        const auto qualified = qualifyOwnedPacket(packet, true);
        if (!qualified) {
            return fail(SelectedFailure::SelectedUpdate, "complete acknowledgment packet was refused", current);
        }
        return commitOwnedPacket(*qualified, decision);
    };
    const auto selectedUpdate = [&]() {
        if (!prefix.empty()) { ++result.work.acknowledgments; }
        if (!update()) { return false; }
        if (!prefix.empty()) {
            decision.repairOutputVersion = ledger.version();
        }
        return true;
    };
    if (!closed) {
        if (selectedGap) {
            packet.front().gap = *selectedGap;
            ++result.work.earlyPublications;
        }
        return commit() && selectedUpdate();
    }
    // A recurring closed word is one selected edit. Replaying its forward half
    // over a backedge before adding its acknowledgment would reject a protocol
    // that is deliberately not complete yet. Select the reply from the actual
    // local forward transfers, then check the complete word on the full graph.
    if (publication != current) {
        return fail(SelectedFailure::MissingParticipation, "closed word requires one common cut", current);
    }
    auto afterForward = currentState();
    auto offset = ledger.word(current).size();
    if (!prefix.empty()) {
        // Reproduce the old intermediate state privately to select the same
        // closed continuation. It is not an independently selected protocol.
        const auto staged = ledger.preparePacket(prefix);
        const auto view = ledger.packetView(staged);
        if (!view) { return fail(SelectedFailure::SelectedUpdate, "acknowledgment prefix changed", current); }
        auto evaluated = evaluateContextual(&*view, producerSupportClasses, producerSupportConsumers, 0);
        ++result.work.acknowledgmentPrefixReplays;
        result.work.acknowledgmentPrefixReplaySites += evaluated.evaluations;
        if (!evaluated.success) {
            return fail(SelectedFailure::SelectedUpdate, evaluated.reason, evaluated.failureCut);
        }
        afterForward = evaluated.cuts[current].before;
        offset = view->word(current).size();
    }
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
        return commit() && selectedUpdate();
    }
    const auto reverse = retained ? binding->second.second : reusable(observer, source, afterForward);
    if (reverse == NoAnalysisId || !canPublish(afterForward, reverse)) {
        if (!retained) {
            auto alternative = decision;
            if (!prefix.empty()) {
                alternative.repairedAcquisition = NoAnalysisId;
                alternative.repairedForwardKey = alternative.repairReverseKey = 0;
                alternative.repairInputVersion = alternative.repairOutputVersion = 0;
            }
            if (auto owned = dormantTransfer(source, observer, publication, true, alternative)) {
                if (*owned) { decision = std::move(alternative); }
                return *owned;
            }
        }
        return fail(SelectedFailure::EventResource,
            "common-cut acknowledgment has no independently reusable reverse key", current);
    }
    const auto reverseNumber = frontier.keys()[reverse].key;
    packet.push_back({current, {Command::Publish, observer, source, reverseNumber},
                      EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, forwardWait});
    packet.push_back({current, {Command::Acquire, observer, source, reverseNumber},
                      EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, forwardWait});
    bool defer = !retained && canDeferCommonReturn();
    bool committed = false;
    if (defer) {
        // Certify the complete fallback first. The dormant endpoints are owned
        // structural records, never executed causal credit.
        const auto qualified = qualifyOwnedPacket(packet, true);
        if (qualified) {
            if (!commitOwnedPacket(*qualified, decision)) { return false; }
            committed = true;
        } else {
            if (!prefix.empty()) {
                return fail(SelectedFailure::SelectedUpdate, "complete acknowledgment packet was refused", current);
            }
            defer = false; // Retain the existing closed construction certificate.
        }
    }
    if (!committed && !commit()) { return false; }
    const auto helperPub = decision.endpoints[decision.endpoints.size() - 2];
    const auto helperWait = decision.endpoints.back();
    rememberReturn(helperPub, helperWait);
    if (defer) {
        deferReturn(helperWait);
    }
    result.work.acknowledgments += !defer;
    ++result.work.commonCutTransfers;
    if (!selectedUpdate()) {
        if (!defer) { return false; }
        // Forward-only continuation refused: realize the preserved complete
        // fallback before rejecting this otherwise supported local decision.
        if (!restoreReturns(key)) { return false; }
        result.failure = SelectedFailure::None;
        result.reason.clear();
        result.cut = NoAnalysisId;
        if (!update()) { return false; }
        if (!prefix.empty()) {
            decision.repairOutputVersion = ledger.version();
        }
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
        const bool deferred = deferredByKey.count(forward) != 0;
        const auto count = closed ? frontier.keys().size() : 1;
        for (Id reverse = 0; reverse < count; ++reverse) {
            const auto& b = frontier.keys()[reverse];
            if (closed && (b.source != observer || b.observer != source || closedKeys.count(reverse))) {
                continue;
            }
            const bool dormant = deferred || ledger.hasDormantUses(a) || (closed && ledger.hasDormantUses(b));
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
    const auto found = rearmingByKey.find(key);
    if (found == rearmingByKey.end()) { return false; }
    std::vector<Id> obligations;
    for (auto id : found->second) {
        if (!ledger.active(rearming.at(id).acquisition)) {
            obligations.push_back(id);
        }
    }
    if (obligations.empty()) { return false; }
    // A single edit can expose several independent deadlines. Grow one private
    // recovery from actual missing-consumption diagnostics, never try subsets.
    // No partial restoration or assumed credit enters the selected ledger.
    std::set<Id> selected(obligations.begin(), obligations.end());
    std::set<Id> expandedKeys{key};
    while (true) {
        auto packet = prepareOwnedPacket({}, {selected.begin(), selected.end()});
        if (!packet || packet->restoredWaits.empty()) { return false; }
        const auto commands = ledger.withPacket(packet->prepared);
        if (!commands) { return false; }
        const auto checked = analyze(program, *commands, {false});
        ++result.work.ownershipChecks;
        result.work.ownershipCheckSites += checked.stats.siteEvaluations;
        if (acceptOwnedPacket(*packet, checked)) {
            SelectedDecision restoration;
            return commitOwnedPacket(*packet, restoration);
        }
        if (!checked.complete || !checked.diagnostics.empty()) { return false; }
        // Include closure owners in the progress measure; a diagnostic for one
        // already staged cannot induce a redundant solve or a recovery cycle.
        selected.insert(packet->restoredWaits.begin(), packet->restoredWaits.end());
        const auto before = selected.size();
        for (const auto& issue : checked.protocol) {
            if (issue.kind != ProtocolObligation::ConsumptionNotEstablished) { continue; }
            const auto failedKey = keyIndex(frontier,
                {Command::Publish, issue.event.source, issue.event.observer, issue.event.key});
            if (!expandedKeys.insert(failedKey).second) { continue; }
            const auto failed = rearmingByKey.find(failedKey);
            if (failed == rearmingByKey.end()) { continue; }
            for (auto id : failed->second) {
                if (!ledger.active(rearming.at(id).acquisition)) {
                    selected.insert(id);
                }
            }
        }
        const bool progress = selected.size() != before;
        if (!progress) { return false; }
    }
}
bool Constructor::restoreRearming(Id key, Cut publication)
{
    return restoreReturns(key) && update() && canPublish(cache.cuts[publication].before, key);
}

void Constructor::rememberReturn(Id publication, Id acquisition)
{
    const auto& endpoint = ledger.endpoint(acquisition);
    const auto consumption = endpoint.acknowledges;
    const auto forward = consumption < ledger.records().size()
        ? keyIndex(frontier, ledger.endpoint(consumption).command) : NoAnalysisId;
    const auto inserted = rearming.emplace(acquisition,
        RearmingObligation{consumption, forward, publication, acquisition});
    if (!inserted.second) { return; }
    const auto& c = endpoint.command;
    pendingRearming[{c.source, c.observer}].push_back(acquisition);
    helperOwners[publication] = acquisition;
    helperOwners[acquisition] = acquisition;
    if (forward != NoAnalysisId) { rearmingByKey[forward].push_back(acquisition); }
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
            if (offset != 0) {
                ++result.work.latentSupportChecks;
                if (latentPublicationAfter(word[offset - 1], helper.command.observer)) {
                    ++result.work.latentSupportRetained;
                    return false;
                }
            }
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
        if (acquired) { continue; }
        if (!word.empty()) {
            ++result.work.latentSupportChecks;
            if (latentPublicationAfter(word.back(), helper.command.observer)) {
                ++result.work.latentSupportRetained;
                return false;
            }
        }
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
            auto& helper = rearming.at(pending->second[h]);
            const bool removable = ledger.active(helper.acquisition) && !helper.required;
            if (!removable) { continue; }
            for (Id r = h < paired.first ? paired.second : 0; r < actuals; ++r) {
                ++result.work.rearmingPairVisits;
                if (!returnBeforeUse(helper.acquisition, returns->second[r])) { continue; }
                helper.supportingReceipt = returns->second[r];
                helper.supportRevision = ledger.version();
                ledger.erase(helper.publication);
                ledger.erase(helper.acquisition);
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
