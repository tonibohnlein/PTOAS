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
bool subset(const std::set<Id>& a, const std::set<Id>& b)
{
    return std::includes(b.begin(), b.end(), a.begin(), a.end());
}
std::set<Id> classes(const std::vector<FrontierRequirement>& values)
{
    std::set<Id> out;
    for (const auto& value : values) { out.insert(accessClass(value)); }
    return out;
}
} // namespace
Id selectRealization(const std::vector<CertifiedRealization>& candidates)
{
    std::vector<Id> order;
    for (Id i = 0; i < candidates.size(); ++i) {
        if (!candidates[i].ownCoverage.empty() && subset(candidates[i].ownCoverage,candidates[i].coverage)) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](Id a, Id b) {
        return std::tie(candidates[a].order, candidates[a].shape) <
               std::tie(candidates[b].order, candidates[b].shape);
    });
    Id winner = NoAnalysisId;
    for (auto i : order) {
        const auto& next = candidates[i];
        if (winner == NoAnalysisId) { winner = i; continue; }
        const auto& old = candidates[winner];
        const auto rank = std::make_pair(next.placementClass, !next.known);
        const auto previous = std::make_pair(old.placementClass, !old.known);
        if (rank < previous || (rank == previous && next.coverage.size() > old.coverage.size() &&
                               subset(old.coverage, next.coverage))) { winner = i; }
    }
    return winner;
}
std::vector<DueObligation> Constructor::normalizeDue(const std::vector<FrontierRequirement>& due)
{
    if (normalCellIdentities.empty()) {
        // Equal complete original incidence signatures preserve occurrence,
        // native scope and partial-write interpretation. Marginal origin equality
        // alone is insufficient to merge physical witnesses into one obligation.
        using Incidence = std::tuple<Cut, bool, bool, bool, Id>;
        std::map<unsigned, std::set<Incidence>> uses;
        for (Cut site = 0; site < control.graph.sites.size(); ++site) {
            const auto op = control.graph.operations[site];
            if (op == NoAnalysisId || !control.reachable[site]) { continue; }
            for (const auto& access : program.operations[op].accesses) {
                ++result.work.normalizationIncidences;
                uses[access.cell].emplace(site, access.read, access.write,
                    access.definiteWrite, access.nativeAccumulatorClass);
            }
        }
        using Signature = std::tuple<bool, Cell::Storage, Cell::Domain, std::set<Incidence>>;
        std::map<Signature, Id> identities;
        normalCellIdentities.resize(program.cells.size());
        for (Id cell = 0; cell < program.cells.size(); ++cell) {
            const auto& facts = program.cells[cell];
            const auto found = identities.emplace(Signature{facts.exclusive, facts.storage, facts.domain,
                std::move(uses[cell])}, identities.size());
            normalCellIdentities[cell] = found.first->second;
        }
    }
    std::map<Id, std::set<Cut>> origins;
    for (const auto& fact : requirements.at(current)) {
        origins[fact.access].insert(fact.relationship.source.site);
    }
    using Identity = std::tuple<Pipe, bool, bool, bool, Id, std::set<Cut>>;
    std::map<Identity, std::set<Id>> normalized;
    for (const auto& r : due) {
        const auto access = accessClass(r);
        // Unknown provenance cannot identify otherwise independent witnesses.
        auto identity = origins[access];
        if (identity.empty()) {
            identity.insert(NoAnalysisId - access);
        }
        normalized[{r.source, r.sourceWrite, r.consumerRead, r.consumerWrite,
                    normalCellIdentities[r.cell], identity}].insert(access);
    }
    std::vector<DueObligation> out;
    for (auto& entry : normalized) { out.push_back({std::move(entry.second)}); }
    return out;
}
std::set<Id> Constructor::normalizedCoverage(
    const std::vector<DueObligation>& universe, const std::vector<FrontierRequirement>& due,
    const std::set<Id>& coverage) const
{
    const auto pending = classes(due);
    std::set<Id> out;
    for (Id i = 0; i < universe.size(); ++i) {
        bool any = false, complete = true;
        for (auto access : universe[i].classes) {
            if (!pending.count(access)) { continue; }
            any = true;
            complete &= coverage.count(access) != 0;
        }
        if (any && complete) { out.insert(i); }
    }
    return out;
}
bool Constructor::preservePublications(const OwnedPacket& packet)
{
    if (!packet.publicationProbe) { return false; }
    if (packet.publicationProbe->preserved) { return true; }
    // A single versioned continuation summary includes every active publication,
    // including publications whose symbolic admission signature is Unknown.
    if (publicationReachVersion != ledger.version()) {
        publicationReach.assign(control.graph.sites.size(), false);
        result.work.normalPublicationSites += publicationReach.size();
        std::vector<Cut> pending;
        std::set<Cut> words;
        for (const auto& endpoint : ledger.records()) {
            if (!ledger.active(endpoint.id) || endpoint.command.kind != Command::Publish) { continue; }
            words.insert(control.canonicalCut[endpoint.cut]);
        }
        for (auto word : words) {
            const auto& sites = control.wordOccurrences[word];
            pending.insert(pending.end(), sites.begin(), sites.end());
        }
        while (!pending.empty()) {
            const auto site = pending.back(); pending.pop_back();
            if (!control.reachable[site] || publicationReach[site]) { continue; }
            publicationReach[site] = true;
            ++result.work.normalPublicationSites;
            const auto& previous = control.predecessors[site];
            pending.insert(pending.end(), previous.begin(), previous.end());
        }
        publicationReachVersion = ledger.version();
    }
    const auto view = ledger.packetView(packet.prepared);
    if (!view) { return false; }
    for (auto word : view->changedCuts()) {
        for (auto site : control.wordOccurrences[control.canonicalCut[word]]) {
            if (publicationReach[site]) { return false; }
        }
    }
    return true;
}
bool Constructor::physicalMilestones(const RecurringRequirement& role) const
{
    if (role.physicalQualified) { return *role.physicalQualified; }
    role.physicalQualified = false;
    std::map<Cut, std::set<Cut>> publications, acquisitions;
    for (const auto& origin : role.publicationOrigins) { publications[origin.first].insert(origin.second); }
    for (const auto& origin : role.acquisitionOrigins) { acquisitions[origin.first].insert(origin.second); }
    for (auto word : role.publications) {
        const auto& sources = publications[word];
        if (sources.empty()) { return false; }
        std::vector<Cut> sourceWords;
        for (auto source : sources) {
            const auto operation = control.graph.operations[source];
            const auto& boundary = control.publicationAfter(source);
            if (operation == NoAnalysisId || program.operations[operation].pipe != role.source ||
                !boundary.proved() || boundary.word != word) { return false; }
            sourceWords.push_back(control.canonicalCut[source]);
        }
        // The first command boundary has no earlier selected word in its
        // payload-to-publication corridor. All copies must have physical origins.
        const auto& matches = control.correspondence(sourceWords, {word});
        if (!matches.proved()) { return false; }
        for (const auto& pair : matches.pairs) {
            if (!sources.count(pair.first)) { return false; }
        }
    }
    for (auto word : role.acquisitions) {
        const auto& targets = acquisitions[word];
        for (auto site : control.wordOccurrences[word]) {
            if (!control.reachable[site]) { continue; }
            const auto op = control.graph.operations[site];
            if (op == NoAnalysisId || program.operations[op].pipe != role.observer ||
                !targets.count(site)) { return false; }
        }
    }
    role.physicalQualified = true;
    return true;
}

std::optional<CertifiedRealization> Constructor::normalOrdinary(
    Group group, const std::vector<FrontierRequirement>& due, const std::vector<DueObligation>& universe,
    bool repair, std::optional<WordGap> prescribedGap, Id prescribedKey,
    const SourceGapQualification* prescribedFacts)
{
    if (group.common) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    const auto& word = ledger.word(group.publication);
    const WordGap gap = prescribedGap.value_or(WordGap{control.canonicalCut[group.publication], NoAnalysisId,
        word.empty() ? NoAnalysisId : word.front()});
    const Cut sourceMilestone = group.publication;
    group.publication = gap.cut;
    const auto computed = prescribedFacts ? SourceGapQualification{} :
        sourceGapFacts(gap, group.source, observer, group.requirements);
    const auto& facts = prescribedFacts ? *prescribedFacts : computed;
    if (!facts.proved()) { return {}; }
    CertifiedRealization out;
    out.version = ledger.version();
    out.sourceMilestone = sourceMilestone;
    out.selectedSourceGap = gap;
    for (const auto& r : due) {
        const auto access = accessClass(r);
        bool covered = true;
        for (const auto& prefix : facts.prefixes) {
            const auto* history = prefix.facts()->history.find(access);
            covered &= history && frontierContains(*history, PipeCount + unsigned(group.source));
        }
        for (const auto& pair : control.correspondence(gap.cut, current).pairs) {
            covered &= freshBetween(pair.first, pair.second, access);
        }
        if (covered) { out.physicalCoverage.insert(access); }
    }
    out.coverage = normalizedCoverage(universe, due, out.physicalCoverage);
    out.ownCoverage = normalizedCoverage(universe, due, classes(group.requirements));
    if (out.coverage.empty() || out.ownCoverage.empty()) { return {}; }
    if (repair) {
        if (!fixedBoundaryPacket(facts, group)) { return {}; }
        out.placementClass = 1;
        out.support.push_back({RealizationSupport::Consumption, group.forwardKey, gap.cut});
    }
    for (Id key = 0; !repair && key < frontier.keys().size(); ++key) {
        if (prescribedKey != NoAnalysisId && key != prescribedKey) { continue; }
        if (!sourceGapKey(facts, key)) { continue; }
        const auto& identity = frontier.keys()[key];
        OrderedPacket endpoints{
            {gap.cut, {Command::Publish, group.source, observer, identity.key},
                EndpointPurpose::Completion, result.decisions.size(), NoAnalysisId, gap},
            {current, {Command::Acquire, group.source, observer, identity.key},
                EndpointPurpose::Completion, result.decisions.size()}};
        auto packet = prepareOwnedPacket(endpoints);
        if (!packet || packet->restoredEndpoints || !preservePublications(*packet)) { continue; }
        packet->qualified = true;
        group.packet = std::move(*packet);
        group.forwardKey = key;
        break;
    }
    if (!group.packet) { return {}; }
    group.supporting.clear();
    const auto motivating = classes(group.requirements);
    for (const auto& r : due) {
        if (out.physicalCoverage.count(accessClass(r)) && !motivating.count(accessClass(r))) {
            group.supporting.push_back(r);
        }
    }
    for (auto obligation : out.coverage) {
        out.support.push_back({RealizationSupport::Completion, obligation, current});
    }
    out.order = {control.frame[group.publication], NoAnalysisId - control.position[group.publication],
                 group.publication, group.source, observer, {gap.cut}, {control.canonicalCut[current]}};
    out.shape.push_back({group.source, observer, {gap.cut}, {control.canonicalCut[current]}});
    if (repair) {
        out.shape.push_back({observer, group.source, {gap.cut}, {gap.cut}});
    }
    out.ordinary = std::move(group);
    return out;
}
std::optional<CertifiedRealization> Constructor::normalAlternative(
    const Group& request, const std::vector<FrontierRequirement>& due,
    const std::vector<DueObligation>& universe)
{
    if (!request.common) { return {}; }
    const auto discovered = discoverSourceFrontier(request.source, request.requirements);
    if (!discovered || discovered->publications.empty()) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    const auto motivating = classes(request.requirements);
    struct AlternativePrefix {
        WordGap gap;
        FrontierState state;
    };
    std::vector<AlternativePrefix> prefixes;
    for (auto cut : discovered->publications) {
        auto state = cache.cuts[cut].incoming.causal;
        if (!state.reachable()) { return {}; }
        const auto& word = ledger.word(cut);
        bool found = false;
        for (Id offset = 0; offset <= word.size(); ++offset) {
            const bool covered = std::all_of(motivating.begin(), motivating.end(), [&](Id access) {
                const auto* history = state.facts()->history.find(access);
                return history && frontierContains(*history, PipeCount + unsigned(request.source));
            });
            if (covered) {
                prefixes.push_back({{cut, offset ? word[offset - 1] : NoAnalysisId,
                    offset < word.size() ? word[offset] : NoAnalysisId}, state});
                found = true;
                break;
            }
            if (offset == word.size()) { break; }
            const auto& command = ledger.endpoint(word[offset]).command;
            // Cross an incoming receipt only when it makes a motivating
            // physical completion available. Otherwise an unrelated WAIT
            // would silently broaden this new class-0 publication.
            if (command.kind == Command::BarrierAll) { break; }
            ++result.work.sourceGapCommands;
            const auto step = frontier.command(state, command, {cut, offset});
            if (!step.applied) { return {}; }
            if (command.kind == Command::Acquire) {
                const bool relevant = std::any_of(motivating.begin(), motivating.end(), [&](Id access) {
                    const auto* before = state.facts()->history.find(access);
                    const auto* after = step.state.facts()->history.find(access);
                    const auto hasSource = [&](const auto* history) {
                        return history && frontierContains(*history, PipeCount + unsigned(request.source));
                    };
                    return !hasSource(before) && hasSource(after);
                });
                if (!relevant) { break; }
            }
            state = step.state;
        }
        if (!found) { return {}; }
    }
    std::set<Id> physical;
    for (const auto& r : due) {
        const auto access = accessClass(r);
        if (discovered->crossedClasses.count(access)) { continue; }
        const bool atSources = std::all_of(prefixes.begin(), prefixes.end(), [&](const auto& prefix) {
            const auto* history = prefix.state.facts()->history.find(access);
            return history && frontierContains(*history, PipeCount + unsigned(request.source));
        });
        if (atSources) { physical.insert(access); }
    }
    const auto own = normalizedCoverage(universe, due, motivating);
    const auto covered = normalizedCoverage(universe, due, physical);
    const bool coversOwn = !own.empty() && subset(own, covered);
    if (!coversOwn) { return {}; }
    std::vector<bool> futureSites;
    std::map<Cut, bool> wordIntersects;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != request.source || identity.observer != observer ||
            !helperFreeKey(key)) { continue; }
        if (!ledger.eventUses(identity).empty()) {
            if (futureSites.empty()) {
                futureSites.resize(control.graph.sites.size());
                result.work.normalKeySites += futureSites.size();
                std::vector<Cut> pending = discovered->publications;
                while (!pending.empty()) {
                    const auto site = pending.back(); pending.pop_back();
                    if (!control.reachable[site] || futureSites[site]) { continue; }
                    futureSites[site] = true;
                    ++result.work.normalKeySites;
                    const auto& next = control.graph.sites[site].successors;
                    pending.insert(pending.end(), next.begin(), next.end());
                }
            }
            if (!selectedKeyUsesOutside(futureSites, key, wordIntersects)) { continue; }
        }
        OrderedPacket endpoints;
        bool sourceReady = true;
        for (const auto& prefix : prefixes) {
            State atGap;
            atGap.causal = prefix.state;
            sourceReady &= canPublish(atGap, key);
            endpoints.push_back({prefix.gap.cut,
                {Command::Publish, request.source, observer, identity.key},
                EndpointPurpose::Completion, result.decisions.size(), NoAnalysisId, prefix.gap});
        }
        if (!sourceReady) { continue; }
        endpoints.push_back({current, {Command::Acquire, request.source, observer, identity.key},
            EndpointPurpose::Completion, result.decisions.size(), NoAnalysisId, ledger.tail(current)});
        auto packet = prepareOwnedPacket(endpoints);
        if (!packet || packet->restoredEndpoints || !preservePublications(*packet)) { continue; }
        packet->qualified = true;
        CertifiedRealization out;
        out.version = ledger.version();
        out.physicalCoverage = physical;
        out.coverage = covered;
        out.ownCoverage = own;
        auto group = request;
        group.common = false;
        group.publications = discovered->publications;
        group.publication = *std::min_element(group.publications.begin(), group.publications.end(),
            [&](Cut a, Cut b) { return control.position[a] < control.position[b]; });
        group.forwardKey = key;
        group.version = ledger.version();
        group.packet = std::move(*packet);
        for (const auto& r : due) {
            const auto access = accessClass(r);
            const bool additional = physical.count(access) && !motivating.count(access);
            if (additional) { group.supporting.push_back(r); }
        }
        out.sourceMilestone = group.publication;
        out.order = {control.frame[group.publication], NoAnalysisId - control.position[group.publication],
            group.publication, group.source, observer, group.publications, {control.canonicalCut[current]}};
        out.shape.push_back({group.source, observer, group.publications, {control.canonicalCut[current]}});
        for (auto obligation : covered) {
            out.support.push_back({RealizationSupport::Completion, obligation, current});
        }
        out.ordinary = std::move(group);
        return out;
    }
    return {};
}
void Constructor::indexSelectedReturns()
{
    const auto& endpoints = ledger.records();
    while (indexedReturnEndpoints < endpoints.size()) {
        const auto id = indexedReturnEndpoints++;
        const auto& endpoint = endpoints[id];
        if (endpoint.command.kind != Command::Acquire ||
            (endpoint.purpose != EndpointPurpose::Completion &&
             endpoint.purpose != EndpointPurpose::Fixed) ||
            endpoint.cut >= control.position.size()) { continue; }
        selectedReturnGaps[{endpoint.command.source, endpoint.command.observer}]
            [control.position[endpoint.cut]].push_back(id);
    }
}
std::optional<CertifiedRealization> Constructor::normalCorridor(
    const Group& group, const std::vector<FrontierRequirement>& due,
    const std::vector<DueObligation>& universe)
{
    if (group.common) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    indexSelectedReturns();
    // Skip returns with no actual rearming transition. The first relevant
    // receipt fixes one gap; a failed packet does not start a source search.
    const auto direction = selectedReturnGaps.find({observer, group.source});
    if (direction == selectedReturnGaps.end()) { return {}; }
    const auto first = direction->second.lower_bound(control.position[group.publication]);
    for (auto bucket = first; bucket != direction->second.end(); ++bucket) {
        if (bucket->first > control.position[current]) { break; }
        // Endpoint IDs are append order. The selected word may contain later
        // insertions, so use its actual order for receipts at one position.
        std::set<Id> indexed(bucket->second.begin(), bucket->second.end());
        std::set<Cut> cuts;
        for (auto id : bucket->second) if (ledger.active(id)) {
            cuts.insert(ledger.endpoint(id).cut);
        }
        for (auto cut : cuts) {
            if (!control.straight(group.publication, cut) ||
                !control.straight(cut, current)) { continue; }
            for (auto receipt : ledger.word(cut)) {
                ++result.work.corridorWordEndpoints;
                if (!indexed.count(receipt)) { continue; }
                ++result.work.corridorReceiptScans;
                const auto before = cache.beforeReceiptPublishable.find(receipt);
                const auto after = cache.afterEndpoint.find(receipt);
                if (before == cache.beforeReceiptPublishable.end() ||
                    after == cache.afterEndpoint.end()) { continue; }
                std::vector<Id> transitioned;
                for (Id key = 0; key < frontier.keys().size(); ++key) {
                    const auto& identity = frontier.keys()[key];
                    if (identity.source == group.source && identity.observer == observer &&
                        helperFreeKey(key) && !before->second.count(key) &&
                        canPublish(after->second, key)) {
                        transitioned.push_back(key);
                    }
                }
                if (transitioned.empty()) { continue; }
                const auto gap = ledger.gapAfter(receipt);
                if (!gap) { return {}; }
                const auto facts = sourceGapFacts(*gap, group.source, observer,
                    group.requirements);
                if (!facts.proved()) { return {}; }
                for (auto key : transitioned) {
                    if (!sourceGapKey(facts, key)) { continue; }
                    auto candidate = normalOrdinary(group, due, universe, false, gap,
                        key, &facts);
                    if (!candidate) { continue; }
                    candidate->placementClass = 2;
                    candidate->supportingReceipt = receipt;
                    candidate->support.push_back({RealizationSupport::Consumption, receipt, gap->cut});
                    return candidate;
                }
                return {};
            }
        }
    }
    return {};
}
std::optional<CertifiedRealization> Constructor::normalCommonCut(
    const Group& group, const std::vector<FrontierRequirement>& due,
    const std::vector<DueObligation>& universe)
{
    if (!group.common || !terminalCommonCut()) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    const auto gap = ledger.tail(current);
    const auto facts = sourceGapFacts(gap, group.source, observer, group.requirements);
    if (!facts.proved()) { return {}; }
    const auto offset = ledger.word(gap.cut).size();
    const auto payload = control.graph.operations[current];
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        if (!sourceGapKey(facts, key)) { continue; }
        const auto& identity = frontier.keys()[key];
        const Command publish{Command::Publish, group.source, observer, identity.key};
        const Command acquire{Command::Acquire, group.source, observer, identity.key};
        bool terminal = true;
        for (const auto& prefix : facts.prefixes) {
            const auto sent = frontier.command(prefix, publish, {gap.cut, offset});
            if (!sent.applied) { terminal = false; break; }
            const auto received = frontier.command(sent.state, acquire, {gap.cut, offset + 1});
            if (!received.applied) { terminal = false; break; }
            const auto checked = frontier.inspect(received.state, payload);
            if (!checked.applied && checked.failure != FrontierFailure::Payload) {
                terminal = false;
                break;
            }
            terminal &= std::none_of(checked.residuals.begin(), checked.residuals.end(),
                [&](const auto& r) { return r.source != observer; });
        }
        if (!terminal) { continue; }
        Group ordinary = group;
        ordinary.common = false; // Qualify the same direct packet as ordinary.
        ordinary.publication = gap.cut;
        auto candidate = normalOrdinary(ordinary, due, universe, false, gap, key, &facts);
        if (!candidate) { continue; }
        candidate->placementClass = 3;
        candidate->ordinary.common = true;
        return candidate;
    }
    return {};
}
std::shared_ptr<const Constructor::SupportClosure> Constructor::normalSupport(Id id)
{
    if (normalSupportVersion != ledger.version()) {
        normalClosures.clear();
        normalSupportVersion = ledger.version();
    }
    const auto cached = normalClosures.find(id);
    if (cached != normalClosures.end()) { return cached->second; }
    auto out = std::make_shared<SupportClosure>();
    std::set<Id> closure, roles;
    std::vector<Id> pending{id};
    std::map<Id, std::vector<Id>> previous;
    std::set<Cut> consumers;
    while (!pending.empty()) {
        const auto member = pending.back(); pending.pop_back();
        if (!closure.insert(member).second) { continue; }
        const auto& family = recurringFrontiers.families[member];
        roles.insert(family.roles.begin(), family.roles.end());
        const auto& links = recurringSupportLinks(member);
        if (!links.complete) { normalClosures.emplace(id, out); return out; }
        const auto& scope = recurringScopes.at(member);
        for (unsigned pipe = 0; pipe < PipeCount; ++pipe) {
            out->scope.classes[pipe].insert(scope.classes[pipe].begin(), scope.classes[pipe].end());
            out->scope.seeds[pipe].insert(out->scope.seeds[pipe].end(),
                scope.seeds[pipe].begin(), scope.seeds[pipe].end());
        }
        out->scope.ordinary.insert(out->scope.ordinary.end(),
            links.ordinary.begin(), links.ordinary.end());
        consumers.insert(scope.consumers.begin(), scope.consumers.end());
        for (auto next : links.families) { previous[next].push_back(member); pending.push_back(next); }
    }
    out->complete = true;
    out->families.assign(closure.begin(), closure.end());
    out->roles.assign(roles.begin(), roles.end());
    out->scope.consumers.assign(consumers.begin(), consumers.end());
    for (auto& seeds : out->scope.seeds) {
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    }
    std::sort(out->scope.ordinary.begin(), out->scope.ordinary.end());
    out->scope.ordinary.erase(std::unique(out->scope.ordinary.begin(), out->scope.ordinary.end(),
        [](const auto& a, const auto& b) { return !(a < b) && !(b < a); }), out->scope.ordinary.end());
    // Roots mutually reachable from this root have exactly the same finite
    // closure. Share its representation and its eventual probe, not F copies.
    std::set<Id> equivalent;
    pending = {id};
    while (!pending.empty()) {
        const auto member = pending.back(); pending.pop_back();
        if (!equivalent.insert(member).second) { continue; }
        normalClosures.emplace(member, out);
        const auto& before = previous[member];
        pending.insert(pending.end(), before.begin(), before.end());
    }
    return out;
}
std::optional<CertifiedRealization> Constructor::normalRecurring(
    const std::vector<Id>& roots, const std::vector<FrontierRequirement>& due,
    const std::vector<DueObligation>& universe)
{
    const auto id = roots.front();
    if (activeFamilies[id]) { return {}; }
    auto closure = std::make_shared<SupportClosure>();
    std::set<const SupportClosure*> visited;
    std::set<Id> families;
    std::set<Cut> consumers;
    for (auto root : roots) {
        const auto support = normalSupport(root);
        if (!visited.insert(support.get()).second) {
            continue;
        }
        if (!support->complete) { return {}; }
        if (closure->roles.empty()) { closure->roles = support->roles; }
        if (closure->roles != support->roles) { return {}; }
        families.insert(support->families.begin(), support->families.end());
        consumers.insert(support->scope.consumers.begin(), support->scope.consumers.end());
        for (unsigned pipe=0; pipe<PipeCount; ++pipe) {
            closure->scope.classes[pipe].insert(support->scope.classes[pipe].begin(),
                                               support->scope.classes[pipe].end());
            closure->scope.seeds[pipe].insert(closure->scope.seeds[pipe].end(),
                support->scope.seeds[pipe].begin(), support->scope.seeds[pipe].end());
        }
        closure->scope.ordinary.insert(closure->scope.ordinary.end(),
            support->scope.ordinary.begin(), support->scope.ordinary.end());
    }
    // Exact COMPLETE role identity collects independent physical witnesses.
    // Sharing only one endpoint/role never joins these proof populations.
    closure->families.assign(families.begin(),families.end());
    closure->scope.consumers.assign(consumers.begin(),consumers.end());
    for (auto& seeds : closure->scope.seeds) {
        std::sort(seeds.begin(), seeds.end());
        seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
    }
    std::sort(closure->scope.ordinary.begin(), closure->scope.ordinary.end());
    closure->scope.ordinary.erase(std::unique(closure->scope.ordinary.begin(), closure->scope.ordinary.end(),
        [](const auto& a, const auto& b) { return !(a < b) && !(b < a); }), closure->scope.ordinary.end());
    CertifiedRealization out;
    out.version = ledger.version();
    out.families = closure->families;
    std::vector<RecurringRequirement> requests;
    for (auto role : closure->roles) {
        if (activeRoles.count(role)) { continue; }
        if (!physicalMilestones(recurringFrontiers.roles[role])) { return {}; }
        out.newRoles.push_back(role);
        requests.push_back(recurringFrontiers.roles[role]);
        out.support.push_back({RealizationSupport::Consumption, role, current});
    }
    if (requests.empty()) { return {}; }
    for (auto member : closure->families) { out.support.push_back({RealizationSupport::Induction, member, current}); }
    if (!closure->scope.ordinary.empty()) { out.placementClass = 1; }
    for (const auto& row : closure->scope.ordinary) {
        out.support.push_back({RealizationSupport::OrdinaryRepair, row.access, row.consumer});
    }
    std::string reason;
    auto packet = prepareRecurring(requests, reason, &closure->scope, &out.families, true);
    if (!packet || !packet->localCertificate || !preservePublications(packet->packet)) { return {}; }
    for (Id i = 0; i < requests.size(); ++i) {
        for (auto cut : requests[i].publications) {
            bool incoming = false;
            for (auto endpoint : ledger.word(cut)) {
                const auto& command = ledger.endpoint(endpoint).command;
                const auto& key = frontier.keys()[packet->keys[i]];
                const bool selectedSource = command.kind == Command::Publish && command.source == key.source &&
                    command.observer == key.observer && command.key == key.key;
                if (selectedSource && incoming) { return {}; }
                incoming |= command.kind != Command::Publish;
            }
        }
    }
    out.physicalCoverage = packet->guaranteed[current];
    std::set<Id> own, motivatingRoles;
    const std::set<Id> triggering(roots.begin(),roots.end());
    // A closure is one prescribed realization. Its participating roots are the
    // requests selecting that closure; support-only improvement is insufficient.
    for (const auto& r : due) {
        if (!out.physicalCoverage.count(accessClass(r))) {
            continue;
        }
        const auto indexed = recurringFrontiers.at.find({control.canonicalCut[current],r.cell});
        if (indexed == recurringFrontiers.at.end()) { continue; }
        for (auto root : indexed->second) {
            if (!triggering.count(root) || !supportsRecurring(root, current, r)) {
                continue;
            }
            own.insert(accessClass(r));
            const auto observer = program.operations[control.graph.operations[current]].pipe;
            for (auto roleId : recurringFrontiers.families[root].roles) {
                const auto& role = recurringFrontiers.roles[roleId];
                const bool incoming = role.observer == observer &&
                    (role.source == r.source || (r.source == observer && r.sourceWrite && r.consumerWrite));
                if (incoming) { motivatingRoles.insert(roleId); }
            }
        }
    }
    out.ownCoverage = normalizedCoverage(universe, due, own);
    if (out.ownCoverage.empty()) { return {}; }
    out.coverage = normalizedCoverage(universe, due, out.physicalCoverage);
    if (out.coverage.empty()) { return {}; }
    if (motivatingRoles.empty()) { return {}; }
    bool first = true;
    for (auto roleId : motivatingRoles) {
        const auto& role = recurringFrontiers.roles[roleId];
        const auto source = role.publications.front();
        const bool comparable = control.straight(source, current) &&
            !control.components[control.component[source]].cyclic;
        const CertifiedRealization::Order order{
            comparable ? control.frame[source] : NoAnalysisId,
            comparable ? NoAnalysisId - control.position[source] : role.owner,
            source, role.source, role.observer, role.publications, role.acquisitions};
        if (first || order < out.order) { out.order = order; first = false; }
    }
    for (auto id : closure->roles) {
        const auto& role = recurringFrontiers.roles[id];
        out.shape.push_back({role.source, role.observer, role.publications, role.acquisitions});
    }
    std::set<std::tuple<Pipe, Cut, std::shared_ptr<const std::vector<Cut>>>> ordinaryShapes;
    for (const auto& row : closure->scope.ordinary) {
        ordinaryShapes.emplace(row.source, row.consumer, row.seeds);
    }
    for (const auto& [pipe, consumer, seeds] : ordinaryShapes) {
        out.shape.push_back({pipe, pipe, *seeds, {consumer}});
    }
    std::sort(out.shape.begin(), out.shape.end());
    out.shape.erase(std::unique(out.shape.begin(), out.shape.end()), out.shape.end());
    out.recurring = std::move(*packet);
    return out;
}
std::optional<bool> Constructor::selectNormal(const std::vector<DueObligation>& universe)
{
    const auto due = residual();
    if (due.empty()) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    const auto labels = reasons(current);
    std::map<std::pair<Pipe, bool>, std::vector<FrontierRequirement>> demands;
    std::set<Id> known, families;
    for (const auto& r : due) {
        const auto found = labels.find(accessClass(r));
        const bool isKnown = found != labels.end() && (found->second & (KnownReadiness | KnownReuse));
        if (isKnown) {
            known.insert(accessClass(r));
        }
        if (r.source != observer) { demands[{r.source, isKnown}].push_back(r); }
        const auto indexed = recurringFrontiers.at.find({control.canonicalCut[current], r.cell});
        if (indexed == recurringFrontiers.at.end()) { continue; }
        for (auto family : indexed->second) {
            if (supportsRecurring(family, current, r)) {
                families.insert(family);
            }
        }
    }
    const auto normalizedKnown = normalizedCoverage(universe,due,known);
    std::vector<CertifiedRealization> candidates;
    auto retain = [&](std::optional<CertifiedRealization> candidate) {
        if (!candidate) { return; }
        candidate->known = std::any_of(normalizedKnown.begin(), normalizedKnown.end(), [&](Id obligation) {
            return candidate->coverage.count(obligation) != 0;
        });
        if (candidate->placementClass == 0) { ++result.work.normalCandidates; }
        else { ++result.work.repairCandidates; }
        candidates.push_back(std::move(*candidate));
    };
    std::vector<Group> ordinaryRequests;
    for (const auto& demand : demands) {
        ordinaryRequests.push_back(sourceGroup(demand.first.first, demand.second, due, false));
        retain(normalOrdinary(ordinaryRequests.back(), due, universe));
        retain(normalAlternative(ordinaryRequests.back(), due, universe));
    }
    std::map<std::vector<Id>, std::vector<Id>> roots;
    for (auto family : families) {
        if (activeFamilies[family]) { continue; }
        const auto support = normalSupport(family);
        if (support->complete) { roots[support->roles].push_back(family); }
    }
    for (const auto& entry : roots) {
        // One probe for a canonical finite packet; every triggering root is
        // retained for its own-obligation check below.
        auto candidate = normalRecurring(entry.second, due, universe);
        if (candidate) {
            retain(std::move(candidate));
        }
    }
    const bool hasNormal = std::any_of(candidates.begin(), candidates.end(), [](const auto& candidate) {
        return candidate.placementClass == 0;
    });
    if (!hasNormal) {
        for (const auto& request : ordinaryRequests) {
            retain(normalOrdinary(request, due, universe, true));
        }
    }
    const bool hasPreferred = std::any_of(candidates.begin(), candidates.end(),
        [](const auto& candidate) { return candidate.placementClass <= 1; });
    if (!hasPreferred) {
        for (const auto& request : ordinaryRequests) {
            retain(normalCorridor(request, due, universe));
        }
    }
    const bool terminalCandidate = candidates.empty() && terminalCommonCut();
    if (terminalCandidate) {
        // A common-cut packet for one source can also cover another source's
        // requirement. Preserve ANY qualified earlier source at this deadline
        // until the structured clients enter the same local binding policy.
        const bool earlierSource = std::any_of(ordinaryRequests.begin(), ordinaryRequests.end(),
            [&](const Group& request) {
                return discoverSourceFrontier(request.source, request.requirements) ||
                    earlierLoopEntrySource(request.source, request.requirements);
            });
        if (!earlierSource) {
            for (const auto& request : ordinaryRequests) {
                retain(normalCommonCut(request, due, universe));
            }
        }
    }
    const auto winner = selectRealization(candidates);
    if (winner == NoAnalysisId) { return {}; }
    auto& selected = candidates[winner];
    if (selected.version != ledger.version()) {
        return fail(SelectedFailure::SelectedUpdate, "realization changed during private selection", current);
    }
    SelectedRealizationChoice choice;
    choice.deadline = current;
    choice.placementClass = selected.placementClass;
    choice.recurring = bool(selected.recurring);
    choice.known = selected.known;
    choice.covered = selected.coverage.size();
    choice.inputVersion = ledger.version();
    for (const auto& candidate : candidates) {
        if (candidate.recurring) { ++choice.recurringCandidates; }
        else { ++choice.ordinaryCandidates; }
    }
    if (selected.recurring) {
        const auto first = result.channels.size();
        if (!commitRecurring(*selected.recurring)) { return false; }
        for (Id i = 0; i < selected.newRoles.size(); ++i) { activeRoles.emplace(selected.newRoles[i], first + i); }
        for (auto family : selected.families) { activeFamilies[family] = true; }
        ++result.work.recurringActivations;
        ++result.work.normalRecurringSelected;
        result.activations.push_back({current, selected.families, due, residual(), ledger.version()});
    } else {
        const auto& group = selected.ordinary;
        SelectedDecision decision;
        decision.consumer = current;
        decision.publication = group.publication;
        decision.publicationFrontier = group.publications;
        decision.commonCut = group.common;
        decision.sourceMilestone = selected.sourceMilestone;
        decision.publicationGapLeft = selected.selectedSourceGap.left;
        decision.publicationGapRight = selected.selectedSourceGap.right;
        decision.supportingReceipt = selected.supportingReceipt;
        decision.enlargedPrefix = selected.placementClass == 2;
        decision.source = group.source;
        decision.observer = observer;
        decision.stage = selected.known ? RequirementStage::Known : RequirementStage::Overlap;
        decision.required = group.requirements;
        decision.supporting = group.supporting;
        decision.lifecycles = requirements.demandsAt(current, group.requirements);
        if (group.repairKey != NoAnalysisId) {
            decision.repairedForwardKey = frontier.keys()[group.forwardKey].key;
            decision.repairReverseKey = frontier.keys()[group.repairKey].key;
            decision.repairInputVersion = ledger.version();
        }
        const auto oldWord = ledger.word(group.publication);
        if (!commitOwnedPacket(*group.packet, decision) || !update() || !settleRearming(decision)) {
            return false;
        }
        const bool priorIncoming = std::any_of(oldWord.begin(), oldWord.end(), [&](Id id) {
            const auto& c = ledger.endpoint(id).command;
            return c.kind == Command::Acquire && c.observer == group.source;
        });
        const bool newEarlyPublication = group.publications.empty() &&
            selected.selectedSourceGap.left == NoAnalysisId && priorIncoming;
        if (newEarlyPublication) {
            ++result.work.earlyPublications;
        }
        if (group.repairKey != NoAnalysisId) {
            decision.repairOutputVersion = ledger.version();
            ++result.work.acknowledgments;
            ++result.work.joinedAcknowledgments;
        }
        if (decision.commonCut) { ++result.work.commonCutTransfers; }
        result.decisions.push_back(std::move(decision));
    }
    auto predicted = classes(due);
    for (auto access : selected.physicalCoverage) { predicted.erase(access); }
    if (!subset(classes(residual()), predicted)) {
        return fail(SelectedFailure::SelectedUpdate, "actual realization lost certified coverage", current);
    }
    choice.outputVersion = ledger.version();
    result.realizationChoices.push_back(choice);
    if (selected.placementClass == 0) { ++result.work.normalSelected; }
    else { ++result.work.repairSelected; }
    return true;
}
} // namespace mlir::pto::oahs::selected
