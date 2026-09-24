// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>

namespace mlir::pto::oahs::selected {
namespace {
bool properSubset(const std::set<Id>& a, const std::set<Id>& b)
{
    return a.size() < b.size() && std::includes(b.begin(), b.end(), a.begin(), a.end());
}
} // namespace
std::map<Id, unsigned> Constructor::reasons(Cut site) const
{
    return requirements.reasons(site);
}
std::set<Id> Constructor::sourceHistoryCoverage(
    const std::vector<Cut>& sources, Pipe source,
    const std::vector<FrontierRequirement>& required) const
{
    std::set<Id> out;
    if (sources.empty()) {
        return out;
    }
    for (const auto& r : required) {
        out.insert(accessClass(r));
    }
    for (auto cut : sources) {
        const auto& state = cache.cuts[cut].before.causal;
        if (!state.reachable()) {
            return {};
        }
        for (auto it = out.begin(); it != out.end();) {
            const auto* history = state.facts()->history.find(*it);
            if (!history || !frontierContains(*history, PipeCount + unsigned(source))) {
                it = out.erase(it);
            } else {
                ++it;
            }
        }
    }
    return out;
}
std::set<Id> Constructor::coverage(
    Cut cut, Pipe source, const std::vector<FrontierRequirement>& required) const
{
    auto out = sourceHistoryCoverage({cut}, source, required);
    for (auto it = out.begin(); it != out.end();) {
        if (!freshBetween(cut, current, *it)) {
            it = out.erase(it);
        } else {
            ++it;
        }
    }
    return out;
}
bool Constructor::freshBetween(Cut source, Cut target, Id access) const
{
    if (source == target) return true;
    if (!control.straight(source, target)) return false;
    return !control.lookahead.hasIssueBetween(
        control.frame[source], access, control.position[source], control.position[target]);
}
std::optional<Constructor::SourceFrontierFacts> Constructor::discoverSourceFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required)
{
    // This extension is intentionally acyclic: static predecessor identities
    // are not a bank-generation correspondence across an unqualified loop.
    if (sourceFrontierVersion != cache.version) {
        sourceFrontierCache.clear();
        sourceFrontierVersion = cache.version;
    }
    std::set<Id> needed;
    for (const auto& requirement : required) { needed.insert(accessClass(requirement)); }
    const SourceFrontierKey key{cache.version, current, source, needed};
    const auto previous = sourceFrontierCache.find(key);
    if (previous != sourceFrontierCache.end()) { return previous->second; }
    auto reject = [&]() -> std::optional<SourceFrontierFacts> {
        sourceFrontierCache.emplace(key, std::nullopt);
        return {};
    };
    if (required.empty() || control.components[activeComponent].cyclic ||
        control.canonicalCut[current] != current) { return reject(); }
    auto uniqueWord = [&](Cut cut) {
        const bool canonical = cut < control.canonicalCut.size() && control.canonicalCut[cut] == cut;
        if (!canonical) { return false; }
        const auto& occurrences = control.wordOccurrences[cut];
        return std::count_if(occurrences.begin(), occurrences.end(),
                            [&](Id site) { return control.reachable[site]; }) == 1;
    };
    if (!uniqueWord(current)) { return reject(); }
    std::map<Id, const SelectedSource*> availableSources;
    for (const auto& handle : result.sources) {
        if (handle.pipe == source && handle.version == cache.version && handle.snapshot.reachable()) {
            availableSources.emplace(handle.origin, &handle);
        }
    }
    std::set<Cut> publications;
    std::set<Id> crossedClasses;
    std::vector<bool> seen(control.graph.sites.size());
    auto todo = control.predecessors[current];
    while (!todo.empty()) {
        const auto site = todo.back(); todo.pop_back();
        if (!control.reachable[site] || seen[site]) { continue; }
        seen[site] = true;
        const auto component = control.component[site];
        if (component == NoAnalysisId || control.components[component].cyclic) { return reject(); }
        const auto operation = control.graph.operations[site];
        bool touchesRequiredClass = false;
        if (operation != NoAnalysisId) {
            const auto& op = program.operations[operation];
            for (const auto& access : op.accesses) {
                const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                touchesRequiredClass |= (access.read && needed.count(base)) ||
                                        (access.write && needed.count(base + 1));
            }
        }
        if (touchesRequiredClass) {
            const auto found = availableSources.find(site);
            if (found == availableSources.end()) { return reject(); }
            const auto& handle = *found->second;
            if (handle.cut == current || !uniqueWord(handle.cut)) { return reject(); }
            const auto sourceComponent = control.component[handle.cut];
            if (sourceComponent == NoAnalysisId || control.components[sourceComponent].cyclic) { return reject(); }
            const auto& snapshot = cache.cuts[handle.cut].before.causal;
            if (!snapshot.reachable()) { return reject(); }
            // Do not assume that an absent class on one arm means an optional
            // corresponding producer. That case needs a separate qualifier.
            for (auto access : needed) {
                const auto* history = snapshot.facts()->history.find(access);
                if (!history || !frontierContains(*history, PipeCount + unsigned(source))) { return reject(); }
            }
            publications.insert(handle.cut);
            continue;
        }
        // This is the existing source-discovery traversal. Retain all classes
        // issued strictly after its stopping publications, not a path per class.
        if (operation != NoAnalysisId) {
            const auto& op = program.operations[operation];
            for (const auto& access : op.accesses) {
                const auto base = (Id(access.cell) * PipeCount + unsigned(op.pipe)) * 2;
                if (access.read) { crossedClasses.insert(base); }
                if (access.write) { crossedClasses.insert(base + 1); }
            }
        }
        if (site == control.graph.entry || control.predecessors[site].empty()) { return reject(); }
        const auto& before = control.predecessors[site];
        todo.insert(todo.end(), before.begin(), before.end());
    }
    std::vector<Cut> cuts(publications.begin(), publications.end());
    if (!control.correspondence(cuts, std::vector<Cut>{current}).proved()) {
        return reject();
    }
    SourceFrontierFacts facts{std::move(cuts), std::move(crossedClasses)};
    sourceFrontierCache.emplace(key, facts);
    return facts;
}

bool Constructor::sourceFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all, Group& group)
{
    const auto discovered = discoverSourceFrontier(source, required);
    if (!discovered) { return false; }
    const auto& cuts = discovered->publications;
    // Retain every original publication. Virgin keys use their local source
    // certificate; dormant ownership submits the same complete packet.
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    Id selected = NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != source || identity.observer != observer || closedKeys.count(key)) continue;
        const bool existing = !ledger.eventUses(identity).empty();
        if (!existing) {
            if (!std::all_of(cuts.begin(), cuts.end(), [&](Cut cut) {
                    return canPublish(cache.cuts[cut].before, key);
                })) { continue; }
        }
        OrderedPacket packet;
        for (auto cut : cuts) {
            packet.push_back({cut, {Command::Publish, source, observer, identity.key},
                EndpointPurpose::Completion, result.decisions.size()});
        }
        packet.push_back({current, {Command::Acquire, source, observer, identity.key},
            EndpointPurpose::Completion, result.decisions.size()});
        const auto qualified = qualifyOwnedPacket(packet, existing);
        if (!qualified) { continue; }
        group.packet = *qualified;
        selected = key;
        break;
    }
    if (selected == NoAnalysisId) return false;
    group.publications = cuts;
    group.publication = *std::min_element(group.publications.begin(), group.publications.end(),
        [&](Cut a, Cut b) { return control.position[a] < control.position[b]; });
    group.forwardKey = selected;
    group.version = cache.version;
    group.common = false;
    group.coverage = sourceHistoryCoverage(group.publications, source, all);
    for (auto access : discovered->crossedClasses) {
        group.coverage.erase(access);
    } // Coverage is opportunity metadata until an actual receipt propagates.
    return true;
}
bool Constructor::earlierLoopEntrySource(
    Pipe source, const std::vector<FrontierRequirement>& required) const
{
    if (!program.observed || required.empty()) { return false; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    for (const auto& loop : control.loopEntries) {
        const auto& first = loop.firstConsumers[unsigned(observer)];
        const bool containsDeadline =
            std::find(loop.sites.begin(), loop.sites.end(), current) != loop.sites.end();
        const bool eligible = !first.empty() && control.canonicalCut[loop.entry] == loop.entry &&
            containsDeadline;
        if (!eligible) { continue; }
        if (!std::all_of(first.begin(), first.end(), [&](Cut cut) {
            return std::all_of(required.begin(), required.end(), [&](const auto& r) {
                const auto roles = requirements.use(cut, r.cell).roles;
                return (roles & 2) || ((roles & 1) && r.sourceWrite);
            });
        })) { continue; }
        // This only retains an existing earlier option for the structured
        // binder. Its key and complete packet are still checked there.
        if (std::any_of(result.sources.begin(), result.sources.end(), [&](const auto& handle) {
            return handle.pipe == source && handle.version == cache.version &&
                handle.snapshot.reachable() && control.straight(handle.cut, loop.entry);
        })) { return true; }
        if (!loop.issuedPipes.count(source)) { return true; }
    }
    return false;
}

std::optional<Constructor::EntrySourceFact> Constructor::nextEntrySource(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::set<Id>& needed, Id& cursor) const
{
    const bool available = program.observed && !required.empty() &&
        current < control.loopEntriesAtSite.size();
    if (!available) { return {}; }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    const auto& containing = control.loopEntriesAtSite[current];
    while (cursor < containing.size()) {
        const auto& loop = control.loopEntries[containing[cursor++]];
        const auto& first = loop.firstConsumers[unsigned(observer)];
        const bool eligible = !first.empty() && control.canonicalCut[loop.entry] == loop.entry;
        if (!eligible) { continue; }
        const bool sameRequirements = std::all_of(first.begin(), first.end(), [&](Cut cut) {
            return std::all_of(required.begin(), required.end(), [&](const auto& r) {
                const auto roles = requirements.use(cut, r.cell).roles;
                return (roles & 2) || ((roles & 1) && r.sourceWrite);
            });
        });
        if (!sameRequirements) { continue; }
        const bool communicates = std::any_of(
            loop.crossedWords[unsigned(observer)].begin(),
            loop.crossedWords[unsigned(observer)].end(), [&](Cut cut) {
                return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                    const auto& command = ledger.endpoint(id).command;
                    return command.kind == Command::BarrierAll ||
                        (command.kind == Command::Publish && command.source == observer);
                });
            });
        if (communicates) { continue; }
        const bool refreshed = std::any_of(needed.begin(), needed.end(), [&](Id access) {
            return loop.issuedClasses.count(access) != 0;
        });
        if (refreshed) { continue; }
        const SelectedSource* selected = nullptr;
        for (const auto& handle : result.sources) {
            if (handle.pipe != source || handle.version != cache.version || !handle.snapshot.reachable() ||
                !control.straight(handle.cut, loop.entry)) { continue; }
            bool covered = true;
            const auto& snapshot = cache.cuts[handle.cut].before.causal;
            if (!snapshot.reachable()) { continue; }
            for (auto access : needed) {
                const auto* history = snapshot.facts()->history.find(access);
                covered &= freshBetween(handle.cut, loop.entry, access) && history &&
                    frontierContains(*history, PipeCount + unsigned(source));
            }
            if (covered && (!selected || control.position[handle.cut] < control.position[selected->cut])) {
                selected = &handle;
            }
        }
        const bool regional = !selected && !loop.issuedPipes.count(source) &&
            std::none_of(loop.sites.begin(), loop.sites.end(), [&](Cut cut) {
                return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                    const auto& c = ledger.endpoint(id).command;
                    return c.kind == Command::BarrierAll || c.source == source ||
                        ((c.kind == Command::Publish || c.kind == Command::Acquire) && c.observer == source);
                });
            });
        if (!selected && !regional) { continue; }
        const auto publication = selected ? selected->cut : loop.entry;
        if (!regional && !control.correspondence(publication, loop.entry).proved()) { continue; }
        return EntrySourceFact{&loop, publication, regional};
    }
    return {};
}
bool Constructor::loopEntryFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all, Group& group)
{
    if (!program.observed || required.empty()) { return false; }
    std::set<Id> needed;
    for (const auto& r : required) { needed.insert(accessClass(r)); }
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    Id cursor = 0;
    while (auto fact = nextEntrySource(source, required, needed, cursor)) {
        const auto& loop = *fact->loop;
        const auto publication = fact->publication;
        const bool regional = fact->regional;
        auto candidates = [&](Pipe a, Pipe b) {
            std::vector<Id> keys;
            for (Id key = 0; key < frontier.keys().size(); ++key) {
                const auto& e = frontier.keys()[key];
                const bool eligible = e.source == a && e.observer == b &&
                    !closedKeys.count(key) && !recurringKeys.count(key);
                if (eligible) {
                    keys.push_back(key);
                }
            }
            return keys;
        };
        const auto forwardKeys = candidates(source, observer);
        const auto reverseKeys = candidates(observer, source);
        const bool repeats = control.components[control.component[publication]].cyclic;
        Id forward = NoAnalysisId, reverse = NoAnalysisId;
        std::optional<OwnedPacket> selectedPacket;
        const auto request = result.decisions.size();
        // Freeze the source/entry shape. Every helper-free key gets its one
        // complete probe before any candidate restores or adds a helper.
        std::map<Id, bool> missingConsumption;
        for (bool repair : {false, true}) {
            for (auto key : forwardKeys) {
                if (!repair && !helperFreeKey(key)) { continue; }
                const auto number = frontier.keys()[key].key;
                OrderedPacket packet{
                    {publication, {Command::Publish, source, observer, number},
                        EndpointPurpose::Completion, request},
                    {loop.entry, {Command::Acquire, source, observer, number},
                        EndpointPurpose::Completion, request}};
                auto probed = missingConsumption.find(key);
                if (probed == missingConsumption.end()) {
                    AnalysisResult trial;
                    const auto beforeSites = result.work.ownershipCheckSites;
                    auto prepared = qualifyOwnedPacket(packet, true, {}, &trial);
                    result.work.loopEntryAnalysisSites += result.work.ownershipCheckSites - beforeSites;
                    if (prepared) {
                        forward = key;
                        selectedPacket = std::move(*prepared);
                        break;
                    }
                    const bool needsConsumption = std::any_of(trial.protocol.begin(), trial.protocol.end(),
                        [&](const auto& r) {
                            return r.kind == ProtocolObligation::ConsumptionNotEstablished &&
                                r.event.source == source && r.event.observer == observer && r.event.key == number;
                        });
                    const bool repairable = trial.complete && trial.diagnostics.empty() &&
                        repeats && needsConsumption;
                    probed = missingConsumption.emplace(key, repairable).first;
                }
                // The failed helper-free packet is not replayed a second time.
                // Only its actual missing-consumption result admits the fixed repair.
                if (!repair || !probed->second) { continue; }
                for (auto reply : reverseKeys) {
                    auto complete = packet;
                    const auto replyNumber = frontier.keys()[reply].key;
                    complete.push_back({loop.entry, {Command::Publish, observer, source, replyNumber},
                        EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
                    complete.push_back({loop.entry, {Command::Acquire, observer, source, replyNumber},
                        EndpointPurpose::ConsumptionAcknowledgment, request, NoAnalysisId, {}, 1});
                    const auto beforeSites = result.work.ownershipCheckSites;
                    auto qualified = qualifyOwnedPacket(complete, true);
                    result.work.loopEntryAnalysisSites += result.work.ownershipCheckSites - beforeSites;
                    if (!qualified) { continue; }
                    forward = key;
                    reverse = reply;
                    selectedPacket = std::move(*qualified);
                    break;
                }
                if (selectedPacket) { break; }
            }
            if (selectedPacket) { break; }
        }
        if (!selectedPacket) { continue; }
        group.publication = publication;
        group.publications = {publication};
        group.entryAcquisition = loop.entry;
        group.entryReturnKey = reverse;
        group.packet = std::move(*selectedPacket);
        group.entryRepeats = repeats;
        group.forwardKey = forward;
        group.version = ledger.version();
        std::vector<Cut> sources;
        for (auto cut : control.wordOccurrences[control.canonicalCut[publication]]) {
            if (control.reachable[cut]) { sources.push_back(cut); }
        }
        group.coverage = sourceHistoryCoverage(sources, source, all);
        for (auto it = group.coverage.begin(); it != group.coverage.end();) {
            bool fresh = !loop.issuedClasses.count(*it);
            if (!regional) {
                const auto& pairs = control.correspondence(publication, loop.entry).pairs;
                for (const auto& pair : pairs) {
                    fresh &= freshBetween(pair.first, pair.second, *it);
                }
            }
            if (!fresh) {
                it = group.coverage.erase(it);
            } else {
                ++it;
            }
        }
        // Retain motivating obligations already qualified above. Extra classes
        // require all-occurrence history plus unchanged entry-to-deadline use.
        group.coverage.insert(needed.begin(), needed.end());
        return true;
    }
    return false;
}
Group Constructor::sourceGroup(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all, bool structured)
{
    Group group;
    group.source = source;
    group.requirements = required;
    std::set<Id> needed;
    for (const auto& requirement : required) needed.insert(accessClass(requirement));
    const SelectedSource* selected = nullptr;
    for (const auto& handle : result.sources) {
        if (handle.pipe != source || handle.version != cache.version || !handle.snapshot.reachable() ||
            !control.straight(handle.cut, current)) continue;
        const auto covered = coverage(handle.cut, source, required);
        if (!std::includes(covered.begin(), covered.end(), needed.begin(), needed.end())) continue;
        if (!selected || control.position[handle.cut] < control.position[selected->cut]) selected = &handle;
    }
    const bool comparable = selected != nullptr;
    if (structured && !comparable && sourceFrontier(source, required, all, group)) { return group; }
    if (structured && !comparable && loopEntryFrontier(source, required, all, group)) { return group; }
    group.publication = comparable ? selected->cut : current;
    group.common = !comparable;
    group.coverage = coverage(group.publication, source, all);
    return group;
}
std::vector<Group> Constructor::groups(
    const std::vector<FrontierRequirement>& all, RequirementStage stage)
{
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    if (std::none_of(all.begin(), all.end(), [&](const auto& r) { return r.source != observer; })) {
        return {};
    }
    const auto labels = stage == RequirementStage::Known ? reasons(current) : std::map<Id, unsigned>{};
    std::map<Pipe, std::vector<FrontierRequirement>> sources, overlapSources;
    std::set<Id> known;
    for (const auto& r : all) {
        const auto found = labels.find(accessClass(r));
        const auto flags = found == labels.end() ? unsigned(AdditionalOverlap) : found->second;
        if (r.source == observer) { continue; }
        if (stage == RequirementStage::Known && !(flags & (KnownReadiness | KnownReuse))) {
            overlapSources[r.source].push_back(r);
            continue;
        }
        sources[r.source].push_back(r);
        known.insert(accessClass(r));
    }
    std::vector<Group> pending, ordered;
    for (const auto& source : sources) {
        pending.push_back(sourceGroup(source.first, source.second, all));
    }
    // An independently required provider may already carry Known completion.
    // Preserve each existing Known provider's early source: never union its
    // motivating demand with later same-engine overlap demands.
    if (!known.empty()) {
        for (const auto& source : overlapSources) {
            if (sources.count(source.first)) { continue; }
            auto group = sourceGroup(source.first, source.second, all);
            for (const auto& r : all) {
                const bool coversKnown = known.count(accessClass(r)) && group.coverage.count(accessClass(r));
                if (coversKnown) {
                    group.supporting.push_back(r);
                }
            }
            if (!group.supporting.empty()) { pending.push_back(std::move(group)); }
        }
    }
    if (!pending.empty()) {
        Id selected = NoAnalysisId;
        for (Id i = 0; i < pending.size(); ++i) {
            bool maximal = true;
            for (const auto& other : pending) {
                maximal &= !properSubset(pending[i].coverage, other.coverage);
            }
            if (!maximal) {
                continue;
            }
            if (selected == NoAnalysisId) {
                selected = i;
                continue;
            }
            const auto a = pending[i].publication, b = pending[selected].publication;
            const bool commonFrame = control.straight(a, b) || control.straight(b, a);
            if ((commonFrame && control.position[a] > control.position[b]) ||
                ((!commonFrame || a == b) && std::make_pair(a, pending[i].source) <
                                                std::make_pair(b, pending[selected].source))) {
                selected = i;
            }
        }
        auto& winner = pending[selected];
        std::set<Id> motivating;
        for (const auto& r : winner.requirements) {
            motivating.insert(accessClass(r));
        }
        winner.supporting.clear();
        for (const auto& r : all) {
            const auto access = accessClass(r);
            const bool additional = !motivating.count(access) && winner.coverage.count(access);
            if (additional) {
                winner.supporting.push_back(r);
            }
        }
        ordered.push_back(std::move(winner));
        // Actual receipt propagation invalidates the remaining ranking.
    }
    return ordered;
}
bool Constructor::consume()
{
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId || !currentState().causal.reachable()) {
        return true;
    }
    if (!ensureRecurringBaseline()) { return false; }
    const auto observer = program.operations[operation].pipe;
    const auto universe = normalizeDue(residual());
    result.work.normalizedDue += universe.size();
    auto physical = [](const std::vector<FrontierRequirement>& due) {
        std::set<Id> out;
        for (const auto& r : due) { out.insert(accessClass(r)); }
        return out;
    };
    while (true) {
        const auto before = residual();
        if (before.empty()) { break; }
        const auto oldClasses = physical(before);
        const auto oldDue = normalizedCoverage(universe, before, oldClasses);
        const auto normal = selectNormal(universe);
        if (normal && !*normal) { return false; }
        if (!normal) {
            // Transitional recipes run only when no normal realization was
            // certified. They do not reserve resources during normal probes.
            const auto version = ledger.version();
            if (!activateRecurring()) { return false; }
            if (version == ledger.version()) {
                auto requests = groups(before, RequirementStage::Known);
                auto stage = RequirementStage::Known;
                if (requests.empty()) {
                    stage = RequirementStage::Overlap;
                    requests = groups(before, stage);
                }
                if (requests.empty()) { break; }
                if (!bind(requests.front(), stage)) { return false; }
            }
        }
        const auto after = residual();
        const auto nextClasses = physical(after);
        const auto nextDue = normalizedCoverage(universe, after, nextClasses);
        const bool progress = nextDue.size() < oldDue.size() &&
            std::includes(oldDue.begin(), oldDue.end(), nextDue.begin(), nextDue.end()) &&
            std::includes(oldClasses.begin(), oldClasses.end(), nextClasses.begin(), nextClasses.end());
        if (!progress) {
            return fail(SelectedFailure::MissingParticipation,
                "selected realization did not reduce the fixed complete due residual", current);
        }
    }
    const auto missing = residual();
    for (const auto& r : missing) {
        if (r.source != observer) {
            return fail(SelectedFailure::MissingParticipation,
                "cross-engine requirement survived its selected transfer", current);
        }
    }
    if (!missing.empty()) {
        if (!program.target.barriers[unsigned(observer)]) {
            return fail(SelectedFailure::UnsupportedContract,
                "no qualified named fence for the remaining same-engine requirement", current);
        }
        const auto version = ledger.version();
        result.fences.push_back({current, observer, version, missing});
        // Repair this occurrence at its actual deadline. A future analytical
        // copy may receive completion from transfers selected before it is
        // reached. Ledger canonicalization still emits one command when
        // several analytical sites genuinely share the same authored word.
        ledger.append(current, {Command::Barrier, observer, Pipe::S, 0},
                      EndpointPurpose::LocalFence);
        if (!update()) {
            return false;
        }
    }
    const auto checked = frontier.inspect(currentState().causal, operation);
    if (!checked.applied) {
        return fail(SelectedFailure::FinalValidation, checked.reason, current);
    }
    return true;
}
} // namespace mlir::pto::oahs::selected
