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
    std::map<Id, unsigned> out;
    for (const auto& relationship : storage.relationshipsAt(site)) {
        const auto source = program.operations[relationship.source.operation].pipe;
        const bool write = relationship.kind != StorageRelationship::WAR;
        const auto index = (Id(relationship.cell) * PipeCount + unsigned(source)) * 2 + Id(write);
        out[index] |= storage.describeRequirement(relationship).reasons;
    }
    return out;
}
std::set<Id> Constructor::coverage(
    Cut cut, Pipe source, const std::vector<FrontierRequirement>& requirements) const
{
    std::set<Id> out;
    const auto& atSource = cache.cuts[cut].before;
    if (!atSource.causal.reachable()) {
        return out;
    }
    const auto& history = atSource.causal.facts()->history;
    for (const auto& r : requirements) {
        const auto index = accessClass(r);
        const auto* reached = history.find(index);
        if (freshBetween(cut, current, index) && reached &&
            frontierContains(*reached, PipeCount + unsigned(source))) {
            out.insert(index);
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
bool Constructor::sourceFrontier(
    Pipe source, const std::vector<FrontierRequirement>& required, Group& group) const
{
    // This extension is intentionally acyclic: static predecessor identities
    // are not a bank-generation correspondence across an unqualified loop.
    if (required.empty() || control.components[activeComponent].cyclic ||
        control.canonicalCut[current] != current) return false;
    auto uniqueWord = [&](Cut cut) {
        if (cut >= control.canonicalCut.size() || control.canonicalCut[cut] != cut) return false;
        const auto& occurrences = control.wordOccurrences[cut];
        return std::count_if(occurrences.begin(), occurrences.end(),
                            [&](Id site) { return control.reachable[site]; }) == 1;
    };
    if (!uniqueWord(current)) return false;
    std::set<Id> needed;
    for (const auto& requirement : required) needed.insert(accessClass(requirement));
    std::map<Id, const SelectedSource*> availableSources;
    for (const auto& handle : result.sources) {
        if (handle.pipe == source && handle.version == cache.version && handle.snapshot.reachable())
            availableSources.emplace(handle.origin, &handle);
    }
    std::set<Cut> publications;
    std::vector<bool> seen(control.graph.sites.size());
    auto todo = control.predecessors[current];
    while (!todo.empty()) {
        const auto site = todo.back(); todo.pop_back();
        if (!control.reachable[site] || seen[site]) continue;
        seen[site] = true;
        const auto component = control.component[site];
        if (component == NoAnalysisId || control.components[component].cyclic) return false;
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
            if (found == availableSources.end()) return false;
            const auto& handle = *found->second;
            if (handle.cut == current || !uniqueWord(handle.cut)) return false;
            const auto sourceComponent = control.component[handle.cut];
            if (sourceComponent == NoAnalysisId || control.components[sourceComponent].cyclic) return false;
            const auto& snapshot = cache.cuts[handle.cut].before.causal;
            if (!snapshot.reachable()) return false;
            // Do not assume that an absent class on one arm means an optional
            // corresponding producer. That case needs a separate qualifier.
            for (auto access : needed) {
                const auto* history = snapshot.facts()->history.find(access);
                if (!history || !frontierContains(*history, PipeCount + unsigned(source))) return false;
            }
            publications.insert(handle.cut);
            continue;
        }
        if (site == control.graph.entry || control.predecessors[site].empty()) return false;
        const auto& before = control.predecessors[site];
        todo.insert(todo.end(), before.begin(), before.end());
    }
    std::vector<Cut> cuts(publications.begin(), publications.end());
    if (!control.lookahead.balancedTransfer(cuts, current, control.graph.entry, control.graph.exit)) return false;
    // No edit/retry: only offer this vocabulary when an unused physical key has
    // its complete source-time certificate at ALL alternative publications.
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    Id selected = NoAnalysisId;
    for (Id key = 0; key < frontier.keys().size(); ++key) {
        const auto& identity = frontier.keys()[key];
        if (identity.source != source || identity.observer != observer || closedKeys.count(key)) continue;
        const bool used = std::any_of(ledger.records().begin(), ledger.records().end(), [&](const auto& endpoint) {
            const auto& command = endpoint.command;
            return (command.kind == Command::Publish || command.kind == Command::Acquire) &&
                command.source == source && command.observer == observer && command.key == identity.key;
        });
        if (used || !std::all_of(cuts.begin(), cuts.end(), [&](Cut cut) {
                return canPublish(cache.cuts[cut].before, key);
            })) continue;
        selected = key;
        break;
    }
    if (selected == NoAnalysisId) return false;
    group.publications = std::move(cuts);
    group.publication = *std::min_element(group.publications.begin(), group.publications.end(),
        [&](Cut a, Cut b) { return control.position[a] < control.position[b]; });
    group.forwardKey = selected;
    group.version = cache.version;
    group.common = false;
    group.coverage = std::move(needed); // conservative joint credit; no hypothetical receipt
    return true;
}
bool Constructor::loopEntryFrontier(Pipe source, const std::vector<FrontierRequirement>& required, Group& group)
{
    if (!program.observed || required.empty()) return false;
    std::set<Id> needed;
    for (const auto& r : required) needed.insert(accessClass(r));
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    for (const auto& loop : control.loopEntries) {
        const auto& first = loop.firstConsumers[unsigned(observer)];
        if (std::find(first.begin(), first.end(), current) == first.end() ||
            control.canonicalCut[loop.entry] != loop.entry) continue;
        if (!std::all_of(first.begin(), first.end(), [&](Cut cut) {
            const auto& op = program.operations[control.graph.operations[cut]];
            return std::all_of(required.begin(), required.end(), [&](const auto& r) {
                return std::any_of(op.accesses.begin(), op.accesses.end(), [&](const auto& a) {
                    return a.cell == r.cell && (a.write || (a.read && r.sourceWrite));
                });
            });
        })) continue;
        // Moving a wait ahead of Q's first payload can still order another
        // engine through an earlier Q publication. Consult actual selected
        // words, not just payload order. Existing entry commands precede the
        // appended acquisition; existing deadline commands follow it only if
        // we hoist, so the latter must also be checked.
        const bool communicates = std::any_of(
            loop.crossedWords[unsigned(observer)].begin(),
            loop.crossedWords[unsigned(observer)].end(), [&](Cut cut) {
                return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                    const auto& command = ledger.endpoint(id).command;
                    return command.kind == Command::BarrierAll ||
                        (command.kind == Command::Publish && command.source == observer);
                });
            });
        if (communicates) continue;
        // No relevant source-class occurrence may be refreshed inside the
        // region. Readiness/release belongs to this bank, not to a maximum
        // operation number or to every operation on its source engine.
        if (std::any_of(needed.begin(), needed.end(), [&](Id access) {
            return loop.issuedClasses.count(access) != 0;
        })) continue;
        const SelectedSource* selected = nullptr;
        for (const auto& handle : result.sources) {
            if (handle.pipe != source || handle.version != cache.version || !handle.snapshot.reachable() ||
                !control.straight(handle.cut, loop.entry)) continue;
            bool covered = true;
            const auto& snapshot = cache.cuts[handle.cut].before.causal;
            if (!snapshot.reachable()) continue;
            for (auto access : needed) {
                const auto* history = snapshot.facts()->history.find(access);
                covered &= freshBetween(handle.cut, loop.entry, access) && history &&
                    frontierContains(*history, PipeCount + unsigned(source));
            }
            if (covered && (!selected || control.position[handle.cut] < control.position[selected->cut]))
                selected = &handle;
        }
        // With no saved source in this invocation, a source-inactive region
        // can establish its incoming completion at entry. All alternative first
        // observer payloads must need these same classes: a branch containing
        // unrelated observer work is not a reason to advance its deadline.
        const bool regional = !selected && !loop.issuedPipes.count(source) &&
            std::none_of(loop.sites.begin(), loop.sites.end(), [&](Cut cut) {
                return std::any_of(ledger.word(cut).begin(), ledger.word(cut).end(), [&](Id id) {
                    const auto& c = ledger.endpoint(id).command;
                    return c.kind == Command::BarrierAll || c.source == source ||
                        ((c.kind == Command::Publish || c.kind == Command::Acquire) && c.observer == source);
                });
            });
        if (!selected && !regional) continue;
        const auto publication = selected ? selected->cut : loop.entry;
        if (!regional && !control.lookahead.balancedTransfer({publication}, loop.entry,
                                       control.graph.entry, control.graph.exit)) continue;
        auto unused = [&](Pipe a, Pipe b) {
            for (Id key = 0; key < frontier.keys().size(); ++key) {
                const auto& e = frontier.keys()[key];
                if (e.source != a || e.observer != b || closedKeys.count(key) || recurringKeys.count(key)) continue;
                if (std::none_of(ledger.records().begin(), ledger.records().end(), [&](const auto& r) {
                    if (!ledger.active(r.id)) return false;
                    const auto& c = r.command;
                    return (c.kind == Command::Publish || c.kind == Command::Acquire) &&
                        c.source == a && c.observer == b && c.key == e.key;
                })) return key;
            }
            return NoAnalysisId;
        };
        const auto forward = unused(source, observer);
        if (forward == NoAnalysisId) continue;
        const bool repeats = control.components[control.component[publication]].cyclic;
        Id reverse = NoAnalysisId;
        auto commands = ledger.commands();
        commands[publication].push_back({Command::Publish, source, observer, frontier.keys()[forward].key});
        commands[loop.entry].push_back({Command::Acquire, source, observer, frontier.keys()[forward].key});
        auto trial = analyze(program, commands, {false});
        result.work.loopEntryAnalysisSites += trial.stats.siteEvaluations;
        const bool needsConsumption = std::any_of(trial.protocol.begin(), trial.protocol.end(), [&](const auto& r) {
            return r.kind == ProtocolObligation::ConsumptionNotEstablished &&
                r.event.source == source && r.event.observer == observer &&
                r.event.key == frontier.keys()[forward].key;
        });
        // Existing causal paths get the first opportunity to prove reuse. A
        // return is justified by this key's missing consumption certificate,
        // not merely by being textually inside a repeated component.
        if (trial.complete && trial.diagnostics.empty() && repeats && needsConsumption) {
            reverse = unused(observer, source);
            if (reverse == NoAnalysisId) continue;
            commands[loop.entry].push_back({Command::Publish, observer, source, frontier.keys()[reverse].key});
            commands[loop.entry].push_back({Command::Acquire, observer, source, frontier.keys()[reverse].key});
            trial = analyze(program, commands, {false});
            result.work.loopEntryAnalysisSites += trial.stats.siteEvaluations;
        }
        if (!trial.complete || !trial.diagnostics.empty() || !trial.protocol.empty()) continue;
        group.publication = publication;
        group.publications = {publication};
        group.entryAcquisition = loop.entry;
        group.entryReturnKey = reverse;
        group.entryRepeats = repeats;
        group.forwardKey = forward;
        group.version = ledger.version();
        group.coverage = needed;
        return true;
    }
    return false;
}
Group Constructor::sourceGroup(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all)
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
    if (!comparable && sourceFrontier(source, required, group)) return group;
    if (!comparable && loopEntryFrontier(source, required, group)) return group;
    group.publication = comparable ? selected->cut : current;
    group.common = !comparable;
    group.coverage = coverage(group.publication, source, all);
    return group;
}
std::vector<Group> Constructor::groups(
    const std::vector<FrontierRequirement>& all, RequirementStage stage)
{
    const auto labels = reasons(current);
    const auto observer = program.operations[control.graph.operations[current]].pipe;
    std::map<Pipe, std::vector<FrontierRequirement>> sources;
    for (const auto& r : all) {
        const auto found = labels.find(accessClass(r));
        const auto flags = found == labels.end() ? unsigned(AdditionalOverlap) : found->second;
        if (r.source == observer || (stage == RequirementStage::Known && !(flags & (KnownReadiness | KnownReuse)))) {
            continue;
        }
        sources[r.source].push_back(r);
    }
    std::vector<Group> pending, ordered;
    for (const auto& source : sources) {
        pending.push_back(sourceGroup(source.first, source.second, all));
    }
    while (!pending.empty()) {
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
        ordered.push_back(std::move(pending[selected]));
        pending.erase(pending.begin() + selected);
    }
    return ordered;
}
bool Constructor::consume()
{
    const auto operation = control.graph.operations[current];
    if (operation == NoAnalysisId || !currentState().causal.reachable()) {
        return true;
    }
    const auto observer = program.operations[operation].pipe;
    auto crossCount = [&](const std::vector<FrontierRequirement>& values) {
        return std::count_if(values.begin(), values.end(),
            [&](const auto& r) { return r.source != observer; });
    };
    for (auto stage : {RequirementStage::Known, RequirementStage::Overlap}) {
        while (true) {
            const auto before = residual();
            auto requests = groups(before, stage);
            if (requests.empty()) break;
            if (!bind(requests.front(), stage)) return false;
            // A real acquisition can change which remaining prefix is best.
            // Re-form groups from the new frontier; never retain a stale source
            // requirement merely because it was in an earlier candidate list.
            // No payload was issued, so strict residual decrease is the finite
            // progress measure, not an iteration limit or a retry budget.
            if (crossCount(residual()) >= crossCount(before)) {
                return fail(SelectedFailure::MissingParticipation,
                    "selected transfer did not reduce the cross-engine residual", current);
            }
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
        ledger.append(current, {Command::Barrier, observer, Pipe::S, 0}, EndpointPurpose::LocalFence);
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
