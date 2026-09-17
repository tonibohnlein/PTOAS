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
