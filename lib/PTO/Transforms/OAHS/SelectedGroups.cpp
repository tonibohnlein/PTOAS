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
    const auto& atConsumer = cache.cuts[current].before;
    if (!atSource.causal.reachable()) {
        return out;
    }
    const auto& history = atSource.causal.facts()->history;
    for (const auto& r : requirements) {
        const auto index = accessClass(r);
        const auto origin = atSource.latest.get(index);
        const bool sameOccurrence = cut == current ||
            (origin != NoAnalysisId && origin == atConsumer.latest.get(index));
        const auto* reached = history.find(index);
        if (sameOccurrence && reached && frontierContains(*reached, PipeCount + unsigned(source))) {
            out.insert(index);
        }
    }
    return out;
}
Group Constructor::sourceGroup(
    Pipe source, const std::vector<FrontierRequirement>& required,
    const std::vector<FrontierRequirement>& all)
{
    Group group;
    group.source = source;
    group.requirements = required;
    Id latest = NoAnalysisId;
    bool comparable = true;
    for (const auto& r : required) {
        const auto origin = currentState().latest.get(accessClass(r));
        if (origin == NoAnalysisId || !control.straight(origin, current)) {
            comparable = false;
            break;
        }
        if (latest == NoAnalysisId || control.position[latest] < control.position[origin]) {
            latest = origin;
        }
    }
    group.publication = comparable && latest != NoAnalysisId ? control.after(latest) : current;
    if (group.publication == NoAnalysisId || !control.straight(group.publication, current)) {
        group.publication = current;
        comparable = false;
    }
    if (comparable) {
        const auto saved = std::find_if(result.sources.begin(), result.sources.end(), [&](const auto& handle) {
            return handle.origin == latest && handle.pipe == source && handle.cut == group.publication &&
                   handle.version == cache.version && handle.snapshot.reachable();
        });
        if (saved == result.sources.end()) {
            comparable = false;
            group.publication = current;
        }
    }
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
    for (auto stage : {RequirementStage::Known, RequirementStage::Overlap}) {
        auto requests = groups(residual(), stage);
        for (auto& group : requests) {
            const auto missing = residual();
            const bool needed = std::any_of(missing.begin(), missing.end(), [&](const auto& r) {
                return std::any_of(group.requirements.begin(), group.requirements.end(), [&](const auto& g) {
                    return accessClass(r) == accessClass(g);
                });
            });
            if (needed && !bind(group, stage)) {
                return false;
            }
        }
    }
    const auto missing = residual();
    const auto observer = program.operations[operation].pipe;
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
