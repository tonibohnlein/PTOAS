// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared immutable lifecycle view for placement, recurrence and binding.
// Physical histories and possible returns are candidates, never causal credit.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
RequirementFrontiers::RequirementFrontiers(
    const Program& program, const Control& control, const StorageFrontierAnalysis& storageAnalysis)
{
    if (!control.complete || !storageAnalysis.complete()) {
        error = control.complete ? storageAnalysis.reason() : control.reason;
        return;
    }
    storage = &storageAnalysis;
    this->program = &program;
    this->control = &control;
    byDeadline.resize(control.graph.sites.size());
    for (Cut site = 0; site < control.graph.sites.size(); ++site) {
        if (!control.reachable[site]) {
            continue;
        }
        const auto targetOperation = control.graph.operations[site];
        if (targetOperation == NoAnalysisId) {
            continue;
        }
        for (const auto& relationship : storageAnalysis.relationshipsAt(site)) {
            if (relationship.source.operation >= program.operations.size()) {
                error = "storage requirement has no original source operation";
                return;
            }
            RequirementFrontier frontier;
            frontier.relationship = relationship;
            frontier.source = program.operations[relationship.source.operation].pipe;
            frontier.observer = program.operations[targetOperation].pipe;
            const bool write = relationship.kind != StorageRelationship::WAR;
            frontier.access =
                (Id(relationship.cell) * PipeCount + unsigned(frontier.source)) * 2 + Id(write);
            frontier.deadline = control.canonicalCut[site];
            const auto& sourceUse = use(relationship.source.site, relationship.cell);
            auto& indexed = uses.at({relationship.source.site, relationship.cell});
            indexed.deadlines.push_back(site);
            if (relationship.kind == StorageRelationship::WAR) {
                indexed.returns.push_back(site);
            }
            const auto sourceComponent = control.component[relationship.source.site];
            const auto targetComponent = control.component[site];
            if (sourceComponent != NoAnalysisId && targetComponent != NoAnalysisId &&
                !control.components[sourceComponent].cyclic && !control.components[targetComponent].cyclic &&
                control.straight(relationship.source.site, site)) {
                const auto after = sourceUse.release;
                if (after != NoAnalysisId) {
                    frontier.publication = control.canonicalCut[after];
                    ++boundedSources;
                }
            }
            const auto sourceMode = sourceUse.occurrence;
            const auto targetMode = use(site, relationship.cell).occurrence;
            auto guarded = [&](Id occurrence) {
                if (!program.observed || occurrence >= program.observed->sites.size()) {
                    return false;
                }
                const auto observation = program.observed->sites[occurrence].observation;
                return observation != NoAnalysisId && observation < program.observed->observations.size() &&
                    std::any_of(program.observed->observations[observation].atoms.begin(),
                                program.observed->observations[observation].atoms.end(), [](const auto& atom) {
                                    return atom.kind == ObservationAtom::OriginalBoolean;
                                });
            };
            if (frontier.publication != NoAnalysisId) {
                frontier.occurrence = RequirementOccurrence::Acyclic;
            } else if (guarded(relationship.source.site) || guarded(site)) {
                frontier.occurrence = RequirementOccurrence::Guarded;
            } else if (sourceMode == targetMode) {
                frontier.occurrence = RequirementOccurrence::SameVisit;
            } else if (sourceMode.valid && targetMode.valid && sourceMode.owner == targetMode.owner &&
                       sourceMode.period == targetMode.period && sourceMode.residue == targetMode.residue &&
                       sourceMode.next && targetMode.previous) {
                frontier.occurrence = RequirementOccurrence::PreviousUse;
            } else if (!sourceMode.valid && targetMode.valid && !targetMode.previous) {
                frontier.occurrence = RequirementOccurrence::RegionEntry;
            } else if (sourceMode.valid && !sourceMode.next && !targetMode.valid) {
                frontier.occurrence = RequirementOccurrence::RegionContinuation;
            }
            ++occurrences[unsigned(frontier.occurrence)];
            byDeadline[site].push_back(std::move(frontier));
            ++population;
        }
    }
    ready = true;
}

const std::vector<RequirementFrontier>& RequirementFrontiers::at(Cut site) const
{
    static const std::vector<RequirementFrontier> empty;
    return ready && site < byDeadline.size() ? byDeadline[site] : empty;
}

std::map<Id, unsigned> RequirementFrontiers::reasons(Cut site) const
{
    std::map<Id, unsigned> out;
    if (!ready || site >= byDeadline.size() || !storage) {
        return out;
    }
    for (auto& frontier : byDeadline[site]) {
        if (!frontier.described) {
            frontier.provenance = storage->describeRequirement(frontier.relationship);
            frontier.described = true;
        }
        out[frontier.access] |= frontier.provenance.reasons;
    }
    return out;
}

const LifecycleUse& RequirementFrontiers::use(Cut site, unsigned cell) const
{
    static const LifecycleUse unknown;
    const bool valid = program && control && site < control->graph.sites.size() && cell < program->cells.size();
    if (!valid || !control->reachable[site]) {
        return unknown;
    }
    const auto operation = control->graph.operations[site];
    if (operation == NoAnalysisId) {
        return unknown;
    }
    const auto key = std::make_pair(site, cell);
    const auto found = uses.find(key);
    if (found != uses.end()) {
        return found->second;
    }
    LifecycleUse fact;
    fact.cell = cell;
    fact.origin = {site, operation, control->canonicalCut[site], control->graph.cutContexts[site]};
    fact.pipe = program->operations[operation].pipe;
    for (const auto& access : program->operations[operation].accesses) {
        if (access.cell == cell) {
            fact.roles |= unsigned(access.read) | (unsigned(access.write) << 1);
        }
    }
    if (!fact.roles) {
        return unknown;
    }
    fact.occurrence = occurrenceMode(*program, site);
    fact.release = fact.roles ? control->after(site) : NoAnalysisId;
    return uses.emplace(key, std::move(fact)).first->second;
}

Cut RequirementFrontiers::recurringRelease(Cut site, unsigned cell) const
{
    if (!use(site, cell).roles) {
        return NoAnalysisId;
    }
    const auto& boundary = control->publicationAfter(site);
    return boundary.proved() ? boundary.word : NoAnalysisId;
}

const PhysicalUseFrontier& RequirementFrontiers::nextUses(
    const std::vector<Cut>& starts, unsigned cell, const std::vector<Cut>& stops) const
{
    static const PhysicalUseFrontier unknown;
    return ready && storage ? storage->nearestUses(starts, cell, stops) : unknown;
}

std::vector<SelectedLifecycleDemand> RequirementFrontiers::demandsAt(
    Cut site, const std::vector<FrontierRequirement>& required) const
{
    std::set<Id> classes;
    for (const auto& demand : required) {
        classes.insert((Id(demand.cell) * PipeCount + unsigned(demand.source)) * 2 + Id(demand.sourceWrite));
    }
    std::vector<SelectedLifecycleDemand> result;
    for (const auto& fact : at(site)) {
        if (!classes.count(fact.access)) {
            continue;
        }
        const auto& source = use(fact.relationship.source.site, fact.relationship.cell);
        SelectedLifecycleDemand demand{fact.relationship, source.release, fact.deadline, {}};
        // A consumer's subsequent storage return may carry consumption back to
        // this producer. This is only a subscription to the required deadline.
        for (auto next : use(site, fact.relationship.cell).returns) {
            if (use(next, fact.relationship.cell).pipe == source.pipe) {
                demand.returnDeadlines.push_back(control->canonicalCut[next]);
            }
        }
        result.push_back(std::move(demand));
    }
    return result;
}
} // namespace mlir::pto::oahs::selected
