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

const std::vector<EndpointRequirement>& RequirementFrontiers::endpoints(Id originalOwner) const
{
    static const std::vector<EndpointRequirement> empty;
    if (!ready) {
        return empty;
    }
    if (!endpointsIndexed) {
        // One immutable index, built only for consumers that request endpoint
        // discovery. Normal construction does not pay for unused native facts.
        std::map<Id, std::vector<Id>> contextOwners;
        contextOwners.emplace(NoAnalysisId, std::vector<Id>{});
        auto owners = [&](Id context) -> const std::vector<Id>& {
            std::vector<Id> path;
            auto current = context;
            while (!contextOwners.count(current)) {
                path.push_back(current);
                current = control->graph.contexts[current].parent;
            }
            while (!path.empty()) {
                const auto at = path.back();
                path.pop_back();
                const auto& scope = control->graph.contexts[at];
                auto list = contextOwners.at(scope.parent);
                if (scope.kind == AnalysisContext::ForBody) {
                    list.push_back(scope.ownerSite);
                }
                contextOwners.emplace(at, std::move(list));
            }
            return contextOwners.at(context);
        };
        for (const auto& frontiers : byDeadline) {
            for (const auto& frontier : frontiers) {
                if (frontier.source == frontier.observer) {
                    continue;
                }
                const auto& relationship = frontier.relationship;
                auto request = [&](EndpointRequirement::Role role, const StorageOrigin& access) {
                    for (auto owner : owners(control->graph.cutContexts[access.site])) {
                        byOwner[owner].push_back({role, owner, access, relationship});
                    }
                };
                if (relationship.kind == StorageRelationship::RAW) {
                    request(EndpointRequirement::FirstConsumer, relationship.target);
                } else {
                    request(EndpointRequirement::FirstWrite, relationship.target);
                }
                if (relationship.kind == StorageRelationship::WAR) {
                    request(EndpointRequirement::FinalReader, relationship.source);
                }
            }
        }
        endpointsIndexed = true;
    }
    const auto found = byOwner.find(originalOwner);
    return found != byOwner.end() ? found->second : empty;
}

bool RequirementFrontiers::needsOccurrenceSeparation(Id originalOwner) const
{
    for (const auto& request : endpoints(originalOwner)) {
        // Discover the original read frontiers before analytical refinement.
        // This requests endpoint vocabulary; it grants neither guard availability
        // nor a matched recurring protocol. Unknown retains the marginal query.
        if (request.role != EndpointRequirement::FirstWrite) {
            const auto reader = program->operations[request.access.operation].pipe;
            const auto& interval = storage->originalReaderFrontiers(
                {originalOwner, request.relationship.cell, reader});
            if (interval.complete()) {
                if (interval.separatesVisits) { return true; }
                continue;
            }
        }
        const bool backward = request.role != EndpointRequirement::FinalReader;
        const auto site = request.access.site;
        ++endpointWork;
        const auto key = std::make_tuple(site, request.relationship.cell, backward);
        const auto cached = separationByUse.find(key);
        if (cached != separationByUse.end()) {
            if (cached->second) {
                return true;
            }
            continue;
        }
        auto& separate = separationByUse[key];
        const auto& frontier = storage->nearestUses(
            backward ? control->predecessors[site] : control->graph.sites[site].successors,
            request.relationship.cell, {}, backward);
        if (!frontier.complete) {
            continue;
        }
        unsigned roles = frontier.boundaries.empty() ? 0u : 4u;
        for (const auto& access : frontier.accesses) {
            ++endpointWork;
            const auto role = use(access.site, request.relationship.cell).roles;
            // A write-bearing access is one production boundary, not two
            // alternative paths. Preserve its read obligation in physical facts.
            roles |= (role & 2u) ? 2u : (role & 1u);
        }
        const bool readerAndOther = (roles & 1u) && (roles & 6u);
        const bool initialAndWriter = (roles & 2u) && (roles & 4u);
        separate = readerAndOther || (backward && initialAndWriter);
        if (separate) {
            return true;
        }
    }
    return false;
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
        if (!frontier.classified) {
            frontier.reasons = storage->classifyRequirement(frontier.relationship);
            frontier.classified = true;
        }
        out[frontier.access] |= frontier.reasons;
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

const ReaderFrontiers& RequirementFrontiers::readerBoundaries(Cut site, unsigned cell, Id owner) const
{
    static const ReaderFrontiers unknown;
    if (!ready || !storage || !control || site >= control->graph.sites.size() || use(site, cell).roles != 1) {
        return unknown;
    }
    const auto key = std::make_tuple(owner, site, cell);
    const auto cached = readerFrontiers.find(key);
    if (cached != readerFrontiers.end()) { return cached->second; }
    auto& result = readerFrontiers[key];
    result.owner = owner;
    OriginalUseQuery query;
    query.cell = cell;
    query.owner = owner;
    query.occurrence = site;
    query.starts = control->predecessors[site];
    query.backward = true;
    result.preceding = storage->nearestUseSummary(query);
    query.backward = false;
    query.starts = control->graph.sites[site].successors;
    result.following = storage->nearestUseSummary(query);
    const bool complete = result.preceding.complete() && result.following.complete();
    if (!complete) {
        result.reason = "original-use interval is unavailable";
        return result;
    }
    const auto role = [](const PhysicalUseSummary& summary) {
        unsigned mask = 0;
        if (summary.roles & PhysicalUseSummary::Read) { mask |= 1; }
        if (summary.roles & (PhysicalUseSummary::FullWrite | PhysicalUseSummary::PartialWrite)) { mask |= 2; }
        if (summary.roles & (PhysicalUseSummary::Entry | PhysicalUseSummary::Exit)) { mask |= 4; }
        if (summary.roles & (PhysicalUseSummary::ReadWrite | PhysicalUseSummary::IntervalStop)) { mask |= 8; }
        return mask;
    };
    const auto previous = role(result.preceding), next = role(result.following);
    const bool firstUniform = previous == 1 || previous == 2;
    const bool finalUniform = next == 1 || next == 2 || next == 4 || next == 6;
    if (!firstUniform || !finalUniform) {
        result.reason = "mixed participation needs an available original endpoint predicate";
        return result;
    }
    const auto observation = program->observed ? program->observed->sites[site].observation : NoAnalysisId;
    result.first = {previous == 2 ? ReaderEndpoint::Status::Exact : ReaderEndpoint::Status::NoHit, site, observation};
    result.final = {next != 1 ? ReaderEndpoint::Status::Exact : ReaderEndpoint::Status::NoHit, site, observation};
    return result;
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
