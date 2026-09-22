// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedInternal.h"
#include <algorithm>
#include <chrono>

namespace mlir::pto::oahs::selected {
Constructor::Constructor(const Program& p)
    : program(p), frontier(p), control(p), storage(p), requirements(p, control, storage),
      ledger(p, control.canonicalCut), finalized(control.graph.sites.size())
{
}
bool Constructor::fail(SelectedFailure failure, std::string reason, Cut cut)
{
    result.failure = failure;
    result.reason = std::move(reason);
    result.cut = cut;
    result.success = false;
    result.commands.clear();
    result.certificate = {};
    result.loops.clear();
    return false;
}
bool Constructor::finish()
{
    const auto exit = invocationExitCut(program);
    if (program.invocation.retirement == Program::InvocationContract::DrainAllAtReturn) {
        const auto& word = ledger.word(exit);
        if (word.empty() || ledger.endpoint(word.back()).command.kind != Command::BarrierAll) {
            ledger.append(exit, {Command::BarrierAll, Pipe::S, Pipe::S, 0}, EndpointPurpose::Retirement);
        }
    }
    auto commands = ledger.commands();
    result.certificate = checkCausalFrontier(program, commands);
    result.work.invariantSiteEvaluations += result.certificate.siteEvaluations;
    if (!result.certificate.accepted) {
        return fail(SelectedFailure::FinalValidation, result.certificate.reason, result.certificate.cut);
    }
    for (auto& source : result.sources) {
        if (source.cut < result.certificate.cuts.size()) {
            source.snapshot = result.certificate.cuts[source.cut].beforeIssue;
            source.version = ledger.version();
        }
    }
    if (program.observed && !recurringKeys.empty()) {
        for (const auto& region : program.observed->loops) {
            if (std::none_of(result.channels.begin(), result.channels.end(),
                            [&](const auto& channel) { return channel.owner == region.owner; })) continue;
            SelectedLoopInterface loop;
            loop.owner = region.owner;
            loop.entry = region.entry;
            loop.exit = region.exit;
            loop.version = ledger.version();
            loop.incoming = result.certificate.cuts[loop.entry].incoming;
            loop.outgoing = result.certificate.cuts[loop.exit].incoming;
            for (auto site : region.sites) {
                const auto observation = program.observed->sites[site].observation;
                const auto& snapshot = result.certificate.cuts[site];
                if (observation != NoAnalysisId && snapshot.incoming.reachable())
                    loop.clauses.push_back({site, observation, loop.version,
                        snapshot.incoming, snapshot.beforeIssue, snapshot.outgoing});
            }
            result.loops.push_back(std::move(loop));
        }
    }
    result.commands = std::move(commands);
    result.success = true;
    return true;
}
void Constructor::registerSource()
{
    const auto operation = control.graph.operations[current];
    const auto after = control.after(current);
    if (operation == NoAnalysisId || after == NoAnalysisId) return;
    sourcesAtCut[after].push_back(result.sources.size());
    result.sources.push_back({program.operations[operation].pipe, current, after, ledger.version(), {}});
    if (needsContextualReplay) refreshSources(after);
}

SelectedPlan Constructor::run(const Commands& fixed, bool useRecurring)
{
    const auto start = std::chrono::steady_clock::now();
    auto complete = [&]() {
        // Internal endpoint identities survive online discharge. Export a
        // compact ledger and remap all public provenance to its live records.
        std::vector<Id> remap(ledger.records().size(), NoAnalysisId);
        for (const auto& endpoint : ledger.records()) if (ledger.active(endpoint.id)) {
            remap[endpoint.id] = result.ledger.size();
            auto live = endpoint;
            live.id = result.ledger.size();
            result.ledger.push_back(live);
        }
        for (auto& endpoint : result.ledger)
            if (endpoint.acknowledges != NoAnalysisId) endpoint.acknowledges = remap[endpoint.acknowledges];
        for (auto& decision : result.decisions) {
            std::vector<Id> live;
            for (auto id : decision.endpoints) if (remap[id] != NoAnalysisId) live.push_back(remap[id]);
            decision.endpoints = std::move(live);
            if (decision.repairedAcquisition != NoAnalysisId)
                decision.repairedAcquisition = remap[decision.repairedAcquisition];
        }
        result.work.sourceHandles = result.sources.size();
        result.work.constructedSites = control.graph.sites.size();
        result.work.loopEntryPreparationSites = control.loopEntryPreparationSites;
        result.work.commandWords = commandCutCount(program);
        result.work.cells = program.cells.size();
        result.work.eligibleKeys = frontier.keys().size();
        result.work.components = control.components.size();
        result.work.cyclicComponents = std::size_t(std::count_if(
            control.components.begin(), control.components.end(),
            [](const Component& block) { return block.cyclic; }));
        result.work.requirementFrontiers = requirements.size();
        result.work.qualifiedSourceFrontiers = requirements.sourceBoundaries();
        result.work.unqualifiedSourceFrontiers = requirements.size() - requirements.sourceBoundaries();
        const auto& occurrences = requirements.occurrenceCounts();
        result.work.frontierAcyclic = occurrences[unsigned(RequirementOccurrence::Acyclic)];
        result.work.frontierSameVisit = occurrences[unsigned(RequirementOccurrence::SameVisit)];
        result.work.frontierPreviousUse = occurrences[unsigned(RequirementOccurrence::PreviousUse)];
        result.work.frontierRegionEntry = occurrences[unsigned(RequirementOccurrence::RegionEntry)];
        result.work.frontierRegionContinuation =
            occurrences[unsigned(RequirementOccurrence::RegionContinuation)];
        result.work.frontierGuarded = occurrences[unsigned(RequirementOccurrence::Guarded)];
        result.work.frontierUnknown = occurrences[unsigned(RequirementOccurrence::Unknown)];
        result.work.elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start).count();
        return std::move(result);
    };
    if (!control.complete || !storage.complete() || !requirements.complete()) {
        fail(control.validInput ? SelectedFailure::UnqualifiedControl : SelectedFailure::InvalidInput,
             !control.complete ? control.reason : (!storage.complete() ? storage.reason() : requirements.reason()));
        return complete();
    }
    if (!frontier.complete()) {
        fail(SelectedFailure::UnsupportedContract, frontier.reason());
        return complete();
    }
    std::string reason;
    if (!ledger.initialize(fixed, reason)) {
        fail(SelectedFailure::InvalidInput, reason);
        return complete();
    }
    if (useRecurring) {
        const auto channels = qualifyCyclicFrontiers(program, control, requirements);
        result.work.recurringProposals = channels.size();
        if (!channels.empty() && !recurring(channels)) {
            return complete();
        }
    }
    for (activeComponent = 0; activeComponent < control.components.size(); ++activeComponent) {
        // End compiler role reservations, not physical event state. Actual D/S
        // facts and balances continue through the selected-body fixed point.
        closedBindings.clear();
        closedKeys = recurringKeys;
        const auto& block = control.components[activeComponent];
        for (activeOffset = 0; activeOffset < block.order.size(); ++activeOffset) {
            current = block.order[activeOffset];
            ++result.work.frontierVisits;
            if (!advance() || !consume()) {
                return complete();
            }
            auto outgoing = currentState();
            if (outgoing.causal.reachable() && !payload(outgoing, current, cache)) {
                fail(SelectedFailure::FinalValidation, cache.reason, current);
                return complete();
            }
            cache.cuts[current].outgoing = std::move(outgoing);
            finalized[current] = true;
            registerSource();
        }

    }
    finish();
    return complete();
}
} // namespace mlir::pto::oahs::selected

namespace mlir::pto::oahs {
bool hasQualifiedRecurringAccesses(const Program& program)
{
    selected::Control control(program);
    StorageFrontierAnalysis storage(program);
    selected::RequirementFrontiers requirements(program, control, storage);
    return control.complete && storage.complete() && requirements.complete() &&
        !selected::qualifyCyclicFrontiers(program, control, requirements).empty();
}
SelectedPlan constructSelectedPlan(const Program& program, const Commands& fixed)
{
    const auto start = std::chrono::steady_clock::now();
    auto attempt = [&](bool useRecurring) {
        const auto preparing = std::chrono::steady_clock::now();
        selected::Constructor constructor(program);
        const auto prepared = std::chrono::steady_clock::now();
        auto plan = constructor.run(fixed, useRecurring);
        plan.work.preparationMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
            prepared - preparing).count();
        return plan;
    };
    auto result = attempt(true);
    if (!result.success && result.work.recurringProposals) {
        // Optional protocols may fit their own key population yet starve an
        // ordinary deadline. Discard the whole private attempt, including its
        // ledgers and reservations, before constructing without specialization.
        // This is one deterministic retry, never subset or deletion search.
        DeclinedRecurringAttempt declined{result.failure, result.reason, result.cut, result.work};
        result = attempt(false);
        result.declinedRecurring = std::move(declined);
    }
    result.work.elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
} // namespace mlir::pto::oahs
