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
Constructor::Constructor(const Program& p, SelectedOptions settings)
    : program(p), options(settings), frontier(p), control(p), storage(p), requirements(p, control, storage),
      ledger(p, control.canonicalCut, control.wordSpan), finalized(control.graph.sites.size())
{
    // Final-source proofs query every original occurrence, including exits.
    needsContextualReplay = options.finalReadSources && !control.finalReadGaps.empty();
    // A word shared by first/repeated/final observations executes in several
    // incoming contexts. The hypothesis-seeded traversal has only one of them
    // when it first repairs the word; its local certificate is insufficient.
    // Use the existing whole-original-control pending-effects evaluator before
    // selecting endpoints, so both storage and key queries see every copy.
    for (const auto& occurrences : control.wordOccurrences)
        if (std::count_if(occurrences.begin(), occurrences.end(),
                [&](Cut cut) { return control.reachable[cut]; }) > 1)
            needsContextualReplay = true;
    // Distinct observation words can also describe copies of one payload,
    // such as a counted loop's first/interior/final continuations.
    std::vector<bool> represented(program.operations.size());
    for (Cut cut = 0; cut < control.graph.operations.size(); ++cut) {
        const auto operation = control.graph.operations[cut];
        if (!control.reachable[cut] || operation == NoAnalysisId) continue;
        if (represented[operation]) needsContextualReplay = true;
        represented[operation] = true;
    }
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
    // Reciprocal storage transfers can close each other's physical-key
    // rearming obligations.  Consider the complete helper population for one
    // unordered engine pair at a time: this composes actual selected transfers,
    // never changes a completion publication/acquisition, and needs at most one
    // cold certificate per target engine pair rather than one trial per helper.
    const auto helperStart = std::chrono::steady_clock::now();
    using Pair = std::pair<Pipe, Pipe>;
    std::set<std::pair<Pipe, Pipe>> completionDirections;
    std::map<Pair, std::vector<Id>> helpers;
    for (const auto& endpoint : ledger.records()) if (ledger.active(endpoint.id)) {
        const auto& command = endpoint.command;
        if (endpoint.purpose == EndpointPurpose::Completion && command.kind == Command::Acquire)
            completionDirections.insert({command.source, command.observer});
        if (endpoint.purpose == EndpointPurpose::ConsumptionAcknowledgment) {
            const auto pair = std::minmax(command.source, command.observer);
            helpers[{pair.first, pair.second}].push_back(endpoint.id);
        }
    }
    for (const auto& [pair, endpoints] : helpers) {
        if (!options.finalHelperTrials) break;
        if (!completionDirections.count({pair.first, pair.second}) ||
            !completionDirections.count({pair.second, pair.first})) continue;
        ++result.work.helperCompositionTrials;
        Ledger trial = ledger;
        for (auto id : endpoints) trial.erase(id);
        if (!trial.publicationPrefixesValid()) {
            continue;
        }
        auto checked = checkCausalFrontier(program, trial.commands());
        result.work.invariantSiteEvaluations += checked.siteEvaluations;
        result.work.helperCompositionSiteEvaluations += checked.siteEvaluations;
        if (!checked.accepted) continue;
        for (auto id : endpoints) ledger.erase(id);
        const auto removed = std::count_if(endpoints.begin(), endpoints.end(), [&](Id id) {
            return ledger.endpoint(id).command.kind == Command::Acquire;
        });
        result.work.acknowledgments -= removed;
        result.work.rearmingDischarged += removed;
        result.work.rearmingComposed += removed;
    }
    result.work.helperCompositionMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - helperStart).count();
    if (!ledger.publicationPrefixesValid()) {
        return fail(SelectedFailure::FinalValidation, "final edit invalidated a publication prefix", exit);
    }
    const auto certificateStart = std::chrono::steady_clock::now();
    auto commands = ledger.commands();
    result.certificate = checkCausalFrontier(program, commands);
    result.work.finalCertificateMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - certificateStart).count();
    result.work.finalCertificateSiteEvaluations = result.certificate.siteEvaluations;
    result.work.invariantSiteEvaluations += result.certificate.siteEvaluations;
    if (!result.certificate.accepted) {
        return fail(SelectedFailure::FinalValidation, result.certificate.reason, result.certificate.cut);
    }
    for (auto& source : result.sources) {
        if (source.cut < result.certificate.cuts.size()) {
            source.snapshot = result.certificate.cuts[source.cut].beforeIssue;
            source.postOrigin = result.certificate.cuts[source.cut].incoming;
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
    result.sources.push_back({program.operations[operation].pipe, current, after, ledger.version(), {}, {}});
    if (needsContextualReplay) refreshSources(after);
}

SelectedPlan Constructor::run(const Commands& fixed)
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
        for (auto& obligation : result.rearming) {
            obligation.acquisition = remap[obligation.acquisition];
            std::vector<Id> live;
            for (auto id : obligation.reusePublications) {
                if (remap[id] != NoAnalysisId) {
                    live.push_back(remap[id]);
                }
            }
            obligation.reusePublications = std::move(live);
        }
        result.work.sourceHandles = result.sources.size();
        result.work.constructedSites = control.graph.sites.size();
        result.work.loopEntryPreparationSites = control.loopEntryPreparationSites;
        result.work.choicePreparationSites = control.choicePreparationSites;
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
    const auto qualificationStart = std::chrono::steady_clock::now();
    const auto channels = qualifyCyclicFrontiers(
        program, control, requirements, ledger.records().empty(), options.movingFrontiers,
        options.shareReaderReturns, &result.work);
    result.work.recurringQualificationMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - qualificationStart).count();
    if (options.recurring && !channels.empty() && !recurring(channels)) return complete();
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
SelectedPlan constructSelectedPlan(const Program& program, const Commands& fixed, SelectedOptions options)
{
    const auto start = std::chrono::steady_clock::now();
    selected::Constructor constructor(program, options);
    const auto prepared = std::chrono::steady_clock::now();
    auto result = constructor.run(fixed);
    result.work.preparationMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        prepared - start).count();
    result.work.elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    return result;
}
} // namespace mlir::pto::oahs
