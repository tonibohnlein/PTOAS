// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
namespace {
SequenceAnalysis finishSequence(std::shared_ptr<SequenceAnalysisState> state)
{
    SequenceAnalysis result;
    auto& composer = *state;
    composer.costs = {};
    for (const auto& child : composer.children) {
        const auto& cost = child.regional.cost;
        if (cost.arithmeticRegions > UINT64_MAX - composer.costs.arithmeticRegions ||
            cost.boundaryBytes > UINT64_MAX - composer.costs.boundaryBytes) {
            result.error = "regional arithmetic cost count exceeds representation";
            return result;
        }
        composer.costs.arithmeticRegions += cost.arithmeticRegions;
        composer.costs.boundaryBytes += cost.boundaryBytes;
        composer.costs.physicalFragments += cost.physicalFragments;
        composer.costs.rotatingResidues += cost.rotatingResidues;
        composer.costs.numericVisits += cost.numericVisits;
        composer.costs.repeatedRegions += cost.repeatedRegions;
        composer.costs.phaseDescriptions += cost.phaseDescriptions;
        composer.costs.selectorComparisons += cost.selectorComparisons;
        composer.costs.crossingCandidates += cost.crossingCandidates;
        composer.costs.implicationChecks += cost.implicationChecks;
    }
    if (!composer.importSummaries(composer.requireEndpoints)) { result.error = composer.error; return result; }
    composer.bridges();
    if (composer.reconstructPrerequisites && !composer.valueBridges()) { result.error = composer.error; return result; }
    if (!composer.closure()) { result.error = composer.error; return result; }
    if (!composer.expressions.constructionError().empty()) {
        result.error = composer.expressions.constructionError(); return result;
    }
    for (const auto& port : composer.ports) {
        const auto& child = composer.children[port.child];
        result.occurrences.push_back({port.child, port.type, child.anchors[port.type],
                                      child.regional.occurrenceLoops[port.type], port.ordinal, port.visits});
    }
    // Parent selectors are guarded folds. They preserve original occurrence
    // identities and remain reusable independently of incoming storage state.
    auto append = [](auto& destination, const auto& source) {
        for (auto value : source) { destination.push_back({value.port, value.present}); }
    };
    auto mask = [&](auto& selected, Expr predicate) {
        for (auto& value : selected) { value.present = composer.both(value.present, predicate); }
    };
    for (uint32_t cell = 0; cell < composer.cells.size(); ++cell) {
        SequenceStorageBoundary out;
        out.cell = composer.cells[cell];
        auto seenWriter = composer.no();
        for (uint32_t child = 0; child < composer.children.size(); ++child) {
            const auto& local = composer.boundaries[child][cell];
            auto written = composer.no();
            for (auto writer : local.firstWriters) {
                written = composer.either(written, writer.present);
                out.firstWriters.push_back({writer.port, composer.both(writer.present, composer.negate(seenWriter))});
            }
            for (const auto& [pipe, reads] : local.firstReaders) {
                auto previouslyRead = composer.no();
                for (auto prior : out.firstReaders[pipe]) {
                    previouslyRead = composer.either(previouslyRead, prior.present);
                }
                for (auto read : reads) {
                    out.firstReaders[pipe].push_back({read.port, composer.both(read.present,
                        composer.negate(composer.either(seenWriter, previouslyRead)))});
                }
            }
            mask(out.lastWriters, composer.negate(written));
            append(out.lastWriters, local.lastWriters);
            for (auto& [pipe, reads] : out.lastReaders) {
                auto replaced = written;
                auto found = local.lastReaders.find(pipe);
                if (found != local.lastReaders.end()) {
                    for (auto read : found->second) { replaced = composer.either(replaced, read.present); }
                }
                mask(reads, composer.negate(replaced));
            }
            for (const auto& [pipe, reads] : local.lastReaders) { append(out.lastReaders[pipe], reads); }
            seenWriter = composer.either(seenWriter, written);
        }
        result.storageBoundary.push_back(std::move(out));
    }
    for (uint32_t childId = 0; childId < composer.children.size(); ++childId) {
        for (const auto& [pipe, values] : composer.nativeFirst[childId]) {
            auto seen = composer.no(), nonempty = composer.no();
            for (auto previous : result.firstPayloads[pipe]) { seen = composer.either(seen, previous.present); }
            for (auto selected : values) {
                nonempty = composer.either(nonempty, selected.present);
                result.firstPayloads[pipe].push_back({selected.port,
                    composer.both(selected.present, composer.negate(seen))});
            }
            mask(result.lastPayloads[pipe], composer.negate(nonempty));
            auto found = composer.nativeLast[childId].find(pipe);
            if (found != composer.nativeLast[childId].end()) {
                append(result.lastPayloads[pipe], found->second);
            }
        }
    }
    result.cost = composer.costs;
    result.cost.children = composer.children.size(); result.cost.cells = composer.cells.size();
    result.cost.ports = composer.ports.size(); result.cost.crossings = composer.crossings.size();
    result.cost.expressionNodes = composer.expressions.size();
    if (!composer.expressions.constructionError().empty()) {
        result.error = composer.expressions.constructionError(); return result;
    }
    result.state = std::move(state);
    return result;
}
} // namespace
SequenceAnalysis analyzeSequence(func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    return analyzeSequenceRegion(function, input, program, 0, std::make_shared<RegionExpressions>());
}
SequenceAnalysis analyzeSequenceRegion(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<PhaseIndex> sharedIndex)
{
    SequenceAnalysis result;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !expressions ||
        !expressions->constructionError().empty()) {
        result.error = "sequence route requires a single structured function body"; return result;
    }
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        result.error = "sequence endpoint arithmetic requires a 64-bit index representation"; return result;
    }
    auto state = std::make_shared<SequenceAnalysisState>(function, std::move(expressions), std::move(sharedIndex));
    state->input = &input;
    state->program = &program;
    state->completeInvocation = node == 0;
    if (node < program.nodes.size()) { state->requiredOuterLoops = program.nodes[node].loops; }
    if (!state->collect(node) || !state->partition()) { result.error = state->error; return result; }
    state->summarize();
    state->bindAdapters();
    return finishSequence(std::move(state));
}
SequenceAnalysis composeRegionalSequence(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children,
    bool reconstructPrerequisites, bool requireEndpoints)
{
    SequenceAnalysis result;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !expressions ||
        !expressions->constructionError().empty()) {
        result.error = "sequence requires a valid common regional expression arena"; return result;
    }
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (bits.isScalable() || bits.getFixedValue() != 64) {
        result.error = "sequence endpoint arithmetic requires a 64-bit index representation"; return result;
    }
    auto state = std::make_shared<SequenceAnalysisState>(function, std::move(expressions));
    state->reconstructPrerequisites = reconstructPrerequisites;
    state->requireEndpoints = requireEndpoints;
    SmallVector<const CompoundInstanceElement*> phases;
    DenseSet<const CompoundInstanceElement*> seen;
    for (const auto& regional : children) {
        for (const auto& anchor : regional.anchors) {
            if (anchor.phase && seen.insert(anchor.phase).second) {
                phases.push_back(anchor.phase);
            }
        }
    }
    if (reconstructPrerequisites && failed(state->index.build(function, phases))) {
        result.error = "cannot reconstruct regional value prerequisites";
        return result;
    }
    for (auto& regional : children) {
        Child child; child.regional = std::move(regional);
        state->children.push_back(std::move(child));
    }
    return finishSequence(std::move(state));
}
RegionExpressions* sequenceExpressions(SequenceAnalysis& analysis)
{
    return analysis.state ? &analysis.state->expressions : nullptr;
}
std::pair<uint64_t, uint64_t> sequencePreparationCounts(const SequenceAnalysis& analysis)
{
    if (!analysis.state) { return {0, 0}; }
    return {analysis.state->childPreparationOperations, analysis.state->crossingPreparationOperations};
}
std::optional<RegionExpressions::Id> sequenceEventReachability(const SequenceAnalysis& analysis,
    uint32_t sourcePort, PeriodicEventKind sourceKind, uint32_t targetPort, PeriodicEventKind targetKind)
{
    const auto validKind = [](PeriodicEventKind kind) {
        return kind == PeriodicEventKind::Start || kind == PeriodicEventKind::Completion;
    };
    if (!analysis.state || !validKind(sourceKind) || !validKind(targetKind) ||
        sourcePort >= analysis.occurrences.size() || targetPort >= analysis.occurrences.size()) {
        return std::nullopt;
    }
    auto source = 2*sourcePort + (sourceKind == PeriodicEventKind::Completion);
    auto target = 2*targetPort + (targetKind == PeriodicEventKind::Completion);
    return analysis.state->eventReachability(source, target);
}
std::optional<RegionExpressions::Id> sequenceEventReachability(SequenceAnalysis& analysis,
    SequenceEvent source, SequenceEvent target)
{
    if (!analysis.state || !analysis.error.empty()) { return std::nullopt; }
    auto& state = *analysis.state;
    auto valid = [&](const SequenceEvent& event) {
        return event.child < state.children.size() && validRegionalEvent(state.children[event.child].regional,
                   {event.type, event.ordinal, event.kind, event.visits});
    };
    if (!valid(source) || !valid(target)) { return std::nullopt; }
    return state.eventReachability(std::move(source), std::move(target));
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(SequenceAnalysis& analysis)
{
    if (!analysis.state || !analysis.error.empty()) { return failure(); }
    analysis.insertionError.clear();
    auto result = analysis.state->prepare();
    if (failed(result)) { analysis.insertionError = analysis.state->error; }
    else {
        (*result)->allocationCertificate = regionalAllocationCertificate(sequenceRegionalResult(analysis), **result);
        if (!(*result)->allocationCertificate) {
            (*result)->allocationCertificate =
                finiteRegionalAllocationCertificate(sequenceRegionalResult(analysis), **result);
        }
    }
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
    std::string& error, SequenceCost* cost)
{
    auto analysis = analyzeSequence(function, input, program);
    if (!analysis.error.empty()) { error = analysis.error; return failure(); }
    auto prepared = prepareSequenceInsertion(analysis);
    error = failed(prepared) ? analysis.insertionError : analysis.error;
    if (cost) { *cost = analysis.cost; }
    return prepared;
}
} // namespace mlir::pto::frontiersynch
