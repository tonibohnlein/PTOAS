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
        composer.costs.physicalFragments += cost.physicalFragments;
        composer.costs.rotatingResidues += cost.rotatingResidues;
        composer.costs.numericVisits += cost.numericVisits;
        composer.costs.selectorComparisons += cost.selectorComparisons;
        composer.costs.crossingCandidates += cost.crossingCandidates;
        composer.costs.implicationChecks += cost.implicationChecks;
    }
    if (!composer.importSummaries()) { result.error = composer.error; return result; }
    composer.bridges();
    if (!composer.valueBridges()) { result.error = composer.error; return result; }
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
        const auto& child = composer.children[childId].regional;
        for (const auto& [pipe, values] : child.firstPayloads) {
            auto seen = composer.no(), nonempty = composer.no();
            for (auto previous : result.firstPayloads[pipe]) { seen = composer.either(seen, previous.present); }
            for (auto selected : values) {
                nonempty = composer.either(nonempty, selected.present);
                auto event = composer.port(childId, selected.event);
                result.firstPayloads[pipe].push_back({event,
                    composer.both(selected.present, composer.negate(seen))});
            }
            mask(result.lastPayloads[pipe], composer.negate(nonempty));
            auto found = child.lastPayloads.find(pipe);
            if (found != child.lastPayloads.end()) {
                for (auto selected : found->second) {
                    result.lastPayloads[pipe].push_back({composer.port(childId, selected.event), selected.present});
                }
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
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions)
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
    auto state = std::make_shared<SequenceAnalysisState>(function, std::move(expressions));
    state->input = &input;
    state->program = &program;
    state->completeInvocation = node == 0;
    state->requiresOuterBinding = node < program.nodes.size() && !program.nodes[node].loops.empty();
    if (!state->collect(node) || !state->partition()) { result.error = state->error; return result; }
    state->summarize();
    state->bindAdapters();
    return finishSequence(std::move(state));
}
SequenceAnalysis composeRegionalSequence(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children)
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
    SmallVector<const CompoundInstanceElement*> phases;
    DenseSet<const CompoundInstanceElement*> seen;
    for (const auto& regional : children) {
        for (const auto& anchor : regional.anchors) {
            if (anchor.phase && seen.insert(anchor.phase).second) {
                phases.push_back(anchor.phase);
            }
        }
    }
    if (failed(state->index.build(function, phases))) {
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
    return analysis.state->graph[source][target];
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
    auto sourcePort = state.portIds.find({source.child, source.type, source.ordinal, source.visits});
    auto targetPort = state.portIds.find({target.child, target.type, target.ordinal, target.visits});
    if (sourcePort != state.portIds.end() && targetPort != state.portIds.end()) {
        return state.graph[2*sourcePort->second + (source.kind == PeriodicEventKind::Completion)]
                          [2*targetPort->second + (target.kind == PeriodicEventKind::Completion)];
    }
    auto local = [&](SequenceEvent a, SequenceEvent b) -> std::optional<Expr> {
        if (a.child != b.child) { return state.no(); }
        return regionalReachability(state.children[a.child].regional,
            {a.type, a.ordinal, a.kind, a.visits}, {b.type, b.ordinal, b.kind, b.visits});
    };
    if (source.child == target.child) { return local(source, target); }
    if (source.child > target.child) { return state.no(); }
    Expr result = state.no();
    std::vector<std::array<Expr, 2>> firsts(analysis.occurrences.size(), {state.no(), state.no()});
    std::vector<std::array<Expr, 2>> lasts(analysis.occurrences.size(), {state.no(), state.no()});
    for (uint32_t id = 0; id < analysis.occurrences.size(); ++id) {
        const auto& port = state.ports[id];
        for (unsigned completion = 0; completion < 2; ++completion) {
            auto kind = completion ? PeriodicEventKind::Completion : PeriodicEventKind::Start;
            if (port.child == source.child) {
                auto answer = local(source, {port.child, port.type, port.ordinal, kind, port.visits});
                if (!answer) { return std::nullopt; } firsts[id][completion] = *answer;
            }
            if (port.child == target.child) {
                auto answer = local({port.child, port.type, port.ordinal, kind, port.visits}, target);
                if (!answer) { return std::nullopt; } lasts[id][completion] = *answer;
            }
        }
    }
    for (uint32_t a = 0; a < analysis.occurrences.size(); ++a) {
        for (uint32_t b = 0; b < analysis.occurrences.size(); ++b) {
            for (unsigned ac = 0; ac < 2; ++ac) {
                for (unsigned bc = 0; bc < 2; ++bc) {
                    result = state.either(result, state.both(firsts[a][ac],
                        state.both(state.graph[2*a+ac][2*b+bc], lasts[b][bc])));
                }
            }
        }
    }
    return result;
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
