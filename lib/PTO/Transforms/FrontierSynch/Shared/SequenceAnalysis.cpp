// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "PrefixOccurrence.h"
#include "RegionalRelationsInternal.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalRelations.h"
namespace mlir::pto::frontiersynch {
bool sameOriginalCoordinates(const RegionalAnalysis& a, const RegionalAnalysis& b)
{
    const bool sameDomain = a.expressions == b.expressions && a.anchors.size() == b.anchors.size() &&
        a.occurrenceLoops == b.occurrenceLoops && a.accessModel == b.accessModel &&
        a.gmAliasPolicy == b.gmAliasPolicy;
    if (!sameDomain) { return false; }
    // Specialized phase views need their own explicit coordinate adapter.
    if (a.firstOrdinal != b.firstOrdinal || a.endpointSiteGuard != b.endpointSiteGuard ||
        a.endpointInvocationGuard != b.endpointInvocationGuard) { return false; }
    for (std::size_t i = 0; i < a.anchors.size(); ++i) {
        const auto& x = a.anchors[i];
        const auto& y = b.anchors[i];
        if (x.phase != y.phase || x.before.block != y.before.block || x.before.before != y.before.before ||
            x.after.block != y.after.block || x.after.before != y.after.before ||
            x.coordinates.size() != y.coordinates.size()) { return false; }
        for (std::size_t j = 0; j < x.coordinates.size(); ++j) {
            if (x.coordinates[j].loop != y.coordinates[j].loop ||
                x.coordinates[j].induction != y.coordinates[j].induction) { return false; }
        }
        const auto ax = a.outerLoops.empty() ? ArrayRef<scf::ForOp>() : ArrayRef<scf::ForOp>(a.outerLoops[i]);
        const auto bx = b.outerLoops.empty() ? ArrayRef<scf::ForOp>() : ArrayRef<scf::ForOp>(b.outerLoops[i]);
        if (ax != bx) { return false; }
        for (std::size_t j = 0; j < ax.size(); ++j) {
            const auto ad = a.outerDivisors.empty() ? 1 : a.outerDivisors[i][j];
            const auto bd = b.outerDivisors.empty() ? 1 : b.outerDivisors[i][j];
            if (ad != bd) { return false; }
        }
    }
    return true;
}
namespace {
SequenceAnalysis finishSequence(std::shared_ptr<SequenceAnalysisState> state)
{
    SequenceAnalysis result;
    auto& composer = *state;
    composer.costs = {};
    for (const auto& child : composer.children) {
        const auto& cost = child.regional.cost;
        for (auto member : {&RegionalCost::arithmeticRegions, &RegionalCost::boundaryBytes,
            &RegionalCost::retainedExpressionNodes, &RegionalCost::physicalFragments,
            &RegionalCost::rotatingResidues, &RegionalCost::numericVisits, &RegionalCost::repeatedRegions,
            &RegionalCost::phaseDescriptions, &RegionalCost::selectorComparisons, &RegionalCost::crossingCandidates,
            &RegionalCost::implicationChecks, &RegionalCost::selectorPartitionVisits,
            &RegionalCost::selectorPartitionAtoms, &RegionalCost::numericalLeafQueries,
            &RegionalCost::numericalIndexOperations, &RegionalCost::numericalMerges,
            &RegionalCost::numericalReusedChildren}) {
            accumulateCost(composer.costs.*member, cost.*member);
        }
    }
    if (!composer.importSummaries(composer.requireEndpoints)) {
        const bool symbolic = llvm::any_of(composer.children, [](const Child& child) {
            return !child.regional.symbolicStorageEffects.empty();
        });
        const SyncInput* input = composer.input;
        for (const auto& child : composer.children) {
            if (!input && child.regional.relations) { input = child.regional.relations->data.input; }
            if (!input && child.regional.arithmeticRelations) { input = child.regional.arithmeticRelations->input; }
        }
        if (symbolic && input && !composer.children.empty()) {
            std::vector<RegionalAnalysis> children;
            for (const auto& child : composer.children) { children.push_back(child.regional); }
            std::string diagnostic;
            if (!composer.indexReady) {
                auto facts = composer.index.sourceFactsFor(composer.function, input->instructions());
                if (failed(composer.index.build(composer.function, input->instructions(), std::move(facts)))) {
                    result.error = "symbolic composition shared prerequisite index unavailable"; return result;
                }
                composer.indexReady = true;
            }
            ++composer.relationalCompositions;
            auto composed = composeSymbolicRegionalSequence(children, composer.function, *input,
                composer.index, diagnostic, composer.allocationPolicy);
            if (succeeded(composed) && (!composed->capabilities.endpointRecipes ||
                (!composed->prepare && !composed->prepareWithVisits))) {
                result.insertionError = composed->relations && !composed->relations->exportObligation.empty() ?
                    composed->relations->exportObligation : "symbolic sequence endpoint recipes are unavailable";
            }
            if (succeeded(composed)) {
                composer.error.clear();
                composer.relationalResult = std::move(*composed);
                result.cost = composer.relationalResult->cost;
                result.state = std::move(state);
                return result;
            }
            if (!diagnostic.empty()) { composer.error += "; " + diagnostic; }
        }
        result.error = composer.error;
        return result;
    }
    composer.bridges();
    if (composer.reconstructPrerequisites && !composer.valueBridges()) { result.error = composer.error; return result; }
    if (!composer.closure()) {
        // Retain failed construction records without publishing a mathematical
        // interface. Every query/preparation adapter rejects result.error.
        result.error = !composer.error.empty() ? composer.error :
            (!composer.expressions.constructionError().empty() ? composer.expressions.constructionError() :
             "sequence crossing construction failed");
        result.cost = composer.costs;
        composer.resolveOriginal = {};
        result.state = std::move(state);
        return result;
    }
    if (!composer.expressions.constructionError().empty()) {
        result.error = composer.expressions.constructionError(); return result;
    }
    for (const auto& port : composer.ports) {
        const auto& child = composer.children[port.child];
        result.occurrences.push_back({port.child, port.type, child.anchors[port.type],
                                      child.regional.occurrenceLoops[port.type], port.ordinal, port.visits});
    }
    if (composer.numericalBoundary) {
        result.storageBoundary = composer.numericalBoundary->storageBoundary;
        result.firstPayloads = composer.numericalBoundary->firstPayloads;
        result.lastPayloads = composer.numericalBoundary->lastPayloads;
    } else { composer.summarizeBoundaries(result); }
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
void SequenceAnalysisState::summarizeBoundaries(SequenceAnalysis& result)
{
    auto& composer = *this;
    // First and last boundaries are independent prefix/suffix selections.
    // Visit each child alternative a bounded number of times; do not remask
    // accumulated output or assume an order among simultaneous envelopes.
    composer.boundarySummaryVisits = 0;
    auto lookup = [&](const auto& seen, uint32_t pipe) {
        auto found = seen.find(pipe);
        return found == seen.end() ? composer.no() : found->second;
    };
    auto append = [&](auto& destination, const auto& source, Expr masked, bool reverse) {
        auto one = [&](const auto& value) {
            ++composer.boundarySummaryVisits;
            auto present = composer.both(value.present, composer.negate(masked));
            if (composer.expressions.constantValue(present) != 0) { destination.push_back({value.port, present}); }
        };
        if (reverse) { for (const auto& value : llvm::reverse(source)) { one(value); } }
        else { for (const auto& value : source) { one(value); } }
    };
    auto any = [&](const auto& values) {
        auto result = composer.no();
        for (const auto& value : values) {
            ++composer.boundarySummaryVisits;
            result = composer.either(result, value.present);
        }
        return result;
    };
    for (uint32_t cell = 0; cell < composer.cells.size(); ++cell) {
        SequenceStorageBoundary out;
        out.cell = composer.cells[cell];
        auto seenWriter = composer.no();
        std::map<uint32_t, Expr> seenReaders;
        for (const auto& [child, local] : composer.boundaries[cell]) {
            append(out.firstWriters, local.firstWriters, seenWriter, false);
            for (const auto& [pipe, reads] : local.firstReaders) {
                auto seen = lookup(seenReaders, pipe);
                append(out.firstReaders[pipe], reads, composer.either(seenWriter, seen), false);
                seenReaders[pipe] = composer.either(seen, any(reads));
            }
            seenWriter = composer.either(seenWriter, any(local.firstWriters));
        }
        seenWriter = composer.no(); seenReaders.clear();
        for (const auto& [child, local] : llvm::reverse(composer.boundaries[cell])) {
            append(out.lastWriters, local.lastWriters, seenWriter, true);
            for (const auto& [pipe, reads] : local.lastReaders) {
                auto seen = lookup(seenReaders, pipe);
                append(out.lastReaders[pipe], reads, composer.either(seenWriter, seen), true);
                seenReaders[pipe] = composer.either(seen, any(reads));
            }
            seenWriter = composer.either(seenWriter, any(local.firstWriters));
        }
        std::reverse(out.lastWriters.begin(), out.lastWriters.end());
        for (auto& [pipe, reads] : out.lastReaders) { std::reverse(reads.begin(), reads.end()); }
        result.storageBoundary.push_back(std::move(out));
    }
    std::map<uint32_t, Expr> seenPipes;
    for (uint32_t child = 0; child < composer.children.size(); ++child) {
        for (const auto& [pipe, values] : composer.nativeFirst[child]) {
            auto seen = lookup(seenPipes, pipe);
            append(result.firstPayloads[pipe], values, seen, false);
            seenPipes[pipe] = composer.either(seen, any(values));
        }
    }
    seenPipes.clear();
    for (std::size_t child = composer.children.size(); child > 0; --child) {
        for (const auto& [pipe, values] : composer.nativeFirst[child - 1]) {
            auto seen = lookup(seenPipes, pipe);
            auto found = composer.nativeLast[child - 1].find(pipe);
            if (found != composer.nativeLast[child - 1].end()) {
                append(result.lastPayloads[pipe], found->second, seen, true);
            }
            seenPipes[pipe] = composer.either(seen, any(values));
        }
    }
    for (auto& [pipe, values] : result.lastPayloads) { std::reverse(values.begin(), values.end()); }
}
SequenceAnalysis analyzeSequence(func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    return analyzeSequenceRegion(function, input, program, 0, std::make_shared<RegionExpressions>());
}
void recordSequenceContractAttempt(ProgramRecognition& program, const SyncInput& input,
                                   const SequenceAnalysis& analysis)
{
    ProgramContractCandidate candidate;
    candidate.kind = ContractClass::Sequence;
    candidate.node = 0;
    candidate.membership = ContractStatus::Unproved;
    candidate.demands = ContractImplementation::Unavailable;
    candidate.implementationError = analysis.error;
    const auto& state = analysis.state;
    const bool original = state && state->program == &program && state->input == &input &&
        state->completeInvocation && state->reconstructPrerequisites && state->requiredOuterLoops.empty() &&
        !program.nodes.empty() && program.nodes[0].kind == StructureKind::Sequence && !program.nodes[0].parent &&
        program.nodes[0].anchor == state->function.getOperation() &&
        program.nodes[0].region == &state->function.getBody();
    const bool retainedRelations = state && state->relationalResult && state->relationalResult->relations &&
        state->relationalResult->relations->data.completeRequiredOrder;
    const bool established = original && analysis.error.empty() && (retainedRelations ||
        (state->error.empty() && state->expressions.constructionError().empty()));
    // A failed endpoint/export attempt cannot erase an already owned exact
    // relational result. Keep its diagnostic separate from class membership.
    if (established && !analysis.insertionError.empty()) {
        candidate.implementationError = analysis.insertionError;
        candidate.endpointRecipes = ContractImplementation::Unavailable;
    }
    // finishSequence publishes a state only after importing exact child
    // queries/selectors, reconciling shared storage, reconstructing original
    // value prerequisites and completing the crossing closure. Composed views
    // lack the original program/input identities and cannot prove this claim.
    for (const char* obligation : {"original-structured-occurrence-tree", "shared-context-exact-child-queries",
        state && state->partialStorage() ? "complete-local-boundaries-and-certified-external-omissions" :
            "complete-storage-boundary-selectors", "native-first-last-and-supplied-prerequisite-coverage"}) {
        candidate.obligations.push_back({obligation,
            established ? ContractStatus::Established : ContractStatus::Unproved});
    }
    if (established) {
        candidate.membership = ContractStatus::Established;
        candidate.demands = ContractImplementation::Available;
    } else if (candidate.implementationError.empty()) {
        candidate.implementationError = original && !state->error.empty() ? state->error :
            "sequence result does not certify the original complete invocation before endpoint preparation";
    }
    program.sequenceContract = std::move(candidate);
    refreshProgramContractAudit(program);
}
SequenceAnalysis analyzeSequenceRegion(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<PhaseIndex> sharedIndex, bool requireEndpoints)
{
    return analyzeSequenceRegionWithResolver(function, input, program, node, std::move(expressions),
                                             std::move(sharedIndex), requireEndpoints, {});
}
SequenceAnalysis analyzeSequenceRegionWithResolver(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<PhaseIndex> sharedIndex, bool requireEndpoints, SequenceRegionResolver resolver)
{
    SequenceAnalysis result;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !expressions ||
        !expressions->constructionError().empty()) {
        result.error = "sequence route requires a single structured function body"; return result;
    }
    auto state = std::make_shared<SequenceAnalysisState>(function, std::move(expressions),
        std::move(sharedIndex), program.allocationPolicy);
    state->input = &input;
    state->finiteInput = resolver.finiteInput;
    if (state->finiteInput && &state->finiteInput->storage() != &input) {
        result.error = "finite sequence input belongs to different source facts"; return result;
    }
    state->constructionOrigin = "requested-region";
    state->diagnosticRegion = node;
    state->originalRegionId = node;
    state->program = &program;
    state->resolveOriginal = std::move(resolver);
    state->completeInvocation = node == 0;
    // Mathematical session requests pass false. A distinct stronger snapshot
    // requests endpoint-capable children before reducing complete crossings,
    // including at the root, without replacing the earlier exact mathematics.
    state->requireEndpoints = requireEndpoints;
    if (node < program.nodes.size()) {
        state->requiredOuterLoops = program.nodes[node].loops;
        state->constructionAnchor = program.nodes[node].anchor;
    }
    if (!state->collect(node) || !state->partition()) { result.error = state->error; return result; }
    // Construction callbacks borrow the session; retained results must not.
    state->resolveOriginal = {};
    state->summarize();
    state->bindAdapters();
    return finishSequence(std::move(state));
}
std::optional<SequenceAnalysis> upgradeSequenceEndpoints(const SequenceAnalysis& retained,
    std::size_t node, SequenceRegionResolver resolver)
{
    const auto& source = retained.state;
    if (!source || !retained.error.empty() || !source->program || !source->input || !resolver ||
        source->originalRegionId != node || node >= source->program->nodes.size()) {
        return std::nullopt;
    }
    if (source->relationalResult && (!source->relationalResult->relations ||
        !source->relationalResult->relations->preparation)) { return std::nullopt; }
    const bool conditional = source->program->nodes[node].kind == StructureKind::Conditional;
    if ((!conditional && source->program->nodes[node].kind != StructureKind::Sequence) ||
        llvm::any_of(source->children, [&](const Child& child) {
            return child.endpointRepresentationUpgrade ||
                (child.originalArmGuard && !conditional) || (!child.originalNode &&
                (!child.regional.capabilities.endpointRecipes ||
                 (!child.regional.prepare && !child.regional.prepareWithVisits)));
        })) { return std::nullopt; }
    SequenceAnalysis result;
    auto children = source->children;
    bool changed = false;
    for (auto& child : children) {
        if (!child.originalNode) { ++result.upgradeWork.retainedChildren; continue; }
        Region* arm = nullptr;
        if (conditional) {
            if (!child.originalArmGuard || *child.originalNode >= source->program->nodes.size()) {
                return std::nullopt;
            }
            arm = source->program->nodes[*child.originalNode].region;
            auto branch = dyn_cast_or_null<scf::IfOp>(source->constructionAnchor);
            if (!arm || !branch || (arm != &branch.getThenRegion() && arm != &branch.getElseRegion())) {
                return std::nullopt;
            }
            const auto condition = source->expressions.input(branch.getCondition());
            const auto expected = arm == &branch.getThenRegion() ? condition : source->expressions.lnot(condition);
            if (*child.originalArmGuard != expected) { return std::nullopt; }
        }
        std::string error;
        auto upgraded = resolver(*child.originalNode, true, error);
        if (failed(upgraded)) { result.error = "sequence endpoint child upgrade: " + error; return result; }
        if (!upgraded->capabilities.endpointRecipes ||
            (!upgraded->prepare && !upgraded->prepareWithVisits)) {
            result.error = "sequence endpoint child upgrade lacks an owned preparation recipe"; return result;
        }
        if (arm) {
            if (upgraded->expressions != source->arena) { return std::nullopt; }
            // Retain the old guarded mathematics. Only the new preparation
            // callbacks need the original arm mask: the default path invokes
            // these wrappers once; resolver-based preparation masks its fresh
            // raw arm plan once instead of invoking the stored callbacks.
            *upgraded = guardOriginalArm(std::move(*upgraded), *child.originalArmGuard, *arm);
        }
        if (sameOriginalCoordinates(child.regional, *upgraded)) {
            // Only preparation is an upgrade. The original semantic callbacks,
            // selected occurrences and all crossing expressions stay unchanged.
            child.regional.prepare = upgraded->prepare;
            child.regional.prepareWithVisits = upgraded->prepareWithVisits;
            child.regional.prepareFiltered = upgraded->prepareFiltered;
            child.regional.capabilities.endpointRecipes = true;
            ++result.upgradeWork.retainedChildren;
        } else {
            // A changed guarded representation needs the existing conditional
            // producer to re-establish its arm-specific composition contract.
            if (conditional || source->relationalResult) { return std::nullopt; }
            Child replacement;
            replacement.originalNode = child.originalNode;
            replacement.prefixMathematics = upgraded->retainedMathematics.lock();
            replacement.anchors = upgraded->anchors;
            replacement.regional = std::move(*upgraded);
            child = std::move(replacement);
            ++result.upgradeWork.replacedChildren;
            changed = true;
        }
    }
    auto work = result.upgradeWork;
    if (!changed) {
        auto state = std::make_shared<SequenceAnalysisState>(*source);
        state->upgradeSource = source;
        state->children = std::move(children);
        state->relationalCompositions = 0;
        if (source->relationalResult) {
            std::vector<RegionalAnalysis> preparationChildren;
            for (const auto& child : state->children) { preparationChildren.push_back(child.regional); }
            auto rebound = rebindRegionalRelationPreparation(*source->relationalResult, preparationChildren);
            if (failed(rebound)) { return std::nullopt; }
            state->relationalResult = std::move(*rebound);
            work.reusedRelationalResults = 1;
        }
        state->requireEndpoints = true;
        state->resolveOriginal = {};
        state->childPreparationOperations = state->crossingPreparationOperations = 0;
        state->constructionOrigin = "sequence-endpoint-upgrade";
        state->crossingPresence.reset();
        if (source->hasCertifiedCrossingPresence()) { state->sealCrossingPresence(); }
        result = retained;
        result.insertionError.clear();
        result.state = std::move(state);
        work.reusedCrossings = source->crossings.size();
    } else {
        // Changed coordinates invalidate the global partition and transitive
        // closure, even between unchanged endpoints. Child mathematics does
        // not depend on that parent construction and remains owned/reused.
        auto state = std::make_shared<SequenceAnalysisState>(source->function, source->arena,
            source->indexOwner, source->allocationPolicy);
        state->upgradeSource = source;
        state->input = source->input; state->program = source->program;
        state->finiteInput = source->finiteInput;
        state->completeInvocation = source->completeInvocation;
        state->reconstructPrerequisites = source->reconstructPrerequisites;
        state->requiredOuterLoops = source->requiredOuterLoops;
        state->constructionOrigin = "sequence-endpoint-recomposition";
        state->diagnosticRegion = source->diagnosticRegion;
        state->originalRegionId = source->originalRegionId;
        state->constructionAnchor = source->constructionAnchor;
        state->children = std::move(children);
        result = finishSequence(std::move(state));
        if (result.state) { work.recomposedCrossings = result.state->crossings.size(); }
    }
    result.upgradeWork = work;
    return result;
}
SequenceAnalysis composeRegionalSequence(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children,
    bool reconstructPrerequisites, bool requireEndpoints,
    std::shared_ptr<const PhaseSourceFacts> sourceFacts, ArithmeticRegionalAllocationPolicy allocationPolicy)
{
    return composeRegionalSequenceWithin(function, std::move(expressions), std::move(children),
        reconstructPrerequisites, requireEndpoints, {}, std::move(sourceFacts), allocationPolicy);
}
SequenceAnalysis composeRegionalSequenceWithin(func::FuncOp function,
    std::shared_ptr<RegionExpressions> expressions, std::vector<RegionalAnalysis> children,
    bool reconstructPrerequisites, bool requireEndpoints, ArrayRef<scf::ForOp> enclosing,
    std::shared_ptr<const PhaseSourceFacts> sourceFacts, ArithmeticRegionalAllocationPolicy allocationPolicy)
{
    SequenceAnalysis result;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock() || !expressions ||
        !expressions->constructionError().empty()) {
        result.error = "sequence requires a valid common regional expression arena"; return result;
    }
    auto state = std::make_shared<SequenceAnalysisState>(function, std::move(expressions),
        std::shared_ptr<PhaseIndex>{}, allocationPolicy);
    state->requiredOuterLoops.assign(enclosing.begin(), enclosing.end());
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
    // Retain supplied source facts even when value links were supplied by the
    // caller. An empty local phase view adds no prerequisite edges; a later
    // symbolic fallback can still reuse discovery for its full-input view.
    if (reconstructPrerequisites || sourceFacts) {
        ArrayRef<const CompoundInstanceElement*> selected = reconstructPrerequisites ?
            ArrayRef<const CompoundInstanceElement*>(phases) : ArrayRef<const CompoundInstanceElement*>{};
        if (failed(state->index.build(function, selected, std::move(sourceFacts)))) {
            result.error = "cannot reconstruct regional value prerequisites";
            return result;
        }
    }
    for (auto& regional : children) {
        Child child;
        child.prefixMathematics = regional.retainedMathematics.lock();
        child.regional = std::move(regional);
        state->children.push_back(std::move(child));
    }
    return finishSequence(std::move(state));
}
NumericalChainQueryCost sequenceNumericalQueryCounts(const SequenceAnalysis& analysis)
{
    return analysis.state ? analysis.state->numericalQueryCost : NumericalChainQueryCost{};
}
uint64_t sequenceBoundarySummaryVisits(const SequenceAnalysis& analysis)
{
    return analysis.state ? analysis.state->boundarySummaryVisits : 0;
}
SequenceBridgeWork sequenceBridgeWork(const SequenceAnalysis& analysis)
{
    return analysis.state ? analysis.state->bridgeWork : SequenceBridgeWork{};
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
    if (!analysis.state || !analysis.error.empty() || !validKind(sourceKind) || !validKind(targetKind) ||
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
    if (state.relationalResult) {
        auto map = [&](const SequenceEvent& event) {
            uint32_t type = event.type;
            for (uint32_t i = 0; i < event.child; ++i) { type += state.children[i].regional.anchors.size(); }
            return RegionalEvent{type, event.ordinal, event.kind, event.visits};
        };
        return regionalReachability(*state.relationalResult, map(source), map(target));
    }
    return state.eventReachability(std::move(source), std::move(target));
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceLogicalInsertion(SequenceAnalysis& analysis,
    ArrayRef<scf::ForOp> enclosing, const SequenceEndpointResolver& resolver)
{
    if (!analysis.state || !analysis.error.empty()) { return failure(); }
    std::string diagnostic;
    auto result = analysis.state->prepare(diagnostic, enclosing, resolver);
    analysis.insertionError = failed(result) ? std::move(diagnostic) : std::string{};
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareSequenceInsertion(SequenceAnalysis& analysis)
{
    auto result = prepareSequenceLogicalInsertion(analysis);
    if (succeeded(result)) { prepareAllocationSupport(**result); }
    const bool needsAllocation = succeeded(result) && !(*result)->allocationCertificate;
    if (needsAllocation) {
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
