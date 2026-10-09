// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Existing mathematical producers behind the session request protocol. No
// endpoint code or allocation is constructed by these adapters.
#include "AnalysisSessionInternal.h"
#include "FiniteExpansionPlan.h"
#include "NumericTemplatePlan.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRecognition.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRegional.h"
namespace mlir::pto::frontiersynch {
uint64_t FrontierAnalysis::finiteExpansionPreflights() const
{
    return sessionState ? sessionState->finiteExpansionPlans.size() : 0;
}
uint64_t FrontierAnalysis::numericTemplatePreflights() const
{
    return sessionState ? sessionState->numericTemplatePlans.size() : 0;
}
const NumericTemplate& FrontierAnalysis::materializeNumericRegion(std::size_t region)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    auto& node = program->nodes[region];
    if (node.numericTemplate) { return *node.numericTemplate; }
    auto found = sessionState->numericTemplatePlans.find(region);
    if (found == sessionState->numericTemplatePlans.end()) {
        auto plan = std::make_shared<const NumericTemplatePlan>(
            preflightNumericTemplate(cast<scf::ForOp>(node.anchor), *structuralIndex, *storage));
        found = sessionState->numericTemplatePlans.emplace(region, std::move(plan)).first;
    }
    node.numericTemplate = materializeNumericTemplate(*found->second, *structuralIndex, *storage);
    return *node.numericTemplate;
}
void FrontierAnalysis::recordWholeRegion(AnalysisBackend backend)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    sessionState->wholeRegionEvidence.insert(backend);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareRotatingFunction(bool guarded)
{
    if (failed(recognizeStructure())) { return failure(); }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    auto result = requestBackend(guarded ? AnalysisBackend::GuardedRotating : AnalysisBackend::Rotating, {});
    if (result.status != AnalysisStatus::Ready) { return failure(); }
    auto prepared = prepareLogical(result);
    if (succeeded(prepared)) { (void)attachAllocation(result, **prepared); }
    return prepared;
}
namespace {
const FiniteExpansionPlan& expansionPlan(AnalysisSessionState& session, func::FuncOp function,
    const ProgramRecognition& program, std::size_t region, const PhaseIndex& index, const SyncInput& input)
{
    auto found = session.finiteExpansionPlans.find(region);
    if (found != session.finiteExpansionPlans.end()) { return *found->second; }
    ArithmeticRegionContext context{function, function};
    if (region) {
        const auto& node = program.nodes[region];
        if (node.kind == StructureKind::ExplicitRun) { context.roots = node.operations; }
        else if (node.kind == StructureKind::Sequence && node.region && node.region->hasOneBlock()) {
            for (auto& operation : node.region->front()) {
                if (!operation.hasTrait<OpTrait::IsTerminator>()) { context.roots.push_back(&operation); }
            }
        } else if (node.kind == StructureKind::Loop || node.kind == StructureKind::Conditional) {
            context.roots.push_back(node.anchor);
        }
        context.root = context.roots.empty() ? nullptr : context.roots.front();
    }
    auto plan = std::make_shared<const FiniteExpansionPlan>(preflightFiniteExpansion(context, index, input));
    auto inserted = session.finiteExpansionPlans.emplace(region, std::move(plan));
    return *inserted.first->second;
}
const StructureNode* wholeLoop(func::FuncOp function, const SyncInput& input,
                              const ProgramRecognition& program, const PhaseIndex& index)
{
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        const bool rootLoop = node.kind == StructureKind::Loop && node.anchor->getParentOp() == function;
        if (!rootLoop) { continue; }
        if (selected || node.unsupportedContext) { return nullptr; }
        selected = &node;
    }
    if (!selected || llvm::any_of(input.instructions(), [&](const auto* phase) {
            return !selected->anchor->isProperAncestor(phase->elementOp);
        })) { return nullptr; }
    RecognitionResult outside;
    for (auto& operation : function.front()) {
        if (&operation == selected->anchor) { continue; }
        if (operation.getNumRegions()) { return nullptr; }
        detail::inspectLeaf(operation, index, outside);
    }
    return outside.state == RecognitionState::Applicable ? selected : nullptr;
}
} // namespace
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceLoopBackend(
    AnalysisBackend backend, std::string& error, std::size_t region)
{
    const auto* node = region == 0 ? wholeLoop(function, *storage, *program, *structuralIndex) :
                                    &program->nodes[region];
    if (!node || node->kind != StructureKind::Loop || node->unsupportedContext) {
        error = "route requires an original certified loop"; return {};
    }
    const auto original = static_cast<std::size_t>(node - program->nodes.data());
    auto& attempt = sessionState->loopAttempts[{original, backend}];
    if (!attempt.produced) {
        attempt.produced = true;
        attempt.mathematical = constructLoopBackend(backend, original, attempt.demandError);
    }
    error = attempt.demandError;
    if (!attempt.mathematical) { return {}; }
    auto result = std::make_shared<MathematicalResult>(*attempt.mathematical);
    result->region = region;
    if (region == 0 && result->boundedDemands) { boundedAnalysis = result->boundedDemands; }
    return result;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::constructLoopBackend(
    AnalysisBackend backend, std::size_t region, std::string& error)
{
    const auto* node = &program->nodes[region];
    const auto loop = cast<scf::ForOp>(node->anchor);
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    owned->region = region;
    switch (backend) {
    case AnalysisBackend::VaryingBoundary: {
        auto& local = program->nodes[region];
        if (!local.varyingRotating || local.varyingRotating->result.state != RecognitionState::Applicable) {
            error = "affine repeating-boundary form is not established";
            return {};
        }
        if (!local.varyingDemands) {
            if (construction.varyingBoundaryReductions != UINT64_MAX) { ++construction.varyingBoundaryReductions; }
            local.varyingDemands = analyzeVaryingRotating(*local.varyingRotating, *structuralIndex, *storage);
        }
        if (!local.varyingDemands->error.empty()) {
            error = local.varyingDemands->error;
            return {};
        }
        // The unchanged recognition tree owns this once-assigned certificate.
        // An aliasing owner avoids copying the quotient and crossing tables.
        owned->varyingBoundaryDemands = std::shared_ptr<const AffineRotatingVisits>(program, &*local.varyingDemands);
        owned->varyingNode = region;
        owned->backend = "repeating-boundary-interface";
        break;
    }
    case AnalysisBackend::FiniteVisit: {
        auto found = finiteVisitAnalyses.find(region);
        if (found == finiteVisitAnalyses.end()) {
            auto analyzed = std::make_shared<FiniteVisitAnalysis>(
                analyzeFiniteVisitLoop(function, *storage, *program, region, structuralIndex));
            found = finiteVisitAnalyses.emplace(region, std::move(analyzed)).first;
        }
        const auto& result = *found->second;
        for (auto& candidate : program->finiteVisitContracts) {
            if (candidate.node == region) { candidate = result.recognition.contract; }
        }
        refreshProgramContractAudit(*program);
        const bool exact = result.demands && result.demands->error.empty() &&
            result.recognition.contract.membership == ContractStatus::Established &&
            result.recognition.contract.demands == ContractImplementation::Available;
        if (!exact) {
            error = result.recognition.contract.implementationError;
            return {};
        }
        owned->finiteVisitDemands = found->second;
        owned->backend = "finite-visit-types";
        break;
    }
    case AnalysisBackend::NumericalPeriodic: {
        auto& local = program->nodes[region];
        (void)materializeNumericRegion(region);
        if (local.numericTemplate && local.numericTemplate->result.state == RecognitionState::Applicable &&
            !local.periodicAnalysis) {
            local.periodicAnalysis = analyzeNumericTemplate(*local.numericTemplate);
        }
        const bool available = node->numericTemplate && node->periodicAnalysis &&
            node->periodicAnalysis->error.empty() && !node->numericTemplate->specializedBody;
        if (!available) { return {}; }
        owned->numericNode = static_cast<std::size_t>(node - program->nodes.data());
        owned->backend = "numerical-periodic";
        break;
    }
    case AnalysisBackend::Rotating: {
        if (!node->rotatingResult || node->rotatingResult->state != RecognitionState::Applicable) { return {}; }
        if (construction.rotatingReductions != UINT64_MAX) { ++construction.rotatingReductions; }
        auto demands = std::make_shared<RotatingAnalysis>(
            analyzeRotating(loop, *structuralIndex, *storage, *node->rotatingResult, false));
        if (!demands->error.empty()) { error = demands->error; return {}; }
        owned->rotatingDemands = std::move(demands);
        owned->backend = "rotating";
        break;
    }
    case AnalysisBackend::GuardedRotating: {
        if (!node->guardedRotatingResult ||
            node->guardedRotatingResult->result.state != RecognitionState::Applicable) { return {}; }
        if (construction.guardedRotatingReductions != UINT64_MAX) { ++construction.guardedRotatingReductions; }
        auto demands = std::make_shared<GuardedRotatingAnalysis>(
            analyzeGuardedRotating(loop, *storage, *node->guardedRotatingResult, *structuralIndex,
                                  sessionState->expressions));
        if (!demands->error.empty()) { error = demands->error; return {}; }
        owned->guardedRotatingDemands = std::move(demands);
        owned->backend = "guarded-rotating";
        break;
    }
    case AnalysisBackend::BoundedLifetime: {
        if (!node->boundedLifetime ||
            node->boundedLifetime->skeleton.result.state != RecognitionState::Applicable) { return {}; }
        auto demands = cachedBoundedLifetimeRegion(function, *node, *structuralIndex, *storage, error);
        if (failed(demands)) { return {}; }
        owned->boundedDemands = *demands;
        owned->backend = "bounded-lifetime";
        break;
    }
    default:
        error = "backend is not a loop producer";
        return {};
    }
    return owned;
}
std::shared_ptr<const RegionalAnalysis> FrontierAnalysis::varyingExports(
    const MathematicalResult& demands, bool selectors, std::string& error)
{
    if (!demands.varyingNode || !demands.varyingBoundaryDemands) { return {}; }
    auto id = *demands.varyingNode;
    auto& attempt = sessionState->loopAttempts[{id, AnalysisBackend::VaryingBoundary}];
    if (!attempt.varyingQueriesAttempted) {
        attempt.varyingQueriesAttempted = true;
        if (construction.varyingQueryBuilds != UINT64_MAX) { ++construction.varyingQueryBuilds; }
        attempt.varyingExports = buildVaryingQueries(function, *program->nodes[id].varyingRotating,
            *structuralIndex, *storage, sessionState->expressions, demands.varyingBoundaryDemands,
            attempt.varyingQueryError, storage);
    }
    if (!attempt.varyingExports) { error = attempt.varyingQueryError; return {}; }
    if (!selectors) { return varyingQueryResult(*attempt.varyingExports); }
    if (!attempt.varyingSelectorsAttempted) {
        attempt.varyingSelectorsAttempted = true;
        if (construction.varyingSelectorBuilds != UINT64_MAX) { ++construction.varyingSelectorBuilds; }
    }
    return varyingSelectedResult(*attempt.varyingExports, *storage, error);
}
std::shared_ptr<const RegionalAnalysis> FrontierAnalysis::arithmeticExports(
    const MathematicalResult& demands, bool selectors, std::string& error)
{
    if (!demands.arithmeticRegionalDemands) { return {}; }
    auto& attempt = sessionState->arithmeticRegionAttempts[demands.region];
    if (!attempt.arithmeticQueriesAttempted) {
        attempt.arithmeticQueriesAttempted = true;
        if (construction.arithmeticQueryBuilds != UINT64_MAX) { ++construction.arithmeticQueryBuilds; }
        auto queries = exportArithmeticRegion(*demands.arithmeticRegionalDemands,
            sessionState->expressions, false, attempt.arithmeticQueryError, storage);
        if (succeeded(queries)) {
            attempt.arithmeticQueries = std::make_shared<const RegionalAnalysis>(std::move(*queries));
        }
    }
    if (!attempt.arithmeticQueries) { error = attempt.arithmeticQueryError; return {}; }
    if (!selectors) { return attempt.arithmeticQueries; }
    if (!attempt.arithmeticSelectorsAttempted) {
        attempt.arithmeticSelectorsAttempted = true;
        if (construction.arithmeticSelectorBuilds != UINT64_MAX) { ++construction.arithmeticSelectorBuilds; }
        auto selected = exportArithmeticRegion(*demands.arithmeticRegionalDemands,
            sessionState->expressions, true, attempt.arithmeticSelectorError, storage);
        if (succeeded(selected)) {
            attempt.arithmeticSelectors = std::make_shared<const RegionalAnalysis>(std::move(*selected));
        }
    }
    error = attempt.arithmeticSelectorError;
    return attempt.arithmeticSelectors;
}
SequenceRegionResolver FrontierAnalysis::regionalResolver()
{
    SequenceRegionResolver resolver;
    resolver.region = [this](std::size_t region, bool endpoints, std::string& error) -> FailureOr<RegionalAnalysis> {
        AnalysisRequest request;
        request.region = region;
        request.needs.queries = request.needs.selectors = true;
        request.needs.synchronization = endpoints;
        auto result = analyze(request);
        auto exports = result.regionalExports ? result.regionalExports :
            (result.mathematical ? result.mathematical->regionalDemands : nullptr);
        if (result.status == AnalysisStatus::Ready && exports) {
            return *exports;
        }
        for (const auto& obligation : result.obligations) {
            if (!error.empty()) { error += "; "; }
            error += obligation.diagnostic;
        }
        return failure();
    };
    resolver.demands = [this](std::size_t region, AnalysisBackend backend) {
        std::string error;
        return backend == AnalysisBackend::Arithmetic ? produceArithmeticRegion(region, error) :
                                                        produceLoopBackend(backend, error, region);
    };
    resolver.exports = [this](std::size_t region, AnalysisBackend backend, std::string& error)
        -> FailureOr<RegionalAnalysis> {
        auto demands = backend == AnalysisBackend::Arithmetic ? produceArithmeticRegion(region, error) :
            produceLoopBackend(backend, error, region);
        if (!demands) { return failure(); }
        auto result = backend == AnalysisBackend::Arithmetic ? arithmeticExports(*demands, true, error) :
            varyingExports(*demands, true, error);
        if (!result) { return failure(); }
        return *result;
    };
    return resolver;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceRegionBackend(
    AnalysisBackend backend, std::size_t region, std::string& error)
{
    if (backend == AnalysisBackend::NumericalPeriodic || backend == AnalysisBackend::Rotating ||
        backend == AnalysisBackend::GuardedRotating || backend == AnalysisBackend::BoundedLifetime ||
        backend == AnalysisBackend::FiniteVisit || backend == AnalysisBackend::VaryingBoundary) {
        return produceLoopBackend(backend, error, region);
    }
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    owned->region = region;
    if (backend == AnalysisBackend::ExpandedFinite) {
        const auto& node = program->nodes[region];
        if (node.unsupportedContext) {
            error = "finite expansion requires a supported original region"; return {};
        }
        const auto& plan = expansionPlan(*sessionState, function, *program, region, *structuralIndex, *storage);
        auto demands = std::make_shared<FiniteGuardedAnalysis>(analyzeExpandedFinite(plan, *structuralIndex, *storage));
        if (!demands->error.empty()) { error = demands->error; return {}; }
        owned->finiteGuardedDemands = std::move(demands);
        owned->backend = "expanded-finite-guarded";
        return owned;
    }
    if (backend == AnalysisBackend::Sequence) {
        auto sequence = std::make_shared<SequenceAnalysis>(analyzeSequenceRegionWithResolver(
            function, *storage, *program, region, sessionState->expressions, structuralIndex,
            false, regionalResolver()));
        if (!sequence->error.empty()) { error = sequence->error; return {}; }
        owned->regionalDemands = std::make_shared<const RegionalAnalysis>(sequenceRegionalResult(*sequence));
        owned->sequenceDemands = std::move(sequence);
        owned->backend = "sequence";
        return owned;
    }
    if (backend == AnalysisBackend::Arithmetic) { return produceArithmeticRegion(region, error); }
    error = "backend has no standalone regional adapter";
    return {};
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceArithmeticRegion(
    std::size_t region, std::string& error)
{
    auto& cached = sessionState->arithmeticRegionAttempts[region];
    if (!cached.produced) {
        cached.produced = true;
        const auto& node = program->nodes[region];
        const bool applicable = node.kind == StructureKind::Loop || node.kind == StructureKind::Conditional;
        if (!applicable || !node.anchor) {
            cached.demandError = "arithmetic region has no supported original root";
        } else {
            auto owned = std::make_shared<MathematicalResult>();
            owned->input = storage;
            owned->recognition = program;
            owned->region = region;
            owned->backend = "arithmetic";
            if (sessionState->arithmeticRegionBuilds != UINT64_MAX) { ++sessionState->arithmeticRegionBuilds; }
            owned->arithmeticRegionalDemands = analyzeArithmeticRegionDemands(
                {function, node.anchor}, *structuralIndex, *storage,
                recognizeArithmeticRegion(region), cached.demandError);
            if (owned->arithmeticRegionalDemands) { cached.mathematical = std::move(owned); }
        }
    }
    error = cached.demandError;
    return cached.mathematical;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceBackend(
    AnalysisBackend backend, std::string& error)
{
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    switch (backend) {
    case AnalysisBackend::Explicit:
        if (failed(analyzeExplicitFunction())) {
            if (nativeScalarOnly) {
                owned->explicitDemands = std::make_shared<ExplicitAnalysis>();
                owned->backend = "native-scalar";
                return owned;
            }
            error = explicitAnalysis ? explicitAnalysis->error : "explicit form unavailable";
            return {};
        }
        owned->explicitDemands = explicitAnalysis;
        owned->backend = "explicit";
        return owned;
    case AnalysisBackend::NumericalPeriodic:
    case AnalysisBackend::Rotating:
    case AnalysisBackend::GuardedRotating:
    case AnalysisBackend::BoundedLifetime:
    case AnalysisBackend::FiniteVisit:
    case AnalysisBackend::VaryingBoundary:
        return produceLoopBackend(backend, error);
    case AnalysisBackend::MixedStride: {
        auto result = analyzeMixedStrideFunction(function, *storage, *program, *structuralIndex, error);
        if (failed(result)) { return {}; }
        owned->mixedStrideDemands = *result;
        owned->backend = "mixed-stride";
        return owned;
    }
    case AnalysisBackend::CompactBounding: {
        auto result = analyzeCompactBounding(function, storage, *structuralIndex);
        if (!result.error.empty()) { error = result.error; return {}; }
        owned->compactDemands = result.owner;
        owned->backend = "compact-bounding";
        return owned;
    }
    case AnalysisBackend::Sequence: {
        const auto* sequence = analyzeSequenceFunction();
        const bool complete = sequence && sequence->error.empty() && program->sequenceContract &&
            program->sequenceContract->membership == ContractStatus::Established &&
            program->sequenceContract->demands == ContractImplementation::Available;
        if (!complete) {
            error = sequenceAnalysis ? sequenceAnalysis->error : "sequence form unavailable";
            if (error.empty()) { error = "sequence does not certify the complete original invocation"; }
            return {};
        }
        owned->sequenceDemands = sequenceAnalysis;
        owned->backend = "sequence";
        return owned;
    }
    case AnalysisBackend::ArithmeticPeriodic:
        if (failed(analyzeArithmeticPeriodicFunction())) { return {}; }
        owned->arithmeticPeriodicDemands = arithmeticPeriodicAnalysis;
        owned->backend = "arithmetic-periodic";
        return owned;
    case AnalysisBackend::Arithmetic:
        if (failed(analyzeArithmeticFunction())) { return {}; }
        owned->arithmeticDemands = arithmeticAnalysis;
        owned->generalArithmeticDemands = generalArithmeticAnalysis;
        owned->backend = "arithmetic";
        return owned;
    case AnalysisBackend::ExpandedFinite: {
        auto demands = std::make_shared<FiniteGuardedAnalysis>(
            analyzeExpandedFinite(expansionPlan(*sessionState, function, *program, 0, *structuralIndex, *storage),
                *structuralIndex, *storage));
        if (!demands->error.empty()) { error = demands->error; return {}; }
        owned->finiteGuardedDemands = std::move(demands);
        owned->backend = "expanded-finite-guarded";
        return owned;
    }
    case AnalysisBackend::FiniteGuarded: {
        SmallVector<Operation*> roots;
        for (auto& operation : function.front()) {
            if (!operation.hasTrait<OpTrait::IsTerminator>()) { roots.push_back(&operation); }
        }
        auto demands = std::make_shared<FiniteGuardedAnalysis>(
            analyzeFiniteGuarded(function, roots, *structuralIndex, *storage));
        if (!demands->error.empty()) { error = demands->error; return {}; }
        owned->finiteGuardedDemands = std::move(demands);
        owned->backend = "finite-guarded";
        return owned;
    }
    default:
        error = "unregistered analysis backend";
        return {};
    }
}
} // namespace mlir::pto::frontiersynch
