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
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
namespace mlir::pto::frontiersynch {
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
SequenceRegionResolver FrontierAnalysis::regionalResolver()
{
    SequenceRegionResolver resolver;
    resolver.region = [this](std::size_t region, bool endpoints, std::string& error) -> FailureOr<RegionalAnalysis> {
        AnalysisRequest request;
        request.region = region;
        request.needs.queries = request.needs.selectors = true;
        request.needs.synchronization = endpoints;
        auto result = analyze(request);
        if (result.status == AnalysisStatus::Ready && result.mathematical->regionalDemands) {
            return *result.mathematical->regionalDemands;
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
    return resolver;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceRegionBackend(
    AnalysisBackend backend, std::size_t region, std::string& error)
{
    if (backend == AnalysisBackend::NumericalPeriodic || backend == AnalysisBackend::Rotating ||
        backend == AnalysisBackend::GuardedRotating || backend == AnalysisBackend::BoundedLifetime ||
        backend == AnalysisBackend::FiniteVisit) {
        return produceLoopBackend(backend, error, region);
    }
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    owned->region = region;
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
            auto regional = analyzeArithmeticRegionRetained({function, node.anchor}, *structuralIndex, *storage,
                sessionState->expressions, owned->arithmeticRegionalDemands, cached.demandError,
                recognizeArithmeticRegion(region));
            if (succeeded(regional)) {
                owned->regionalDemands = std::make_shared<const RegionalAnalysis>(std::move(*regional));
            }
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
