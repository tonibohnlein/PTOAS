// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Request protocol adapters. Cache mathematical construction independently of
// export checks; detached command fragments always have fresh ownership.
#include "AnalysisSessionInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticAllocation.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
AnalysisOutcome FrontierAnalysis::minimumDemands(std::size_t region, AnalysisNeeds exports)
{
    exports.synchronization = false;
    return analyze({region, AnalysisMode::MinimumExact, exports});
}
AnalysisOutcome FrontierAnalysis::requestBackend(AnalysisBackend backend, const AnalysisRequest& request)
{
    auto& attempt = sessionState->attempts[request.region][backend];
    if (!attempt.produced) {
        attempt.produced = true;
        if (construction.mathematicalAttempts != UINT64_MAX) { ++construction.mathematicalAttempts; }
        bool descends = true;
        if (!sessionState->activeRegions.empty()) {
            const auto parent = sessionState->activeRegions.back();
            auto ancestor = program->nodes[request.region].parent;
            while (ancestor && *ancestor != parent) { ancestor = program->nodes[*ancestor].parent; }
            descends = ancestor.has_value();
        }
        if (descends) {
            sessionState->activeRegions.push_back(request.region);
            attempt.mathematical = request.region == 0 ? produceBackend(backend, attempt.demandError) :
                produceRegionBackend(backend, request.region, attempt.demandError);
            sessionState->activeRegions.pop_back();
        } else { attempt.demandError = "recursive request must descend to an original child"; }
    }
    AnalysisOutcome result;
    result.mathematical = attempt.mathematical;
    if (result.mathematical && request.region == 0) { recordWholeRegion(backend); }
    if (!result.mathematical) {
        result.stage = AnalysisStage::Demands;
        result.obligations.push_back({result.stage, attempt.demandError});
        return result;
    }
    const auto& demands = *result.mathematical;
    result.available.queries = demands.explicitDemands || demands.rotatingDemands ||
        demands.guardedRotatingDemands || demands.numericNode || demands.sequenceDemands ||
        demands.finiteGuardedDemands;
    result.available.selectors = demands.explicitDemands || demands.sequenceDemands || demands.finiteGuardedDemands;
    if (demands.regionalDemands) {
        result.available.queries = demands.regionalDemands->capabilities.exactQueries;
        result.available.selectors = demands.regionalDemands->capabilities.exactSelectors;
    }
    if (demands.backend == "native-scalar") { result.available = {}; }
    result.available.synchronization = attempt.endpoints.value_or(false);
    result.status = AnalysisStatus::Ready;
    result.stage = AnalysisStage::None;
    if (request.needs.queries && !result.available.queries) { result.stage = AnalysisStage::Queries; }
    if (request.needs.selectors && !result.available.selectors) { result.stage = AnalysisStage::Selectors; }
    if (request.needs.evaluation == AnalysisEvaluation::Stateful) { result.stage = AnalysisStage::Demands; }
    if (result.stage == AnalysisStage::None && request.needs.synchronization) {
        if (!attempt.endpoints) {
            auto prepared = prepareLogical(result);
            attempt.endpoints = succeeded(prepared);
        }
        result.available.synchronization = *attempt.endpoints;
        if (!*attempt.endpoints) { result.stage = AnalysisStage::Synchronization; }
    }
    if (result.stage != AnalysisStage::None) {
        result.status = AnalysisStatus::UnmetObligation;
        result.obligations.push_back({result.stage, demands.backend + ": requested export is unavailable"});
    }
    return result;
}
AnalysisOutcome FrontierAnalysis::analyze(const AnalysisRequest& request)
{
    AnalysisOutcome result;
    if (!storage || !structuralIndex) {
        result.obligations.push_back({AnalysisStage::Form, "original region request is unavailable"});
        return result;
    }
    if (failed(recognizeStructure())) {
        result.obligations.push_back({AnalysisStage::Form, "original region request is unavailable"});
        return result;
    }
    if (request.region >= program->nodes.size()) {
        result.obligations.push_back({AnalysisStage::Form, "original region identity is invalid"});
        return result;
    }
    if (request.needs.evaluation == AnalysisEvaluation::Stateful) {
        auto uniform = request;
        uniform.needs.evaluation = AnalysisEvaluation::Uniform;
        uniform.needs.synchronization = false;
        auto retained = analyze(uniform);
        retained.status = AnalysisStatus::UnmetObligation;
        retained.stage = AnalysisStage::Demands;
        retained.obligations.push_back({retained.stage, "stateful evaluation is unavailable"});
        return retained;
    }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    for (auto backend : {AnalysisBackend::Explicit, AnalysisBackend::NumericalPeriodic,
            AnalysisBackend::Rotating, AnalysisBackend::GuardedRotating, AnalysisBackend::BoundedLifetime,
            AnalysisBackend::Sequence, AnalysisBackend::ArithmeticPeriodic, AnalysisBackend::Arithmetic,
            AnalysisBackend::FiniteGuarded}) {
        auto attempt = requestBackend(backend, request);
        if (attempt.status == AnalysisStatus::Ready) { return attempt; }
        if (!result.mathematical && attempt.mathematical) {
            result.mathematical = attempt.mathematical;
            result.available = attempt.available;
            result.status = AnalysisStatus::UnmetObligation;
            result.stage = attempt.stage;
        }
        llvm::append_range(result.obligations, attempt.obligations);
    }
    // Fallback may not weaken a retained exact result. Conservative adapters
    // are registered separately from these mathematical producers.
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareLogical(const AnalysisOutcome& result)
{
    if (!result.mathematical || !sessionState || result.mathematical->input != storage) { return failure(); }
    const bool retained = llvm::any_of(sessionState->attempts[result.mathematical->region], [&](const auto& attempt) {
        return attempt.second.mathematical == result.mathematical;
    });
    if (!retained) { return failure(); }
    if (construction.logicalPreparations != UINT64_MAX) { ++construction.logicalPreparations; }
    auto prepared = prepareRetained(*result.mathematical);
    if (succeeded(prepared)) { (*prepared)->mathematicalOwner = result.mathematical; }
    return prepared;
}
namespace {
DictionaryAttr allocateRetained(const MathematicalResult& demands, const ProgramRecognition& program,
    const BackendAttempt& attempt, PreparedLogicalPlan& prepared, MLIRContext* context)
{
    const auto plan = prepared.planId;
    if (demands.explicitDemands) { return explicitAllocationCertificate(*demands.explicitDemands, plan, context); }
    if (demands.numericNode) {
        return encodePeriodicSharedAllocation(*program.nodes[*demands.numericNode].periodicAnalysis, plan, context);
    }
    if (demands.rotatingDemands) {
        return encodePeriodicSharedAllocation(demands.rotatingDemands->periodic, plan, context);
    }
    if (demands.guardedRotatingDemands) {
        return guardedAllocationCertificate(*demands.guardedRotatingDemands, plan, context);
    }
    if (demands.arithmeticPeriodicDemands) {
        const auto& conversion = demands.arithmeticPeriodicDemands->conversion;
        return guardedPeriodicAllocationCertificate(*conversion.expressions, conversion.payloads,
            conversion.generators, *conversion.guarded, plan, context);
    }
    if (demands.sequenceDemands || demands.finiteGuardedDemands) {
        auto region = demands.sequenceDemands ? sequenceRegionalResult(*demands.sequenceDemands) :
            finiteGuardedRegionalResult(*demands.finiteGuardedDemands);
        auto certificate = regionalAllocationCertificate(region, prepared);
        return certificate ? certificate : finiteRegionalAllocationCertificate(region, prepared);
    }
    if (!demands.arithmeticDemands && !demands.generalArithmeticDemands) { return {}; }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.arithmetic->sites) {
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    if (demands.generalArithmeticDemands) {
        return generalArithmeticAllocationCertificate(*demands.generalArithmeticDemands, pipes,
            attempt.arithmeticRecords, plan, context);
    }
    auto certificate = arithmeticAllocationCertificate(*demands.arithmeticDemands, pipes,
                                                       attempt.arithmeticRecords, plan, context);
    if (certificate) { return certificate; }
    auto proof = buildArithmeticHandoffAllocation(*demands.arithmeticDemands, pipes);
    return encodeGeneralArithmeticAllocationCertificate(proof, pipes, attempt.arithmeticRecords, plan, context);
}
} // namespace
LogicalResult FrontierAnalysis::attachAllocation(const AnalysisOutcome& result, PreparedLogicalPlan& plan)
{
    if (!result.mathematical || !sessionState || result.mathematical->input != storage ||
        plan.mathematicalOwner != result.mathematical) { return failure(); }
    auto& attempts = sessionState->attempts[result.mathematical->region];
    auto selected = llvm::find_if(attempts, [&](const auto& entry) {
        return entry.second.mathematical == result.mathematical;
    });
    if (selected == attempts.end()) { return failure(); }
    if (plan.allocationCertificate) { return success(); }
    auto& attempt = selected->second;
    auto cached = attempt.allocation.find(plan.planId);
    if (cached == attempt.allocation.end()) {
        if (construction.allocationExports != UINT64_MAX) { ++construction.allocationExports; }
        auto certificate = allocateRetained(*result.mathematical, *program, attempt,
                                           plan, function.getContext());
        cached = attempt.allocation.emplace(plan.planId, certificate).first;
    }
    plan.allocationCertificate = cached->second;
    return success(static_cast<bool>(plan.allocationCertificate));
}
void FrontierAnalysis::invalidate()
{
    // Dropping modeled input makes all old handles ineligible for preparation.
    // initialize rebuilds every dependent cache before accepting another request.
    explicitAnalysis.reset();
    boundedAnalysis.reset();
    sequenceAnalysis.reset();
    arithmeticAnalysis.reset();
    generalArithmeticAnalysis.reset();
    arithmeticGeneratorStage.reset();
    arithmeticPeriodicAnalysis.reset();
    finiteVisitAnalyses.clear();
    periodicExports = {};
    nativeScalarOnly = false;
    sessionState.reset();
    program.reset();
    structuralIndex.reset();
    storage.reset();
    initialized = false;
}
} // namespace mlir::pto::frontiersynch
