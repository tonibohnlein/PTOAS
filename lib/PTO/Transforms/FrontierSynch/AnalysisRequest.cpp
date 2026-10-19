// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Request protocol adapters. Cache mathematical construction independently of
// export checks; detached command fragments always have fresh ownership.
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
#include "AnalysisSessionInternal.h"
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "llvm/ADT/ScopeExit.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "Shared/CompletionInlineChild.h"
#include "PTO/Transforms/FrontierSynch/Shared/AllocationProfile.h"
#include "Shared/SequenceAnalysisInternal.h"
#include "WholeInvocationIdentity.h"
#include "Structured/OrderCoveredPeriodic.h"
#include "Shared/NormalizedControl.h"
#include "Shared/RegionalAllocationCandidates.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Structured/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/GeneralArithmeticAllocation.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticPeriodicConversion.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicSharedCertificate.h"
#include "PTO/Transforms/FrontierSynch/Structured/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/Finite/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
AnalysisExports mathematicalExports(const MathematicalResult& demands)
{
    AnalysisExports available;
    available.queries = demands.finiteDemands || demands.rotatingDemands ||
        demands.guardedRotatingDemands || demands.numericNode || demands.sequenceDemands ||
        demands.expandedFiniteDemands;
    available.selectors = demands.finiteDemands || demands.sequenceDemands || demands.expandedFiniteDemands;
    if (demands.sequenceDemands && demands.sequenceDemands->state &&
        demands.sequenceDemands->state->partialStorage()) { available.selectors = false; }
    if (demands.regionalDemands) {
        available.queries = demands.regionalDemands->capabilities.exactQueries;
        available.selectors = demands.regionalDemands->capabilities.exactSelectors;
    }
    if (demands.compactDemands) {
        const auto& exports = demands.compactDemands->boundary->nativeExports();
        available.queries = exports.capabilities.exactQueries;
        available.selectors = exports.capabilities.exactSelectors;
    }
    if (demands.backend == "native-scalar" || demands.backend == "expanded-finite-guarded") { available = {}; }
    return available;
}
namespace {
std::shared_ptr<const RegionalAnalysis> retainedCandidateExports(
    const AnalysisSessionState& state, const ProgramRecognition& program,
    const MathematicalResult& math, const BackendAttempt& candidate)
{
    // Exports are owned by canonical producer attempts, not
    // necessarily the request adapter's preparation attempt.
    const BackendAttempt* exports = &candidate;
    if (math.arithmeticRegionalDemands) {
        auto found = state.arithmeticRegionAttempts.find(math.region);
        if (found != state.arithmeticRegionAttempts.end()) { exports = &found->second; }
    } else if (math.boundedDemands || math.varyingNode) {
        auto region = math.varyingNode;
        if (!region) {
            auto node = llvm::find_if(program.nodes, [&](const auto& node) {
                return node.anchor == math.boundedDemands->loop.getOperation();
            });
            if (node != program.nodes.end()) { region = std::distance(program.nodes.begin(), node); }
        }
        if (region) {
            auto found = state.loopAttempts.find({*region, math.varyingNode ?
                AnalysisBackend::VaryingBoundary : AnalysisBackend::BoundedLifetime});
            if (found != state.loopAttempts.end()) { exports = &found->second; }
        }
    }
    auto regional = exports->arithmeticSelectors ? exports->arithmeticSelectors :
        exports->boundedSelectors ? exports->boundedSelectors : exports->varyingSelectors;
    if (!regional) {
        regional = exports->arithmeticQueries ? exports->arithmeticQueries :
            exports->boundedQueries ? exports->boundedQueries : exports->varyingQueries;
    }
    return regional;
}
void preferRetainedControl(AnalysisSessionState& state, const ProgramRecognition& program,
    const AnalysisRequest& request, AnalysisOutcome& answer)
{
    // Compare only completed retained preparations. Selection never causes
    // another producer/export or builds speculative scalar operations.
    if (request.needs.synchronization && answer.status == AnalysisStatus::Ready &&
        answer.mathematical && !answer.completion) {
        auto& attempts = state.attempts[request.region];
        auto incumbent = llvm::find_if(attempts, [&](const auto& entry) {
            return entry.second.mathematical == answer.mathematical;
        });
        if (incumbent != attempts.end() && incumbent->second.pendingLogical) {
            auto* selected = incumbent->second.pendingLogical.get();
            std::optional<uint64_t> selectedCost;
            for (auto& [backend, candidate] : attempts) {
                (void)backend;
                if (!candidate.pendingLogical || candidate.pendingLogical.get() == selected ||
                    !candidate.endpoints.value_or(false) || !candidate.mathematical ||
                    candidate.mathematical->input != answer.mathematical->input) { continue; }
                auto available = mathematicalExports(*candidate.mathematical);
                auto regional = retainedCandidateExports(state, program, *candidate.mathematical, candidate);
                if (regional) {
                    available.queries |= regional->capabilities.exactQueries;
                    available.selectors |= regional->capabilities.exactSelectors;
                }
                if ((request.needs.queries && !available.queries) ||
                    (request.needs.selectors && !available.selectors)) { continue; }
                // The replacement must realize exactly the common required
                // order. Against a covering incumbent it cannot add payload
                // order; mathematical exactness alone is insufficient.
                const auto before = describeResultGuarantees(*selected);
                const auto after = describeResultGuarantees(*candidate.pendingLogical);
                const bool incumbentCovers = before.realization == RealizationOrder::Equivalent ||
                    before.realization == RealizationOrder::CoversSelectedOrder ||
                    before.realization == RealizationOrder::MayStrengthen;
                if (!incumbentCovers || after.realization != RealizationOrder::Equivalent ||
                    before.modeledOrder != InputOrderGuarantee::InputOrderEquivalent ||
                    after.modeledOrder != InputOrderGuarantee::InputOrderEquivalent ||
                    selected->completeInvocation != candidate.pendingLogical->completeInvocation) { continue; }
                AnalysisProfileScope comparison(state.ledger, "retained-control-comparison",
                    request.region, candidate.mathematical->backend);
                if (!selectedCost) { selectedCost = logicalFamilyConstructionOperations(*selected); }
                const auto cost = logicalFamilyConstructionOperations(*candidate.pendingLogical);
                comparison.recordWork({{"known", selectedCost && cost ? 1u : 0u},
                    {"incumbent", selectedCost.value_or(0)}, {"candidate", cost.value_or(0)}});
                if (!selectedCost || !cost || *cost >= *selectedCost) { continue; }
                selected = candidate.pendingLogical.get();
                selectedCost = cost;
                answer.mathematical = candidate.mathematical;
                answer.regionalExports = regional;
                answer.available = available;
                answer.available.synchronization = true;
            }
        }
    }
}
} // namespace
AnalysisOutcome FrontierAnalysis::minimumDemands(std::size_t region, AnalysisNeeds exports)
{
    exports.synchronization = false;
    return analyze({region, AnalysisMode::MinimumExact, exports});
}
AnalysisOutcome FrontierAnalysis::requestWholeFunctionBackend(AnalysisBackend backend)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", 0, analysisBackendName(backend), true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    if (!storage || failed(recognizeStructure()) || program->nodes.empty()) {
        result.obligations.push_back({AnalysisStage::Form, "whole-function original region is unavailable"});
        return result;
    }
    result = requestBackend(backend, {});
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
LogicalResult FrontierAnalysis::analyzeExplicitFunction()
{
    const auto result = requestWholeFunctionBackend(AnalysisBackend::Explicit);
    return success(result.status == AnalysisStatus::Ready && result.mathematical &&
        result.mathematical->backend == "explicit");
}
LogicalResult FrontierAnalysis::analyzeArithmeticFunction()
{
    return success(requestWholeFunctionBackend(AnalysisBackend::Arithmetic).status == AnalysisStatus::Ready);
}
LogicalResult FrontierAnalysis::analyzeArithmeticPeriodicFunction()
{
    return success(requestWholeFunctionBackend(AnalysisBackend::ArithmeticPeriodic).status == AnalysisStatus::Ready);
}
SequenceAnalysis* FrontierAnalysis::auditSequenceFunction()
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "audit-sequence", 0, "concrete-composition", true);
    return produceSequenceFunction();
}
AnalysisOutcome FrontierAnalysis::analyzeNumericalRegion(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyzeNumericalRegion", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    const bool valid = succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) {
        result.obligations.push_back({AnalysisStage::Form, "numerical original region is unavailable"});
        return result;
    }
    result = requestBackend(AnalysisBackend::NumericalPeriodic, request);
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
AnalysisOutcome FrontierAnalysis::analyzeArithmeticRegional(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyzeArithmeticRegional", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    const bool valid = succeeded(recognizeStructure()) && request.region && request.region < program->nodes.size();
    if (!valid) {
        result.obligations.push_back({AnalysisStage::Form, "arithmetic original child region is unavailable"});
        return result;
    }
    result = requestBackend(AnalysisBackend::Arithmetic, request);
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
AnalysisOutcome FrontierAnalysis::analyzeFiniteVisit(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyzeFiniteVisit", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    const bool valid = succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) {
        result.obligations.push_back({AnalysisStage::Form, "finite-visit original region is unavailable"});
        return result;
    }
    result = requestBackend(AnalysisBackend::FiniteVisit, request);
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
AnalysisOutcome FrontierAnalysis::analyzeVaryingBoundary(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyzeVaryingBoundary", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    const bool valid = succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) {
        result.obligations.push_back({AnalysisStage::Form, "repeating-boundary original region is unavailable"});
        return result;
    }
    result = requestBackend(AnalysisBackend::VaryingBoundary, request);
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
AnalysisOutcome FrontierAnalysis::analyzeFiniteExpansion(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyzeFiniteExpansion", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    const bool valid = succeeded(recognizeStructure()) && request.region < program->nodes.size();
    if (!valid) {
        result.obligations.push_back({AnalysisStage::Form, "finite expansion original region is unavailable"});
        return result;
    }
    result = requestBackend(AnalysisBackend::ExpandedFinite, request);
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    return result;
}
std::vector<RegionCertification> FrontierAnalysis::certifyRegions()
{
    const bool formsAvailable = succeeded(recognizeStructure());
    if (!formsAvailable) { return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    if (sessionState->certifications) { return *sessionState->certifications; }
    sessionState->certifying = true;
    std::vector<RegionCertification> results(program->nodes.size());
    // Children first reuse the same session when a parent requests exports.
    // Requests themselves enforce strict original-tree descent.
    for (std::size_t remaining = results.size(); remaining; --remaining) {
        const auto id = remaining - 1;
        auto& result = results[id];
        result.region = id;
        result.analysis = minimumDemands(id);
        if (!result.analysis.mathematical) { continue; }
        result.status = CertificationStatus::Recognized;
        const MathematicalResult* selected = result.analysis.mathematical.get();
        while (selected->identityComposition) { selected = selected->identityComposition->child.get(); }
        const auto& math = *selected;
        if (math.backend == "native-scalar") {
            result.status = CertificationStatus::Unresolved;
            result.analysis.obligations.push_back({AnalysisStage::Form,
                "native-only exact demands do not certify finite-occurrence class membership"});
        } else if (math.numericNode || math.backend == "numerical-sequence") {
            result.selectedClass = "periodic-storage";
            if (math.normalizedInput && math.normalizedInput->original) {
                result.representation = "small-count-expanded";
            }
        } else if (math.orderCoveredDemands) {
            result.selectedClass = "periodic-required-order";
            result.representation = "order-covered-periodic";
        } else if (math.rotatingDemands || math.mixedStrideDemands || math.arithmeticPeriodicDemands) {
            result.selectedClass = "periodic-storage";
        } else if (math.guardedRotatingDemands) {
            result.selectedClass = "invariant-guarded-periodic-storage";
        } else if (math.boundedDemands) {
            result.selectedClass = "bounded-lifetime";
        } else if (math.arithmeticDemands || math.generalArithmeticDemands || math.arithmeticRegionalDemands) {
            result.selectedClass = "restricted-arithmetic";
        } else if (math.expandedFiniteDemands || (math.finiteDemands && math.finiteDemands->guarded())) {
            result.selectedClass = "finite-guarded-occurrences";
            if (math.expandedFiniteDemands && math.expandedFiniteDemands->expandedProgram) {
                result.representation = "small-count-expanded";
            }
        } else if (math.finiteVisitDemands) {
            result.selectedClass = "finite-visit-types";
        } else if (math.varyingBoundaryDemands) {
            result.selectedClass = "repeating-boundary-interface";
        } else if (math.concreteDemands()) {
            result.selectedClass = "finite-occurrences";
        } else if (math.sequenceDemands) {
            const auto& node = program->nodes[id];
            result.selectedClass = node.kind == StructureKind::ExplicitRun ? "finite-occurrences" :
                (node.kind == StructureKind::Loop ? "regional-repetition" :
                (node.kind == StructureKind::Conditional ? "regional-conditional" : "regional-sequence"));
        }
        // A retained result alone is insufficient to certify a form we cannot
        // name. In particular, never relabel a conservative bounding result.
        if (result.selectedClass.empty()) { result.status = CertificationStatus::Unresolved; }
    }
    sessionState->certifications = results;
    sessionState->certifying = false;
    refreshProgramContractAudit(*program);
    return results;
}
AnalysisOutcome FrontierAnalysis::requestBackend(AnalysisBackend backend, const AnalysisRequest& request,
    bool sameNodeContinuation)
{
    sessionState->requestsStarted = true;
    AnalysisProfileScope profile(sessionState->ledger, "backend-request", request.region,
        analysisBackendName(backend));
    auto& attempt = sessionState->attempts[request.region][backend];
    RetainedAttemptPublication publication(*sessionState, attempt);
    if (attempt.produced) { profile.cacheHit(); }
    if (backend == AnalysisBackend::NumericalSequence) {
        const auto key = analysisRequestKey(request);
        const auto identity = std::make_pair(request.region, key);
        if (sessionState->numericalSequenceEstimates.insert(identity).second) {
            AnalysisCostEstimate estimate;
            auto found = sessionState->expandedFormEstimates.find(identity);
            if (found != sessionState->expandedFormEstimates.end()) { estimate = found->second; }
            sessionState->ledger->recordEstimate({request.region, key, "numerical-sequence", estimate});
        }
    }
    if (!attempt.produced) {
        AnalysisProfileScope producer(sessionState->ledger, "producer-attempt", request.region,
            analysisBackendName(backend));
        producer.dispatched();
        attempt.produced = true;
        if (construction.mathematicalAttempts != UINT64_MAX) { ++construction.mathematicalAttempts; }
        bool descends = true;
        if (!sessionState->activeRegions.empty()) {
            const auto parent = sessionState->activeRegions.back();
            auto ancestor = program->nodes[request.region].parent;
            while (ancestor && *ancestor != parent) { ancestor = program->nodes[*ancestor].parent; }
            // Only the consumed unary Sequence attempt may continue on the
            // same original region; ordinary recursive requests still descend.
            const bool continuation = sameNodeContinuation && parent == request.region &&
                backend != AnalysisBackend::Sequence && backend != AnalysisBackend::SequenceEndpoints;
            descends = ancestor.has_value() || continuation;
        }
        if (descends) {
            sessionState->activeRegions.push_back(request.region);
            attempt.mathematical = request.region == 0 ? produceBackend(backend, request, attempt.demandError) :
                produceRegionBackend(backend, request, attempt.demandError);
            sessionState->activeRegions.pop_back();
        } else { attempt.demandError = "recursive request must descend to an original child"; }
    }
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    result.mathematical = attempt.mathematical;
    if (result.mathematical && request.region == 0 && backend != AnalysisBackend::CompactBounding) {
        recordWholeRegion(backend);
    }
    if (!result.mathematical) {
        result.stage = attempt.failure.stage;
        result.status = attempt.failure.status;
        result.obligations.push_back({result.stage, attempt.demandError.empty() ?
            "exact producer or adapter did not establish its required contract" : attempt.demandError});
        for (const auto& diagnostic : attempt.failure.diagnostics) {
            result.obligations.push_back({result.stage, recognitionName(diagnostic.issue).str()});
        }
        for (const auto& diagnostic : attempt.failure.arithmeticDiagnostics) {
            result.obligations.push_back({result.stage, recognitionName(diagnostic.issue).str()});
        }
        return result;
    }
    const auto& demands = *result.mathematical;
    if (demands.identityComposition) {
        auto forwarded = requestIdentity(request, attempt);
        forwarded.ledger = sessionState->ledger;
        forwarded.requestId = profile.requestId();
        return forwarded;
    }
    if (demands.orderCoveredDemands) {
        for (const auto& diagnostic : attempt.failure.diagnostics) {
            result.obligations.push_back({AnalysisStage::Form,
                "ordinary rotating form: " + recognitionName(diagnostic.issue).str()});
        }
    }
    result.available = mathematicalExports(demands);
    if (demands.boundedDemands && (request.needs.queries || request.needs.selectors)) {
        std::string error;
        result.regionalExports = boundedExports(demands, false, error);
        result.available.queries = static_cast<bool>(result.regionalExports);
        if (request.needs.selectors) {
            auto selected = boundedExports(demands, true, error);
            result.available.selectors = static_cast<bool>(selected);
            if (selected) { result.regionalExports = std::move(selected); }
        }
        if (!error.empty()) { result.obligations.push_back({result.available.queries ?
            AnalysisStage::Selectors : AnalysisStage::Queries, error}); }
    }
    if (demands.arithmeticRegionalDemands && (request.needs.queries || request.needs.selectors)) {
        std::string error;
        result.regionalExports = arithmeticExports(demands, false, error);
        result.available.queries = static_cast<bool>(result.regionalExports);
        if (request.needs.selectors) {
            auto selected = arithmeticExports(demands, true, error);
            result.available.selectors = static_cast<bool>(selected);
            if (selected) { result.regionalExports = std::move(selected); }
        }
        if (!error.empty()) { result.obligations.push_back({result.available.queries ?
            AnalysisStage::Selectors : AnalysisStage::Queries, error}); }
    }
    const bool expandedExports = request.needs.queries || request.needs.selectors || attempt.expandedQueries;
    if (demands.backend == "expanded-finite-guarded" && expandedExports) {
        if (!attempt.expandedQueries) {
            AnalysisProfileScope exportProfile(sessionState->ledger, "queries", request.region, "expanded-finite");
            exportProfile.constructed();
            attempt.expandedQueries = std::make_shared<const RegionalAnalysis>(
                expandedFiniteRegionalQueries(*demands.expandedFiniteDemands, storage));
        }
        result.regionalExports = attempt.expandedQueries;
        result.available.queries = result.regionalExports->capabilities.exactQueries;
        if (request.needs.selectors && !attempt.expandedSelectorsAttempted) {
            AnalysisProfileScope exportProfile(sessionState->ledger, "selectors", request.region, "expanded-finite");
            exportProfile.constructed();
            attempt.expandedSelectorsAttempted = true;
            if (construction.expandedSelectorBuilds != UINT64_MAX) { ++construction.expandedSelectorBuilds; }
            auto selected = expandedFiniteRegionalSelectors(*demands.expandedFiniteDemands,
                *attempt.expandedQueries, attempt.expandedSelectorError, storage,
                &construction.expandedSelectorChecks);
            if (succeeded(selected)) {
                attempt.expandedSelectors = std::make_shared<const RegionalAnalysis>(std::move(*selected));
            }
        }
        if (request.needs.selectors && attempt.expandedSelectors) {
            result.regionalExports = attempt.expandedSelectors;
            result.available.selectors = result.regionalExports->capabilities.exactSelectors;
        } else if (request.needs.selectors && !attempt.expandedSelectorError.empty()) {
            result.obligations.push_back({AnalysisStage::Selectors, attempt.expandedSelectorError});
        }
    }
    if (demands.varyingBoundaryDemands &&
        (request.needs.queries || request.needs.selectors || request.needs.synchronization)) {
        std::string error;
        result.regionalExports = varyingExports(demands, false, error);
        result.available.queries = static_cast<bool>(result.regionalExports);
        if (request.needs.selectors) {
            auto selected = varyingExports(demands, true, error);
            result.available.selectors = static_cast<bool>(selected);
            if (selected) { result.regionalExports = std::move(selected); }
        }
        if (!error.empty()) { result.obligations.push_back({request.needs.selectors ?
            AnalysisStage::Selectors : AnalysisStage::Queries, error}); }
    }
    result.available.synchronization = attempt.endpoints.value_or(false);
    result.status = AnalysisStatus::Ready;
    result.stage = AnalysisStage::None;
    if (request.needs.queries && !result.available.queries) { result.stage = AnalysisStage::Queries; }
    if (request.needs.selectors && !result.available.selectors) { result.stage = AnalysisStage::Selectors; }
    if (request.needs.evaluation == AnalysisEvaluation::Stateful) { result.stage = AnalysisStage::Demands; }
    if (result.stage == AnalysisStage::None && request.needs.synchronization) {
        const auto sequence = demands.sequenceDemands;
        const bool upgrade = backend == AnalysisBackend::Sequence && sequence &&
            sequenceNeedsEndpointUpgrade(*sequence);
        if (upgrade) {
            // This is the one retained compact-to-phase capability alternative,
            // not another mathematical selection on a weak request. A refused
            // upgrade leaves the incumbent's own preparation available.
            auto stronger = requestBackend(AnalysisBackend::SequenceEndpoints, request);
            if (stronger.status == AnalysisStatus::Ready) { return stronger; }
        }
        if (!attempt.endpoints) {
            auto prepared = prepareLogical(result);
            attempt.endpoints = succeeded(prepared);
            if (succeeded(prepared)) { attempt.pendingLogical = std::move(*prepared); }
        }
        result.available.synchronization = *attempt.endpoints;
        if (!*attempt.endpoints) {
            result.stage = AnalysisStage::Synchronization;
            if (!attempt.prefixEmissionError.empty()) {
                result.obligations.push_back({result.stage, attempt.prefixEmissionError});
            }
            // Report the retained endpoint obligation, including terminal
            // cleanup/history failures, without rerunning preparation.
            if (demands.sequenceDemands && !demands.sequenceDemands->insertionError.empty()) {
                result.obligations.push_back({result.stage, demands.sequenceDemands->insertionError});
            }
            if (demands.finiteDemands && !demands.finiteDemands->insertionError().empty()) {
                result.obligations.push_back({result.stage, demands.finiteDemands->insertionError()});
            }
            if (demands.expandedFiniteDemands && !demands.expandedFiniteDemands->insertionError.empty()) {
                result.obligations.push_back({result.stage, demands.expandedFiniteDemands->insertionError});
            }
        }
    }
    if (result.stage != AnalysisStage::None) {
        result.status = AnalysisStatus::UnmetObligation;
        result.obligations.push_back({result.stage, demands.backend + ": requested export is unavailable"});
    }
    return result;
}
std::vector<AnalysisBackend> FrontierAnalysis::exactAttemptFamilies(
    const AnalysisRequest& request, AnalysisForm form, bool alternative)
{
    if (form == AnalysisForm::SmallCountExpanded) {
        return {AnalysisBackend::NumericalPeriodic, AnalysisBackend::NumericalSequence,
            AnalysisBackend::ExpandedFinite};
    }
    std::vector<AnalysisBackend> result;
    if (!alternative) { result.push_back(AnalysisBackend::NumericalPeriodic); }
    for (auto backend : {AnalysisBackend::Rotating, AnalysisBackend::MixedStride, AnalysisBackend::GuardedRotating,
            AnalysisBackend::BoundedLifetime, AnalysisBackend::Sequence, AnalysisBackend::VaryingBoundary,
            AnalysisBackend::FiniteVisit, AnalysisBackend::ExpandedFinite, AnalysisBackend::ArithmeticPeriodic,
            AnalysisBackend::FiniteGuarded}) {
        if (backend == AnalysisBackend::ExpandedFinite && alternative) { continue; }
        result.push_back(backend);
        if (backend == AnalysisBackend::Sequence && request.needs.synchronization) {
            result.push_back(AnalysisBackend::SequenceEndpoints);
        }
    }
    return result;
}
std::shared_ptr<const CompletionRegion> FrontierAnalysis::completionRegion(std::size_t region, std::string& error)
{
    auto& cached = sessionState->completions[region];
    if (cached.current(sessionState->retainedEvidenceGeneration)) {
        error = cached.error; return cached.owner;
    }
    if (cached.building) { error = "recursive completion capability request"; return {}; }
    cached.building = true;
    cached.owner.reset();
    cached.error.clear();
    auto finish = llvm::make_scope_exit([&]() {
        cached.building = false;
        if (cached.owner) {
            sessionState->issuedCompletions.insert(cached.owner);
            // First availability can enable a parent. Refreshing an available
            // snapshot after producer evidence changed is not new evidence:
            // otherwise alternating parent/child requests invalidate forever.
            if (!cached.published) {
                cached.published = true;
                sessionState->retainedEvidencePublished();
            }
        }
        cached.evidenceGeneration = sessionState->retainedEvidenceGeneration;
        cached.attempted = true;
    });
    SmallVector<Operation*> roots;
    const auto& node = program->nodes[region];
    if (node.kind == StructureKind::ExplicitRun) { roots = node.operations; }
    else if (node.kind == StructureKind::Sequence && node.region && node.region->hasOneBlock()) {
        for (auto& op : node.region->front().without_terminator()) { roots.push_back(&op); }
    } else if (node.kind == StructureKind::Loop || node.kind == StructureKind::Conditional ||
               (node.anchor && isa<scf::WhileOp>(node.anchor))) {
        roots.push_back(node.anchor);
    } else if (region) {
        cached.error = error = "completion request has no supported contiguous original scope"; return {};
    }
    SmallVector<std::shared_ptr<const CompletionInlineChild>> children;
    SmallVector<std::size_t> pending(node.children.rbegin(), node.children.rend());
    while (!pending.empty()) {
        const auto id = pending.pop_back_val();
        const auto& childNode = program->nodes[id];
        std::shared_ptr<const CompletionInlineChild> child;
        // Providers own qualification and the immutable fragment. The parent
        // requests this capability without inspecting a selected representation.
        if (auto found = sessionState->attempts.find(id); found != sessionState->attempts.end()) {
            for (const auto& attempt : found->second) {
                const auto& math = attempt.second.mathematical;
                if (!math || !math->completionChild) { continue; }
                child = math->completionChild();
                if (child) { break; }
            }
        }
        if (!child) {
            auto found = sessionState->completions.find(id);
            if (found != sessionState->completions.end()) {
                std::string childError;
                child = CompletionInlineChild::qualify(completionRegion(id, childError));
            }
        }
        if (!child && childNode.kind == StructureKind::ExplicitRun) {
            AnalysisRequest request; request.region = id;
            auto answer = requestBackend(AnalysisBackend::Sequence, request);
            if (answer.mathematical && answer.mathematical->completionChild) {
                child = answer.mathematical->completionChild();
            }
        }
        if (child) { children.push_back(std::move(child)); }
        else { pending.append(childNode.children.rbegin(), childNode.children.rend()); }
    }
    if (!sessionState->completionSource) {
        sessionState->completionSource =
            CompletionRegion::captureSource(function, storage, *structuralIndex, cached.error);
        if (!sessionState->completionSource) { error = cached.error; return {}; }
    }
    cached.owner = region ? CompletionRegion::buildRegion(function, storage, *structuralIndex, roots, children,
        cached.error, sessionState->completionSource) : CompletionRegion::buildWithChildren(
        function, storage, *structuralIndex, children, cached.error, sessionState->completionSource);
    error = cached.error;
    return cached.owner;
}
AnalysisOutcome FrontierAnalysis::analyze(const AnalysisRequest& request)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "analyze", true);
    AnalysisOutcome result;
    result.ledger = sessionState->ledger;
    result.requestId = profile.requestId();
    result.status = AnalysisStatus::UnmetObligation;
    if (!storage || !structuralIndex) {
        result.obligations.push_back({AnalysisStage::Form, "original region request is unavailable"});
        return result;
    }
    if (!program && failed(recognizeStructure())) {
        result.obligations.push_back({AnalysisStage::Form, "original region request is unavailable"});
        return result;
    }
    if (request.region >= program->nodes.size()) {
        result.obligations.push_back({AnalysisStage::Form, "original region identity is invalid"});
        return result;
    }
    const auto identity = std::make_tuple(request.region, analysisRequestKey(request), sessionState->certifying);
    if (auto found = sessionState->selectedAnswers.find(identity); found != sessionState->selectedAnswers.end()) {
        profile.cacheHit();
        auto retained = found->second;
        retained.ledger = sessionState->ledger;
        retained.requestId = profile.requestId();
        return retained;
    }
    const auto exhaustedKey = sessionState->exhaustedKey(request);
    bool descent = sessionState->activeRegions.empty();
    if (!descent) {
        auto ancestor = program->nodes[request.region].parent;
        while (ancestor && *ancestor != sessionState->activeRegions.back()) {
            ancestor = program->nodes[*ancestor].parent;
        }
        descent = ancestor.has_value();
    }
    if (descent) {
        if (const auto* cached = sessionState->exhausted(exhaustedKey)) {
            profile.cacheHit();
            if (construction.exhaustedWalkHits != UINT64_MAX) { ++construction.exhaustedWalkHits; }
            auto answer = cached->outcome;
            answer.ledger = sessionState->ledger; answer.requestId = profile.requestId();
            return answer;
        }
    }
    bool exhaustedWalk = false;
    auto finish = [&](AnalysisOutcome answer) {
        preferRetainedControl(*sessionState, *program, request, answer);

        answer.ledger = sessionState->ledger;
        answer.requestId = profile.requestId();
        // Positive exact selections remain independent of negative walk recipes.
        // A fallback or missing export expires on any new retained evidence.
        if (answer.status == AnalysisStatus::Ready && answer.mathematical && !answer.completion &&
            !answer.mathematical->compactDemands) {
            sessionState->selectedAnswers.emplace(identity, answer);
        } else if (exhaustedWalk && descent) {
            sessionState->retainExhausted(exhaustedKey, answer);
        }
        return answer;
    };
    if (request.needs.evaluation == AnalysisEvaluation::Uniform) {
        // Complete immutable answers can satisfy a different capability subset.
        // Preserve mode/audit context and exact-key precedence. This inspection
        // neither dispatches candidates nor prepares a mutable logical plan.
        const AnalysisOutcome* retained = nullptr;
        const auto context = analysisRequestKey(request) & ~uint8_t{7};
        auto begin = sessionState->selectedAnswers.lower_bound({request.region, uint8_t{0}, false});
        for (auto it = begin; it != sessionState->selectedAnswers.end() &&
             std::get<0>(it->first) == request.region; ++it) {
            if ((std::get<1>(it->first) & ~uint8_t{7}) != context ||
                std::get<2>(it->first) != sessionState->certifying) { continue; }
            const auto& answer = it->second;
            if (answer.status != AnalysisStatus::Ready || !answer.mathematical || answer.completion ||
                answer.mathematical->compactDemands ||
                (request.needs.queries && !answer.available.queries) ||
                (request.needs.selectors && !answer.available.selectors) ||
                (request.needs.synchronization && !answer.available.synchronization)) { continue; }
            if (!retained || (!retained->mathematical->concreteDemands() && answer.mathematical->concreteDemands())) {
                retained = &answer;
            }
        }
        // A failed stronger Explicit export may still answer this weaker
        // request. Leave it first in ordinary dispatch rather than bypassing it
        // with another family's cached selection.
        if (retained && !retained->mathematical->concreteDemands()) {
            auto region = sessionState->attempts.find(request.region);
            if (region != sessionState->attempts.end()) {
                auto explicitAttempt = region->second.find(AnalysisBackend::Explicit);
                if (explicitAttempt != region->second.end() && explicitAttempt->second.mathematical &&
                    explicitAttempt->second.mathematical->concreteDemands() &&
                    (!request.needs.synchronization || explicitAttempt->second.endpoints.value_or(true))) {
                    retained = nullptr;
                }
            }
        }
        if (retained) {
            profile.cacheHit();
            return finish(*retained);
        }
    }
    if (construction.requestWalks != UINT64_MAX) { ++construction.requestWalks; }
    if (request.needs.evaluation == AnalysisEvaluation::Stateful) {
        auto uniform = request;
        uniform.needs.evaluation = AnalysisEvaluation::Uniform;
        uniform.needs.synchronization = false;
        auto retained = analyze(uniform);
        retained.requestId = profile.requestId();
        retained.status = AnalysisStatus::UnmetObligation;
        retained.stage = AnalysisStage::Demands;
        retained.obligations.push_back({retained.stage, "stateful evaluation is unavailable"});
        exhaustedWalk = true;
        return finish(std::move(retained));
    }
    auto attemptBackend = [&](AnalysisBackend backend) {
        auto attempt = requestBackend(backend, request);
        if (sessionState->ledger->profiling) {
            for (auto& selected : sessionState->ledger->selections) {
                if (selected.request == profile.requestId() && selected.region == request.region &&
                    selected.method == analysisBackendName(backend)) {
                    selected.attempted = true;
                    selected.failureStage = attempt.status == AnalysisStatus::Ready ? 0 : unsigned(attempt.stage);
                }
            }
        }
        // Native protection proves an empty exact demand set, but is not a
        // finite-list certificate. Keep it while trying actual class forms.
        if (attempt.status == AnalysisStatus::Ready && sessionState->certifying &&
            attempt.mathematical && attempt.mathematical->backend == "native-scalar") {
            result = std::move(attempt);
            return false;
        }
        if (attempt.status == AnalysisStatus::Ready) { result = std::move(attempt); return true; }
        if (!result.mathematical && attempt.mathematical) {
            result.mathematical = attempt.mathematical;
            result.available = attempt.available;
            result.regionalExports = attempt.regionalExports;
            result.status = AnalysisStatus::UnmetObligation;
            result.stage = attempt.stage;
        }
        if (!result.mathematical && attempt.stage == AnalysisStage::Demands) {
            result.stage = AnalysisStage::Demands;
        }
        llvm::append_range(result.obligations, attempt.obligations);
        return false;
    };
    // The straight-line fast path needs neither normalization nor arithmetic.
    if (attemptBackend(AnalysisBackend::Explicit)) {
        return finish(std::move(result));
    }
    const auto forms = exactForms(request);
    const bool alternative = forms.size() > 1;
    std::vector<AnalysisBackend> candidates;
    std::map<AnalysisBackend, AnalysisForm> candidateForms;
    for (const auto form : forms) {
        for (const auto family : exactAttemptFamilies(request, form, alternative)) {
            const auto methods = family == AnalysisBackend::ArithmeticPeriodic ? arithmeticMethods(request) :
                std::vector<AnalysisBackend>{family};
            for (const auto backend : methods) {
                if (candidateForms.emplace(backend, form).second) { candidates.push_back(backend); }
            }
        }
    }
    while (!candidates.empty()) {
        // A failed export can retain useful mathematics or qualifications.
        // Reprice only remaining candidates against that new immutable work.
        const auto order = rankExactCandidates(request, candidates);
        const auto backend = order.front();
        sessionState->formAttempts[{request.region, candidateForms.at(backend)}] = 1;
        if (attemptBackend(backend)) { return finish(std::move(result)); }
        candidates.erase(llvm::find(candidates, backend));
    }
    exhaustedWalk = true;
    // A missing export can never authorize weakening an exact whole-region order.
    if (!result.mathematical && request.mode == AnalysisMode::Fallback) {
        auto conservative = requestBackend(AnalysisBackend::CompactBounding, request);
        llvm::append_range(conservative.obligations, result.obligations);
        conservative.ledger = sessionState->ledger;
        conservative.requestId = profile.requestId();
        if (conservative.status == AnalysisStatus::Ready) { return finish(std::move(conservative)); }
        std::string completionError;
        auto owner = completionRegion(request.region, completionError);
        if (owner) {
            AnalysisOutcome completion;
            completion.completion = owner;
            completion.available.synchronization = true;
            completion.available.allocation = true; // Closed child palettes remain subject to the requested pool.
            completion.status = request.needs.queries || request.needs.selectors ?
                AnalysisStatus::UnmetObligation : AnalysisStatus::Ready;
            completion.stage = AnalysisStage::Synchronization;
            completion.obligations = std::move(conservative.obligations);
            if (completion.status != AnalysisStatus::Ready) {
                completion.obligations.push_back({AnalysisStage::Queries,
                    "completion transfer does not export exact queries or storage selectors"});
            }
            completion.ledger = sessionState->ledger;
            completion.requestId = profile.requestId();
            return finish(std::move(completion));
        }
        conservative.obligations.push_back({AnalysisStage::Synchronization, completionError});
        return finish(std::move(conservative));
    }
    return finish(std::move(result));
}
namespace {
void retainFamilyTemplate(BackendAttempt& attempt, const PreparedLogicalPlan& plan)
{
    if (!attempt.logicalFamilyTemplate) {
        attempt.logicalFamilyTemplate = LogicalFamilyFragment::capture(plan, LogicalFamilyFragment::Context::Session);
    }
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareLogical(const AnalysisOutcome& result)
{
    if (result.completion) {
        if (result.mathematical || !sessionState || !sessionState->ownsCompletion(result.completion) ||
            result.completion->inputOwner() != storage) { return failure(); }
        auto plan = std::make_unique<PreparedLogicalPlan>(0);
        plan->completeInvocation = result.completion->completesInvocation();
        plan->completionOwner = result.completion;
        return plan;
    }
    if (!result.mathematical || !sessionState || result.mathematical->input != storage) { return failure(); }
    auto& attempts = sessionState->attempts[result.mathematical->region];
    auto retained = llvm::find_if(attempts, [&](const auto& attempt) {
        return attempt.second.mathematical == result.mathematical;
    });
    if (retained == attempts.end()) { return failure(); }
    RetainedAttemptPublication publication(*sessionState, retained->second);
    AnalysisProfileScope profile(sessionState->ledger, "logical-preparation", result.mathematical->region,
        result.mathematical->backend, false, result.requestId);
    if (retained->second.pendingLogical) {
        profile.cacheHit();
        retainFamilyTemplate(retained->second, *retained->second.pendingLogical);
        return std::move(retained->second.pendingLogical);
    }
    if (retained->second.logicalFamilyTemplate) {
        profile.cacheHit();
        auto plan = retained->second.logicalFamilyTemplate->instantiate();
        plan->mathematicalOwner = result.mathematical;
        return plan;
    }
    profile.constructed();
    if (construction.logicalPreparations != UINT64_MAX) { ++construction.logicalPreparations; }
    auto prepared = prepareRetained(*result.mathematical);
    const bool rootFallback = failed(prepared) && result.mathematical->region == 0;
    if (rootFallback) {
        prepared = prepareRetainedPrefix(result.mathematical, retained->second.prefixEmissionError);
    }
    if (succeeded(prepared)) {
        (*prepared)->mathematicalOwner = result.mathematical;
        retainFamilyTemplate(retained->second, **prepared);
    }
    refreshProgramContractAudit(*program);
    return prepared;
}
namespace {
// Borrow existing regional mathematics; requesting allocation must not invoke
// another demand analysis or stronger query export merely to select a candidate.
std::shared_ptr<const RegionalAnalysis> allocationRegion(const MathematicalResult& demands)
{
    if (demands.regionalDemands) { return demands.regionalDemands; }
    if (demands.sequenceDemands) {
        return std::make_shared<const RegionalAnalysis>(sequenceRegionalResult(*demands.sequenceDemands));
    }
    if (demands.finiteDemands && demands.finiteDemands->guarded()) {
        return std::make_shared<const RegionalAnalysis>(finiteRegionalResult(demands.finiteDemands));
    }
    return {};
}
struct IdentityAllocationSource {
    const MathematicalResult* demands = nullptr;
    const BackendAttempt* attempt = nullptr;
    AnalysisBackend backend = AnalysisBackend::Explicit;
};
// Identity composition changes neither the command graph nor its namespaces.
// Both allocation requests use this same ownership check before forwarding.
IdentityAllocationSource identityAllocationSource(const MathematicalResult& demands,
    const SyncInput* input, const PhaseIndex* index, const ProgramRecognition* program,
    func::FuncOp function, const AnalysisSessionState& session)
{
    const auto* retained = &demands;
    auto backend = AnalysisBackend::Explicit;
    while (retained->identityComposition) {
        const auto& owner = *retained->identityComposition;
        if (retained->region != owner.region || !owner.applies(input, index, program, function)) { return {}; }
        backend = owner.childBackend;
        retained = owner.child.get();
    }
    auto region = session.attempts.find(retained->region);
    if (region == session.attempts.end()) { return {}; }
    auto found = region->second.find(backend);
    if (found == region->second.end() || found->second.mathematical.get() != retained) { return {}; }
    return {retained, &found->second, backend};
}
DictionaryAttr allocateRetained(const MathematicalResult& demands, const ProgramRecognition& program,
    const BackendAttempt& attempt, PreparedLogicalPlan& prepared, MLIRContext* context,
    uint64_t& guardedImportWork, AllocationProfile* allocationProfile)
{
    const auto plan = prepared.planId;
    if (demands.concreteDemands()) { return explicitAllocationCertificate(*demands.concreteDemands(), plan, context); }
    if (demands.numericNode) {
        return demands.numericalDemands ?
            encodePeriodicSharedAllocation(demands.numericalDemands->analysis, plan, context) : DictionaryAttr{};
    }
    if (demands.rotatingDemands) {
        if (demands.orderCoveredDemands) {
            const auto& owner = *demands.orderCoveredDemands;
            auto originalLoop = owner.retained.loop;
            if (!owner.applies(demands.input.get(), owner.index.get(), demands.rotatingDemands->loop) ||
                &owner.retained != demands.rotatingDemands.get() || owner.originalRegion >= program.nodes.size() ||
                program.nodes[owner.originalRegion].anchor != originalLoop.getOperation()) { return {}; }
        }
        return encodePeriodicSharedAllocation(demands.rotatingDemands->periodic, plan, context);
    }
    if (demands.mixedStrideDemands) {
        return mixedStrideAllocationCertificate(*demands.mixedStrideDemands, plan, context);
    }
    if (demands.compactDemands) { return compactBoundingAllocationCertificate(*demands.compactDemands, prepared); }
    if (demands.guardedRotatingDemands) {
        auto imported = importGuardedRotating(*demands.guardedRotatingDemands,
            std::make_shared<RegionExpressions>(), &guardedImportWork);
        return succeeded(imported) ? guardedAllocationCertificate(*imported, plan, context) : DictionaryAttr{};
    }
    if (demands.arithmeticPeriodicDemands) {
        const auto& conversion = demands.arithmeticPeriodicDemands->conversion;
        if (conversion.numerical) { return encodePeriodicSharedAllocation(*conversion.numerical, plan, context); }
        if (!conversion.guarded) { return {}; }
        return guardedPeriodicAllocationCertificate(*conversion.expressions, conversion.payloads,
            conversion.generators, *conversion.guarded, plan, context);
    }
    if (demands.sequenceDemands || (demands.finiteDemands && demands.finiteDemands->guarded())) {
        auto region = allocationRegion(demands);
        auto certificate = regionalAllocationCertificate(*region, prepared, allocationProfile);
        return certificate ? certificate : finiteRegionalAllocationCertificate(*region, prepared);
    }
    if (!demands.arithmeticDemands && !demands.generalArithmeticDemands) { return {}; }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.arithmetic->sites) {
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    if (demands.generalArithmeticDemands) {
        return generalArithmeticAllocationCertificate(
            *requestArithmeticRequiredOrder(*demands.generalArithmeticDemands), pipes,
            attempt.arithmeticRecords, plan, context);
    }
    const auto complete = requestArithmeticRequiredOrder(*demands.arithmeticDemands);
    auto certificate = arithmeticAllocationCertificate(*complete, pipes,
                                                       attempt.arithmeticRecords, plan, context);
    if (certificate) { return certificate; }
    auto proof = buildArithmeticHandoffAllocation(*complete, pipes);
    return encodeGeneralArithmeticAllocationCertificate(proof, pipes, attempt.arithmeticRecords, plan, context);
}
} // namespace
DictionaryAttr FrontierAnalysis::allocateIdentity(const MathematicalResult& demands, PreparedLogicalPlan& plan,
    AllocationProfile* allocationProfile)
{
    auto source = identityAllocationSource(demands, storage.get(), structuralIndex.get(),
        program.get(), function, *sessionState);
    if (!source.demands) { return {}; }
    const auto* retained = source.demands;
    auto certificate = allocateRetained(*retained, *program, *source.attempt, plan, function.getContext(),
        sessionState->guardedImportWork, allocationProfile);
    if (certificate || !plan.regionalAllocation) { return certificate; }
    AnalysisRequest request;
    request.region = retained->region;
    request.needs.queries = true;
    auto exported = requestBackend(source.backend, request);
    if (exported.mathematical.get() != retained || !exported.available.queries) { return {}; }
    auto region = exported.regionalExports ? exported.regionalExports : retained->regionalDemands;
    return region ? regionalAllocationCertificate(*region, plan, allocationProfile) : DictionaryAttr{};
}
LogicalResult FrontierAnalysis::attachAllocation(const AnalysisOutcome& result, PreparedLogicalPlan& plan,
    ArrayRef<int64_t> eligibleIds)
{
    llvm::SmallDenseSet<int64_t, 8> uniqueIds;
    for (auto id : eligibleIds) {
        if (!isNativeEventId(id) || !uniqueIds.insert(id).second) { return failure(); }
    }
    if (result.completion) {
        return success(!result.mathematical && sessionState && sessionState->ownsCompletion(result.completion) &&
                       plan.completionOwner == result.completion && result.completion->inputOwner() == storage);
    }
    // Private invocation prefixes require their own lifetime export. Never reuse
    // a direct-endpoint certificate cached for the same mathematical graph.
    if (plan.privateState) { return failure(); }
    // Allocation is a separate request on this prepared graph. Its readiness is
    // returned here and stored in plan.allocationCertificate, not in the earlier
    // Analyze outcome's available exports or logical Ready status.
    if (!result.mathematical || !sessionState || result.mathematical->input != storage ||
        plan.mathematicalOwner != result.mathematical) { return failure(); }
    auto& attempts = sessionState->attempts[result.mathematical->region];
    auto selected = llvm::find_if(attempts, [&](const auto& entry) {
        return entry.second.mathematical == result.mathematical;
    });
    if (selected == attempts.end()) { return failure(); }
    const bool emptyInvocation = program && structuralIndex && staticallyEmptyWholeInvocation(
        *result.mathematical, function, *storage, *program, *structuralIndex);
    if (emptyInvocation && (plan.completeInvocation || plan.familyRecipe || plan.familyProtocol ||
        plan.completionOwner || plan.executedOutput || plan.physicalOutput || plan.allocationPreparation ||
        plan.regionalAllocation || plan.compactBoundingOwner || !plan.families.empty() ||
        !plan.endpoints.empty() || !plan.preparation.empty())) { return failure(); }
    if (failed(normalizeLogicalFamilies(plan))) {
        return failure();
    }
    AnalysisProfileScope profile(sessionState->ledger, "allocation-support", result.mathematical->region,
        result.mathematical->backend, false, result.requestId);
    if (!emptyInvocation && !eligibleIds.empty()) {
        auto region = result.regionalExports ? result.regionalExports : allocationRegion(*result.mathematical);
        if (!region && result.mathematical->identityComposition) {
            auto source = identityAllocationSource(*result.mathematical, storage.get(), structuralIndex.get(),
                program.get(), function, *sessionState);
            if (source.demands) { region = allocationRegion(*source.demands); }
        }
        if (region) {
            AnalysisProfileScope candidateWork(sessionState->ledger, "allocation-candidates",
                result.mathematical->region, result.mathematical->backend, false, result.requestId);
            candidateWork.constructed();
            AllocationProfile candidateProfile(sessionState->ledger);
            if (succeeded(selectRegionalAllocationCandidate(function, *region, plan, eligibleIds, &candidateProfile))) {
                profile.constructed();
                if (construction.allocationExports != UINT64_MAX) { ++construction.allocationExports; }
                // Capacity-specific evidence stays on this plan, outside the
                // unrestricted split-export cache used by fresh preparations.
                return success();
            }
        }
    }
    if (plan.allocationCertificate) { profile.cacheHit(); return success(); }
    AllocationProfile allocationWork(sessionState->ledger);
    auto* allocationProfile = &allocationWork;
    auto& attempt = selected->second;
    const auto protocol =
        plan.familyProtocol ? normalizedFamilyProtocolKey(*plan.familyProtocol) : ArrayRef<uint64_t>{};
    const auto cacheKey = std::make_pair(plan.planId, std::vector<uint64_t>(protocol.begin(), protocol.end()));
    auto cached = attempt.allocation.find(cacheKey);
    if (cached != attempt.allocation.end()) { profile.cacheHit(); }
    {
        AllocationProfile::Scope summary(allocationProfile, AllocationProfile::Summary);
        prepareAllocationSupport(plan, allocationProfile);
    }
    if (cached == attempt.allocation.end()) {
        profile.constructed();
        if (construction.allocationExports != UINT64_MAX) { ++construction.allocationExports; }
        auto certificate = plan.allocationCertificate;
        if (!certificate) {
            if (emptyInvocation) {
                certificate = explicitAllocationCertificate(ExplicitAnalysis{}, plan.planId, function.getContext());
            } else {
                certificate = result.mathematical->identityComposition ?
                    allocateIdentity(*result.mathematical, plan, allocationProfile) :
                    allocateRetained(*result.mathematical, *program, attempt, plan, function.getContext(),
                        sessionState->guardedImportWork, allocationProfile);
            }
        }
        cached =
            attempt.allocation.emplace(cacheKey, BackendAttempt::CachedAllocation{plan.familyProtocol, certificate})
                .first;
    }
    plan.allocationCertificate = cached->second.certificate;
    return success(static_cast<bool>(plan.allocationCertificate));
}
void FrontierAnalysis::invalidate()
{
    // Dropping modeled input makes all old handles ineligible for preparation.
    // initialize rebuilds every dependent cache before accepting another request.
    concreteFinite.reset();
    finiteInput.reset();
    sequenceAnalysis.reset();
    arithmeticAnalysis.reset();
    generalArithmeticAnalysis.reset();
    arithmeticForms.clear();
    arithmeticPeriodicAnalysis.reset();
    finiteVisitAnalyses.clear();
    nativeScalarOnly = false;
    sessionState.reset();
    program.reset();
    structuralIndex.reset();
    storage.reset();
    initialized = false;
}
} // namespace mlir::pto::frontiersynch
