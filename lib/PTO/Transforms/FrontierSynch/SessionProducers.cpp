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
#include "PTO/Transforms/FrontierSynch/NumericTemplateRegional.h"
#include "SequenceAnalysisInternal.h"
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
namespace {
void recordSequenceCosts(AnalysisSessionState& session, const SequenceAnalysis& sequence, std::size_t region)
{
    if (!sequence.state) { return; }
    for (auto record : sequence.state->crossingMethods) {
        if (record.method.empty()) { continue; }
        record.region = region;
        record.request = sequence.state->requireEndpoints ? 4 : 0;
        session.costs.push_back(std::move(record));
    }
}
void recordFormFailure(ProducerFailure& failure, const RecognitionResult& form)
{
    failure.stage = AnalysisStage::Form;
    failure.status = form.state == RecognitionState::NotApplicable ?
        AnalysisStatus::NotApplicable : AnalysisStatus::UnmetObligation;
    failure.diagnostics = form.diagnostics;
    // Expansion caps restrict a representation attempt, never the paper class.
    for (const auto& diagnostic : form.diagnostics) {
        if (diagnostic.issue == RecognitionIssue::TemplateExpansionLimit) {
            failure.status = AnalysisStatus::UnmetObligation;
        }
    }
}
void recordArithmeticFormFailure(ProducerFailure& failure, const ArithmeticProgram* form)
{
    if (!form) { failure.stage = AnalysisStage::Form; return; }
    if (form->extraction.state != RecognitionState::Applicable) {
        recordFormFailure(failure, form->extraction);
    } else if (form->recognition.state != RecognitionState::Applicable) {
        failure.stage = AnalysisStage::Form;
        failure.status = form->recognition.state == RecognitionState::NotApplicable ?
            AnalysisStatus::NotApplicable : AnalysisStatus::UnmetObligation;
        failure.arithmeticDiagnostics = form->recognition.diagnostics;
    }
}
ArithmeticRegionContext originalContext(func::FuncOp function, const ProgramRecognition& program, std::size_t region)
{
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
    return context;
}
std::shared_ptr<const NormalizedControlDescription> normalizedInput(AnalysisSessionState& session,
    func::FuncOp function, const ProgramRecognition& program, std::size_t region,
    const PhaseIndex& index, const SyncInput& input)
{
    auto found = session.normalizedInputs.find(region);
    if (found != session.normalizedInputs.end()) { return found->second; }
    auto description = normalizeSmallCountControl(originalContext(function, program, region), index, input);
    return session.normalizedInputs.emplace(region, std::move(description)).first->second;
}
} // namespace
uint64_t FrontierAnalysis::normalizationConstructions() const
{
    return sessionState ? sessionState->normalizedInputs.size() : 0;
}
uint64_t FrontierAnalysis::finiteExpansionPreflights() const
{
    return sessionState ? sessionState->finiteExpansionPlans.size() : 0;
}
uint64_t FrontierAnalysis::numericTemplatePreflights() const
{
    return sessionState ? sessionState->numericTemplatePlans.size() : 0;
}
uint64_t FrontierAnalysis::numericalRegionConstructions() const
{
    return sessionState ? sessionState->numericalRegionBuilds : 0;
}
const NumericTemplate& FrontierAnalysis::materializeNumericRegion(std::size_t region)
{
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    auto& node = program->nodes[region];
    if (node.numericTemplate) { return *node.numericTemplate; }
    auto found = sessionState->numericTemplatePlans.find({region, false});
    if (found == sessionState->numericTemplatePlans.end()) {
        auto plan = std::make_shared<const NumericTemplatePlan>(
            preflightNumericTemplate(cast<scf::ForOp>(node.anchor), *structuralIndex, *storage, {}, false, {}, {},
                normalizedInput(*sessionState, function, *program, region, *structuralIndex, *storage)));
        found = sessionState->numericTemplatePlans.emplace(std::make_pair(region, false), std::move(plan)).first;
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
const FiniteExpansionPlan& expansionPlan(AnalysisSessionState& session,
    const PhaseIndex& index, const SyncInput& input,
    std::shared_ptr<const NormalizedControlDescription> normalized)
{
    auto found = session.finiteExpansionPlans.find(normalized.get());
    if (found != session.finiteExpansionPlans.end()) { return *found->second; }
    const auto* identity = normalized.get();
    auto context = normalized->context;
    auto plan = std::make_shared<const FiniteExpansionPlan>(
        preflightFiniteExpansion(context, index, input, {}, std::move(normalized)));
    auto inserted = session.finiteExpansionPlans.emplace(identity, std::move(plan));
    return *inserted.first->second;
}
const StructureNode* wholeLoop(func::FuncOp function, const SyncInput& input,
                              const ProgramRecognition& program, const PhaseIndex& index,
                              ProducerFailure* failure = nullptr)
{
    if (failure) { failure->stage = AnalysisStage::Form; }
    auto excluded = [&]() { if (failure) { failure->status = AnalysisStatus::NotApplicable; } };
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        const bool rootLoop = node.kind == StructureKind::Loop && node.anchor->getParentOp() == function;
        if (!rootLoop) { continue; }
        if (node.unsupportedContext) { return nullptr; }
        if (selected) { excluded(); return nullptr; }
        selected = &node;
    }
    if (!selected || llvm::any_of(input.instructions(), [&](const auto* phase) {
            return !selected->anchor->isProperAncestor(phase->elementOp);
        })) { excluded(); return nullptr; }
    RecognitionResult outside;
    for (auto& operation : function.front()) {
        if (&operation == selected->anchor) { continue; }
        if (operation.getNumRegions()) { excluded(); return nullptr; }
        detail::inspectLeaf(operation, index, outside);
    }
    if (outside.state != RecognitionState::Applicable && failure) { recordFormFailure(*failure, outside); }
    return outside.state == RecognitionState::Applicable ? selected : nullptr;
}
} // namespace
std::shared_ptr<const NormalizedControlDescription> FrontierAnalysis::normalizedRegion(std::size_t region)
{
    // A complete root-loop view shares its original loop's representation.
    // Whole-invocation scope checks remain separate from that representation.
    if (!region) {
        if (const auto* loop = wholeLoop(function, *storage, *program, *structuralIndex)) {
            region = static_cast<std::size_t>(loop - program->nodes.data());
        }
    }
    return normalizedInput(*sessionState, function, *program, region, *structuralIndex, *storage);
}
const NumericTemplatePlan* FrontierAnalysis::numericalPreflight(std::size_t region)
{
    const auto* node = region ? &program->nodes[region] : wholeLoop(function, *storage, *program, *structuralIndex);
    const bool valid = node && node->kind == StructureKind::Loop && !node->unsupportedContext;
    if (!valid) { return nullptr; }
    const auto original = static_cast<std::size_t>(node - program->nodes.data());
    const auto key = std::make_pair(original, true);
    auto found = sessionState->numericTemplatePlans.find(key);
    if (found == sessionState->numericTemplatePlans.end()) {
        auto plan = std::make_shared<const NumericTemplatePlan>(preflightNumericTemplate(
            cast<scf::ForOp>(node->anchor), *structuralIndex, *storage, {}, true, {}, {}, normalizedRegion(original)));
        found = sessionState->numericTemplatePlans.emplace(key, std::move(plan)).first;
    }
    return found->second.get();
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceExpandedFinite(std::size_t region,
    std::string& error)
{
    if (region && program->nodes[region].unsupportedContext) {
        error = "finite expansion requires a supported original region"; return {};
    }
    auto normalized = normalizedRegion(region);
    auto& cached = sessionState->expandedAttempts[normalized.get()];
    if (!cached.produced) {
        cached.produced = true;
        const auto& plan = expansionPlan(*sessionState, *structuralIndex, *storage, normalized);
        auto demands = std::make_shared<FiniteGuardedAnalysis>(
            analyzeExpandedFinite(plan, *structuralIndex, *storage, sessionState->expressions, structuralIndex));
        cached.demandError = demands->error;
        if (plan.result.state != RecognitionState::Applicable) {
            recordFormFailure(cached.failure, plan.result);
        }
        if (cached.demandError.empty()) {
            auto owned = std::make_shared<MathematicalResult>();
            owned->input = storage; owned->recognition = program;
            owned->normalizedInput = normalized; owned->finiteGuardedDemands = std::move(demands);
            owned->backend = "expanded-finite-guarded";
            cached.mathematical = std::move(owned);
        }
    }
    error = cached.demandError;
    sessionState->attempts[region][AnalysisBackend::ExpandedFinite].failure = cached.failure;
    if (!cached.mathematical) { return {}; }
    auto owned = std::make_shared<MathematicalResult>(*cached.mathematical);
    owned->region = region;
    return owned;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceLoopBackend(
    AnalysisBackend backend, std::string& error, std::size_t region)
{
    auto& requestedFailure = sessionState->attempts[region][backend].failure;
    const auto* node = region == 0 ? wholeLoop(function, *storage, *program, *structuralIndex, &requestedFailure) :
                                    &program->nodes[region];
    if (!node || node->kind != StructureKind::Loop || node->unsupportedContext) {
        auto& producerFailure = sessionState->attempts[region][backend].failure;
        producerFailure.stage = AnalysisStage::Form;
        if (node && node->kind != StructureKind::Loop) { producerFailure.status = AnalysisStatus::NotApplicable; }
        error = "route requires an original certified loop"; return {};
    }
    const auto original = static_cast<std::size_t>(node - program->nodes.data());
    auto& attempt = sessionState->loopAttempts[{original, backend}];
    if (!attempt.produced) {
        attempt.produced = true;
        attempt.mathematical = constructLoopBackend(backend, original, attempt.demandError);
    }
    error = attempt.demandError;
    sessionState->attempts[region][backend].failure = attempt.failure;
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
    auto& producerFailure = sessionState->loopAttempts[{region, backend}].failure;
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    owned->region = region;
    switch (backend) {
    case AnalysisBackend::VaryingBoundary: {
        auto& local = program->nodes[region];
        if (!local.varyingRotating || local.varyingRotating->result.state != RecognitionState::Applicable) {
            if (local.varyingRotating) { recordFormFailure(producerFailure, local.varyingRotating->result); }
            else { producerFailure.stage = AnalysisStage::Form; }
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
            const auto& contract = result.recognition.contract;
            if (contract.membership != ContractStatus::Established) {
                producerFailure.stage = AnalysisStage::Form;
                producerFailure.status = contract.membership == ContractStatus::Violated ?
                    AnalysisStatus::NotApplicable : AnalysisStatus::UnmetObligation;
                RecognitionResult form;
                form.state = contract.membership == ContractStatus::Violated ?
                    RecognitionState::NotApplicable : RecognitionState::MissingPremise;
                form.diagnostics = contract.diagnostics;
                recordFormFailure(producerFailure, form);
            }
            error = contract.implementationError;
            return {};
        }
        owned->finiteVisitDemands = found->second;
        owned->backend = "finite-visit-types";
        break;
    }
    case AnalysisBackend::NumericalPeriodic: {
        const auto* plan = numericalPreflight(region);
        if (!plan) { producerFailure.stage = AnalysisStage::Form;
            error = "numerical original-loop preflight unavailable"; return {}; }
        auto result = std::make_shared<NumericalRegionDemands>();
        if (sessionState->numericalRegionBuilds != UINT64_MAX) { ++sessionState->numericalRegionBuilds; }
        result->form = materializeNumericTemplate(*plan, *structuralIndex, *storage);
        if (result->form.result.state != RecognitionState::Applicable) {
            recordFormFailure(producerFailure, result->form.result);
            error = "regional numerical template form is unavailable"; return {};
        }
        result->analysis = analyzeNumericTemplate(result->form);
        if (!result->analysis.error.empty()) { error = result->analysis.error; return {}; }
        owned->normalizedInput = plan->normalized;
        owned->numericalDemands = std::move(result);
        owned->numericNode = region;
        owned->backend = "numerical-periodic";
        break;
    }
    case AnalysisBackend::Rotating: {
        if (!node->rotatingResult || node->rotatingResult->state != RecognitionState::Applicable) {
            if (node->rotatingResult) { recordFormFailure(producerFailure, *node->rotatingResult); }
            else { producerFailure.stage = AnalysisStage::Form; }
            error = "rotating form is not established"; return {};
        }
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
            node->guardedRotatingResult->result.state != RecognitionState::Applicable) {
            if (node->guardedRotatingResult) {
                recordFormFailure(producerFailure, node->guardedRotatingResult->result);
            }
            else { producerFailure.stage = AnalysisStage::Form; }
            error = "guarded rotating form is not established"; return {};
        }
        GuardedRotatingSpecialization request; request.loop = loop;
        auto mathematics = specializedGuardedDemands(request, error);
        if (!mathematics || !error.empty()) { return {}; }
        owned->guardedRotatingDemands = std::make_shared<GuardedRotatingAnalysis>(mathematics->demands);
        owned->backend = "guarded-rotating";
        break;
    }
    case AnalysisBackend::BoundedLifetime: {
        if (!node->boundedLifetime ||
            node->boundedLifetime->skeleton.result.state != RecognitionState::Applicable) {
            if (node->boundedLifetime) { recordFormFailure(producerFailure, node->boundedLifetime->skeleton.result); }
            else { producerFailure.stage = AnalysisStage::Form; }
            error = "bounded lifetime form is not established"; return {};
        }
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
uint64_t FrontierAnalysis::specializedArithmeticConstructions() const
{
    return sessionState ? sessionState->specializedArithmeticBuilds : 0;
}
std::shared_ptr<const ArithmeticRegionalRelations> FrontierAnalysis::specializedArithmeticDemands(
    ArithmeticRegionContext context, const ArithmeticEntryConstant& constants, std::string& error,
    ArrayRef<ArithmeticLimits> profiles)
{
    const auto owned = [&](Operation* root) {
        return root && (root == function || function->isProperAncestor(root));
    };
    const bool valid = initialized && storage && structuralIndex && context.function == function &&
        owned(context.root) && llvm::all_of(context.roots, owned);
    if (!valid) {
        error = "specialized arithmetic request belongs to a different invocation"; return {};
    }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    if (profiles.empty()) { profiles = regionalArithmeticProfiles; }
    std::vector<std::tuple<unsigned, unsigned, uint64_t, uint64_t>> profileKey;
    for (const auto& value : profiles) {
        profileKey.emplace_back(value.pipes, value.dimensions, value.period, value.coefficient);
    }
    const auto binding = [&](Value value) { return constants ? constants(value) : std::optional<int64_t>{}; };
    auto& variants = sessionState->specializedArithmetic[context.root];
    for (const auto& attempt : variants) {
        if (attempt.roots != context.roots || attempt.profiles != profileKey) { continue; }
        const bool sameBindings = llvm::all_of(attempt.bindings, [&](const auto& entry) {
            return binding(entry.second.first) == entry.second.second;
        });
        if (sameBindings) { error = attempt.error; return attempt.mathematics; }
    }
    SpecializedArithmeticAttempt attempt;
    attempt.roots = context.roots; attempt.profiles = std::move(profileKey);
    auto observed = [&](Value value) {
        auto result = binding(value);
        attempt.bindings.emplace(value.getAsOpaquePointer(), std::make_pair(value, result));
        return result;
    };
    ++sessionState->specializedArithmeticBuilds;
    auto mathematics = analyzeSpecializedArithmeticDemands(
        context, *structuralIndex, *storage, profiles, observed, attempt.error);
    if (mathematics) {
        auto ownedMathematics = std::make_shared<ArithmeticRegionalRelations>(*mathematics);
        ownedMathematics->inputOwner = storage; ownedMathematics->indexOwner = structuralIndex;
        attempt.mathematics = std::move(ownedMathematics);
    }
    error = attempt.error;
    auto result = attempt.mathematics;
    variants.push_back(std::move(attempt));
    return result;
}
uint64_t FrontierAnalysis::specializedGuardedConstructions() const
{
    return sessionState ? sessionState->specializedGuardedBuilds : 0;
}
uint64_t FrontierAnalysis::guardedExpressionImportWork() const
{
    return sessionState ? sessionState->guardedImportWork : 0;
}
std::shared_ptr<const GuardedRotatingMathematics> FrontierAnalysis::specializedGuardedDemands(
    const GuardedRotatingSpecialization& request, std::string& error)
{
    error.clear();
    auto originalValue = [&](Value value) {
        auto* region = value ? value.getParentRegion() : nullptr;
        auto* owner = region ? region->getParentOp() : nullptr;
        return owner && (owner == function || function->isProperAncestor(owner));
    };
    const bool valid = initialized && storage && structuralIndex && program && request.loop &&
        function->isProperAncestor(request.loop) &&
        (request.sliced || (request.choices.empty() && request.guards.empty())) &&
        llvm::all_of(request.choices, [&](const auto& entry) {
            return originalValue(entry.first) && entry.first.getType().isInteger(1);
        }) && llvm::all_of(request.guards, [&](Value value) {
            return originalValue(value) && value.getType().isInteger(1);
        }) && (request.bindings.empty() || request.expressions);
    if (!valid) { error = "guarded specialization requires original invocation values and a binding arena"; return {}; }
    auto node = llvm::find_if(program->nodes, [&](const auto& value) {
        return value.kind == StructureKind::Loop && value.anchor == request.loop;
    });
    if (node == program->nodes.end()) { error = "guarded specialization requires an original loop"; return {}; }
    SmallVector<Value> keys;
    SmallVector<RegionExpressions::Id> roots;
    for (const auto& [value, expression] : request.bindings) {
        const bool typed = originalValue(value) && value.getType().isIntOrIndex() &&
            expression < request.expressions->size() &&
            request.expressions->isBoolean(expression) == value.getType().isInteger(1);
        if (!typed) { error = "guarded specialization contains an invalid typed binding"; return {}; }
        auto chosen = request.choices.find(value);
        const bool consistent = chosen == request.choices.end() ||
            request.expressions->constantValue(expression) == std::optional<uint64_t>(chosen->second);
        if (!consistent) { error = "guarded slice choice is not proved by its binding"; return {}; }
        keys.push_back(value); roots.push_back(expression);
    }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    auto arena = sessionState->guardedExpressions;
    // Canonicalize the actual circuits, not their source IDs. Import commits
    // before the producer transaction so failed attempts cannot recycle keys.
    DenseMap<Value, RegionExpressions::Id> bindings;
    if (!roots.empty()) {
        auto imported = arena->import(*request.expressions, roots, &sessionState->guardedImportWork);
        if (failed(imported)) { error = "guarded specialization binding import failed"; return {}; }
        for (auto [value, expression] : llvm::zip(keys, *imported)) { bindings[value] = expression; }
    }
    auto& variants = sessionState->specializedGuarded[request.loop];
    auto equalMap = [](const auto& a, const auto& b) {
        return a.size() == b.size() && llvm::all_of(a, [&](const auto& entry) {
            auto found = b.find(entry.first); return found != b.end() && found->second == entry.second;
        });
    };
    for (const auto& attempt : variants) {
        const bool same = attempt.sliced == request.sliced && attempt.guards == request.guards &&
            equalMap(attempt.choices, request.choices) && equalMap(attempt.bindings, bindings);
        if (same) { error = attempt.error; return attempt.mathematics; }
    }
    auto mathematics = std::make_shared<GuardedRotatingMathematics>();
    mathematics->inputOwner = storage; mathematics->indexOwner = structuralIndex;
    if (!sessionState->guardedInputOwner) {
        sessionState->guardedInputOwner = std::make_shared<std::pair<std::shared_ptr<const SyncInput>,
            std::shared_ptr<const PhaseIndex>>>(storage, structuralIndex);
    }
    mathematics->exportInputOwner = sessionState->guardedInputOwner;

    if (request.sliced) {
        mathematics->recognition = detail::recognizeRotatingSlice(
            request.loop, *structuralIndex, *storage, request.choices, request.guards);
    } else if (node->guardedRotatingResult) { mathematics->recognition = *node->guardedRotatingResult; }
    else {
        mathematics->recognition = recognizeGuardedRotating(
            request.loop, *structuralIndex, *storage, storage->accesses());
    }
    if (mathematics->recognition.result.state == RecognitionState::Applicable) {
        RegionExpressions::Transaction transaction(*arena);
        if (sessionState->specializedGuardedBuilds != UINT64_MAX) { ++sessionState->specializedGuardedBuilds; }
        if (construction.guardedRotatingReductions != UINT64_MAX) { ++construction.guardedRotatingReductions; }
        auto demands = analyzeGuardedRotating(request.loop, *storage, mathematics->recognition,
            *structuralIndex, arena, bindings);
        error = demands.error;
        if (error.empty()) { mathematics->demands = std::move(demands); transaction.commit(); }
    } else { error = "guarded rotating form is unavailable in this specialization"; }
    if (!error.empty()) { mathematics->demands.error = error; }
    variants.push_back({request.sliced, request.choices, request.guards, std::move(bindings), mathematics, error});
    return mathematics;
}
std::shared_ptr<GuardedRotatingMathematics> FrontierAnalysis::instantiateGuardedDemands(
    const GuardedRotatingSpecialization& request, std::shared_ptr<RegionExpressions> target, std::string& error)
{
    auto retained = specializedGuardedDemands(request, error);
    const bool unavailable = !retained || !error.empty();
    if (unavailable) {
        return retained ? std::make_shared<GuardedRotatingMathematics>(*retained) : nullptr;
    }
    if (!target) { error = "guarded instantiation requires an export arena"; return {}; }
    auto imported = importGuardedRotating(retained->demands, target, &sessionState->guardedImportWork);
    if (failed(imported)) { error = "guarded mathematical circuit import failed"; return {}; }
    target->retainInputOwner(retained->exportInputOwner);
    auto result = std::make_shared<GuardedRotatingMathematics>(*retained);
    result->demands = std::move(*imported); return result;
}
uint64_t FrontierAnalysis::specializedNumericConstructions() const
{
    return sessionState ? sessionState->specializedNumericBuilds : 0;
}
std::shared_ptr<const NumericBodyMathematics> FrontierAnalysis::specializedNumericDemands(scf::ForOp loop,
    const TemplateGeometryConstant& geometry, const TemplateControlConstant& control, std::string& error,
    NumericTemplateLimits limits, std::shared_ptr<const NormalizedControlDescription> normalized,
    TemplateGeometryPolicy policy)
{
    const bool valid = initialized && storage && structuralIndex && program && loop &&
        function->isProperAncestor(loop);
    if (!valid) { error = "specialized numeric body belongs to a different invocation"; return {}; }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    const auto node = llvm::find_if(program->nodes, [&](const auto& node) { return node.anchor == loop; });
    if (node == program->nodes.end()) { error = "specialized numeric body requires an original loop"; return {}; }
    if (!normalized) {
        auto description = normalizedInput(*sessionState, function, *program,
            static_cast<std::size_t>(node - program->nodes.begin()), *structuralIndex, *storage);
        normalized = description->original ? description->original : description;
    }
    const bool sameContext = normalized->index == structuralIndex.get() && normalized->input == storage.get() &&
        normalized->context.function == function && normalized->context.root == loop;
    if (!sameContext) { error = "specialized numeric normalization belongs to a different context"; return {}; }
    const auto geometryValue = [&](Value value) { return geometry ? geometry(value) : std::optional<int64_t>{}; };
    const auto controlValue = [&](Value value) { return control ? control(value) : std::optional<bool>{}; };
    const auto key = std::make_tuple(limits.visits, limits.payloads, limits.fragments, limits.depth);
    auto& variants = sessionState->specializedNumeric[loop];
    for (const auto& attempt : variants) {
        if (attempt.limits != key || attempt.normalized != normalized || attempt.policy != policy) { continue; }
        const bool sameGeometry = llvm::all_of(attempt.geometry, [&](const auto& entry) {
            return geometryValue(entry.second.first) == entry.second.second;
        });
        const bool sameControl = llvm::all_of(attempt.control, [&](const auto& entry) {
            return controlValue(entry.second.first) == entry.second.second;
        });
        if (sameGeometry && sameControl) { error = attempt.error; return attempt.mathematics; }
    }
    SpecializedNumericAttempt attempt; attempt.limits = key; attempt.normalized = normalized; attempt.policy = policy;
    const auto observedGeometry = [&](Value value) {
        auto answer = geometryValue(value);
        attempt.geometry.emplace(value.getAsOpaquePointer(), std::make_pair(value, answer)); return answer;
    };
    const auto observedControl = [&](Value value) {
        auto answer = controlValue(value);
        attempt.control.emplace(value.getAsOpaquePointer(), std::make_pair(value, answer)); return answer;
    };
    ++sessionState->specializedNumericBuilds;
    auto body = recognizeSpecializedNumericBody(loop, *structuralIndex, *storage,
        observedGeometry, observedControl, limits, normalized, policy);
    auto demands = analyzeNumericBody(body, attempt.error);
    if (demands) {
        attempt.mathematics = std::make_shared<const NumericBodyMathematics>(
            NumericBodyMathematics{std::move(body), std::move(demands), storage, structuralIndex, normalized});
    }
    error = attempt.error;
    auto result = attempt.mathematics; variants.push_back(std::move(attempt)); return result;
}
uint64_t FrontierAnalysis::finiteArithmeticStorageProbes() const
{
    if (!sessionState) { return 0; }
    return llvm::count_if(sessionState->arithmeticRegionAttempts,
        [](const auto& entry) { return entry.second.finiteArithmeticStorage.has_value(); });
}
FailureOr<RegionalAnalysis> FrontierAnalysis::finiteArithmeticRegion(std::size_t region, std::string& error)
{
    const bool valid = initialized && storage && structuralIndex && program && region < program->nodes.size() &&
        program->nodes[region].kind == StructureKind::Loop && program->nodes[region].anchor;
    if (!valid) { error = "finite arithmetic request requires an original session loop"; return failure(); }
    if (!sessionState) { sessionState = std::make_shared<AnalysisSessionState>(); }
    auto& cached = sessionState->arithmeticRegionAttempts[region];
    if (!cached.finiteArithmeticStorage) {
        const auto* form = recognizeArithmeticRegion(region);
        if (!form) { cached.finiteArithmeticStorageError = "original arithmetic form is unavailable"; }
        cached.finiteArithmeticStorage = form && hasFiniteArithmeticStorage(
            *form, *storage, sessionState->expressions, cached.finiteArithmeticStorageError);
    }
    if (!*cached.finiteArithmeticStorage) { error = cached.finiteArithmeticStorageError; return failure(); }
    auto mathematics = produceArithmeticRegion(region, error);
    if (!mathematics) { return failure(); }
    auto exported = arithmeticExports(*mathematics, true, error);
    if (!exported) { return failure(); }
    return *exported;
}
SequenceRegionResolver FrontierAnalysis::regionalResolver(
    std::shared_ptr<const MathematicalResult> retainedNumerical)
{
    SequenceRegionResolver resolver;
    resolver.region = [this, retainedNumerical](std::size_t region, bool endpoints, std::string& error)
        -> FailureOr<RegionalAnalysis> {
        AnalysisRequest request;
        request.region = region;
        request.needs.queries = request.needs.selectors = true;
        request.needs.synchronization = endpoints;
        const bool numericalAdapter = retainedNumerical && retainedNumerical->numericNode == region;
        auto result = numericalAdapter ? requestBackend(AnalysisBackend::NumericalSequence, request) : analyze(request);
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
    resolver.normalized = [this](std::size_t region) {
        auto description = normalizedRegion(region);
        const bool current = !sessionState->activeRegions.empty() && sessionState->activeRegions.back() == region;
        return current && description->original ? description->original : description;
    };
    resolver.demands = [this, retainedNumerical](std::size_t region, AnalysisBackend backend) {
        const bool currentNumerical = backend == AnalysisBackend::NumericalPeriodic &&
            !sessionState->activeRegions.empty() && sessionState->activeRegions.back() == region;
        if (retainedNumerical && backend == AnalysisBackend::NumericalPeriodic &&
            retainedNumerical->numericNode == region) { return retainedNumerical; }
        if (currentNumerical) {
            auto description = normalizedRegion(region);
            const auto cached = sessionState->loopAttempts.find({region, backend});
            const bool reusable = !description->original && cached != sessionState->loopAttempts.end();
            return reusable ? cached->second.mathematical : std::shared_ptr<const MathematicalResult>{};
        }
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
    resolver.finiteArithmetic = [this](std::size_t region, std::string& error) {
        return finiteArithmeticRegion(region, error);
    };
    resolver.specializedGuarded = [this](const GuardedRotatingSpecialization& request,
        std::shared_ptr<RegionExpressions> target, std::string& error) {
        return instantiateGuardedDemands(request, std::move(target), error);
    };
    resolver.specializedNumeric = [this](scf::ForOp loop, const TemplateGeometryConstant& geometry,
        const TemplateControlConstant& control, std::shared_ptr<const NormalizedControlDescription> normalized,
        TemplateGeometryPolicy policy, std::string& error) {
        return specializedNumericDemands(loop, geometry, control, error, {}, std::move(normalized), policy);
    };
    resolver.specializedDemands = [this](ArithmeticRegionContext context,
        const ArithmeticEntryConstant& constants, std::string& error) {
        return specializedArithmeticDemands(context, constants, error);
    };
    return resolver;
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::adaptNumericalSequence(
    std::size_t region, std::string& error)
{
    // An explicit adapter of retained mathematics, never another producer or
    // an original-form cache entry. Root eligibility was checked by its owner.
    std::shared_ptr<const MathematicalResult> retained;
    auto scope = sessionState->attempts.find(region);
    if (scope != sessionState->attempts.end()) {
        auto producer = scope->second.find(AnalysisBackend::NumericalPeriodic);
        if (producer != scope->second.end()) { retained = producer->second.mathematical; }
    }
    // Root promotion may already own the canonical loop without a named child
    // request. Reading that entry must not invoke or insert a producer.
    if (!retained && region) {
        auto canonical = sessionState->loopAttempts.find({region, AnalysisBackend::NumericalPeriodic});
        if (canonical != sessionState->loopAttempts.end()) { retained = canonical->second.mathematical; }
    }
    if (!retained || !retained->numericalDemands || !retained->numericNode) {
        error = "numerical sequence adapter requires retained exact numerical mathematics"; return {};
    }
    auto sequence = std::make_shared<SequenceAnalysis>(analyzeSequenceRegionWithResolver(
        function, *storage, *program, region, sessionState->expressions, structuralIndex,
        false, regionalResolver(retained)));
    recordSequenceCosts(*sessionState, *sequence, region);
    if (!sequence->error.empty()) { error = sequence->error; return {}; }
    bool imported = retained->numericalDemands->form.emptyInvocation;
    if (region && sequence->state) {
        imported |= llvm::any_of(sequence->state->children, [&](const auto& child) {
            return child.numericTemplate.get() == &retained->numericalDemands->form;
        });
    } else if (!region && sequence->state) {
        for (const auto& child : sequence->state->children) {
            if (child.originalNode != retained->numericNode) { continue; }
            auto attempts = sessionState->attempts.find(*child.originalNode);
            if (attempts == sessionState->attempts.end()) { continue; }
            auto adapter = attempts->second.find(AnalysisBackend::NumericalSequence);
            const bool available = adapter != attempts->second.end() && adapter->second.mathematical;
            if (!available) { continue; }
            // This named child adapter passed the direct pointer-consumption
            // check above; the root resolver forced this exact cached result.
            imported |= adapter->second.mathematical->normalizedInput == retained->normalizedInput;
        }
    }
    if (!imported) { error = "sequence selected another representation instead of numerical import"; return {}; }
    if (!region) {
        recordSequenceContractAttempt(*program, *storage, *sequence);
        const bool complete = program->sequenceContract &&
            program->sequenceContract->membership == ContractStatus::Established &&
            program->sequenceContract->demands == ContractImplementation::Available;
        if (!complete) { error = "numerical sequence adapter does not certify the complete invocation"; return {}; }
    }
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    owned->region = region;
    owned->normalizedInput = retained->normalizedInput;
    owned->regionalDemands = std::make_shared<const RegionalAnalysis>(sequenceRegionalResult(*sequence));
    owned->sequenceDemands = std::move(sequence);
    owned->backend = "numerical-sequence";
    return owned;
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
    if (backend == AnalysisBackend::ExpandedFinite) { return produceExpandedFinite(region, error); }
    if (backend == AnalysisBackend::NumericalSequence) { return adaptNumericalSequence(region, error); }
    if (backend == AnalysisBackend::Sequence || backend == AnalysisBackend::SequenceEndpoints) {
        const bool endpoints = backend == AnalysisBackend::SequenceEndpoints;
        auto sequence = std::make_shared<SequenceAnalysis>(analyzeSequenceRegionWithResolver(
            function, *storage, *program, region, sessionState->expressions, structuralIndex,
            endpoints, regionalResolver()));
        recordSequenceCosts(*sessionState, *sequence, region);
        if (!sequence->error.empty()) { error = sequence->error; return {}; }
        if (region == 0) {
            auto previous = program->sequenceContract;
            recordSequenceContractAttempt(*program, *storage, *sequence);
            const bool complete = program->sequenceContract &&
                program->sequenceContract->membership == ContractStatus::Established &&
                program->sequenceContract->demands == ContractImplementation::Available;
            if (!complete) {
                error = "sequence adapter does not certify the complete invocation";
                const bool retained = previous && previous->membership == ContractStatus::Established &&
                    previous->demands == ContractImplementation::Available;
                if (retained) {
                    program->sequenceContract = std::move(previous);
                    noteSequenceEndpointOutcome(error);
                }
                return {};
            }
        }
        owned->regionalDemands = std::make_shared<const RegionalAnalysis>(sequenceRegionalResult(*sequence));
        owned->sequenceDemands = std::move(sequence);
        owned->backend = endpoints ? "sequence-endpoints" : "sequence";
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
            const auto* form = recognizeArithmeticRegion(region);
            owned->arithmeticRegionalDemands = analyzeArithmeticRegionDemands(
                {function, node.anchor}, *structuralIndex, *storage, form, cached.demandError);
            if (!owned->arithmeticRegionalDemands) { recordArithmeticFormFailure(cached.failure, form); }
            if (owned->arithmeticRegionalDemands) { cached.mathematical = std::move(owned); }
        }
    }
    error = cached.demandError;
    sessionState->attempts[region][AnalysisBackend::Arithmetic].failure = cached.failure;
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
    case AnalysisBackend::NumericalSequence:
        return adaptNumericalSequence(0, error);
    case AnalysisBackend::SequenceEndpoints:
        return produceRegionBackend(backend, 0, error);
    case AnalysisBackend::Sequence: {
        const auto* sequence = analyzeSequenceFunction();
        if (sequence) { recordSequenceCosts(*sessionState, *sequence, 0); }
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
        if (failed(analyzeArithmeticPeriodicFunction())) {
            recordArithmeticFormFailure(sessionState->attempts[0][backend].failure,
                program->arithmetic ? &*program->arithmetic : nullptr);
            error = arithmeticPeriodicAnalysis ? arithmeticPeriodicAnalysis->conversion.diagnostic :
                "arithmetic periodic form is unavailable";
            return {};
        }
        owned->arithmeticPeriodicDemands = arithmeticPeriodicAnalysis;
        owned->backend = "arithmetic-periodic";
        return owned;
    case AnalysisBackend::Arithmetic:
        if (failed(analyzeArithmeticFunction())) {
            recordArithmeticFormFailure(sessionState->attempts[0][backend].failure,
                program->arithmetic ? &*program->arithmetic : nullptr);
            return {};
        }
        owned->arithmeticDemands = arithmeticAnalysis;
        owned->generalArithmeticDemands = generalArithmeticAnalysis;
        owned->backend = "arithmetic";
        return owned;
    case AnalysisBackend::ExpandedFinite:
        return produceExpandedFinite(0, error);
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
