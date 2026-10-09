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
    return prepareLogical(result);
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
    AnalysisBackend backend, std::string& error)
{
    const auto* node = wholeLoop(function, *storage, *program, *structuralIndex);
    if (!node) { error = "route requires one certified whole-invocation loop"; return {}; }
    const auto loop = cast<scf::ForOp>(node->anchor);
    auto owned = std::make_shared<MathematicalResult>();
    owned->input = storage;
    owned->recognition = program;
    switch (backend) {
    case AnalysisBackend::NumericalPeriodic: {
        if (failed(analyzeNumericCandidates())) { return {}; }
        const bool available = node->numericTemplate && node->periodicAnalysis &&
            node->periodicAnalysis->error.empty() && !node->numericTemplate->specializedBody;
        if (!available) { return {}; }
        owned->numericNode = static_cast<std::size_t>(node - program->nodes.data());
        owned->backend = "numerical-periodic";
        break;
    }
    case AnalysisBackend::Rotating: {
        if (!node->rotatingResult || node->rotatingResult->state != RecognitionState::Applicable) { return {}; }
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
        auto demands = std::make_shared<GuardedRotatingAnalysis>(
            analyzeGuardedRotating(loop, *storage, *node->guardedRotatingResult, *structuralIndex));
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
        boundedAnalysis = *demands;
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
        explicitMathematical = owned;
        return owned;
    case AnalysisBackend::NumericalPeriodic:
    case AnalysisBackend::Rotating:
    case AnalysisBackend::GuardedRotating:
    case AnalysisBackend::BoundedLifetime:
        return produceLoopBackend(backend, error);
    case AnalysisBackend::Sequence: {
        const auto* sequence = analyzeSequenceFunction();
        if (!sequence || !sequence->error.empty()) {
            error = sequenceAnalysis ? sequenceAnalysis->error : "sequence form unavailable";
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
