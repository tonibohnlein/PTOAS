// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared pass state: input extraction followed by structural route recognition.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ExecutionContexts.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
namespace mlir::pto::frontiersynch {
namespace {
// Whole-invocation shortcut only: every payload uses the protected scalar
// pipe, and every other effect is accounted for by the shared leaf contract.
// No storage geometry, visit enumeration or regional selector is required.
bool nativeScalarRequirements(func::FuncOp function, const SyncInput& input)
{
    auto scalar = ptoStorageProtection().scalarPipe;
    if (!scalar || function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) ||
        llvm::any_of(input.instructions(), [&](const auto* phase) {
            return static_cast<uint32_t>(phase->kPipeValue) != *scalar ||
                llvm::any_of(phase->elementOp->getResults(), [&](Value value) {
                    return resultAvailability(*phase, value) == SyncResultAvailability::RequiresCompletion;
                });
        })) { return false; }
    PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    RecognitionResult checked;
    function.walk([&](Operation* op) {
        if (op == function.getOperation()) { return; }
        if (llvm::any_of(index.prerequisitesFor(op), [](const auto& prerequisite) {
                return !prerequisite.native;
            })) {
            checked.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (op->getNumRegions()) {
            if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
                checked.note(RecognitionIssue::UnsupportedControl, op, true);
            }
            return;
        }
        detail::inspectLeaf(*op, index, checked);
    });
    return checked.state == RecognitionState::Applicable;
}
} // namespace
LogicalResult FrontierAnalysis::initialize(GMAliasPolicy requestedPolicy, bool requireStructure) {
    if (initialized && policy == requestedPolicy) {
        if (!storage) { return failure(); }
        return nativeScalarOnly && !requireStructure ? success() : recognizeStructure();
    }
    nativeScalarOnly = false;
    explicitAnalysis.reset();
    program.reset();
    storage.reset();
    initialized = true;
    policy = requestedPolicy;
    if (!function) {
        return failure();
    }
    auto pending = std::make_unique<SyncInput>(policy);
    if (failed(pending->build(function, SyncInstructionView::PipeEnvelopes))) {
        return failure();
    }
    nativeScalarOnly = nativeScalarRequirements(function, *pending);
    storage = std::move(pending);
    return nativeScalarOnly && !requireStructure ? success() : recognizeStructure();
}
LogicalResult FrontierAnalysis::recognizeStructure() {
    if (program) { return success(); }
    if (!storage) { return failure(); }
    auto recognized = recognizeProgram(function, *storage);
    if (failed(recognized)) { return failure(); }
    program = std::move(*recognized);
    return success();
}
LogicalResult FrontierAnalysis::analyzeNumericCandidates() {
    if (failed(recognizeStructure())) { return failure(); }
    for (auto& node : program->nodes) {
        if (!node.numericTemplate || node.periodicAnalysis ||
            node.numericTemplate->result.state != RecognitionState::Applicable) { continue; }
        node.periodicAnalysis = analyzeNumericTemplate(*node.numericTemplate);
        if (!node.periodicAnalysis->error.empty()) { continue; }
        node.logicalEndpoints = buildNumericTemplateEndpoints(*node.numericTemplate, *node.periodicAnalysis);
        if (!node.logicalEndpoints->logical.error.empty()) { continue; }
        node.periodicAllocation = buildPeriodicAllocation(*node.periodicAnalysis);
    }
    return success();
}
LogicalResult FrontierAnalysis::analyzeExplicitFunction() {
    if (!storage || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    if (!explicitAnalysis) {
        PhaseIndex index;
        if (failed(index.build(function, *storage))) {
            return failure();
        }
        explicitAnalysis = analyzeExplicit(function.front(), index, *storage);
    }
    return success(explicitAnalysis->error.empty());
}
LogicalResult FrontierAnalysis::recognizeArithmetic() {
    if (failed(recognizeStructure())) { return failure(); }
    if (program->arithmetic || function.isDeclaration()) {
        return success();
    }
    PhaseIndex index;
    if (failed(index.build(function, *storage))) {
        return failure();
    }
    // Try the smaller configured residue class first. Ordinary affine regions
    // need no parity expansion; fixed-modulus/step-two inputs use the second
    // class when the complete period-one primitive contract does not apply.
    auto arithmetic = recognizeArithmeticProgram(function, index, *storage, storage->accesses(), {8, 8, 1, 8});
    if (arithmetic.extraction.state != RecognitionState::Applicable ||
        arithmetic.recognition.state != RecognitionState::Applicable) {
        arithmetic = recognizeArithmeticProgram(function, index, *storage, storage->accesses(), {8, 8, 2, 8});
    }
    program->arithmetic = std::move(arithmetic);
    return success();
}
} // namespace mlir::pto::frontiersynch
namespace mlir::pto {
#define GEN_PASS_DEF_PTOFRONTIERANALYSIS
#include "PTO/Transforms/Passes.h.inc"
namespace {
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepareWholeFunctionArithmetic(
    func::FuncOp function, frontiersynch::FrontierAnalysis& analysis, std::string& error,
    bool requireAllocationCertificate)
{
    if (failed(analysis.recognizeArithmetic()) || !analysis.result()->arithmetic) { return failure(); }
    const auto& arithmetic = *analysis.result()->arithmetic;
    const auto protection = frontiersynch::structuredProtection(analysis.input()->accesses());
    if (arithmetic.recognition.arithmeticClass == frontiersynch::ArithmeticClass::Differences) {
        auto demands = frontiersynch::analyzeArithmeticDemandsWithProtection(arithmetic, protection);
        if (!demands.error.empty()) { error += "; arithmetic: " + demands.error; return failure(); }
        auto prepared = frontiersynch::prepareArithmeticInsertion(function, arithmetic, demands, error);
        if (succeeded(prepared) && requireAllocationCertificate && !(*prepared)->allocationCertificate) {
            return failure();
        }
        return prepared;
    }
    // The existing general arithmetic producer has no allocation export.
    if (requireAllocationCertificate) { return failure(); }
    auto demands = frontiersynch::analyzeGeneralArithmeticDemandsWithProtection(arithmetic, protection);
    if (!demands.error.empty()) { error += "; arithmetic: " + demands.error; return failure(); }
    return frontiersynch::prepareGeneralArithmeticInsertion(function, arithmetic, demands, error);
}
FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepareFunction(
    func::FuncOp function, GMAliasPolicy policy)
{
    frontiersynch::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize(policy, /*requireStructure=*/false))) {
        return failure();
    }
    FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepared = failure();
    std::string routeError;
    bool sequencePrepared = false;
    const bool straight = !function.isDeclaration() && llvm::hasSingleElement(function.getBody()) &&
        llvm::all_of(function.front(), [](Operation& op) { return op.getNumRegions() == 0; });
    if (analysis.hasOnlyNativeScalarRequirements()) {
        auto scalar = std::make_unique<frontiersynch::PreparedLogicalPlan>(0);
        // Protected scalar accesses need no internal demands. Keep the
        // invocation drain: protection is not completion at issue.
        scalar->completeInvocation = !analysis.input()->instructions().empty();
        frontiersynch::ExplicitAnalysis empty;
        scalar->allocationCertificate = frontiersynch::explicitAllocationCertificate(
            empty, scalar->planId, function.getContext());
        prepared = std::move(scalar);
    } else if (straight) {
        if (succeeded(analysis.analyzeExplicitFunction())) {
            prepared = frontiersynch::prepareExplicitInsertion(function, *analysis.explicitResult());
        } else {
            routeError = analysis.explicitResult() ? analysis.explicitResult()->error :
                         "explicit analysis unavailable";
        }
    } else {
        // Preserve the existing certified numerical route (including its
        // allocation export) when available. Direct rotating extraction
        // covers symbolic rotations which have no fixed local effect word.
        if (failed(analysis.analyzeNumericCandidates())) { return failure(); }
        const bool numerical = llvm::any_of(analysis.result()->nodes, [](const auto& node) {
            return node.numericTemplate &&
                node.numericTemplate->result.state == frontiersynch::RecognitionState::Applicable &&
                node.logicalEndpoints && node.logicalEndpoints->logical.error.empty();
        });
        if (!numerical) {
            prepared = frontiersynch::prepareRotatingInsertion(function, *analysis.input(), *analysis.result());
        }
        if (failed(prepared) && numerical) {
            prepared = frontiersynch::prepareNumericTemplateInsertion(function, *analysis.result());
        }
        if (failed(prepared)) {
            prepared = frontiersynch::prepareGuardedRotatingInsertion(
                function, *analysis.input(), *analysis.result());
        }
        if (failed(prepared)) {
            prepared = frontiersynch::prepareSequenceInsertion(function, *analysis.input(),
                                                               *analysis.result(), routeError);
            sequencePrepared = succeeded(prepared);
        }
    }
    if (failed(prepared) && straight && analysis.result()) {
        prepared = frontiersynch::prepareSequenceInsertion(function, *analysis.input(),
                                                           *analysis.result(), routeError);
        sequencePrepared = succeeded(prepared);
    }
    if (failed(prepared)) {
        prepared = prepareWholeFunctionArithmetic(function, analysis, routeError, false);
    } else if (sequencePrepared && !(*prepared)->allocationCertificate && !(*prepared)->regionalAllocation &&
               llvm::any_of((*prepared)->endpoints, [](const auto& endpoint) {
                   return endpoint.kind != frontiersynch::LogicalCommandKind::Barrier;
               })) {
        // Keep the accepted logical plan unless another exact route also
        // supplies the existing allocation interface. Both preparations
        // remain detached; this neither assigns IDs nor repairs scarcity.
        std::string allocationRouteError;
        auto alternative = prepareWholeFunctionArithmetic(function, analysis, allocationRouteError, true);
        if (succeeded(alternative)) { prepared = std::move(alternative); }
    }
    if (failed(prepared)) { function.emitError(routeError); }
    return prepared;
}
class PTOFrontierAnalysisPass : public impl::PTOFrontierAnalysisBase<PTOFrontierAnalysisPass> {
public:
    using Base = impl::PTOFrontierAnalysisBase<PTOFrontierAnalysisPass>;
    PTOFrontierAnalysisPass() = default;
    explicit PTOFrontierAnalysisPass(const PTOFrontierAnalysisOptions& options) : Base(options) {}
    void runOnOperation() override {
        if (gmAlias != "may-alias" && gmAlias != "may-not-alias") {
            getOperation().emitError("gm-alias must be may-alias or may-not-alias");
            signalPassFailure();
            return;
        }
        auto function = getOperation();
        if (function.isDeclaration() || hasManualOnCoreSynchronization(function)) {
            return;
        }
        auto policy = gmAlias == "may-alias" ? GMAliasPolicy::MayAlias : GMAliasPolicy::MayNotAlias;
        if (frontiersynch::hasPhysicalSections(function)) {
            if (failed(frontiersynch::insertContextSynchronization(function, [&](func::FuncOp projected) {
                    return prepareFunction(projected, policy);
                }))) { signalPassFailure(); }
            return;
        }
        auto prepared = prepareFunction(function, policy);
        if (failed(prepared) || failed(frontiersynch::insertLogicalSynchronization(function, **prepared))) {
            signalPassFailure();
        }
        // Inserted guards and commands invalidate structural analysis.
    }
};
} // namespace
std::unique_ptr<Pass> createPTOFrontierAnalysisPass() {
    return std::make_unique<PTOFrontierAnalysisPass>();
}
std::unique_ptr<Pass> createPTOFrontierAnalysisPass(const PTOFrontierAnalysisOptions& options) {
    return std::make_unique<PTOFrontierAnalysisPass>(options);
}
} // namespace mlir::pto
