// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared pass state: input extraction followed by structural route recognition.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
namespace mlir::pto::frontiersynch {
LogicalResult FrontierAnalysis::initialize(GMAliasPolicy requestedPolicy) {
    if (initialized && policy == requestedPolicy) {
        return success(program.has_value());
    }
    explicitAnalysis.reset();
    program.reset();
    storage.reset();
    initialized = true;
    policy = requestedPolicy;
    if (!function) {
        return failure();
    }
    auto pending = std::make_unique<SyncInput>(policy);
    if (failed(pending->build(function))) {
        return failure();
    }
    auto recognized = recognizeProgram(function, *pending);
    if (failed(recognized)) {
        return failure();
    }
    storage = std::move(pending);
    program = std::move(*recognized);
    return success();
}
LogicalResult FrontierAnalysis::analyzeExplicitFunction() {
    if (!program || !storage || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
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
    if (!program || !storage) {
        return failure();
    }
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
        auto policy = gmAlias == "may-alias" ? GMAliasPolicy::MayAlias : GMAliasPolicy::MayNotAlias;
        auto& analysis = getAnalysis<frontiersynch::FrontierAnalysis>();
        if (failed(analysis.initialize(policy))) {
            signalPassFailure();
            return;
        }
        FailureOr<std::unique_ptr<frontiersynch::PreparedLogicalPlan>> prepared = failure();
        auto function = getOperation();
        std::string routeError;
        const bool straight = !function.isDeclaration() && llvm::hasSingleElement(function.getBody()) &&
            llvm::all_of(function.front(), [](Operation& op) { return op.getNumRegions() == 0; });
        if (straight) {
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
            }
        }
        if (failed(prepared) && succeeded(analysis.recognizeArithmetic()) && analysis.result()->arithmetic) {
            const auto& arithmetic = *analysis.result()->arithmetic;
            SmallVector<const CompoundInstanceElement*> phases;
            for (const auto& site : arithmetic.sites) { phases.push_back(site.phase); }
            if (!frontiersynch::mayHaveHardwareProtectedPair(*analysis.input(), phases)) {
                if (arithmetic.recognition.arithmeticClass == frontiersynch::ArithmeticClass::Differences) {
                    auto demands = frontiersynch::analyzeArithmeticDemands(arithmetic);
                    if (demands.error.empty()) {
                        prepared = frontiersynch::prepareArithmeticInsertion(function, arithmetic, demands, routeError);
                    } else { routeError += "; arithmetic: " + demands.error; }
                } else {
                    auto demands = frontiersynch::analyzeGeneralArithmeticDemands(arithmetic);
                    if (demands.error.empty()) {
                        prepared = frontiersynch::prepareGeneralArithmeticInsertion(
                            function, arithmetic, demands, routeError);
                    } else { routeError += "; arithmetic: " + demands.error; }
                }
            }
        }
        if (failed(prepared)) { function.emitError(routeError); }
        if (failed(prepared) || failed(frontiersynch::insertLogicalSynchronization(getOperation(), **prepared))) {
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
