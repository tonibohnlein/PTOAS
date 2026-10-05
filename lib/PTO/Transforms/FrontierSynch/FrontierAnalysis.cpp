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
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
namespace mlir::pto::frontiersynch {
LogicalResult FrontierAnalysis::initialize(GMAliasPolicy requestedPolicy) {
    if (initialized && policy == requestedPolicy) {
        return success(program.has_value());
    }
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
    auto recognized = recognizeProgram(function, *pending, {8, 8, 2, 8});
    if (failed(recognized)) {
        return failure();
    }
    storage = std::move(pending);
    program = std::move(*recognized);
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
        if (failed(frontiersynch::insertLogicalSynchronization(getOperation(), *analysis.result()))) {
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
