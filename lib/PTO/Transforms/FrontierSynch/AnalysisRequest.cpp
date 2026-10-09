// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Request protocol adapters. Cache mathematical construction independently of
// export checks; detached command fragments always have fresh ownership.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
namespace mlir::pto::frontiersynch {
AnalysisOutcome FrontierAnalysis::minimumDemands(std::size_t region, AnalysisNeeds exports)
{
    exports.synchronization = false;
    return analyze({region, AnalysisMode::MinimumExact, exports});
}
AnalysisOutcome FrontierAnalysis::analyze(const AnalysisRequest& request)
{
    AnalysisOutcome result;
    if (!storage || !structuralIndex) {
        result.obligations.push_back({AnalysisStage::Form, "analysis session is not initialized"});
        return result;
    }
    if (request.region != 0) {
        result.obligations.push_back({AnalysisStage::Form, "regional request adapter is unavailable"});
        return result;
    }
    if (failed(analyzeExplicitFunction())) {
        result.status = explicitAnalysis ? AnalysisStatus::UnmetObligation : AnalysisStatus::NotApplicable;
        result.stage = explicitAnalysis ? AnalysisStage::Demands : AnalysisStage::Form;
        result.obligations.push_back({result.stage,
            explicitAnalysis ? explicitAnalysis->error : "explicit function form is unavailable"});
        return result;
    }
    if (!explicitMathematical) {
        auto owned = std::make_shared<MathematicalResult>();
        owned->input = storage;
        owned->explicitDemands = explicitAnalysis;
        owned->backend = "explicit";
        explicitMathematical = std::move(owned);
    }
    result.mathematical = explicitMathematical;
    result.available.queries = true;
    result.available.selectors = true;
    result.status = AnalysisStatus::Ready;
    result.stage = AnalysisStage::None;
    if (request.needs.evaluation == AnalysisEvaluation::Stateful) {
        result.status = AnalysisStatus::UnmetObligation;
        result.stage = AnalysisStage::Demands;
        result.obligations.push_back({result.stage, "stateful evaluation is unavailable"});
    } else if (request.needs.synchronization) {
        if (!explicitEndpointOutcome) {
            auto prepared = prepareLogical(result);
            explicitEndpointOutcome = succeeded(prepared);
        }
        result.available.synchronization = *explicitEndpointOutcome;
        if (!result.available.synchronization) {
            result.status = AnalysisStatus::UnmetObligation;
            result.stage = AnalysisStage::Synchronization;
            result.obligations.push_back({result.stage, "explicit endpoint preparation is unavailable"});
        }
    }
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareLogical(const AnalysisOutcome& result)
{
    if (!result.mathematical || result.mathematical != explicitMathematical ||
        result.mathematical->input != storage) {
        return failure();
    }
    if (construction.logicalPreparations != UINT64_MAX) { ++construction.logicalPreparations; }
    auto prepared = prepareExplicitInsertion(function, *result.mathematical->explicitDemands, false);
    if (succeeded(prepared)) { (*prepared)->mathematicalOwner = result.mathematical; }
    return prepared;
}
LogicalResult FrontierAnalysis::attachAllocation(const AnalysisOutcome& result, PreparedLogicalPlan& plan)
{
    if (!result.mathematical || result.mathematical != explicitMathematical ||
        result.mathematical->input != storage || plan.mathematicalOwner != result.mathematical) {
        return failure();
    }
    if (construction.allocationExports != UINT64_MAX) { ++construction.allocationExports; }
    plan.allocationCertificate = explicitAllocationCertificate(
        *result.mathematical->explicitDemands, plan.planId, function.getContext());
    return success(static_cast<bool>(plan.allocationCertificate));
}
void FrontierAnalysis::invalidate()
{
    // Dropping modeled input makes all old handles ineligible for preparation.
    // initialize rebuilds every dependent cache before accepting another request.
    explicitMathematical.reset();
    explicitEndpointOutcome.reset();
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
    program.reset();
    structuralIndex.reset();
    storage.reset();
    initialized = false;
}
} // namespace mlir::pto::frontiersynch
