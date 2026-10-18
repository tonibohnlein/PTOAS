// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticInsertion.h"
#include "GeneralArithmeticEndpointRecipe.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/GeneralArithmeticAllocation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticLogicalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error, ArithmeticRecordMap& records)
{
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.hasRelations(ArithmeticRelationView::Demands) ||
        bits.isScalable() || bits.getFixedValue() != 64 || !analysis.period || analysis.period > INT64_MAX ||
        analysis.parameterCount != program.parameters.size()) {
        error = "arithmetic insertion requires reduced demands and a supported index representation";
        return failure();
    }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.sites) {
        if (!site.phase) { return failure(); }
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    auto selectors = buildArithmeticSelectors(analysis, pipes);
    if (!selectors.error.empty()) { error = selectors.error; return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !program.sites.empty();
    detail::DifferenceArithmeticEndpointRecipeBuilder preparer(program, selectors, *plan, error);
    if (failed(preparer.run(records))) {
        if (error.empty()) { error = "arithmetic selector cannot be emitted at its original cut"; }
        return failure();
    }
    return plan;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error)
{
    ArithmeticRecordMap records;
    auto logical = prepareArithmeticLogicalInsertion(function, program, analysis, error, records);
    if (failed(logical)) { return failure(); }
    auto plan = std::move(*logical);
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.sites) { pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue)); }
    plan->allocationCertificate = arithmeticAllocationCertificate(analysis, pipes, records,
                                                                  plan->planId, function.getContext());
    if (!plan->allocationCertificate && program.context.root == function.getOperation()) {
        auto proof = buildArithmeticHandoffAllocation(analysis, pipes);
        plan->allocationCertificate = encodeGeneralArithmeticAllocationCertificate(
            proof, pipes, records, plan->planId, function.getContext());
    }
    return plan;
}
namespace {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneral(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error, bool regional,
    ArithmeticRecordMap* records = nullptr)
{
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.hasRelations(ArithmeticRelationView::Demands) ||
        bits.isScalable() || bits.getFixedValue() != 64 || !analysis.period || analysis.period > INT64_MAX ||
        analysis.parameterCount != program.parameters.size()) {
        error = "arithmetic insertion requires reduced demands and a supported index representation";
        return failure();
    }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.sites) {
        if (!site.phase) { return failure(); }
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    auto selectors = buildGeneralArithmeticSelectors(analysis, pipes);
    if (!selectors.error.empty()) { error = selectors.error; return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !program.sites.empty();
    ArithmeticRecordMap retainedRecords;
    detail::GeneralArithmeticEndpointRecipeBuilder preparer(program, selectors, *plan, error);
    if (failed(preparer.run(retainedRecords))) {
        if (error.empty()) { error = "arithmetic selector recipe unavailable at its original cut"; }
        return failure();
    }
    // Both output routes retain the same exact matching tuple. Counter palettes
    // remain keyed by original-record provenance within the grouped namespace.
    if (regional) { plan->completeInvocation = false; }
    if (records) { *records = retainedRecords; }
    if (!records && !regional && program.context.root == function.getOperation()) {
        plan->allocationCertificate = generalArithmeticAllocationCertificate(
            analysis, pipes, retainedRecords, plan->planId, function.getContext());
    }
    return plan;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticLogicalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error, ArithmeticRecordMap& records)
{
    return prepareGeneral(function, program, analysis, error, false, &records);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error)
{
    return prepareGeneral(function, program, analysis, error, false);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticRegionalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error)
{
    return prepareGeneral(function, program, analysis, error, true);
}
} // namespace mlir::pto::frontiersynch
