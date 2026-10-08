// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingEndpoints(
    func::FuncOp function, GuardedRotatingAnalysis& analysis, std::string& error,
    const RegionalDemandFilter& filter)
{
    if (!analysis.error.empty()) { error = analysis.error; return failure(); }
    GuardedPeriodicEndpointInput input{analysis.loop, analysis.expressions, analysis.phases,
        analysis.payloads, analysis.generators, &analysis.periodic};
    auto plan = prepareGuardedPeriodicEndpoints(function, input, error, filter);
    if (succeeded(plan) && !filter) {
        (*plan)->allocationCertificate = guardedAllocationCertificate(analysis, (*plan)->planId, function.getContext());
    }
    return plan;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    if (function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) { return failure(); }
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (node.kind == StructureKind::Loop && node.anchor->getParentOp() == function) {
            if (selected || !node.guardedRotatingResult ||
                node.guardedRotatingResult->result.state != RecognitionState::Applicable) { return failure(); }
            selected = &node;
        }
    }
    if (!selected) { return failure(); }
    auto loop = dyn_cast<scf::ForOp>(selected->anchor);
    if (!loop || llvm::any_of(input.instructions(), [&](const auto* phase) {
        return !loop->isProperAncestor(phase->elementOp);
    })) { return failure(); }
    PhaseIndex index;
    if (failed(index.build(function, input))) { return failure(); }
    RecognitionResult outside;
    for (auto& operation : function.front()) {
        if (&operation != loop.getOperation()) {
            detail::inspectLeaf(operation, index, outside);
            if (operation.getNumRegions()) { return failure(); }
        }
    }
    if (outside.state != RecognitionState::Applicable) { return failure(); }
    auto analysis = analyzeGuardedRotating(loop, input, *selected->guardedRotatingResult);
    std::string error;
    return prepareGuardedRotatingEndpoints(function, analysis, error);
}
} // namespace mlir::pto::frontiersynch
