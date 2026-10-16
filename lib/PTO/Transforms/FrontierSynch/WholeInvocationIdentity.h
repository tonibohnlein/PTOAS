// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// An identity adapter retains the selected graph, not a universal relation.
#ifndef PTO_FRONTIERSYNCH_WHOLEINVOCATIONIDENTITY_H
#define PTO_FRONTIERSYNCH_WHOLEINVOCATIONIDENTITY_H
#include "AnalysisSessionInternal.h"
namespace mlir::pto::frontiersynch {
const StructureNode* wholeInvocationLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, const PhaseIndex& index);
// This is a proof about the complete original invocation, not an empty demand
// set or a selected empty child surrounded by executing payloads.
bool staticallyEmptyWholeInvocation(const MathematicalResult& demands, func::FuncOp function,
    const SyncInput& input, const ProgramRecognition& program, const PhaseIndex& index);
// Pure retained-owner lookup for the restricted same-node continuation.
std::optional<std::pair<AnalysisOutcome, AnalysisBackend>> retainedIdentityAnswer(
    const AnalysisSessionState& session, const AnalysisRequest& request, ArrayRef<AnalysisBackend> eligible);
struct WholeInvocationIdentity {
    std::shared_ptr<const SyncInput> input;
    std::shared_ptr<const PhaseIndex> index;
    std::shared_ptr<const ProgramRecognition> program;
    std::shared_ptr<const MathematicalResult> child;
    func::FuncOp function;
    Operation* loop = nullptr;
    std::size_t region = 0, childRegion = 0;
    AnalysisBackend childBackend = AnalysisBackend::Explicit;
    bool applies(const SyncInput* expectedInput, const PhaseIndex* expectedIndex,
        const ProgramRecognition* expectedProgram, func::FuncOp expectedFunction) const;
};
} // namespace mlir::pto::frontiersynch
#endif
