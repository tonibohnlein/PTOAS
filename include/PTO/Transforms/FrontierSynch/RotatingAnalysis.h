// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGANALYSIS_H
#include "PTO/Transforms/FrontierSynch/RotatingExtraction.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
namespace mlir::pto::frontiersynch {
struct RotatingAnalysis {
    std::string error;
    scf::ForOp loop;
    SmallVector<const CompoundInstanceElement*> phases;
    std::vector<RotatingFragment> fragments;
    RotatingExtraction extraction;
    PeriodicAnalysis periodic;
    NumericTemplateEndpoints endpoints;
};
// Certified normalized fragments feed the pure extractor and global quotient.
// This exports completion-origin queries and endpoint recipes, not a regional
// storage selector or physical-allocation certificate.
RotatingAnalysis analyzeRotating(scf::ForOp loop, const PhaseIndex& index,
    const SyncInput& input, const RecognitionResult& recognition);
// Select only one whole-function fixed-body loop. No boundary obligations may
// be omitted. Not applicable returns failure without emitting a diagnostic.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareRotatingInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program);
} // namespace mlir::pto::frontiersynch
#endif
