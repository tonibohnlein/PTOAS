// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Session-local requests name an original structural node (zero is the root).
// Mathematical ownership is independent of endpoint and allocation outcomes.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISREQUEST_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ANALYSISREQUEST_H
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include <memory>
namespace mlir::pto::frontiersynch {
enum class AnalysisMode { MinimumExact, Fallback };
enum class AnalysisEvaluation { Uniform, Stateful };
enum class AnalysisStatus { Ready, NotApplicable, UnmetObligation };
enum class AnalysisStage { None, Form, Demands, Queries, Selectors, Synchronization, Allocation };
struct AnalysisNeeds {
    bool queries = false;
    bool selectors = false;
    bool synchronization = false;
    AnalysisEvaluation evaluation = AnalysisEvaluation::Uniform;
};
struct AnalysisRequest {
    std::size_t region = 0;
    AnalysisMode mode = AnalysisMode::MinimumExact;
    AnalysisNeeds needs;
};
struct AnalysisExports {
    bool queries = false;
    bool selectors = false;
    bool synchronization = false;
    bool allocation = false;
};
struct AnalysisObligation {
    AnalysisStage stage = AnalysisStage::None;
    std::string diagnostic;
};
// This handle owns the modeled accesses referenced by the original occurrences.
// Original MLIR must still outlive it and remain unchanged, as with MLIR analyses.
struct MathematicalResult {
    std::shared_ptr<const SyncInput> input;
    std::shared_ptr<const ExplicitAnalysis> explicitDemands;
    std::size_t region = 0;
    std::string backend;
};
struct AnalysisOutcome {
    AnalysisStatus status = AnalysisStatus::NotApplicable;
    AnalysisStage stage = AnalysisStage::Form;
    std::shared_ptr<const MathematicalResult> mathematical;
    AnalysisExports available;
    std::vector<AnalysisObligation> obligations;
};
struct AnalysisConstructionCounts {
    uint64_t structuralIndices = 0;
    uint64_t explicitReductions = 0;
    uint64_t logicalPreparations = 0;
    uint64_t allocationExports = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
