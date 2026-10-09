// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact finite potential-event analysis; guards remain shared circuits.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEGUARDEDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEGUARDEDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
namespace mlir::pto::frontiersynch {
struct FiniteGuardedState;
struct ArithmeticProgram;
struct FiniteGuardedAnalysis {
    std::string error;
    std::string insertionError;
    RegionalCost cost;
    std::shared_ptr<FiniteGuardedState> state;
    std::shared_ptr<const ArithmeticProgram> expandedProgram;
};
// Original roots are adjacent operations in the function body. Nested if/else
// is allowed, loops and unresolved value prerequisites are not. Input/IR remain
// borrowed and must stay unchanged. No branch valuation is enumerated.
FiniteGuardedAnalysis analyzeFiniteGuarded(func::FuncOp function, ArrayRef<Operation*> roots,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions = {});
FiniteGuardedAnalysis analyzeExpandedFinite(func::FuncOp function, Operation* root,
    const PhaseIndex& index, const SyncInput& input);
FiniteGuardedAnalysis analyzeExpandedFinite(func::FuncOp function, ArrayRef<Operation*> roots,
    const PhaseIndex& index, const SyncInput& input);
RegionalAnalysis finiteGuardedRegionalResult(const FiniteGuardedAnalysis& analysis);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteGuardedLogicalInsertion(FiniteGuardedAnalysis& analysis);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareFiniteGuardedInsertion(FiniteGuardedAnalysis& analysis);
} // namespace mlir::pto::frontiersynch
#endif
