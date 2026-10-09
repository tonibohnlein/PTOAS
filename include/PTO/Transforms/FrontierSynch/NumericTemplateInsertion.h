// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical-template adapter for the common logical inserter.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEINSERTION_H
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
namespace mlir::pto::frontiersynch {
// Recognition borrows the unchanged function and SyncInput. Preparation validates
// the complete template and creates detached arithmetic, leaving original IR intact.
// Shared counted-loop endpoint arithmetic. Caller supplies certified recipes
// and matching original cuts; no numeric effect-template assumption is made.
// Preserve whole-invocation bounds, scope and recipe checks for an owned result.
LogicalResult validateNumericTemplateInsertion(func::FuncOp function, const NumericalRegionDemands& demands,
                                              const NumericTemplateEndpoints& endpoints, const SyncInput& input);
LogicalResult prepareCountedEndpointCode(func::FuncOp function, const NumericTemplateEndpoints& endpoints,
                                         PreparedLogicalPlan& prepared);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareNumericTemplateLogicalInsertion(
    func::FuncOp function, const ProgramRecognition& program, int64_t planId = 0);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareNumericTemplateInsertion(
    func::FuncOp function, const ProgramRecognition& program, int64_t planId = 0);
} // namespace mlir::pto::frontiersynch
#endif
