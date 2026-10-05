// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICINSERTION_H
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
// Analysis and selectors remain available if original cuts cannot carry their
// executable predicates. Preparation is detached; no physical IDs are assigned.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
