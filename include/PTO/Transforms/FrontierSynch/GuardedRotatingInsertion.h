// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGINSERTION_H
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
namespace mlir::pto::frontiersynch {
// Prepare compact endpoints, hoisting immutable decisions before the loop.
// Failure leaves original IR unchanged, including failed availability checks.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingEndpoints(
    func::FuncOp function, GuardedRotatingAnalysis& analysis, std::string& error);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGuardedRotatingInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program);
} // namespace mlir::pto::frontiersynch
#endif
