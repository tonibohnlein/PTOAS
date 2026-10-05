// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded actual-IR execution traces for independent logical insertion checks.
#ifndef PTO_TEST_SYNC_LOGICAL_INSERTION_CHECKS_H
#define PTO_TEST_SYNC_LOGICAL_INSERTION_CHECKS_H
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
#include "llvm/Support/JSON.h"
llvm::json::Object traceLogicalInsertion(mlir::func::FuncOp function,
                                        const mlir::pto::frontiersynch::NumericTemplate& input);
mlir::LogicalResult runLogicalInsertionChecks(mlir::func::FuncOp function, mlir::pto::GMAliasPolicy policy,
                                             bool physical = false);
mlir::LogicalResult runPreparedInsertionChecks(mlir::func::FuncOp function);
#endif
