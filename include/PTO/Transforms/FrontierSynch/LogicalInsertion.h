// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Insert certified logical synchronization while preserving physical ID freedom.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALINSERTION_H
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
namespace mlir::pto::frontiersynch {
// Requires fresh recognition borrowing this unchanged function and SyncInput.
// Preflights the complete whole-function numeric-template plan before mutation.
// Success invalidates recognition; unsupported input leaves original IR intact.
// Emits logical SET/WAIT and local barriers, never physical event IDs.
LogicalResult insertLogicalSynchronization(func::FuncOp function, const ProgramRecognition& program);
} // namespace mlir::pto::frontiersynch
#endif
