// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Physical sections select independent executions. Temporary analysis projections
// preserve source cuts; only prepared synchronization is mapped back to the IR.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_EXECUTIONCONTEXTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_EXECUTIONCONTEXTS_H
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir::pto::frontiersynch {
inline constexpr llvm::StringLiteral ContextPlansAttr = "pto.sync_context_plans";
inline constexpr llvm::StringLiteral ActiveContextAttr = "pto.active_sync_context";
inline constexpr llvm::StringLiteral ContextAttr = "pto.sync_context";
inline constexpr llvm::StringLiteral ContextLoopsAttr = "pto.sync_context_loops";
bool hasPhysicalSections(func::FuncOp function);
// When no context is active, preserves the ordinary whole-function behavior.
bool belongsToActiveContext(func::FuncOp function, Operation* operation);
using PrepareContext = llvm::function_ref<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(func::FuncOp)>;
LogicalResult insertContextSynchronization(func::FuncOp function, PrepareContext prepare);
LogicalResult allocateContextSynchronization(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
