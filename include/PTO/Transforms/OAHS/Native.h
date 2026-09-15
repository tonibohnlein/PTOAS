// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_NATIVE_H
#define PTO_TRANSFORMS_OAHS_NATIVE_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir::pto::oahs {
// Imports and validates complete shared semantics without changing the IR.
LogicalResult analyzeHandoffSync(func::FuncOp function);
// Runs through production translation/alias analysis and SyncCodegen. Failure
// leaves the original function unchanged; no backend fallback is performed.
LogicalResult runHandoffSync(func::FuncOp function);
namespace testing {
// Mutation occurs only on the private working copy, before reconstruction.
LogicalResult runHandoffSyncWithMutation(func::FuncOp function,
    llvm::function_ref<void(func::FuncOp)> mutate);
}
}
#endif
