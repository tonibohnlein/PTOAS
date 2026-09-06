// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCAUDITPATHS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCAUDITPATHS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace mlir::pto {
// Exhaustive for result-free choices and literal finite loops within budgets.
// Dynamic loops are unsupported, never "verified" by sampling trip counts.
struct InsertSyncAuditPaths {
    SmallVector<SmallVector<Operation*>> paths;
    Operation* witness = nullptr;
    std::string unsupported;
};
InsertSyncAuditPaths buildInsertSyncAuditPaths(func::FuncOp function);
} // namespace mlir::pto
#endif
