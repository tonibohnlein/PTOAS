// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCAUDIT_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCAUDIT_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>

namespace mlir::pto {
// Never conflate an unsupported checker domain with a successful verification.
enum class InsertSyncAuditStatus { VerifiedLocal, Uncovered, InvalidToken, Unsupported };
struct InsertSyncAuditResult {
    InsertSyncAuditStatus status = InsertSyncAuditStatus::Unsupported;
    Operation* source = nullptr;
    Operation* target = nullptr;
    std::string reason;
};

// Fresh emitted-IR extraction, independent of InsertSync's alias/dependency
// analysis and mutable synchronization records. Domain: A2/A3 ordinary vector
// UB, exhaustive result-free choices and bounded literal loops, static flags
// and terminal PIPE_ALL. Symbolic loops remain unsupported. Bounds are
// conservative; an uncovered bound is not a device-race witness.
InsertSyncAuditResult auditInsertSyncLocal(func::FuncOp function);
StringRef stringifyInsertSyncAuditStatus(InsertSyncAuditStatus status);
} // namespace mlir::pto
#endif
