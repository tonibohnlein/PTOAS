// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Bounded occurrence expansion in the disposable emission module only.
#ifndef PTO_PROTOCOLSYNC_STRUCTUREDNORMALIZATION_H
#define PTO_PROTOCOLSYNC_STRUCTUREDNORMALIZATION_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir::pto::protocol_sync {
struct SyncNormalizationResult {
    unsigned loops = 0;
    unsigned iterations = 0;
    unsigned restrictedDomains = 0;
};
SyncNormalizationResult expandSyncInnerOccurrences(func::FuncOp function);
/// A once-only choice with a scalar condition that can be computed at entry.
/// Each specialization retains the complete physical prefix and suffix.
scf::IfOp findSyncPathChoice(func::FuncOp function);
Value materializeSyncPathCondition(OpBuilder& builder, scf::IfOp choice);
void specializeSyncPath(scf::IfOp choice, bool thenPath);
} // namespace mlir::pto::protocol_sync
#endif
