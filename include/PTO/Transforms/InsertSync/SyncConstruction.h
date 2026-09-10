// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCCONSTRUCTION_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCCONSTRUCTION_H
#include "PTO/Transforms/InsertSync/SyncConstructionResult.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::logical_sync {
// Explicit reference engine during migration. The pass dispatcher need not
// include its Presburger-backed requirements to call this historical entry.
ConstructionResult constructLogicalSync(func::FuncOp, InsertSyncGMAliasMode, bool, uint64_t);
}
#endif
