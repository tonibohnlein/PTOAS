// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Caller alias contracts are not completion or GM publication guarantees.
// Root tracing is adapted from ProtocolSync without its schedule/admission gate.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCGMALIAS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCGMALIAS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::pto {

enum class InsertSyncGMAliasMode { MayAlias, DisjointArguments };

struct InsertSyncGMRoots {
    llvm::SmallVector<Value, 2> arguments;
    bool complete = false;
};

FailureOr<InsertSyncGMAliasMode> resolveInsertSyncGMAlias(func::FuncOp function, llvm::StringRef overrideMode);
InsertSyncGMRoots traceInsertSyncGMRoots(func::FuncOp function, Value value);
bool disjointInsertSyncGMRoots(func::FuncOp function, Value first, Value second);

} // namespace mlir::pto
#endif
