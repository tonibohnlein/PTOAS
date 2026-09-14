// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Caller alias contracts are not completion or GM publication guarantees.
// Qualified root tracing is independent of planning and completion.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCGMALIAS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCGMALIAS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir::pto {

enum class InsertSyncGMAliasMode { MayAlias, DisjointArguments };

// Possible backing origins and range precision are independent. A complete
// result contains every possible origin, including initialization and loop
// backedges; unknown/unsupported paths never certify disjointness.
struct InsertSyncMemoryOrigins {
    llvm::SmallVector<Value, 2> values;
    // At least one supported path has no finite origin description. This is
    // independent of values: a result may retain {A, B} and also be unknown.
    bool hasUnknown = true;
    bool complete = false;
    // False when an unqualified view/selector can shift a local access beyond
    // the backing allocation's recorded interval. Origins remain useful.
    bool preservesRootRange = false;
    uint64_t work = 0;
};
InsertSyncMemoryOrigins traceInsertSyncMemoryOrigins(func::FuncOp function, Value value);
struct InsertSyncGMRange {
    Value root;
    uint64_t lower = 0, upper = 0;
};
// Checked, constant, contiguous byte interval through supported pointer/views.
// Absence loses only range precision, never the independently traced origins.
std::optional<InsertSyncGMRange> traceInsertSyncGMRange(func::FuncOp function, Value value);

struct InsertSyncGMRoots {
    llvm::SmallVector<Value, 2> arguments;
    bool complete = false;
};

FailureOr<InsertSyncGMAliasMode> resolveInsertSyncGMAlias(func::FuncOp function, llvm::StringRef overrideMode);
InsertSyncGMRoots traceInsertSyncGMRoots(func::FuncOp function, Value value);
bool disjointInsertSyncGMRoots(func::FuncOp function, Value first, Value second);

} // namespace mlir::pto
#endif
