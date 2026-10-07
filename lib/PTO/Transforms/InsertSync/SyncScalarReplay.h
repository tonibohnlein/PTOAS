// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared admissibility predicate for replaying deterministic scalar expressions.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARREPLAY_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARREPLAY_H
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::pto::detail {
// A replay keeps the original integer operations and widths. Purity alone is
// insufficient: an operation such as LLVM freeze is not deterministic on replay.
inline bool canReplayScalar(Operation* operation)
{
    if (!operation || operation->getNumRegions() || operation->getNumSuccessors()) { return false; }
    const auto dialect = operation->getName().getDialectNamespace();
    return (dialect == "arith" || dialect == "index") &&
           isMemoryEffectFree(operation) && isSpeculatable(operation);
}
} // namespace mlir::pto::detail
#endif
