// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Record a safe operand-first recipe without changing the original IR.
#include "RecognitionInternal.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::pto::frontiersynch::detail {
bool entryExpression(Value value, Operation* entry, const PhaseIndex& index,
                     SmallVectorImpl<Operation*>& recipe)
{
    SmallVector<std::pair<Value, bool>> work{{value, false}};
    DenseSet<Value> complete;
    SmallVector<Operation*> pending;
    while (!work.empty()) {
        auto [current, finish] = work.pop_back_val();
        if (complete.contains(current)) {
            continue;
        }
        if (index.valueAvailable(current, entry, Boundary::Before)) {
            complete.insert(current);
            continue;
        }
        Operation* op = current.getDefiningOp();
        const bool safe = op && !op->getNumRegions() && index.phasesFor(op).empty() &&
                          isMemoryEffectFree(op) && isSpeculatable(op);
        if (!safe) {
            return false;
        }
        if (!finish) {
            work.push_back({current, true});
            for (Value operand : op->getOperands()) {
                work.push_back({operand, false});
            }
            continue;
        }
        complete.insert(current);
        pending.push_back(op);
    }
    llvm::SmallPtrSet<Operation*, 16> seen(recipe.begin(), recipe.end());
    for (Operation* op : pending) {
        if (seen.insert(op).second) {
            recipe.push_back(op);
        }
    }
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
