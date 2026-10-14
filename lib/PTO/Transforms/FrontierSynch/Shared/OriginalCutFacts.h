// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Original SSA availability and branch decisions at one insertion cut.
#ifndef PTO_FRONTIERSYNCH_ORIGINALCUTFACTS_H
#define PTO_FRONTIERSYNCH_ORIGINALCUTFACTS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::frontiersynch::detail {
inline bool originalValueAvailable(Value value, Operation* cut, DominanceInfo& dominance)
{
    if (!value || !cut) { return false; }
    auto function = cut->getParentOfType<func::FuncOp>();
    auto* region = value.getParentRegion();
    auto* owner = region ? region->getParentOp() : nullptr;
    return function && owner && (owner == function || function->isProperAncestor(owner)) &&
        dominance.properlyDominates(value, cut);
}
inline llvm::DenseMap<Value, bool> originalCutFacts(Operation* cut)
{
    llvm::DenseMap<Value, bool> facts;
    for (Operation* nested = cut; nested && nested->getParentOp(); nested = nested->getParentOp()) {
        if (auto branch = dyn_cast<scf::IfOp>(nested->getParentOp())) {
            facts[branch.getCondition()] = nested->getParentRegion() == &branch.getThenRegion();
        }
    }
    return facts;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
