// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Original SSA branch decisions at one original insertion cut.
#ifndef PTO_FRONTIERSYNCH_ORIGINALCUTFACTS_H
#define PTO_FRONTIERSYNCH_ORIGINALCUTFACTS_H
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
namespace mlir::pto::frontiersynch::detail {
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
