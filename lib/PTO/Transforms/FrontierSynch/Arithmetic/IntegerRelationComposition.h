// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared exact composition primitive; not a new public analysis capability.
#ifndef PTO_FRONTIERSYNCH_INTEGERRELATIONCOMPOSITION_H
#define PTO_FRONTIERSYNCH_INTEGERRELATIONCOMPOSITION_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
namespace mlir::pto::frontiersynch {
// Exactly left.remap(...).intersect(right.remap(...)), normalized once.
// Duplicate coordinates identify variables; fresh coordinates are unconstrained.
FailureOr<IntegerSystem> intersectRemappedIntegerSystems(
    const IntegerSystem& left, const IntegerSystem& right, unsigned newDimensions,
    llvm::ArrayRef<unsigned> leftToNew, llvm::ArrayRef<unsigned> rightToNew);
} // namespace mlir::pto::frontiersynch
#endif
