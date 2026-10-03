// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private lowering for certified invocation-owned event pools. Counters follow
// endpoint execution through existing SCF controls, independently for SET/WAIT.
#ifndef PTO_FRONTIERSYNCH_STRUCTURED_EVENT_COUNTERS_H
#define PTO_FRONTIERSYNCH_STRUCTURED_EVENT_COUNTERS_H
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>
#include <string>
namespace mlir::pto::frontiersynch {
struct StructuredEventPool {
    PIPE source;
    PIPE target;
    SmallVector<unsigned> eligibleIds;
};
LogicalResult preflightStructuredCounters(func::FuncOp pending, std::string& reason);
LogicalResult lowerStructuredCounters(func::FuncOp pending, ArrayRef<StructuredEventPool> pools,
    const DenseMap<Operation*, std::size_t>& endpoints, std::string& reason);
} // namespace mlir::pto::frontiersynch
#endif
