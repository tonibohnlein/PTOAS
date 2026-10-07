// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Transactional lowering of certified executed-handoff ranks.
#ifndef PTO_FRONTIERSYNCH_PHYSICALEXECUTEDCOUNTERS_H
#define PTO_FRONTIERSYNCH_PHYSICALEXECUTEDCOUNTERS_H
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
namespace mlir::pto::frontiersynch {
// Closed whole-function arithmetic scope only. Each width>1 family receives
// separate SET/WAIT SSA counters, both zero at invocation entry. They advance
// modulo width only when the corresponding logical command executes. Existing
// scf.for/scf.if results are preserved; unknown region semantics are rejected.
// Family palettes are disjoint in one global pool, excluding all hidden macro
// reservations. Failure leaves the original function unchanged.
LogicalResult allocateExecutedFamilyCounters(func::FuncOp function,
    DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
