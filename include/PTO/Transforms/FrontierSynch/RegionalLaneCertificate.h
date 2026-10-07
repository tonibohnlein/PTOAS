// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared allocation of contracted child lanes with guarded boundary evidence.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALLANECERTIFICATE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALLANECERTIFICATE_H
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
DictionaryAttr regionalLaneEvidence(const RegionalAnalysis& region,
    const RegionalAllocationSummary& summary, ArrayAttr groups);
// Returns one physical palette per group, indexed by the original local lane.
FailureOr<SmallVector<SmallVector<int64_t>>> allocateRegionalLanes(func::FuncOp function,
    DictionaryAttr evidence, ArrayAttr groups, ArrayRef<int64_t> eligibleIds);
}
#endif
