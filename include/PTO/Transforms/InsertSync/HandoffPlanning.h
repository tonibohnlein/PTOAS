// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_HANDOFFPLANNING_H
#define PTO_TRANSFORMS_INSERTSYNC_HANDOFFPLANNING_H
#include "PTO/Transforms/InsertSync/StorageFrontierAnalysis.h"
namespace mlir::pto {
struct HandoffPlanningResult {
    unsigned advanced = 0, delayed = 0, split = 0, unitsRemoved = 0;
    unsigned setsRemoved = 0, waitsRemoved = 0, barriersRemoved = 0, attempts = 0;
    uint64_t work = 0;
    bool internalError = false;
    std::string reason;
};
// Current InsertSync is the feasible seed. Ownership is explicit; authored or
// hidden participants prevent changes to their complete concrete key family.
// Requirements are retained independently of each trial, including GM and
// qualified target resource obligations. Unsupported trials preserve the seed.
HandoffPlanningResult planInsertSyncHandoffs(func::FuncOp function,
    ArrayRef<Operation*> ownedEvents, ArrayRef<Operation*> ownedBarriers,
    bool useMmadChains, insert_sync_frontier::Budget& budget);
}
#endif
