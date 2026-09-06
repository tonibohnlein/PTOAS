// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- EventLifetime.h - Once-only event consumption order ------*- C++ -*-===//
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_EVENTLIFETIME_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_EVENTLIFETIME_H

#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::pto::protocol_sync {

/// Ordering of event actions, not completion of physical memory accesses.
/// Inputs must be target-qualified, complete handoffs. Concrete verification
/// reconstructs its inputs from actual instructions, never planner certificates.
class SyncEventConsumptionOrder {
public:
    bool provesConsumedBeforeSet(SyncEventGenerationId first, SyncEventGenerationId second) const;
    bool exceededBudget() const { return budgetExceeded; }

private:
    friend SyncEventConsumptionOrder buildEventConsumptionOrder(llvm::ArrayRef<SyncEventGeneration>);
    llvm::SmallVector<llvm::BitVector, 16> before;
    bool budgetExceeded = false;
};

/// Prove only unguarded, non-recurring handoffs in a function's entry block.
/// Same-lane instruction order and set-to-wait causality are the only edges.
/// Missing anchors, structured control and budget overflow provide no proof;
/// callers must retain conservative interference, not infer event scarcity.
SyncEventConsumptionOrder buildEventConsumptionOrder(llvm::ArrayRef<SyncEventGeneration> generations);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_EVENTLIFETIME_H
