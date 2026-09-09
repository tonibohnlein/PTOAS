// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCPLAN_H
#define PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCPLAN_H
#include "PTO/Transforms/InsertSync/SyncOccurrences.h"
#include "PTO/Transforms/InsertSync/InsertSyncOptions.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::pto::logical_sync {
// Immutable obligations consumed by construction, completion and emitted checks.
// Access pointers belong to the translated input and are valid during the
// synchronous observation callback. The occurrence tuple retains all guards
// and loop invocations; IDs are independent of any selected event or repair.
struct OrderingRequirement {
    enum Kind { RAW, WAR, WAW, AccResource, Retirement } kind;
    unsigned source, target;
    const BaseMemInfo* sourceAccess = nullptr;
    const BaseMemInfo* targetAccess = nullptr;
    Relation occurrences;
    // Retirement targets the original function return, not a physical lane.
    // Its conservative realization is an explicit all-pipeline terminal drain;
    // ordinary destination-lane completion cannot discharge this obligation.
    // Native footprints are currently conservative may-accesses. This does
    // not assert a definite overwrite or an exact last value production.
};
// Outcome describes the constructor that actually ran. Expected failure never
// commits a partial candidate; InternalError is not a fallback permission.
struct ConstructionResult {
    enum Status { Applied, Unsupported, AnalysisLimit, Unproved, AllocationFailure, InternalError };
    Status status = Unsupported;
    std::string reason;
    uint64_t work = 0;
    unsigned requirements = 0, handoffs = 0, barriers = 0;
};
ConstructionResult constructLogicalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm, bool useMmad, uint64_t budget = kDefaultLogicalSyncWorkBudget);
namespace testing {
// Exercise the actual pre-emission expansion bound without constructing an
// exponentially large IR. No production option or acceptance override.
bool guardEmissionFits(ArrayRef<unsigned> clauseSizes, bool shortCircuit, uint64_t allowance);
// Differential check of cached, directly complemented native boundary atoms
// against general integer subtraction; no production acceptance override.
bool checkBoundaryConditions(const SyncOccurrences& facts, uint64_t budget);
using RequirementObserver = llvm::function_ref<void(
    const SyncOccurrences&, ArrayRef<const CompoundInstanceElement*>, ArrayRef<OrderingRequirement>)>;
// Native reconstruction challenge only: mutation receives the emitted clone,
// never planner facts or a way to override acceptance. No CLI/environment hook.
ConstructionResult constructWithEmissionMutation(
    func::FuncOp function, InsertSyncGMAliasMode gm, uint64_t budget, llvm::function_ref<void(func::FuncOp)> mutate,
    RequirementObserver observe = {});
} // namespace testing
} // namespace mlir::pto::logical_sync
#endif
