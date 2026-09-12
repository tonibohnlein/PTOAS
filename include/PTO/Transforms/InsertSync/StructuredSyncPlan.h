// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCPLAN_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCPLAN_H
#include "PTO/Transforms/InsertSync/SyncConstructionResult.h"
#include "PTO/Transforms/InsertSync/StructuredSyncCore.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir::pto::structured_sync {
// During migration enablePrecision=false selects the general conservative
// composition engine. The default retains the previously qualified periodic
// implementation until overlap recovery passes the frozen quality gates.
// It NEVER calls the Presburger or legacy planner, including on refusal.
logical_sync::ConstructionResult constructStructuredSync(
    func::FuncOp function, InsertSyncGMAliasMode gm,
    HardwareContract hardware = HardwareContract::Conservative,
    bool enablePrecision = true);
namespace testing {
logical_sync::ConstructionResult constructCompositionalSync(
    func::FuncOp function, InsertSyncGMAliasMode gm,
    llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware = HardwareContract::Conservative,
    bool precision = false);
logical_sync::ConstructionResult constructWithEmissionMutation(
    func::FuncOp function, InsertSyncGMAliasMode gm,
    llvm::function_ref<void(func::FuncOp)> mutate,
    HardwareContract hardware = HardwareContract::Conservative);
}
}
#endif
