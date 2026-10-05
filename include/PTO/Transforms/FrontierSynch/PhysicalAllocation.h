// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Allocation-only lowering of a certified logical plan. No scarcity repair.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALALLOCATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALALLOCATION_H
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
namespace mlir::pto::frontiersynch {
inline constexpr llvm::StringLiteral CyclicAllocationAttr = "pto.cyclic_allocation";
// Certificate of a producer-proved uniform cyclic assignment. This is an
// internal pass interface, not a proof checker for arbitrary annotated IR.
// Guards, logical matching, occurrence ordinals and payload order must remain
// unchanged between production and consumption of this certificate.
DictionaryAttr encodeCyclicAllocation(const PeriodicAllocation& allocation, int64_t planId, MLIRContext* context);
struct PhysicalRecordAllocation {
    int64_t record = 0;
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    uint64_t stride = 0;
    uint64_t phase = 0;
    SmallVector<int64_t> ids;
};
struct PhysicalAllocationPlan {
    int64_t planId = 0;
    std::vector<PhysicalRecordAllocation> records;
};
FailureOr<PhysicalAllocationPlan> decodeCyclicAllocation(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
// Internal emission helpers. Compaction preserves each executed command and ID;
// the returned pointers identify erased logical endpoints, never dereferenced.
void emitAllocatedCommand(OpBuilder& builder, Operation* endpoint,
                          const PhysicalRecordAllocation& record, Value phaseOffset = {});
SmallVector<Operation*> compactAllocatedEndpoints(func::FuncOp function, const PhysicalAllocationPlan& plan);
// Success lowers all logical commands; failure leaves the function unchanged.
// The caller supplies an eligible subset of 0..5; 6/7 are reserved. Every
// supported direction may use this numeric list:
// (source, destination, numeric ID), not numeric ID alone, identifies an event.
LogicalResult allocatePhysicalEventIds(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
