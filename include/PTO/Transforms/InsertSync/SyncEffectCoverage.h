// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCEFFECTCOVERAGE_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCEFFECTCOVERAGE_H
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
// Scalar SSA prerequisites, excluding asynchronous scalar production. Shared
// by effect coverage and the structural descriptor/physical-phase adapter.
bool isInsertSyncScalarPrerequisite(Value value);
// Verify translator completeness, not scheduling correctness. A missing
// summary or discarded memory operand is an explicit unsupported diagnostic.
LogicalResult checkInsertSyncEffectCoverage(func::FuncOp function, const SyncIRs& syncIR);

// A missing model is not a demonstrated code-generation defect. Report mode
// preserves legacy translation and diagnoses the gap; strict mode gates it.
// Explicitly malformed helper contracts fail in either mode. A successful
// return contains true only when the translator coverage check was complete;
// it is not a safety or device-correctness certificate.
FailureOr<bool> inspectInsertSyncEffectCoverage(
    func::FuncOp function, const SyncIRs& syncIR, bool strict);
} // namespace mlir::pto
#endif
