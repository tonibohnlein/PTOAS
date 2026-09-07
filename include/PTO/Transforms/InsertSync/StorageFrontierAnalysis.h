// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERANALYSIS_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERANALYSIS_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>
#include <cstdint>

namespace mlir::pto {
struct StorageFrontierRefinementResult {
  unsigned removed = 0;
  unsigned guarded = 0;
  unsigned atoms = 0;
  unsigned requirements = 0;
  unsigned occurrenceProofs = 0;
  uint64_t work = 0;
  bool internalError = false;
  std::string reason;
};
// Uses the existing translated accesses. No new admission gate or speculative
// effect classification. Static events and physical operation order are retained.
// The analysis is optional; unproved/budget outcomes leave the function intact.
StorageFrontierRefinementResult refineInsertSyncStorageFrontiers(
    func::FuncOp function, const SyncIRs &syncIR, ArrayRef<Operation *> candidates,
    bool useMmadChains = false);
} // namespace mlir::pto
#endif
