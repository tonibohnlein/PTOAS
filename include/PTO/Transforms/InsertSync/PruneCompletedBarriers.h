// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_PRUNECOMPLETEDBARRIERS_H
#define PTO_TRANSFORMS_INSERTSYNC_PRUNECOMPLETEDBARRIERS_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <string>

namespace mlir::pto {
struct CompletedBarrierPruningResult {
  unsigned removed = 0;
  std::string reason;
};

// Optional post-allocation refinement. Only named barriers already implied by
// source-prefix completion are removed. No events or endpoints are changed.
// Outside the explicitly supported domain, leave the function unchanged.
CompletedBarrierPruningResult pruneProvenCompletedBarriers(func::FuncOp function);
} // namespace mlir::pto
#endif
