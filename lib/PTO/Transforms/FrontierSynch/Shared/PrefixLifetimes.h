// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Producer-owned prefix lifetimes; no payload-order repair or guessed reuse.
#ifndef PTO_PREFIX_LIFETIMES_H
#define PTO_PREFIX_LIFETIMES_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::frontiersynch {
// Internal saved-plan contract: logical guards, counters, matching and cuts must
// remain unchanged. Export checks the actual coordinator command inventory.
LogicalResult exportPrefixLifetimes(func::FuncOp function, std::string& error);
// Fresh invocation and qualified already-drained epochs share the certified map
// lowering. Failure is transactional and not a minimum-capacity lower bound.
LogicalResult allocatePrefixLifetimes(func::FuncOp function, ArrayRef<int64_t> eligible);
} // namespace mlir::pto::frontiersynch
#endif
