// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERSYNCH_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERSYNCH_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {
class SyncInput;
namespace frontiersynch {
// Construction entry point. Input is exactly the shared InsertSync translation;
// no OAHS-specific opcode, datatype or phase filter intervenes. This foundation
// does not yet construct synchronization and reports that limitation explicitly.
LogicalResult run(func::FuncOp function, const SyncInput &input);
} // namespace frontiersynch
} // namespace mlir::pto
#endif
