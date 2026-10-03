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
#include "PTO/Transforms/FrontierSynch/CostLedger.h"

namespace mlir::pto {
class SyncInput;
namespace frontiersynch {
// Analyze shared demands and optionally dump diagnostic evidence. Emit direct logical endpoints
// when their premises are established. Physical event-ID assignment is separate.
struct DiagnosticOptions {
    bool dumpDemands = false;
    bool reportCosts = false;
    // Explicit restriction of permitted internal repair mechanisms, not a new
    // analysis algorithm or a hardware claim that reverse directions are absent.
    bool finiteOneWayFamily = false;
};
LogicalResult run(func::FuncOp function, const SyncInput& input, DiagnosticOptions options, CostLedger& costs);
void reportImportFailure(func::FuncOp function, const CostLedger& costs, llvm::StringRef reason, bool dumpDemands);
} // namespace frontiersynch
} // namespace mlir::pto
#endif
