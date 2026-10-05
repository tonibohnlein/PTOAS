// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owning MLIR analysis for the frontier synchronization pass. Step 0 storage
// outlives all borrowed structure/recognition links. Default MLIR invalidation
// discards this state after an IR-changing pass.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FRONTIERANALYSIS_H
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
namespace mlir::pto::frontiersynch {
class FrontierAnalysis {
public:
    explicit FrontierAnalysis(Operation* operation) : function(dyn_cast<func::FuncOp>(operation)) {}
    LogicalResult initialize(GMAliasPolicy policy = GMAliasPolicy::MayNotAlias);
    // Build and cache the whole-function arithmetic candidate only on request.
    // Requires successful initialization; uses fixed class limits, not input-derived limits.
    LogicalResult recognizeArithmetic();
    const SyncInput* input() const { return program ? storage.get() : nullptr; }
    const ProgramRecognition* result() const { return program ? &*program : nullptr; }
private:
    func::FuncOp function;
    bool initialized = false;
    GMAliasPolicy policy = GMAliasPolicy::MayNotAlias;
    std::unique_ptr<SyncInput> storage;
    std::optional<ProgramRecognition> program;
};
} // namespace mlir::pto::frontiersynch
#endif
