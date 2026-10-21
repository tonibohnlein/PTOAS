// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite structured graph over existing invocation-wide may-overlap classes.
#ifndef PTO_FRONTIERSYNCH_PREFIXINPUT_H
#define PTO_FRONTIERSYNCH_PREFIXINPUT_H
#include "PTO/Transforms/FrontierSynch/Shared/SourcePipeServices.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::frontiersynch {
struct PrefixInput {
    std::vector<SourcePipeNode> graph{{std::nullopt, {}}};
    std::vector<SourcePipeSite> sites;
    std::vector<std::size_t> order;
    std::size_t classes = 0;
    double classSeconds = 0, graphSeconds = 0;
    std::string error;
};
// Site numbers refer to this exact instruction inventory. No selector answers,
// refreshes, original footprint exactness or closed child imports are inferred.
PrefixInput buildPrefixInput(func::FuncOp function, ArrayRef<const CompoundInstanceElement*> phases);
} // namespace mlir::pto::frontiersynch
#endif
