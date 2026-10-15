// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_REGIONCONSTRUCTION_H
#define PTO_FRONTIER_REGIONCONSTRUCTION_H
#include "ScalarConstruction.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
namespace mlir::pto::frontiersynch::detail {
struct RegionConstructionContext {
    llvm::DenseMap<RegionExpressions::Id, ScalarToken> values;
    llvm::DenseMap<Value, ScalarToken> inputs;
    uint64_t replayBindings = 0, sharedRequests = 0, sharedImports = 0;
};
struct RegionConstructionBindings {
    llvm::function_ref<ScalarToken(RegionExpressions::Id)> values;
    llvm::function_ref<ScalarToken(Value)> inputs;
};
// The context is owned by one original cut and immutable shared environment.
// All contexts exchanging tokens use one comparison-wide ScalarConstruction;
// token serials from different construction owners must never be mixed.
// This service consumes the already qualified CutExpressionRecipe; it neither
// reruns dependence analysis nor creates a different availability contract.
struct RegionConstructionAccess {
    static FailureOr<ScalarToken> construct(RegionExpressions& arena, const CutExpressionRecipe& recipe,
        ScalarConstruction& construction, Operation* cut, RegionConstructionContext& context,
        const RegionConstructionBindings* shared = nullptr);
    template<class Memo> static FailureOr<ScalarToken> schedule(RegionExpressions& arena,
        const CutExpressionRecipe& recipe, ScalarConstruction& construction, Operation* cut,
        Memo& memo, const RegionConstructionBindings* shared);
};
} // namespace mlir::pto::frontiersynch::detail
#endif
