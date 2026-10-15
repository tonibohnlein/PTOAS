// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared lookup is indexed by expression/input, then checked at the receiving
// cut. This avoids copying all shared values into every command-cut cache.
#include "FamilyMaterialization.h"
namespace mlir::pto::frontiersynch {
FailureOr<Value> FamilyMaterialization::emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
    OpBuilder& builder, Operation* cut, const FamilyEndpointRecipe* environment)
{
    return at(arena, cut, environment, [&](auto& memo, bool published, auto& value, auto& input) {
        RegionExpressions::SharedCutBindings bindings{value, input};
        return arena.emitPreparedContextual(recipe, builder, cut, memo, published ? &bindings : nullptr);
    });
}
FailureOr<detail::ScalarToken> FamilyConstruction::emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
    Operation* cut, const FamilyEndpointRecipe* environment)
{
    return at(arena, cut, environment, [&](auto& memo, bool published, auto& value, auto& input) {
        detail::RegionConstructionBindings bindings{value, input};
        return detail::RegionConstructionAccess::construct(
            arena, recipe, construction, cut, memo, published ? &bindings : nullptr);
    });
}
} // namespace mlir::pto::frontiersynch
