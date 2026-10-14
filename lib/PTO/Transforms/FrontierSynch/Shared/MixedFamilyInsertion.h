// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Keep qualified numerical pieces alongside residual guarded recipes.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDFAMILYINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_MIXEDFAMILYINSERTION_H
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
// Each numerical entry certifies exact activity at both original cuts: the
// supplied coordinate point plus the counted-loop displacement bounds, or just
// the point with source ordinal zero when outer is absent. Necessary domains
// alone do not authorize numerical qualification. Residual activity is exact.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildMixedFamilyEndpointRecipe(
    ContextualFamilyExpressions& expressions, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<uint8_t> numerical, scf::ForOp outer = {},
    ArrayRef<ContextualSharedExpression> shared = {});
// Diagnostic inventory from the authoritative composition traversal; no
// separately maintained semantic metadata or generated SSA is required.
struct FamilyEndpointForms {
    std::size_t numericalSites = 0, contextualSites = 0, mixedLeaves = 0;
};
FailureOr<FamilyEndpointForms> familyEndpointForms(const FamilyEndpointRecipe& recipe);
} // namespace mlir::pto::frontiersynch
#endif
