// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Prepare structured logical endpoint families directly from analysis coordinates.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYINSERTION_H
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
class FamilyEndpointRecipe;
// Split construction from lowering. Recipes retain exact pieces and validated
// selector maps, but allocate no MLIR operations. Original IR stays immutable.
// The caller certifies cuts, bounds, 64-bit indices and correspondence with the
// plan families. Lowering stages arithmetic in detached blocks and preserves
// original record identities across independently partitioned endpoint sides.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFamilyEndpointRecipe(const NumericTemplateEndpoints& plan);
void materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared);
std::size_t familyEndpointPieceCount(const FamilyEndpointRecipe& recipe);
} // namespace mlir::pto::frontiersynch
#endif
