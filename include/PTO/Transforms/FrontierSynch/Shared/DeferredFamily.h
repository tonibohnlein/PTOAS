// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_DEFERREDFAMILY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_DEFERREDFAMILY_H
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
// Numerical point selectors need only their original loop bindings and finite
// domains. This value contains no generated SSA, callbacks or analysis owners.
struct NumericalFamilyRecipe {
    scf::ForOp outer;
    std::vector<EndpointPiece> pieces;
};
FailureOr<NumericalFamilyRecipe> numericalFamilyRecipe(PreparedLogicalPlan& plan);
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> restoreNumericalFamilyRecipe(
    const NumericalFamilyRecipe& recipe);
// Plain owned recipe inventory. References are to earlier nodes, never callbacks.
// Original cuts/values are rebound by the artifact codec before restoration.
struct DeferredFamilyNode {
    enum class Kind { Leaf, Append, Invocation, Activity };
    Kind kind = Kind::Leaf;
    int64_t previous = -1, child = -1;
    uint32_t recordBase = 0;
    std::shared_ptr<RegionExpressions> arena;
    scf::ForOp outer;
    std::vector<EndpointPiece> pieces;
    std::vector<ContextualEndpointFamily> bindings;
    SmallVector<uint8_t> numerical;
    std::vector<ContextualSharedExpression> shared;
    std::map<Operation*, RegionExpressions::Id> contextCuts;
};
FailureOr<std::vector<DeferredFamilyNode>> deferredFamilyRecipe(PreparedLogicalPlan& plan);
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> restoreDeferredFamilyRecipe(
    ArrayRef<DeferredFamilyNode> nodes);
LogicalResult insertDeferredFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> readDeferredFamilyOutput(func::FuncOp function);
LogicalResult allocateDeferredFamilyOutput(func::FuncOp function, ArrayRef<int64_t> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
