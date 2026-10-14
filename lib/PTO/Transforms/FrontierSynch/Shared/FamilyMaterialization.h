// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One staged lowering owns scalar caches. Shared-root environments are isolated
// by their immutable leaf owner; self-contained leaves and wrappers share only
// within the same arena and original cut. No cache survives staged failure.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYMATERIALIZATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYMATERIALIZATION_H
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
class FamilyMaterialization {
public:
    FamilyMaterialization() = default;
    FamilyMaterialization(const FamilyMaterialization&) = delete;
    FamilyMaterialization& operator=(const FamilyMaterialization&) = delete;
    FailureOr<Value> emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
        OpBuilder& builder, Operation* cut, const FamilyEndpointRecipe* environment = nullptr);
    // Publish only after every shared root in the environment has lowered.
    // Index each binding once; command cuts request only reachable entries.
    void publish(RegionExpressions& arena, const FamilyEndpointRecipe& environment);
    void bind(RegionExpressions& arena, const FamilyEndpointRecipe& environment,
              Operation* cut, RegionExpressions::Id expression, Value value);
    FamilyMaterializationWork work() const;
private:
    struct LocatedValue { Operation* cut; Value value; };
    struct Environment {
        std::map<Operation*, RegionExpressions::CutEmission> cuts;
        llvm::DenseMap<RegionExpressions::Id, SmallVector<LocatedValue>> values;
        llvm::DenseMap<Value, SmallVector<LocatedValue>> inputs;
        bool published = false;
    };
    std::map<std::pair<RegionExpressions*, const FamilyEndpointRecipe*>, Environment> environments;
    std::map<Operation*, std::unique_ptr<DominanceInfo>> dominance;
    uint64_t sharedCandidates = 0;
    Value available(ArrayRef<LocatedValue> bindings, Operation* cut);
};
} // namespace mlir::pto::frontiersynch
#endif
