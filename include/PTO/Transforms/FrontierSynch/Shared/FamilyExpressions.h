// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Build exact endpoint-family selectors directly from original loop coordinates.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYEXPRESSIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FAMILYEXPRESSIONS_H
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplate.h"
#include "mlir/IR/Builders.h"
#include <memory>
#include <optional>
#include <string>
namespace mlir::pto::frontiersynch {
struct FamilyExpressions {
    std::string error;
    Value present;
    Value member; // Selected label (default: list ordinal); meaningful only when present.
};
// Immutable, validated coordinate domain and selected-label mapping. It owns
// no generated SSA values and borrows only original induction variables.
class FamilyExpressionRecipe;
enum class FamilyExpressionOutputs { PresenceAndMember, Member };
struct FamilyExpressionPlan {
    std::string error;
    std::shared_ptr<const FamilyExpressionRecipe> recipe;
};
FamilyExpressionPlan buildFamilyExpressions(ArrayRef<SmallVector<TemplateCoordinate>> members,
    ArrayRef<int64_t> labels = {}, uint64_t modulus = 0,
    FamilyExpressionOutputs outputs = FamilyExpressionOutputs::PresenceAndMember);
// Deterministic structural estimate of the live requested outputs. No SSA is
// constructed. This is a bounded implementation heuristic, not a latency model.
struct FamilyExpressionEstimate {
    uint64_t operations = 0, weighted = 0;
    bool cheap = true;
};
std::optional<FamilyExpressionEstimate> estimateFamilyExpressions(const FamilyExpressionRecipe& recipe);
// Attempt-local structural identity; original SSA pointers are opaque key data.
void appendFamilyExpressionKey(const FamilyExpressionRecipe& recipe, SmallVectorImpl<uint64_t>& key);
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location, const FamilyExpressionRecipe& recipe);
// Slice only the already retained exact activity; no selected-label SSA.
Value emitFamilyPresence(OpBuilder& builder, Location location, const FamilyExpressionRecipe& recipe);
// The caller establishes a 64-bit index layout and availability of every loop
// induction value at the insertion point. Each tuple uses the same ordered loop
// schema. Literal lower/step bounds permit lattice normalization; otherwise
// exact raw induction coordinates preserve environment-evaluated nested bounds.
// Invalid inputs return an error without creating IR. Empty input denotes no
// members; a singleton empty tuple denotes one unconditional member. Emitted
// arithmetic is total even outside the exact member domain. Optional labels map
// each coordinate tuple to its original record identity without requiring paired
// endpoints to have the same piece partition. An empty label list means ordinals.
// A nonzero modulus (at most eight) selects labels modulo that modulus and permits
// verified modular affine expressions; zero retains the exact-label mapping.
// Member-only requests require the caller to establish the exact member domain.
// They return no presence Value and build no unused domain guard. Coordinate
// validation and exact label fitting still cover every supplied member.
FamilyExpressions emitFamilyExpressions(OpBuilder& builder, Location location,
    ArrayRef<SmallVector<TemplateCoordinate>> members, ArrayRef<int64_t> labels = {}, uint64_t modulus = 0,
    FamilyExpressionOutputs outputs = FamilyExpressionOutputs::PresenceAndMember);
} // namespace mlir::pto::frontiersynch
#endif
