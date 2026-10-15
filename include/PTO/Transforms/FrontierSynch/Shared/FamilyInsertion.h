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
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include <map>
#include <functional>
namespace mlir::pto::frontiersynch {
class FamilyEndpointRecipe;
// Split construction from lowering. Recipes retain exact pieces and validated
// selector maps, but allocate no MLIR operations. Original IR stays immutable.
// The caller certifies cuts, bounds, 64-bit indices and correspondence with the
// plan families. Lowering stages arithmetic in detached blocks and preserves
// original record identities across independently partitioned endpoint sides.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFamilyEndpointRecipe(const NumericTemplateEndpoints& plan);
// Finite point batches use the same family compiler with source ordinal zero.
// No outer loop or displacement is admitted. The producer must prove that at
// each original cut exact activity equals the supplied coordinate point, not
// merely that the coordinates are a necessary domain for that activity.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildFiniteFamilyEndpointRecipe(
    const NumericTemplateEndpoints& plan);
// Residual guarded families retain the producer's exact activity and matching
// expressions. Each piece has one provenance member; numerical batches use the
// grouped builders above. Availability is certified without generated SSA.
struct ContextualEndpointFamily {
    EndpointPiece piece;
    RegionExpressions::Id activity = RegionExpressions::invalid;
    RegionExpressions::Id ordinal = RegionExpressions::invalid;
    RegionExpressions::Id residueCoordinate = RegionExpressions::invalid;
    uint64_t modulus = 0, residue = 0;
};
// One preparation attempt shares successful availability recipes across
// protocol selection and family construction. The arena is append-only and
// original cuts are unchanged; this owner must not survive an arena rollback.
class ContextualFamilyExpressions {
public:
    using SharedLookup = std::function<bool(RegionExpressions::Id, Operation*)>;
    // The optional lookup has one immutable binding environment for this owner.
    // It is used during preparation only; returned recipes retain reached keys.
    explicit ContextualFamilyExpressions(std::shared_ptr<RegionExpressions> arena, SharedLookup shared = {})
        : owner(std::move(arena)), sharedLookup(std::move(shared)) {}
    const std::shared_ptr<RegionExpressions>& arena() const { return owner; }
    // Shared roots must have validated recipes at dominating original cuts.
    // Bind their immutable expression identities before specializing a root.
    FailureOr<std::shared_ptr<const CutExpressionRecipe>> prepare(RegionExpressions::Id expression, Operation* cut,
        ArrayRef<RegionExpressions::Id> shared = {});
    std::pair<uint64_t, uint64_t> sharedBindingWork() const;
private:
    std::shared_ptr<RegionExpressions> owner;
    SharedLookup sharedLookup;
    std::map<Operation*, RegionExpressions::CutEmission> contexts;
    std::map<Operation*, std::map<RegionExpressions::Id, std::shared_ptr<const CutExpressionRecipe>>> recipes;
};
struct ContextualSharedExpression {
    RegionExpressions::Id expression = RegionExpressions::invalid;
    Operation* cut = nullptr;
};
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildContextualFamilyEndpointRecipe(
    std::shared_ptr<RegionExpressions> arena, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<ContextualSharedExpression> shared = {});
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> buildContextualFamilyEndpointRecipe(
    ContextualFamilyExpressions& expressions, ArrayRef<ContextualEndpointFamily> families,
    ArrayRef<ContextualSharedExpression> shared = {});
// Append immutable child recipes without copying their selector inventories.
// endpointIndex counts concrete endpoints preceding the child; recordBase is
// the checked namespace translation applied equally to both matching sides.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> appendFamilyEndpointRecipe(
    std::shared_ptr<const FamilyEndpointRecipe> previous,
    std::shared_ptr<const FamilyEndpointRecipe> child, uint32_t recordBase, std::size_t endpointIndex);
// Prefix one enclosing invocation coordinate onto every nonlocal matching
// tuple. Availability is planned once per original cut, without generated SSA.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> prefixFamilyInvocation(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id visit);
// Retain an exact activity mask through composition. Like invocation binding,
// availability is established once per original cut; barriers are masked too.
FailureOr<std::shared_ptr<const FamilyEndpointRecipe>> maskFamilyActivity(
    std::shared_ptr<const FamilyEndpointRecipe> child, std::shared_ptr<RegionExpressions> arena,
    RegionExpressions::Id activity);
// One namespace per direction/displacement, shared by concrete and delayed
// sequence endpoints. Record provenance and member coordinates stay distinct.
std::map<uint32_t, uint32_t> familyRecordNamespaces(ArrayRef<EndpointFamily> families);
LogicalResult normalizeFamilyEndpointPlan(PreparedLogicalPlan& plan);
// Explicit combined-output request. Decode retained certified assignments and
// compose their maps with the frozen protocol before any family scalar lowering.
LogicalResult planPhysicalFamilyOutput(func::FuncOp function, PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds);
bool validPhysicalFamilyOutput(const PreparedLogicalPlan& plan);
bool validFamilyEndpointProtocol(const PreparedLogicalPlan& plan);
bool validPhysicalFamilyEndpoint(const PreparedLogicalPlan& plan, const PreparedLogicalEndpoint& endpoint);
// Composition creates a fresh protocol owner after applying its transforms.
std::shared_ptr<const FamilyEndpointRecipe> familyRecipeForComposition(
    std::shared_ptr<const FamilyEndpointRecipe> recipe);
// Attempt-local complete protocol key. It owns no callback/SSA state and is
// meaningful only while the retained mathematical/source owners are unchanged.
ArrayRef<uint64_t> normalizedFamilyProtocolKey(const FamilyEndpointRecipe& recipe);
LogicalResult materializeFamilyEndpointCode(const FamilyEndpointRecipe& recipe, PreparedLogicalPlan& prepared);
std::size_t familyEndpointPieceCount(const FamilyEndpointRecipe& recipe);
// Exact matching-tuple sizes for nonlocal records, before SSA. Count the source
// ordinal and enclosing invocation coordinates, excluding the provenance label.
// Derived through the same composition walk as lowering; conflicting shapes fail.
FailureOr<std::map<uint32_t, uint64_t>> familyMatchingArities(const FamilyEndpointRecipe& recipe);
} // namespace mlir::pto::frontiersynch
#endif
