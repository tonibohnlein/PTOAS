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
    // Defined-input totality for Boolean relation projection. Integer recipes
    // remain outside this optimization's explicitly qualified vocabulary.
    static bool totalScalarDag(const RegionExpressions& arena, ArrayRef<RegionExpressions::Id> roots,
        uint64_t& visited);
    // Trusted internal equivalence producers only; these never change the
    // authoritative query result or authorize different capture/proof inputs.
    static RegionAlternativeWork& alternativeConstructionWork(RegionExpressions& arena);
    static std::vector<std::shared_ptr<const RegionAlternativeWork>> alternativeConstructionOwners(
        const RegionExpressions& arena);
    static void inheritAlternativeConstruction(RegionExpressions& arena, const RegionExpressions& source);
    static bool retainAlternative(RegionExpressions& arena, RegionExpressions::Id original,
        RegionExpressions::Id alternate, uint64_t* work = nullptr);
    static std::optional<RegionExpressions::Id> alternative(
        RegionExpressions& arena, RegionExpressions::Id original, uint64_t* work = nullptr);
    static void transportQualifiedAlternative(RegionExpressions& arena, RegionExpressions::Id original,
        RegionExpressions::Id bound, const RegionExpressions::Substitution& substitution);
    static void transportAlternatives(RegionExpressions& arena,
        const RegionExpressions::Substitution& substitution, ArrayRef<RegionExpressions::Id> visited,
        uint64_t* work);
    static void importAlternatives(RegionExpressions& arena, const RegionExpressions& source,
        ArrayRef<RegionExpressions::Id> visited,
        const llvm::DenseMap<RegionExpressions::Id, RegionExpressions::Id>& imported, uint64_t* work);
    static FailureOr<ScalarToken> construct(RegionExpressions& arena, const CutExpressionRecipe& recipe,
        ScalarConstruction& construction, Operation* cut, RegionConstructionContext& context,
        const RegionConstructionBindings* shared = nullptr);
    template<class Memo> static FailureOr<ScalarToken> schedule(RegionExpressions& arena,
        const CutExpressionRecipe& recipe, ScalarConstruction& construction, Operation* cut,
        Memo& memo, const RegionConstructionBindings* shared);
};
} // namespace mlir::pto::frontiersynch::detail
#endif
