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
#include "RegionConstruction.h"
namespace mlir::pto::frontiersynch {
template<class Scalar, class Memo>
class FamilyScalarEnvironments {
public:
    template<class Emit> auto at(RegionExpressions& arena, Operation* cut,
        const FamilyEndpointRecipe* environment, Emit&& emit)
    {
        auto& state = environments[{&arena, environment}];
        auto value = [&](RegionExpressions::Id id) {
            auto found = state.values.find(id);
            return found == state.values.end() ? Scalar{} : available(found->second, cut);
        };
        auto input = [&](Value original) {
            auto found = state.inputs.find(original);
            return found == state.inputs.end() ? Scalar{} : available(found->second, cut);
        };
        return emit(state.cuts[cut], state.published, value, input);
    }
    void publish(RegionExpressions& arena, const FamilyEndpointRecipe& environment)
    {
        auto& state = environments[{&arena, &environment}];
        if (state.published) { return; }
        for (const auto& [cut, memo] : state.cuts) {
            for (const auto& value : memo.values) { state.values[value.first].push_back({cut, value.second}); }
            for (const auto& input : memo.inputs) { state.inputs[input.first].push_back({cut, input.second}); }
        }
        state.published = true;
    }
    void bind(RegionExpressions& arena, const FamilyEndpointRecipe& environment,
        Operation* cut, RegionExpressions::Id expression, Scalar value)
    {
        environments[{&arena, &environment}].cuts[cut].values[expression] = value;
    }
    FamilyMaterializationWork work() const
    {
        FamilyMaterializationWork result;
        result.sharedCandidates = sharedCandidates;
        auto add = [](uint64_t& target, uint64_t value) {
            target = value > UINT64_MAX - target ? UINT64_MAX : target + value;
        };
        for (const auto& item : environments) {
            const auto& state = item.second;
            add(result.contexts, state.cuts.size());
            for (const auto& value : state.values) { add(result.sharedPublished, value.second.size()); }
            for (const auto& input : state.inputs) { add(result.sharedPublished, input.second.size()); }
            for (const auto& cut : state.cuts) {
                add(result.sharedRequests, cut.second.sharedRequests);
                add(result.sharedImports, cut.second.sharedImports);
                add(result.localEntries, cut.second.values.size());
                add(result.localEntries, cut.second.inputs.size());
            }
        }
        return result;
    }
private:
    struct LocatedValue { Operation* cut; Scalar value; };
    struct Environment {
        std::map<Operation*, Memo> cuts;
        llvm::DenseMap<RegionExpressions::Id, SmallVector<LocatedValue>> values;
        llvm::DenseMap<Value, SmallVector<LocatedValue>> inputs;
        bool published = false;
    };
    std::map<std::pair<RegionExpressions*, const FamilyEndpointRecipe*>, Environment> environments;
    std::map<Operation*, std::unique_ptr<DominanceInfo>> dominance;
    uint64_t sharedCandidates = 0;
    Scalar available(ArrayRef<LocatedValue> bindings, Operation* cut)
    {
        if (bindings.empty()) { return {}; }
        auto* function = cut->getParentOfType<func::FuncOp>().getOperation();
        auto& info = dominance[function];
        if (!info) { info = std::make_unique<DominanceInfo>(function); }
        for (const auto& binding : bindings) {
            if (sharedCandidates != UINT64_MAX) { ++sharedCandidates; }
            if (binding.cut == cut || info->properlyDominates(binding.cut, cut)) { return binding.value; }
        }
        return {};
    }
};
class FamilyMaterialization : public FamilyScalarEnvironments<Value, RegionExpressions::CutEmission> {
public:
    FamilyMaterialization() = default;
    FamilyMaterialization(const FamilyMaterialization&) = delete;
    FamilyMaterialization& operator=(const FamilyMaterialization&) = delete;
    FailureOr<Value> emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
        OpBuilder& builder, Operation* cut, const FamilyEndpointRecipe* environment = nullptr);
};
class FamilyConstruction : public FamilyScalarEnvironments<detail::ScalarToken, detail::RegionConstructionContext> {
public:
    explicit FamilyConstruction(detail::ScalarConstruction& construction) : construction(construction) {}
    FamilyConstruction(const FamilyConstruction&) = delete;
    FamilyConstruction& operator=(const FamilyConstruction&) = delete;
    FailureOr<detail::ScalarToken> emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
        Operation* cut, const FamilyEndpointRecipe* environment = nullptr);
private:
    detail::ScalarConstruction& construction;
};
} // namespace mlir::pto::frontiersynch
#endif
