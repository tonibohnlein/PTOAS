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
Value FamilyMaterialization::available(ArrayRef<LocatedValue> bindings, Operation* cut)
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
FailureOr<Value> FamilyMaterialization::emit(RegionExpressions& arena, const CutExpressionRecipe& recipe,
    OpBuilder& builder, Operation* cut, const FamilyEndpointRecipe* environment)
{
    auto& state = environments[{&arena, environment}];
    auto& memo = state.cuts[cut];
    auto value = [&](RegionExpressions::Id id) {
        auto found = state.values.find(id);
        return found == state.values.end() ? Value{} : available(found->second, cut);
    };
    auto input = [&](Value original) {
        auto found = state.inputs.find(original);
        return found == state.inputs.end() ? Value{} : available(found->second, cut);
    };
    RegionExpressions::SharedCutBindings bindings{value, input};
    return arena.emitPreparedContextual(recipe, builder, cut, memo, state.published ? &bindings : nullptr);
}
void FamilyMaterialization::bind(RegionExpressions& arena, const FamilyEndpointRecipe& environment,
    Operation* cut, RegionExpressions::Id expression, Value value)
{
    environments[{&arena, &environment}].cuts[cut].values[expression] = value;
}
void FamilyMaterialization::publish(RegionExpressions& arena, const FamilyEndpointRecipe& environment)
{
    auto& state = environments[{&arena, &environment}];
    if (state.published) { return; }
    for (const auto& [cut, memo] : state.cuts) {
        for (const auto& value : memo.values) { state.values[value.first].push_back({cut, value.second}); }
        for (const auto& input : memo.inputs) { state.inputs[input.first].push_back({cut, input.second}); }
    }
    state.published = true;
}
FamilyMaterializationWork FamilyMaterialization::work() const
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
} // namespace mlir::pto::frontiersynch
