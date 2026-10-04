// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Original phase/coordinate anchors for numeric-template logical recipes.
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
namespace mlir::pto::frontiersynch {
NumericTemplateEndpoints buildNumericTemplateEndpoints(const NumericTemplate& input,
                                                       const PeriodicAnalysis& analysis)
{
    NumericTemplateEndpoints result;
    const bool certified = input.result.state == RecognitionState::Applicable &&
        input.period == 1 && input.refresh == 1 && input.outer && input.payloads.size() == analysis.payloads.size();
    if (!certified) {
        result.logical.error = "numeric endpoint template mismatch";
        return result;
    }
    result.logical = buildLogicalEndpoints(analysis);
    if (!result.logical.error.empty()) {
        return result;
    }
    for (std::size_t i = 0; i < input.payloads.size(); ++i) {
        const auto& payload = input.payloads[i];
        if (!payload.phase || static_cast<uint32_t>(payload.phase->kPipeValue) != analysis.payloads[i].pipe) {
            NumericTemplateEndpoints failure;
            failure.logical.error = "numeric endpoint pipe mismatch";
            return failure;
        }
        auto* operation = payload.phase->elementOp;
        if (!operation || !operation->getBlock()) {
            NumericTemplateEndpoints failure;
            failure.logical.error = "numeric endpoint has no legal operation cut";
            return failure;
        }
        result.anchors.push_back({payload.phase, payload.coordinates,
            {operation->getBlock(), operation}, {operation->getBlock(), operation->getNextNode()}});
    }
    DenseMap<std::pair<Block*, Operation*>, uint32_t> cuts;
    for (uint32_t i = 0; i < result.logical.recipes.size(); ++i) {
        const auto& recipe = result.logical.recipes[i];
        const auto& anchor = result.anchors[recipe.kind == EndpointKind::Set ? recipe.source : recipe.target];
        const auto cut = recipe.kind == EndpointKind::Set ? anchor.after : anchor.before;
        auto [entry, inserted] = cuts.try_emplace({cut.block, cut.before}, result.groups.size());
        if (inserted) {
            result.groups.push_back({cut, {}});
        }
        result.groups[entry->second].recipes.push_back(i);
    }
    result.outer = input.outer;
    return result;
}
} // namespace mlir::pto::frontiersynch
