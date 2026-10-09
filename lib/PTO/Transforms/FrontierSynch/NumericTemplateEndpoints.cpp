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
FailureOr<std::vector<TemplateEndpointAnchor>> numericTemplateAnchors(
    const NumericTemplate& input, const PeriodicAnalysis& analysis, std::string& error)
{
    std::vector<TemplateEndpointAnchor> anchors;
    const bool certified = input.result.state == RecognitionState::Applicable &&
        input.period == 1 && input.refresh == 1 && input.outer && input.payloads.size() == analysis.payloads.size();
    if (!certified) { error = "numeric endpoint template mismatch"; return failure(); }
    for (std::size_t i = 0; i < input.payloads.size(); ++i) {
        const auto& payload = input.payloads[i];
        if (!payload.phase || static_cast<uint32_t>(payload.phase->kPipeValue) != analysis.payloads[i].pipe) {
            error = "numeric endpoint pipe mismatch"; return failure();
        }
        auto* operation = payload.phase->elementOp;
        if (!operation || !operation->getBlock()) {
            error = "numeric endpoint has no legal operation cut"; return failure();
        }
        anchors.push_back({payload.phase, payload.coordinates,
            {operation->getBlock(), operation}, {operation->getBlock(), operation->getNextNode()}});
    }
    return anchors;
}
NumericTemplateEndpoints buildNumericTemplateEndpoints(const NumericTemplate& input,
                                                       const PeriodicAnalysis& analysis)
{
    std::string error;
    auto anchors = numericTemplateAnchors(input, analysis, error);
    if (failed(anchors)) {
        NumericTemplateEndpoints result; result.logical.error = std::move(error); return result;
    }
    return bindPeriodicEndpoints(input.outer, *anchors, analysis);
}
NumericTemplateEndpoints bindPeriodicEndpoints(scf::ForOp outer,
    ArrayRef<TemplateEndpointAnchor> anchors, const PeriodicAnalysis& analysis)
{
    NumericTemplateEndpoints result;
    if (!outer || anchors.size() != analysis.payloads.size()) {
        result.logical.error = "periodic endpoint anchor count mismatch";
        return result;
    }
    result.logical = buildLogicalEndpoints(analysis);
    if (!result.logical.error.empty()) {
        return result;
    }
    for (auto [i, anchor] : llvm::enumerate(anchors)) {
        if (!anchor.phase || static_cast<uint32_t>(anchor.phase->kPipeValue) != analysis.payloads[i].pipe) {
            result.logical.error = "periodic endpoint pipe mismatch";
            return result;
        }
        result.anchors.push_back(anchor);
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
    result.outer = outer;
    return result;
}
} // namespace mlir::pto::frontiersynch
