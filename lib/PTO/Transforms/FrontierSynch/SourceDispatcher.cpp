// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Production selection consumes one typed regional request. Route admission,
// selector qualification and failures share its invocation-local cache.
#include "RegionalRequests.h"
namespace mlir::pto::frontiersynch {
SelectedAnalysisHandle selectAnalysis(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& source,
    SmallVectorImpl<AnalysisAttempt>& attempts, std::string& reason, CostLedger& costs, const AnalysisNeeds& needs)
{
    RegionalRequests requests(function, input, source, costs);
    auto exact = needs;
    exact.allowSoundUpper = false;
    auto requested = requests.request(function, requests.context(), RegionalMode::Modeled, exact,
        RegionalRepresentation::NativeSummaries, RegionalPreparation::SelectorMatching);
    if (!requested.analysis && needs.allowSoundUpper && !needs.suppliedExactEffects) {
        requested = requests.request(function, requests.context(), RegionalMode::Modeled, needs,
            RegionalRepresentation::NativeSummaries, RegionalPreparation::SelectorMatching);
    }
    // Preserve the admitted late precision extension, but never force it
    // before an economical upper result has had a chance to realize matching.
    if (!requested.analysis) {
        auto priorObligation = requested.obligation;
        requested = requests.request(function, requests.context(), RegionalMode::Modeled, needs,
            RegionalRepresentation::NativeSummaries, RegionalPreparation::SelectorMatching,
            RegionalRoutePolicy::GeneralExtension);
        if (!requested.analysis && !priorObligation.empty()) {
            requested.obligation = std::move(priorObligation);
        }
    }
    llvm::append_range(attempts, requests.attempts());
    if (!requested.analysis) {
        reason = requested.obligation.empty() ? "no registered route established selector matching" :
                                               requested.obligation;
        attempts.push_back(
            {&function.getBody(), "approximate", "unmet-obligation",
             "no registered constructor established the requested upper/query interfaces", 0});
        return {};
    }
    auto selected = std::make_shared<SelectedAnalysis>(*requested.analysis);
    if (selected->kind == SelectedAnalysis::Kind::Explicit) { selected->route = "explicit-retention-sink"; }
    if (selected->kind == SelectedAnalysis::Kind::Periodic && !selected->stationary && !selected->upper) {
        selected->route = "certified-periodic-quotient";
    }
    selected->attempts.assign(attempts.begin(), attempts.end());
    selected->inspectedRegions += requests.regionCount();
    selected->cacheHits += requests.cacheHits();
    if (selected->regionalChildren.empty()) { selected->regionalChildren = std::move(requested.children); }
    reason.clear();
    return selected;
}
} // namespace mlir::pto::frontiersynch
