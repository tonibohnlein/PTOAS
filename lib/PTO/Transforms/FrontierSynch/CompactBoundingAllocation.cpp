// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactBoundingAllocation.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include <set>
namespace mlir::pto::frontiersynch {
bool attachCompactBoundingAllocation(const RegionalAnalysis& region, const PeriodicAnalysis& selected,
    RegionExpressions::Id trips, PreparedLogicalPlan& plan, std::string& error)
{
    error.clear();
    if (!region.expressions || !selected.error.empty() || trips >= region.expressions->size() ||
        region.expressions->isBoolean(trips) || region.anchors.size() != selected.payloads.size() ||
        !plan.groupedFamilies || !plan.independentPieces || plan.planId < 0) {
        error = "compact allocation requires a paired canonical grouped plan and its selected periodic graph";
        return false;
    }
    std::set<uint32_t> records;
    for (const auto& family : plan.families) {
        for (const auto& member : family.members) {
            if (member.record >= selected.generators.size() || !records.insert(member.record).second) {
                error = "compact allocation family has a duplicate or unknown selected record"; return false;
            }
            const auto& edge = selected.generators[member.record];
            if (member.source != edge.source || member.target != edge.target ||
                family.displacement != edge.displacement || edge.source >= selected.payloads.size() ||
                edge.target >= selected.payloads.size() || family.sourcePipe != selected.payloads[edge.source].pipe ||
                family.targetPipe != selected.payloads[edge.target].pipe ||
                family.local != (family.sourcePipe == family.targetPipe)) {
                error = "compact allocation family differs from its selected graph"; return false;
            }
        }
    }
    if (records.size() != selected.retained.size() ||
        llvm::any_of(selected.retained, [&](uint32_t id) { return !records.count(id); })) {
        error = "compact allocation plan omits a selected record"; return false;
    }
    if (llvm::any_of(plan.endpoints, [](const auto& endpoint) { return !endpoint.before; }) ||
        (records.empty() && !plan.endpoints.empty())) {
        error = "compact allocation endpoints differ from the selected record set"; return false;
    }
    MLIRContext* context = nullptr;
    if (!plan.endpoints.empty()) {
        context = plan.endpoints.front().before->getContext();
    }
    else if (!region.anchors.empty() && region.anchors.front().before.before) {
        context = region.anchors.front().before.before->getContext();
    }
    // A balanced empty plan has no command cuts and may have only abstract
    // anchors. Its original counted invocation still owns the certificate.
    if (!context) {
        for (auto loop : region.occurrenceLoops) {
            if (loop) { context = loop->getContext(); break; }
        }
    }
    if (!context) { error = "compact allocation has no concrete plan context"; return false; }
    auto certificate = encodePeriodicSharedAllocation(selected, plan.planId, context);
    if (!certificate) { error = "compact uniform shared-ID reuse certificate unavailable"; return false; }
    auto summary = periodicRegionalAllocation(region, selected, trips);
    plan.allocationCertificate = certificate;
    plan.regionalAllocation = std::move(summary);
    return true;
}
} // namespace mlir::pto::frontiersynch
