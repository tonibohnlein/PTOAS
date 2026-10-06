// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed nested identities and flat-backend compatibility checks.
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
bool validRegionalEvent(const RegionalAnalysis& region, const RegionalEvent& event)
{
    if (!region.expressions || (!region.anchors.empty() && event.type >= region.anchors.size()) ||
        region.occurrenceLoops.size() != region.anchors.size() ||
        (event.kind != PeriodicEventKind::Start && event.kind != PeriodicEventKind::Completion)) { return false; }
    const auto& e = *region.expressions;
    auto integer = [&](RegionExpressions::Id id) { return id < e.size() && !e.isBoolean(id); };
    if (!integer(event.ordinal)) { return false; }
    if (region.outerLoops.empty()) { return event.visits.empty(); }
    if (region.outerLoops.size() != region.anchors.size() ||
        region.outerLoops[event.type].size() != event.visits.size()) { return false; }
    return llvm::all_of(event.visits, integer);
}
std::optional<RegionExpressions::Id> regionalPresence(const RegionalAnalysis& region, RegionalEvent event)
{
    if (!validRegionalEvent(region, event) || !region.presence) { return std::nullopt; }
    return region.presence(std::move(event));
}
std::optional<RegionExpressions::Id> regionalReachability(
    const RegionalAnalysis& region, RegionalEvent source, RegionalEvent target)
{
    if (!validRegionalEvent(region, source) || !validRegionalEvent(region, target) || !region.reachability) {
        return std::nullopt;
    }
    return region.reachability(std::move(source), std::move(target));
}
std::optional<RegionExpressions::Id> regionalReferenceBefore(
    const RegionalAnalysis& region, RegionalEvent source, RegionalEvent target)
{
    if (!validRegionalEvent(region, source) || !validRegionalEvent(region, target)) { return std::nullopt; }
    if (region.referenceBefore) { return region.referenceBefore(std::move(source), std::move(target)); }
    if (!source.visits.empty() || !target.visits.empty()) { return std::nullopt; }
    auto& e = *region.expressions;
    return e.lor(e.lt(source.ordinal, target.ordinal),
        e.land(e.eq(source.ordinal, target.ordinal), e.boolean(source.type < target.type)));
}
} // namespace mlir::pto::frontiersynch
