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
    if (region.outerLoops.empty()) { return region.outerDivisors.empty() && event.visits.empty(); }
    if (region.outerLoops.size() != region.anchors.size() ||
        region.outerLoops[event.type].size() != event.visits.size()) { return false; }
    if (!region.outerDivisors.empty() && (region.outerDivisors.size() != region.anchors.size() ||
        region.outerDivisors[event.type].size() != event.visits.size() ||
        llvm::any_of(region.outerDivisors[event.type], [](uint64_t value) { return value == 0; }))) { return false; }
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
    // A completion can reach a payload start only at a later reference
    // occurrence. Discharge absent or backward endpoints before asking a
    // compact child to construct its full reachability circuit.
    if (region.capabilities.exactQueries) {
        auto ps = regionalPresence(region, source), pt = regionalPresence(region, target);
        auto& e = *region.expressions;
        if (ps && pt && e.constantValue(e.land(*ps, *pt)) == 0) { return e.boolean(false); }
        if (source.kind == PeriodicEventKind::Completion && target.kind == PeriodicEventKind::Start) {
            auto before = regionalReferenceBefore(region, source, target);
            if (before && e.constantValue(*before) == 0) { return e.boolean(false); }
        }
    }
    // Ordered starts/completions and forward-only modeled demands already
    // decide these native event-kind pairs. C->S still needs the full query.
    if (region.capabilities.exactQueries && source.type < region.anchors.size() &&
        target.type < region.anchors.size() &&
        region.anchors[source.type].phase && region.anchors[target.type].phase &&
        region.anchors[source.type].phase->kPipeValue == region.anchors[target.type].phase->kPipeValue &&
        !(source.kind == PeriodicEventKind::Completion && target.kind == PeriodicEventKind::Start)) {
        auto reverse = regionalReferenceBefore(region, target, source);
        auto ps = regionalPresence(region, source), pt = regionalPresence(region, target);
        if (reverse && ps && pt) {
            // Present occurrences have a total reference order. Non-strict
            // forward order is therefore the absence of strict reverse order;
            // this includes equal occurrences for S->S, C->C and S->C. Keeping
            // it as one comparison also recognizes a first occurrence before
            // any symbolic later ordinal without a separate equality circuit.
            auto& e = *region.expressions;
            return e.land(e.land(*ps, *pt), e.lnot(*reverse));
        }
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
