// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve invariant repeated crossings as numerical families before lowering.
#include "RepeatedRegionInternal.h"
#include "OccurrenceBinding.h"
#include "CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
FailureOr<bool> prepareRepeatedCrossingFamilies(RepeatedRegionState& state, PreparedLogicalPlan& plan,
    std::vector<std::pair<uint32_t, std::size_t>>& allocation, uint64_t& nextRecord)
{
    // The numerical family's ordinal and finite bounds must be those of the
    // original loop. Quotient phases and clipped invocations keep their exact
    // guarded producer until they export the corresponding family maps.
    if (state.phaseCount != 1 || !state.typePhases.empty() ||
        state.originalBegin != RegionExpressions::invalid) { return false; }
    auto counted = CountedLoop::get(state.loop);
    if (!counted || counted->trips(state.e()) != state.trips) { return false; }
    auto& body = state.body;
    FixedPointActivity proof(state.e());
    std::size_t count = 0;
    for (const auto& crossing : state.crossings) {
        if (crossing.native || state.e().constantValue(crossing.guard) == 0) { continue; }
        if (!validRegionalEvent(body, crossing.source) || !validRegionalEvent(body, crossing.target) ||
            crossing.source.type >= body.anchors.size() || crossing.target.type >= body.anchors.size()) {
            return false;
        }
        const auto& a = body.anchors[crossing.source.type];
        const auto& b = body.anchors[crossing.target.type];
        if (!a.phase || !b.phase ||
            !proof.prove(body, crossing.source, a, a.after.before, crossing.guard) ||
            !proof.prove(body, crossing.target, b, b.before.before, crossing.guard)) { return false; }
        ++count;
    }
    if (!count) { return true; }
    if (nextRecord >= UINT32_MAX || count > UINT32_MAX - nextRecord) { return failure(); }
    if (count > UINT32_MAX / 2) { return false; }
    NumericTemplateEndpoints endpoints;
    endpoints.outer = state.loop;
    endpoints.anchors = body.anchors;
    std::map<Operation*, std::size_t> cuts;
    auto append = [&](uint32_t record, const RepeatedCrossing& crossing, uint32_t pipe, EndpointKind kind) {
        const auto& anchor = body.anchors[kind == EndpointKind::Set ? crossing.source.type : crossing.target.type];
        const auto cut = kind == EndpointKind::Set ? anchor.after : anchor.before;
        auto [entry, added] = cuts.emplace(cut.before, endpoints.groups.size());
        if (added) { endpoints.groups.push_back({cut, {}}); }
        endpoints.groups[entry->second].recipes.push_back(endpoints.logical.recipes.size());
        endpoints.logical.recipes.push_back({record, crossing.source.type, crossing.target.type,
                                            pipe, crossing.displacement, kind});
    };
    std::vector<EndpointFamily> families;
    std::vector<std::pair<uint32_t, std::size_t>> crossings;
    auto record = static_cast<uint32_t>(nextRecord);
    for (const auto& crossing : state.crossings) {
        if (crossing.native || state.e().constantValue(crossing.guard) == 0) { continue; }
        const auto& a = body.anchors[crossing.source.type];
        const auto& b = body.anchors[crossing.target.type];
        const auto p = static_cast<uint32_t>(a.phase->kPipeValue), q = static_cast<uint32_t>(b.phase->kPipeValue);
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.displacement = crossing.displacement;
        family.sourceCut = a.after; family.targetCut = b.before;
        family.members.push_back({record, crossing.source.type, crossing.target.type, a.coordinates, b.coordinates});
        families.push_back(std::move(family));
        crossings.emplace_back(record, &crossing - state.crossings.data());
        if (p != q) { append(record, crossing, p, EndpointKind::Set); }
        append(record, crossing, q, p == q ? EndpointKind::Barrier : EndpointKind::Wait);
        ++record;
    }
    auto recipe = buildFamilyEndpointRecipe(endpoints);
    if (failed(recipe)) { return false; }
    auto joined = appendFamilyEndpointRecipe(plan.familyRecipe, *recipe, 0, plan.endpoints.size());
    if (failed(joined)) { return failure(); }
    plan.familyRecipe = std::move(*joined);
    llvm::append_range(plan.families, std::move(families));
    llvm::append_range(allocation, std::move(crossings));
    nextRecord = record;
    return true;
}
} // namespace mlir::pto::frontiersynch
