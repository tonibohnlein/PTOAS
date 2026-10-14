// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Keep finite crossing points as recipes before preparing any scalar endpoint code.
#include "SequenceAnalysisInternal.h"
#include "OccurrenceBinding.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
FailureOr<bool> SequenceAnalysisState::prepareFiniteCrossings(PreparedLogicalPlan& plan,
    std::vector<RegionalAllocationGroup>& allocation, ArrayRef<uint32_t> typeOffsets,
    uint32_t& nextRecord, std::vector<uint8_t>& qualified)
{
    qualified.assign(crossings.size(), 0);
    bool complete = true;
    FixedPointActivity proof(expressions);
    std::size_t count = 0;
    for (const auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) == 0) { continue; }
        const auto& a = ports[edge.source];
        const auto& b = ports[edge.target];
        const auto& ca = children[a.child];
        const auto& cb = children[b.child];
        auto active = edge.guard;
        if (ca.regional.endpointInvocationGuard) { active = both(active, *ca.regional.endpointInvocationGuard); }
        if (cb.regional.endpointInvocationGuard) { active = both(active, *cb.regional.endpointInvocationGuard); }
        const bool numerical =
            proof.prove(ca.regional, a.event(), ca.anchors[a.type], ca.anchors[a.type].after.before, active) &&
            proof.prove(cb.regional, b.event(), cb.anchors[b.type], cb.anchors[b.type].before.before, active);
        qualified[&edge - crossings.data()] = numerical;
        complete &= numerical;
        ++count;
    }
    if (!complete) { return false; }
    if (!count) { return true; }
    if (count > UINT32_MAX - nextRecord) { return failure(); }
    if (count > UINT32_MAX / 2) { return false; }
    // Every endpoint guard contains its full fixed-coordinate equality tests.
    // At that point the remaining exact guard is true under original cut facts.
    // Thus the finite point is exact activity here, not a necessary-domain guess.
    NumericTemplateEndpoints endpoints;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> types;
    std::map<Operation*, std::size_t> cuts;
    auto type = [&](const Port& port) {
        auto [entry, added] = types.emplace(std::make_pair(port.child, port.type), endpoints.anchors.size());
        if (added) { endpoints.anchors.push_back(children[port.child].anchors[port.type]); }
        return entry->second;
    };
    auto append = [&](uint32_t record, uint32_t a, uint32_t b, uint32_t pipe, EndpointKind kind) {
        const auto& anchor = endpoints.anchors[kind == EndpointKind::Set ? a : b];
        const auto cut = kind == EndpointKind::Set ? anchor.after : anchor.before;
        auto [entry, added] = cuts.emplace(cut.before, endpoints.groups.size());
        if (added) { endpoints.groups.push_back({cut, {}}); }
        endpoints.groups[entry->second].recipes.push_back(endpoints.logical.recipes.size());
        endpoints.logical.recipes.push_back({record, a, b, pipe, 0, kind});
    };
    std::vector<EndpointFamily> families;
    std::vector<RegionalAllocationGroup> groups;
    uint32_t record = nextRecord;
    for (const auto& edge : crossings) {
        if (expressions.constantValue(edge.guard) == 0) { continue; }
        const auto& a = ports[edge.source];
        const auto& b = ports[edge.target];
        const auto at = type(a), bt = type(b);
        const auto& aa = endpoints.anchors[at];
        const auto& ab = endpoints.anchors[bt];
        const auto p = static_cast<uint32_t>(aa.phase->kPipeValue);
        const auto q = static_cast<uint32_t>(ab.phase->kPipeValue);
        EndpointFamily family;
        family.id = record;
        family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
        family.sourceCut = aa.after; family.targetCut = ab.before;
        family.members.push_back({record, a.type, b.type, aa.coordinates, ab.coordinates});
        families.push_back(std::move(family));
        if (p != q) {
            append(record, at, bt, p, EndpointKind::Set);
            groups.push_back({p, q, 1, {{record, 0, 0,
                {typeOffsets[a.child] + a.type, a.ordinal, PeriodicEventKind::Start, a.visits},
                {typeOffsets[b.child] + b.type, b.ordinal, PeriodicEventKind::Completion, b.visits},
                edge.guard, {}, true}}, {}, false});
        }
        append(record, at, bt, q, p == q ? EndpointKind::Barrier : EndpointKind::Wait);
        ++record;
    }
    auto recipe = buildFiniteFamilyEndpointRecipe(endpoints);
    // An unavailable compact representation leaves the ordinary realization
    // available; qualification must not create a new export refusal.
    if (failed(recipe)) { return false; }
    auto joined = appendFamilyEndpointRecipe(plan.familyRecipe, *recipe, 0, plan.endpoints.size());
    if (failed(joined)) { return failure(); }
    plan.familyRecipe = std::move(*joined);
    llvm::append_range(plan.families, std::move(families));
    llvm::append_range(allocation, std::move(groups));
    nextRecord = record;
    return true;
}
} // namespace mlir::pto::frontiersynch
