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
#include "OriginalCutFacts.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
namespace mlir::pto::frontiersynch {
namespace {
class FixedPointActivity {
public:
    explicit FixedPointActivity(RegionExpressions& arena) : arena(arena) {}
    bool prove(const RegionalAnalysis& region, const RegionalEvent& event,
               const TemplateEndpointAnchor& anchor, Operation* cut, Expr active)
    {
        if (region.expressions.get() != &arena || event.type >= region.anchors.size() ||
            !validRegionalEvent(region, event) || arena.constantValue(event.ordinal) != 0 ||
            !event.visits.empty() || region.occurrenceLoops[event.type] ||
            (!region.outerLoops.empty() && !region.outerLoops[event.type].empty()) || !cut) { return false; }
        if (anchor.coordinates.size() != region.anchors[event.type].coordinates.size()) { return false; }
        std::set<Operation*> axes;
        for (auto [original, supplied] : llvm::zip(region.anchors[event.type].coordinates, anchor.coordinates)) {
            auto loop = supplied.loop;
            if (!loop || original.loop != loop || original.induction != supplied.induction ||
                !axes.insert(loop.getOperation()).second) { return false; }
        }
        std::string error;
        auto conditions = regionalOccurrenceConstraints(region, event, error);
        if (failed(conditions)) { return false; }
        if (region.endpointSiteGuard) { conditions->push_back(*region.endpointSiteGuard); }
        conditions->push_back(active);
        // Reuse the authoritative original-cut facts, not a second branch or
        // availability analysis. Substitution never emits or replays payloads.
        auto key = std::make_pair(&anchor, cut);
        auto& context = contexts[key];
        if (!context) {
            SmallVector<std::pair<Expr, Expr>> bindings;
            for (const auto& [value, truth] : detail::originalCutFacts(cut)) {
                auto input = arena.input(value);
                if (!arena.constantValue(input)) { bindings.push_back({input, arena.boolean(truth)}); }
            }
            for (auto coordinate : anchor.coordinates) {
                bindings.push_back({arena.input(coordinate.loop.getInductionVar()),
                                    arena.constant(coordinate.induction)});
            }
            context = std::make_unique<RegionExpressions::Substitution>(bindings);
        }
        return llvm::all_of(*conditions, [&](Expr condition) {
            return arena.constantValue(arena.substitute(condition, *context)) == 1;
        });
    }
private:
    RegionExpressions& arena;
    std::map<std::pair<const TemplateEndpointAnchor*, Operation*>,
             std::unique_ptr<RegionExpressions::Substitution>> contexts;
};
} // namespace
FailureOr<bool> SequenceAnalysisState::prepareFiniteCrossings(PreparedLogicalPlan& plan,
    std::vector<RegionalAllocationGroup>& allocation, ArrayRef<uint32_t> typeOffsets, uint32_t& nextRecord)
{
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
        if (!proof.prove(ca.regional, a.event(), ca.anchors[a.type], ca.anchors[a.type].after.before, active) ||
            !proof.prove(cb.regional, b.event(), cb.anchors[b.type], cb.anchors[b.type].before.before, active)) {
            return false;
        }
        ++count;
    }
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
