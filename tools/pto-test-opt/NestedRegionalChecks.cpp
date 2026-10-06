// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check hierarchical identities through two sequence exports, without insertion.
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FiniteOverlayInsertion.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
bool runNestedRegionalChecks(func::FuncOp function)
{
    Operation* anchor = nullptr;
    function.walk([&](Operation* op) { if (op->hasAttr("test.nested")) { anchor = op; } });
    if (!anchor) { return false; }
    auto inner = anchor->getParentOfType<scf::ForOp>();
    auto outer = inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!outer) { return false; }
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_V, anchor->getName());
    phase.elementOp = anchor;
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0), one = e.constant(1), two = e.constant(2), three = e.constant(3);
    fs::RegionalAnalysis child;
    child.expressions = arena;
    child.anchors.push_back({&phase, {}, {anchor->getBlock(), anchor}, {anchor->getBlock(), anchor->getNextNode()}});
    child.occurrenceLoops.push_back({});
    child.outerLoops.push_back({outer, inner});
    child.capabilities = {true, true, true, true};
    child.presence = [arena](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
        if (event.type || event.visits.size() != 2) { return std::nullopt; }
        auto& e = *arena;
        return e.land(e.eq(event.ordinal, e.constant(0)),
            e.land(e.lt(event.visits[0], e.constant(3)), e.lt(event.visits[1], e.constant(4))));
    };
    child.referenceBefore = [arena](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        if (a.visits.size() != 2 || b.visits.size() != 2) { return std::nullopt; }
        auto& e = *arena;
        return e.lor(e.lt(a.visits[0], b.visits[0]),
            e.land(e.eq(a.visits[0], b.visits[0]), e.lt(a.visits[1], b.visits[1])));
    };
    child.reachability = [before = child.referenceBefore, present = child.presence, arena]
        (fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b), less = before(a, b);
        if (!pa || !pb || !less) { return std::nullopt; }
        auto& e = *arena;
        auto equal = e.land(e.eq(a.visits[0], b.visits[0]), e.eq(a.visits[1], b.visits[1]));
        auto sameOrder = e.boolean(a.kind == fs::PeriodicEventKind::Start ||
                                  b.kind == fs::PeriodicEventKind::Completion);
        return e.land(e.land(*pa, *pb), e.lor(*less, e.land(equal, sameOrder)));
    };
    child.prepare = []() -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
        return std::make_unique<fs::PreparedLogicalPlan>(0);
    };
    fs::RegionalSelector first{{0, zero, fs::PeriodicEventKind::Start, {zero, zero}}, e.boolean(true)};
    fs::RegionalSelector last{{0, zero, fs::PeriodicEventKind::Start, {two, three}}, e.boolean(true)};
    child.firstPayloads[1].push_back(first); child.lastPayloads[1].push_back(last);
    fs::RegionalStorageBoundary cell;
    cell.cell = {pto::AddressSpace::ACC, 0, 4};
    cell.firstWriters.push_back(first); cell.lastWriters.push_back(last);
    child.storageBoundary.push_back(cell);
    child.prepareFiltered = [prepare = child.prepare](const fs::RegionalDemandFilter&) { return prepare(); };
    auto overlay = fs::analyzeFiniteOverlay(child, {}, child.referenceBefore);
    if (!overlay.error.empty()) { return false; }
    fs::RegionalEvent early{0, zero, fs::PeriodicEventKind::Start, {zero, three}};
    fs::RegionalEvent late{0, zero, fs::PeriodicEventKind::Start, {one, zero}};
    auto forward = fs::regionalReachability(overlay.regional, early, late);
    auto reverse = fs::regionalReachability(overlay.regional, late, early);
    if (!forward || !reverse || e.constantValue(*forward) != 1 || e.constantValue(*reverse) != 0) { return false; }
    if (fs::finiteOverlayRegionalResult(function, overlay).capabilities.endpointRecipes) { return false; }
    std::string overlayError;
    if (succeeded(fs::prepareFiniteOverlayInsertion(function, overlay, overlayError))) { return false; }
    auto flat = child; flat.outerLoops.clear();
    if (fs::regionalPresence(flat, first.event)) { return false; }
    auto malformed = first.event; malformed.visits[0] = e.boolean(true);
    if (fs::regionalPresence(child, malformed)) { return false; }
    auto invalid = child;
    invalid.anchors.push_back(child.anchors.front());
    invalid.occurrenceLoops.push_back({});
    invalid.firstPayloads.clear(); invalid.lastPayloads.clear(); invalid.storageBoundary.clear();
    if (fs::composeRegionalSequence(function, arena, {invalid}).error.empty()) { return false; }
    invalid = child;
    invalid.outerLoops[0] = {inner, outer};
    if (fs::composeRegionalSequence(function, arena, {invalid}).error.empty()) { return false; }
    for (unsigned level = 0; level < 2; ++level) {
        auto composed = fs::composeRegionalSequence(function, arena, {child});
        if (!composed.error.empty()) { return false; }
        child = fs::sequenceRegionalResult(composed);
        if (child.firstPayloads[1][0].event.visits != first.event.visits ||
            child.lastPayloads[1][0].event.visits != last.event.visits ||
            child.storageBoundary[0].lastWriters[0].event.visits != last.event.visits) { return false; }
        fs::RegionalEvent a{0, zero, fs::PeriodicEventKind::Completion, {zero, three}};
        fs::RegionalEvent b{0, zero, fs::PeriodicEventKind::Start, {one, zero}};
        auto forward = fs::regionalReachability(child, a, b), reverse = fs::regionalReachability(child, b, a);
        if (!forward || !reverse || e.constantValue(*forward) != 1 || e.constantValue(*reverse) != 0) { return false; }
        if (succeeded(fs::prepareSequenceInsertion(composed))) { return false; }
    }
    fs::PreparedLogicalPlan plan(0);
    return !fs::finiteRegionalAllocation(child, plan) && e.error().empty();
}
