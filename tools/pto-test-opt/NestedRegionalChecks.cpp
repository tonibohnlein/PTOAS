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
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::optional<uint64_t> guardValue(Value value, Value outer, Value inner, uint64_t i, uint64_t j)
{
    if (value == outer) { return i; }
    if (value == inner) { return j; }
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) { return constant.getZExtValue(); }
    auto* op = value.getDefiningOp();
    if (!op || op->getNumOperands() != 2) { return std::nullopt; }
    auto a = guardValue(op->getOperand(0), outer, inner, i, j);
    auto b = guardValue(op->getOperand(1), outer, inner, i, j);
    if (!a || !b) { return std::nullopt; }
    if (isa<arith::SubIOp>(op)) { return *a - *b; }
    if (isa<arith::DivUIOp>(op) && *b) { return *a / *b; }
    if (isa<arith::RemUIOp>(op) && *b) { return *a % *b; }
    if (isa<arith::AndIOp>(op)) { return *a & *b; }
    if (isa<arith::OrIOp>(op)) { return *a | *b; }
    if (isa<arith::XOrIOp>(op)) { return *a ^ *b; }
    if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
        switch (compare.getPredicate()) {
        case arith::CmpIPredicate::eq: return *a == *b;
        case arith::CmpIPredicate::ult: return *a < *b;
        case arith::CmpIPredicate::ule: return *a <= *b;
        default: return std::nullopt;
        }
    }
    return std::nullopt;
}
} // namespace
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
    if (!fs::finiteOverlayRegionalResult(function, overlay).capabilities.endpointRecipes) { return false; }
    std::string overlayError;
    if (failed(fs::prepareFiniteOverlayInsertion(function, overlay, overlayError))) { return false; }
    // A finite cross-pipe overlay selects one precise occurrence in each nested
    // frame. Its detached guards must depend on both original induction values.
    auto nestedBase = child;
    pto::CompoundInstanceElement otherPhase(1, {}, {}, pto::PipelineType::PIPE_MTE1, anchor->getName());
    otherPhase.elementOp = anchor;
    nestedBase.anchors.push_back(nestedBase.anchors.front());
    nestedBase.anchors.back().phase = &otherPhase;
    nestedBase.occurrenceLoops.push_back({}); nestedBase.outerLoops.push_back({outer, inner});
    nestedBase.outerDivisors = {{1, 2}, {1, 2}};
    nestedBase.endpointEventGuard = [arena, inner](fs::RegionalEvent) mutable
        -> std::optional<fs::RegionExpressions::Id> {
        auto& e = *arena;
        auto visit = e.div(e.sub(e.input(inner.getInductionVar()), e.input(inner.getLowerBound())),
            e.input(inner.getStep()));
        return e.eq(e.rem(visit, e.constant(2)), e.constant(0));
    };
    nestedBase.presence = [present = child.presence](fs::RegionalEvent value) {
        value.type = 0; return present(value);
    };
    nestedBase.reachability = [query = child.reachability, arena](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        if (a.type != b.type) { return arena->boolean(false); }
        a.type = b.type = 0; return query(a, b);
    };
    fs::RegionalEvent addedSource{0, zero, fs::PeriodicEventKind::Completion, {zero, one}};
    fs::RegionalEvent addedTarget{1, zero, fs::PeriodicEventKind::Start, {one, zero}};
    auto nestedOverlay = fs::analyzeFiniteOverlay(nestedBase,
        {{addedSource, addedTarget, e.boolean(true)}}, nestedBase.referenceBefore);
    auto nestedPlan = fs::prepareFiniteOverlayInsertion(function, nestedOverlay, overlayError);
    if (failed(nestedPlan) || (*nestedPlan)->endpoints.size() != 2 || !(*nestedPlan)->regionalAllocation ||
        (*nestedPlan)->regionalAllocation->groups.size() != 1) { return false; }
    for (const auto& endpoint : (*nestedPlan)->endpoints) {
        SmallVector<Value> work{endpoint.guard};
        bool hasOuter = false, hasInner = false;
        llvm::SmallPtrSet<Operation*, 16> seen;
        while (!work.empty()) {
            auto value = work.pop_back_val();
            hasOuter |= value == outer.getInductionVar(); hasInner |= value == inner.getInductionVar();
            auto* op = value.getDefiningOp();
            if (op && seen.insert(op).second) { work.append(op->operand_begin(), op->operand_end()); }
        }
        if (!hasOuter || !hasInner || endpoint.records != SmallVector<uint32_t>{0}) { return false; }
        for (uint64_t i = 0; i < 3; ++i) {
            for (uint64_t j = 0; j < 4; ++j) {
                const bool expected = endpoint.kind == fs::LogicalCommandKind::Set ? i == 0 && j == 2 :
                    i == 1 && j == 0;
                auto actual = guardValue(endpoint.guard, outer.getInductionVar(), inner.getInductionVar(), i, j);
                if (actual != uint64_t(expected)) { return false; }
            }
        }
    }
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
    auto empty = fs::finiteRegionalAllocation(child, plan);
    if (!empty || !empty->groups.empty()) { return false; }
    fs::EndpointFamily local;
    local.local = true;
    plan.families.push_back(local);
    empty = fs::finiteRegionalAllocation(child, plan);
    if (!empty || !empty->groups.empty()) { return false; }
    // An unexported repeated notification still cannot acquire a finite rule.
    plan.families.back().local = false;
    if (fs::finiteRegionalAllocation(child, plan)) { return false; }
    plan.families.clear();

    // Three typed finite spans use the same site and local ordinal but distinct
    // visit coordinates. The third overlaps both earlier spans; a cache that
    // omits visits incorrectly borrows the first pair's reuse proof.
    plan.regionalAllocation = std::make_shared<fs::RegionalAllocationSummary>();
    auto span = [&](uint32_t record, fs::RegionExpressions::Id firstVisit,
                    fs::RegionExpressions::Id lastVisit) {
        fs::RegionalEvent source{0, zero, fs::PeriodicEventKind::Start, {zero, firstVisit}};
        fs::RegionalEvent target{0, zero, fs::PeriodicEventKind::Completion, {zero, lastVisit}};
        plan.regionalAllocation->groups.push_back({1, 2, 1,
            {{record, 0, 0, source, target, e.boolean(true)}}});
    };
    span(0, zero, one);
    span(1, two, three);
    span(2, zero, three);
    plan.regionalAllocation->groups[1].members[0].tupleRule =
        fs::PhysicalTupleRule{3, 0, {{2, 5, 7, 1, 1}}};
    auto certificate = fs::regionalAllocationCertificate(child, plan);
    if (!certificate) { return false; }
    auto groups = certificate.getAs<ArrayAttr>("groups");
    if (!groups || groups.size() != 3) { return false; }
    auto conflicts = [&](std::size_t group) {
        return cast<DictionaryAttr>(groups[group]).getAs<DenseI64ArrayAttr>("conflicts");
    };
    if (!conflicts(0).empty() || !conflicts(1).empty() || conflicts(2).size() != 2 ||
        conflicts(2)[0] != 0 || conflicts(2)[1] != 1) { return false; }
    auto coalesced = fs::coalesceRegionalAllocation(child, *plan.regionalAllocation);
    if (!coalesced || coalesced->groups.size() != 1 || coalesced->groups[0].budget != 2 ||
        coalesced->groups[0].members.size() != 3) { return false; }
    const auto& members = coalesced->groups[0].members;
    if (!members[0].tupleRule || !members[1].tupleRule || !members[2].tupleRule ||
        members[0].tupleRule->base != members[1].tupleRule->base ||
        members[2].tupleRule->base == members[0].tupleRule->base) { return false; }
    for (const auto& member : members) {
        if (!fs::validPhysicalTupleRule(*member.tupleRule, 2)) { return false; }
    }
    auto unequal = *plan.regionalAllocation;
    unequal.groups[0].budget = 2;
    unequal.groups[0].members[0].stride = 3;
    unequal.groups[0].members[0].phase = 1;
    unequal.groups[1].budget = 2;
    unequal.groups[1].members[0].tupleRule = fs::PhysicalTupleRule{3, 0, {{2, 1, 1, 2, 1}}};
    auto colored = fs::coalesceRegionalAllocation(child, unequal);
    if (!colored || colored->groups.size() != 1 || colored->groups[0].budget != 3) { return false; }
    const auto& varied = colored->groups[0].members;
    for (uint64_t ordinal = 0; ordinal < 8; ++ordinal) {
        auto evaluate = [&](const fs::PhysicalTupleRule& rule) {
            uint64_t lane = rule.base;
            for (const auto& term : rule.terms) {
                uint64_t value = term.coordinate == 0 ? ordinal : ordinal + 1;
                lane += term.scale * ((term.stride * value + term.phase) % term.modulus);
            }
            return lane;
        };
        if (evaluate(*varied[0].tupleRule) != (3 * ordinal + 1) % 2 ||
            evaluate(*varied[1].tupleRule) != (ordinal + 2) % 2 ||
            evaluate(*varied[2].tupleRule) != 2 || varied[1].tupleRule->coordinateCount != 3) { return false; }
    }
    auto decoded = fs::decodeRegionalAllocation(function, certificate, {0, 1});
    if (failed(decoded) || decoded->records.size() != 3 || !decoded->records[1].tupleRule) { return false; }
    const auto& rule = *decoded->records[1].tupleRule;
    if (rule.coordinateCount != 3 || rule.base || rule.terms.size() != 1 ||
        rule.terms[0].coordinate != 2 || rule.terms[0].stride != 5 || rule.terms[0].phase != 7 ||
        rule.terms[0].modulus != 1 || rule.terms[0].scale != 1) { return false; }
    fs::PhysicalTupleRule bounded{2, 0, {{0, 1, 0, 2, 1}, {1, 1, 0, 3, 2}}};
    if (!fs::validPhysicalTupleRule(bounded, 6) || fs::validPhysicalTupleRule(bounded, 5)) { return false; }
    bounded.terms[0].scale = INT64_MAX;
    if (fs::validPhysicalTupleRule(bounded, 6)) { return false; }
    plan.regionalAllocation->groups.back().members.front().lastTarget.visits.pop_back();
    return !fs::regionalAllocationCertificate(child, plan) &&
        !fs::coalesceRegionalAllocation(child, *plan.regionalAllocation) && e.error().empty();
}
