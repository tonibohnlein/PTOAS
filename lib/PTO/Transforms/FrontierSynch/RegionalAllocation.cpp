// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compose certified cyclic palettes using required-order lifetime envelopes.
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/Builders.h"
#include "PTO/IR/PTO.h"
#include "llvm/ADT/DenseSet.h"
#include <map>
#include <numeric>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool hasNestedFrame(const RegionalAnalysis& region)
{
    return llvm::any_of(region.outerLoops, [](const auto& loops) { return !loops.empty(); });
}
} // namespace
std::shared_ptr<RegionalAllocationSummary> periodicRegionalAllocation(
    const RegionalAnalysis& region, const PeriodicAnalysis& periodic, RegionExpressions::Id trips)
{
    if (hasNestedFrame(region)) { return {}; }
    auto allocation = buildPeriodicAllocation(periodic);
    if (!allocation.error.empty() || !region.expressions || region.anchors.size() != periodic.payloads.size()) {
        return {};
    }
    auto result = std::make_shared<RegionalAllocationSummary>();
    auto& a = *region.expressions;
    for (const auto& direction : allocation.directions) {
        if (!direction.uniformBudget || !*direction.uniformBudget) { return {}; }
        RegionalAllocationGroup group{direction.sourcePipe, direction.targetPipe, *direction.uniformBudget, {}};
        for (auto [phase, handoff] : llvm::enumerate(direction.handoffs)) {
            auto active = a.lt(a.constant(handoff.displacement), trips);
            group.members.push_back({handoff.record, direction.handoffs.size(), phase,
                {handoff.source, a.constant(0), PeriodicEventKind::Start},
                {handoff.target, a.sub(trips, a.constant(1)), PeriodicEventKind::Completion}, active});
        }
        result->groups.push_back(std::move(group));
    }
    return result;
}
std::shared_ptr<RegionalAllocationSummary> finiteRegionalAllocation(
    const RegionalAnalysis& region, const PreparedLogicalPlan& plan)
{
    // No notification lifetime exists when every emitted family is local.
    // This remains true for arbitrarily nested payload/barrier occurrences.
    if (llvm::all_of(plan.families, [](const EndpointFamily& family) { return family.local; })) {
        return std::make_shared<RegionalAllocationSummary>();
    }
    if (hasNestedFrame(region) || !region.expressions || !region.presence ||
        region.anchors.size() != region.occurrenceLoops.size()) {
        return {};
    }
    using CutPipe = std::pair<Operation*, uint32_t>;
    std::map<CutPipe, uint32_t> starts, finishes;
    for (uint32_t i = 0; i < region.anchors.size(); ++i) {
        const auto& anchor = region.anchors[i];
        if (!anchor.phase || !anchor.before.before || !anchor.after.before ||
            region.occurrenceLoops[i] || !anchor.coordinates.empty() ||
            !starts.emplace(CutPipe{anchor.before.before,
                static_cast<uint32_t>(anchor.phase->kPipeValue)}, i).second ||
            !finishes.emplace(CutPipe{anchor.after.before,
                static_cast<uint32_t>(anchor.phase->kPipeValue)}, i).second) {
            return {};
        }
    }
    auto result = std::make_shared<RegionalAllocationSummary>();
    auto& a = *region.expressions;
    for (const auto& family : plan.families) {
        if (family.local) { continue; }
        auto source = finishes.find({family.sourceCut.before, family.sourcePipe});
        auto target = starts.find({family.targetCut.before, family.targetPipe});
        if (source == finishes.end() || target == starts.end() || family.members.size() != 1) { return {}; }
        RegionalEvent first{source->second, a.constant(0), PeriodicEventKind::Start};
        RegionalEvent last{target->second, a.constant(0), PeriodicEventKind::Completion};
        auto sp = region.presence(first), tp = region.presence(last);
        if (!sp || !tp) { return {}; }
        result->groups.push_back({family.sourcePipe, family.targetPipe, 1,
            {{family.members.front().record, 0, 0, first, last, a.land(*sp, *tp), {}, true}}});
    }
    return result;
}
DictionaryAttr regionalAllocationCertificate(const RegionalAnalysis& region, const PreparedLogicalPlan& plan)
{
    // A supplied summary certifies its own internal reuse. Nested coordinates
    // are supported for its selected span endpoints, without inferring an
    // allocation for arbitrary repeated handoff families.
    if (!plan.regionalAllocation || !region.expressions || !region.capabilities.exactQueries ||
        !region.reachability || region.anchors.empty() ||
        (!region.outerLoops.empty() && region.outerLoops.size() != region.anchors.size())) {
        return {};
    }
    auto& a = *region.expressions;
    Builder b(region.anchors.front().before.before->getContext());
    SmallVector<Attribute> groups;
    using EventKey = std::tuple<uint32_t, RegionExpressions::Id, PeriodicEventKind,
                                std::vector<RegionExpressions::Id>>;
    using Query = std::pair<EventKey, EventKey>;
    std::map<Query, RegionExpressions::Id> queries;
    auto reaches = [&](RegionalEvent source, RegionalEvent target, bool handoff = false)
        -> std::optional<RegionExpressions::Id> {
        if (handoff && source.type < region.anchors.size() && target.type < region.anchors.size() &&
            region.anchors[source.type].phase && region.anchors[target.type].phase &&
            region.anchors[source.type].phase->kPipeValue == region.anchors[target.type].phase->kPipeValue) {
            source.kind = target.kind = PeriodicEventKind::Start;
        }
        auto key = Query{{source.type, source.ordinal, source.kind, source.visits},
                         {target.type, target.ordinal, target.kind, target.visits}};
        auto found = queries.find(key);
        if (found != queries.end()) { return found->second; }
        auto answer = regionalReachability(region, source, target);
        if (answer) { queries.emplace(key, *answer); }
        return answer;
    };
    // A hidden notification can overlap surrounding commands even after the
    // host has returned from the macro call. Its bounds are pipe events, not
    // lexical positions. Include pipes having no memory effects as well.
    using OwnerPipe = std::pair<Operation*, uint32_t>;
    std::map<OwnerPipe, uint32_t> envelopeTypes;
    std::set<Operation*> owners;
    for (uint32_t i = 0; i < region.anchors.size(); ++i) {
        auto* phase = region.anchors[i].phase;
        if (!phase || phase->macroOpInstanceId < 0) { continue; }
        if (region.occurrenceLoops.size() != region.anchors.size() || region.occurrenceLoops[i] ||
            !region.anchors[i].coordinates.empty() ||
            (!region.outerLoops.empty() && !region.outerLoops[i].empty()) ||
            !envelopeTypes.emplace(OwnerPipe{phase->elementOp, static_cast<uint32_t>(phase->kPipeValue)}, i).second) {
            return {}; // Repeated reservations require coordinate-qualified lifetimes.
        }
        owners.insert(phase->elementOp);
    }
    struct Reservation {
        uint32_t source, target;
        SmallVector<unsigned> ids;
        RegionalEvent first, last;
        RegionExpressions::Id active;
    };
    std::vector<Reservation> reservations;
    for (auto* owner : owners) {
        auto model = getSyncMacroModel(owner);
        if (!model) { return {}; }
        for (const auto& hidden : model->hiddenEvents) {
            auto p = static_cast<uint32_t>(hidden.srcPipe), q = static_cast<uint32_t>(hidden.dstPipe);
            auto from = envelopeTypes.find({owner, p}), to = envelopeTypes.find({owner, q});
            if (from == envelopeTypes.end() || to == envelopeTypes.end()) { return {}; }
            RegionalEvent first{from->second, a.constant(0), PeriodicEventKind::Start};
            RegionalEvent last{to->second, a.constant(0), PeriodicEventKind::Completion};
            auto present = regionalPresence(region, first);
            if (!present) { return {}; }
            reservations.push_back({p, q, hidden.eventIds, first, last, *present});
        }
    }
    const auto& palettes = plan.regionalAllocation->groups;
    for (std::size_t i = 0; i < palettes.size(); ++i) {
        const auto& x = palettes[i];
        if (!x.budget || x.budget > INT64_MAX) { return {}; }
        SmallVector<int64_t> conflicts, records, strides, phases;
        std::set<int64_t> forbidden;
        SmallVector<Attribute> tupleRules;
        for (const auto& member : x.members) {
            if (member.stride > INT64_MAX || member.phase > INT64_MAX ||
                !validRegionalEvent(region, member.firstSource) ||
                !validRegionalEvent(region, member.lastTarget) ||
                member.active >= a.size() || !a.isBoolean(member.active) ||
                (member.tupleRule && !validPhysicalTupleRule(*member.tupleRule, x.budget))) { return {}; }
            records.push_back(member.record); strides.push_back(member.stride); phases.push_back(member.phase);
            if (!member.tupleRule) {
                tupleRules.push_back(b.getDictionaryAttr({}));
                continue;
            }
            const auto& rule = *member.tupleRule;
            SmallVector<Attribute> terms;
            for (const auto& term : rule.terms) {
                terms.push_back(b.getDenseI64ArrayAttr({static_cast<int64_t>(term.coordinate),
                    static_cast<int64_t>(term.stride), static_cast<int64_t>(term.phase),
                    static_cast<int64_t>(term.modulus), static_cast<int64_t>(term.scale)}));
            }
            tupleRules.push_back(b.getDictionaryAttr({
                b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(rule.coordinateCount)),
                b.getNamedAttr("base", b.getI64IntegerAttr(rule.base)),
                b.getNamedAttr("terms", b.getArrayAttr(terms))}));
        }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& y = palettes[j];
            bool compatible = true;
            for (const auto& xm : x.members) {
                for (const auto& ym : y.members) {
                    auto active = a.land(xm.active, ym.active);
                    if (a.constantValue(active) == 0) { continue; }
                    auto xy = reaches(xm.lastTarget, ym.firstSource, true);
                    auto yx = reaches(ym.lastTarget, xm.firstSource, true);
                    if (!xy || !yx) { return {}; }
                    if (!a.implies(active, a.lor(*xy, *yx))) { compatible = false; break; }
                }
                if (!compatible) { break; }
            }
            if (!compatible) { conflicts.push_back(j); }
        }
        for (const auto& hidden : reservations) {
            bool disjoint = true;
            for (const auto& member : x.members) {
                auto active = a.land(member.active, hidden.active);
                if (a.constantValue(active) == 0) { continue; }
                auto before = reaches(member.lastTarget, hidden.first);
                auto after = reaches(hidden.last, member.firstSource);
                if (!before || !after) { return {}; }
                if (!a.implies(active, a.lor(*before, *after))) { disjoint = false; break; }
            }
            if (!disjoint) { forbidden.insert(hidden.ids.begin(), hidden.ids.end()); }
        }
        SmallVector<int64_t> forbiddenIds(forbidden.begin(), forbidden.end());
        groups.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(x.sourcePipe)),
            b.getNamedAttr("target", b.getI64IntegerAttr(x.targetPipe)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(x.budget)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(strides)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(phases)),
            b.getNamedAttr("tuple_rules", b.getArrayAttr(tupleRules)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr(conflicts)),
            b.getNamedAttr("forbidden_ids", b.getDenseI64ArrayAttr(forbiddenIds))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("kind", b.getStringAttr("finite")),
        b.getNamedAttr("plan", b.getI64IntegerAttr(plan.planId)),
        b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
        b.getNamedAttr("macro_reservations", b.getUnitAttr()),
        b.getNamedAttr("groups", b.getArrayAttr(groups))});
}
namespace {
struct Palette {
    uint32_t source = 0, target = 0;
    uint64_t budget = 0;
    DenseI64ArrayAttr records, strides, phases, conflicts;
    SmallVector<int64_t> ids;
    std::vector<std::optional<PhysicalTupleRule>> tupleRules;
    SmallVector<int64_t> forbidden;
};
std::optional<int64_t> integer(DictionaryAttr attr, StringRef key)
{
    auto value = attr.getAs<IntegerAttr>(key);
    if (!value || !value.getType().isInteger(64)) { return std::nullopt; }
    return value.getInt();
}
bool decodeTupleRule(Attribute attr, uint64_t budget, std::optional<PhysicalTupleRule>& result)
{
    auto entry = dyn_cast<DictionaryAttr>(attr);
    if (!entry) { return false; }
    if (entry.empty()) { return true; }
    auto count = integer(entry, "coordinate_count"), base = integer(entry, "base");
    auto terms = entry.getAs<ArrayAttr>("terms");
    if (!count || *count <= 0 || !base || *base < 0 || !terms) { return false; }
    PhysicalTupleRule rule{static_cast<uint64_t>(*count), static_cast<uint64_t>(*base), {}};
    for (auto value : terms) {
        auto term = dyn_cast<DenseI64ArrayAttr>(value);
        if (!term || term.size() != 5 ||
            llvm::any_of(term.asArrayRef(), [](int64_t item) { return item < 0; })) { return false; }
        rule.terms.push_back({static_cast<uint64_t>(term[0]), static_cast<uint64_t>(term[1]),
            static_cast<uint64_t>(term[2]), static_cast<uint64_t>(term[3]), static_cast<uint64_t>(term[4])});
    }
    if (!validPhysicalTupleRule(rule, budget)) { return false; }
    result = std::move(rule);
    return true;
}
} // namespace
FailureOr<PhysicalAllocationPlan> decodeRegionalAllocation(func::FuncOp function,
    DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    auto version = integer(certificate, "version"), plan = integer(certificate, "plan");
    auto groups = certificate.getAs<ArrayAttr>("groups");
    auto strategy = certificate.getAs<StringAttr>("strategy");
    if (!version || *version != 2 || !plan || *plan < 0 || !groups ||
        !strategy || strategy.getValue() != "regional-palettes") {
        return function.emitError("malformed regional allocation certificate"), failure();
    }
    SmallVector<Palette> palettes;
    llvm::DenseSet<int64_t> recordsSeen;
    auto validPipe = [](int64_t pipe) {
        return pipe >= 0 && pipe <= static_cast<int64_t>(PIPE::PIPE_FIX) &&
            pipe != static_cast<int64_t>(PIPE::PIPE_ALL);
    };
    for (auto entry : groups) {
        auto attr = dyn_cast<DictionaryAttr>(entry);
        if (!attr) { return function.emitError("malformed allocation palette"), failure(); }
        auto source = integer(attr, "source"), target = integer(attr, "target"), budget = integer(attr, "budget");
        auto records = attr.getAs<DenseI64ArrayAttr>("records"), strides = attr.getAs<DenseI64ArrayAttr>("strides");
        auto phases = attr.getAs<DenseI64ArrayAttr>("phases"), conflicts = attr.getAs<DenseI64ArrayAttr>("conflicts");
        if (!source || !target || !budget || !validPipe(*source) || !validPipe(*target) ||
            *source == *target || *budget <= 0 || !records || records.empty() || !strides || !phases ||
            !conflicts || strides.size() != records.size() || phases.size() != records.size()) {
            return function.emitError("malformed allocation palette"), failure();
        }
        for (std::size_t i = 0; i < static_cast<std::size_t>(records.size()); ++i) {
            if (records[i] < 0 || strides[i] < 0 || phases[i] < 0 || !recordsSeen.insert(records[i]).second) {
                return function.emitError("invalid or duplicate regional allocation record"), failure();
            }
        }
        std::vector<std::optional<PhysicalTupleRule>> tupleRules(records.size());
        if (auto raw = attr.get("tuple_rules")) {
            auto rules = dyn_cast<ArrayAttr>(raw);
            if (!rules || rules.size() != static_cast<std::size_t>(records.size())) {
                return function.emitError("malformed regional tuple-rule list"), failure();
            }
            for (std::size_t i = 0; i < rules.size(); ++i) {
                if (!decodeTupleRule(rules[i], static_cast<uint64_t>(*budget), tupleRules[i])) {
                    return function.emitError("invalid or out-of-palette regional tuple rule"), failure();
                }
            }
        }
        int64_t previous = -1;
        for (auto conflict : conflicts.asArrayRef()) {
            if (conflict <= previous || conflict < 0 || static_cast<uint64_t>(conflict) >= palettes.size()) {
                return function.emitError("invalid regional allocation conflict index"), failure();
            }
            previous = conflict;
        }
        SmallVector<int64_t> forbiddenIds;
        if (auto raw = attr.get("forbidden_ids")) {
            auto values = dyn_cast<DenseI64ArrayAttr>(raw);
            if (!values || llvm::any_of(values.asArrayRef(), [](int64_t id) {
                    return id < 0 || id > UINT32_MAX || !symbolizeEVENT(static_cast<uint32_t>(id));
                })) {
                return function.emitError("invalid hidden event ID reservation"), failure();
            }
            forbiddenIds.append(values.asArrayRef().begin(), values.asArrayRef().end());
        }
        palettes.push_back({static_cast<uint32_t>(*source), static_cast<uint32_t>(*target),
            static_cast<uint64_t>(*budget), records, strides, phases, conflicts, {},
            std::move(tupleRules), std::move(forbiddenIds)});
    }
    // Preserve every child cycle. Coloring only chooses its palette, never adds
    // ordering or changes the logical endpoints. Largest palettes go first.
    SmallVector<std::size_t> order(palettes.size());
    std::iota(order.begin(), order.end(), 0);
    llvm::stable_sort(order, [&](std::size_t x, std::size_t y) { return palettes[x].budget > palettes[y].budget; });
    for (auto i : order) {
        auto& palette = palettes[i];
        std::set<int64_t> unavailable(palette.forbidden.begin(), palette.forbidden.end());
        for (std::size_t j = 0; j < palettes.size(); ++j) {
            if (j == i || palettes[j].ids.empty()) { continue; }
            const auto& later = palettes[std::max(i, j)];
            if (llvm::is_contained(later.conflicts.asArrayRef(), static_cast<int64_t>(std::min(i, j)))) {
                unavailable.insert(palettes[j].ids.begin(), palettes[j].ids.end());
            }
        }
        for (auto id : eligibleIds) {
            if (!unavailable.count(id) && palette.ids.size() < palette.budget) { palette.ids.push_back(id); }
        }
        if (palette.ids.size() != palette.budget) {
            return function.emitError("regional allocation not certified within supplied capacity: direction ")
                << palette.source << " -> " << palette.target << "; sufficient palette needs " << palette.budget
                << " IDs; no minimum-capacity claim; scarcity repair not implemented yet", failure();
        }
    }
    PhysicalAllocationPlan result;
    result.planId = *plan;
    for (const auto& palette : palettes) {
        for (std::size_t i = 0; i < static_cast<std::size_t>(palette.records.size()); ++i) {
            PhysicalRecordAllocation record{palette.records[i], palette.source, palette.target,
                static_cast<uint64_t>(palette.strides[i]), static_cast<uint64_t>(palette.phases[i]), {}};
            record.ids.append(palette.ids.begin(), palette.ids.end());
            record.tupleRule = palette.tupleRules[i];
            result.records.push_back(std::move(record));
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
