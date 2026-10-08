// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Keep record formulas unchanged while matching their certified child lanes.
#include "PTO/Transforms/FrontierSynch/RegionalLaneCertificate.h"
#include "PTO/Transforms/FrontierSynch/SharedHandoffAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalLaneExports.h"
#include <functional>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Lane {
    uint32_t group = 0, local = 0;
    RegionalAllocationLane boundary;
};
bool proves(RegionExpressions& e, Id premise, Id consequence)
{
    return e.implies(premise, consequence) || e.constantUnder(premise, premise) == 0 ||
        e.constantUnder(premise, consequence) == 1;
}
std::optional<int64_t> integer(DictionaryAttr d, StringRef key)
{
    auto attr = d.getAs<IntegerAttr>(key);
    if (!attr || !attr.getType().isInteger(64)) { return std::nullopt; }
    return attr.getInt();
}
}
DictionaryAttr regionalLaneEvidence(const RegionalAnalysis& region,
    const RegionalAllocationSummary& summary, ArrayAttr groups)
{
    if (!region.expressions || groups.size() != summary.groups.size()) { return {}; }
    auto& e = *region.expressions;
    std::vector<Lane> lanes;
    for (uint32_t g = 0; g < summary.groups.size(); ++g) {
        auto group = summary.groups[g];
        if (llvm::any_of(group.members, [&](const auto& member) {
            return regionalAllocationDirection(group, member) != std::make_pair(group.sourcePipe, group.targetPipe);
        })) { return {}; }
        exportConstantRegionalLanes(group);
        // A wider internal cycle cannot fit the six-ID allocation-only path.
        // Do not enumerate arbitrary encoded budgets just to discover that.
        if (!group.budget || group.budget > 6 ||
            (!group.lanes.empty() && group.lanes.size() != group.budget)) { return {}; }
        RegionalAllocationLane envelope;
        if (group.lanes.empty()) {
            for (const auto& m : group.members) {
                envelope.firstSources.push_back({m.firstSource, m.active});
                envelope.lastTargets.push_back({m.lastTarget, m.active});
            }
        }
        for (uint32_t l = 0; l < group.budget; ++l) {
            auto boundary = group.lanes.empty() ? envelope : group.lanes[l];
            for (const auto* selectors : {&boundary.firstSources, &boundary.lastTargets}) {
                for (const auto& s : *selectors) {
                    if (!validRegionalEvent(region, s.event) || s.present >= e.size() || !e.isBoolean(s.present)) {
                        return {};
                    }
                }
            }
            lanes.push_back({g, l, std::move(boundary)});
        }
    }
    if (lanes.size() > UINT32_MAX) { return {}; }
    using Key = std::tuple<uint32_t, Id, std::vector<Id>, uint32_t, Id, std::vector<Id>>;
    std::map<Key, std::optional<Id>> queries;
    auto before = [&](const Lane& a, const Lane& b) {
        for (const auto& x : a.boundary.lastTargets) {
            for (const auto& y : b.boundary.firstSources) {
                auto active = e.land(x.present, y.present);
                if (proves(e, active, e.boolean(false))) { continue; }
                Key key{x.event.type, x.event.ordinal, x.event.visits,
                        y.event.type, y.event.ordinal, y.event.visits};
                auto found = queries.find(key);
                if (found == queries.end()) {
                    found = queries.emplace(std::move(key), regionalHandoffReuse(region, x.event, y.event)).first;
                }
                if (!found->second || !proves(e, active, *found->second)) { return false; }
            }
        }
        return true;
    };
    std::vector<std::vector<bool>> order(lanes.size(), std::vector<bool>(lanes.size()));
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        for (std::size_t j = 0; j < lanes.size(); ++j) {
            const auto& a = summary.groups[lanes[i].group];
            const auto& b = summary.groups[lanes[j].group];
            if (a.sourcePipe != b.sourcePipe || a.targetPipe != b.targetPipe) { continue; }
            if (i != j) { order[i][j] = before(lanes[i], lanes[j]); }
        }
    }
    Builder b(groups.getContext());
    SmallVector<Attribute> rows;
    for (uint32_t i = 0; i < lanes.size(); ++i) {
        SmallVector<int64_t> successors, conflicts;
        for (uint32_t j = 0; j < lanes.size(); ++j) {
            if (order[i][j] && (!order[j][i] || i < j)) { successors.push_back(j); }
            const auto& a = summary.groups[lanes[i].group];
            const auto& b = summary.groups[lanes[j].group];
            if (j < i && a.sourcePipe == b.sourcePipe && a.targetPipe == b.targetPipe &&
                !order[i][j] && !order[j][i]) { conflicts.push_back(j); }
        }
        rows.push_back(b.getDictionaryAttr({
            b.getNamedAttr("group", b.getI64IntegerAttr(lanes[i].group)),
            b.getNamedAttr("local", b.getI64IntegerAttr(lanes[i].local)),
            b.getNamedAttr("successors", b.getDenseI64ArrayAttr(successors)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr(conflicts))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("lanes", b.getArrayAttr(rows))});
}
namespace {
struct DecodedLane {
    uint32_t group = 0, local = 0;
    std::vector<uint32_t> successors, conflicts;
    std::set<int64_t> forbidden;
};
bool indices(DictionaryAttr d, StringRef key, uint64_t bound, std::vector<uint32_t>& values)
{
    auto raw = d.getAs<DenseI64ArrayAttr>(key);
    if (!raw) { return false; }
    int64_t previous = -1;
    for (auto value : raw.asArrayRef()) {
        if (value <= previous || value < 0 || uint64_t(value) >= bound) { return false; }
        values.push_back(value); previous = value;
    }
    return true;
}
// Assign distinct eligible IDs to matched chains, respecting all hidden events
// excluded by any member of that chain. At most six chains can be accepted.
bool placeChains(const std::vector<DecodedLane>& lanes, const SharedHandoffAllocation& assignment,
                 ArrayRef<int64_t> eligible, SmallVector<int64_t>& physical)
{
    if (assignment.budget > eligible.size()) { return false; }
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        for (auto prior : lanes[i].conflicts) {
            if (assignment.lanes[i] == assignment.lanes[prior]) { return false; }
        }
    }
    std::vector<std::set<int64_t>> forbidden(assignment.budget);
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        auto& values = forbidden[assignment.lanes[i]];
        values.insert(lanes[i].forbidden.begin(), lanes[i].forbidden.end());
    }
    SmallVector<int64_t> owner(eligible.size(), -1), matched(assignment.budget, -1);
    std::function<bool(uint32_t, SmallVector<bool>&)> augment = [&](uint32_t chain, SmallVector<bool>& seen) {
        for (uint32_t j = 0; j < eligible.size(); ++j) {
            if (seen[j] || forbidden[chain].count(eligible[j])) { continue; }
            seen[j] = true;
            if (owner[j] < 0 || augment(owner[j], seen)) { owner[j] = chain; matched[chain] = j; return true; }
        }
        return false;
    };
    for (uint32_t i = 0; i < assignment.budget; ++i) {
        SmallVector<bool> seen(eligible.size(), false);
        if (!augment(i, seen)) { return false; }
    }
    for (auto lane : assignment.lanes) { physical.push_back(eligible[matched[lane]]); }
    return true;
}
bool placeDirectedChains(const std::vector<DecodedLane>& lanes, const std::vector<SharedHandoff>& handoffs,
                         ArrayRef<int64_t> eligible, SmallVector<int64_t>& physical)
{
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < handoffs.size(); ++i) {
        groups[{handoffs[i].sourcePipe, handoffs[i].targetPipe}].push_back(i);
    }
    physical.assign(lanes.size(), -1);
    for (const auto& [direction, indices] : groups) {
        std::map<uint32_t, uint32_t> local;
        for (uint32_t i = 0; i < indices.size(); ++i) { local[indices[i]] = i; }
        std::vector<DecodedLane> part;
        std::vector<std::vector<uint32_t>> successors;
        for (auto i : indices) {
            auto lane = lanes[i]; lane.successors.clear(); lane.conflicts.clear();
            for (auto next : lanes[i].successors) {
                if (auto found = local.find(next); found != local.end()) { lane.successors.push_back(found->second); }
            }
            for (auto prior : lanes[i].conflicts) {
                if (auto found = local.find(prior); found != local.end()) { lane.conflicts.push_back(found->second); }
            }
            successors.push_back(lane.successors); part.push_back(std::move(lane));
        }
        std::vector<SharedHandoff> domain(indices.size(), {direction.first, direction.second});
        auto assigned = allocateSharedHandoffs(domain, successors, eligible.size());
        SmallVector<int64_t> ids;
        if (!assigned.error.empty() || !placeChains(part, assigned, eligible, ids)) { return false; }
        for (std::size_t i = 0; i < indices.size(); ++i) { physical[indices[i]] = ids[i]; }
    }
    return true;
}
}
FailureOr<SmallVector<SmallVector<int64_t>>> allocateRegionalLanes(func::FuncOp function,
    DictionaryAttr evidence, ArrayAttr groups, ArrayRef<int64_t> eligible)
{
    std::set<int64_t> ids(eligible.begin(), eligible.end());
    if (eligible.size() > 6 || ids.size() != eligible.size() ||
        llvm::any_of(eligible, [](int64_t id) { return id < 0 || id >= 6; })) {
        return function.emitError("invalid shared event-ID pool"), failure();
    }
    auto raw = evidence.getAs<ArrayAttr>("lanes");
    if (!raw || raw.size() > UINT32_MAX) { return function.emitError("invalid regional lane table"), failure(); }
    SmallVector<SmallVector<int64_t>> result;
    for (auto group : groups) {
        auto d = dyn_cast<DictionaryAttr>(group);
        auto budget = d ? integer(d, "budget") : std::nullopt;
        if (!budget || *budget <= 0 || *budget > 6) {
            return function.emitError("invalid regional lane budget"), failure();
        }
        auto source = integer(d, "source"), target = integer(d, "target");
        auto sources = d.getAs<DenseI64ArrayAttr>("sources"), targets = d.getAs<DenseI64ArrayAttr>("targets");
        if (!source || !target || (sources && llvm::any_of(sources.asArrayRef(),
                [&](int64_t value) { return value != *source; })) ||
            (targets && llvm::any_of(targets.asArrayRef(), [&](int64_t value) { return value != *target; }))) {
            return function.emitError("regional lane group contains mixed directions"), failure();
        }
        result.push_back(SmallVector<int64_t>(*budget, -1));
    }
    std::vector<DecodedLane> lanes;
    std::vector<SharedHandoff> handoffs;
    for (auto entry : raw) {
        auto d = dyn_cast<DictionaryAttr>(entry);
        auto group = d ? integer(d, "group") : std::nullopt, local = d ? integer(d, "local") : std::nullopt;
        if (!group || !local || *group < 0 || uint64_t(*group) >= result.size() || *local < 0 ||
            uint64_t(*local) >= result[*group].size() || result[*group][*local] != -1) {
            return function.emitError("invalid or repeated regional lane identity"), failure();
        }
        DecodedLane lane{uint32_t(*group), uint32_t(*local), {}, {}, {}};
        if (!indices(d, "successors", raw.size(), lane.successors) ||
            !indices(d, "conflicts", lanes.size(), lane.conflicts)) {
            return function.emitError("invalid regional lane relation"), failure();
        }
        auto palette = cast<DictionaryAttr>(groups[*group]);
        auto reserved = palette.getAs<DenseI64ArrayAttr>("forbidden_ids");
        if (reserved) { lane.forbidden.insert(reserved.asArrayRef().begin(), reserved.asArrayRef().end()); }
        auto p = integer(palette, "source"), q = integer(palette, "target");
        if (!p || !q) { return function.emitError("missing regional lane direction"), failure(); }
        handoffs.push_back({uint32_t(*p), uint32_t(*q)});
        result[*group][*local] = lanes.size(); lanes.push_back(std::move(lane));
    }
    for (const auto& group : result) {
        if (llvm::is_contained(group, int64_t(-1))) {
            return function.emitError("incomplete regional lane table"), failure();
        }
    }
    // Pairwise guarded compatibility need not be transitive through an absent
    // middle lane. Matching is used only when its strict-order validation holds.
    SmallVector<int64_t> physical;
    if (!placeDirectedChains(lanes, handoffs, eligible, physical)) {
        physical.clear();
        for (std::size_t i = 0; i < lanes.size(); ++i) {
            const auto& lane = lanes[i];
            std::set<int64_t> used = lane.forbidden;
            for (auto prior : lane.conflicts) {
                if (handoffs[i].sourcePipe == handoffs[prior].sourcePipe &&
                    handoffs[i].targetPipe == handoffs[prior].targetPipe) { used.insert(physical[prior]); }
            }
            auto id = llvm::find_if(eligible, [&](int64_t value) { return !used.count(value); });
            if (id == eligible.end()) {
                return function.emitError("regional lane assignment not certified within directed capacity; "
                    "no minimum-capacity claim; scarcity repair not implemented yet"), failure();
            }
            physical.push_back(*id);
        }
    }
    for (auto& group : result) { for (auto& index : group) { index = physical[index]; } }
    return result;
}
} // namespace mlir::pto::frontiersynch
