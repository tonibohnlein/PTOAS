// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/StorageLaneAllocation.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Key = std::tuple<uint32_t, uint64_t, uint64_t, uint32_t>;
constexpr uint64_t maximumLabels = 6;
bool proves(RegionExpressions& e, Id premise, Id consequence)
{
    return e.implies(premise, consequence) || e.constantUnder(premise, premise) == 0 ||
           e.constantUnder(premise, consequence) == 1;
}
bool descriptors(llvm::ArrayRef<StorageLaneCell> cells, std::map<uint32_t, StorageLaneCell>& result)
{
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> geometry;
    std::map<uint32_t, std::set<std::pair<uint64_t, uint64_t>>> atoms;
    for (const auto& cell : cells) {
        if (cell.begin >= cell.end || !cell.slots || cell.stride >= cell.slots || cell.offset >= cell.slots ||
            !result.emplace(cell.cell, cell).second) { return false; }
        const auto shape = std::make_pair(cell.slots, cell.stride);
        auto [entry, added] = geometry.emplace(cell.family, shape);
        if (!added && entry->second != shape) { return false; }
        atoms[cell.family].insert({cell.begin, cell.end});
    }
    for (const auto& [family, ranges] : atoms) {
        uint64_t end = 0;
        for (const auto& atom : ranges) {
            if (atom.first < end) { return false; }
            end = atom.second;
        }
    }
    return true;
}
Key key(const StorageLaneCell& cell, uint32_t pipe)
{
    return {cell.family, cell.begin, cell.end, pipe};
}
} // namespace
StorageLaneAllocation buildStorageLaneAllocation(RegionExpressions& e,
    const LifetimeWindowInput& window, const LifetimeWindowAnalysis& analysis,
    llvm::ArrayRef<StorageLaneCell> cells, Id sourceOrdinal, bool allowPartial)
{
    StorageLaneAllocation result;
    auto fail = [&](const char* message) {
        result.error = message; result.lanes.clear(); result.records.clear(); result.budget = 0; return result;
    };
    if (!analysis.error.empty() || sourceOrdinal >= e.size() || e.isBoolean(sourceOrdinal) ||
        !window.sites || window.payloads.empty() || analysis.sourceWitnesses.size() != analysis.sourceDemands.size()) {
        return fail("storage lanes require valid retained-demand provenance and a source ordinal");
    }
    std::map<uint32_t, StorageLaneCell> mapping;
    if (!descriptors(cells, mapping)) { return fail("storage lane physical-cell normalization is unavailable"); }
    std::map<Key, uint64_t> widths, bases;
    for (std::size_t r = 0; r < analysis.sourceDemands.size(); ++r) {
        const auto& edge = analysis.sourceDemands[r];
        if (edge.source >= window.sites || edge.source >= window.payloads.size() ||
            edge.target >= window.payloads.size() || edge.guard >= e.size() || !e.isBoolean(edge.guard)) {
            return fail("storage lane retained endpoint is invalid");
        }
        const auto p = window.payloads[edge.source].pipe, q = window.payloads[edge.target].pipe;
        if (p == q) { continue; }
        auto covered = e.boolean(false);
        std::map<Key, uint64_t> candidateWidths;
        for (const auto& witness : analysis.sourceWitnesses[r]) {
            if (witness.guard >= e.size() || !e.isBoolean(witness.guard)) {
                return fail("storage lane witness guard is invalid");
            }
            if (witness.hazard == StorageHazard::Supplied || proves(e, witness.guard, e.boolean(false))) { continue; }
            auto cell = mapping.find(witness.cell);
            const auto expected = witness.hazard == StorageHazard::WAR ? p : q;
            if (cell == mapping.end() || witness.pipe != expected ||
                (witness.hazard != StorageHazard::RAW && witness.hazard != StorageHazard::WAR &&
                 witness.hazard != StorageHazard::WAW)) {
                return fail("storage lane witness lacks a physical mapping");
            }
            covered = e.lor(covered, witness.guard);
            candidateWidths.emplace(key(cell->second, witness.pipe), cell->second.slots);
        }
        if (!proves(e, edge.guard, covered)) {
            if (allowPartial) { continue; }
            return fail("retained demand lacks complete storage-lane provenance; separate prerequisite proof required");
        }
        if (r > UINT32_MAX) { return fail("storage lane record index is unrepresentable"); }
        result.records.push_back(r);
        widths.insert(candidateWidths.begin(), candidateWidths.end());
    }
    for (const auto& [name, width] : widths) {
        if (width > maximumLabels - result.budget) {
            return fail("storage-lane sufficient palette exceeds six labels; no minimum-capacity claim");
        }
        bases[name] = result.budget; result.budget += width;
    }
    for (std::size_t r = 0; r < analysis.sourceDemands.size(); ++r) {
        auto lane = e.constant(0);
        if (!std::binary_search(result.records.begin(), result.records.end(), r)) {
            result.lanes.push_back(lane); continue;
        }
        auto witnesses = analysis.sourceWitnesses[r];
        std::stable_sort(witnesses.begin(), witnesses.end(), [](const auto& a, const auto& b) {
            return std::tie(a.cell, a.pipe, a.hazard) < std::tie(b.cell, b.pipe, b.hazard);
        });
        for (auto it = witnesses.rbegin(); it != witnesses.rend(); ++it) {
            const auto cell = mapping.find(it->cell);
            if (it->hazard == StorageHazard::Supplied || cell == mapping.end()) { continue; }
            auto base = bases.find(key(cell->second, it->pipe));
            if (base == bases.end()) { continue; }
            const auto& c = cell->second;
            // Reduce before multiplying: the admitted physical palette has at
            // most six slots, so even UINT64_MAX source ordinals cannot wrap.
            auto slot = e.constant(c.offset), residue = e.rem(sourceOrdinal, e.constant(c.slots));
            for (uint64_t step = 0; step < c.stride; ++step) { slot = e.add(slot, residue); }
            slot = e.rem(slot, e.constant(c.slots));
            lane = e.select(it->guard, e.add(e.constant(base->second), slot), lane);
        }
        result.lanes.push_back(lane);
    }
    return result;
}
DictionaryAttr storageLaneAllocationCertificate(func::FuncOp function,
    const LifetimeWindowInput& window, llvm::ArrayRef<GuardedRankEdge> demands,
    const StorageLaneAllocation& allocation, int64_t plan)
{
    if (!function || plan < 0 || !allocation.error.empty() || allocation.budget > maximumLabels ||
        allocation.lanes.size() != demands.size()) { return {}; }
    Builder b(function.getContext());
    SmallVector<int64_t> records, sources, targets;
    std::optional<uint32_t> previous;
    for (auto r : allocation.records) {
        if (r >= demands.size() || (previous && r <= *previous) ||
            demands[r].source >= window.payloads.size() || demands[r].target >= window.payloads.size()) { return {}; }
        previous = r;
        const auto p = window.payloads[demands[r].source].pipe, q = window.payloads[demands[r].target].pipe;
        if (p == q) { return {}; }
        records.push_back(r); sources.push_back(p); targets.push_back(q);
    }
    SmallVector<Attribute> groups;
    if (!records.empty()) {
        if (!allocation.budget) { return {}; }
        std::set<int64_t> hidden;
        function.walk([&](Operation* op) {
            if (auto model = getSyncMacroModel(op)) {
                for (const auto& event : model->hiddenEvents) {
                    hidden.insert(event.eventIds.begin(), event.eventIds.end());
                }
            }
        });
        SmallVector<int64_t> forbidden(hidden.begin(), hidden.end()), zeros(records.size(), 0);
        auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(2)),
            b.getNamedAttr("base", b.getI64IntegerAttr(0)),
            b.getNamedAttr("terms", b.getArrayAttr({
                b.getDenseI64ArrayAttr({1, 1, 0, int64_t(allocation.budget), 1})}))});
        SmallVector<Attribute> rules(records.size(), rule);
        groups.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(sources.front())),
            b.getNamedAttr("target", b.getI64IntegerAttr(targets.front())),
            b.getNamedAttr("sources", b.getDenseI64ArrayAttr(sources)),
            b.getNamedAttr("targets", b.getDenseI64ArrayAttr(targets)),
            b.getNamedAttr("budget", b.getI64IntegerAttr(allocation.budget)),
            b.getNamedAttr("records", b.getDenseI64ArrayAttr(records)),
            b.getNamedAttr("strides", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("phases", b.getDenseI64ArrayAttr(zeros)),
            b.getNamedAttr("tuple_rules", b.getArrayAttr(rules)),
            b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
            b.getNamedAttr("forbidden_ids", b.getDenseI64ArrayAttr(forbidden))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("kind", b.getStringAttr("finite")), b.getNamedAttr("plan", b.getI64IntegerAttr(plan)),
        b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
        b.getNamedAttr("macro_reservations", b.getUnitAttr()), b.getNamedAttr("groups", b.getArrayAttr(groups))});
}
DictionaryAttr combineStorageLaneCertificates(DictionaryAttr storage, DictionaryAttr residual)
{
    if (!storage || !residual || storage.getContext() != residual.getContext() ||
        storage.get("plan") != residual.get("plan") || storage.get("version") != residual.get("version") ||
        storage.get("strategy") != residual.get("strategy") || !storage.get("macro_reservations") ||
        !residual.get("macro_reservations")) { return {}; }
    Builder b(storage.getContext());
    SmallVector<Attribute> groups;
    std::set<int64_t> records;
    uint64_t budget = 0;
    for (auto certificate : {storage, residual}) {
        auto entries = certificate.getAs<ArrayAttr>("groups");
        if (!entries) { return {}; }
        for (auto entry : entries) {
            auto group = dyn_cast<DictionaryAttr>(entry);
            auto width = group ? group.getAs<IntegerAttr>("budget") : IntegerAttr{};
            auto ids = group ? group.getAs<DenseI64ArrayAttr>("records") : DenseI64ArrayAttr{};
            if (!width || !width.getType().isInteger(64) || width.getInt() <= 0 || !ids ||
                uint64_t(width.getInt()) > maximumLabels - budget) { return {}; }
            for (auto record : ids.asArrayRef()) {
                if (record < 0 || !records.insert(record).second) { return {}; }
            }
            SmallVector<int64_t> conflicts;
            for (std::size_t i = 0; i < groups.size(); ++i) { conflicts.push_back(i); }
            NamedAttrList attributes(group);
            attributes.set("conflicts", b.getDenseI64ArrayAttr(conflicts));
            groups.push_back(attributes.getDictionary(storage.getContext()));
            budget += width.getInt();
        }
    }
    NamedAttrList attributes(storage);
    attributes.erase("lane_evidence");
    attributes.set("groups", b.getArrayAttr(groups));
    return attributes.getDictionary(storage.getContext());
}
} // namespace mlir::pto::frontiersynch
