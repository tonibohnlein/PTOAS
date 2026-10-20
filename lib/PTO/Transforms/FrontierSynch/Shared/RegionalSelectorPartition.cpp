// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalSelectorPartition.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "llvm/ADT/MapVector.h"
#include <algorithm>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool disjointDomain(const RegionalAnalysis& region, AddressSpace space, Value base, RegionalByteAddress address)
{
    if (space != address.space) { return true; }
    if (base == address.base) { return false; }
    SmallVector<SyncStorageCell> domains{{space, 0, 1, base}, {address.space, 0, 1, address.base}};
    return storageBasesAreComparable(domains, region.gmAliasPolicy);
}
bool effectAbsent(const RegionalAnalysis& region, std::size_t id, RegionalByteAddress address, bool wholeDomain = false)
{
    if (!region.accessModel || id >= region.accessModel->effects().size()) { return false; }
    const auto& effect = region.accessModel->effects()[id];
    if (!effect.memory) { return false; }
    if (effect.memory->scope != address.space) { return true; }
    if (effect.rangesMaterialized) {
        auto offset = region.expressions->constantValue(address.offset);
        return llvm::all_of(effect.ranges, [&](const auto& range) {
            if (disjointDomain(region, range.space, range.base, address)) { return true; }
            return !wholeDomain && range.base == address.base && offset &&
                (*offset < range.begin || *offset >= range.end);
        });
    }
    return !effect.regions.empty() && llvm::all_of(effect.regions, [&](const auto& piece) {
        return disjointDomain(region, effect.memory->scope, piece.base, address);
    });
}
bool finiteDomainCovered(const RegionalAnalysis& region, AddressSpace space, Value base)
{
    if (!region.expressions) { return false; }
    const RegionalByteAddress address{space, base, region.expressions->constant(0)};
    for (const auto* side : {&region.accessBoundary, &region.deferredAccessBoundary}) {
        for (const auto& access : *side) {
            if (!access.representedByCells && !effectAbsent(region, access.effect, address, true)) { return false; }
        }
    }
    return llvm::all_of(region.symbolicStorageEffects,
        [&](std::size_t id) { return effectAbsent(region, id, address, true); });
}
} // namespace
void qualifyRegionalCellCoverage(RegionalAnalysis& region)
{
    if (!region.accessModel) { return; }
    using Interval = std::pair<uint64_t, uint64_t>;
    std::map<AddressSpace, llvm::MapVector<Value, std::vector<Interval>>> domains;
    for (const auto& cell : region.storageBoundary) {
        if (cell.cell.begin < cell.cell.end) {
            domains[cell.cell.space][cell.cell.base].push_back({cell.cell.begin, cell.cell.end});
        }
    }
    for (auto& space : domains) {
        for (auto& entry : space.second) {
            auto& ranges = entry.second;
            std::sort(ranges.begin(), ranges.end());
            std::vector<Interval> merged;
            for (const auto& range : ranges) {
                if (!merged.empty() && range.first <= merged.back().second) {
                    merged.back().second = std::max(merged.back().second, range.second);
                } else { merged.push_back(range); }
            }
            ranges = std::move(merged);
        }
    }
    for (auto& access : region.accessBoundary) {
        if (!access.representedByCells || access.effect >= region.accessModel->effects().size()) {
            access.representedByCells = false;
            continue;
        }
        const auto& effect = region.accessModel->effects()[access.effect];
        access.representedByCells = effect.rangesMaterialized && llvm::all_of(effect.ranges, [&](const auto& range) {
            if (range.begin == range.end) { return true; }
            auto space = domains.find(range.space);
            if (space == domains.end()) { return false; }
            auto base = space->second.find(range.base);
            if (base == space->second.end()) { return false; }
            const auto& ranges = base->second;
            auto next = std::upper_bound(ranges.begin(), ranges.end(), range.begin,
                [](uint64_t begin, const Interval& interval) { return begin < interval.first; });
            return next != ranges.begin() && std::prev(next)->second >= range.end;
        });
    }
}

RegionalCellCoverage regionalCellCoverage(const RegionalAnalysis& region, RegionalByteAddress address)
{
    if (!region.expressions || address.offset >= region.expressions->size() ||
        region.expressions->isBoolean(address.offset)) { return RegionalCellCoverage::Unavailable; }
    for (const auto* side : {&region.accessBoundary, &region.deferredAccessBoundary}) {
        for (const auto& access : *side) {
            if (!access.representedByCells && !effectAbsent(region, access.effect, address)) {
                return RegionalCellCoverage::Unavailable;
            }
        }
    }
    for (auto id : region.symbolicStorageEffects) {
        if (!effectAbsent(region, id, address)) { return RegionalCellCoverage::Unavailable; }
    }
    bool domain = false;
    for (const auto& cell : region.storageBoundary) {
        if (disjointDomain(region, cell.cell.space, cell.cell.base, address)) { continue; }
        if (cell.cell.base != address.base) { return RegionalCellCoverage::Unavailable; }
        domain = true;
    }
    return domain ? RegionalCellCoverage::Exact : RegionalCellCoverage::Absent;
}
std::optional<RegionalStorageSelectors> regionalStorageSelectors(
    const RegionalAnalysis& region, RegionalByteAddress address)
{
    if (!region.expressions || address.offset >= region.expressions->size() ||
        region.expressions->isBoolean(address.offset)) { return std::nullopt; }
    if (region.storageSelectors) { return region.storageSelectors(address); }
    const auto coverage = regionalCellCoverage(region, address);
    if (coverage == RegionalCellCoverage::Unavailable) { return std::nullopt; }
    RegionalStorageSelectors out;
    if (coverage == RegionalCellCoverage::Absent) { return out; }
    auto& e = *region.expressions;
    for (const auto& cell : region.storageBoundary) {
        if (cell.cell.space != address.space || cell.cell.base != address.base) { continue; }
        const auto active = e.land(e.le(e.constant(cell.cell.begin), address.offset),
                                  e.lt(address.offset, e.constant(cell.cell.end)));
        auto append = [&](auto& destination, const auto& source) {
            for (auto value : source) {
                value.present = e.land(active, value.present);
                destination.push_back(std::move(value));
            }
        };
        append(out.firstWriters, cell.firstWriters); append(out.lastWriters, cell.lastWriters);
        for (const auto& [pipe, values] : cell.firstReaders) { append(out.firstReaders[pipe], values); }
        for (const auto& [pipe, values] : cell.lastReaders) { append(out.lastReaders[pipe], values); }
    }
    return out;
}
std::optional<std::vector<SyncStorageCell>> selectorSupportRanges(
    const RegionalAnalysis& region, uint64_t& visits)
{
    const auto& certificate = region.symbolicStorage;
    if (certificate && certificate->expressions == region.expressions &&
        certificate->accessModel == region.accessModel && certificate->gmAliasPolicy == region.gmAliasPolicy &&
        certificate->selectorSupport) {
        auto ranges = certificate->selectorSupport(visits);
        if (!ranges) { return std::nullopt; }
        for (const auto& range : *ranges) {
            if (range.begin > range.end || range.end > uint64_t(INT64_MAX)) { return std::nullopt; }
        }
        return ranges;
    }
    if (!region.symbolicStorageEffects.empty() || !region.deferredAccessBoundary.empty() ||
        !region.capabilities.completeStorageModel || !region.capabilities.exactSelectors ||
        llvm::any_of(region.accessBoundary, [](const auto& access) { return !access.representedByCells; })) {
        return std::nullopt;
    }
    std::vector<SyncStorageCell> ranges;
    for (const auto& boundary : region.storageBoundary) {
        accumulateCost(visits, 1);
        ranges.push_back(boundary.cell);
    }
    return ranges;
}
std::optional<std::vector<SyncStorageCell>> selectorDomainSupportRanges(
    const RegionalAnalysis& region, AddressSpace space, Value base, uint64_t& visits)
{
    if (!region.expressions) { return std::nullopt; }
    const auto& certificate = region.symbolicStorage;
    if (certificate && certificate->expressions == region.expressions &&
        certificate->accessModel == region.accessModel && certificate->gmAliasPolicy == region.gmAliasPolicy &&
        certificate->selectorDomainSupport) {
        auto ranges = certificate->selectorDomainSupport(space, base, visits);
        if (!ranges || llvm::any_of(*ranges, [&](const auto& range) {
                return range.space != space || range.base != base || range.begin > range.end ||
                    range.end > uint64_t(INT64_MAX);
            })) { return std::nullopt; }
        return ranges;
    }
    // A complete global envelope is also sufficient for a requested domain.
    if (certificate && certificate->selectorSupport) {
        auto all = selectorSupportRanges(region, visits);
        if (all) {
            std::vector<SyncStorageCell> selected;
            for (const auto& range : *all) {
                if (disjointDomain(region, range.space, range.base, {space, base, 0})) { continue; }
                if (range.base != base) { return std::nullopt; }
                selected.push_back(range);
            }
            return selected;
        }
    }
    if (!finiteDomainCovered(region, space, base)) { return std::nullopt; }
    std::vector<SyncStorageCell> ranges;
    for (const auto& boundary : region.storageBoundary) {
        accumulateCost(visits, 1);
        if (disjointDomain(region, boundary.cell.space, boundary.cell.base, {space, base, 0})) { continue; }
        if (boundary.cell.base != base) { return std::nullopt; }
        ranges.push_back(boundary.cell);
    }
    return ranges;
}
// Consume producer-certified uniform atoms before the per-byte compatibility
// adapter. Validate shape and ownership; semantic uniformity is the producer's
// contract and cannot be established from one selector query.
std::optional<std::vector<uint64_t>> uniformSelectorCuts(
    const RegionalAnalysis& region, SyncStorageCell range, uint64_t& visits)
{
    if (range.begin >= range.end || range.end > uint64_t(INT64_MAX)) { return std::nullopt; }
    const auto& certificate = region.symbolicStorage;
    if (certificate && certificate->expressions == region.expressions &&
        certificate->accessModel == region.accessModel && certificate->gmAliasPolicy == region.gmAliasPolicy &&
        certificate->selectorCuts) {
        auto cuts = certificate->selectorCuts(range, visits);
        if (cuts && cuts->size() >= 2 && cuts->size() - 1 <= maxRegionalSlotVisits &&
            cuts->front() == range.begin && cuts->back() == range.end &&
            std::adjacent_find(cuts->begin(), cuts->end(), std::greater_equal<uint64_t>()) == cuts->end()) {
            return cuts;
        }
    }
    if (!finiteDomainCovered(region, range.space, range.base)) { return std::nullopt; }
    std::set<uint64_t> cuts{range.begin, range.end};
    for (const auto& boundary : region.storageBoundary) {
        accumulateCost(visits, 1);
        const auto& cell = boundary.cell;
        if (cell.space != range.space) { continue; }
        if (cell.base != range.base) {
            SmallVector<SyncStorageCell> domains{cell, range};
            if (!storageBasesAreComparable(domains, region.gmAliasPolicy)) { return std::nullopt; }
            continue;
        }
        if (cell.begin > cell.end) { return std::nullopt; }
        if (range.begin < cell.begin && cell.begin < range.end) { cuts.insert(cell.begin); }
        if (range.begin < cell.end && cell.end < range.end) { cuts.insert(cell.end); }
        if (cuts.size() - 1 > maxRegionalSlotVisits) { return std::nullopt; }
    }
    return std::vector<uint64_t>(cuts.begin(), cuts.end());
}
std::optional<std::vector<uint64_t>> finiteSelectorCuts(
    const RegionalAnalysis& region, SyncStorageCell range, uint64_t& visits)
{
    if (auto uniform = uniformSelectorCuts(region, range, visits)) { return uniform; }
    if (range.begin >= range.end || range.end > uint64_t(INT64_MAX) ||
        range.end - range.begin > maxRegionalSlotVisits) { return std::nullopt; }
    std::vector<uint64_t> cuts;
    for (auto point = range.begin; point < range.end; ++point) {
        accumulateCost(visits, 1);
        cuts.push_back(point);
    }
    cuts.push_back(range.end);
    return cuts;
}
// Union requested atoms while retaining every certified cut. Overlapping
// counterpart ranges must not create overlapping selector alternatives.
bool normalizeSelectorSupport(SmallVector<SyncStorageCell>& cells)
{
    std::map<AddressSpace, llvm::MapVector<Value, std::map<uint64_t, int64_t>>> domains;
    for (const auto& cell : cells) {
        if (cell.begin >= cell.end || cell.end > uint64_t(INT64_MAX)) { return false; }
        ++domains[cell.space][cell.base][cell.begin];
        --domains[cell.space][cell.base][cell.end];
    }
    cells.clear();
    for (const auto& [space, bases] : domains) {
        for (const auto& [base, points] : bases) {
            int64_t active = 0;
            uint64_t previous = 0;
            for (const auto& [point, delta] : points) {
                if (active > 0 && previous < point) {
                    if (cells.size() == maxRegionalSlotVisits) { return false; }
                    cells.push_back({space, previous, point, base});
                }
                active += delta;
                previous = point;
            }
        }
    }
    return true;
}
std::optional<std::vector<uint64_t>> commonSelectorCuts(
    ArrayRef<RegionalAnalysis> children, SyncStorageCell range, uint64_t& visits)
{
    if (range.begin >= range.end || range.end > uint64_t(INT64_MAX)) { return std::nullopt; }
    std::set<uint64_t> cuts{range.begin, range.end};
    for (const auto& child : children) {
        auto local = uniformSelectorCuts(child, range, visits);
        if (!local) { return std::nullopt; }
        cuts.insert(local->begin(), local->end());
        if (cuts.size() - 1 > maxRegionalSlotVisits) { return std::nullopt; }
    }
    return std::vector<uint64_t>(cuts.begin(), cuts.end());
}
} // namespace mlir::pto::frontiersynch
