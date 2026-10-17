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
    if (!region.symbolicStorageEffects.empty() || !region.deferredAccessBoundary.empty() ||
        !region.capabilities.completeStorageModel || !region.capabilities.exactSelectors ||
        llvm::any_of(region.accessBoundary, [](const auto& access) { return !access.representedByCells; })) {
        return std::nullopt;
    }
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
