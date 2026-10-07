// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite symbolic support is queried per byte; no interval-constant selector assumption.
#include "RepeatedReadOnlyStorage.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>
namespace mlir::pto::frontiersynch {
namespace {
std::optional<SmallVector<SyncStorageCell>> support(const RegionalAnalysis& body)
{
    if (!body.accessModel->hasMaterializedCellPartition(body.symbolicStorageEffects)) { return std::nullopt; }
    SmallVector<SyncStorageCell> bytes, domains;
    for (const auto& boundary : body.storageBoundary) { domains.push_back(boundary.cell); }
    for (auto id : body.symbolicStorageEffects) {
        for (const auto& range : body.accessModel->effects()[id].ranges) {
            // The shared model validates identity and complete materialization;
            // this export additionally checks its own alias context and budget.
            if (range.begin > range.end || range.end - range.begin > maxRegionalSlotVisits) { return std::nullopt; }
            domains.push_back(range);
            for (auto byte = range.begin; byte < range.end; ++byte) {
                auto same = [&](const SyncStorageCell& old) {
                    return sameStorageDomain(old, range) && old.begin == byte;
                };
                if (llvm::any_of(bytes, same)) { continue; }
                if (bytes.size() == maxRegionalSlotVisits) { return std::nullopt; }
                bytes.push_back({range.space, byte, byte + 1, range.base});
            }
        }
    }
    if (!storageBasesAreComparable(domains, body.gmAliasPolicy)) { return std::nullopt; }
    return bytes;
}
} // namespace
std::string materializeRepeatedSymbolicStorage(
    RegionalAnalysis& body, scf::ForOp loop, RegionExpressions::Id trips)
{
    if (!loop || !body.expressions || !body.storageSelectors || !body.accessModel ||
        trips >= body.expressions->size()) {
        return "finite symbolic boundary requires a complete shared model and exact byte selectors";
    }
    if (body.expressions->constantValue(trips) == uint64_t(0)) {
        // No crossings or storage interface entries exist at zero trips. Avoid
        // asking for support of an unexecuted symbolic footprint.
        body.storageBoundary.clear();
        body.accessBoundary.clear();
        body.deferredAccessBoundary.clear();
        body.firstPayloads.clear();
        body.lastPayloads.clear();
        body.firstSitePayloads.clear();
        return {};
    }
    for (auto id : body.symbolicStorageEffects) {
        auto matches = [id](const RegionalAccessBoundary& access) { return access.effect == id; };
        if (!llvm::any_of(body.accessBoundary, matches) && !llvm::any_of(body.deferredAccessBoundary, matches)) {
            return "finite repeated support lacks effect extrema for unresolved storage relationships";
        }
    }
    const auto effects = body.accessModel->effects();
    for (std::size_t id = 0; id < effects.size(); ++id) {
        const auto& effect = effects[id];
        if (!effect.phase || !effect.phase->elementOp || !loop->isProperAncestor(effect.phase->elementOp)) { continue; }
        const bool relevant = llvm::any_of(body.symbolicStorageEffects,
            [&](std::size_t symbolic) { return body.accessModel->mayConflict(id, symbolic); });
        if (!relevant) { continue; }
        auto matches = [id](const RegionalAccessBoundary& access) { return access.effect == id; };
        if (!llvm::any_of(body.accessBoundary, matches) && !llvm::any_of(body.deferredAccessBoundary, matches)) {
            return "repeated symbolic conflict has no exported or deferred effect extrema";
        }
    }
    auto bytes = support(body);
    if (!bytes) {
        return "symbolic repeated crossings need complete finite selector support within the regional expansion limit";
    }
    std::vector<RegionalStorageBoundary> boundaries;
    auto& e = *body.expressions;
    for (const auto& byte : *bytes) {
        auto selectors = body.storageSelectors({byte.space, byte.base, e.constant(byte.begin)});
        if (!selectors) { return "exact per-byte selectors unavailable for finite repeated support"; }
        boundaries.push_back({byte, std::move(selectors->firstWriters), std::move(selectors->lastWriters),
            std::move(selectors->firstReaders), std::move(selectors->lastReaders)});
    }
    if (boundaries.size() > UINT64_MAX - body.cost.cells || boundaries.size() > UINT64_MAX - body.cost.boundaryBytes) {
        return "finite repeated selector representation cost overflow";
    }
    // Existing explicit cells retain their exact selectors. Overlapping atoms
    // are combined by the ordinary sequence interface's selector normalization.
    body.cost.cells += boundaries.size();
    body.cost.boundaryBytes += boundaries.size();
    body.storageBoundary.insert(body.storageBoundary.end(),
        std::make_move_iterator(boundaries.begin()), std::make_move_iterator(boundaries.end()));
    return {};
}
} // namespace mlir::pto::frontiersynch
