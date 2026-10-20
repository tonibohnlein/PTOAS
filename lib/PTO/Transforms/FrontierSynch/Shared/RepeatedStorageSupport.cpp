// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite geometry for exact repeated selectors, independently per storage domain.
#include "RepeatedStorageInternal.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using State = RepeatedStorage::State;
bool disjointDomain(const State& state, AddressSpace space, Value base, std::pair<AddressSpace, Value> domain)
{
    if (space != domain.first) {
        return true;
    }
    if (base == domain.second) {
        return false;
    }
    SmallVector<SyncStorageCell> ranges{{space, 0, 1, base}, {domain.first, 0, 1, domain.second}};
    return storageBasesAreComparable(ranges, state.body.gmAliasPolicy);
}
bool completeDomain(const State& state, std::optional<std::pair<AddressSpace, Value>> domain)
{
    for (const auto* side : {&state.body.accessBoundary, &state.body.deferredAccessBoundary}) {
        for (const auto& access : *side) {
            if (access.representedByCells || llvm::is_contained(state.effectIds, access.effect)) {
                continue;
            }
            if (!state.body.accessModel || access.effect >= state.body.accessModel->effects().size()) {
                return false;
            }
            const auto& effect = state.body.accessModel->effects()[access.effect];
            if (!effect.memory) {
                return false;
            }
            if (domain && effect.memory->scope != domain->first) {
                continue;
            }
            if (!domain) {
                return false;
            }
            const bool absent =
                effect.rangesMaterialized ?
                    llvm::all_of(
                        effect.ranges,
                        [&](const auto& cell) { return disjointDomain(state, cell.space, cell.base, *domain); }) :
                    !effect.regions.empty() && llvm::all_of(effect.regions, [&](const auto& piece) {
                        return disjointDomain(state, effect.memory->scope, piece.base, *domain);
                    });
            if (!absent) {
                return false;
            }
        }
    }
    return true;
}
std::optional<std::vector<SyncStorageCell>> support(
    const State& state, std::optional<std::pair<AddressSpace, Value>> domain, uint64_t& visits)
{
    const auto* owned = &state;
    auto& e = owned->expressions();
    auto trips = e.constantValue(owned->trips);
    if (!trips) {
        return std::nullopt;
    }
    std::vector<SyncStorageCell> ranges;
    if (!*trips) {
        return ranges;
    }
    if (!completeDomain(state, domain)) {
        return std::nullopt;
    }
    for (const auto& cell : owned->body.storageBoundary) {
        if (domain && disjointDomain(state, cell.cell.space, cell.cell.base, *domain)) {
            continue;
        }
        if (domain && cell.cell.base != domain->second) {
            return std::nullopt;
        }
        ranges.push_back(cell.cell);
    }
    for (const auto& family : owned->families) {
        if (domain && disjointDomain(state, family.spec.space, family.spec.origin.base, *domain)) {
            continue;
        }
        if (domain && family.spec.origin.base != domain->second) {
            return std::nullopt;
        }
        accumulateCost(visits, 1);
        auto origin = e.constantValue(family.origin);
        if (!origin || *origin > uint64_t(INT64_MAX) || family.extent > uint64_t(INT64_MAX) - *origin) {
            return std::nullopt;
        }
        auto end = *origin + family.extent;
        const auto stride = family.spec.stride;
        if (stride && *trips - 1 > (uint64_t(INT64_MAX) - end) / stride) {
            return std::nullopt;
        }
        end += (*trips - 1) * stride;
        if (*origin < end) {
            ranges.push_back({family.spec.space, *origin, end, family.spec.origin.base});
        }
    }
    return ranges;
}
// Return only translations intersecting the requested range. The existing atom
// limit bounds the resulting partition, not the numerical invocation count.
std::pair<uint64_t, uint64_t> intersecting(
    uint64_t begin, uint64_t end, uint64_t stride, uint64_t count, SyncStorageCell range)
{
    if (!count || begin >= end || range.end <= begin) {
        return {0, 0};
    }
    if (!stride) {
        return {0, range.begin < end ? 1 : 0};
    }
    const auto first = range.begin < end ? uint64_t(0) : (range.begin - end) / stride + 1;
    const auto distance = range.end - begin;
    const auto limit = distance / stride + uint64_t(distance % stride != 0);
    return {std::min(first, count), std::min(limit, count)};
}
} // namespace
void attachRepeatedStorageSupport(RegionalSymbolicStorageCertificate& certificate, std::shared_ptr<State> state)
{
    certificate.selectorSupport = [state](uint64_t& visits) { return support(*state, std::nullopt, visits); };
    certificate.selectorDomainSupport = [state](AddressSpace space, Value base, uint64_t& visits) {
        return support(*state, std::make_pair(space, base), visits);
    };
    certificate.selectorCuts = [owned = state](
                                   SyncStorageCell range, uint64_t& visits) -> std::optional<std::vector<uint64_t>> {
        if (range.begin >= range.end || range.end > uint64_t(INT64_MAX)) {
            return std::nullopt;
        }
        auto& e = owned->expressions();
        auto trips = e.constantValue(owned->trips);
        if (!trips) {
            return std::nullopt;
        }
        std::set<uint64_t> cuts{range.begin, range.end};
        if (!*trips) {
            return std::vector<uint64_t>(cuts.begin(), cuts.end());
        }
        if (!completeDomain(*owned, std::make_pair(range.space, range.base))) {
            return std::nullopt;
        }
        for (const auto& cell : owned->body.storageBoundary) {
            if (disjointDomain(*owned, cell.cell.space, cell.cell.base, {range.space, range.base})) {
                continue;
            }
            if (cell.cell.base != range.base) {
                return std::nullopt;
            }
            for (const auto point : {cell.cell.begin, cell.cell.end}) {
                if (range.begin < point && point < range.end) {
                    cuts.insert(point);
                }
            }
        }
        for (const auto& family : owned->families) {
            if (disjointDomain(*owned, family.spec.space, family.spec.origin.base, {range.space, range.base})) {
                continue;
            }
            if (family.spec.origin.base != range.base) {
                return std::nullopt;
            }
            auto origin = e.constantValue(family.origin);
            if (!origin) {
                return std::nullopt;
            }
            for (const auto& piece : family.pieces) {
                const auto innerTrips = piece.inner ? e.constantValue(piece.inner->trips) : std::optional<uint64_t>(1);
                if (!innerTrips) {
                    return std::nullopt;
                }
                if (!*innerTrips) {
                    continue;
                }
                const auto innerStride = piece.inner ? piece.inner->stride : uint64_t(0);
                if (*origin > uint64_t(INT64_MAX) || piece.end > uint64_t(INT64_MAX) - *origin ||
                    (innerStride && *innerTrips - 1 > (uint64_t(INT64_MAX) - *origin - piece.end) / innerStride)) {
                    return std::nullopt;
                }
                const auto span = piece.end + (*innerTrips - 1) * innerStride;
                const auto interval =
                    intersecting(*origin + piece.begin, *origin + span, family.spec.stride, *trips, range);
                for (auto visit = interval.first; visit < interval.second; ++visit) {
                    if (family.spec.stride && visit > (uint64_t(INT64_MAX) - *origin - span) / family.spec.stride) {
                        return std::nullopt;
                    }
                    const auto start = *origin + visit * family.spec.stride;
                    const auto innerRange =
                        intersecting(start + piece.begin, start + piece.end, innerStride, *innerTrips, range);
                    for (auto inner = innerRange.first; inner < innerRange.second; ++inner) {
                        accumulateCost(visits, 1);
                        const auto shift = start + inner * innerStride;
                        for (const auto point : {shift + piece.begin, shift + piece.end}) {
                            if (range.begin < point && point < range.end) {
                                cuts.insert(point);
                            }
                        }
                        if (cuts.size() - 1 > maxRegionalSlotVisits) {
                            return std::nullopt;
                        }
                    }
                }
            }
        }
        return std::vector<uint64_t>(cuts.begin(), cuts.end());
    };
}
} // namespace mlir::pto::frontiersynch
