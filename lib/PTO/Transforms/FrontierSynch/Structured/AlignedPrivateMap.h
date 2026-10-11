// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bind exact original machine affine maps to a whole-invocation private segment proof.
#ifndef PTO_FRONTIERSYNCH_ALIGNEDPRIVATEMAP_H
#define PTO_FRONTIERSYNCH_ALIGNEDPRIVATEMAP_H
#include "AlignedModularMap.h"
#include "AlignedPrivateTranslation.h"
namespace mlir::pto::frontiersynch::detail {
inline bool hasPrivateBytes(const AlignedModularMap& map)
{
    return llvm::any_of(map.access.intervals, [](auto range) { return range.first < range.second; });
}
struct AlignedPrivateMapGroup {
    Value base, origin;
    unsigned width, alignmentBits;
    uint64_t byteStride, trips;
    llvm::APInt ordinalStride;
    llvm::APInt lower, upper, step;
    std::vector<AlignedModularAccess> accesses;
    SmallVector<SyncStorageCell> relativeSegments;
};
inline std::optional<uint64_t> privateConstantTrips(scf::ForOp loop, unsigned width,
    llvm::APInt& lower, llvm::APInt& upper, llvm::APInt& step)
{
    if (!loop || !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || lower.getBitWidth() != width ||
        upper.getBitWidth() != width || step.getBitWidth() != width || !step.isStrictlyPositive()) {
        return std::nullopt;
    }
    const auto distance = upper.sext(128) - lower.sext(128);
    const auto count = distance.isStrictlyPositive() ?
        (distance - 1).udiv(step.sext(128)) + 1 : llvm::APInt(128, 0);
    return count.isIntN(64) ? std::optional<uint64_t>(count.getZExtValue()) : std::nullopt;
}
// Full segment coverage is deliberately sufficient: distinct-payload writes
// sharing a segment decline even when a finer residual test could prove safety.
inline bool privateWithinVisit(ArrayRef<AlignedModularMap> maps, unsigned bits,
    ArrayRef<std::size_t> payloads, ArrayRef<uint8_t> writes)
{
    struct Seen { std::size_t first; bool multiple = false; std::optional<std::size_t> writer; };
    std::map<uint64_t, Seen> seen;
    for (unsigned i = 0; i < maps.size(); ++i) {
        if (!hasPrivateBytes(maps[i])) { continue; }
        const auto label = maps[i].access.offset.lshr(bits).getZExtValue();
        auto& cell = seen.try_emplace(label, Seen{payloads[i], false, {}}).first->second;
        if (cell.writer && *cell.writer != payloads[i]) { return false; }
        cell.multiple |= cell.first != payloads[i];
        if (writes[i]) {
            if (cell.multiple) { return false; }
            cell.writer = payloads[i];
        }
    }
    return true;
}
inline std::optional<unsigned> privateMapAlignment(ArrayRef<AlignedModularMap> maps, uint64_t budget)
{
    if (maps.empty() || maps.size() > budget || budget > UINT32_MAX) { return std::nullopt; }
    const auto& first = maps.front();
    if (!first.width || first.width > 64 || first.ordinalStride.getBitWidth() != first.width) {
        return std::nullopt;
    }
    unsigned bits = first.alignmentBits;
    for (const auto& map : maps) {
        if (map.base != first.base || map.origin != first.origin || map.width != first.width ||
            map.byteStride != first.byteStride || map.alignmentBits >= map.width ||
            map.access.offset.getBitWidth() != map.width || map.ordinalStride.getBitWidth() != map.width ||
            map.ordinalStride != first.ordinalStride || budget < 2) { return std::nullopt; }
        budget -= 2; // One map descriptor and one relative segment.
        if (map.access.intervals.size() > budget) { return std::nullopt; }
        budget -= map.access.intervals.size();
        bits = std::min(bits, map.alignmentBits);
    }
    if (!first.byteStride || first.byteStride > UINT64_MAX / (uint64_t{1} << bits)) { return std::nullopt; }
    const auto span = first.byteStride * (uint64_t{1} << bits);
    for (const auto& map : maps) {
        for (auto range : map.access.intervals) {
            if (range.first > range.second || range.second > span) { return std::nullopt; }
        }
    }
    return bits;
}
inline std::optional<AlignedPrivateMapGroup> provePrivateAlignedMaps(scf::ForOp loop,
    ArrayRef<AlignedModularMap> maps, ArrayRef<std::size_t> payloads, ArrayRef<uint8_t> writes,
    uint64_t maximumFragments, std::function<void(uint64_t)> attempted = {})
{
    if (maps.size() != payloads.size() || maps.size() != writes.size()) { return std::nullopt; }
    auto bits = privateMapAlignment(maps, maximumFragments);
    if (!bits) { return std::nullopt; }
    const auto& first = maps.front();
    llvm::APInt lower, upper, step;
    auto trips = privateConstantTrips(loop, first.width, lower, upper, step);
    if (!trips) { return std::nullopt; }
    if (attempted) { attempted(maps.size()); }
    if (!privateWithinVisit(maps, *bits, payloads, writes)) { return std::nullopt; }
    SmallVector<llvm::APInt> offsets;
    if (attempted) { attempted(maps.size()); }
    for (const auto& map : maps) {
        if (hasPrivateBytes(map)) { offsets.push_back(map.access.offset); }
    }
    if (attempted) { attempted(offsets.size()); }
    auto cells = privateAlignedSegments(first.width, *bits, offsets, first.ordinalStride, *trips);
    if (!cells) { return std::nullopt; }
    AlignedPrivateMapGroup result{first.base, first.origin, first.width, *bits, first.byteStride, *trips,
        first.ordinalStride, lower, upper, step, {}, std::move(*cells)};
    for (const auto& map : maps) {
        if (attempted) { attempted(1 + map.access.intervals.size()); }
        result.accesses.push_back(map.access);
    }
    return result;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
