// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Partition exact relative intervals inside aligned signed machine segments.
#ifndef PTO_FRONTIERSYNCH_ALIGNEDMODULARCELLS_H
#define PTO_FRONTIERSYNCH_ALIGNEDMODULARCELLS_H
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
// This numerical lemma does not establish an SSA origin, input scope or alias
// fact. Its caller must separately certify one common invariant machine origin
// aligned to 2^alignmentBits and the exact original byte-map decomposition.
struct AlignedModularAccess {
    llvm::APInt offset;
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
};
struct AlignedModularAtom {
    uint64_t segment, begin, end;
};
struct AlignedModularPartition {
    unsigned width, alignmentBits;
    uint64_t byteStride;
    std::vector<AlignedModularAtom> atoms;
    std::vector<std::vector<uint32_t>> references;
};
class AlignedModularPartitionBuilder {
public:
    AlignedModularPartitionBuilder(unsigned width, unsigned alignmentBits, uint64_t byteStride,
        llvm::ArrayRef<AlignedModularAccess> accesses, uint64_t maximumFragments,
        std::function<void(uint64_t)> attempted = {})
        : result{width, alignmentBits, byteStride, {}, {}}, accesses(accesses),
          maximumFragments(maximumFragments), fragments(accesses.size()), attempted(std::move(attempted)) {}

    std::optional<AlignedModularPartition> build()
    {
        if (attempted) { attempted(accesses.size()); }
        if (!collectBoundaries() || !partitionAtoms() || !collectReferences()) {
            return std::nullopt;
        }
        return std::move(result);
    }

private:
    bool charge()
    {
        if (fragments == maximumFragments) { return false; }
        ++fragments;
        if (attempted) { attempted(1); }
        return true;
    }

    bool collectBoundaries()
    {
        // The signed representative cut must lie on a segment boundary.
        if (!result.width || result.width > 64 || result.alignmentBits >= result.width ||
            !result.byteStride || accesses.size() > maximumFragments || maximumFragments > UINT32_MAX) {
            return false;
        }
        const auto granularity = uint64_t{1} << result.alignmentBits;
        if (result.byteStride > UINT64_MAX / granularity) { return false; }
        const auto span = granularity * result.byteStride;
        for (const auto& access : accesses) {
            if (access.offset.getBitWidth() != result.width ||
                (access.offset.getZExtValue() & (granularity - 1))) { return false; }
            auto& events = boundaries[access.offset.lshr(result.alignmentBits).getZExtValue()];
            for (auto interval : access.intervals) {
                if (interval.first > interval.second || interval.second > span || !charge()) { return false; }
                if (interval.first == interval.second) { continue; }
                ++events[interval.first];
                --events[interval.second];
            }
        }
        return true;
    }

    bool partitionAtoms()
    {
        for (const auto& [segment, events] : boundaries) {
            int64_t active = 0;
            uint64_t previous = 0;
            for (auto [position, delta] : events) {
                if (active && previous < position) {
                    if (!charge()) { return false; }
                    segmentAtoms[segment].push_back(static_cast<uint32_t>(result.atoms.size()));
                    result.atoms.push_back({segment, previous, position});
                }
                active += delta;
                previous = position;
            }
            if (active) { return false; }
        }
        return true;
    }

    bool collectReferences()
    {
        result.references.reserve(accesses.size());
        for (const auto& access : accesses) {
            std::vector<uint32_t> references;
            const auto found = segmentAtoms.find(access.offset.lshr(result.alignmentBits).getZExtValue());
            if (found != segmentAtoms.end()) {
                const auto& candidates = found->second;
                for (auto interval : access.intervals) {
                    if (interval.first == interval.second) { continue; }
                    auto first = std::lower_bound(candidates.begin(), candidates.end(), interval.first,
                        [&](uint32_t atom, uint64_t begin) { return result.atoms[atom].end <= begin; });
                    for (; first != candidates.end() && result.atoms[*first].begin < interval.second; ++first) {
                        if (!charge()) { return false; }
                        references.push_back(*first);
                    }
                }
            }
            std::sort(references.begin(), references.end());
            references.erase(std::unique(references.begin(), references.end()), references.end());
            result.references.push_back(std::move(references));
        }
        return true;
    }

    AlignedModularPartition result;
    llvm::ArrayRef<AlignedModularAccess> accesses;
    uint64_t maximumFragments;
    uint64_t fragments;
    std::function<void(uint64_t)> attempted;
    std::map<uint64_t, std::map<uint64_t, int64_t>> boundaries;
    std::map<uint64_t, std::vector<uint32_t>> segmentAtoms;
};

inline std::optional<AlignedModularPartition> partitionAlignedModularCells(unsigned width,
    unsigned alignmentBits, uint64_t byteStride, llvm::ArrayRef<AlignedModularAccess> accesses,
    uint64_t maximumFragments, std::function<void(uint64_t)> attempted = {})
{
    return AlignedModularPartitionBuilder(width, alignmentBits, byteStride, accesses, maximumFragments,
        std::move(attempted)).build();
}
} // namespace mlir::pto::frontiersynch::detail
#endif
