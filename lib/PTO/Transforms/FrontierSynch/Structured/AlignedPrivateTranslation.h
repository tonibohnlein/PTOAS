// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient finite translation proof in relative segment coordinates only.
#ifndef PTO_FRONTIERSYNCH_ALIGNEDPRIVATETRANSLATION_H
#define PTO_FRONTIERSYNCH_ALIGNEDPRIVATETRANSLATION_H
#include "../Shared/DisjointTranslations.h"
#include <cstdint>
#include <optional>
namespace mlir::pto::frontiersynch::detail {
// Caller separately proves one common aligned invariant origin, original-width
// scalar decomposition, exact residuals inside each segment, and absence of
// conflicting accesses between distinct payloads within one visit. These cells
// are relative segment labels, never absolute byte addresses or storage exports.
inline std::optional<SmallVector<SyncStorageCell>> privateAlignedSegments(
    unsigned width, unsigned alignmentBits, ArrayRef<APInt> offsets,
    const APInt& ordinalStride, uint64_t trips)
{
    if (!width || width > 64 || alignmentBits >= width || ordinalStride.getBitWidth() != width ||
        ordinalStride.isNegative()) { return std::nullopt; }
    const auto mask = APInt::getLowBitsSet(width, alignmentBits);
    if (!(ordinalStride & mask).isZero()) { return std::nullopt; }
    const auto stride = ordinalStride.zext(128).lshr(alignmentBits);
    const auto count = APInt(128, 1).shl(width - alignmentBits);
    const auto advance = stride * APInt(128, trips ? trips - 1 : 0);
    SmallVector<SyncStorageCell> cells;
    for (const auto& offset : offsets) {
        if (offset.getBitWidth() != width || !(offset & mask).isZero()) { return std::nullopt; }
        const auto first = offset.zext(128).lshr(alignmentBits);
        const auto end = first + APInt(128, 1);
        if ((end + advance).ugt(count) || !end.isIntN(64)) { return std::nullopt; }
        cells.push_back({AddressSpace::GM, first.getZExtValue(), end.getZExtValue()});
    }
    if (!disjointTranslations(cells, stride, trips)) { return std::nullopt; }
    return cells;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
