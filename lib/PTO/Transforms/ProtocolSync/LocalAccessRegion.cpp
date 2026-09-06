// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalAccessRegion.cpp - Independently recover local footprints -------===//

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/IR/Matchers.h"

#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool supportedCubeLayout(TileBufType type, std::uint64_t bytes)
{
    const auto shape = type.getShape();
    const bool ordinaryElement = type.getElementType().isIntOrFloat() &&
                                 type.getElementType().getIntOrFloatBitWidth() == 8 * bytes && bytes <= 4;
    const bool ordinaryShape = shape.size() == 2 && shape[0] > 0 && shape[1] > 0;
    if (!ordinaryElement || !ordinaryShape || bytes == 0) {
        return false;
    }
    if (32 % bytes != 0) {
        return false;
    }
    const auto outer = type.getBLayoutValueI32();
    const auto inner = type.getSLayoutValueI32();
    const bool rowOuter = outer == static_cast<int32_t>(BLayout::RowMajor);
    const bool colOuter = outer == static_cast<int32_t>(BLayout::ColMajor);
    if (!rowOuter && !colOuter) {
        return false;
    }
    if (inner == static_cast<int32_t>(SLayout::NoneBox)) {
        const auto minor = static_cast<std::uint64_t>(rowOuter ? shape[1] : shape[0]);
        return minor % (32 / bytes) == 0 && type.getCompactModeI32() == static_cast<int32_t>(CompactMode::Null);
    }
    const bool rowInner = inner == static_cast<int32_t>(SLayout::RowMajor);
    const bool colInner = inner == static_cast<int32_t>(SLayout::ColMajor);
    // Qualified complete-box layouts are permutations of a dense allocation.
    // Reject partial boxes, MX/sub-byte packing and unqualified layout variants.
    const bool ordinaryBox = rowInner || (rowOuter && colInner);
    const auto fractal = type.getSFractalSizeI32();
    if (!ordinaryBox || (fractal != 512 && fractal != 1024)) {
        return false;
    }
    // C-fractal qualification is limited to the ordinary four-byte layout.
    if (fractal == 1024 && bytes != 4) {
        return false;
    }
    const std::uint64_t rows = fractal == 1024 || rowInner ? 16 : 32 / bytes;
    const std::uint64_t columns = fractal == 1024 || colInner ? 16 : 32 / bytes;
    const bool fullCompact =
        type.getCompactModeI32() == static_cast<int32_t>(CompactMode::Normal) && type.getValidShape() == shape;
    const bool compact = type.getCompactModeI32() == static_cast<int32_t>(CompactMode::Null) || fullCompact;
    return compact && static_cast<std::uint64_t>(shape[0]) % rows == 0 &&
           static_cast<std::uint64_t>(shape[1]) % columns == 0;
}

SyncLocalAccessRegion recoverAllocationRegion(const SyncAccess& access, AllocTileOp allocation)
{
    SyncLocalAccessRegion region;
    region.access = access.id;
    const auto storageSpace = access.storage.space;
    const bool local = storageSpace == AddressSpace::VEC || storageSpace == AddressSpace::MAT ||
                       storageSpace == AddressSpace::LEFT || storageSpace == AddressSpace::RIGHT ||
                       storageSpace == AddressSpace::ACC;
    if (!allocation || access.slot || !local) {
        return region;
    }
    auto type = allocation.getResult().getType();
    auto space = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
    IntegerAttr address;
    const bool knownAddress = allocation.getAddr() && matchPattern(allocation.getAddr(), m_Constant(&address));
    const bool validAddress =
        knownAddress && address.getValue().getBitWidth() <= 64 && !address.getValue().isNegative();
    const bool packedStride = type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne);
    const bool matchingSpace = space && space.getAddressSpace() == storageSpace;
    if (!matchingSpace || !validAddress || packedStride) {
        return region;
    }
    std::uint64_t bytes = getPTOStorageElemByteSize(type.getElementType());
    if (bytes == 0) {
        return region;
    }
    const bool unsupportedLayout = storageSpace != AddressSpace::VEC && !supportedCubeLayout(type, bytes);
    if (unsupportedLayout) {
        return region;
    }
    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    for (std::int64_t dimension : type.getShape()) {
        if (dimension <= 0 || bytes > maximum / static_cast<std::uint64_t>(dimension)) {
            return region;
        }
        bytes *= static_cast<std::uint64_t>(dimension);
    }
    const std::uint64_t begin = address.getValue().getZExtValue();
    if (begin > maximum - bytes) {
        return region;
    }
    region.interval = {begin, bytes};
    region.space = storageSpace;
    // MemoryEffectOpInterface identifies a tile, not exactly which of its bytes
    // the instruction touches. Even a static allocation is only an upper bound.
    region.precision = SyncRegionPrecision::Conservative;
    return region;
}

std::optional<std::uint64_t> nonnegativeConstant(Value value)
{
    IntegerAttr attribute;
    const bool known = value && matchPattern(value, m_Constant(&attribute));
    const bool invalid = !known || attribute.getValue().getBitWidth() > 64 || attribute.getValue().isNegative();
    if (invalid) {
        return std::nullopt;
    }
    return attribute.getValue().getZExtValue();
}

bool isOrdinaryRowLayout(TileBufType type)
{
    return type.getRank() == 2 && type.getBLayoutValueI32() == static_cast<int32_t>(BLayout::RowMajor) &&
           type.getSLayoutValueI32() == static_cast<int32_t>(SLayout::NoneBox) &&
           type.getCompactModeI32() == static_cast<int32_t>(CompactMode::Null);
}

SyncLocalAccessRegion recoverRowSubview(const SyncAccess& access, SubViewOp view)
{
    SyncLocalAccessRegion unknown;
    unknown.access = access.id;
    auto allocation = view.getSource().getDefiningOp<AllocTileOp>();
    const bool unsupportedView = !allocation || view.getOffsets().size() != 2 || view.getSizes().size() != 2 ||
                                 view.getValidRow() || view.getValidCol();
    if (unsupportedView) {
        return unknown;
    }
    const TileBufType parent = allocation.getResult().getType();
    const TileBufType child = view.getResult().getType();
    const bool compatibleTypes = isOrdinaryRowLayout(parent) && isOrdinaryRowLayout(child) &&
                                 parent.getElementType() == child.getElementType() &&
                                 parent.getMemorySpace() == child.getMemorySpace() &&
                                 parent.getConfigAttr() == child.getConfigAttr();
    const bool supportedShape = compatibleTypes && child.getValidShape() == child.getShape() &&
                                parent.getShape()[0] > 0 && child.getShape()[0] > 0 &&
                                child.getShape()[1] == parent.getShape()[1];
    if (!supportedShape) {
        return unknown;
    }
    for (unsigned dimension = 0; dimension < 2; ++dimension) {
        auto size = dyn_cast<IntegerAttr>(view.getSizes()[dimension]);
        const bool matchingSize =
            size && size.getValue().getBitWidth() <= 64 && size.getInt() == child.getShape()[dimension];
        if (!matchingSize) {
            return unknown;
        }
    }
    const auto row = nonnegativeConstant(view.getOffsets()[0]);
    const auto column = nonnegativeConstant(view.getOffsets()[1]);
    const auto parentRows = static_cast<std::uint64_t>(parent.getShape()[0]);
    const auto childRows = static_cast<std::uint64_t>(child.getShape()[0]);
    if (!row || !column || *column != 0 || childRows > parentRows || *row > parentRows - childRows) {
        return unknown;
    }
    SyncLocalAccessRegion region = recoverAllocationRegion(access, allocation);
    if (region.precision == SyncRegionPrecision::Unknown) {
        return unknown;
    }
    // Only full-width row slices: their inherited and result-type strides
    // coincide, including after ResolveBufferSelect. Column slices can retain
    // a larger physical type; do not infer a footprint from logical sizes.
    // The allocation calculation proved size and end cannot overflow, and the
    // dimensional containment above bounds both products and the adjusted end.
    const std::uint64_t rowBytes = region.interval.size / parentRows;
    region.interval = {region.interval.begin + *row * rowBytes, childRows * rowBytes};
    return region;
}

} // namespace

SyncLocalAccessRegion mlir::pto::protocol_sync::recoverLocalAccessRegion(const SyncAccess& access)
{
    if (access.value && !access.slot && access.storage.space == AddressSpace::VEC) {
        if (auto view = access.value.getDefiningOp<SubViewOp>()) {
            return recoverRowSubview(access, view);
        }
    }
    auto allocation = access.value ? access.value.getDefiningOp<AllocTileOp>() : AllocTileOp();
    return recoverAllocationRegion(access, allocation);
}
