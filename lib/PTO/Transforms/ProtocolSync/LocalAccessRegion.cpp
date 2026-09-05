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

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

SyncLocalAccessRegion mlir::pto::protocol_sync::recoverLocalAccessRegion(const SyncAccess& access)
{
    SyncLocalAccessRegion region;
    region.access = access.id;
    auto allocation = access.value ? access.value.getDefiningOp<AllocTileOp>() : AllocTileOp();
    if (!allocation || access.slot || access.storage.space != AddressSpace::VEC) {
        return region;
    }
    auto type = allocation.getResult().getType();
    auto space = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
    IntegerAttr address;
    const bool knownAddress = allocation.getAddr() && matchPattern(allocation.getAddr(), m_Constant(&address));
    const bool validAddress =
        knownAddress && address.getValue().getBitWidth() <= 64 && !address.getValue().isNegative();
    const bool packedStride = type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne);
    const bool vectorSpace = space && space.getAddressSpace() == AddressSpace::VEC;
    if (!vectorSpace || !validAddress || packedStride) {
        return region;
    }
    std::uint64_t bytes = getPTOStorageElemByteSize(type.getElementType());
    if (bytes == 0) {
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
    // MemoryEffectOpInterface identifies a tile, not exactly which of its bytes
    // the instruction touches. Even a static allocation is only an upper bound.
    region.precision = SyncRegionPrecision::Conservative;
    return region;
}
