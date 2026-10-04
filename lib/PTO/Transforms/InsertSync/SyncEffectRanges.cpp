// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Geometry comes from the shared translator records. Read/write modes come only
// from SyncInput. Buffer descriptors do not establish accessed byte sets.
#include "SyncEffectRanges.h"
#include <limits>

namespace mlir::pto::detail {
namespace {
SmallVector<SyncStorageCell> sharedRanges(const BaseMemInfo& memory)
{
    if (memory.aliasesUnknownRange || !memory.hasKnownPhysicalAddresses ||
        memory.scope == AddressSpace::GM || memory.scope == AddressSpace::Zero || !memory.allocateSize) {
        return {};
    }
    SmallVector<SyncStorageCell> result;
    for (uint64_t begin : memory.baseAddresses) {
        if (memory.allocateSize > UINT64_MAX - begin) {
            return {};
        }
        result.push_back({memory.scope, begin, begin + memory.allocateSize});
    }
    return result;
}

} // namespace

SmallVector<SyncStorageCell> physicalSlotRanges(const SyncInput& input, const BaseMemInfo& memory)
{
    auto root = input.buffers().find(memory.rootBuffer);
    if (root == input.buffers().end() || root->second.size() != 1) {
        return {};
    }
    return sharedRanges(*root->second.front());
}

void resolveEffectRanges(const SyncInput& input, SyncStorageEffect& effect)
{
    const auto& memory = *effect.memory;
    if (!memory.rootBuffer || !memory.baseBuffer) {
        return;
    }
    effect.descriptorRegion = resolveBufferRegion(input, memory.baseBuffer, effect.phase->elementOp);
    // The common interface supplies read/write operands, not an exact accessed
    // region or a promise of containment in the allocation. Keep the geometry
    // available to clients without guessing instruction semantics or kills.
    effect.precision = SyncAccessPrecision::Unknown;
    effect.precisionReason = "shared read/write interface does not specify an access region";
}
} // namespace mlir::pto::detail
