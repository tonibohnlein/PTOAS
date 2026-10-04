// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Geometry comes from the shared translator records. Read/write modes come only
// from SyncInput. Allocation extents and alternative slots are upper bounds.
#include "SyncEffectRanges.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

namespace mlir::pto::detail {
namespace {
std::optional<uint64_t> constant(Value value)
{
    APInt integer;
    if (!value || !matchPattern(value, m_ConstantInt(&integer)) || integer.isNegative() ||
        integer.getActiveBits() > 64) {
        return std::nullopt;
    }
    return integer.getZExtValue();
}

std::optional<uint64_t> elementBytes(TileBufType type)
{
    auto element = type.getElementType();
    if (!element.isIntOrFloat()) {
        return std::nullopt;
    }
    auto bits = element.getIntOrFloatBitWidth();
    return bits && bits % 8 == 0 ? std::optional<uint64_t>(bits / 8) : std::nullopt;
}

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

// This optional scalar refinement needs contiguous coordinates. It does not
// restrict generic buffer bounds, which are supplied for every translated op.
bool hasScalarCoordinates(const BaseMemInfo& memory)
{
    auto type = dyn_cast<TileBufType>(memory.baseBuffer.getType());
    if (!type || type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox) ||
        type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne)) {
        return false;
    }
    if (memory.baseBuffer == memory.rootBuffer) {
        return true;
    }
    auto get = memory.baseBuffer.getDefiningOp<MultiTileGetOp>();
    return get && get.getSource() == memory.rootBuffer && constant(get.getSlot()).has_value();
}

void normalize(SmallVectorImpl<SyncStorageCell>& ranges)
{
    llvm::sort(ranges, [](const auto& a, const auto& b) { return a.begin < b.begin; });
    std::size_t count = 0;
    for (const auto& range : ranges) {
        if (count && range.begin <= ranges[count - 1].end) {
            ranges[count - 1].end = std::max(ranges[count - 1].end, range.end);
        } else {
            ranges[count++] = range;
        }
    }
    ranges.resize(count);
}

void refineScalarElement(const SyncInput& input, SyncStorageEffect& effect, bool fixedCoordinates)
{
    auto* operation = effect.phase->elementOp;
    Value operand, index;
    if (auto read = dyn_cast<TGetValOp>(operation); read && effect.mode == SyncAccessMode::Read) {
        operand = read.getSrc();
        index = read.getOffset();
    } else if (auto write = dyn_cast<TSetValOp>(operation); write && effect.mode == SyncAccessMode::Write) {
        operand = write.getDst();
        index = write.getOffset();
    } else {
        return;
    }
    auto type = dyn_cast<TileBufType>(operand.getType());
    auto found = input.buffers().find(operand);
    if (!fixedCoordinates || !type || effect.ranges.size() != 1 || operand != effect.memory->baseBuffer ||
        found == input.buffers().end() || found->second.size() != 1) {
        return;
    }
    auto offset = constant(index);
    auto bytes = elementBytes(type);
    if (!offset || !bytes) {
        return;
    }
    auto& range = effect.ranges.front();
    if (*offset >= (range.end - range.begin) / *bytes) {
        // Out-of-range accesses are not proved contained in the allocation.
        effect.ranges.clear();
        effect.precision = SyncAccessPrecision::Unknown;
        return;
    }
    range.begin += *offset * *bytes;
    range.end = range.begin + *bytes;
    effect.precision = SyncAccessPrecision::Exact;
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
    effect.ranges = sharedRanges(memory);
    if (effect.ranges.empty()) {
        return;
    }
    const bool fixedCoordinates = hasScalarCoordinates(memory) && effect.ranges.size() == 1;
    normalize(effect.ranges);
    effect.precision = SyncAccessPrecision::UpperBound;
    refineScalarElement(input, effect, fixedCoordinates);
}
} // namespace mlir::pto::detail
