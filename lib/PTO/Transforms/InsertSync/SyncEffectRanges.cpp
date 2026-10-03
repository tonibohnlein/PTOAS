// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Geometry is recovered from shared record anchors. Read/write modes come only
// from SyncInput. Allocation extents and alternative slots are upper bounds.
#include "SyncEffectRanges.h"
#include "PTO/IR/PTOMultiBuffer.h"
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

std::optional<uint64_t> plainTileBytes(TileBufType type)
{
    if (type.getRank() != 2 || type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox) ||
        type.getCompactModeI32() == static_cast<int32_t>(CompactMode::RowPlusOne)) {
        return std::nullopt;
    }
    auto size = elementBytes(type);
    for (auto dimension : type.getShape()) {
        if (!size || dimension <= 0 || static_cast<uint64_t>(dimension) > UINT64_MAX / *size) {
            return std::nullopt;
        }
        *size *= static_cast<uint64_t>(dimension);
    }
    return size;
}

SmallVector<SyncStorageCell> rootRanges(const BaseMemInfo& memory)
{
    SmallVector<SyncStorageCell> result;
    if (memory.aliasesUnknownRange || memory.scope == AddressSpace::GM || memory.scope == AddressSpace::Zero) {
        return result;
    }
    auto allocation = memory.rootBuffer.getDefiningOp<AllocTileOp>();
    auto multi = memory.rootBuffer.getDefiningOp<AllocMultiTileOp>();
    if (!allocation && !multi) {
        return result;
    }
    TileBufType type = allocation ? cast<TileBufType>(allocation.getResult().getType()) :
                                   multi.getResult().getType().getSlotType();
    auto bytes = plainTileBytes(type);
    if (!bytes) {
        return result;
    }
    auto address = constant(allocation ? allocation.getAddr() : multi.getAddr());
    auto planned = multi ? multi->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName) :
                           DenseI64ArrayAttr{};
    const uint64_t count = multi ? multi.getResult().getType().getCount() : 1;
    if ((!planned && !address) || (planned && static_cast<uint64_t>(planned.size()) != count)) {
        return result;
    }
    for (uint64_t slot = 0; slot < count; ++slot) {
        uint64_t begin = 0;
        if (planned) {
            if (planned[slot] < 0) {
                return {};
            }
            begin = static_cast<uint64_t>(planned[slot]);
        } else {
            if (slot > (UINT64_MAX - *address) / *bytes) {
                return {};
            }
            begin = *address + slot * *bytes;
        }
        if (*bytes > UINT64_MAX - begin) {
            return {};
        }
        result.push_back({memory.scope, begin, begin + *bytes});
    }
    return result;
}

// Only immediate, constant, plain-layout subviews are narrowed here. A nested
// or dynamic view retains the whole root bound, not a guessed view address.
std::optional<SmallVector<SyncStorageCell>> subviewRanges(
    const BaseMemInfo& memory, ArrayRef<SyncStorageCell> roots)
{
    auto view = memory.baseBuffer.getDefiningOp<SubViewOp>();
    if (!view || view.getSource() != memory.rootBuffer || roots.size() != 1) {
        return std::nullopt;
    }
    auto type = cast<TileBufType>(view.getSource().getType());
    auto row = constant(view.getOffsets()[0]), col = constant(view.getOffsets()[1]);
    auto sizes = view.getSizes();
    const int64_t height = cast<IntegerAttr>(sizes[0]).getInt();
    const int64_t width = cast<IntegerAttr>(sizes[1]).getInt();
    const auto rows = static_cast<uint64_t>(type.getShape()[0]);
    const auto cols = static_cast<uint64_t>(type.getShape()[1]);
    if (!row || !col || height <= 0 || width <= 0 || *row > rows || *col > cols ||
        static_cast<uint64_t>(height) > rows - *row || static_cast<uint64_t>(width) > cols - *col) {
        return std::nullopt;
    }
    const bool rowMajor = type.getBLayoutValueI32() == static_cast<int32_t>(BLayout::RowMajor);
    const auto start = rowMajor ? *row : *col;
    const auto count = static_cast<uint64_t>(rowMajor ? height : width);
    const auto offset = rowMajor ? *col : *row;
    const auto stride = rowMajor ? cols : rows;
    const auto length = static_cast<uint64_t>(rowMajor ? width : height);
    const auto bytes = *elementBytes(type);
    SmallVector<SyncStorageCell> result;
    for (uint64_t i = 0; i < count; ++i) {
        // Bounds above and checked root size prove all products and sums fit.
        const auto begin = roots.front().begin + ((start + i) * stride + offset) * bytes;
        result.push_back({memory.scope, begin, begin + length * bytes});
    }
    return result;
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

SmallVector<SyncStorageCell> physicalSlotRanges(const BaseMemInfo& memory)
{
    return rootRanges(memory);
}

void resolveEffectRanges(const SyncInput& input, SyncStorageEffect& effect)
{
    const auto& memory = *effect.memory;
    if (!memory.rootBuffer || !memory.baseBuffer) {
        return;
    }
    effect.ranges = rootRanges(memory);
    if (effect.ranges.empty()) {
        return;
    }
    bool fixedCoordinates = memory.baseBuffer == memory.rootBuffer && effect.ranges.size() == 1;
    if (auto narrowed = subviewRanges(memory, effect.ranges)) {
        effect.ranges = std::move(*narrowed);
        fixedCoordinates = true;
    } else if (auto get = memory.baseBuffer.getDefiningOp<MultiTileGetOp>();
               get && get.getSource() == memory.rootBuffer) {
        if (auto slot = constant(get.getSlot()); slot && *slot < effect.ranges.size()) {
            const auto range = effect.ranges[*slot];
            effect.ranges.assign(1, range);
            fixedCoordinates = true;
        }
    }
    normalize(effect.ranges);
    effect.precision = SyncAccessPrecision::UpperBound;
    refineScalarElement(input, effect, fixedCoordinates);
}
} // namespace mlir::pto::detail
