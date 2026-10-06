// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Derive fixed within-slot bytes by subtracting the shared selected-slot map.
// No instruction-specific footprint rules or sampled loop iterations are used.
#include "RotationPattern.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::pto::frontiersynch::detail {
namespace {
std::optional<SyncAccessRegion> slotOrigin(const SyncStorageEffect& effect, const SyncInput& input)
{
    Value operand = effect.memory->baseBuffer;
    while (auto* op = operand.getDefiningOp()) {
        if (auto get = dyn_cast<MultiTileGetOp>(op)) {
            const bool matchesFamily = effect.selection && get.getSource() == effect.selection->family;
            if (!matchesFamily) {
                return std::nullopt;
            }
            auto descriptor = resolveBufferRegion(input, operand, effect.phase->elementOp);
            if (descriptor) {
                SmallVector<AffineExpr> zero(descriptor->extents.size(),
                    getAffineConstantExpr(0, op->getContext()));
                descriptor->byteOffset = descriptor->byteOffset.replaceDims(zero);
                descriptor->extents.clear();
            }
            return descriptor;
        }
        auto view = dyn_cast<ViewLikeOpInterface>(op);
        const bool stop = !view || view.getViewSource() == operand;
        if (stop) {
            break;
        }
        operand = view.getViewSource();
    }
    auto slots = mlir::pto::detail::physicalSlotRanges(input, *effect.memory);
    const bool fixedBase = slots.size() == 1 && slots.front().begin <= INT64_MAX;
    if (!fixedBase) {
        return std::nullopt;
    }
    SyncAccessRegion origin;
    origin.byteOffset = getAffineConstantExpr(slots.front().begin, effect.phase->elementOp->getContext());
    return origin;
}

bool relativeRanges(const SyncAccessRegion& region, const SyncAccessRegion& origin, AddressSpace space,
                    SmallVectorImpl<SyncStorageCell>& ranges)
{
    if (region.base != origin.base) {
        return false;
    }
    auto* context = region.byteOffset.getContext();
    SmallVector<Value> symbols(region.symbols);
    SmallVector<AffineExpr> replacements;
    for (auto symbol : origin.symbols) {
        auto found = llvm::find(symbols, symbol);
        auto id = found - symbols.begin();
        if (found == symbols.end()) {
            symbols.push_back(symbol);
        }
        replacements.push_back(getAffineSymbolExpr(id, context));
    }
    SyncAccessRegion relative = region;
    relative.base = {};
    relative.byteOffset = simplifyAffineExpr(
        region.byteOffset - origin.byteOffset.replaceSymbols(replacements), region.extents.size(), symbols.size());
    // A fixed footprint must be independent of every loop/parameter symbol.
    for (unsigned i = 0; i < symbols.size(); ++i) {
        if (relative.byteOffset.isFunctionOfSymbol(i)) {
            return false;
        }
        for (auto extent : relative.extents) {
            if (extent.isFunctionOfSymbol(i)) {
                return false;
            }
        }
    }
    relative.symbols.clear();
    return mlir::pto::detail::materializeRegion(relative, space, ranges);
}
} // namespace

std::optional<SlotRanges> withinSlotRanges(const SyncStorageEffect& effect,
                                                           const SyncInput& input, uint64_t bytes)
{
    if (effect.regions.empty()) {
        return std::nullopt;
    }
    auto origin = slotOrigin(effect, input);
    if (!origin) {
        return std::nullopt;
    }
    SmallVector<SyncStorageCell> ranges;
    for (const auto& region : effect.regions) {
        SmallVector<SyncStorageCell> piece;
        if (!relativeRanges(region, *origin, effect.memory->scope, piece)) {
            return std::nullopt;
        }
        llvm::append_range(ranges, piece);
    }
    llvm::sort(ranges, [](const auto& a, const auto& b) { return a.begin < b.begin; });
    SlotRanges normalized;
    for (const auto& range : ranges) {
        if (range.end > bytes || range.begin > range.end) {
            return std::nullopt;
        }
        if (range.begin == range.end) {
            continue;
        }
        const bool touches = !normalized.empty() && range.begin <= normalized.back().second;
        if (touches) {
            normalized.back().second = std::max(normalized.back().second, range.end);
        } else {
            normalized.push_back({range.begin, range.end});
        }
    }
    return normalized;
}

} // namespace mlir::pto::frontiersynch::detail
