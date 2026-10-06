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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

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

std::optional<SyncSlotSelection> slotSelection(const SyncInput& input, Value operand)
{
    while (auto* operation = operand.getDefiningOp()) {
        if (auto selected = dyn_cast<MultiTileGetOp>(operation)) {
            auto family = input.buffers().find(selected.getSource());
            if (family == input.buffers().end() || family->second.size() != 1 ||
                !family->second.front()->hasKnownPhysicalAddresses) {
                return std::nullopt;
            }
            return SyncSlotSelection{selected.getSource(), selected.getSlot(), family->second.front()->baseAddresses};
        }
        auto view = dyn_cast<ViewLikeOpInterface>(operation);
        if (!view || view.getViewSource() == operand) {
            break;
        }
        operand = view.getViewSource();
    }
    return std::nullopt;
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

namespace {
void retainBufferBound(const SyncInput& input, SyncStorageEffect& effect)
{
    effect.region.reset();
    effect.regions.clear();
    effect.ranges.clear();
    effect.cells.clear();
    effect.rangesMaterialized = false;
    // The translator's ranges enclose accesses through this buffer, including
    // slot alternatives. A cached initial alias of a loop-carried pointer does
    // not enclose all visits, so it cannot be used as a bound here.
    if (effect.sharedProvenanceComplete) {
        effect.ranges = sharedRanges(*effect.memory);
    }
    effect.rangesMaterialized = !effect.ranges.empty();
    if (effect.selection) {
        // Preserve the selected allocation's bound, rather than forgetting
        // which slot is used merely because its selector varies in a loop.
        auto slots = physicalSlotRanges(input, *effect.memory);
        Value operand = effect.memory->baseBuffer;
        while (auto* op = operand.getDefiningOp()) {
            if (isa<MultiTileGetOp>(op)) { break; }
            auto view = dyn_cast<ViewLikeOpInterface>(op);
            if (!view || view.getViewSource() == operand) { break; }
            operand = view.getViewSource();
        }
        auto origin = resolveBufferRegion(input, operand, effect.phase->elementOp);
        if (origin && !slots.empty() && slots.front().end >= slots.front().begin) {
            const auto bytes = slots.front().end - slots.front().begin;
            if (bytes <= INT64_MAX && llvm::all_of(slots, [&](const auto& slot) {
                    return slot.end >= slot.begin && slot.end - slot.begin == bytes;
                })) {
                auto* context = effect.phase->elementOp->getContext();
                SmallVector<AffineExpr> zeros(origin->extents.size(), getAffineConstantExpr(0, context));
                origin->byteOffset = origin->byteOffset.replaceDims(zeros) + getAffineDimExpr(0, context);
                origin->extents = {getAffineConstantExpr(bytes, context)};
                origin->elementBytes = 1;
                SmallVector<Value> symbols;
                SmallVector<AffineExpr> replacements;
                for (auto [position, symbol] : llvm::enumerate(origin->symbols)) {
                    if (origin->byteOffset.isFunctionOfSymbol(position)) {
                        replacements.push_back(getAffineSymbolExpr(symbols.size(), context));
                        symbols.push_back(symbol);
                    } else { replacements.push_back(getAffineConstantExpr(0, context)); }
                }
                origin->byteOffset = origin->byteOffset.replaceSymbols(replacements);
                origin->symbols = std::move(symbols);
                effect.regions.push_back(*origin);
                effect.region = *origin;
                effect.ranges.clear();
                effect.rangesMaterialized = materializeRegion(*origin, effect.memory->scope, effect.ranges);
                return;
            }
        }
    }
    // An enclosing allocation interval is the supplied access model when no
    // narrower selection is available. It is not an instruction admission test.
    auto* context = effect.phase->elementOp->getContext();
    for (const auto& range : effect.ranges) {
        if (range.end > INT64_MAX) {
            // A partial affine union must never stand for the complete bound.
            effect.regions.clear();
            return;
        }
        SyncAccessRegion region;
        region.base = range.base;
        region.byteOffset = getAffineConstantExpr(range.begin, context) + getAffineDimExpr(0, context);
        region.extents.push_back(getAffineConstantExpr(range.end - range.begin, context));
        region.elementBytes = 1;
        effect.regions.push_back(std::move(region));
    }
    if (effect.regions.size() == 1) { effect.region = effect.regions.front(); }
}

std::optional<SyncAccessRegion> declaredRegion(const SyncInput& input, const SyncStorageEffect& effect,
                                              const SyncMemoryEffect& declaration)
{
    if (declaration.getResource() != SideEffects::DefaultResource::get()) {
        return std::nullopt;
    }
    auto parameters = declaration.getParameters();
    if (auto contract = dyn_cast_or_null<DictionaryAttr>(parameters)) {
        return resolveSelectedRegion(input, effect.memory->baseBuffer, effect.phase->elementOp, contract);
    }
    return declaration.getEffectOnFullRegion() && !parameters ? effect.descriptorRegion : std::nullopt;
}
} // namespace

void applyAccessCoverage(const SyncInput& input, SyncStorageEffect& effect,
                         ArrayRef<SyncMemoryEffect> declarations)
{
    retainBufferBound(input, effect);
    auto aliases = input.buffers().find(effect.memory->baseBuffer);
    if (effect.phase->macroOpInstanceId >= 0 ||
        aliases == input.buffers().end() || aliases->second.size() != 1) {
        return;
    }
    SmallVector<SyncAccessRegion> regions;
    for (const auto& declared : declarations) {
        const bool read = isa<MemoryEffects::Read>(declared.getEffect());
        const bool write = isa<MemoryEffects::Write>(declared.getEffect());
        if (declared.getValue() != effect.memory->baseBuffer ||
            (effect.mode == SyncAccessMode::Read ? !read : !write)) {
            continue;
        }
        auto region = declaredRegion(input, effect, declared);
        if (!region) {
            return; // An unresolved declaration must not be hidden by an exact one.
        }
        regions.push_back(std::move(*region));
    }
    if (regions.empty()) {
        return;
    }
    effect.regions = std::move(regions);
    effect.region.reset();
    if (effect.regions.size() == 1) {
        effect.region = effect.regions.front();
    }
    effect.ranges.clear();
    effect.rangesMaterialized = true;
    for (const auto& region : effect.regions) {
        SmallVector<SyncStorageCell> piece;
        if (!materializeRegion(region, effect.memory->scope, piece)) {
            effect.rangesMaterialized = false;
            effect.ranges.clear();
            break;
        }
        effect.ranges.append(piece);
    }
}

void resolveEffectRanges(const SyncInput& input, SyncStorageEffect& effect)
{
    const auto& memory = *effect.memory;
    if (!memory.rootBuffer || !memory.baseBuffer) {
        return;
    }
    effect.selection = slotSelection(input, memory.baseBuffer);
    effect.descriptorRegion = resolveBufferRegion(input, memory.baseBuffer, effect.phase->elementOp);
    applyAccessCoverage(input, effect, input.effectsFor(*effect.phase));
}
} // namespace mlir::pto::detail
