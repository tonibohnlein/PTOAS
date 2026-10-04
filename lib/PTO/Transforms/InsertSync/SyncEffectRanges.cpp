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

void applyAccessCoverage(const SyncInput& input, SyncStorageEffect& effect,
                         ArrayRef<SyncMemoryEffect> declarations)
{
    effect.region.reset();
    effect.ranges.clear();
    effect.cells.clear();
    effect.exactRanges = false;
    effect.precision = SyncAccessPrecision::Unknown;
    effect.precisionReason = "shared read/write interface does not specify an access region";
    auto aliases = input.buffers().find(effect.memory->baseBuffer);
    if (!effect.descriptorRegion || effect.phase->macroOpInstanceId >= 0 ||
        aliases == input.buffers().end() || aliases->second.size() != 1) {
        return;
    }
    bool matched = false;
    for (const auto& declared : declarations) {
        const bool read = isa<MemoryEffects::Read>(declared.getEffect());
        const bool write = isa<MemoryEffects::Write>(declared.getEffect());
        if (declared.getValue() != effect.memory->baseBuffer ||
            (effect.mode == SyncAccessMode::Read ? !read : !write)) {
            continue;
        }
        // Unknown parameter/resource semantics must not be interpreted as a
        // whole-buffer access, nor may one full declaration hide a partial one.
        if (!declared.getEffectOnFullRegion() || declared.getParameters() ||
            declared.getResource() != SideEffects::DefaultResource::get()) {
            return;
        }
        matched = true;
    }
    if (!matched) {
        return;
    }
    effect.region = effect.descriptorRegion;
    effect.precision = SyncAccessPrecision::Exact;
    effect.precisionReason.clear();
    effect.exactRanges = materializeRegion(*effect.region, effect.memory->scope, effect.ranges);
    if (!effect.exactRanges) {
        effect.ranges.clear();
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
