// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Physical byte relations: fixed exact effects, or residue-selected scalar slots.
#include "ArithmeticProgramInternal.h"
#include "RotationPattern.h"
#include "../InsertSync/SyncEffectRanges.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
void emitRange(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect,
               const SyncStorageCell& range, std::optional<std::pair<unsigned, uint64_t>> filter = std::nullopt,
               uint64_t modulus = 1)
{
    if (range.begin >= range.end || range.end > static_cast<uint64_t>(INT64_MAX)) {
        builder.output.extraction.note(RecognitionIssue::UnknownGeometry, effect.phase->elementOp);
        return;
    }
    const auto& site = builder.output.sites[siteId];
    const unsigned depth = site.loops.size();
    auto relation = builder.relation(effect.mode == SyncAccessMode::Read ? PrimitiveKind::Reads : PrimitiveKind::Writes,
                                     depth + 1);
    relation.sourceSite = siteId;
    relation.sourceDimensions = depth;
    relation.storageSpace = range.space;
    relation.coordinates[depth] = {"byte", CoordinateKind::Storage};
    auto rows = builder.domain(site, 0);
    auto byte = getAffineDimExpr(depth, builder.context);
    rows.push_back(byte - static_cast<int64_t>(range.begin));
    rows.push_back(static_cast<int64_t>(range.end - 1) - byte);
    builder.emit(relation, rows, filter, modulus);
    builder.output.primitives.relations.push_back(std::move(relation));
}
void rotating(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect, const SyncInput& input)
{
    const auto& memory = *effect.memory;
    auto* anchor = effect.phase->elementOp;
    auto alias = input.buffers().find(memory.baseBuffer);
    auto multi = memory.rootBuffer ? dyn_cast<MultiTileBufType>(memory.rootBuffer.getType()) : MultiTileBufType{};
    if (!effect.selection || !multi || effect.selection->family != memory.rootBuffer ||
        alias == input.buffers().end() ||
        alias->second.size() != 1 || memory.aliasesUnknownRange) {
        builder.output.extraction.note(effect.precision == SyncAccessPrecision::Exact ?
                                       RecognitionIssue::SymbolicGeometry : RecognitionIssue::InexactFootprint, anchor);
        return;
    }
    const auto count = multi.getCount();
    if (!count || builder.limits.period % count) {
        builder.output.extraction.note(RecognitionIssue::ArithmeticPeriod, anchor, true);
        return;
    }
    auto slots = mlir::pto::detail::physicalSlotRanges(input, memory);
    if (slots.size() != count) {
        builder.output.extraction.note(RecognitionIssue::UnknownGeometry, anchor);
        return;
    }
    const auto& site = builder.output.sites[siteId];
    for (auto [dimension, storedLoop] : llvm::enumerate(site.loops)) {
        auto loop = storedLoop;
        auto pattern = matchSlot(effect.selection->selector, loop.getInductionVar(), count);
        if (!pattern || !pattern->arithmeticProven || pattern->stride != 1 % count || pattern->offset != 0) {
            continue;
        }
        for (auto [slot, range] : llvm::enumerate(slots)) {
            auto atom = withinSlotRange(effect, input, range.end - range.begin);
            if (!atom) {
                builder.output.extraction.note(effect.precision == SyncAccessPrecision::Exact ?
                    RecognitionIssue::WithinSlotFootprint : RecognitionIssue::InexactFootprint, anchor);
                return;
            }
            emitRange(builder, siteId, effect, {range.space, range.begin + atom->first, range.begin + atom->second},
                      std::make_pair(dimension, slot), count);
        }
        return;
    }
    builder.output.extraction.note(RecognitionIssue::SlotExpression, anchor, true);
}
}
void extractAccesses(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects)
{
    for (auto [siteId, site] : llvm::enumerate(builder.output.sites)) {
        for (auto id : effects.effectsFor(site.phase)) {
            const auto& effect = effects.effects()[id];
            if (effect.precision == SyncAccessPrecision::Exact && effect.exactRanges) {
                for (const auto& range : effect.ranges) {
                    emitRange(builder, siteId, effect, range);
                }
            } else {
                rotating(builder, siteId, effect, input);
            }
        }
    }
}
} // namespace mlir::pto::frontiersynch::detail
