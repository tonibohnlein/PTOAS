// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact physical byte relations from shared access maps. A symbolic origin and
// a fixed local byte union stay compact; unsupported local geometry is rejected.
#include "ArithmeticProgramInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
AffineExpr add(AffineExpr a, AffineExpr b)
{
    return mlir::pto::detail::checkedAdd(a, b);
}
AffineExpr negate(AffineExpr expression)
{
    return expression ? mlir::pto::detail::checkedMul(expression,
        getAffineConstantExpr(-1, expression.getContext())) : AffineExpr{};
}
void emitRange(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect,
               const SyncStorageCell& range, AffineExpr origin, Value base = {})
{
    const bool valid = range.begin < range.end && range.end <= static_cast<uint64_t>(INT64_MAX) && origin;
    if (!valid) {
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
    relation.storageBase = base ? base : range.base;
    relation.coordinates[depth] = {"byte", CoordinateKind::Storage};
    auto rows = builder.domain(site, 0);
    auto byte = getAffineDimExpr(depth, builder.context);
    auto begin = add(origin, getAffineConstantExpr(range.begin, builder.context));
    auto end = add(origin, getAffineConstantExpr(range.end - 1, builder.context));
    rows.push_back(add(byte, negate(begin)));
    rows.push_back(add(end, -byte));
    builder.emitForSites(relation, rows, {{&site, 0}});
    builder.output.primitives.relations.push_back(std::move(relation));
}
bool symbolicRegion(ProgramBuilder& builder, std::size_t siteId, const SyncStorageEffect& effect,
                    const SyncAccessRegion& region)
{
    if (!region.byteOffset) {
        return false;
    }
    SmallVector<AffineExpr> zeros(region.extents.size(), getAffineConstantExpr(0, builder.context));
    SmallVector<AffineExpr> symbols, occurrenceSymbols;
    for (auto [id, symbol] : llvm::enumerate(region.symbols)) {
        symbols.push_back(getAffineSymbolExpr(id, builder.context));
        auto value = builder.value(symbol, builder.output.sites[siteId], 0);
        if (!value) {
            return false;
        }
        occurrenceSymbols.push_back(value);
    }
    auto localOrigin = mlir::pto::detail::substitute(region.byteOffset, zeros, symbols);
    auto origin = mlir::pto::detail::substitute(region.byteOffset, zeros, occurrenceSymbols);
    auto offset = add(region.byteOffset, negate(localOrigin));
    if (!origin || !offset) {
        return false;
    }
    SyncAccessRegion local = region;
    // Materialize only the local byte union; preserve the physical base tag on
    // the resulting relations rather than pretending it is an absolute address.
    local.base = {};
    local.byteOffset = simplifyAffineExpr(offset, region.extents.size(), symbols.size());
    for (unsigned i = 0; i < symbols.size(); ++i) {
        if (local.byteOffset.isFunctionOfSymbol(i)) {
            return false;
        }
        for (auto extent : local.extents) {
            if (extent.isFunctionOfSymbol(i)) {
                return false;
            }
        }
    }
    local.symbols.clear();
    SmallVector<SyncStorageCell> pieces;
    if (!mlir::pto::detail::materializeRegion(local, effect.memory->scope, pieces)) {
        return false;
    }
    for (const auto& piece : pieces) {
        emitRange(builder, siteId, effect, piece, origin, region.base);
    }
    return true;
}
bool checkStorageBases(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects)
{
    SmallVector<SyncStorageCell> identities;
    for (const auto& site : builder.output.sites) {
        if (builder.staticallyEmpty(site)) {
            continue;
        }
        for (auto id : effects.effectsFor(site.phase)) {
            const auto& effect = effects.effects()[id];
            if (effect.exactRanges) { llvm::append_range(identities, effect.ranges); }
            for (const auto& region : effect.regions) {
                if (region.empty()) { continue; }
                if (region.base && (effect.memory->scope != AddressSpace::GM ||
                                    effect.memory->rootBuffer != region.base)) {
                    builder.output.extraction.note(RecognitionIssue::SymbolicGeometry, site.phase->elementOp);
                    return false;
                }
                // A one-byte witness checks the identity of a symbolic region;
                // this does not materialize or approximate its access domain.
                identities.push_back({effect.memory->scope, 0, 1, region.base});
            }
        }
    }
    if (!storageBasesAreComparable(identities, input.memory().gmPolicy())) {
        builder.output.extraction.note(RecognitionIssue::SymbolicGeometry,
                                      builder.output.sites.front().phase->elementOp);
        return false;
    }
    return true;
}
}
void extractAccesses(ProgramBuilder& builder, const SyncInput& input, const SyncStorageEffects& effects)
{
    if (!checkStorageBases(builder, input, effects)) {
        return;
    }
    for (auto [siteId, site] : llvm::enumerate(builder.output.sites)) {
        if (builder.staticallyEmpty(site)) {
            continue;
        }
        for (auto id : effects.effectsFor(site.phase)) {
            const auto& effect = effects.effects()[id];
            if (effect.precision != SyncAccessPrecision::Exact) {
                builder.output.extraction.note(RecognitionIssue::InexactFootprint, site.phase->elementOp);
                continue;
            }
            if (effect.exactRanges) {
                for (const auto& range : effect.ranges) {
                    emitRange(builder, siteId, effect, range, getAffineConstantExpr(0, builder.context));
                }
                continue;
            }
            if (effect.regions.empty()) {
                builder.output.extraction.note(RecognitionIssue::SymbolicGeometry, site.phase->elementOp);
            }
            for (const auto& region : effect.regions) {
                if (!symbolicRegion(builder, siteId, effect, region)) {
                    builder.output.extraction.note(RecognitionIssue::SymbolicGeometry, site.phase->elementOp);
                }
            }
        }
    }
}
} // namespace mlir::pto::frontiersynch::detail
