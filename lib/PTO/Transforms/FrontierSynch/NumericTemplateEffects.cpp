// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact substitution of shared access maps; no instruction footprint recovery.
#include "NumericTemplateInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
namespace mlir::pto::frontiersynch::detail {
namespace {
std::optional<TemplateRegion> substitute(TemplateBuilder& builder, const SyncAccessRegion& source)
{
    SmallVector<AffineExpr> dimensions, symbols;
    SmallVector<Value> invariants;
    for (unsigned i = 0; i < source.extents.size(); ++i) {
        dimensions.push_back(getAffineDimExpr(i, builder.context()));
    }
    for (auto value : source.symbols) {
        auto expression = builder.scalar(value, source.base ? &invariants : nullptr);
        if (!expression) {
            return std::nullopt;
        }
        symbols.push_back(expression);
    }
    auto replace = [&](AffineExpr expression) {
        auto result = mlir::pto::detail::substitute(expression, dimensions, symbols);
        return result ? simplifyAffineExpr(result, dimensions.size(), 1 + invariants.size()) : AffineExpr{};
    };
    TemplateRegion result{source.base, replace(source.byteOffset), {}, source.elementBytes, invariants};
    if (!result.byteOffset) {
        return std::nullopt;
    }
    for (auto extent : source.extents) {
        auto value = replace(extent);
        if (!value) {
            return std::nullopt;
        }
        result.extents.push_back(value);
    }
    return result;
}
bool localRanges(const TemplateRegion& map, AddressSpace space, SmallVectorImpl<SyncStorageCell>& result)
{
    const bool varying = map.byteOffset.isFunctionOfSymbol(0) || llvm::any_of(map.extents, [](AffineExpr extent) {
        return extent.isFunctionOfSymbol(0);
    });
    if (map.base || varying) {
        return false;
    }
    SyncAccessRegion local;
    local.byteOffset = map.byteOffset;
    local.extents = map.extents;
    local.elementBytes = map.elementBytes;
    return mlir::pto::detail::materializeRegion(local, space, result);
}
}
bool TemplateBuilder::payload(const CompoundInstanceElement* phase)
{
    TemplatePayload payload;
    payload.phase = phase;
    payload.coordinates = path;
    for (auto id : input.accesses().effectsFor(phase)) {
        const auto& source = input.accesses().effects()[id];
        if (source.precision != SyncAccessPrecision::Exact || source.regions.empty()) {
            output.result.note(RecognitionIssue::InexactFootprint, phase->elementOp);
            return false;
        }
        TemplateEffect effect;
        effect.sourceEffect = id;
        effect.mode = source.mode;
        for (const auto& original : source.regions) {
            auto map = substitute(*this, original);
            if (!map || !charge(1, output.fragments, output.limits.fragments, phase->elementOp)) {
                output.result.note(RecognitionIssue::SymbolicGeometry, phase->elementOp);
                return false;
            }
            if (source.memory->scope != AddressSpace::GM) {
                SmallVector<SyncStorageCell> ranges;
                if (!localRanges(*map, source.memory->scope, ranges)) {
                    output.result.note(RecognitionIssue::SymbolicGeometry, phase->elementOp);
                    return false;
                }
                if (!charge(ranges.size(), output.fragments, output.limits.fragments, phase->elementOp)) {
                    return false;
                }
                effect.ranges.append(ranges);
            }
            effect.regions.push_back(std::move(*map));
        }
        payload.effects.push_back(std::move(effect));
    }
    output.payloads.push_back(std::move(payload));
    return true;
}
} // namespace mlir::pto::frontiersynch::detail
