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
// A globally read-only GM base needs no iteration-coordinate normalization.
// Check whole pointer origins, not same-iteration disjointness: two translated
// subregions of one base can overlap on different visits.
Value canonicalOrigin(const SyncStorageEffect& effect)
{
    Value base;
    for (const auto& region : effect.regions) {
        if (!isCanonicalGMBase(region.base) || (base && base != region.base)) {
            return {};
        }
        base = region.base;
    }
    if (base) {
        return base;
    }
    // A descriptor map resolves its pointer origin even when an IV in its
    // offsets prevents use of the legacy all-operands provenance flag.
    if (effect.descriptorRegion && isCanonicalGMBase(effect.descriptorRegion->base)) {
        return effect.descriptorRegion->base;
    }
    return effect.sharedProvenanceComplete && isCanonicalGMBase(effect.memory->rootBuffer) ?
        effect.memory->rootBuffer : Value{};
}
bool readOnlyOrigin(const SyncInput& input, const SyncStorageEffect& source)
{
    const auto base = canonicalOrigin(source);
    if (source.mode != SyncAccessMode::Read || source.memory->scope != AddressSpace::GM || !base) {
        return false;
    }
    return llvm::all_of(input.accesses().effects(), [&](const SyncStorageEffect& other) {
        if (other.mode != SyncAccessMode::Write) {
            return true;
        }
        if (other.memory->scope == AddressSpace::Zero) {
            return false;
        }
        if (other.memory->scope != AddressSpace::GM) {
            return true;
        }
        const auto otherBase = canonicalOrigin(other);
        SmallVector<SyncStorageCell> domains{{AddressSpace::GM, 0, 1, base},
                                            {AddressSpace::GM, 0, 1, otherBase}};
        return otherBase && otherBase != base &&
            storageBasesAreComparable(domains, input.memory().gmPolicy());
    });
}
std::optional<TemplateRegion> substitute(TemplateBuilder& builder, const SyncAccessRegion& source,
                                         bool specializeGeometry)
{
    SmallVector<AffineExpr> dimensions, symbols;
    SmallVector<Value> invariants;
    for (unsigned i = 0; i < source.extents.size(); ++i) {
        dimensions.push_back(getAffineDimExpr(i, builder.context()));
    }
    for (auto value : source.symbols) {
        auto expression = builder.scalar(value, source.base ? &invariants : nullptr, false, specializeGeometry);
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
    if (phase->macroOpInstanceId >= 0) {
        output.result.note(RecognitionIssue::MultiplePhases, phase->elementOp, true);
        return false;
    }
    TemplatePayload payload;
    payload.phase = phase;
    payload.coordinates = path;
    for (auto id : input.accesses().effectsFor(phase)) {
        const auto& source = input.accesses().effects()[id];
        TemplateEffect effect;
        effect.sourceEffect = id;
        effect.mode = source.mode;
        if (source.regions.empty() && source.rangesMaterialized) {
            effect.ranges = source.ranges;
        }
        for (const auto& original : source.regions) {
            const bool preserveGlobal = output.geometryPolicy == TemplateGeometryPolicy::PreserveGlobalCoordinates &&
                source.memory->scope == AddressSpace::GM;
            auto map = substitute(*this, original, !preserveGlobal);
            if (!map && readOnlyOrigin(input, source)) {
                effect.regions.clear();
                effect.ranges.clear();
                effect.discharge = TemplateDischarge::ReadOnlyBase;
                break;
            }
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
