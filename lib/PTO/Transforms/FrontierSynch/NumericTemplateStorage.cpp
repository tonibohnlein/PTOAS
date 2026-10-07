// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Whole-base GM discharges and a common partition of absolute local storage.
#include "NumericTemplateInternal.h"
#include "DisjointTranslations.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include <map>
namespace mlir::pto::frontiersynch::detail {
namespace {
using EffectRef = std::pair<std::size_t, std::size_t>;
struct GlobalBase {
    SmallVector<EffectRef> effects;
    bool writes = false;
};
bool collectGlobal(TemplateBuilder& builder, DenseMap<Value, GlobalBase>& bases)
{
    for (auto [id, payload] : llvm::enumerate(builder.output.payloads)) {
        for (auto [effectId, effect] : llvm::enumerate(payload.effects)) {
            const auto& original = builder.input.accesses().effects()[effect.sourceEffect];
            if (original.memory->scope != AddressSpace::GM || effect.regions.empty()) {
                continue;
            }
            Value base;
            for (const auto& map : effect.regions) {
                auto argument = dyn_cast_or_null<BlockArgument>(map.base);
                auto function = argument ? dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp()) : func::FuncOp{};
                auto type = map.base ? dyn_cast<PtrType>(map.base.getType()) : PtrType{};
                const bool valid = function && type && type.getMemorySpace().getAddressSpace() == AddressSpace::GM &&
                    original.memory->rootBuffer == map.base && (!base || base == map.base);
                if (!valid) {
                    builder.output.result.note(RecognitionIssue::GMDischarge, payload.phase->elementOp);
                    return false;
                }
                base = map.base;
            }
            auto& summary = bases[base];
            summary.effects.push_back({id, effectId});
            summary.writes |= effect.mode == SyncAccessMode::Write;
        }
    }
    return true;
}
std::optional<int64_t> translation(const TemplateRegion& map, SyncAccessRegion& fixed)
{
    for (auto extent : map.extents) {
        if (extent.isFunctionOfSymbol(0)) {
            return std::nullopt;
        }
    }
    auto parts = mlir::pto::detail::splitTranslation(map.byteOffset, map.extents.size(),
                                                    1 + map.invariantSymbols.size(), 0);
    if (!parts) { return std::nullopt; }
    fixed.byteOffset = parts->first;
    fixed.extents = map.extents;
    fixed.elementBytes = map.elementBytes;
    return parts->second;
}
// Translation tests may cancel an unknown address origin only when every
// footprint of the writing payload has that same loop-invariant origin.
// Relative offsets and extents must then be concrete before interval checks.
bool relativeOrigin(const TemplateRegion& map, SyncAccessRegion& fixed,
                    std::optional<AffineExpr>& common, SmallVectorImpl<Value>& symbols)
{
    auto* context = map.byteOffset.getContext();
    SmallVector<AffineExpr> zeros(map.extents.size(), getAffineConstantExpr(0, context));
    auto origin = simplifyAffineExpr(fixed.byteOffset.replaceDims(zeros), 0, 1 + map.invariantSymbols.size());
    // Keep each region's constant displacement. Only the common symbolic
    // origin cancels; independent stores may cover different local columns.
    SmallVector<AffineExpr> zeroSymbols(1 + map.invariantSymbols.size(), getAffineConstantExpr(0, context));
    auto local = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(origin.replaceSymbols(zeroSymbols), 0, 0));
    if (!local || local.getValue() == INT64_MIN) { return false; }
    origin = simplifyAffineExpr(origin - local.getValue(), 0, 1 + map.invariantSymbols.size());
    if (common && (*common != origin || ArrayRef<Value>(symbols) != ArrayRef<Value>(map.invariantSymbols))) {
        return false;
    }
    common = origin;
    symbols.assign(map.invariantSymbols.begin(), map.invariantSymbols.end());
    auto relative = mlir::pto::detail::checkedAdd(fixed.byteOffset,
        mlir::pto::detail::checkedMul(origin, getAffineConstantExpr(-1, context)));
    if (!relative) {
        return false;
    }
    fixed.byteOffset = simplifyAffineExpr(relative, map.extents.size(), 1 + map.invariantSymbols.size());
    for (unsigned i = 0; i <= map.invariantSymbols.size(); ++i) {
        if (fixed.byteOffset.isFunctionOfSymbol(i) || llvm::any_of(fixed.extents, [i](AffineExpr extent) {
                return extent.isFunctionOfSymbol(i);
            })) {
            return false;
        }
    }
    return true;
}
bool discharge(TemplateBuilder& builder, const GlobalBase& base)
{
    std::optional<int64_t> commonStride;
    std::optional<AffineExpr> commonOrigin;
    SmallVector<Value> originSymbols;
    const bool relative = llvm::any_of(base.effects, [&](EffectRef ref) {
        return llvm::any_of(builder.output.payloads[ref.first].effects[ref.second].regions,
                           [](const TemplateRegion& map) { return !map.invariantSymbols.empty(); });
    });
    SmallVector<SyncStorageCell> unionRanges;
    struct Access { SyncStorageCell range; std::size_t payload; bool writes; };
    SmallVector<Access> accesses;
    for (auto [payloadId, effectId] : base.effects) {
        auto& effect = builder.output.payloads[payloadId].effects[effectId];
        if (!base.writes) {
            effect.discharge = TemplateDischarge::ReadOnlyBase;
            continue;
        }
        for (const auto& map : effect.regions) {
            SyncAccessRegion fixed;
            auto stride = translation(map, fixed);
            SmallVector<SyncStorageCell> ranges;
            if (!stride || (commonStride && *stride != *commonStride) ||
                (relative && !relativeOrigin(map, fixed, commonOrigin, originSymbols)) ||
                !mlir::pto::detail::materializeRegion(fixed, AddressSpace::GM, ranges) ||
                !builder.charge(ranges.size(), builder.output.fragments, builder.output.limits.fragments,
                                builder.output.payloads[payloadId].phase->elementOp)) {
                return false;
            }
            commonStride = stride;
            for (const auto& range : ranges) {
                unionRanges.push_back(range);
                accesses.push_back({range, payloadId, effect.mode == SyncAccessMode::Write});
            }
            effect.ranges.append(ranges);
        }
        effect.outerStride = commonStride.value_or(0);
        effect.discharge = TemplateDischarge::IterationPrivateBase;
    }
    if (!base.writes) {
        return true;
    }
    const auto stride = commonStride.value_or(0);
    const uint64_t distance = stride < 0 ? static_cast<uint64_t>(-(stride + 1)) + 1 : static_cast<uint64_t>(stride);
    // Discharging this base removes it from the internal scan as well. Prove
    // that distinct payloads have no within-visit conflicts before doing so.
    const bool multiplePayloads = !accesses.empty() && llvm::any_of(accesses, [&](const Access& access) {
        return access.payload != accesses.front().payload;
    });
    for (std::size_t i = 0; multiplePayloads && i < accesses.size(); ++i) {
        const auto& a = accesses[i];
        for (std::size_t j = 0; j < i; ++j) {
            const auto& b = accesses[j];
            if (a.payload != b.payload && (a.writes || b.writes) &&
                a.range.begin < b.range.end && b.range.begin < a.range.end) { return false; }
        }
    }
    uint64_t trips = UINT64_MAX;
    APInt upper;
    if (matchPattern(builder.output.outer.getUpperBound(), m_ConstantInt(&upper)) && upper.isSignedIntN(64)) {
        auto span = upper.sextOrTrunc(128) - APInt(128, builder.output.lower, true);
        trips = span.isStrictlyPositive() ?
            ((span - 1).udiv(APInt(128, builder.output.step)) + 1).getZExtValue() : 0;
    }
    return disjointTranslations(unionRanges, APInt(128, distance), trips);
}
bool partition(TemplateBuilder& builder)
{
    std::map<AddressSpace, std::map<uint64_t, int64_t>> endpoints;
    for (const auto& payload : builder.output.payloads) {
        for (const auto& effect : payload.effects) {
            if (effect.discharge != TemplateDischarge::None) {
                continue;
            }
            for (const auto& range : effect.ranges) {
                ++endpoints[range.space][range.begin];
                --endpoints[range.space][range.end];
            }
        }
    }
    std::map<AddressSpace, std::pair<std::size_t, std::size_t>> atomSpans;
    for (const auto& [space, points] : endpoints) {
        auto& span = atomSpans[space];
        span.first = builder.output.atoms.size();
        int64_t active = 0;
        uint64_t previous = 0;
        for (const auto& [point, change] : points) {
            if (active && previous < point) {
                if (!builder.charge(1, builder.output.fragments, builder.output.limits.fragments,
                                    builder.output.outer)) {
                    return false;
                }
                builder.output.atoms.push_back({space, previous, point});
            }
            active += change;
            previous = point;
        }
        span.second = builder.output.atoms.size();
    }
    for (auto& payload : builder.output.payloads) {
        for (auto& effect : payload.effects) {
            if (effect.discharge != TemplateDischarge::None) {
                continue;
            }
            for (const auto& range : effect.ranges) {
                const auto span = atomSpans.at(range.space);
                auto begin = builder.output.atoms.begin() + span.first;
                auto end = builder.output.atoms.begin() + span.second;
                auto atom = std::lower_bound(begin, end, range.begin, [](const SyncStorageCell& item, uint64_t point) {
                    return item.begin < point;
                });
                for (; atom != end && atom->begin < range.end; ++atom) {
                    if (!builder.charge(1, builder.output.fragments, builder.output.limits.fragments,
                                        payload.phase->elementOp)) {
                        return false;
                    }
                    effect.atoms.push_back(atom - builder.output.atoms.begin());
                }
            }
            llvm::sort(effect.atoms);
            effect.atoms.erase(std::unique(effect.atoms.begin(), effect.atoms.end()), effect.atoms.end());
        }
    }
    return true;
}
} // namespace
bool prepareTemplateEffects(TemplateBuilder& builder)
{
    DenseMap<Value, GlobalBase> bases;
    if (!collectGlobal(builder, bases)) {
        return false;
    }
    for (const auto& item : bases) {
        if (!discharge(builder, item.second)) {
            builder.output.result.note(RecognitionIssue::GMDischarge, builder.output.outer);
            return false;
        }
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (const auto& payload : builder.output.payloads) { phases.push_back(payload.phase); }
    if (builder.input.accesses().hasUniformRelationships(phases)) {
        for (uint32_t a = 0; a < builder.output.payloads.size(); ++a) {
            for (uint32_t b = a; b < builder.output.payloads.size(); ++b) {
                bool conflict = false;
                for (const auto& x : builder.output.payloads[a].effects) {
                    for (const auto& y : builder.output.payloads[b].effects) {
                        conflict |= builder.input.accesses().uniformConflict(x.sourceEffect, y.sourceEffect);
                    }
                }
                if (conflict) { builder.output.uniformConflicts.emplace_back(a, b); }
            }
        }
    }
    return partition(builder);
}
} // namespace mlir::pto::frontiersynch::detail
