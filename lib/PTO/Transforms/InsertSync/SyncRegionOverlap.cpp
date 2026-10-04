// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Concrete spans and conservative symbolic interval comparisons. Symbolic
// addresses remain exact descriptions when an overlap query is unresolved.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "SyncEffectRanges.h"
#include "SyncRegionArithmetic.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::pto {
namespace {
AffineExpr simplify(AffineExpr expression, const SyncAccessRegion& region)
{
    return expression ? simplifyAffineExpr(expression, region.extents.size(), region.symbols.size()) : AffineExpr{};
}
std::optional<int64_t> integer(AffineExpr expression)
{
    auto constant = dyn_cast_or_null<AffineConstantExpr>(expression);
    return constant ? std::optional<int64_t>(constant.getValue()) : std::nullopt;
}
std::optional<SmallVector<int64_t>> sizes(const SyncAccessRegion& region)
{
    SmallVector<int64_t> result;
    for (auto extent : region.extents) {
        auto size = integer(extent);
        if (!size || *size < 0) {
            return std::nullopt;
        }
        result.push_back(*size);
    }
    return result;
}
// An affine rectangular map has monotone extrema when all dimension strides
// are nonnegative constants. Nonlinear layout maps keep conservative overlap.
std::optional<std::pair<AffineExpr, AffineExpr>> bounds(const SyncAccessRegion& region)
{
    auto shape = sizes(region);
    if (!shape) {
        return std::nullopt;
    }
    SmallVector<int64_t> maximums;
    for (auto size : *shape) {
        maximums.push_back(std::max<int64_t>(1, size));
    }
    if (!detail::magnitude(region.byteOffset, maximums)) {
        return std::nullopt;
    }
    auto* context = region.byteOffset.getContext();
    SmallVector<AffineExpr> low(shape->size(), getAffineConstantExpr(0, context));
    auto begin = simplify(region.byteOffset.replaceDims(low), region);
    auto end = begin;
    auto reconstructed = begin;
    for (unsigned i = 0; i < shape->size(); ++i) {
        low[i] = getAffineConstantExpr(1, context);
        auto stride = integer(simplify(region.byteOffset.replaceDims(low) - begin, region));
        low[i] = getAffineConstantExpr(0, context);
        if (!stride || *stride < 0) {
            return std::nullopt;
        }
        int64_t distance = 0;
        if (llvm::MulOverflow(*stride, std::max<int64_t>(0, (*shape)[i] - 1), distance)) {
            return std::nullopt;
        }
        end = detail::checkedAdd(end, getAffineConstantExpr(distance, context));
        if (!end) {
            return std::nullopt;
        }
        reconstructed = reconstructed + getAffineDimExpr(i, context) * *stride;
    }
    if (integer(simplify(region.byteOffset - reconstructed, region)) != std::optional<int64_t>(0)) {
        return std::nullopt;
    }
    end = detail::checkedAdd(end, getAffineConstantExpr(region.elementBytes, context));
    return end ? std::optional<std::pair<AffineExpr, AffineExpr>>(std::make_pair(begin, end)) : std::nullopt;
}
} // namespace

bool regionsProvablyDisjoint(const SyncAccessRegion& a, const SyncAccessRegion& b)
{
    // A static phase can execute at different iteration coordinates. Equal SSA
    // symbols under repeated control do not denote equal dynamic values across
    // those occurrences. This query has no occurrence context, so do not cancel
    // them (the exact symbolic footprint itself is retained).
    auto invariant = [](Value value) {
        if (!value) {
            return true;
        }
        Operation* owner = value.getDefiningOp();
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            owner = argument.getOwner()->getParentOp();
        }
        for (auto* parent = owner; parent; parent = parent->getParentOp()) {
            if (isa<LoopLikeOpInterface>(parent)) {
                return false;
            }
        }
        return true;
    };
    if (!invariant(a.base) || !invariant(b.base) ||
        !llvm::all_of(a.symbols, invariant) || !llvm::all_of(b.symbols, invariant)) {
        return false;
    }
    if (a.empty() || b.empty()) {
        return true;
    }
    if (a.base != b.base) {
        return false;
    }
    auto aBounds = bounds(a), bBounds = bounds(b);
    if (!aBounds || !bBounds) {
        return false;
    }
    SmallVector<Value> symbols(a.symbols);
    SmallVector<AffineExpr> replacements;
    for (auto symbol : b.symbols) {
        auto found = llvm::find(symbols, symbol);
        if (found == symbols.end()) {
            symbols.push_back(symbol);
            found = std::prev(symbols.end());
        }
        replacements.push_back(getAffineSymbolExpr(found - symbols.begin(), a.byteOffset.getContext()));
    }
    auto delta = [&](AffineExpr left, AffineExpr right) {
        auto difference = detail::checkedAdd(left, detail::checkedMul(right.replaceSymbols(replacements),
            getAffineConstantExpr(-1, a.byteOffset.getContext())));
        return difference ? integer(simplifyAffineExpr(difference, 0, symbols.size())) : std::nullopt;
    };
    auto first = delta(aBounds->second, bBounds->first);
    auto second = delta(aBounds->first, bBounds->second);
    return (first && *first <= 0) || (second && *second >= 0);
}

namespace detail {
bool materializeRegion(const SyncAccessRegion& region, AddressSpace space,
                       SmallVectorImpl<SyncStorageCell>& result)
{
    if (region.empty()) {
        result.clear();
        return true;
    }
    if (region.base || !region.symbols.empty()) {
        return false;
    }
    auto shape = sizes(region);
    if (!shape || !region.elementBytes) {
        return false;
    }
    SmallVector<int64_t> maximums;
    for (auto size : *shape) {
        maximums.push_back(std::max<int64_t>(1, size));
    }
    if (!magnitude(region.byteOffset, maximums)) {
        return false;
    }
    auto* context = region.byteOffset.getContext();
    SmallVector<AffineExpr> coordinates(shape->size(), getAffineConstantExpr(0, context));
    auto at = [&]() { return integer(simplify(region.byteOffset.replaceDims(coordinates), region)); };
    SmallVector<SyncStorageCell> pending;
    // Collapse a physically contiguous dimension before visiting the remaining
    // slices. Large contiguous buffers therefore produce one interval.
    int contiguous = -1;
    auto origin = at();
    if (!origin || *origin < 0) {
        return false;
    }
    for (unsigned i = 0; i < shape->size(); ++i) {
        auto shifted = region.byteOffset.replaceDims([&] {
            SmallVector<AffineExpr> dims;
            for (unsigned j = 0; j < shape->size(); ++j) {
                auto d = getAffineDimExpr(j, context);
                dims.push_back(j == i ? d + 1 : d);
            }
            return dims;
        }());
        if (integer(simplify(shifted - region.byteOffset, region)) ==
            std::optional<int64_t>(region.elementBytes)) {
            contiguous = i;
            break;
        }
    }
    uint64_t span = region.elementBytes;
    if (contiguous >= 0) {
        if (static_cast<uint64_t>((*shape)[contiguous]) > UINT64_MAX / span) {
            return false;
        }
        span *= (*shape)[contiguous];
    }
    // Concrete cells are an optional view of an exact symbolic footprint.
    // Avoid expanding large blocked/strided buffers into one record per element.
    constexpr uint64_t maxMaterializedSlices = 4096;
    uint64_t slices = 1;
    for (unsigned i = 0; i < shape->size(); ++i) {
        if (static_cast<int>(i) != contiguous) {
            if (static_cast<uint64_t>((*shape)[i]) > maxMaterializedSlices / slices) {
                return false;
            }
            slices *= (*shape)[i];
        }
    }
    SmallVector<int64_t> indices(shape->size(), 0);
    bool done = false;
    while (!done) {
        auto begin = at();
        if (!begin || *begin < 0 || span > UINT64_MAX - static_cast<uint64_t>(*begin)) {
            return false;
        }
        pending.push_back({space, static_cast<uint64_t>(*begin), static_cast<uint64_t>(*begin) + span});
        done = true;
        for (unsigned i = 0; i < shape->size(); ++i) {
            if (static_cast<int>(i) == contiguous) {
                continue;
            }
            if (++indices[i] < (*shape)[i]) {
                coordinates[i] = getAffineConstantExpr(indices[i], context);
                done = false;
                break;
            }
            indices[i] = 0;
            coordinates[i] = getAffineConstantExpr(0, context);
        }
    }
    result.assign(pending.begin(), pending.end());
    return true;
}
} // namespace detail
} // namespace mlir::pto
