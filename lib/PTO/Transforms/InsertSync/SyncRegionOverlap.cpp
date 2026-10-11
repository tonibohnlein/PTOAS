// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Concrete spans, independent modular byte bands and invariant interval
// comparisons. Unresolved overlap preserves the original symbolic addresses.
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "SyncEffectRanges.h"
#include "SyncRegionArithmetic.h"
#include "SyncRegionDigits.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
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
// are nonnegative constants. With a residue period, the end instead encloses
// residue contributions after whole-period strides are removed. Nonlinear
// layout maps keep conservative overlap.
std::optional<std::pair<AffineExpr, AffineExpr>> bounds(const SyncAccessRegion& region, uint64_t residuePeriod = 0)
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
        // A complete row stride divisible by the residue period changes no
        // byte residue, independently of the dimension coordinate.
        if (residuePeriod && static_cast<uint64_t>(*stride) % residuePeriod == 0) { distance = 0; }
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

// The low bits hold independently for every valuation of every symbol. In
// particular, equal SSA values in different loop visits are never cancelled.
struct LowBits {
    unsigned count = 0;
    uint64_t residue = 0;
};
uint64_t bitMask(unsigned count)
{
    return count ? (uint64_t{1} << count) - 1 : 0;
}
LowBits combineBits(AffineExpr expression, const DenseMap<AffineExpr, LowBits>& known)
{
    if (auto value = integer(expression)) {
        return {63, static_cast<uint64_t>(*value) & bitMask(63)};
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
    if (!binary) { return {}; }
    if (expression.getKind() == AffineExprKind::Add) {
        auto a = known.lookup(binary.getLHS()), b = known.lookup(binary.getRHS());
        const auto count = std::min(a.count, b.count);
        return {count, (a.residue + b.residue) & bitMask(count)};
    }
    if (expression.getKind() != AffineExprKind::Mul) { return {}; }
    auto coefficient = integer(binary.getRHS());
    auto operand = binary.getLHS();
    if (!coefficient) { coefficient = integer(binary.getLHS()); operand = binary.getRHS(); }
    if (!coefficient) { return {}; }
    const auto multiplier = static_cast<uint64_t>(*coefficient);
    if (!multiplier) { return {63, 0}; }
    auto value = known.lookup(operand);
    const auto count = std::min<unsigned>(63, value.count + llvm::countr_zero(multiplier));
    return {count, (value.residue * multiplier) & bitMask(count)};
}
LowBits lowBits(AffineExpr expression)
{
    DenseMap<AffineExpr, LowBits> known;
    SmallVector<std::pair<AffineExpr, bool>> pending{{expression, false}};
    while (!pending.empty()) {
        auto [current, ready] = pending.pop_back_val();
        if (known.count(current)) { continue; }
        auto binary = dyn_cast<AffineBinaryOpExpr>(current);
        const bool combine = binary && (current.getKind() == AffineExprKind::Add ||
                                         current.getKind() == AffineExprKind::Mul);
        if (combine && !ready) {
            pending.push_back({current, true});
            pending.push_back({binary.getRHS(), false});
            pending.push_back({binary.getLHS(), false});
            continue;
        }
        known.try_emplace(current, combineBits(current, known));
    }
    return known.lookup(expression);
}
using ByteBand = std::pair<uint64_t, uint64_t>;
SmallVector<ByteBand, 2> residueBands(uint64_t begin, uint64_t span, uint64_t period)
{
    const auto end = begin + span; // Both operands are below period <= 2^63.
    if (end <= period) { return {{begin, end}}; }
    return {{begin, period}, {0, end - period}};
}
bool modularBandsDisjoint(const SyncAccessRegion& a, const SyncAccessRegion& b)
{
    if (a.base != b.base || !isCanonicalGMBase(a.base) || !a.byteOffset || !b.byteOffset ||
        a.byteOffset.getContext() != b.byteOffset.getContext() || !a.elementBytes || !b.elementBytes) {
        return false;
    }
    auto function = cast<func::FuncOp>(cast<BlockArgument>(a.base).getOwner()->getParentOp());
    const auto width = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) { return false; }
    auto first = bounds(a), second = bounds(b);
    if (!first || !second) { return false; }
    auto span = [](const auto& endpoints, const SyncAccessRegion& region) {
        auto negative = detail::checkedMul(endpoints.first, getAffineConstantExpr(-1, region.byteOffset.getContext()));
        auto difference = detail::checkedAdd(endpoints.second, negative);
        return integer(simplify(difference, region));
    };
    const auto bitsA = lowBits(first->first), bitsB = lowBits(second->first);
    const auto count = std::min(bitsA.count, bitsB.count);
    if (!count) { return false; }
    const uint64_t period = uint64_t{1} << count;
    // The checks above validate the original linear maps. Only their
    // modular enclosing spans omit whole-period dimension contributions.
    auto reducedA = bounds(a, period), reducedB = bounds(b, period);
    if (!reducedA || !reducedB) { return false; }
    auto spanA = span(*reducedA, a), spanB = span(*reducedB, b);
    if (!spanA || !spanB || *spanA <= 0 || *spanB <= 0) { return false; }
    if (static_cast<uint64_t>(*spanA) >= period || static_cast<uint64_t>(*spanB) >= period) { return false; }
    const auto firstBands = residueBands(bitsA.residue & (period - 1), *spanA, period);
    const auto secondBands = residueBands(bitsB.residue & (period - 1), *spanB, period);
    for (const auto& x : firstBands) {
        for (const auto& y : secondBands) {
            if (std::max(x.first, y.first) < std::min(x.second, y.second)) { return false; }
        }
    }
    return true;
}
} // namespace

bool isCanonicalGMBase(Value base)
{
    auto argument = dyn_cast_or_null<BlockArgument>(base);
    auto type = base ? dyn_cast<PtrType>(base.getType()) : PtrType();
    if (!argument || !type || type.getMemorySpace().getAddressSpace() != AddressSpace::GM) {
        return false;
    }
    auto function = dyn_cast<func::FuncOp>(argument.getOwner()->getParentOp());
    return function && !function.isDeclaration() && argument.getOwner() == &function.front();
}

bool regionsProvablyDisjoint(const SyncAccessRegion& a, const SyncAccessRegion& b)
{
    if (a.empty() || b.empty()) { return true; }
    if (modularBandsDisjoint(a, b)) { return true; }
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
    if (region.base) {
        if (space != AddressSpace::GM || !isCanonicalGMBase(region.base)) {
            return false;
        }
        // Materialize offsets with the existing finite mapper, then restore
        // their identity. Clearing a base is never an alias/disjointness proof.
        SyncAccessRegion relative = region;
        relative.base = {};
        SmallVector<SyncStorageCell> pieces;
        if (!materializeRegion(relative, space, pieces)) {
            return false;
        }
        for (auto& piece : pieces) { piece.base = region.base; }
        result.assign(pieces.begin(), pieces.end());
        return true;
    }
    if (!region.symbols.empty()) {
        return false;
    }
    if (materializeDigitRegion(region, space, result)) {
        return true;
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
