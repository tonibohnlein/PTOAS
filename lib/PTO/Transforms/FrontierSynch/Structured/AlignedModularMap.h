// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Certify one original shared map as a signed modular origin plus fixed displacement.
#ifndef PTO_FRONTIERSYNCH_ALIGNEDMODULARMAP_H
#define PTO_FRONTIERSYNCH_ALIGNEDMODULARMAP_H
#include "AlignedModularCells.h"
#include "../../InsertSync/SyncEffectRanges.h"
#include "../../InsertSync/SyncRegionArithmetic.h"
#include "PTO/Transforms/FrontierSynch/Shared/ModularScalar.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
namespace mlir::pto::frontiersynch::detail {
struct AlignedModularMap {
    Value base;
    Value origin;
    unsigned width;
    unsigned alignmentBits;
    uint64_t byteStride;
    AlignedModularAccess access;
    llvm::APInt ordinalStride;
};

struct AlignedModularMapGroup {
    Value base;
    Value origin;
    AlignedModularPartition partition;
};
// Callers must supply every access to this writing base in the selected word.
// This helper checks numerical identity; the owning producer proves completeness.
inline std::optional<AlignedModularMapGroup> partitionAlignedModularMaps(
    llvm::ArrayRef<AlignedModularMap> maps, uint64_t maximumFragments,
    std::function<void(uint64_t)> attempted = {})
{
    if (maps.empty() || maps.size() > maximumFragments || maximumFragments > UINT32_MAX) {
        return std::nullopt;
    }
    const auto& first = maps.front();
    unsigned bits = first.alignmentBits;
    for (const auto& map : maps) {
        if (map.base != first.base || map.origin != first.origin || map.width != first.width ||
            map.byteStride != first.byteStride || !map.ordinalStride.isZero()) { return std::nullopt; }
        bits = std::min(bits, map.alignmentBits);
    }
    std::vector<AlignedModularAccess> accesses;
    accesses.reserve(maps.size());
    for (const auto& map : maps) {
        if (attempted) { attempted(1 + map.access.intervals.size()); }
        accesses.push_back(map.access);
    }
    auto partition = partitionAlignedModularCells(first.width, bits, first.byteStride, accesses,
        maximumFragments, std::move(attempted));
    if (!partition) { return std::nullopt; }
    return AlignedModularMapGroup{first.base, first.origin, std::move(*partition)};
}

// Bound the existing linear rectangle materializer before it allocates slices.
// One contiguous dimension is sufficient; additional coalescing only lowers work.
inline std::optional<uint64_t> alignedMaterializationBound(const SyncAccessRegion& region, uint64_t limit)
{
    auto* context = region.byteOffset.getContext();
    SmallVector<AffineExpr> coordinates(region.extents.size(), getAffineConstantExpr(0, context));
    auto at = [&](ArrayRef<AffineExpr> point) {
        return dyn_cast<AffineConstantExpr>(simplifyAffineExpr(region.byteOffset.replaceDims(point), 0, 0));
    };
    auto origin = at(coordinates);
    if (!origin || origin.getValue() < 0) { return std::nullopt; }
    AffineExpr rebuilt = origin;
    int contiguous = -1;
    SmallVector<uint64_t> extents;
    for (unsigned i = 0; i < region.extents.size(); ++i) {
        auto extent = dyn_cast<AffineConstantExpr>(region.extents[i]);
        if (!extent || extent.getValue() < 0) { return std::nullopt; }
        if (!extent.getValue()) { return uint64_t{0}; }
        extents.push_back(static_cast<uint64_t>(extent.getValue()));
        coordinates[i] = getAffineConstantExpr(1, context);
        auto unit = at(coordinates);
        coordinates[i] = getAffineConstantExpr(0, context);
        if (!unit) { return std::nullopt; }
        auto stride = llvm::APInt(128, unit.getValue(), true) - llvm::APInt(128, origin.getValue(), true);
        if (!stride.isSignedIntN(64) || stride.isNegative()) { return std::nullopt; }
        const auto value = stride.getSExtValue();
        auto term = mlir::pto::detail::checkedMul(getAffineDimExpr(i, context), getAffineConstantExpr(value, context));
        rebuilt = mlir::pto::detail::checkedAdd(rebuilt, term);
        if (!rebuilt) { return std::nullopt; }
        if (contiguous < 0 && static_cast<uint64_t>(value) == region.elementBytes) { contiguous = i; }
    }
    auto negative = mlir::pto::detail::checkedMul(rebuilt, getAffineConstantExpr(-1, context));
    auto difference = mlir::pto::detail::checkedAdd(region.byteOffset, negative);
    auto zero = difference ? dyn_cast<AffineConstantExpr>(simplifyAffineExpr(difference, region.extents.size(), 0)) :
        AffineConstantExpr{};
    if (!zero || zero.getValue()) { return std::nullopt; }
    uint64_t count = 1;
    for (unsigned i = 0; i < extents.size(); ++i) {
        if (static_cast<int>(i) == contiguous) { continue; }
        if (extents[i] > limit / count) { return std::nullopt; }
        count *= extents[i];
    }
    return count <= limit ? std::optional<uint64_t>(count) : std::nullopt;
}

// Fixed values must be exact original-width machine values at this occurrence.
// No origin equality is inferred from expression similarity or pointer aliasing.
class AlignedModularMapBuilder {
public:
    using Constant = llvm::function_ref<std::optional<llvm::APInt>(Value)>;
    AlignedModularMapBuilder(scf::ForOp loop, ModularScalar& scalar, Constant constant, bool allowOrdinal = false)
        : loop(loop), scalar(scalar), constant(constant), allowOrdinal(allowOrdinal) {}

    std::optional<AlignedModularMap> build(const SyncAccessRegion& source, uint64_t maximumRanges = UINT32_MAX,
        std::function<void(uint64_t)> attempted = {})
    {
        if (attempted) { attempted(1); }
        if (!isCanonicalGMBase(source.base) || !source.byteOffset || !source.elementBytes) {
            return std::nullopt;
        }
        SmallVector<AffineExpr> dimensions, symbols;
        for (unsigned i = 0; i < source.extents.size(); ++i) {
            dimensions.push_back(getAffineDimExpr(i, source.byteOffset.getContext()));
        }
        std::optional<Origin> origin;
        for (auto symbol : source.symbols) {
            if (auto fixed = checkedConstant(symbol)) {
                symbols.push_back(getAffineConstantExpr(fixed->getSExtValue(), source.byteOffset.getContext()));
            } else {
                if (origin) { return std::nullopt; }
                origin = decompose(symbol, 0);
                if (!origin) { return std::nullopt; }
                symbols.push_back(getAffineSymbolExpr(0, source.byteOffset.getContext()));
            }
        }
        if (!origin || (!origin->value && (!allowOrdinal || origin->stride.isZero()))) { return std::nullopt; }
        auto map = mlir::pto::detail::substitute(source.byteOffset, dimensions, symbols);
        if (!map) { return std::nullopt; }
        auto split = mlir::pto::detail::splitTranslation(map, dimensions.size(), 1, 0);
        if (!split || split->second <= 0) { return std::nullopt; }
        SyncAccessRegion residual;
        residual.byteOffset = split->first;
        residual.elementBytes = source.elementBytes;
        for (auto extent : source.extents) {
            auto fixed = mlir::pto::detail::substitute(extent, dimensions, symbols);
            if (!fixed || fixed.isFunctionOfSymbol(0)) { return std::nullopt; }
            residual.extents.push_back(fixed);
        }
        auto bound = alignedMaterializationBound(residual, maximumRanges);
        if (!bound) { return std::nullopt; }
        // This is a conservative charged slice count, not measured allocation work.
        if (attempted) { attempted(*bound); }
        SmallVector<SyncStorageCell> ranges;
        if (!mlir::pto::detail::materializeRegion(residual, AddressSpace::GM, ranges)) {
            return std::nullopt;
        }
        const auto originBits = origin->value ? alignment(origin->value, 0) : origin->offset.getBitWidth();
        const auto bits = std::min({originBits, origin->offset.countTrailingZeros(),
                                    origin->stride.countTrailingZeros()});
        // A whole-circle segment would cross the signed representative cut.
        const auto alignmentBits = std::min(bits, origin->offset.getBitWidth() - 1);
        AlignedModularMap result{source.base, origin->value, origin->offset.getBitWidth(),
            alignmentBits, static_cast<uint64_t>(split->second), {origin->offset, {}}, origin->stride};
        for (const auto& range : ranges) { result.access.intervals.emplace_back(range.begin, range.end); }
        const auto granularity = uint64_t{1} << result.alignmentBits;
        if (result.byteStride > UINT64_MAX / granularity) { return std::nullopt; }
        const auto span = granularity * result.byteStride;
        for (auto interval : result.access.intervals) {
            if (interval.first > interval.second || interval.second > span) { return std::nullopt; }
        }
        return result;
    }

private:
    struct Origin {
        Value value;
        llvm::APInt offset;
        llvm::APInt stride;
    };
    std::optional<llvm::APInt> checkedConstant(Value value)
    {
        auto found = constants.find(value);
        if (found != constants.end()) { return found->second; }
        auto result = outerDependent(value, 0) ? std::nullopt : constant(value);
        const auto width = scalar.width(value.getType());
        if (!result || !width || width > 64 || result->getBitWidth() != width) { result = std::nullopt; }
        constants.try_emplace(value, result);
        return result;
    }
    bool outerDependent(Value value, unsigned depth)
    {
        auto found = dependencies.find(value);
        if (found != dependencies.end()) { return found->second; }
        bool dependent = false;
        if (!loop.isDefinedOutsideOfLoop(value)) {
            if (auto argument = dyn_cast<BlockArgument>(value)) {
                auto enclosing = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
                dependent = !enclosing || enclosing == loop || argument != enclosing.getInductionVar();
            } else if (auto* operation = value.getDefiningOp()) {
                dependent = depth > 128 || llvm::any_of(operation->getOperands(), [&](Value operand) {
                    return outerDependent(operand, depth + 1);
                });
            } else {
                dependent = true;
            }
        }
        dependencies.try_emplace(value, dependent);
        return dependent;
    }
    std::optional<Origin> decompose(Value value, unsigned depth)
    {
        auto found = decompositions.find(value);
        if (found != decompositions.end()) { return found->second; }
        auto result = decomposeValue(value, depth);
        decompositions.try_emplace(value, result);
        return result;
    }
    std::optional<Origin> combine(Origin left, Origin right, bool subtract)
    {
        if (right.value && (subtract || left.value)) { return std::nullopt; }
        if (right.value) { left.value = right.value; }
        left.offset += subtract ? -right.offset : right.offset;
        left.stride += subtract ? -right.stride : right.stride;
        return left;
    }
    std::optional<Origin> decomposeValue(Value value, unsigned depth)
    {
        const auto width = scalar.width(value.getType());
        if (!width || width > 64 || depth > 64) { return std::nullopt; }
        if (auto fixed = checkedConstant(value)) { return Origin{{}, *fixed, llvm::APInt(width, 0)}; }
        if (allowOrdinal && value == loop.getInductionVar()) {
            llvm::APInt lower, step;
            if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
                !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
                lower.getBitWidth() != width || step.getBitWidth() != width) { return std::nullopt; }
            return Origin{{}, lower, step};
        }
        if (loop.isDefinedOutsideOfLoop(value)) {
            return Origin{value, llvm::APInt(width, 0), llvm::APInt(width, 0)};
        }
        auto* op = value.getDefiningOp();
        if (!op || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op) ||
            scalar.width(op->getOperand(0).getType()) != width ||
            scalar.width(op->getOperand(1).getType()) != width) { return std::nullopt; }
        auto left = decompose(op->getOperand(0), depth + 1);
        auto right = decompose(op->getOperand(1), depth + 1);
        if (!left || !right) { return std::nullopt; }
        if (!isa<arith::MulIOp>(op)) { return combine(*left, *right, isa<arith::SubIOp>(op)); }
        if (left->value || right->value) { return std::nullopt; }
        if (!left->stride.isZero() && !right->stride.isZero()) { return std::nullopt; }
        return Origin{{}, left->offset * right->offset,
            left->stride * right->offset + right->stride * left->offset};
    }
    unsigned alignment(Value value, unsigned depth)
    {
        auto found = alignments.find(value);
        if (found != alignments.end()) { return found->second; }
        const auto result = computeAlignment(value, depth);
        alignments.try_emplace(value, result);
        return result;
    }
    unsigned computeAlignment(Value value, unsigned depth)
    {
        const auto width = scalar.width(value.getType());
        if (!width || width > 64 || depth > 64) { return 0; }
        if (auto fixed = checkedConstant(value)) { return fixed->countTrailingZeros(); }
        auto* op = value.getDefiningOp();
        if (!op || !isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op) ||
            scalar.width(op->getOperand(0).getType()) != width ||
            scalar.width(op->getOperand(1).getType()) != width) { return 0; }
        const auto left = alignment(op->getOperand(0), depth + 1);
        const auto right = alignment(op->getOperand(1), depth + 1);
        return isa<arith::MulIOp>(op) ? std::min(width, left + right) : std::min(left, right);
    }
    scf::ForOp loop;
    ModularScalar& scalar;
    Constant constant;
    bool allowOrdinal;
    DenseMap<Value, std::optional<Origin>> decompositions;
    DenseMap<Value, std::optional<llvm::APInt>> constants;
    DenseMap<Value, unsigned> alignments;
    DenseMap<Value, bool> dependencies;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
