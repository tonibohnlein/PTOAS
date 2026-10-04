// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared byte selections for typed scalar and contiguous-vector accesses.
#ifndef PTO_IR_PTOLINEARACCESS_H
#define PTO_IR_PTOLINEARACCESS_H
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOAccessRegion.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::pto {
// Only types whose native object stride equals their complete byte width.
// In particular, odd-size LLVM vectors can have a larger allocation stride.
inline unsigned linearAccessBytes(Type type)
{
    uint64_t count = 1;
    if (auto vector = dyn_cast<VectorType>(type)) {
        if (vector.isScalable() || vector.getRank() != 1 || vector.getDimSize(0) <= 0 ||
            !llvm::isPowerOf2_64(vector.getDimSize(0))) {
            return 0;
        }
        count = static_cast<uint64_t>(vector.getDimSize(0));
        type = vector.getElementType();
    }
    if (!(type.isInteger(8) || type.isInteger(16) || type.isInteger(32) || type.isInteger(64) ||
          type.isF16() || type.isBF16() || type.isF32() || type.isF64())) {
        return 0;
    }
    auto width = type.getIntOrFloatBitWidth() / 8;
    if (count > std::numeric_limits<unsigned>::max() / width) {
        return 0;
    }
    return static_cast<unsigned>(count * width);
}

inline bool addLinearAccess(SmallVectorImpl<MemoryEffects::EffectInstance>& effects,
                            OpOperand& operand, OpOperand& offset, Type elementType,
                            MemoryEffects::Effect* mode, bool tileScalar)
{
    const auto width = linearAccessBytes(elementType);
    if (!width) {
        return false;
    }
    if (tileScalar) {
        auto tile = dyn_cast<TileBufType>(operand.get().getType());
        auto space = tile ? dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace()) : AddressSpaceAttr{};
        if (!tile || !space || space.getAddressSpace() != AddressSpace::VEC ||
            tile.getElementType() != elementType || isa<VectorType>(elementType)) {
            return false;
        }
    } else {
        auto pointer = dyn_cast<PtrType>(operand.get().getType());
        if (!pointer || pointer.getElementType() != elementType) {
            return false;
        }
    }
    auto* context = operand.getOwner()->getContext();
    auto index = getAffineSymbolExpr(0, context);
    // Tile::GetValue/SetValue accepts uint32_t, while pointer subscripts keep
    // the index type chosen by lowering. Both count objects, not bytes.
    auto nativeIndex = tileScalar ? index % INT64_C(4294967296) : index;
    auto domain = AffineMap::get(0, 1, {}, context);
    auto selection = AffineMap::get(0, 1, {nativeIndex * width}, context);
    auto region = makeByteAccessRegion(operand, domain, selection, width,
        {static_cast<int64_t>(offset.getOperandNumber())});
    addAccessRegion(effects, operand, mode, region);
    return true;
}
} // namespace mlir::pto
#endif
