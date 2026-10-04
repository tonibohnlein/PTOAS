// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Access selections carried by the existing MemoryEffectOpInterface.
#ifndef PTO_IR_PTOACCESSREGION_H
#define PTO_IR_PTOACCESSREGION_H
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::pto {
// Exact access coordinates in the affected operand. Dimensions range over the
// shape operand's valid extents, or static capacity when capacity is true.
// Map symbols refer to scalar operands of this same operation, by operand index.
// This describes an access, not the allocation or an overwrite of its padding.
// Missing contracts retain the original read/write declaration as unresolved.
inline DictionaryAttr makeAccessRegion(OpOperand& shape, bool capacity,
                                       AffineMap coordinates, ArrayRef<int64_t> symbols = {})
{
    Builder builder(shape.getOwner()->getContext());
    return builder.getDictionaryAttr({
        builder.getNamedAttr("pto.access_region", builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("shape_operand", builder.getI64IntegerAttr(shape.getOperandNumber())),
        builder.getNamedAttr("capacity", builder.getBoolAttr(capacity)),
        builder.getNamedAttr("coordinates", AffineMapAttr::get(coordinates)),
        builder.getNamedAttr("symbol_operands", builder.getDenseI64ArrayAttr(symbols))});
}

inline void addAccessRegion(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects, OpOperand& operand,
    MemoryEffects::Effect* mode, DictionaryAttr region)
{
    effects.emplace_back(mode, &operand, region);
}
} // namespace mlir::pto
#endif
