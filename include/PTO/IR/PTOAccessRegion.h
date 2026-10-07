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

// The selection is exact only when all current valid shapes match these
// (operand index, rows, columns) triples. Consumers resolve descriptor metadata
// at the accessing operation; an unknown or unequal shape rejects the entire
// selection rather than substituting the expected dimensions. Repeat the same
// preconditions on every effect of an operation whose execution requires them.
inline DictionaryAttr makeCheckedAccessRegion(OpOperand& shape, AffineMap coordinates,
                                              ArrayRef<int64_t> validShapes)
{
    NamedAttrList parameters(makeAccessRegion(shape, false, coordinates));
    parameters.set("valid_shapes", DenseI64ArrayAttr::get(shape.getOwner()->getContext(), validShapes));
    return parameters.getDictionary(shape.getOwner()->getContext());
}

// Version 2 supplies the access domain independently of another operand.
// Extents have no dimensions; their symbols and the selection's symbols both
// refer to symbol_operands. No extents means one access, not an empty access.
// coordinates selects logical elements; bytes selects offsets relative to the
// affected operand's physical base and uses byte_width bytes at each offset.
inline DictionaryAttr makeExplicitAccessRegion(OpOperand& operand, AffineMap extents,
                                               AffineMap selection, ArrayRef<int64_t> symbols = {},
                                               unsigned byteWidth = 0)
{
    Builder builder(operand.getOwner()->getContext());
    return builder.getDictionaryAttr({
        builder.getNamedAttr("pto.access_region", builder.getI64IntegerAttr(2)),
        builder.getNamedAttr("extents", AffineMapAttr::get(extents)),
        builder.getNamedAttr("coordinates", AffineMapAttr::get(selection)),
        builder.getNamedAttr("symbol_operands", builder.getDenseI64ArrayAttr(symbols)),
        builder.getNamedAttr("addressing", builder.getStringAttr(byteWidth ? "bytes" : "coordinates")),
        builder.getNamedAttr("byte_width", builder.getI64IntegerAttr(byteWidth))});
}

inline DictionaryAttr makeByteAccessRegion(OpOperand& operand, AffineMap extents,
                                           AffineMap byteOffsets, unsigned byteWidth,
                                           ArrayRef<int64_t> symbols = {})
{
    NamedAttrList parameters(makeExplicitAccessRegion(operand, extents, byteOffsets, symbols, byteWidth));
    parameters.set("addressing", StringAttr::get(operand.getOwner()->getContext(), "bytes"));
    return parameters.getDictionary(operand.getOwner()->getContext());
}

// Descriptor effects preserve the ordinary read/write declaration for MLIR
// scheduling, but do not access the physical bytes described by the operand.
// Consumers must retain these operations and their scalar dependencies.
inline DictionaryAttr makeDescriptorEffect(MLIRContext* context)
{
    Builder builder(context);
    return builder.getDictionaryAttr({builder.getNamedAttr(
        "pto.descriptor_state", builder.getUnitAttr())});
}

inline bool hasOnlyDescriptorEffects(Operation* operation)
{
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface || operation->getNumRegions()) { return false; }
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    if (effects.empty()) { return false; }
    for (const auto& effect : effects) {
        auto parameters = dyn_cast_or_null<DictionaryAttr>(effect.getParameters());
        if (!parameters || !parameters.getAs<UnitAttr>("pto.descriptor_state") ||
            !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect())) {
            return false;
        }
    }
    return true;
}

inline void addAccessRegion(
    SmallVectorImpl<MemoryEffects::EffectInstance>& effects, OpOperand& operand,
    MemoryEffects::Effect* mode, DictionaryAttr region)
{
    effects.emplace_back(mode, &operand, region);
}
} // namespace mlir::pto
#endif
