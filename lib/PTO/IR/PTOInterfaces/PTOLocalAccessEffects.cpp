// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Instruction-family contracts for plain local vector tiles. Descriptor guards
// are checked at the access, so mutable valid shapes cannot reuse stale type
// metadata. Unsupported effects retain the shared allocation bound.
static pto::TileBufType localAccessTile(Value value, bool allowBFloat = false)
{
    if (!plainVectorAccessTile(value, 32)) { return {}; }
    auto type = cast<pto::TileBufType>(value.getType());
    auto element = type.getElementType();
    auto valid = type.getValidShape();
    const bool floating = element.isF16() || element.isF32() || (allowBFloat && element.isBF16());
    if (!floating ||
        valid.size() != 2 ||
        valid[0] <= 0 || valid[1] <= 0 || valid[0] > type.getShape()[0] || valid[1] > type.getShape()[1]) {
        return {};
    }
    return type;
}

static pto::TileBufType pointwiseAccessTile(Value value)
{
    auto type = localAccessTile(value);
    if (!type) { return {}; }
    // A3 arithmetic row-repeat strides narrow to eight-bit block counts;
    // A5's flattened path narrows its total vector count to uint16_t.
    const auto width = pto::linearAccessBytes(type.getElementType());
    const auto strideBlocks = type.getShape()[1] * width / 32;
    const auto valid = type.getValidShape();
    const uint64_t vectors = (static_cast<uint64_t>(valid[0]) * valid[1] * width + 255) / 256;
    if (strideBlocks > UINT8_MAX || vectors > UINT16_MAX) { return {}; }
    return type;
}

static void appendLocalShapeGuard(SmallVectorImpl<int64_t>& guards, OpOperand& operand)
{
    auto valid = cast<pto::TileBufType>(operand.get().getType()).getValidShape();
    guards.append({static_cast<int64_t>(operand.getOperandNumber()), valid[0], valid[1]});
}

static bool matchingLocalShapes(pto::TileBufType source, pto::TileBufType destination)
{
    return source && destination && source.getValidShape() == destination.getValidShape();
}

static bool addLocalPointwiseAccess(PTOEffectList& effects, ArrayRef<OpOperand*> sources,
                                    OpOperand& destination)
{
    auto dst = pointwiseAccessTile(destination.get());
    if (!dst) { return false; }
    SmallVector<int64_t> guards;
    appendLocalShapeGuard(guards, destination);
    for (auto* operand : sources) {
        auto src = operand ? pointwiseAccessTile(operand->get()) : pto::TileBufType{};
        const bool compatible = matchingLocalShapes(src, dst) && src.getElementType() == dst.getElementType();
        if (!compatible) { return false; }
        appendLocalShapeGuard(guards, *operand);
    }
    const bool a3 = pto::getTargetArch(destination.getOwner()) == pto::PTOArch::A3;
    const auto rowBytes = dst.getValidShape()[1] * pto::linearAccessBytes(dst.getElementType());
    for (auto* operand : sources) {
        // A3 masks the arithmetic memory operands. A5 loads full vectors before
        // predicated computation/stores; a tail read retains its enclosing bound.
        if (a3 || rowBytes % 256 == 0) {
            addCheckedVectorAccess(effects, *operand, *operand, MemoryEffects::Read::get(), guards);
        } else {
            addEffect(effects, operand, MemoryEffects::Read::get());
        }
    }
    addCheckedVectorAccess(effects, destination, destination, MemoryEffects::Write::get(), guards);
    return true;
}

static void addPointwiseEffects(PTOEffectList& effects, ArrayRef<OpOperand*> sources, OpOperand& destination)
{
    const bool selected = addLocalPointwiseAccess(effects, sources, destination) ||
                          addAlignedPointwiseAccessEffects(effects, sources, destination);
    if (selected) {
        return;
    }
    for (auto* source : sources) { addEffect(effects, source, MemoryEffects::Read::get()); }
    addEffect(effects, &destination, MemoryEffects::Write::get());
}

static bool addConversionAccess(pto::TCvtOp op, PTOEffectList& effects)
{
    auto src = localAccessTile(op.getSrc(), true);
    auto dst = localAccessTile(op.getDst(), true);
    const bool compatible = matchingLocalShapes(src, dst) &&
                            (src.getElementType().isF32() != dst.getElementType().isF32());
    if (!compatible) { return false; }
    // Ordinary f16/bf16 <-> f32 conversion. Both sides cover whole vector batches;
    // packed, integer, and tail forms require different per-operand contracts.
    const auto columns = src.getValidShape()[1];
    const bool fullBatches = columns * pto::linearAccessBytes(src.getElementType()) % 256 == 0 &&
                             columns * pto::linearAccessBytes(dst.getElementType()) % 256 == 0;
    const uint64_t elements = static_cast<uint64_t>(columns) * src.getValidShape()[0];
    const auto vectors = (elements + 63) / 64;
    if (!fullBatches || vectors > UINT16_MAX) {
        return false;
    }
    SmallVector<int64_t> guards;
    appendLocalShapeGuard(guards, op.getSrcMutable());
    appendLocalShapeGuard(guards, op.getDstMutable());
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    addA2A3ScratchEffects(effects, op.getOperation(), op.getTmpMutable());
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}

static bool addLocalMoveAccess(pto::TMovOp op, PTOEffectList& effects)
{
    const bool specialForm = op.getFp() || op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
                             op.getGrpAxisAttr() || op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (specialForm) {
        return false;
    }
    auto src = localAccessTile(op.getSrc());
    auto dst = localAccessTile(op.getDst());
    const bool compatible = matchingLocalShapes(src, dst) && src.getElementType() == dst.getElementType();
    if (!compatible) { return false; }
    const auto bytes = src.getValidShape()[1] * pto::linearAccessBytes(src.getElementType());
    const bool a3 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A3;
    // A3 uses rounded 32-byte DMA blocks, with a uint16_t contiguous length.
    const auto blocks = static_cast<uint64_t>(bytes) * src.getValidShape()[0] / 32;
    if (a3 && (bytes % 32 || blocks > UINT16_MAX)) {
        return false;
    }
    SmallVector<int64_t> guards;
    appendLocalShapeGuard(guards, op.getSrcMutable());
    appendLocalShapeGuard(guards, op.getDstMutable());
    if (a3 || bytes % 256 == 0) {
        addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    } else {
        addEffect(effects, &op.getSrcMutable(), MemoryEffects::Read::get());
    }
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}

static bool addSingleRowSumAccess(pto::TRowSumOp op, PTOEffectList& effects)
{
    const bool a3 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A3;
    if (!a3) { return false; }
    auto src = localAccessTile(op.getSrc());
    auto dst = localAccessTile(op.getDst());
    const bool compatible = src && dst && src.getElementType() == dst.getElementType();
    if (!compatible) { return false; }
    auto valid = src.getValidShape();
    auto result = dst.getValidShape();
    if (valid[0] != 1 || result[0] != 1 || result[1] != 1 ||
        valid[1] * pto::linearAccessBytes(src.getElementType()) > 256) {
        return false;
    }
    // The A3 single-repeat floating reduction masks its source and writes one
    // result. Preserve existing scratch effects independently of these exports.
    SmallVector<int64_t> guards;
    appendLocalShapeGuard(guards, op.getSrcMutable());
    appendLocalShapeGuard(guards, op.getDstMutable());
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    addA2A3ScratchEffects(effects, op.getOperation(), op.getTmpMutable());
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}
