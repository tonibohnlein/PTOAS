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
static pto::TileBufType localAccessTile(Value value, pto::TileValidShapeQuery validShape, bool allowBFloat = false)
{
    if (!plainVectorAccessTile(value, 32)) { return {}; }
    auto type = cast<pto::TileBufType>(value.getType());
    auto current = validShape(value);
    if (!current || (*current)[0] <= 0 || (*current)[1] <= 0 ||
        (*current)[0] > type.getShape()[0] || (*current)[1] > type.getShape()[1]) { return {}; }
    // Reuse immutable type-based qualification without changing the original
    // descriptor type, allocation capacity, layout, element type or padding.
    type = pto::TileBufType::get(value.getContext(), type.getShape(), type.getElementType(),
        type.getMemorySpace(), *current, type.getConfigAttr());
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

static pto::TileBufType pointwiseAccessTile(Value value, pto::TileValidShapeQuery validShape)
{
    auto type = localAccessTile(value, validShape);
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

static void appendLocalShapeGuard(SmallVectorImpl<int64_t>& guards, OpOperand& operand, pto::TileBufType type)
{
    auto valid = type.getValidShape();
    guards.append({static_cast<int64_t>(operand.getOperandNumber()), valid[0], valid[1]});
}

static bool matchingLocalShapes(pto::TileBufType source, pto::TileBufType destination)
{
    return source && destination && source.getValidShape() == destination.getValidShape();
}

static bool addLocalPointwiseAccess(PTOEffectList& effects, ArrayRef<OpOperand*> sources,
                                    OpOperand& destination, pto::TileValidShapeQuery validShape)
{
    auto dst = pointwiseAccessTile(destination.get(), validShape);
    if (!dst) { return false; }
    SmallVector<int64_t> guards;
    appendLocalShapeGuard(guards, destination, dst);
    for (auto* operand : sources) {
        auto src = operand ? pointwiseAccessTile(operand->get(), validShape) : pto::TileBufType{};
        const bool compatible = matchingLocalShapes(src, dst) && src.getElementType() == dst.getElementType();
        if (!compatible) { return false; }
        appendLocalShapeGuard(guards, *operand, src);
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

static void addPointwiseEffects(PTOEffectList& effects, ArrayRef<OpOperand*> sources, OpOperand& destination,
                                pto::TileValidShapeQuery validShape)
{
    const bool selected = addLocalPointwiseAccess(effects, sources, destination, validShape) ||
                          addAlignedPointwiseAccessEffects(effects, sources, destination);
    if (selected) {
        return;
    }
    for (auto* source : sources) { addEffect(effects, source, MemoryEffects::Read::get()); }
    addEffect(effects, &destination, MemoryEffects::Write::get());
}

static bool addConversionAccess(pto::TCvtOp op, PTOEffectList& effects, pto::TileValidShapeQuery validShape)
{
    auto src = localAccessTile(op.getSrc(), validShape, true);
    auto dst = localAccessTile(op.getDst(), validShape, true);
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
    appendLocalShapeGuard(guards, op.getSrcMutable(), src);
    appendLocalShapeGuard(guards, op.getDstMutable(), dst);
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    addA2A3ScratchEffects(effects, op.getOperation(), op.getTmpMutable());
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}

static bool addLocalMoveAccess(pto::TMovOp op, PTOEffectList& effects, pto::TileValidShapeQuery validShape)
{
    const bool specialForm = op.getFp() || op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
                             op.getGrpAxisAttr() || op.getReluPreMode() != pto::ReluPreMode::NoRelu;
    if (specialForm) {
        return false;
    }
    auto src = localAccessTile(op.getSrc(), validShape);
    auto dst = localAccessTile(op.getDst(), validShape);
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
    appendLocalShapeGuard(guards, op.getSrcMutable(), src);
    appendLocalShapeGuard(guards, op.getDstMutable(), dst);
    if (a3 || bytes % 256 == 0) {
        addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    } else {
        addEffect(effects, &op.getSrcMutable(), MemoryEffects::Read::get());
    }
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}

static bool addSingleRowSumAccess(pto::TRowSumOp op, PTOEffectList& effects, pto::TileValidShapeQuery validShape)
{
    const bool a3 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A3;
    if (!a3) { return false; }
    auto src = localAccessTile(op.getSrc(), validShape);
    auto dst = localAccessTile(op.getDst(), validShape);
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
    appendLocalShapeGuard(guards, op.getSrcMutable(), src);
    appendLocalShapeGuard(guards, op.getDstMutable(), dst);
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), guards);
    addA2A3ScratchEffects(effects, op.getOperation(), op.getTmpMutable());
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), guards);
    return true;
}

// Operand roles are declared once here for both static and contextual queries.
// Nonlocal/special forms retain their existing ordinary interface fallback.
static bool dispatchLocalAccessEffects(Operation* operation, PTOEffectList& effects,
                                       pto::TileValidShapeQuery validShape)
{
    return TypeSwitch<Operation*, bool>(operation)
        .Case<pto::TAddOp, pto::TMulOp, pto::TSubOp>([&](auto op) {
            addPointwiseEffects(effects, {&op.getSrc0Mutable(), &op.getSrc1Mutable()}, op.getDstMutable(), validShape);
            return true;
        })
        .Case<pto::TAddSOp, pto::TExpOp, pto::TLogOp, pto::TSqrtOp>([&](auto op) {
            addPointwiseEffects(effects, {&op.getSrcMutable()}, op.getDstMutable(), validShape);
            return true;
        })
        .Case<pto::TMulSOp>([&](auto op) {
            addPointwiseEffects(effects, {&op.getSrc0Mutable()}, op.getDstMutable(), validShape);
            return true;
        })
        .Case<pto::TCvtOp>([&](auto op) { return addConversionAccess(op, effects, validShape); })
        .Case<pto::TMovOp>([&](auto op) { return addLocalMoveAccess(op, effects, validShape); })
        .Case<pto::TRowSumOp>([&](auto op) { return addSingleRowSumAccess(op, effects, validShape); })
        .Default(false);
}
bool mlir::pto::getContextualLocalAccessEffects(Operation* operation, PTOEffectList& effects,
                                               TileValidShapeQuery validShape)
{
    if (!operation || !validShape) { return false; }
    SmallVector<MemoryEffects::EffectInstance> pending;
    if (!dispatchLocalAccessEffects(operation, pending, validShape)) { return false; }
    llvm::append_range(effects, pending);
    return true;
}
static bool addDefaultLocalAccessEffects(Operation* operation, PTOEffectList& effects)
{
    return pto::getContextualLocalAccessEffects(operation, effects, [](Value value)
        -> std::optional<std::array<int64_t, 2>> {
        auto type = dyn_cast<pto::TileBufType>(value.getType());
        if (!type || type.getValidShape().size() != 2) { return std::nullopt; }
        return std::array<int64_t, 2>{type.getValidShape()[0], type.getValidShape()[1]};
    });
}
