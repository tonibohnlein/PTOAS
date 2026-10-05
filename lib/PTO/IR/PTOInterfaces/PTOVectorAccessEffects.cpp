// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Shared exact selections for ordinary aligned ND transfers and pointwise
// vector operations. DMA fringes, padded loads, and A5 unmasked vector tail
// reads need different selections; the original effect declarations cover them.
static bool plainVectorAccessTile(Value value, unsigned alignment)
{
    auto type = dyn_cast<pto::TileBufType>(value.getType());
    auto space = type ? dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace()) : pto::AddressSpaceAttr{};
    if (!type || !space || space.getAddressSpace() != pto::AddressSpace::VEC ||
        type.getShape().size() != 2 || value.getDefiningOp<pto::SubViewOp>() ||
        type.getBLayoutValueI32() != static_cast<int>(pto::BLayout::RowMajor) ||
        type.getSLayoutValueI32() != static_cast<int>(pto::SLayout::NoneBox) ||
        type.getCompactModeI32() != static_cast<int>(pto::CompactMode::Null) || type.getPadValueI32() != 0) {
        return false;
    }
    const auto width = pto::linearAccessBytes(type.getElementType());
    auto shape = type.getShape();
    return width && width <= 4 && shape[0] > 0 && shape[0] < 4096 &&
        shape[1] > 0 && shape[1] <= 65535 && (shape[1] * width) % alignment == 0;
}

// The native ND DMA consumes rank-two view dimensions and a uint32_t byte
// gap. Qualification keeps every narrowed field representable and excludes
// padding, so the source and destination have the same logical rectangle.
static bool vectorAccessNdView(Value value, Value tile, SmallVectorImpl<int64_t>& shape)
{
    auto layout = getLogicalViewLayout(value);
    if (!layout || *layout != pto::Layout::ND || !getLogicalViewShape(value, shape) || shape.size() != 2) {
        return false;
    }
    auto type = cast<pto::TileBufType>(tile.getType());
    const int64_t width = pto::linearAccessBytes(type.getElementType());
    if (shape[0] <= 0 || shape[0] > type.getShape()[0] || shape[1] <= 0 ||
        shape[1] > type.getShape()[1] || shape[1] * width % 32 != 0) {
        return false;
    }
    Value base = value;
    while (auto part = base.getDefiningOp<pto::PartitionViewOp>()) {
        base = part.getSource();
    }
    auto view = base.getDefiningOp<pto::MakeTensorViewOp>();
    if (!view || view.getStrides().size() != 2) {
        return false;
    }
    auto viewType = dyn_cast<pto::TensorViewType>(view.getResult().getType());
    auto row = getConstIndexValue(view.getStrides()[0]);
    auto col = getConstIndexValue(view.getStrides()[1]);
    return viewType && viewType.getElementType() == type.getElementType() && row && col && *col == 1 &&
        *row >= shape[1] && *row <= INT32_MAX && (*row - shape[1]) <= UINT32_MAX / width;
}

static void addCheckedVectorAccess(PTOEffectList& effects, OpOperand& operand, OpOperand& domain,
                                   MemoryEffects::Effect* mode, ArrayRef<int64_t> conditions)
{
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, operand.getOwner()->getContext());
    pto::addAccessRegion(effects, operand, mode, pto::makeCheckedAccessRegion(domain, identity, conditions));
}

static bool addVectorLoadAccessEffects(pto::TLoadOp op, PTOEffectList& effects)
{
    if (!plainVectorAccessTile(op.getDst(), 32) || op.getPadModeAttr() || op.getPadValue() ||
        op.getLeftPaddingNum() || op.getRightPaddingNum() || op.getInitOutBuffer() || op.getInitCondition() ||
        op.getOffset() || (op.getCachePolicyAttr() &&
                           op.getCachePolicyAttr().getValue() == pto::LoadCachePolicy::L2Bypass)) {
        return false;
    }
    SmallVector<int64_t> shape;
    if (!vectorAccessNdView(op.getSrc(), op.getDst(), shape)) {
        return false;
    }
    SmallVector<int64_t> conditions{static_cast<int64_t>(op.getDstMutable().getOperandNumber()), shape[0], shape[1]};
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getDstMutable(), MemoryEffects::Read::get(), conditions);
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getDstMutable(), MemoryEffects::Write::get(), conditions);
    return true;
}

static bool addVectorStoreAccessEffects(pto::TStoreOp op, PTOEffectList& effects)
{
    if (!plainVectorAccessTile(op.getSrc(), 32) || op.getFp() || op.getPreQuantScalar() ||
        op.getStPhase() != pto::STPhase::Unspecified || op.getAtomicType() != pto::AtomicType::AtomicNone) {
        return false;
    }
    SmallVector<int64_t> shape;
    if (!vectorAccessNdView(op.getDst(), op.getSrc(), shape)) {
        return false;
    }
    SmallVector<int64_t> conditions{static_cast<int64_t>(op.getSrcMutable().getOperandNumber()), shape[0], shape[1]};
    addCheckedVectorAccess(effects, op.getSrcMutable(), op.getSrcMutable(), MemoryEffects::Read::get(), conditions);
    addCheckedVectorAccess(effects, op.getDstMutable(), op.getSrcMutable(), MemoryEffects::Write::get(), conditions);
    return true;
}

// TADD/TADDS share pointwise rectangle semantics. The 256-byte alignment also
// makes A5's unmasked vlds cover exactly the masked write domain (no tail).
static bool addAlignedPointwiseAccessEffects(PTOEffectList& effects, ArrayRef<OpOperand*> sources,
                                             OpOperand& destination)
{
    if (!plainVectorAccessTile(destination.get(), 256)) {
        return false;
    }
    auto type = cast<pto::TileBufType>(destination.get().getType());
    auto element = type.getElementType();
    if (!element.isF16() && !element.isF32() && !element.isInteger(16) && !element.isInteger(32)) {
        return false;
    }
    auto shape = type.getShape();
    // A5 flattens full tiles and narrows the vector repeat count to uint16_t.
    // Tile dimensions above were bounded before this widened multiplication.
    const uint64_t bytes = static_cast<uint64_t>(shape[0]) * shape[1] * pto::linearAccessBytes(element);
    // A2/A3's column-repeat path additionally uses an eight-bit repeat field.
    const uint64_t rowBytes = static_cast<uint64_t>(shape[1]) * pto::linearAccessBytes(element);
    if (bytes / 256 > UINT16_MAX || rowBytes / 256 > UINT8_MAX) {
        return false;
    }
    SmallVector<int64_t> conditions{static_cast<int64_t>(destination.getOperandNumber()), shape[0], shape[1]};
    for (auto* source : sources) {
        if (!source || !plainVectorAccessTile(source->get(), 256)) {
            return false;
        }
        auto sourceType = cast<pto::TileBufType>(source->get().getType());
        if (sourceType.getElementType() != element || sourceType.getShape() != shape) {
            return false;
        }
        conditions.append({static_cast<int64_t>(source->getOperandNumber()), shape[0], shape[1]});
    }
    for (auto* source : sources) {
        addCheckedVectorAccess(effects, *source, *source, MemoryEffects::Read::get(), conditions);
    }
    addCheckedVectorAccess(effects, destination, destination, MemoryEffects::Write::get(), conditions);
    return true;
}
