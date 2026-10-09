// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Shared selections for ordinary ND/DN transfers and aligned pointwise operations.
// GM and local effects are qualified separately: a byte-counted GM transfer
// remains exact when the local DMA fringe needs an enclosing allocation bound.
static bool plainVectorAccessTile(Value value, unsigned alignment, bool allowColumnMajor = false)
{
    auto type = dyn_cast<pto::TileBufType>(value.getType());
    auto space = type ? dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace()) : pto::AddressSpaceAttr{};
    if (!type || !space || space.getAddressSpace() != pto::AddressSpace::VEC ||
        type.getShape().size() != 2 || value.getDefiningOp<pto::SubViewOp>() ||
        (type.getBLayoutValueI32() != static_cast<int>(pto::BLayout::RowMajor) &&
         (!allowColumnMajor || type.getBLayoutValueI32() != static_cast<int>(pto::BLayout::ColMajor))) ||
        type.getSLayoutValueI32() != static_cast<int>(pto::SLayout::NoneBox) ||
        type.getCompactModeI32() != static_cast<int>(pto::CompactMode::Null) || type.getPadValueI32() != 0) {
        return false;
    }
    const auto width = pto::linearAccessBytes(type.getElementType());
    auto shape = type.getShape();
    const unsigned inner = type.getBLayoutValueI32() == static_cast<int>(pto::BLayout::RowMajor) ? 1 : 0;
    const unsigned outer = 1 - inner;
    return width && width <= 4 && shape[outer] > 0 && shape[outer] < 4096 &&
        shape[inner] > 0 && shape[inner] <= 65535 && (shape[inner] * width) % alignment == 0;
}

// The native ND/DN DMA consumes rank-two view dimensions and a uint32_t byte
// gap. Qualification keeps every narrowed field representable. The GM side
// uses the byte count even for short rows; local fringes are handled separately.
static bool vectorTransferView(Value value, Value tile, SmallVectorImpl<int64_t>& shape)
{
    auto layout = getLogicalViewLayout(value);
    if (!layout ||
        (*layout != pto::Layout::ND && *layout != pto::Layout::DN) ||
        !getLogicalViewShape(value, shape) || shape.size() != 2) {
        return false;
    }
    auto type = cast<pto::TileBufType>(tile.getType());
    const int64_t width = pto::linearAccessBytes(type.getElementType());
    if (shape[0] <= 0 || shape[0] > type.getShape()[0] || shape[1] <= 0 ||
        shape[1] > type.getShape()[1]) {
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
    const unsigned inner = type.getBLayoutValueI32() == static_cast<int>(pto::BLayout::RowMajor) ? 1 : 0;
    const unsigned outer = 1 - inner;
    auto contiguous = getConstIndexValue(view.getStrides()[inner]);
    auto stride = getConstIndexValue(view.getStrides()[outer]);
    const auto expectedLayout = inner == 1 ? pto::Layout::ND : pto::Layout::DN;
    // Singleton dimensions admit both ND and DN. The consumer selects the
    // orientation, while the descriptor still determines the same GM bytes.
    SmallVector<int64_t> baseShape;
    const bool singleton = getLogicalViewShape(base, baseShape) && baseShape.size() == 2 && baseShape[outer] == 1;
    if (*layout != expectedLayout && !singleton) {
        return false;
    }
    if (!viewType ||
        viewType.getElementType() != type.getElementType() || !contiguous ||
        *contiguous != 1 || !stride || *stride < 0 || *stride > INT32_MAX) {
        return false;
    }
    // A single burst never consumes the inter-burst gap.
    return shape[outer] == 1 || (*stride >= shape[inner] && (*stride - shape[inner]) <= UINT32_MAX / width);
}

static void addCheckedVectorAccess(PTOEffectList& effects, OpOperand& operand, OpOperand& domain,
                                   MemoryEffects::Effect* mode, ArrayRef<int64_t> conditions)
{
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, operand.getOwner()->getContext());
    pto::addAccessRegion(effects, operand, mode, pto::makeCheckedAccessRegion(domain, identity, conditions));
}

// Both A2/A3 and A5 ND/DN transfers use the valid contiguous extent
// times sizeof(T) GM bytes per burst. Local stride/padding rules differ for
// short bursts. Export the exact GM rectangle independently; do not assert a
// local overwrite of padding.
static void addVectorTransferAccess(PTOEffectList& effects, OpOperand& global, OpOperand& local,
                                    ArrayRef<int64_t> shape, bool load)
{
    SmallVector<int64_t> conditions{static_cast<int64_t>(local.getOperandNumber()), shape[0], shape[1]};
    MemoryEffects::Effect* read = MemoryEffects::Read::get();
    MemoryEffects::Effect* write = MemoryEffects::Write::get();
    auto* globalMode = load ? read : write;
    auto* localMode = load ? write : read;
    addCheckedVectorAccess(effects, global, local, globalMode, conditions);
    auto type = cast<pto::TileBufType>(local.get().getType());
    const unsigned inner = type.getBLayoutValueI32() == static_cast<int>(pto::BLayout::RowMajor) ? 1 : 0;
    const auto burstBytes = shape[inner] * pto::linearAccessBytes(type.getElementType());
    if (burstBytes % 32 == 0) {
        addCheckedVectorAccess(effects, local, local, localMode, conditions);
    } else {
        addEffect(effects, &local, localMode);
    }
}

static bool addVectorLoadAccessEffects(pto::TLoadOp op, PTOEffectList& effects)
{
    const bool supportedTile = plainVectorAccessTile(op.getDst(), 32, true);
    if (!supportedTile ||
        op.getPadModeAttr() || op.getPadValue() ||
        op.getLeftPaddingNum() || op.getRightPaddingNum() || op.getInitOutBuffer() || op.getInitCondition() ||
        op.getOffset() || (op.getCachePolicyAttr() &&
                           op.getCachePolicyAttr().getValue() == pto::LoadCachePolicy::L2Bypass)) {
        return false;
    }
    SmallVector<int64_t> shape;
    if (!vectorTransferView(op.getSrc(), op.getDst(), shape)) {
        return false;
    }
    addVectorTransferAccess(effects, op.getSrcMutable(), op.getDstMutable(), shape, true);
    return true;
}

static bool addVectorStoreAccessEffects(pto::TStoreOp op, PTOEffectList& effects)
{
    const bool supportedTile = plainVectorAccessTile(op.getSrc(), 32, true);
    if (!supportedTile ||
        op.getFp() || op.getPreQuantScalar() ||
        op.getStPhase() != pto::STPhase::Unspecified || op.getAtomicType() != pto::AtomicType::AtomicNone) {
        return false;
    }
    SmallVector<int64_t> shape;
    if (!vectorTransferView(op.getDst(), op.getSrc(), shape)) {
        return false;
    }
    addVectorTransferAccess(effects, op.getDstMutable(), op.getSrcMutable(), shape, false);
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

#include "PTOLocalAccessEffects.cpp"
