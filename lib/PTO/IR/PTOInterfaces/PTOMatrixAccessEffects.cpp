// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included beside the shared read/write declarations. These selections describe
// ordinary full matrix operands; consumers do not dispatch on operation names.
static bool fullMatrixAccessTile(Value value, pto::AddressSpace space,
                                 pto::BLayout block, pto::SLayout inner, int fractal)
{
    auto type = dyn_cast<pto::TileBufType>(value.getType());
    auto memory = type ? dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace()) : pto::AddressSpaceAttr{};
    if (!type || !memory || memory.getAddressSpace() != space || type.getShape().size() != 2 ||
        value.getDefiningOp<pto::SubViewOp>()) {
        return false;
    }
    auto shape = type.getShape();
    return shape[0] > 0 && shape[0] % 16 == 0 && shape[1] > 0 && shape[1] % 16 == 0 &&
        type.getPadValueI32() == 0 &&
        type.getCompactModeI32() == static_cast<int>(pto::CompactMode::Null) &&
        type.getBLayoutValueI32() == static_cast<int>(block) &&
        type.getSLayoutValueI32() == static_cast<int>(inner) && type.getSFractalSizeI32() == fractal;
}

static bool fullAccumulatorAccess(Value value)
{
    auto type = dyn_cast<pto::TileBufType>(value.getType());
    return type && type.getElementType().isF32() &&
        fullMatrixAccessTile(value, pto::AddressSpace::ACC, pto::BLayout::ColMajor, pto::SLayout::RowMajor, 1024);
}

// tload_common.hpp::TLoadGm2L1Nd2nz and tstore_common.hpp::TStoreAccNz2nd
// use rank-two ND coordinates and the row stride. Restrict the stride to the
// narrower uint16_t load field and require unit inner stride before exporting
// the original view map (rather than silently overlooking native truncation).
static bool matrixAccessNdView(Value value, ArrayRef<int64_t> expectedShape, Type element)
{
    SmallVector<int64_t> shape;
    auto layout = getLogicalViewLayout(value);
    if (!layout || *layout != pto::Layout::ND || !getLogicalViewShape(value, shape) ||
        ArrayRef<int64_t>(shape) != expectedShape) {
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
    auto type = dyn_cast<pto::TensorViewType>(view.getResult().getType());
    auto rows = getConstIndexValue(view.getStrides()[0]);
    auto cols = getConstIndexValue(view.getStrides()[1]);
    return type && type.getElementType() == element && rows && *rows > 0 && *rows <= 65535 && cols && *cols == 1;
}

// Geometry and instruction modes are checked above. Current valid extents are
// checked by the shared consumer at this instruction, including descriptor
// updates and constant extents carried behind dynamically typed operands.
static SmallVector<int64_t> fullMatrixConditions(ArrayRef<OpOperand*> operands)
{
    SmallVector<int64_t> conditions;
    for (auto* operand : operands) {
        auto shape = cast<pto::TileBufType>(operand->get().getType()).getShape();
        conditions.append({static_cast<int64_t>(operand->getOperandNumber()), shape[0], shape[1]});
    }
    return conditions;
}
static void addFullMatrixAccess(PTOEffectList& effects, OpOperand& operand, MemoryEffects::Effect* mode,
                                ArrayRef<int64_t> conditions)
{
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, operand.getOwner()->getContext());
    pto::addAccessRegion(effects, operand, mode, pto::makeCheckedAccessRegion(operand, identity, conditions));
}

// Native ND->NZ DMA reads the source rectangle and writes complete NZ blocks.
// Full 16-aligned shapes remove padding/fringe differences on A2/A3 and A5.
static bool addMatrixLoadAccessEffects(pto::TLoadOp op, PTOEffectList& effects)
{
    auto tile = dyn_cast<pto::TileBufType>(op.getDst().getType());
    if (!tile || (!tile.getElementType().isF16() && !tile.getElementType().isF32()) ||
        op.getPadModeAttr() || op.getPadValue() ||
        op.getLeftPaddingNum() || op.getRightPaddingNum() || op.getInitOutBuffer() || op.getInitCondition() ||
        op.getOffset() ||
        (op.getCachePolicyAttr() && op.getCachePolicyAttr().getValue() == pto::LoadCachePolicy::L2Bypass) ||
        !fullMatrixAccessTile(op.getDst(), pto::AddressSpace::MAT, pto::BLayout::ColMajor,
                              pto::SLayout::RowMajor, 512)) {
        return false;
    }
    auto shape = tile.getShape();
    if (shape[0] > 16384 || shape[1] > 65535 || !matrixAccessNdView(op.getSrc(), shape, tile.getElementType())) {
        return false;
    }
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, op.getContext());
    auto conditions = fullMatrixConditions({&op.getDstMutable()});
    pto::addAccessRegion(effects, op.getSrcMutable(), MemoryEffects::Read::get(),
                        pto::makeCheckedAccessRegion(op.getDstMutable(), identity, conditions));
    addFullMatrixAccess(effects, op.getDstMutable(), MemoryEffects::Write::get(), conditions);
    return true;
}

// TStoreAccNz2nd consumes valid rows/columns and performs the optional f32->f16
// conversion without changing the logical rectangle. The source layout is NZ
// with 16x16 f32 blocks; destination byte widths come from its own view map.
static bool addMatrixStoreAccessEffects(pto::TStoreOp op, PTOEffectList& effects)
{
    if (!fullAccumulatorAccess(op.getSrc()) || op.getFp() || op.getPreQuantScalar() ||
        op.getStPhase() != pto::STPhase::Unspecified || op.getAtomicType() != pto::AtomicType::AtomicNone) {
        return false;
    }
    auto tile = cast<pto::TileBufType>(op.getSrc().getType());
    auto view = dyn_cast<pto::PartitionTensorViewType>(op.getDst().getType());
    auto tensor = dyn_cast<pto::TensorViewType>(op.getDst().getType());
    Type element = view ? view.getElementType() : (tensor ? tensor.getElementType() : Type{});
    auto shape = tile.getShape();
    if (!element || (!element.isF16() && !element.isF32()) || shape[0] > 8192 || shape[1] > 4095 ||
        !matrixAccessNdView(op.getDst(), shape, element)) {
        return false;
    }
    auto conditions = fullMatrixConditions({&op.getSrcMutable()});
    addFullMatrixAccess(effects, op.getSrcMutable(), MemoryEffects::Read::get(), conditions);
    auto identity = mlir::AffineMap::getMultiDimIdentityMap(2, op.getContext());
    pto::addAccessRegion(effects, op.getDstMutable(), MemoryEffects::Write::get(),
                        pto::makeCheckedAccessRegion(op.getSrcMutable(), identity, conditions));
    return true;
}

// a2a3/TMatmul.hpp::TMATMUL_IMPL and a5/TMatmul.hpp::TMATMUL_IMPL use
// lhs valid M,K and rhs valid N. Full aligned, matching shapes make each native
// rectangle the corresponding operand's complete physical region. Both native
// ACC implementations read cOutMatrix; a distinct acc_in is not certified here.
static bool addMatrixMultiplyAccessEffects(PTOEffectList& effects, OpOperand& lhs, OpOperand& rhs,
                                           OpOperand& dst, OpOperand* accumulator)
{
    bool a5 = pto::getTargetArch(lhs.getOwner()) == pto::PTOArch::A5;
    if (!fullMatrixAccessTile(lhs.get(), pto::AddressSpace::LEFT,
            a5 ? pto::BLayout::ColMajor : pto::BLayout::RowMajor, pto::SLayout::RowMajor, 512) ||
        !fullMatrixAccessTile(rhs.get(), pto::AddressSpace::RIGHT,
                              pto::BLayout::RowMajor, pto::SLayout::ColMajor, 512) ||
        !fullAccumulatorAccess(dst.get()) || (accumulator && accumulator->get() != dst.get())) {
        return false;
    }
    auto left = cast<pto::TileBufType>(lhs.get().getType());
    auto right = cast<pto::TileBufType>(rhs.get().getType());
    auto result = cast<pto::TileBufType>(dst.get().getType());
    auto a = left.getShape();
    auto b = right.getShape();
    auto c = result.getShape();
    if ((!left.getElementType().isF16() && !left.getElementType().isF32()) ||
        right.getElementType() != left.getElementType() ||
        a[0] > 4095 || a[1] > 4095 || b[1] > 4095 || a[1] != b[0] || c[0] != a[0] || c[1] != b[1]) {
        return false;
    }
    auto conditions = fullMatrixConditions({&lhs, &rhs, &dst});
    if (accumulator) {
        addFullMatrixAccess(effects, *accumulator, MemoryEffects::Read::get(), conditions);
    }
    addFullMatrixAccess(effects, lhs, MemoryEffects::Read::get(), conditions);
    addFullMatrixAccess(effects, rhs, MemoryEffects::Read::get(), conditions);
    addFullMatrixAccess(effects, dst, MemoryEffects::Write::get(), conditions);
    return true;
}
