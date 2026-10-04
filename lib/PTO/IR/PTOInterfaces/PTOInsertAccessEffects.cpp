// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Included beside TINSERT's shared effects; describes the aligned VEC ND path.
// Native evidence: common/arch/memory/tinsert_common.hpp and a5/TInsert.hpp.
static bool alignedInsertTile(Value value)
{
    auto tile = dyn_cast<pto::TileBufType>(value.getType());
    auto space = tile ? dyn_cast_or_null<pto::AddressSpaceAttr>(tile.getMemorySpace()) : pto::AddressSpaceAttr{};
    if (!tile || !space || space.getAddressSpace() != pto::AddressSpace::VEC ||
        tile.getShape().size() != 2 || tile.getBLayoutValueI32() != static_cast<int>(pto::BLayout::RowMajor) ||
        tile.getSLayoutValueI32() != static_cast<int>(pto::SLayout::NoneBox) ||
        tile.getCompactModeI32() != static_cast<int>(pto::CompactMode::Null)) {
        return false;
    }
    auto width = pto::linearAccessBytes(tile.getElementType());
    auto allocation = value.getDefiningOp<pto::AllocTileOp>();
    return width && width <= 4 && allocation && allocation.getAddr() &&
        extractOffsetAligned(allocation.getAddr(), 32) &&
        llvm::all_of(tile.getShape(), [](int64_t size) { return size > 0 && size <= 65535; }) &&
        (tile.getShape()[1] * width) % 32 == 0;
}

static bool addInsertAccessEffects(pto::TInsertOp op, PTOEffectList& effects)
{
    if (op.getFp() || op.getPreQuantScalar() || op.getAccToVecModeAttr() || op.getTinsertModeAttr() ||
        op.getReluPreMode() != pto::ReluPreMode::NoRelu ||
        !alignedInsertTile(op.getSrc()) || !alignedInsertTile(op.getDst())) {
        return false;
    }
    auto src = cast<pto::TileBufType>(op.getSrc().getType());
    auto dst = cast<pto::TileBufType>(op.getDst().getType());
    auto valid = src.getValidShape();
    const int64_t width = pto::linearAccessBytes(src.getElementType());
    if (src.getElementType() != dst.getElementType() || valid.size() != 2 ||
        llvm::any_of(valid, [](int64_t size) { return size <= 0 || size > 65535; }) ||
        valid[0] > src.getShape()[0] || valid[1] > src.getShape()[1] ||
        src.getShape()[0] > dst.getShape()[0] || src.getShape()[1] > dst.getShape()[1] ||
        valid[1] * width % 32 || !extractOffsetAligned(op.getIndexCol(), 32 / width)) {
        return false;
    }
    // Native transfer lengths/gaps are uint16_t block counts. In the contiguous
    // case the total rectangle, not just one row, is narrowed to this field.
    const int64_t srcCols = src.getShape()[1], dstCols = dst.getShape()[1];
    const int64_t blocks = valid[1] * width / 32;
    if ((valid[1] == srcCols && srcCols == dstCols && valid[0] * blocks > 65535) ||
        blocks > 65535 || (srcCols - valid[1]) * width / 32 > 65535 ||
        (dstCols - valid[1]) * width / 32 > 65535) {
        return false;
    }
    auto* context = op.getContext();
    auto row = getAffineSymbolExpr(0, context) % 65536;
    auto col = getAffineSymbolExpr(1, context) % 65536;
    auto shifted = mlir::AffineMap::get(2, 2,
        {getAffineDimExpr(0, context) + row, getAffineDimExpr(1, context) + col}, context);
    auto source = pto::makeAccessRegion(op.getSrcMutable(), false, mlir::AffineMap::getMultiDimIdentityMap(2, context));
    auto destination = pto::makeAccessRegion(op.getSrcMutable(), false, shifted,
        {static_cast<int64_t>(op.getIndexRowMutable().getOperandNumber()),
         static_cast<int64_t>(op.getIndexColMutable().getOperandNumber())});
    pto::addAccessRegion(effects, op.getSrcMutable(), MemoryEffects::Read::get(), source);
    pto::addAccessRegion(effects, op.getDstMutable(), MemoryEffects::Write::get(), destination);
    return true;
}
