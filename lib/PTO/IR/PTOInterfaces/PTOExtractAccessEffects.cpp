// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included beside the operation's existing read/write declarations. The
// synchronization consumer only sees coordinate maps, never operation names.
// Qualify the ordinary MAT->LEFT/RIGHT transfer; other native forms retain the
// existing read/write declaration until their semantics supply a selection.
static bool ordinaryExtractTile(Value value) {
  auto type = dyn_cast<pto::TileBufType>(value.getType());
  if (!type || !type.getElementType().isIntOrFloat()) {
    return false;
  }
  const auto width = type.getElementType().getIntOrFloatBitWidth();
  return (width == 16 || width == 32) && type.getShape().size() == 2 &&
      type.getShape()[0] > 0 && type.getShape()[0] % 16 == 0 &&
      type.getShape()[1] > 0 && type.getShape()[1] % 16 == 0 &&
      type.getCompactModeI32() == static_cast<int>(pto::CompactMode::Null) &&
      type.getSLayoutValueI32() != static_cast<int>(pto::SLayout::NoneBox) &&
      type.getSFractalSizeI32() == 512 &&
      !value.getDefiningOp<pto::SubViewOp>();
}

// Required alignment is a power of two. Add/multiply preserve these low zero
// bits even with machine wraparound (PTO index arithmetic is at least 32 bits).
static bool extractOffsetAligned(Value value, uint64_t alignment, unsigned depth = 0) {
  if (!value || depth > 32) {
    return false;
  }
  llvm::APInt constant;
  if (mlir::matchPattern(value, mlir::m_ConstantInt(&constant))) {
    return constant.countTrailingZeros() >= llvm::Log2_64(alignment);
  }
  if (auto argument = dyn_cast<mlir::BlockArgument>(value)) {
    auto loop = dyn_cast<mlir::scf::ForOp>(argument.getOwner()->getParentOp());
    // Executed induction values are lower + iteration * step. Both terms
    // must be aligned; the bound may be dynamic or give zero trips. Other
    // loop-carried arguments do not have this recurrence.
    return loop && argument == loop.getInductionVar() &&
        extractOffsetAligned(loop.getLowerBound(), alignment, depth + 1) &&
        extractOffsetAligned(loop.getStep(), alignment, depth + 1);
  }
  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    return extractOffsetAligned(multiply.getLhs(), alignment, depth + 1) ||
        extractOffsetAligned(multiply.getRhs(), alignment, depth + 1);
  }
  if (auto add = value.getDefiningOp<arith::AddIOp>()) {
    return extractOffsetAligned(add.getLhs(), alignment, depth + 1) &&
        extractOffsetAligned(add.getRhs(), alignment, depth + 1);
  }
  return false;
}

static bool addExtractAccessEffects(pto::TExtractOp op, PTOEffectList& effects) {
  if (op.getFp() || op.getPreQuantScalar() ||
      !ordinaryExtractTile(op.getSrc()) || !ordinaryExtractTile(op.getDst())) {
    return false;
  }
  auto source = cast<pto::TileBufType>(op.getSrc().getType());
  auto target = cast<pto::TileBufType>(op.getDst().getType());
  auto sourceSpace = dyn_cast<pto::AddressSpaceAttr>(source.getMemorySpace());
  auto targetSpace = dyn_cast<pto::AddressSpaceAttr>(target.getMemorySpace());
  if (!sourceSpace || !targetSpace || sourceSpace.getAddressSpace() != pto::AddressSpace::MAT ||
      (targetSpace.getAddressSpace() != pto::AddressSpace::LEFT &&
       targetSpace.getAddressSpace() != pto::AddressSpace::RIGHT) ||
      source.getElementType() != target.getElementType()) {
    return false;
  }
  bool a5 = pto::getTargetArch(op.getOperation()) == pto::PTOArch::A5;
  if (!a5 && (!extractOffsetAligned(op.getIndexRow(), 16) ||
              !extractOffsetAligned(op.getIndexCol(), 16))) {
    return false;
  }
  auto* context = op.getContext();
  auto row = getAffineSymbolExpr(0, context) % 65536;
  auto col = getAffineSymbolExpr(1, context) % 65536;
  // Native extraction converts offsets to uint16_t. A5 then drops the low
  // sub-block bits; express both operations in the shared coordinate map.
  if (a5) {
    bool innerRow = source.getSLayoutValueI32() == static_cast<int>(pto::SLayout::RowMajor);
    unsigned bytes = source.getElementType().getIntOrFloatBitWidth() / 8;
    int rowBlock = innerRow ? 16 : 32 / bytes;
    int colBlock = innerRow ? 32 / bytes : 16;
    row = row.floorDiv(rowBlock) * rowBlock;
    col = col.floorDiv(colBlock) * colBlock;
  }
  auto selection = mlir::AffineMap::get(2, 2,
      {getAffineDimExpr(0, context) + row, getAffineDimExpr(1, context) + col}, context);
  auto read = pto::makeAccessRegion(op.getDstMutable(), true, selection,
      {static_cast<int64_t>(op.getIndexRowMutable().getOperandNumber()),
       static_cast<int64_t>(op.getIndexColMutable().getOperandNumber())});
  auto write = pto::makeAccessRegion(op.getDstMutable(), true, mlir::AffineMap::getMultiDimIdentityMap(2, context));
  pto::addAccessRegion(effects, op.getSrcMutable(), MemoryEffects::Read::get(), read);
  pto::addAccessRegion(effects, op.getDstMutable(), MemoryEffects::Write::get(), write);
  return true;
}
