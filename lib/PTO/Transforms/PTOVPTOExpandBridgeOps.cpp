// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOEXPANDBRIDGEOPS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static pto::AddressSpaceAttr getPointerMemorySpace(Attribute memorySpace,
                                                   MLIRContext *ctx) {
  if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
    return addrSpace;
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return pto::AddressSpaceAttr::get(
        ctx, static_cast<pto::AddressSpace>(intAttr.getInt()));
  return pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
}

static Value materializeBufferPointer(Value value, PatternRewriter &rewriter,
                                      Location loc) {
  if (!value)
    return {};

  if (isa<pto::PtrType>(value.getType()))
    return value;

  auto memrefType = dyn_cast<MemRefType>(value.getType());
  if (!memrefType)
    return {};

  auto ptrType =
      pto::PtrType::get(rewriter.getContext(), memrefType.getElementType(),
                        getPointerMemorySpace(memrefType.getMemorySpace(),
                                              rewriter.getContext()));
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, value).getResult();
}

static Value offsetBufferPointer(Value basePtr, Type elementType,
                                 Value elementOffset,
                                 PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  Value offsetIndex = elementOffset;
  if (!offsetIndex.getType().isIndex())
    offsetIndex = rewriter.create<arith::IndexCastUIOp>(loc,
                                                        rewriter.getIndexType(),
                                                        elementOffset);
  return rewriter.create<pto::AddPtrOp>(loc, basePtr.getType(), basePtr,
                                        offsetIndex);
}

static bool isKnownOne(Value value) {
  APInt intValue;
  return value && matchPattern(value, m_ConstantInt(&intValue)) &&
         intValue.isOne();
}

static bool shouldRestoreDmaLoopSize(Value loop1Count, Value loop2Count) {
  if (!loop1Count)
    return false;
  return !isKnownOne(loop1Count) || !isKnownOne(loop2Count);
}

static SmallVector<pto::DmaLoopConfig> collectLoopConfigs(ValueRange counts,
                                                          ValueRange srcStrides,
                                                          ValueRange dstStrides) {
  SmallVector<pto::DmaLoopConfig> loops;
  loops.reserve(counts.size());
  for (auto [count, srcStride, dstStride] :
       llvm::zip(counts, srcStrides, dstStrides))
    loops.push_back({count, srcStride, dstStride});
  return loops;
}

static Value offsetPointerByBytes(Value basePtr, Value byteOffset,
                                  PatternRewriter &rewriter, Location loc) {
  if (!basePtr)
    return {};

  APInt constOffset;
  if (matchPattern(byteOffset, m_ConstantInt(&constOffset)) && constOffset.isZero())
    return basePtr;

  Value baseInt =
      rewriter.create<pto::CastPtrOp>(loc, rewriter.getI64Type(), basePtr);
  Value offsetI64 = byteOffset;
  if (!offsetI64.getType().isInteger(64))
    offsetI64 =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(),
                                              offsetI64);
  Value sum = rewriter.create<arith::AddIOp>(loc, baseInt, offsetI64);
  return rewriter.create<pto::CastPtrOp>(loc, basePtr.getType(), sum);
}

static Value buildAccumulatedByteOffset(Location loc, Value baseOffset,
                                        Value indexI64, Value stride,
                                        PatternRewriter &rewriter) {
  Value delta = rewriter.create<arith::MulIOp>(loc, indexI64, stride);
  return rewriter.create<arith::AddIOp>(loc, baseOffset, delta);
}

template <typename BodyBuilder>
static void buildSoftwareLoopNest(PatternRewriter &rewriter, Location loc,
                                  ArrayRef<pto::DmaLoopConfig> loops,
                                  Value srcOffset, Value dstOffset,
                                  BodyBuilder &&buildLeaf) {
  if (loops.empty()) {
    buildLeaf(srcOffset, dstOffset);
    return;
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value count = rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getIndexType(),
                                                      loops.front().count);
  scf::ForOp forOp = rewriter.create<scf::ForOp>(loc, c0, count, c1);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value ivI64 =
        rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(),
                                              forOp.getInductionVar());
    Value nextSrcOffset = buildAccumulatedByteOffset(
        loc, srcOffset, ivI64, loops.front().srcStride, rewriter);
    Value nextDstOffset = buildAccumulatedByteOffset(
        loc, dstOffset, ivI64, loops.front().dstStride, rewriter);
    buildSoftwareLoopNest(rewriter, loc, loops.drop_front(), nextSrcOffset,
                          nextDstOffset, buildLeaf);
  }
}

struct ExpandUvldPattern : public OpRewritePattern<pto::UvldOp> {
  using OpRewritePattern<pto::UvldOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::UvldOp op,
                                PatternRewriter &rewriter) const override {
    auto vecType = dyn_cast<pto::VRegType>(op.getResult().getType());
    if (!vecType)
      return failure();

    Value basePtr = materializeBufferPointer(op.getSource(), rewriter, op.getLoc());
    if (!basePtr)
      return op.emitOpError(
          "requires a recoverable pointer base for uvld expansion");

    Value loadPtr = offsetBufferPointer(basePtr, vecType.getElementType(),
                                       op.getOffset(), rewriter, op.getLoc());
    auto alignType = pto::AlignType::get(rewriter.getContext());
    Value align =
        rewriter.create<pto::VldasOp>(op.getLoc(), alignType, loadPtr);
    auto load = rewriter.create<pto::VldusOp>(
        op.getLoc(), TypeRange{vecType, alignType},
        ValueRange{loadPtr, align});
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

struct ExpandDmaLoadPattern : public OpRewritePattern<pto::DmaLoadOp> {
  using OpRewritePattern<pto::DmaLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops = ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops = ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Size = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideOutToUbOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      loop2Size = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideOutToUbOp>(
          loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideOutToUbOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, loop2Size, loop1Count);
    }

    Value leftPadding = op.getLeftPaddingCount();
    if (!leftPadding)
      leftPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value rightPadding = op.getRightPaddingCount();
    if (!rightPadding)
      rightPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value dataSelect = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI1Type(),
        rewriter.getBoolAttr(static_cast<bool>(op.getPadValue())));

    if (Value padValue = op.getPadValue())
      rewriter.create<pto::SetMovPadValOp>(loc, padValue);

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyGmToUbufOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              leftPadding, rightPadding, dataSelect, op.getL2CacheCtl(),
              op.getNburstSrcStride(), op.getNburstDstStride());
        });
    if (shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeOutToUbOp>(loc, one, one);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandDmaStorePattern : public OpRewritePattern<pto::DmaStoreOp> {
  using OpRewritePattern<pto::DmaStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaStoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    SmallVector<pto::DmaLoopConfig> loops =
        collectLoopConfigs(op.getLoopCounts(), op.getLoopSrcStrides(),
                           op.getLoopDstStrides());
    ArrayRef<pto::DmaLoopConfig> hwLoops = ArrayRef<pto::DmaLoopConfig>(loops).take_front(2);
    ArrayRef<pto::DmaLoopConfig> swLoops = ArrayRef<pto::DmaLoopConfig>(loops).drop_front(hwLoops.size());

    Value loop1Count;
    Value loop2Size = one;
    if (hwLoops.size() == 2) {
      rewriter.create<pto::SetLoop2StrideUbToOutOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      loop2Size = hwLoops[0].count;
      loop1Count = hwLoops[1].count;
      rewriter.create<pto::SetLoop1StrideUbToOutOp>(
          loc, hwLoops[1].srcStride, hwLoops[1].dstStride);
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
    } else if (hwLoops.size() == 1) {
      loop1Count = hwLoops[0].count;
      rewriter.create<pto::SetLoop1StrideUbToOutOp>(
          loc, hwLoops[0].srcStride, hwLoops[0].dstStride);
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, loop2Size, loop1Count);
    }

    buildSoftwareLoopNest(
        rewriter, loc, swLoops, zero, zero,
        [&](Value srcOffset, Value dstOffset) {
          Value source = offsetPointerByBytes(op.getSource(), srcOffset, rewriter, loc);
          Value destination =
              offsetPointerByBytes(op.getDestination(), dstOffset, rewriter, loc);
          rewriter.create<pto::CopyUbufToGmOp>(
              loc, source, destination, zero, op.getNBurst(), op.getLenBurst(),
              zero, op.getNburstDstStride(), op.getNburstSrcStride());
        });
    if (shouldRestoreDmaLoopSize(loop1Count, loop2Size))
      rewriter.create<pto::SetLoopSizeUbToOutOp>(loc, one, one);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpandDmaCopyPattern : public OpRewritePattern<pto::DmaCopyOp> {
  using OpRewritePattern<pto::DmaCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::DmaCopyOp op,
                                PatternRewriter &rewriter) const override {
    Value zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 64);
    rewriter.replaceOpWithNewOp<pto::CopyUbufToUbufOp>(
        op, op.getSource(), op.getDestination(), zero, op.getNBurst(),
        op.getLenBurst(), op.getSrcStride(), op.getDstStride());
    return success();
  }
};

struct PTOVPTOExpandBridgeOpsPass
    : public pto::impl::PTOVPTOExpandBridgeOpsBase<PTOVPTOExpandBridgeOpsPass> {
  using pto::impl::PTOVPTOExpandBridgeOpsBase<
      PTOVPTOExpandBridgeOpsPass>::PTOVPTOExpandBridgeOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandUvldPattern, ExpandDmaLoadPattern, ExpandDmaStorePattern,
                 ExpandDmaCopyPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVPTOExpandBridgeOpsPass() {
  return std::make_unique<PTOVPTOExpandBridgeOpsPass>();
}
