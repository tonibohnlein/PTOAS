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
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCANONICALIZEIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kLogicalRank2 = 2;
constexpr unsigned kCanonicalRank5 = 5;
constexpr int64_t kUnitExtent = 1;
constexpr unsigned kRank2Rows = 0;
constexpr unsigned kRank2Cols = 1;
constexpr int64_t kRank2ToRank5DimOffset = 3;

static SmallVector<int64_t, kCanonicalRank5>
rightAlignRank2Shape(ArrayRef<int64_t> shape) {
  return {kUnitExtent, kUnitExtent, kUnitExtent, shape[kRank2Rows],
          shape[kRank2Cols]};
}

static Value getOrCreateIndexConstant(OpBuilder &builder, Location loc,
                                      int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static SmallVector<Value, kCanonicalRank5>
prependThreeValues(ValueRange values, Value fill) {
  return {fill, fill, fill, values[kRank2Rows], values[kRank2Cols]};
}

static SmallVector<Value, kCanonicalRank5>
buildCanonicalRank2Strides(MakeTensorViewOp op) {
  Value rowStride = op.getStrides()[kRank2Rows];
  Value colStride = op.getStrides()[kRank2Cols];
  auto layout = op.getLayoutAttr();
  if (layout && layout.getLayout() == Layout::DN)
    return {colStride, colStride, colStride, rowStride, colStride};
  return {rowStride, rowStride, rowStride, rowStride, colStride};
}

static bool isRank2ViewLike(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type))
    return viewType.getRank() == kLogicalRank2;
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type))
    return viewType.getRank() == kLogicalRank2;
  return false;
}

static Type canonicalViewType(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type)) {
    if (viewType.getRank() == kLogicalRank2)
      return TensorViewType::get(type.getContext(),
                                 rightAlignRank2Shape(viewType.getShape()),
                                 viewType.getElementType());
    return type;
  }
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type)) {
    if (viewType.getRank() == kLogicalRank2)
      return PartitionTensorViewType::get(
          type.getContext(), rightAlignRank2Shape(viewType.getShape()),
          viewType.getElementType());
    return type;
  }
  return type;
}

static bool canonicalizeValueType(Value value) {
  Type oldType = value.getType();
  Type newType = canonicalViewType(oldType);
  if (newType == oldType)
    return false;
  value.setType(newType);
  return true;
}

static LogicalResult rewriteMakeTensorView(MakeTensorViewOp op,
                                           IRRewriter &rewriter) {
  auto oldType = dyn_cast<TensorViewType>(op.getResult().getType());
  if (!oldType || oldType.getRank() != kLogicalRank2)
    return success();

  if (op.getShape().size() != kLogicalRank2 ||
      op.getStrides().size() != kLogicalRank2)
    return op.emitOpError(
        "rank-2 tensor_view must have exactly 2 shape and stride operands");

  rewriter.setInsertionPoint(op);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newShape =
      prependThreeValues(op.getShape(), one);
  SmallVector<Value, kCanonicalRank5> newStrides =
      buildCanonicalRank2Strides(op);
  auto newType = cast<TensorViewType>(canonicalViewType(oldType));

  auto newOp = rewriter.create<MakeTensorViewOp>(
      op.getLoc(), newType, op.getPtr(), newShape, newStrides,
      op.getLayoutAttr());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static LogicalResult rewritePartitionView(PartitionViewOp op,
                                          IRRewriter &rewriter) {
  auto sourceType = dyn_cast<TensorViewType>(op.getSource().getType());
  auto resultType = dyn_cast<PartitionTensorViewType>(op.getResult().getType());
  if (!sourceType || !resultType)
    return success();

  if (op.getOffsets().size() != kLogicalRank2 ||
      op.getSizes().size() != kLogicalRank2)
    return success();

  if (sourceType.getRank() != kCanonicalRank5)
    return op.emitOpError(
        "rank-2 partition_tensor_view normalization expects canonical rank-5 "
        "source tensor_view");

  rewriter.setInsertionPoint(op);
  Value zero = getOrCreateIndexConstant(rewriter, op.getLoc(), 0);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newOffsets =
      prependThreeValues(op.getOffsets(), zero);
  SmallVector<Value, kCanonicalRank5> newSizes =
      prependThreeValues(op.getSizes(), one);
  auto newType = cast<PartitionTensorViewType>(canonicalViewType(resultType));

  auto newOp = rewriter.create<PartitionViewOp>(
      op.getLoc(), newType, op.getSource(), newOffsets, newSizes);
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static Value buildCanonicalDimIndex(Value dimIndex, IRRewriter &rewriter,
                                    Location loc) {
  rewriter.setInsertionPointAfterValue(dimIndex);
  Value offset =
      getOrCreateIndexConstant(rewriter, loc, kRank2ToRank5DimOffset);
  return rewriter.create<arith::AddIOp>(loc, dimIndex, offset);
}

static void rewriteTensorViewDimOperand(Operation *op, Value dimIndex,
                                        IRRewriter &rewriter) {
  Value newDim = buildCanonicalDimIndex(dimIndex, rewriter, op->getLoc());
  op->setOperand(1, newDim);
}

static void canonicalizeFunctionType(func::FuncOp func) {
  auto oldType = func.getFunctionType();
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  bool changed = false;

  inputs.reserve(oldType.getNumInputs());
  for (Type type : oldType.getInputs()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    inputs.push_back(newType);
  }

  results.reserve(oldType.getNumResults());
  for (Type type : oldType.getResults()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    results.push_back(newType);
  }

  if (changed)
    func.setFunctionType(FunctionType::get(func.getContext(), inputs, results));
}

static void canonicalizeValueTypes(func::FuncOp func) {
  canonicalizeFunctionType(func);

  func->walk([](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments())
          canonicalizeValueType(arg);
      }
    }

    for (OpResult result : op->getResults())
      canonicalizeValueType(result);
  });
}

struct PTOCanonicalizeIRPass
    : public mlir::pto::impl::PTOCanonicalizeIRBase<PTOCanonicalizeIRPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<MakeTensorViewOp> makeViews;
    SmallVector<PartitionViewOp> partitionViews;
    SmallVector<std::pair<Operation *, Value>> dimIndexOps;

    func.walk([&](MakeTensorViewOp op) {
      if (isRank2ViewLike(op.getResult().getType()))
        makeViews.push_back(op);
    });
    func.walk([&](PartitionViewOp op) {
      if (op.getOffsets().size() == kLogicalRank2 &&
          op.getSizes().size() == kLogicalRank2)
        partitionViews.push_back(op);
    });
    func.walk([&](GetTensorViewDimOp op) {
      if (isRank2ViewLike(op.getTensorView().getType()))
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex());
    });
    func.walk([&](GetTensorViewStrideOp op) {
      if (isRank2ViewLike(op.getTensorView().getType()))
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex());
    });

    IRRewriter rewriter(func.getContext());
    for (MakeTensorViewOp op : makeViews) {
      if (failed(rewriteMakeTensorView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
    for (auto [op, dimIndex] : dimIndexOps)
      rewriteTensorViewDimOperand(op, dimIndex, rewriter);
    canonicalizeValueTypes(func);
    for (PartitionViewOp op : partitionViews) {
      if (failed(rewritePartitionView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCanonicalizeIRPass() {
  return std::make_unique<PTOCanonicalizeIRPass>();
}
