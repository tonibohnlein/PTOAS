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
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VPTOPTRNORMALIZE
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
  return {};
}

static Value buildIndexValue(OpBuilder &builder, Location loc,
                             OpFoldResult ofr) {
  if (auto value = dyn_cast<Value>(ofr))
    return value;
  auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
  return builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
}

static bool needsSubviewPtrConversion(memref::SubViewOp op) {
  auto resultType = dyn_cast<MemRefType>(op.getType());
  if (!resultType)
    return false;
  return static_cast<bool>(
      getPointerMemorySpace(resultType.getMemorySpace(), op.getContext()));
}

static Type convertSubviewResultType(Type type) {
  auto memrefType = dyn_cast<MemRefType>(type);
  if (!memrefType)
    return type;

  auto memorySpace =
      getPointerMemorySpace(memrefType.getMemorySpace(), type.getContext());
  if (!memorySpace)
    return type;

  return pto::PtrType::get(type.getContext(), memrefType.getElementType(),
                           memorySpace);
}

static bool hasPtrNormalizeConvertibleType(Type type) {
  if (isa<pto::PtrType>(type))
    return true;
  auto memrefType = dyn_cast<MemRefType>(type);
  return memrefType && static_cast<bool>(getPointerMemorySpace(
                           memrefType.getMemorySpace(), type.getContext()));
}

static bool hasPtrNormalizeConvertibleType(TypeRange types) {
  return llvm::any_of(
      types, [](Type type) { return hasPtrNormalizeConvertibleType(type); });
}

static bool isMemRefType(Type type) { return isa<BaseMemRefType>(type); }

static Value materializeUnrealizedCast(OpBuilder &builder, Type resultType,
                                       ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return {};
  return builder
      .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
      .getResult(0);
}

static LogicalResult computeSubviewElementOffset(memref::SubViewOp op,
                                                 PatternRewriter &rewriter,
                                                 Value &offset) {
  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
  if (!sourceType)
    return failure();

  SmallVector<int64_t> strides;
  int64_t baseOffset = 0;
  if (failed(getStridesAndOffset(sourceType, strides, baseOffset)))
    return failure();
  // The SSA source already names the base address after bind_tile/pointer_cast
  // normalization. A dynamic memref layout offset here is metadata we can
  // ignore for ptr normalization and model as zero.
  if (baseOffset == ShapedType::kDynamic)
    baseOffset = 0;

  Location loc = op.getLoc();
  Value total = rewriter.create<arith::ConstantIndexOp>(loc, baseOffset);
  ArrayRef<OpFoldResult> mixedOffsets = op.getMixedOffsets();
  if (mixedOffsets.size() != strides.size())
    return failure();

  for (auto [ofr, stride] : llvm::zip(mixedOffsets, strides)) {
    if (stride == 0)
      continue;
    if (stride == ShapedType::kDynamic)
      return failure();

    Value idx = buildIndexValue(rewriter, loc, ofr);
    if (!idx.getType().isIndex())
      return failure();

    if (stride != 1) {
      Value strideValue =
          rewriter.create<arith::ConstantIndexOp>(loc, stride);
      idx = rewriter.create<arith::MulIOp>(loc, idx, strideValue);
    }
    total = rewriter.create<arith::AddIOp>(loc, total, idx);
  }

  offset = total;
  return success();
}

static Value materializeSubviewInputPtr(Value source, PatternRewriter &rewriter,
                                        Location loc) {
  if (!source)
    return {};
  if (isa<pto::PtrType>(source.getType()))
    return source;

  auto memrefType = dyn_cast<MemRefType>(source.getType());
  if (!memrefType)
    return {};

  auto memorySpace =
      getPointerMemorySpace(memrefType.getMemorySpace(), rewriter.getContext());
  if (!memorySpace)
    return {};

  auto ptrType = pto::PtrType::get(rewriter.getContext(),
                                   memrefType.getElementType(), memorySpace);
  return rewriter.create<pto::CastPtrOp>(loc, ptrType, source);
}

static Value materializeScalarAccessPtr(Value source, PatternRewriter &rewriter,
                                        Location loc) {
  if (!source)
    return {};
  if (isa<pto::PtrType>(source.getType()))
    return source;

  if (auto cast = source.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
      return {};
    Value input = cast.getOperands().front();
    if (isa<pto::PtrType>(input.getType()))
      return input;
    return materializeScalarAccessPtr(input, rewriter, loc);
  }

  if (auto cast = source.getDefiningOp<memref::CastOp>())
    return materializeScalarAccessPtr(cast.getSource(), rewriter, loc);

  if (auto subview = source.getDefiningOp<memref::SubViewOp>()) {
    if (!needsSubviewPtrConversion(subview))
      return {};

    Value basePtr =
        materializeScalarAccessPtr(subview.getSource(), rewriter, loc);
    if (!basePtr)
      return {};

    Value offset;
    if (failed(computeSubviewElementOffset(subview, rewriter, offset)))
      return {};

    auto ptrType = dyn_cast<pto::PtrType>(convertSubviewResultType(source.getType()));
    if (!ptrType)
      return {};
    if (basePtr.getType() != ptrType)
      basePtr = rewriter.create<pto::CastPtrOp>(loc, ptrType, basePtr);
    return rewriter.create<pto::AddPtrOp>(loc, ptrType, basePtr, offset);
  }

  if (auto bind = source.getDefiningOp<pto::BindTileOp>())
    return materializeScalarAccessPtr(bind.getSource(), rewriter, loc);

  if (auto pointerCast = source.getDefiningOp<pto::PointerCastOp>()) {
    if (pointerCast.getAddrs().empty())
      return {};
    Value addr = pointerCast.getAddrs().front();
    if (isa<pto::PtrType>(addr.getType()))
      return addr;
    return materializeScalarAccessPtr(addr, rewriter, loc);
  }

  // Restrict normalization to memref views that already sit on top of a ptr-like
  // boundary bridge. Materializing fresh memref -> ptr casts here would leave
  // illegal pto.castptr(memref) behind in this pass.
  return {};
}

static Value materializeBoundaryOperandPtr(Value source,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  if (!source)
    return {};
  if (isa<pto::PtrType>(source.getType()))
    return source;
  return materializeScalarAccessPtr(source, rewriter, loc);
}

template <typename OpTy>
static LogicalResult rewriteBufferLikeBoundaryOp(
    OpTy op, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter,
    StringRef sourceRole, StringRef destinationRole) {
  Value source =
      materializeBoundaryOperandPtr(adaptor.getOperands()[0], rewriter, op.getLoc());
  if (!source)
    return rewriter.notifyMatchFailure(
        op, (Twine("failed to materialize ") + sourceRole + " ptr").str());
  if (!isa<pto::PtrType>(source.getType()))
    return rewriter.notifyMatchFailure(
        op, (Twine("expected ptr-form ") + sourceRole).str());

  Value destination = materializeBoundaryOperandPtr(adaptor.getOperands()[1],
                                                    rewriter, op.getLoc());
  if (!destination)
    return rewriter.notifyMatchFailure(
        op, (Twine("failed to materialize ") + destinationRole + " ptr").str());
  if (!isa<pto::PtrType>(destination.getType()))
    return rewriter.notifyMatchFailure(
        op, (Twine("expected ptr-form ") + destinationRole).str());

  SmallVector<Value> operands(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
  operands[0] = source;
  operands[1] = destination;

  OperationState state(op.getLoc(), op->getName().getStringRef());
  state.addOperands(operands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  state.propertiesAttr = op->getPropertiesAsAttribute();
  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

struct ConvertTileBufAddrToPtrPattern
    : public OpConversionPattern<pto::TileBufAddrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::TileBufAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getDst().getType());
    if (!isa<pto::PtrType>(convertedType))
      return failure();

    rewriter.replaceOpWithNewOp<pto::TileBufAddrOp>(op, convertedType,
                                                    adaptor.getSrc());
    return success();
  }
};

struct ConvertPointerCastToCastPtrPattern
    : public OpConversionPattern<pto::PointerCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    auto ptrType = dyn_cast<pto::PtrType>(convertedType);
    if (!ptrType)
      return failure();

    if (adaptor.getAddrs().empty())
      return rewriter.notifyMatchFailure(op, "expected at least one address");

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, ptrType,
                                                adaptor.getAddrs().front());
    return success();
  }
};

struct ConvertCastPtrPattern : public OpConversionPattern<pto::CastPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CastPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    if (isMemRefType(inputType) || isMemRefType(convertedResultType))
      return rewriter.notifyMatchFailure(op,
                                         "memref castptr must be eliminated");

    if (!isa<pto::PtrType, IntegerType>(inputType) ||
        !isa<pto::PtrType, IntegerType>(convertedResultType))
      return rewriter.notifyMatchFailure(op,
                                         "expected ptr/int castptr operands");

    if (inputType == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }

    rewriter.replaceOpWithNewOp<pto::CastPtrOp>(op, convertedResultType, input);
    return success();
  }
};

struct ConvertBindTileToPtrPattern : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    auto ptrType = dyn_cast<pto::PtrType>(convertedType);
    if (!ptrType)
      return failure();

    Value ptr =
        materializeSubviewInputPtr(adaptor.getSource(), rewriter, op.getLoc());
    if (!ptr)
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize bind_tile input ptr");

    if (ptr.getType() != ptrType)
      ptr = rewriter.create<pto::CastPtrOp>(op.getLoc(), ptrType, ptr);

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct ConvertSubviewToAddPtrPattern
    : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!needsSubviewPtrConversion(op))
      return failure();

    auto ptrType =
        dyn_cast<pto::PtrType>(getTypeConverter()->convertType(op.getType()));
    if (!ptrType)
      return rewriter.notifyMatchFailure(op, "expected ptr result type");

    Value basePtr =
        materializeSubviewInputPtr(adaptor.getSource(), rewriter, op.getLoc());
    if (!basePtr)
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize subview input ptr");

    Value offset;
    if (failed(computeSubviewElementOffset(op, rewriter, offset)))
      return rewriter.notifyMatchFailure(op,
                                         "failed to compute subview element offset");

    rewriter.replaceOpWithNewOp<pto::AddPtrOp>(op, ptrType, basePtr, offset);
    return success();
  }
};

struct ConvertVldsSubviewOperandPattern : public OpConversionPattern<pto::VldsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::VldsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<pto::PtrType>(adaptor.getSource().getType()))
      return failure();

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands({adaptor.getSource(), adaptor.getOffset()});
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertVstsSubviewOperandPattern : public OpConversionPattern<pto::VstsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::VstsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<pto::PtrType>(adaptor.getDestination().getType()))
      return failure();

    OperationState state(op.getLoc(), op->getName().getStringRef());
    state.addOperands(
        {adaptor.getValue(), adaptor.getDestination(), adaptor.getOffset(),
         adaptor.getMask()});
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct ConvertLoadScalarOperandToPtrPattern
    : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr)
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize load_scalar ptr");
    if (!isa<pto::PtrType>(ptr.getType()))
      return rewriter.notifyMatchFailure(op, "expected ptr-form load_scalar input");

    rewriter.replaceOpWithNewOp<pto::LoadScalarOp>(op, op.getValue().getType(),
                                                   ptr, adaptor.getOffset());
    return success();
  }
};

struct ConvertStoreScalarOperandToPtrPattern
    : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = materializeScalarAccessPtr(adaptor.getPtr(), rewriter, op.getLoc());
    if (!ptr)
      return rewriter.notifyMatchFailure(op,
                                         "failed to materialize store_scalar ptr");
    if (!isa<pto::PtrType>(ptr.getType()))
      return rewriter.notifyMatchFailure(op, "expected ptr-form store_scalar input");

    rewriter.replaceOpWithNewOp<pto::StoreScalarOp>(op, ptr,
                                                    adaptor.getOffset(),
                                                    adaptor.getValue());
    return success();
  }
};

struct ConvertCubeLoadOperandPattern
    : public OpConversionPattern<pto::CubeLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CubeLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "cube_load source",
                                       "cube_load destination");
  }
};

struct ConvertCubeStoreOperandPattern
    : public OpConversionPattern<pto::CubeStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CubeStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "cube_store source",
                                       "cube_store destination");
  }
};

struct ConvertBiasLoadOperandPattern
    : public OpConversionPattern<pto::BiasLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::BiasLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "bias_load source",
                                       "bias_load destination");
  }
};

struct ConvertCubeLoadFracOperandPattern
    : public OpConversionPattern<pto::CubeLoadFracOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::CubeLoadFracOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "cube_load_frac source",
                                       "cube_load_frac destination");
  }
};

struct ConvertLeftLoadOperandPattern
    : public OpConversionPattern<pto::LeftLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LeftLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "left_load source",
                                       "left_load destination");
  }
};

struct ConvertRightLoadOperandPattern
    : public OpConversionPattern<pto::RightLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::RightLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "right_load source",
                                       "right_load destination");
  }
};

struct ConvertLeftLoadMxOperandPattern
    : public OpConversionPattern<pto::LeftLoadMxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::LeftLoadMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "left_load_mx source",
                                       "left_load_mx destination");
  }
};

struct ConvertRightLoadMxOperandPattern
    : public OpConversionPattern<pto::RightLoadMxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::RightLoadMxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "right_load_mx source",
                                       "right_load_mx destination");
  }
};

struct ConvertAccStoreOperandPattern
    : public OpConversionPattern<pto::AccStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AccStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter, "acc_store source",
                                       "acc_store destination");
  }
};

struct ConvertAccStoreGmOperandPattern
    : public OpConversionPattern<pto::AccStoreGmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AccStoreGmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "acc_store_gm source",
                                       "acc_store_gm destination");
  }
};

struct ConvertAccStoreUbOperandPattern
    : public OpConversionPattern<pto::AccStoreUbOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pto::AccStoreUbOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteBufferLikeBoundaryOp(op, adaptor, rewriter,
                                       "acc_store_ub source",
                                       "acc_store_ub destination");
  }
};

struct ConvertPtrNormalizeUnrealizedCastOp final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasPtrNormalizeConvertibleType(op->getOperandTypes()) &&
        !hasPtrNormalizeConvertibleType(op->getResultTypes()))
      return failure();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() != convertedResultType)
      return failure();

    rewriter.replaceOp(op, input);
    return success();
  }
};

struct VPTOPtrNormalizePass
    : public pto::impl::VPTOPtrNormalizeBase<VPTOPtrNormalizePass> {
  using pto::impl::VPTOPtrNormalizeBase<
      VPTOPtrNormalizePass>::VPTOPtrNormalizeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion(
        [](Type type) { return convertSubviewResultType(type); });
    typeConverter.addTargetMaterialization(materializeUnrealizedCast);
    typeConverter.addSourceMaterialization(materializeUnrealizedCast);
    typeConverter.addArgumentMaterialization(materializeUnrealizedCast);

    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           scf::SCFDialect>();
    target.addDynamicallyLegalDialect<pto::PTODialect>([](Operation *op) {
      return !isa<pto::TileBufAddrOp, pto::PointerCastOp, pto::CastPtrOp,
                  pto::BindTileOp, pto::VldsOp, pto::VstsOp>(op);
    });
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp op) {
          return !hasPtrNormalizeConvertibleType(op->getOperandTypes()) &&
                 !hasPtrNormalizeConvertibleType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<pto::TileBufAddrOp>([&](pto::TileBufAddrOp op) {
      return op.getDst().getType() ==
             typeConverter.convertType(op.getDst().getType());
    });
    target.addDynamicallyLegalOp<pto::PointerCastOp>(
        [&](pto::PointerCastOp op) {
          return op.getResult().getType() ==
                 typeConverter.convertType(op.getResult().getType());
        });
    target.addDynamicallyLegalOp<pto::CastPtrOp>([&](pto::CastPtrOp op) {
      return !isMemRefType(op.getInput().getType()) &&
             !isMemRefType(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<pto::BindTileOp>([&](pto::BindTileOp op) {
      return op.getResult().getType() ==
             typeConverter.convertType(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<pto::VldsOp>(
        [](pto::VldsOp op) { return isa<pto::PtrType>(op.getSource().getType()); });
    target.addDynamicallyLegalOp<pto::VstsOp>([](pto::VstsOp op) {
      return isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::LoadScalarOp>(
        [](pto::LoadScalarOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::StoreScalarOp>(
        [](pto::StoreScalarOp op) { return isa<pto::PtrType>(op.getPtr().getType()); });
    target.addDynamicallyLegalOp<pto::CubeLoadOp>([](pto::CubeLoadOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::CubeStoreOp>([](pto::CubeStoreOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::BiasLoadOp>([](pto::BiasLoadOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::CubeLoadFracOp>([](pto::CubeLoadFracOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::LeftLoadOp>([](pto::LeftLoadOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::RightLoadOp>([](pto::RightLoadOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::LeftLoadMxOp>([](pto::LeftLoadMxOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::RightLoadMxOp>([](pto::RightLoadMxOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::AccStoreOp>([](pto::AccStoreOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::AccStoreGmOp>([](pto::AccStoreGmOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<pto::AccStoreUbOp>([](pto::AccStoreUbOp op) {
      return isa<pto::PtrType>(op.getSource().getType()) &&
             isa<pto::PtrType>(op.getDestination().getType());
    });
    target.addDynamicallyLegalOp<memref::SubViewOp>(
        [](memref::SubViewOp op) { return !needsSubviewPtrConversion(op); });

    RewritePatternSet patterns(context);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                         target);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    patterns.add<ConvertTileBufAddrToPtrPattern,
                 ConvertPointerCastToCastPtrPattern, ConvertCastPtrPattern,
                 ConvertBindTileToPtrPattern,
                 ConvertSubviewToAddPtrPattern, ConvertVldsSubviewOperandPattern,
                 ConvertVstsSubviewOperandPattern,
                 ConvertLoadScalarOperandToPtrPattern,
                 ConvertStoreScalarOperandToPtrPattern,
                 ConvertCubeLoadOperandPattern,
                 ConvertCubeStoreOperandPattern,
                 ConvertBiasLoadOperandPattern,
                 ConvertCubeLoadFracOperandPattern,
                 ConvertLeftLoadOperandPattern,
                 ConvertRightLoadOperandPattern,
                 ConvertLeftLoadMxOperandPattern,
                 ConvertRightLoadMxOperandPattern,
                 ConvertAccStoreOperandPattern,
                 ConvertAccStoreGmOperandPattern,
                 ConvertAccStoreUbOperandPattern,
                 ConvertPtrNormalizeUnrealizedCastOp>(
        typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVPTOPtrNormalizePass() {
  return std::make_unique<VPTOPtrNormalizePass>();
}
