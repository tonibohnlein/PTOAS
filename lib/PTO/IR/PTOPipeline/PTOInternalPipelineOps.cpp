// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// The unified FIFO handle fixes publication/acquisition/release pipes. GM
// entries are descriptors, so their type alone does not identify that pipe.
static pto::PIPE getFifoBoundaryPipe(Operation* op, Value handle, bool producer, bool release) {
  Operation* init = handle.getDefiningOp();
  int64_t mask = 0;
  bool direct = false;
  if (auto gm = dyn_cast_or_null<InitializeL2G2LPipeOp>(init)) {
    mask = gm.getDirMask();
  } else if (auto local = dyn_cast_or_null<InitializeL2LPipeOp>(init)) {
    mask = local.getDirMask();
    direct = getTargetArch(op) == PTOArch::A5;
  } else { return pto::PIPE::PIPE_UNASSIGNED; }
  bool c2v = mask == 1;
  if (mask == 3) {
    auto core = pto::recoverSyncPhysicalCore(op);
    if (core != pto::SyncPhysicalCore::AIC && core != pto::SyncPhysicalCore::AIV) {
      return pto::PIPE::PIPE_UNASSIGNED;
    }
    c2v = producer ? core == pto::SyncPhysicalCore::AIC : core == pto::SyncPhysicalCore::AIV;
  } else if (mask != 1 && mask != 2) { return pto::PIPE::PIPE_UNASSIGNED; }
  if (producer) {
    return c2v ? pto::PIPE::PIPE_FIX : pto::PIPE::PIPE_MTE3;
  }
  if (release && !c2v && getTargetArch(op) == PTOArch::A5) {
    return pto::PIPE::PIPE_MTE1;
  }
  if (!direct) {
    return pto::PIPE::PIPE_MTE2;
  }
  return c2v ? pto::PIPE::PIPE_V : pto::PIPE::PIPE_MTE1;
}

pto::PIPE TPushOp::getPipe() {
  if (auto tile = dyn_cast<TileBufType>(getTile().getType())) {
    auto space = dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
    if (space && space.getAddressSpace() == AddressSpace::ACC) {
      return pto::PIPE::PIPE_FIX;
    }
    if (space && space.getAddressSpace() == AddressSpace::VEC) {
      return pto::PIPE::PIPE_MTE3;
    }
    return pto::PIPE::PIPE_UNASSIGNED;
  }
  return getFifoBoundaryPipe(getOperation(), getPipeHandle(), true, false);
}
pto::PIPE TAllocOp::getPipe() {
  if (getTargetArch(getOperation()) != PTOArch::A5) {
    return pto::PIPE::PIPE_S;
  }
  return getFifoBoundaryPipe(getOperation(), getPipeHandle(), true, false);
}
pto::PIPE TPopOp::getPipe() {
  if (isa<TensorViewType>(getTile().getType()) && getTargetArch(getOperation()) != PTOArch::A5) {
    // A2/A3 GlobalTensor POP only waits via wait_flag_dev and binds its address.
    return pto::PIPE::PIPE_S;
  }
  if (auto tile = dyn_cast<TileBufType>(getTile().getType())) {
    auto space = dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
    if (!space || (space.getAddressSpace() != AddressSpace::MAT &&
                   space.getAddressSpace() != AddressSpace::VEC)) { return pto::PIPE::PIPE_UNASSIGNED; }
  }
  return getFifoBoundaryPipe(getOperation(), getPipeHandle(), false, false);
}
pto::PIPE TFreeOp::getPipe() {
  return getFifoBoundaryPipe(getOperation(), getPipeHandle(), false, true);
}

static LogicalResult verifyInternalTileOp(
    Operation *op, Value pipeHandle, uint32_t split, bool producerSide,
    Value tile, Value aivSubblockId, pto::PIPE pipe, StringRef pipeError) {
  if (failed(verifyInternalPipeBase(op, pipeHandle, split, producerSide)) ||
      failed(verifyOddSplitTileEntry(op, split, tile.getType())))
    return failure();
  if (isInsideCubeKernelOrSection(op) &&
      failed(verifyFullTileSplitParity(op, split, tile.getType())))
    return failure();
  if (failed(verifyAivSubblockIdOperand(op, aivSubblockId, split,
                                        tile.getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          op, pipeHandle, tile.getType())))
    return failure();
  if (!isa<TensorViewType>(tile.getType()) &&
      pipe == pto::PIPE::PIPE_UNASSIGNED)
    return op->emitOpError(pipeError);
  return success();
}

LogicalResult TPushOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/true,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type must map to a supported producer pipe");
}

LogicalResult TAllocOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/true)) ||
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType())) ||
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType())))
    return failure();
  return success();
}

LogicalResult TPopOp::verify() {
  return verifyInternalTileOp(
      getOperation(), getPipeHandle(), getSplit(), /*producerSide=*/false,
      getTile(), getAivSubblockid(), getPipe(),
      "tile type and target arch must map to a supported consumer pipe");
}

LogicalResult TFreeOp::verify() {
  if (failed(verifyInternalPipeBase(getOperation(), getPipeHandle(), getSplit(),
                                    /*producerSide=*/false)))
    return failure();
  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

struct TFreeParseState {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;
};

static ParseResult parseTFreeOperands(OpAsmParser &parser,
                                      TFreeParseState &state) {
  if (parser.parseLParen() || parser.parseOperand(state.first))
    return failure();
  state.hasEntry = succeeded(parser.parseOptionalComma());
  if (!state.hasEntry) {
    if (parser.parseColonType(state.pipeTy) || parser.parseRParen())
      return failure();
    state.pipe = state.first;
    return success();
  }
  if (parser.parseOperand(state.pipe) || parser.parseColonType(state.firstTy) ||
      parser.parseComma() || parser.parseType(state.pipeTy) ||
      parser.parseRParen())
    return failure();
  return success();
}

static ParseResult parseTFreeAttributes(OpAsmParser &parser,
                                        NamedAttrList &attrs) {
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual())
    return failure();
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs))
    return failure();
  return success();
}

static ParseResult resolveTFreeOperands(OpAsmParser &parser,
                                        OperationState &result,
                                        const TFreeParseState &state) {
  if (state.hasEntry &&
      parser.resolveOperand(state.first, state.firstTy, result.operands))
    return failure();
  return parser.resolveOperand(state.pipe, state.pipeTy, result.operands);
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  TFreeParseState state;
  NamedAttrList attrs;
  if (failed(parseTFreeOperands(parser, state)) ||
      failed(parseTFreeAttributes(parser, attrs))) {
    return failure();
  }
  result.addAttributes(attrs);
  return resolveTFreeOperands(parser, result, state);
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

// SIMT keep/resume verify helpers now live in the shared internal header so
// the validation pass (PTOValidateVPTOIR.cpp) checks the identical rules.
#include "PTOPipeline/PTOSimtKeepResumeShared.h"

using mlir::pto::simt_detail::getFirstNonConstantLikeOp;
using mlir::pto::simt_detail::isOpInRange;
using mlir::pto::simt_detail::kSimtKeepResumeSlotLimit;
using mlir::pto::simt_detail::getSimtKeepResumeRegisterCount;
using mlir::pto::simt_detail::getSimtKeepResumeValueType;
using mlir::pto::simt_detail::verifySimtKeepResumeSlotRange;
using mlir::pto::simt_detail::overlapsEarlierSimtKeepResumeSlotUse;
using mlir::pto::simt_detail::verifyUniqueResumeGroupSlots;
using mlir::pto::simt_detail::verifyUniqueKeepGroupSlots;
using mlir::pto::simt_detail::isSupportedSimtKeepResumeType;

