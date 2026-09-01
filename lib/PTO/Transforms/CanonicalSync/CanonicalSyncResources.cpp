// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

std::optional<CanonicalCore> getSectionCore(Operation *operation) {
  for (Operation *parent = operation; parent; parent = parent->getParentOp()) {
    if (isa<SectionCubeOp>(parent)) {
      return CanonicalCore::AIC;
    }
    if (isa<SectionVectorOp>(parent)) {
      return CanonicalCore::AIV;
    }
  }
  return std::nullopt;
}

std::optional<CanonicalCore> getFunctionCore(func::FuncOp function) {
  auto kind = function->getAttrOfType<FunctionKernelKindAttr>(
      FunctionKernelKindAttr::name);
  if (!kind) {
    return std::nullopt;
  }
  return kind.getKernelKind() == FunctionKernelKind::Cube ? CanonicalCore::AIC
                                                          : CanonicalCore::AIV;
}

std::optional<PIPE> convertPipe(PipelineType pipe) {
  switch (pipe) {
  case PipelineType::PIPE_S:
    return PIPE::PIPE_S;
  case PipelineType::PIPE_V:
    return PIPE::PIPE_V;
  case PipelineType::PIPE_M:
    return PIPE::PIPE_M;
  case PipelineType::PIPE_MTE1:
    return PIPE::PIPE_MTE1;
  case PipelineType::PIPE_MTE2:
    return PIPE::PIPE_MTE2;
  case PipelineType::PIPE_MTE3:
    return PIPE::PIPE_MTE3;
  case PipelineType::PIPE_FIX:
    return PIPE::PIPE_FIX;
  default:
    return std::nullopt;
  }
}

} // namespace

FailureOr<CanonicalPhysicalResource>
mlir::pto::canonical_sync_detail::resolvePhysicalResource(func::FuncOp function,
                                                          Operation *operation,
                                                          PIPE pipe) {
  std::optional<CanonicalCore> core = getSectionCore(operation);
  if (!core) {
    switch (pipe) {
    case PIPE::PIPE_M:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_FIX:
      core = CanonicalCore::AIC;
      break;
    case PIPE::PIPE_S:
      // Scalar instructions execute on the core selected by the surrounding
      // physical section (handled above), or by the function kind when there
      // is no section.  In particular, a scalar load in a cube-only kernel is
      // emitted inside __DAV_CUBE__ and must not be modeled as AIV work.
      core = getFunctionCore(function);
      break;
    case PIPE::PIPE_V:
      core = CanonicalCore::AIV;
      break;
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
      core = getFunctionCore(function);
      break;
    default:
      break;
    }
  }
  if (!core) {
    operation->emitError("canonical sync cannot resolve the physical core for ")
        << stringifyPIPE(pipe)
        << "; normalize physical sections or provide pto.kernel_kind";
    return failure();
  }
  return CanonicalPhysicalResource{*core, pipe};
}

LogicalResult
mlir::pto::canonical_sync_detail::rejectUnsupportedCanonicalSyncInput(
    func::FuncOp function) {
  bool invalid = false;
  function.walk([&](Operation *operation) {
    if (isa<RecordEventOp, WaitEventOp, BarrierSyncOp, SetFlagOp, WaitFlagOp,
            SetFlagDynOp, WaitFlagDynOp, SyncSetOp, SyncWaitOp, SetCrossBlockOp,
            WaitCrossBlockOp, SetIntraBlockOp, WaitIntraBlockOp, BarrierOp,
            GetBufOp, GetBufDynOp, RlsBufOp, RlsBufDynOp>(operation)) {
      operation->emitError(
          "canonical sync requires ownership of physical synchronization; "
          "remove the pre-existing synchronization operation or select a "
          "different sync mode");
      invalid = true;
    }
  });
  if (auto hint =
          function->getAttrOfType<StringAttr>("pto.auto_sync_tail_hint")) {
    const StringRef hintValue = hint.getValue();
    if (hintValue == "mte3-to-s-event0") {
      function.emitError("canonical sync requires a PIPE_ALL epilogue and "
                         "does not support mte3-to-s-event0 tail hints");
      invalid = true;
    }
  }
  return invalid ? failure() : success();
}

SmallVector<unsigned, 6> mlir::pto::canonical_sync_detail::reservedEventIds(
    func::FuncOp function, CanonicalPhysicalResource source,
    CanonicalPhysicalResource target) {
  SmallVector<unsigned, 6> result;
  function.walk([&](Operation *operation) {
    std::optional<SyncMacroModel> model = getSyncMacroModel(operation);
    if (!model) {
      return;
    }
    FailureOr<CanonicalPhysicalResource> macroResource =
        resolvePhysicalResource(function, operation, source.pipe);
    const bool wrongCore =
        succeeded(macroResource) && macroResource->core != source.core;
    const bool unresolvedResource = failed(macroResource);
    if (unresolvedResource || wrongCore) {
      return;
    }
    for (const SyncMacroHiddenEvent &event : model->hiddenEvents) {
      const std::optional<PIPE> sourcePipe = convertPipe(event.srcPipe);
      const std::optional<PIPE> targetPipe = convertPipe(event.dstPipe);
      if (sourcePipe == source.pipe && targetPipe == target.pipe) {
        result.append(event.eventIds.begin(), event.eventIds.end());
      }
    }
  });
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}
