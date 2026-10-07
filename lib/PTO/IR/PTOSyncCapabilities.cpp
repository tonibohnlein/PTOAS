// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTOSyncCapabilities.h"
#include "PTO/IR/PTO.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto {
bool isA5NoSplitPipeOp(Operation *op) {
  if (auto talloc = dyn_cast<pto::TAllocOp>(op)) {
    return talloc.getSplit() == 0;
  }
  if (auto tpush = dyn_cast<pto::TPushOp>(op)) {
    return tpush.getSplit() == 0;
  }
  if (auto tpop = dyn_cast<pto::TPopOp>(op)) {
    return tpop.getSplit() == 0;
  }
  if (auto tfree = dyn_cast<pto::TFreeOp>(op)) {
    return tfree.getSplit() == 0;
  }
  if (auto tpush = dyn_cast<pto::TPushToAivOp>(op)) {
    return tpush.getSplit() == 0;
  }
  if (auto tpush = dyn_cast<pto::TPushToAicOp>(op)) {
    return tpush.getSplit() == 0;
  }
  if (auto talloc = dyn_cast<pto::TAllocToAivOp>(op)) {
    return talloc.getSplit() == 0;
  }
  if (auto talloc = dyn_cast<pto::TAllocToAicOp>(op)) {
    return talloc.getSplit() == 0;
  }
  if (auto tpop = dyn_cast<pto::TPopFromAicOp>(op)) {
    return tpop.getSplit() == 0;
  }
  if (auto tpop = dyn_cast<pto::TPopFromAivOp>(op)) {
    return tpop.getSplit() == 0;
  }
  if (auto tfree = dyn_cast<pto::TFreeFromAicOp>(op)) {
    return tfree.getSplit() == 0;
  }
  if (auto tfree = dyn_cast<pto::TFreeFromAivOp>(op)) {
    return tfree.getSplit() == 0;
  }
  return false;
}

bool hasExplicitSubblockControl(Operation *op) {
  bool hasControl = false;
  op->walk([&](Operation *nested) {
    if (isa<pto::GetSubBlockIdxOp, pto::GetSubBlockNumOp>(nested)) {
      hasControl = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hasControl;
}
bool needsA5NoSplitVectorGuard(Operation *op) {
  // Synchronization normalization may make an originally implicit section
  // predicate explicit while preserving the original function-wide decision.
  if (auto saved = op->getAttrOfType<BoolAttr>("pto.sync_subblock_guard")) {
    return saved.getValue();
  }
  auto arch = getTargetArch(op);
  if (arch != PTOArch::A5) {
    return false;
  }
  bool isVectorScope = isa<pto::SectionVectorOp>(op);
  if (auto func = dyn_cast<func::FuncOp>(op)) {
    if (auto kernelKindAttr =
            func->getAttrOfType<FunctionKernelKindAttr>(
                FunctionKernelKindAttr::name)) {
      isVectorScope =
          kernelKindAttr.getKernelKind() == FunctionKernelKind::Vector;
    }
  }
  if (!isVectorScope) {
    return false;
  }
  if (hasExplicitSubblockControl(op)) {
    return false;
  }

  bool hasNoSplitPipe = false;
  op->walk([&](Operation *nested) {
    if (!isA5NoSplitPipeOp(nested)) {
      return WalkResult::advance();
    }
    hasNoSplitPipe = true;
    return WalkResult::interrupt();
  });
  return hasNoSplitPipe;
}


namespace {
// A2/A3 NPU-2201 AIC/AIV tables, HF-01/HF-06/HF-07 in
// OAHS_Hardware_Facts_Verified_A2A3.docx (2026-09-11).
// https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/intra_core_sync_overview.html
bool classic(StringRef architecture)
{
    return architecture == "a2" || architecture == "a3" || architecture == "a2a3";
}
} // namespace
SyncPhysicalCore recoverSyncPhysicalCore(Operation* anchor)
{
    SyncPhysicalCore result = SyncPhysicalCore::Unknown;
    auto merge = [&](SyncPhysicalCore core) {
        result = result == SyncPhysicalCore::Unknown || result == core ? core : SyncPhysicalCore::Conflict;
    };
    for (Operation* op = anchor; op; op = op->getParentOp()) {
        if (isa<FunctionOpInterface, ModuleOp>(op)) {
            if (auto raw = op->getAttr(FunctionKernelKindAttr::name)) {
                if (auto kind = dyn_cast<FunctionKernelKindAttr>(raw)) {
                    merge(kind.getKernelKind() == FunctionKernelKind::Cube ?
                              SyncPhysicalCore::AIC : SyncPhysicalCore::AIV);
                } else {
                    result = SyncPhysicalCore::Conflict;
                }
            }
        }
        if (isa<SectionCubeOp>(op)) {
            merge(SyncPhysicalCore::AIC);
        } else if (isa<SectionVectorOp>(op)) {
            merge(SyncPhysicalCore::AIV);
        }
    }
    return result;
}
std::optional<bool> getSyncPipelinePresence(StringRef architecture, SyncPhysicalCore core, PIPE pipe)
{
    if (!classic(architecture) || (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV)) {
        return std::nullopt;
    }
    if (pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3) {
        return true;
    }
    if (pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1 || pipe == PIPE::PIPE_FIX) {
        return core == SyncPhysicalCore::AIC;
    }
    if (pipe == PIPE::PIPE_V) {
        return core == SyncPhysicalCore::AIV;
    }
    return std::nullopt;
}
std::optional<bool> getSyncBarrierAvailability(StringRef architecture, SyncPhysicalCore core, PIPE pipe)
{
    auto present = getSyncPipelinePresence(architecture, core, pipe);
    if (!present) {
        return std::nullopt;
    }
    return *present && pipe != PIPE::PIPE_S;
}
SyncEventCapability getSyncEventAvailability(StringRef architecture, SyncPhysicalCore core, PIPE source, PIPE target)
{
    auto a = getSyncPipelinePresence(architecture, core, source);
    auto b = getSyncPipelinePresence(architecture, core, target);
    if (!a || !b) {
        return {};
    }
    if (!*a || !*b || source == target) {
        return {false};
    }
    if (core == SyncPhysicalCore::AIV) {
        return {true};
    }
    if (source == PIPE::PIPE_S || target == PIPE::PIPE_S ||
        (source == PIPE::PIPE_M && target == PIPE::PIPE_MTE3) ||
        (source == PIPE::PIPE_MTE3 && target == PIPE::PIPE_M)) {
        return {false};
    }
    bool noScenario = (target == PIPE::PIPE_FIX && (source == PIPE::PIPE_MTE2 || source == PIPE::PIPE_MTE3)) ||
                      (source == PIPE::PIPE_FIX && (target == PIPE::PIPE_MTE2 || target == PIPE::PIPE_MTE3));
    return {true, noScenario};
}
} // namespace mlir::pto
