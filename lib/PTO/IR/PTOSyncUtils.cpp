// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOSyncUtils.cpp - Shared sync mapping helpers --------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTOSyncUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::pto;

std::string SetFFTsOp::getSyncConfigurationGap() {
  auto function = (*this)->getParentOfType<func::FuncOp>();
  if (!function || function.getBody().empty())
    return "FFTS configuration requires the original function entry block";
  SetFFTsOp initial;
  for (auto &op : function.getBody().front())
    if (auto candidate = dyn_cast<SetFFTsOp>(&op)) { initial = candidate; break; }
  if (!initial)
    return "FFTS configuration requires an unconditional entry setup";
  bool sameAddress = true;
  function.walk([&](SetFFTsOp candidate) {
    sameAddress &= candidate.getFfts() == initial.getFfts();
  });
  if (!sameAddress)
    return "FFTS reconfiguration requires a scoped configuration contract";
  // The lowering is set_ffts_base_addr(pointer), not a byte load/store or a
  // local drain. Only admit initial setup before any issued payload/protocol.
  for (auto &previous : function.getBody().front()) {
    if (&previous == initial.getOperation()) break;
    if (previous.getNumRegions() ||
        (!isMemoryEffectFree(&previous) &&
         !isa<SyncStorageOpInterface>(&previous)))
      return "FFTS setup follows work requiring a configuration lifetime";
  }
  // Repeated original writes of the same SSA address preserve the configured
  // value. Keep them in place, including loop/choice occurrences; do not use
  // them as a completion or cross-core synchronization mechanism.
  return std::string{};
}

FailureOr<SyncOpType> mlir::pto::parseSyncOpTypeLikeAttr(Attribute attr) {
  if (auto a = dyn_cast_or_null<PipeEventTypeAttr>(attr)) {
    return a.getOpType();
  }
  if (auto a = dyn_cast_or_null<SyncOpTypeAttr>(attr)) {
    return a.getOpType();
  }
  return failure();
}

PIPE mlir::pto::mapSyncOpTypeToPipe(SyncOpType opType) {
  switch (opType) {
  case SyncOpType::TLOAD:
    return PIPE::PIPE_MTE2;
  case SyncOpType::TSTORE_VEC:
    return PIPE::PIPE_MTE3;
  case SyncOpType::TSTORE_ACC:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2L:
  case SyncOpType::TMOV_M2B:
    return PIPE::PIPE_MTE1;
  case SyncOpType::TMOV_M2S:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2V:
    return PIPE::PIPE_V;
  case SyncOpType::TMOV_V2M:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMATMUL:
    return PIPE::PIPE_M;
  case SyncOpType::TVEC:
  case SyncOpType::TVECWAIT_EVENT:
    return PIPE::PIPE_V;
  }
  return PIPE::PIPE_UNASSIGNED;
}

bool mlir::pto::isConcreteSyncPipe(PIPE pipe) {
  return pipe != PIPE::PIPE_UNASSIGNED && pipe != PIPE::PIPE_ALL;
}
