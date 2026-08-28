// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace pto {

struct SyncMacroPhase {
  unsigned phaseId{0};
  PipelineType pipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<Value> defValues;
  SmallVector<Value> useValues;
  /// Operation result numbers whose SSA values become complete in this phase.
  /// Every result of a multi-phase macro must be named by exactly one phase.
  SmallVector<unsigned> completedResults;
};

struct SyncMacroHiddenEvent {
  PipelineType srcPipe{PipelineType::PIPE_UNASSIGNED};
  PipelineType dstPipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<unsigned> eventIds;
};

struct SyncMacroModel {
  SmallVector<SyncMacroPhase> phases;
  SmallVector<SyncMacroHiddenEvent> hiddenEvents;

  explicit operator bool() const { return !phases.empty(); }
};

std::optional<SyncMacroModel> getSyncMacroModel(Operation *op);

/// Verify that phase identifiers are unique and every operation result has one
/// authoritative completion phase. Diagnostics are emitted on `op`.
LogicalResult verifySyncMacroModel(Operation *op, const SyncMacroModel &model);

/// Return the authoritative completion phase for a verified macro result.
std::optional<unsigned>
getSyncMacroResultCompletionPhase(const SyncMacroModel &model,
                                  unsigned resultNumber);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H
