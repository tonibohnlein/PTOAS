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
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace pto {

struct SyncMacroPhase {
  unsigned phaseId{0};
  PipelineType pipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<Value> defValues;
  SmallVector<Value> useValues;
};

struct SyncMacroHiddenEvent {
  PipelineType srcPipe{PipelineType::PIPE_UNASSIGNED};
  PipelineType dstPipe{PipelineType::PIPE_UNASSIGNED};
  SmallVector<unsigned> eventIds;
};

// A lowering-owned completion edge between ordered opaque phases. Hidden
// event ownership alone does not imply this edge or expose final completion.
struct SyncMacroCompletionTransfer {
  unsigned sourcePhaseId{0};
  unsigned targetPhaseId{0};
};

struct SyncMacroModel {
  SmallVector<SyncMacroPhase> phases;
  SmallVector<SyncMacroHiddenEvent> hiddenEvents;
  SmallVector<SyncMacroCompletionTransfer> completionTransfers;
  // The lowering owns the core on which its opaque implementation is valid.
  // CUBE_OR_VECTOR leaves the choice to ordinary per-phase lane validation.
  TCoreType coreType{TCoreType::CUBE_OR_VECTOR};

  explicit operator bool() const { return !phases.empty(); }
};

enum class SyncOperationSemanticKind {
  Unsupported,
  SinglePhaseOrdinary,
  Macro,
  ImplicitResource,
  TypedEffect,
};

/// Complete operation-local contract consumed by synchronization planners.
/// Ordinary operations are classified generically from their lowering-owned
/// pipe and memory-effect interfaces. Only operations with genuinely distinct
/// lowering semantics receive a macro, implicit-resource, or typed-effect
/// classification.
struct SyncOperationSemantics {
  SyncOperationSemanticKind kind{SyncOperationSemanticKind::Unsupported};
  PipelineType pipe{PipelineType::PIPE_UNASSIGNED};
  std::optional<SyncMacroModel> macro;
};

inline bool supportsSyncMacroCore(const SyncMacroModel &model,
                                  TCoreType core) {
  return model.coreType == TCoreType::CUBE_OR_VECTOR ||
         model.coreType == core;
}

std::optional<SyncMacroModel> getSyncMacroModel(Operation *op);
SyncOperationSemantics getSyncOperationSemantics(Operation *op);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_SYNCMACROMODEL_H
