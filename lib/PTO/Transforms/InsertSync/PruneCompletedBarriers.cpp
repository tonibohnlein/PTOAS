// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/InsertSync/PruneCompletedBarriers.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/CompletionPrefixState.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {
std::optional<unsigned> ordinaryLane(PIPE pipe) {
  switch (pipe) {
  case PIPE::PIPE_MTE2:
    return 0;
  case PIPE::PIPE_V:
    return 1;
  case PIPE::PIPE_MTE3:
    return 2;
  default:
    return std::nullopt;
  }
}
} // namespace

CompletedBarrierPruningResult
mlir::pto::pruneProvenCompletedBarriers(func::FuncOp function) {
  auto module = function->getParentOfType<ModuleOp>();
  auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch") : StringAttr();
  auto core = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
  if (!arch || (arch.getValue() != "a2" && arch.getValue() != "a3") ||
      !core || core.getKernelKind() != FunctionKernelKind::Vector ||
      !llvm::hasSingleElement(function.getBody())) {
    return {0, "unchanged: requires explicit A2/A3 vector context and one block"};
  }

  // The first rollout supports only the four explicit ordinary tile primitives
  // below, not every operation advertising one of these pipes. In particular,
  // a new/unregistered multi-phase operation cannot authorize barrier deletion.
  // Analyze first, erase last. A later unsupported operation or unproved key
  // lifetime discards all proposed removals; this is never an admission gate.
  insert_sync_detail::CompletionPrefixState<3> state;
  SmallVector<Operation *> removable;
  bool terminalDrain = false;
  unsigned actions = 0;
  for (Operation &op : function.getBody().front()) {
    if (++actions > 4096 || op.getNumRegions() || getSyncMacroModel(&op)) {
      return {0, "unchanged: structured, macro, or over-budget domain"};
    }
    if (auto set = dyn_cast<SetFlagOp>(op)) {
      auto source = ordinaryLane(set.getSrcPipe().getPipe());
      auto target = ordinaryLane(set.getDstPipe().getPipe());
      unsigned id = static_cast<unsigned>(set.getEventId().getEvent());
      if (terminalDrain || !source || !target || id >= 8 ||
          !state.signal(*source, *target, id)) {
        return {0, "unchanged: signal domain or consumption-before-rearm unproved"};
      }
    } else if (auto wait = dyn_cast<WaitFlagOp>(op)) {
      auto source = ordinaryLane(wait.getSrcPipe().getPipe());
      auto target = ordinaryLane(wait.getDstPipe().getPipe());
      unsigned id = static_cast<unsigned>(wait.getEventId().getEvent());
      if (terminalDrain || !source || !target || id >= 8 ||
          !state.wait(*source, *target, id)) {
        return {0, "unchanged: acquisition domain or matching generation unproved"};
      }
    } else if (auto barrier = dyn_cast<BarrierOp>(op)) {
      if (barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
        // Fixed policy and scarcity barriers are never removed or interpreted
        // as a source of body-wide completion for this optimization.
        terminalDrain = true;
        continue;
      }
      auto pipe = ordinaryLane(barrier.getPipe().getPipe());
      if (terminalDrain || !pipe) {
        return {0, "unchanged: unsupported barrier placement"};
      }
      if (state.canOmitBarrier(*pipe)) {
        removable.push_back(&op);
      } else if (!state.keepBarrier(*pipe)) {
        return {0, "unchanged: completion counter limit"};
      }
    } else if (auto physical = dyn_cast<OpPipeInterface>(op)) {
      auto pipe = ordinaryLane(physical.getPipe());
      if (terminalDrain || !pipe ||
          !isa<TLoadOp, TStoreOp, TAbsOp, TAddOp>(op) ||
          !isa<MemoryEffectOpInterface>(op) || !state.issue(*pipe)) {
        return {0, "unchanged: unqualified physical phase"};
      }
    } else if (!isa<AllocTileOp, func::ReturnOp>(op) && !isMemoryEffectFree(&op)) {
      return {0, "unchanged: unmodeled effect"};
    }
  }
  if (!state.allEventsConsumed()) {
    return {0, "unchanged: unmatched event at boundary"};
  }
  unsigned count = removable.size();
  for (Operation *op : removable) {
    op->erase();
  }
  return {count, "whole source-prefix completion already acquired; event order unchanged"};
}

namespace {
// Standalone entry is for native regression/differential tests. Production use
// is the opt-in call from InsertSync after event assignment and emission.
struct PTOPruneCompletedSyncBarriersPass
    : PassWrapper<PTOPruneCompletedSyncBarriersPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOPruneCompletedSyncBarriersPass)
  StringRef getArgument() const final { return "pto-prune-completed-sync-barriers"; }
  StringRef getDescription() const final {
    return "Remove named barriers with existing complete source-prefix supply";
  }
  void runOnOperation() override {
    auto result = pruneProvenCompletedBarriers(getOperation());
    getOperation().emitRemark("InsertSync completed-prefix pruning: ")
        << result.removed << " barriers removed; " << result.reason;
  }
};
PassRegistration<PTOPruneCompletedSyncBarriersPass> registerCompletedPruning;
} // namespace
