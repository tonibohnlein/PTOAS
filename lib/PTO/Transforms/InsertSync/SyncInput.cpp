// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include <map>
#include <climits>
namespace mlir::pto {
bool isPreservedSyncProtocol(Operation* operation) {
  // These explicit visibility boundaries remain in place. They confer no
  // additional reachability credit to the demand reducer.
  if (isa<CmoCacheInvalidOp, FenceBarrierAllOp>(operation)) { return true; }
  if (auto sync = dyn_cast<SyncAllOp>(operation); sync && !sync.getGmWorkspace()) { return true; }
  return isa<BarrierOp, SyncSetOp, SyncWaitOp, SetCrossBlockOp, WaitCrossBlockOp,
             SetIntraBlockOp, WaitIntraBlockOp, GetBufOp, RlsBufOp, GetBufDynOp, RlsBufDynOp>(operation);
}
bool hasManualOnCoreSynchronization(func::FuncOp function) {
  if (!function) {
    return false;
  }
  return function.walk([](Operation* op) {
    return isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp,
               RecordEventOp, WaitEventOp>(op)
        ? WalkResult::interrupt() : WalkResult::advance();
  }).wasInterrupted();
}

SyncResultAvailability resultAvailability(const CompoundInstanceElement& phase, Value result) {
  if (!result || result.getDefiningOp() != phase.elementOp ||
      !isa<IntegerType, IndexType, FloatType>(result.getType())) {
    return SyncResultAvailability::NonScalar;
  }
  return phase.macroOpInstanceId < 0 && phase.kPipeValue == PipelineType::PIPE_S
      ? SyncResultAvailability::SynchronousScalar : SyncResultAvailability::RequiresCompletion;
}
SyncInput::SyncInput(GMAliasPolicy policy)
    : analyzer(policy), resolvedAccesses(std::make_unique<SyncStorageEffects>()) {}
SyncInput::~SyncInput() = default;
const SyncStorageEffects& SyncInput::accesses() const { return *resolvedAccesses; }
LogicalResult SyncInput::build(func::FuncOp function) {
  return build(function, SyncInstructionView::TranslatorStages);
}
LogicalResult SyncInput::build(func::FuncOp function, SyncInstructionView view) {
  resolvedAccesses = std::make_unique<SyncStorageEffects>();
  declaredEffects.clear();
  phases.clear();
  macroEnvelopes.clear();
  nodes.clear();
  storage.clear();
  if (function.isDeclaration()) {
    return success();
  }
  PTOIRTranslator translator(nodes, storage, function, SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build())) {
    return failure();
  }
  if (view == SyncInstructionView::TranslatorStages) {
    for (const auto& node : nodes) {
      if (auto* phase = dyn_cast<CompoundInstanceElement>(node.get())) { phases.push_back(phase); }
    }
  } else {
    if (failed(buildPipeEnvelopes(function))) { return failure(); }
  }
  function.walk([&](Operation* operation) {
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      interface.getEffects(declaredEffects[operation]);
    }
  });
  return resolvedAccesses->build(*this);
}
LogicalResult SyncInput::buildPipeEnvelopes(func::FuncOp function) {
  // The shared macro descriptions are access envelopes, not a serial list of
  // payloads. Export one envelope per (call, pipe), spanning all commands on
  // that pipe, including hidden event use. Never mutate translator records:
  // the existing pass consumes their ordered stages through ir().
  std::map<Operation*, SmallVector<CompoundInstanceElement*>> envelopes;
  const auto originalSize = nodes.size();
  for (std::size_t i = 0; i < originalSize; ++i) {
    auto* phase = dyn_cast<CompoundInstanceElement>(nodes[i].get());
    if (!phase) { continue; }
    if (phase->macroOpInstanceId < 0) { phases.push_back(phase); continue; }
    auto& owned = envelopes[phase->elementOp];
    auto samePipe = llvm::find_if(owned, [&](auto* previous) { return previous->kPipeValue == phase->kPipeValue; });
    if (samePipe != owned.end()) {
      for (auto memory : phase->defVec) {
        if (!llvm::is_contained((*samePipe)->defVec, memory)) { (*samePipe)->defVec.push_back(memory); }
      }
      for (auto memory : phase->useVec) {
        if (!llvm::is_contained((*samePipe)->useVec, memory)) { (*samePipe)->useVec.push_back(memory); }
      }
    } else {
      auto envelope = std::make_unique<CompoundInstanceElement>(*phase);
      owned.push_back(envelope.get());
      phases.push_back(envelope.get());
      macroEnvelopes.push_back(std::move(envelope));
    }
  }
  // Add empty access envelopes for pipes used only by the hidden protocol.
  // Insert them next to their call, never at the end of the reference word.
  SmallVector<const CompoundInstanceElement*> complete;
  for (std::size_t i = 0; i < phases.size(); ++i) {
    auto* phase = phases[i];
    complete.push_back(phase);
    if (phase->macroOpInstanceId < 0 ||
        (i + 1 < phases.size() && phases[i+1]->elementOp == phase->elementOp)) { continue; }
    auto model = getSyncMacroModel(phase->elementOp);
    if (!model) { return phase->elementOp->emitError("shared macro phases have no owning contract"); }
    auto& owned = envelopes[phase->elementOp];
    for (const auto& reservation : model->hiddenEvents) {
      for (auto pipe : {reservation.srcPipe, reservation.dstPipe}) {
        if (llvm::any_of(owned, [&](auto* value) { return value->kPipeValue == pipe; })) { continue; }
        if (nodes.size() >= UINT_MAX || macroEnvelopes.size() >= UINT_MAX - nodes.size()) {
          return function.emitError("shared envelope identity overflow");
        }
        auto extra = std::make_unique<CompoundInstanceElement>(nodes.size() + macroEnvelopes.size(),
            SmallVector<const BaseMemInfo*>(),
            SmallVector<const BaseMemInfo*>(), pipe, phase->opName);
        extra->elementOp = phase->elementOp;
        extra->macroOpInstanceId = owned.size();
        extra->compoundCoreType = phase->compoundCoreType;
        owned.push_back(extra.get());
        complete.push_back(extra.get());
        macroEnvelopes.push_back(std::move(extra));
      }
    }
  }
  phases = std::move(complete);
  return success();
}
ArrayRef<SyncMemoryEffect> SyncInput::effectsFor(const CompoundInstanceElement& phase) const {
  if (phase.macroOpInstanceId >= 0) {
    return {};
  }
  return effectsFor(phase.elementOp);
}
ArrayRef<SyncMemoryEffect> SyncInput::effectsFor(Operation* operation) const {
  auto found = declaredEffects.find(operation);
  return found == declaredEffects.end() ? ArrayRef<SyncMemoryEffect>{} : ArrayRef<SyncMemoryEffect>(found->second);
}
} // namespace mlir::pto
