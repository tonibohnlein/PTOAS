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
namespace mlir::pto {
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
  resolvedAccesses = std::make_unique<SyncStorageEffects>();
  declaredEffects.clear();
  phases.clear();
  nodes.clear();
  storage.clear();
  if (function.isDeclaration()) {
    return success();
  }
  PTOIRTranslator translator(nodes, storage, function, SyncAnalysisMode::NORMALSYNC);
  if (failed(translator.Build())) {
    return failure();
  }
  for (const auto &node : nodes) {
    if (const auto *phase = dyn_cast<CompoundInstanceElement>(node.get())) {
      phases.push_back(phase);
    }
  }
  function.walk([&](Operation* operation) {
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(operation)) {
      interface.getEffects(declaredEffects[operation]);
    }
  });
  return resolvedAccesses->build(*this);
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
