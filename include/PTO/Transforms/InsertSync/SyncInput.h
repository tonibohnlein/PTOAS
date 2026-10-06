// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared owner of translated instruction phases and memory records.
// This extracts effects and pipe assignments, without dependency analysis.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCINPUT_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCINPUT_H
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto {
using SyncMemoryEffect = SideEffects::EffectInstance<MemoryEffects::Effect>;
// Scalar-pipe value operations return an ordinary value only after the scalar
// access completes. This does not establish readiness of the accessed memory.
enum class SyncResultAvailability { NonScalar, SynchronousScalar, RequiresCompletion };
SyncResultAvailability resultAvailability(const CompoundInstanceElement& phase, Value result);

class SyncStorageEffects;

class SyncInput {
public:
  explicit SyncInput(GMAliasPolicy policy = GMAliasPolicy::MayNotAlias);
  ~SyncInput();
  SyncInput(const SyncInput &) = delete;
  SyncInput &operator=(const SyncInput &) = delete;
  SyncInput(SyncInput &&) = delete;
  SyncInput &operator=(SyncInput &&) = delete;
  // Source MLIR must outlive the borrowed operation and value anchors.
  // Failure exposes no partial records. Geometry is a summary, not a full-write proof.
  LogicalResult build(func::FuncOp function);
  SyncIRs &ir() { return nodes; }
  const SyncIRs &ir() const { return nodes; }
  const Buffer2MemInfoMap &buffers() const { return storage; }
  ArrayRef<const CompoundInstanceElement *> instructions() const { return phases; }
  // Original MLIR effect declarations, including coverage, stage and resource.
  // Macro phases need phase-specific declarations and therefore return none.
  ArrayRef<SyncMemoryEffect> effectsFor(const CompoundInstanceElement& phase) const;
  // Retain declarations even for operations with no translated pipe phase.
  // Their classification remains explicit work for the consuming analysis.
  ArrayRef<SyncMemoryEffect> effectsFor(Operation* operation) const;
  const SyncStorageEffects &accesses() const;
  MemoryDependentAnalyzer &memory() { return analyzer; }
  const MemoryDependentAnalyzer &memory() const { return analyzer; }
private:
  MemoryDependentAnalyzer analyzer;
  Buffer2MemInfoMap storage;
  SyncIRs nodes;
  SmallVector<const CompoundInstanceElement *> phases;
  DenseMap<Operation*, SmallVector<SyncMemoryEffect>> declaredEffects;
  std::unique_ptr<SyncStorageEffects> resolvedAccesses;
};
} // namespace mlir::pto
#endif
