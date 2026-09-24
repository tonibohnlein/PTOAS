// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared owning input for synchronization construction. This is the translator's
// instruction/control representation, not a second effects registry or a plan.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCINPUT_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCINPUT_H

#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"

namespace mlir::pto {
class SyncInput {
public:
  SyncInput() = default;
  SyncInput(const SyncInput &) = delete;
  SyncInput &operator=(const SyncInput &) = delete;
  SyncInput(SyncInput &&) = delete;
  SyncInput &operator=(SyncInput &&) = delete;

  // The source function must outlive this input and stay unchanged while its
  // records are consumed. Failure exposes no partially translated records.
  LogicalResult build(func::FuncOp function);
  SyncIRs &ir() { return nodes; }
  const SyncIRs &ir() const { return nodes; }
  MemoryDependentAnalyzer &memory() { return aliases; }
  const MemoryDependentAnalyzer &memory() const { return aliases; }
  const Buffer2MemInfoMap &buffers() const { return storage; }
  // Every translated phase is retained, including multiple phases belonging
  // to one original operation. Views/aliases remain owned by this input.
  ArrayRef<const CompoundInstanceElement *> instructions() const { return phases; }

private:
  MemoryDependentAnalyzer aliases;
  Buffer2MemInfoMap storage;
  SyncIRs nodes;
  SmallVector<const CompoundInstanceElement *> phases;
};
} // namespace mlir::pto
#endif
