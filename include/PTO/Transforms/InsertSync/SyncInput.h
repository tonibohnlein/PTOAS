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
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto {
class SyncInput {
public:
  SyncInput() = default;
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
private:
  Buffer2MemInfoMap storage;
  SyncIRs nodes;
  SmallVector<const CompoundInstanceElement *> phases;
};
} // namespace mlir::pto
#endif
