// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_SYNC_ADDRESS_ANALYSIS_H
#define PTO_SYNC_ADDRESS_ANALYSIS_H
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <string>
namespace mlir::pto {
// Constant/cast semantics only; unsupported arithmetic is unknown, never an
// identity cast. Memoized for one immutable function, including unknown values.
// PTO EmitC uses signed 64-bit index. Conflicting data-layout widths decline
// index interpretation instead of silently assuming the host width.
class SyncAddressEvaluator {
public:
  explicit SyncAddressEvaluator(func::FuncOp function);
  std::optional<llvm::APInt> evaluate(Value value);
  std::optional<int64_t> signedValue(Value value);
  uint64_t visits() const { return visited; }
private:
  std::optional<llvm::APInt> evaluate(Value value, unsigned depth);
  std::optional<unsigned> width(Type type) const;
  llvm::DenseMap<Value, std::optional<llvm::APInt>> cache;
  bool index64 = false;
  uint64_t visited = 0;
};
// Same conservative footprint used for ordinary translated tile effects.
// Returns zero for unrepresentable, nonstatic or empty geometry.
uint64_t getSyncTileFootprintBytes(TileBufType type);
struct SyncAddressAdmission {
  enum Status { Safe, Conservative, Rejected } status = Safe;
  std::string reason;
};
// Independent of planner support. Unknown explicit addresses are safe only
// with conservative same-space aliasing; invalid known intervals reject both
// constructors. Authored-event policy is handled by the pass before this call.
SyncAddressAdmission qualifySyncPhysicalAddresses(func::FuncOp function);
} // namespace mlir::pto
#endif
