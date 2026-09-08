// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_MMADCHAINANALYSIS_H
#define PTO_TRANSFORMS_INSERTSYNC_MMADCHAINANALYSIS_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <string>

namespace mlir::pto {
// Read-only, optional analysis over the existing IR. An unknown operation or
// unsupported region breaks a chain; it never rejects compiler admission.
// The only claim is a target-qualified ACC dependency, not full completion.
class MmadChainAnalysis {
public:
  explicit MmadChainAnalysis(func::FuncOp function);
  bool discharges(const CompoundInstanceElement *source,
                  const CompoundInstanceElement *target,
                  const DepBaseMemInfoPairVec &dependencies) const;
  bool isComplete() const { return complete; }
  bool dischargesWithPredecessors(const CompoundInstanceElement *source,
                  const CompoundInstanceElement *target,
                  const DepBaseMemInfoPairVec &dependencies,
                  ArrayRef<Operation*> immediatePredecessors) const;
  unsigned getEligibleTargets() const { return eligible.size(); }
  unsigned getTransferCount() const { return transfers; }
  StringRef getReason() const { return reason; }

private:
  bool qualifiedPair(const CompoundInstanceElement *source,
                     const CompoundInstanceElement *target,
                     const DepBaseMemInfoPairVec &dependencies) const;
  bool complete = false;
  unsigned transfers = 0;
  std::string reason;
  llvm::DenseMap<Operation *, unsigned> keys;
  llvm::DenseMap<Operation *, Value> accumulators;
  llvm::SmallPtrSet<Operation *, 32> eligible;
};
} // namespace mlir::pto
#endif
