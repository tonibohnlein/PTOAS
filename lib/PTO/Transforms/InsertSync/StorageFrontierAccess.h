// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_INSERTSYNC_STORAGEFRONTIERACCESS_INTERNAL_H
#define PTO_INSERTSYNC_STORAGEFRONTIERACCESS_INTERNAL_H

#include "PTO/Transforms/InsertSync/StorageFrontierDomain.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>
#include <vector>

namespace mlir::pto::insert_sync_frontier {
struct BoundGuard { Value upper; uint64_t limit = 0; };
struct GlobalSlice {
  Value root;
  Operation *carrier = nullptr;
  AffineSlice affine;
  std::optional<BoundGuard> guard;
  bool known = false;
};
struct AccessInfo {
  const BaseMemInfo *legacy = nullptr;
  bool write = false;
  GlobalSlice global;
};
struct PhaseInfo {
  const CompoundInstanceElement *legacy = nullptr;
  std::vector<AccessInfo> accesses;
};
struct OccurrenceProof {
  bool disjoint = false;
  std::vector<BoundGuard> guards;
};
GlobalSlice recoverFrontierGlobalSlice(Value value, Operation *access,
                                       func::FuncOp function);
OccurrenceProof compareFrontierOccurrences(const GlobalSlice &a, Operation *source,
                                           const GlobalSlice &b, Operation *target);
bool localFrontierOverlap(const BaseMemInfo &a, const BaseMemInfo &b);
} // namespace mlir::pto::insert_sync_frontier
#endif
