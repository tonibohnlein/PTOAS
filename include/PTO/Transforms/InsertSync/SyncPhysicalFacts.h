// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCPHYSICALFACTS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCPHYSICALFACTS_H
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include <string>
#include <vector>
namespace mlir::pto {
struct SyncPhysicalFacts {
  enum class Status { Complete, Unsupported, AnalysisLimit, InternalError };
  Status status = Status::Unsupported;
  std::string reason;
  bool cube = false;
  Operation *lifetimeScope = nullptr;
  std::vector<const CompoundInstanceElement *> phases;
  uint64_t work = 0;
};
// Qualification of translated phases only: no lifecycle graph or planner.
SyncPhysicalFacts importSyncPhysicalFacts(func::FuncOp function, const SyncIRs &ir, uint64_t budget);
// Reject unsupported memory forwarding before using the upstream translator.
bool supportsLogicalSyncTranslation(func::FuncOp function);
// Conservative physical overlap with a separately qualified caller contract.
bool logicalSyncMayAlias(const BaseMemInfo *a, const BaseMemInfo *b,
                         func::FuncOp function, InsertSyncGMAliasMode gm);
}
#endif
