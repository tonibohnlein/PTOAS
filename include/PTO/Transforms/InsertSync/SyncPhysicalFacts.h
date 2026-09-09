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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <string>
#include <vector>
#include <utility>
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

struct SyncPhysicalAccess {
  unsigned phase;
  const BaseMemInfo *memory;
  bool write;
  PipelineType lane;
};
struct SyncAccessCandidates {
  enum class Status { Complete, InvalidInput, AnalysisLimit };
  Status status = Status::InvalidInput;
  // Unordered input-access indices, sorted and unique; self-pairs retain WAW
  // between repeated occurrences. Partial results are never returned.
  std::vector<std::pair<unsigned, unsigned>> pairs;
  uint64_t work = 0;
  uint64_t intervalVisits = 0;
  uint64_t candidateVisits = 0;
};
// Conservative candidate enumeration only. Callers still apply alias/contract
// and occurrence queries. Unknown geometry may add candidates, never remove
// them. Read/read is relevant only for ACC resources on different lanes.
// spend charges the caller's budget directly; do not charge result.work again.
SyncAccessCandidates enumerateSyncAccessCandidates(
    ArrayRef<SyncPhysicalAccess> accesses, llvm::function_ref<bool(uint64_t)> spend);
}
#endif
