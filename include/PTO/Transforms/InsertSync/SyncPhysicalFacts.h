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
#include <optional>
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
// Same operation/address qualification, without heuristic visitation ceilings.
// Structured construction terminates by its finite admitted model, not a work
// quota. This does not change the historical reference-import entry above.
SyncPhysicalFacts importStructuredSyncPhysicalFacts(func::FuncOp function, const SyncIRs &ir);
// Reject unsupported memory forwarding before using the upstream translator.
bool supportsLogicalSyncTranslation(func::FuncOp function);
// Conservative physical overlap with a separately qualified caller contract.
bool logicalSyncMayAlias(const BaseMemInfo *a, const BaseMemInfo *b,
                         func::FuncOp function, InsertSyncGMAliasMode gm);

struct SyncPhysicalSlotMapping {
  // Absent for fixed conservative physical intervals. Otherwise bases retains
  // the allocation's ORIGINAL slot order, even for a literal selector.
  Value selector;
  SmallVector<uint64_t> bases;
  uint64_t bytes = 0;
  AddressSpace scope = AddressSpace::Zero;
};
// Qualify existing translated geometry against actual multi_tile_get lowering.
// Does not interpret the selector or assume modulo/count bounds. Nullopt keeps
// the original may-conflict relation. Transparent reshape/bitcast paths retain
// a conservative whole-slot footprint; other forwarding is unavailable.
// Bounded by the import's 2048 physical fragments and 32 forwarding hops. No
// solver is invoked; callers charge the estimate below before qualification.
std::optional<SyncPhysicalSlotMapping> qualifySyncPhysicalSlots(
    const BaseMemInfo *memory, const Buffer2MemInfoMap &buffers);
// Allocation-free bounded preflight accounting, including the root slot table
// sorted by qualification even when a literal get has only one access range.
// Counts beyond qualification's hard limit need no larger allocation/sort:
// qualification declines them before walking those ranges. No buffers mutate.
uint64_t estimateSyncPhysicalSlotQualificationWork(
    const BaseMemInfo *memory, const Buffer2MemInfoMap &buffers);
// Exact geometric overlap for two QUALIFIED mappings, in original slot/index
// coordinates in deterministic sweep order. Sweep work is
// O((N+M) log(N+M)+K), where K is returned overlap. Charges setup before
// allocation and each overlap before appending it. Nullopt means the caller's
// budget refused; no partial pairs escape. This does not interpret selectors.
std::optional<std::vector<std::pair<unsigned, unsigned>>> overlappingSyncPhysicalSlots(
    const SyncPhysicalSlotMapping &a, const SyncPhysicalSlotMapping &b,
    llvm::function_ref<bool(uint64_t)> spend);

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
