// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LocalMemoryAnalysis.h - Canonical ordinary local effects -*- C++ -*-===//
//
// Production, pattern-independent requirements for one straight-line vector UB
// domain. A domain is admitted atomically: no timeline protection is removed if
// an access, alias, control instance, or completion frontier is unmodeled.
// Conservative accesses create may-definitions, never definite generation kills.
// Synchronization requirements summarize outstanding effects through explicit
// completion chains; replacing a frontier does not imply physical completion.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_LOCALMEMORYANALYSIS_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_LOCALMEMORYANALYSIS_H

#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::pto::protocol_sync {

struct SyncLocalAccessRegion {
    SyncAccessId access = kInvalidSyncId;
    SyncByteInterval interval;
    SyncRegionPrecision precision = SyncRegionPrecision::Unknown;
};

struct SyncLocalStorageAtom {
    std::uint32_t id = kInvalidSyncId;
    SyncByteInterval interval;
    llvm::SmallVector<SyncAccessId, 4> accesses;
};

struct SyncLocalMemoryState {
    std::uint32_t id = kInvalidSyncId;
    std::uint32_t atom = kInvalidSyncId;
    SyncPhaseId phase = kInvalidSyncId;
    /// Previous may-definition, not necessarily the unique reaching writer.
    std::uint32_t previousDefinition = kInvalidSyncId;
    bool reads = false;
    bool writes = false;
    SyncRegionPrecision precision = SyncRegionPrecision::Unknown;
};

struct SyncLocalMemoryAnalysis {
    /// Physical domain: this function's vector core UB. Different logical
    /// allocations at the same address deliberately share atoms.
    Operation* scope = nullptr;
    Block* block = nullptr;
    llvm::BitVector coveredAccesses;
    llvm::SmallVector<SyncLocalAccessRegion, 16> regions;
    llvm::SmallVector<SyncLocalStorageAtom, 16> atoms;
    llvm::SmallVector<SyncLocalMemoryState, 32> states;
    llvm::SmallVector<SyncResidualObligation, 16> requirements;
    std::string boundary;
};

/// Recover a conservative access footprint from addressed IR, independently of
/// the legacy allocation map. Unknown aliases/views are not invented intervals.
SyncLocalAccessRegion recoverLocalAccessRegion(const SyncAccess& access);
FailureOr<SyncLocalMemoryAnalysis> analyzeLocalMemory(const StructuredSyncIR& schedule);

/// Exhaustive, non-atomized reference check. Does not consume sparse states,
/// requirements, candidate IDs, or planner coverage. The world must have been
/// reconstructed from concrete synchronization by the caller.
LogicalResult verifyLocalMemoryCoverage(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& concreteWorld, SyncAccessId* uncoveredSource = nullptr,
    SyncAccessId* uncoveredTarget = nullptr);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LOCALMEMORYANALYSIS_H
