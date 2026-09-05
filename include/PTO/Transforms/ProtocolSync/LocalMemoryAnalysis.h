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
// Production, pattern-independent requirements over a shared vector UB atom
// partition. Region transfers retain incoming/outgoing outstanding effects.
// A domain is admitted atomically: no timeline protection is removed if
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

#include <cstddef>

namespace mlir::pto::protocol_sync {

enum class SyncLocalExpandedStateStatus { NotRequested, Complete, LimitExceeded };

struct SyncLocalFlowOptions {
    /// Diagnostic only: production uses sparse predecessor/outstanding links.
    bool expandState = false;
    std::size_t maximumExpandedEntries = 1048576;
};

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
    /// Expanded sets are meaningful only when expandedStateStatus is Complete.
    llvm::SmallVector<std::uint32_t, 4> mayDefinitions;
    llvm::SmallVector<std::uint32_t, 2> mustDefinitions;
    SyncRegionId region = kInvalidSyncId;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
    bool mayHaveLiveIn = true;
};

/// Semantic may-definitions are separate from the sparse outstanding-access
/// frontier. Retiring a frontier only adds mandatory ordering; it is not a kill.
struct SyncLocalAtomFlow {
    llvm::SmallVector<std::uint32_t, 4> mayDefinitions;
    llvm::SmallVector<std::uint32_t, 2> mustDefinitions;
    llvm::SmallVector<SyncAccessId, 2> outstandingWrites;
    llvm::SmallVector<SyncAccessId, 4> outstandingReaders;
    /// A conservative write does not prove all incoming bytes were replaced.
    bool mayHaveLiveIn = true;
};

struct SyncLocalRegionSummary {
    SyncRegionId region = kInvalidSyncId;
    std::uint32_t atom = kInvalidSyncId;
    SyncLocalAtomFlow incoming;
    SyncLocalAtomFlow outgoing;
    /// Reads up to the first write, and that write. These are entry obligations
    /// when composing with a predecessor, not assertions of definite access.
    llvm::SmallVector<SyncAccessId, 4> firstAccesses;
    llvm::SmallVector<SyncObligationId, 8> requirements;
    /// Every path issues a write access; this is NOT a definite byte-set kill.
    bool writesOnEveryPath = false;
    bool complete = false;
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
    llvm::SmallVector<SyncLocalRegionSummary, 8> regionSummaries;
    llvm::SmallVector<SyncResidualObligation, 16> requirements;
    SyncLocalExpandedStateStatus expandedStateStatus = SyncLocalExpandedStateStatus::NotRequested;
    std::string boundary;
};

/// Recover a conservative access footprint from addressed IR, independently of
/// the legacy allocation map. Unknown aliases/views are not invented intervals.
SyncLocalAccessRegion recoverLocalAccessRegion(const SyncAccess& access);
FailureOr<SyncLocalMemoryAnalysis> analyzeLocalMemory(
    const StructuredSyncIR& schedule, SyncLocalFlowOptions options = {});

/// Compose transfers over an already recovered, shared physical atom partition.
/// Requires a frozen schedule and empty states, summaries, and requirements.
/// Unsupported boundary facts do not authorize removing timeline protection.
/// Expansion limits return success with LimitExceeded and no partial output.
/// Without expansion, the caller must establish a single-block/guard domain.
LogicalResult analyzeLocalRegionFlow(
    const StructuredSyncIR& schedule, SyncLocalMemoryAnalysis& result, SyncLocalFlowOptions options = {});

/// Exhaustive, non-atomized reference check. Does not consume sparse states,
/// requirements, candidate IDs, or planner coverage. The world must have been
/// reconstructed from concrete synchronization by the caller.
LogicalResult verifyLocalMemoryCoverage(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& concreteWorld, SyncAccessId* uncoveredSource = nullptr,
    SyncAccessId* uncoveredTarget = nullptr);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LOCALMEMORYANALYSIS_H
