// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- EventAllocation.h - Shared event lifetime allocation ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_EVENTALLOCATION_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_EVENTALLOCATION_H

#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace mlir::pto::protocol_sync {

using SyncEventGenerationId = std::uint32_t;

enum class SyncEventGenerationKind : std::uint8_t {
    OneShot,
    DirectRepair,
    ReadyReleaseReady,
    ReadyReleaseRelease,
};

/// One logical set/wait generation in a directed hardware-event domain.
/// For once-only generations, setAnchor and waitAnchor denote the physical
/// source/target operations after/before which the set and wait are emitted.
/// Concrete reconstruction instead supplies the actual set/wait operations.
/// These points support a bounded event-action consumption proof, never reuse
/// based on lexical cross-pipe wait-before-set order alone. Recurring
/// ReadyRelease lanes span their complete prime/body/drain protocol.
struct SyncEventGeneration {
    SyncEventGenerationId id = kInvalidSyncId;
    SyncEventGenerationKind kind = SyncEventGenerationKind::OneShot;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    Operation* setAnchor = nullptr;
    Operation* waitAnchor = nullptr;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncRegionId recurrenceOwner = kInvalidSyncId;
    bool recurring = false;
    std::optional<unsigned> eventId;
    /// Reserve a once-only boundary key through its enclosing region lifetime.
    /// This is allocation conservatism, not a claim of repeated execution.
    bool persistentReservation = false;
};

enum class SyncEventAllocationStatus : std::uint8_t {
    Allocated,
    ResourceInfeasible,
    AnalysisLimit,
};

/// Failure attribution is not a certificate permitting added serialization.
enum class SyncEventAllocationFailure : std::uint8_t {
    None,
    NoUnreservedIds,
    ConservativeInterference,
    AnalysisLimit,
};

llvm::StringRef stringifySyncEventAllocationFailure(SyncEventAllocationFailure reason);
llvm::StringRef describeSyncEventAllocationFailure(SyncEventAllocationFailure reason);

inline constexpr std::uint64_t kHardMaximumEventBacktrackingNodes = 1000000;
inline constexpr unsigned kHardMaximumExactEventVertices = 128;
inline constexpr unsigned kHardMaximumEventGenerationsPerDomain = 1024;

struct SyncEventAllocationOptions {
    /// Public options may lower, but never raise, these immutable safety caps.
    std::uint64_t maximumBacktrackingNodes = kHardMaximumEventBacktrackingNodes;
    unsigned maximumExactVertices = kHardMaximumExactEventVertices;
    unsigned maximumGenerationsPerDomain = kHardMaximumEventGenerationsPerDomain;
};

struct SyncEventAllocationResult {
    SyncEventAllocationStatus status = SyncEventAllocationStatus::Allocated;
    SyncEventAllocationFailure failureReason = SyncEventAllocationFailure::None;
    SyncPhysicalCore failedCore = SyncPhysicalCore::Unknown;
    PIPE failedSource = PIPE::PIPE_UNASSIGNED;
    PIPE failedTarget = PIPE::PIPE_UNASSIGNED;
    unsigned availableIds = 0;
    llvm::SmallVector<unsigned, 8> eventIds;
    std::uint64_t graphVertices = 0;
    std::uint64_t graphEdges = 0;
    std::uint64_t backtrackingNodes = 0;
    std::uint64_t searchLimitHits = 0;
    std::uint64_t lifetimeLimitHits = 0;
    std::uint64_t eventDomains = 0;
    /// Colors in the computed assignment, not maximum physical live events.
    std::uint64_t maximumDomainPressure = 0;
    std::uint64_t maximumEventIdPlusOne = 0;
};

/// Find a feasible assignment per directed domain, then minimize its color
/// count with bounded deterministic DSATUR. A minimization limit preserves the
/// feasible assignment. A feasibility/input limit reports AnalysisLimit and is
/// never classified as hardware resource exhaustion.
/// ResourceInfeasible refers to the conservative interference graph; it is not
/// by itself a proof that target resources require added serialization.
FailureOr<SyncEventAllocationResult> allocateSyncEventGenerations(
    const ProtocolSyncTarget& target, llvm::ArrayRef<SyncEventReservation> reservations,
    llvm::ArrayRef<SyncEventGeneration> generations, const SyncEventAllocationOptions& options = {});

/// Check a complete or wholly-unallocated assignment against the same explicit
/// lifetime relation. Partial assignments, reserved IDs, and same-ID
/// interference are rejected.
LogicalResult verifySyncEventGenerationAssignment(
    const ProtocolSyncTarget& target, llvm::ArrayRef<SyncEventReservation> reservations,
    llvm::ArrayRef<SyncEventGeneration> generations, bool allowUnallocated = true,
    const SyncEventAllocationOptions& options = {});

void recordSyncEventAllocationStatistics(
    const SyncEventAllocationResult& allocation, ProtocolSyncStatistics& statistics);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_EVENTALLOCATION_H
