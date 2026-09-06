// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Selective, hazard-derived handoffs for an isolated unconditional loop.
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_SELECTIVELOOPREPAIR_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_SELECTIVELOOPREPAIR_H
#include "PTO/Transforms/ProtocolSync/LoopFrontierRepair.h"
#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "PTO/Transforms/ProtocolSync/RecurringEventLifetime.h"

namespace mlir::pto::protocol_sync {
inline constexpr unsigned kSelectiveLoopMaximumPhases = 128;
inline constexpr unsigned kSelectiveLoopMaximumAccesses = 512;
inline constexpr unsigned kSelectiveLoopMaximumOperations = 4096;
inline constexpr unsigned kSelectiveLoopMaximumChannels = 64;
inline constexpr unsigned kSelectiveLoopMaximumBarriers = 64;
inline constexpr std::uint64_t kSelectiveLoopMaximumCompletions = 65536;
struct SyncSelectiveLoopEdge {
    SyncLoopFrontierEdge placement;
    SyncSelectedCompletion completion;
    /// Mechanism identity is independent of concrete event assignment.
    bool isEvent() const { return placement.sourcePipe != placement.targetPipe; }
};
struct SyncSelectiveLoopPlan {
    Operation* loop = nullptr;
    llvm::SmallVector<SyncSelectiveLoopEdge, 16> edges;
    SyncSelectedWorld world;
    unsigned candidateCountBeforeDeletion = 0;
    unsigned deletionAttempts = 0;
    unsigned deletionRemoved = 0;
};

enum class SyncSelectiveLoopStatus : std::uint8_t { Ready, Unsupported, AnalysisLimit, UnprovedTokenContract };
struct SyncSelectiveLoopAttempt {
    SyncSelectiveLoopStatus status = SyncSelectiveLoopStatus::Unsupported;
    std::string detail;
    SyncRecurringEventProof tokenProof;
    std::optional<SyncSelectiveLoopPlan> plan;
};

/// Structural scope only, independent of selected synchronization supply.
std::optional<SyncRegionId> findIsolatedSyncLoop(const StructuredSyncIR& schedule, bool allowFixed);
/// Returns a certified logical plan with no concrete event IDs.
FailureOr<SyncSelectiveLoopAttempt> buildSelectiveLoopRepair(const StructuredSyncIR& schedule);
/// Append recurring resources to the common allocator input. Channels remain
/// conservatively interfering within each domain; this does not prove scarcity.
LogicalResult appendSelectiveLoopEventGenerations(
    const StructuredSyncIR& schedule, const SyncSelectiveLoopPlan& plan,
    llvm::SmallVectorImpl<SyncEventGeneration>& generations);
LogicalResult materializeSelectiveLoopRepair(func::FuncOp function, const SyncSelectiveLoopPlan& plan);
/// Re-extract actual events, boundaries, action order and completion prefixes.
FailureOr<SyncSelectedWorld> reconstructSelectiveLoop(const StructuredSyncIR& schedule);
/// All-pair occurrence check; does not consult sparse requirements or recipes.
LogicalResult verifySelectiveLoopMemory(const StructuredSyncIR& schedule, const SyncSelectedWorld& world);
} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_SELECTIVELOOPREPAIR_H
