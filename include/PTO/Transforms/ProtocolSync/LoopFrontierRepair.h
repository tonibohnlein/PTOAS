// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LoopFrontierRepair.h - Acknowledged phase-frontier cycle -*- C++ -*-===//
// Acknowledged recurring repair with optional prefix/suffix boundary handoffs.
// Native F selection uses the boundary form plus the full residual interpreter.
// The local cycle alone is NOT a visibility or complete-function certificate.
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_LOOPFRONTIERREPAIR_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_LOOPFRONTIERREPAIR_H

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::pto::protocol_sync {

enum class SyncLoopFrontierStatus { Unsupported, ResourceInfeasible, Ready };

struct SyncLoopFrontierEdge {
    Operation* source = nullptr;
    Operation* target = nullptr;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    /// Empty means a target-certified same-pipe barrier.
    std::optional<unsigned> eventId;
};

struct SyncLoopFrontierPlan {
    SyncLoopFrontierStatus status = SyncLoopFrontierStatus::Unsupported;
    Operation* loop = nullptr;
    /// Lexical adjacent phase frontiers; the last edge crosses the backedge.
    llvm::SmallVector<SyncLoopFrontierEdge, 8> edges;
    /// Outer straight-line chain. The loop denotes its last-pipe entry gateway
    /// as a target and its first-pipe drained gateway as a source.
    llvm::SmallVector<SyncLoopFrontierEdge, 8> boundaryEdges;
    llvm::SmallVector<SyncResidualObligation, 16> requirements;
    bool includeBoundaries = false;
    std::uint64_t requirementCount = 0;
    std::string detail;
};

/// Builds a conservative complete local phase-order cycle, not a storage
/// capacity-one protocol. Does not change IR or coverage bits. The default
/// remains the isolated diagnostic API; native F requests includeBoundaries.
FailureOr<SyncLoopFrontierPlan> buildLoopFrontierRepairPlan(
    const StructuredSyncIR& schedule, llvm::ArrayRef<SyncEventReservation> reservations = {},
    bool includeBoundaries = false);

/// Caller owns disposable whole-module staging IR and must discard it on any
/// failure. Successful emission alone does not authorize committing the clone.
LogicalResult materializeLoopFrontierRepair(
    func::FuncOp clone, const IRMapping& mapping, const SyncLoopFrontierPlan& plan);

/// Independently reconstruct the actual phase/event/barrier cycle. Checks
/// arbitrary-trip consumption-before-rearm by the acknowledged cycle invariant,
/// including the zero-trip prime/drain path. Consults no plan or diagnostic tags.
/// Rejects unsupported scope. Certifies local completion only, not GM visibility,
/// or the completeness of shared operation semantics. The boundary form also
/// requires the final function-exit drain; F must still check nonlocal effects.
LogicalResult verifyConcreteLoopFrontierRepair(const StructuredSyncIR& schedule, bool includeBoundaries = false);

/// Logical completion-only certificate; concrete callers must first reconstruct
/// every action with verifyConcreteLoopFrontierRepair(..., true).
FailureOr<SyncSelectedWorld> buildLoopFrontierWorld(const StructuredSyncIR& schedule);
bool loopFrontierOrders(
    const StructuredSyncIR& schedule, SyncRegionId carrier, SyncPhaseId source, SyncPhaseId target,
    const SyncIterationRelation& relation);

} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LOOPFRONTIERREPAIR_H
