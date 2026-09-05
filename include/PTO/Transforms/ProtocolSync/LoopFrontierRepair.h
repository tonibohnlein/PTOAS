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
// Low-level recurring repair for an isolated ordinary loop in disposable IR.
// This is a local completion certificate, NOT a complete function certificate:
// GM visibility, exit drains, fixed supply and F integration remain separate.
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
    std::uint64_t requirementCount = 0;
    std::string detail;
};

/// Builds a conservative complete local phase-order cycle, not a storage
/// capacity-one protocol. Does not change IR, coverage bits, or F selection.
FailureOr<SyncLoopFrontierPlan> buildLoopFrontierRepairPlan(
    const StructuredSyncIR& schedule, llvm::ArrayRef<SyncEventReservation> reservations = {});

/// Caller owns disposable whole-module staging IR and must discard it on any
/// failure. Successful emission alone does not authorize committing the clone.
LogicalResult materializeLoopFrontierRepair(
    func::FuncOp clone, const IRMapping& mapping, const SyncLoopFrontierPlan& plan);

/// Independently reconstruct the actual phase/event/barrier cycle. Checks
/// arbitrary-trip consumption-before-rearm by the acknowledged cycle invariant,
/// including the zero-trip prime/drain path. Consults no plan or diagnostic tags.
/// Rejects unsupported scope. Certifies local completion only, not GM visibility,
/// mandatory function exit, or the completeness of shared operation semantics.
LogicalResult verifyConcreteLoopFrontierRepair(const StructuredSyncIR& schedule);

} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LOOPFRONTIERREPAIR_H
