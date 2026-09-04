// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ReadyReleaseProtocol.h - Atomic recurring ownership ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_READYRELEASEPROTOCOL_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_READYRELEASEPROTOCOL_H

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

enum class SyncReadyReleasePlanStatus : std::uint8_t { Empty, Ready, Unsupported };

enum class SyncReadyReleaseRejection : std::uint8_t {
    None,
    UnsupportedTarget,
    ExistingSynchronization,
    ScheduleFailure,
    SemanticAction,
    UnsupportedControlFlow,
    UnsupportedFunctionShape,
    IncompleteChannelSet,
    NonReadyReleaseChannel,
    UnverifiedChannel,
    UnsupportedCapacity,
    UnsupportedEventDirection,
    UnsupportedVisibility,
    InvalidFrontier,
    InvalidTokenTransfer,
    EventCapacity,
    InternalInvariant,
};

/// One logical ownership lane. Concrete event IDs are deliberately absent
/// until allocation succeeds for the complete protocol candidate.
struct SyncReadyReleaseLane {
    unsigned logicalLane = 0;
    std::optional<unsigned> readyEventId;
    std::optional<unsigned> releaseEventId;
};

/// Finite witnesses plus the inductive steady-state invariant used to certify
/// arbitrary loop trip counts. Every witnessed trip count begins and ends with
/// all lanes Free and no lane Ready.
struct SyncReadyReleaseTokenCertificate {
    unsigned witnessHorizon = 0;
    llvm::SmallVector<unsigned, 8> slotWitness;
    bool zeroTripSafe = false;
    bool oneTripSafe = false;
    bool oddEvenSafe = false;
    bool steadyStateStable = false;
    unsigned transitionApplications = 0;
};

struct SyncReadyReleasePlanRejection {
    SyncChannelId channel = kInvalidSyncId;
    SyncReadyReleaseRejection reason = SyncReadyReleaseRejection::None;
    std::string detail;
};

/// Checkpoint E selects one complete ReadyRelease protocol or none. It never
/// exposes independently selectable prime, body, or drain actions.
struct SyncReadyReleasePlan {
    SyncReadyReleasePlanStatus status = SyncReadyReleasePlanStatus::Empty;
    SyncChannelId channel = kInvalidSyncId;
    SyncGenerationId generation = kInvalidSyncId;
    unsigned capacity = 0;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE producerPipe = PIPE::PIPE_UNASSIGNED;
    PIPE consumerPipe = PIPE::PIPE_UNASSIGNED;
    SyncPhaseId producerPhase = kInvalidSyncId;
    SyncPhaseId consumerPhase = kInvalidSyncId;
    SyncRegionId loopRegion = kInvalidSyncId;
    Operation* loopOperation = nullptr;
    Operation* producerOperation = nullptr;
    Operation* consumerOperation = nullptr;
    std::optional<SyncSlotExpression> slot;
    llvm::SmallVector<SyncReadyReleaseLane, 2> lanes;
    SyncReadyReleaseTokenCertificate tokenCertificate;
    llvm::SmallVector<SyncReadyReleasePlanRejection, 2> rejections;
};

FailureOr<SyncReadyReleasePlan> buildReadyReleaseProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics = nullptr);
LogicalResult allocateReadyReleaseProtocolEvents(
    const StructuredSyncIR& schedule, SyncReadyReleasePlan& plan, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult allocateReadyReleaseProtocolEvents(
    const ProtocolSyncTarget& target, llvm::ArrayRef<SyncEventReservation> reservations, SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
/// Low-level emitter for an already-cloned function. The caller owns rollback
/// if materialization or subsequent verification fails.
LogicalResult materializeReadyReleaseProtocolPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
LogicalResult verifyReadyReleaseProtocolMaterialization(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, func::FuncOp clone,
    const IRMapping& mapping, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult materializeAndVerifyReadyReleaseProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
/// Mutates before verification and must only receive a function inside a
/// disposable module clone. The caller must discard that module on failure.
LogicalResult materializeAndVerifyReadyReleaseProtocolPlanInPlace(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncReadyReleasePlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
void printReadyReleaseProtocolPlan(func::FuncOp function, const SyncReadyReleasePlan& plan, llvm::raw_ostream& output);

llvm::StringRef stringifySyncReadyReleasePlanStatus(SyncReadyReleasePlanStatus status);
llvm::StringRef stringifySyncReadyReleaseRejection(SyncReadyReleaseRejection rejection);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_READYRELEASEPROTOCOL_H
