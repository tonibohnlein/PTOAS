// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackInternal.h - Storage-track helpers ------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKINTERNAL_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKINTERNAL_H

#include "PTO/Transforms/ProtocolSync/StorageTrackAnalysis.h"

namespace mlir::pto::protocol_sync::detail {

using SyncStorageTrackComponent = llvm::SmallVector<SyncStorageTrackId, 8>;

SyncStorageSlotBinding classifyStorageSlotBinding(const SyncAccess& access);
SyncStorageTargetResult mapSameLaneCompletionToStorageTarget(ProtocolSyncSameLaneCompletion completion);
SyncStorageTargetResult queryStorageLaneTransfer(
    const ProtocolSyncTarget& target, const LaneFrontierAnalysisResult& laneFrontiers, SyncExecutionLaneId sourceId,
    SyncExecutionLaneId targetId);
void refineSameLaneStorageTarget(
    const LanePatternAnalysisResult& lanePatterns, SyncStorageTransitionFrontier& transition);
llvm::SmallVector<SyncStorageTrackComponent, 16> buildStorageTrackComponents(
    llvm::ArrayRef<SyncStorageTrack> tracks, llvm::ArrayRef<llvm::SmallVector<SyncStorageTrackId, 2>> accessToTracks);

void auditStorageProjection(
    const StructuredSyncIR& schedule, llvm::ArrayRef<SyncStorageTrack> tracks,
    llvm::ArrayRef<llvm::SmallVector<SyncStorageTrackId, 2>> accessToTracks, SyncStorageProjectionAudit& audit,
    ProtocolSyncStatistics* statistics);
void auditStorageTransitions(
    const LanePatternAnalysisResult& lanePatterns, llvm::ArrayRef<SyncStorageTrack> tracks,
    llvm::ArrayRef<llvm::SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    llvm::ArrayRef<SyncStorageTransitionFrontier> transitions, SyncStorageTransitionAudit& audit,
    ProtocolSyncStatistics* statistics);
void reconstructStorageLifecycles(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    llvm::ArrayRef<SyncStorageTrack> tracks, llvm::ArrayRef<llvm::SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    const SyncReadyReleasePlan* checkpointEPlan,
    llvm::SmallVectorImpl<SyncStorageLifecycleReconstruction>& reconstructions, SyncStorageEDifferential& differential,
    ProtocolSyncStatistics* statistics);

void appendStorageTransitionFrontiers(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const SyncReadyReleasePlan* checkpointEPlan, const LaneFrontierAnalysisResult& laneFrontiers,
    const LanePatternAnalysisResult& lanePatterns, llvm::ArrayRef<SyncStorageTrack> tracks,
    llvm::ArrayRef<llvm::SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    llvm::SmallVectorImpl<SyncStorageTransitionFrontier>& transitions, ProtocolSyncStatistics* statistics);

} // namespace mlir::pto::protocol_sync::detail

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKINTERNAL_H
