// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternAnalysis.h - Read-only lane pattern study -*- C++ -*-===//
//
// This experiment groups lane frontiers and raw residual access intervals into
// recognizable candidate shapes. Candidates are diagnostic observations only:
// target queries report feasibility but never authorize selection or emission.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNANALYSIS_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNANALYSIS_H

#include "PTO/Transforms/ProtocolSync/LaneFrontierAnalysis.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::pto::protocol_sync {

using SyncLanePatternCandidateId = std::uint32_t;
using SyncLaneRawAccessPairId = std::uint32_t;

enum class SyncLanePatternKind : std::uint8_t {
    SharedOneShotFrontier,
    SameLaneCompletionCut,
    ChoiceBalancedRoundTrip,
};

enum class SyncLaneReferencePattern : std::uint8_t {
    SharedEventFrontier,
    MultiDemandPipeBarrier,
    LiftedChoiceReadyRelease,
};

enum class SyncLaneTargetQuery : std::uint8_t {
    Supported,
    UnsupportedTarget,
    UnsupportedMechanism,
};

enum class SyncCheckpointEStatus : std::uint8_t {
    Admitted,
    Rejected,
    NotApplicable,
    Unavailable,
};

enum class SyncLaneRawHazardKind : std::uint8_t {
    ReadAfterWrite,
    WriteAfterRead,
    WriteAfterWrite,
};

struct SyncLaneRawAccessPair {
    SyncLaneRawAccessPairId id = kInvalidSyncId;
    SyncAccessId sourceAccess = kInvalidSyncId;
    SyncAccessId targetAccess = kInvalidSyncId;
    SyncGenerationId sourceGeneration = kInvalidSyncId;
    SyncGenerationId targetGeneration = kInvalidSyncId;
    SyncPhaseId sourcePhase = kInvalidSyncId;
    SyncPhaseId targetPhase = kInvalidSyncId;
    SyncStageId sourceStage = kInvalidSyncId;
    SyncStageId targetStage = kInvalidSyncId;
    SyncExecutionLaneId sourceLane = kInvalidSyncId;
    SyncExecutionLaneId targetLane = kInvalidSyncId;
    SyncProgramPointId sourceAfter = kInvalidSyncId;
    SyncProgramPointId targetBefore = kInvalidSyncId;
    SyncLaneRawHazardKind hazard = SyncLaneRawHazardKind::WriteAfterWrite;
    ProtocolSyncSameLaneCompletion completion = ProtocolSyncSameLaneCompletion::NotApplicable;
};

struct SyncLanePatternCost {
    unsigned logicalCandidates = 1;
    unsigned steadyStateActions = 0;
};

struct SyncLanePatternCandidate {
    SyncLanePatternCandidateId id = kInvalidSyncId;
    SyncLanePatternKind kind = SyncLanePatternKind::SharedOneShotFrontier;
    SyncLaneReferencePattern referencePattern = SyncLaneReferencePattern::SharedEventFrontier;
    SyncExecutionLaneId sourceLane = kInvalidSyncId;
    SyncExecutionLaneId targetLane = kInvalidSyncId;
    SyncProgramFrontier firstSource;
    SyncProgramFrontier firstTarget;
    SyncProgramFrontier secondSource;
    SyncProgramFrontier secondTarget;
    llvm::SmallVector<SyncGenerationId, 4> generations;
    llvm::SmallVector<SyncLaneFrontierExperimentId, 4> frontierMembers;
    llvm::SmallVector<SyncLaneRawAccessPairId, 4> rawPairMembers;
    SyncLaneTargetQuery targetQuery = SyncLaneTargetQuery::UnsupportedMechanism;
    SyncCheckpointEStatus checkpointE = SyncCheckpointEStatus::NotApplicable;
    SyncChannelRejection checkpointERejection = SyncChannelRejection::None;
    SyncLanePatternCost cost;
};

class LanePatternAnalysisResult {
public:
    llvm::ArrayRef<SyncLaneRawAccessPair> getRawAccessPairs() const { return rawAccessPairs; }
    llvm::ArrayRef<SyncLanePatternCandidate> getCandidates() const { return candidates; }

private:
    friend LanePatternAnalysisResult analyzeLanePatterns(
        const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        const LaneFrontierAnalysisResult& frontiers, ProtocolSyncStatistics* statistics);
    llvm::SmallVector<SyncLaneRawAccessPair, 32> rawAccessPairs;
    llvm::SmallVector<SyncLanePatternCandidate, 16> candidates;
};

LanePatternAnalysisResult analyzeLanePatterns(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& frontiers, ProtocolSyncStatistics* statistics = nullptr);
void printLanePatternAnalysis(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LanePatternAnalysisResult& analysis, llvm::raw_ostream& output);
llvm::StringRef stringifySyncLanePatternKind(SyncLanePatternKind kind);
llvm::StringRef stringifySyncLaneReferencePattern(SyncLaneReferencePattern pattern);
llvm::StringRef stringifySyncLaneReferenceMechanism(SyncLanePatternKind kind);
llvm::StringRef stringifySyncLaneTargetQuery(SyncLaneTargetQuery query);
llvm::StringRef stringifySyncCheckpointEStatus(SyncCheckpointEStatus status);
llvm::StringRef stringifySyncLaneRawHazardKind(SyncLaneRawHazardKind kind);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LANEPATTERNANALYSIS_H
