// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LaneFrontierAnalysis.h - Read-only lane experiment ----*- C++ -*-===//
//
// The lane projection groups physical phases by (core, pipe) without turning
// lexical discovery order into a total execution order. Frontier experiments
// are structural observations only: they are not protocol candidates, do not
// prove target ordering, and cannot be selected or materialized.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_LANEFRONTIERANALYSIS_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_LANEFRONTIERANALYSIS_H

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir::pto::protocol_sync {

struct SyncReadyReleasePlan;

using SyncExecutionLaneId = std::uint32_t;
using SyncLaneFrontierExperimentId = std::uint32_t;

enum class SyncLaneDemandKind : std::uint8_t {
    Ready,
    Release,
    Residual,
};

enum class SyncLaneTopology : std::uint8_t {
    SameLane,
    CrossLane,
    CrossCore,
    Unknown,
};

enum class SyncFrontierPlacement : std::uint8_t {
    Exact,
    LinearCoalesced,
    ChoiceEntry,
    ChoiceExit,
    Unavailable,
};

enum class SyncLaneFrontierStatus : std::uint8_t {
    Found,
    TimelineRejected,
    UnresolvedSchedule,
    MissingEndpoint,
    UnresolvedLane,
    MultipleSourceLanes,
    MultipleTargetLanes,
    UnsupportedControl,
    CrossCore,
};

struct SyncLaneOccurrence {
    SyncPhaseId phase = kInvalidSyncId;
    SyncStageId stage = kInvalidSyncId;
    SyncRegionId region = kInvalidSyncId;
    SyncProgramPointId before = kInvalidSyncId;
    SyncProgramPointId after = kInvalidSyncId;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
};

struct SyncExecutionLane {
    SyncExecutionLaneId id = kInvalidSyncId;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    llvm::SmallVector<SyncLaneOccurrence, 8> occurrences;
};

struct SyncLaneFrontierExperiment {
    SyncLaneFrontierExperimentId id = kInvalidSyncId;
    SyncGenerationId generation = kInvalidSyncId;
    SyncLaneDemandKind demand = SyncLaneDemandKind::Residual;
    SyncExecutionLaneId sourceLane = kInvalidSyncId;
    SyncExecutionLaneId targetLane = kInvalidSyncId;
    llvm::SmallVector<SyncStageId, 2> sourceStages;
    llvm::SmallVector<SyncStageId, 4> targetStages;
    SyncProgramFrontier sourceFrontier;
    SyncProgramFrontier targetFrontier;
    SyncFrontierPlacement sourcePlacement = SyncFrontierPlacement::Unavailable;
    SyncFrontierPlacement targetPlacement = SyncFrontierPlacement::Unavailable;
    SyncLaneTopology topology = SyncLaneTopology::Unknown;
    SyncLaneFrontierStatus status = SyncLaneFrontierStatus::MissingEndpoint;
    SyncTimelineRejection timelineRejection = SyncTimelineRejection::None;
    SyncChannelRejection channelRejection = SyncChannelRejection::None;
    unsigned iterationDistance = 0;

    bool isFound() const { return status == SyncLaneFrontierStatus::Found; }
};

class LaneFrontierAnalysisResult {
public:
    llvm::ArrayRef<SyncExecutionLane> getLanes() const { return lanes; }
    llvm::ArrayRef<SyncLaneFrontierExperiment> getExperiments() const { return experiments; }
    const SyncExecutionLane* findLane(SyncExecutionLaneId id) const;
    const SyncExecutionLane* findLaneForPhase(SyncPhaseId phase) const;

private:
    friend LaneFrontierAnalysisResult analyzeLaneFrontiers(
        const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        ProtocolSyncStatistics* statistics);
    llvm::SmallVector<SyncExecutionLane, 8> lanes;
    llvm::SmallVector<SyncExecutionLaneId, 32> phaseToLane;
    llvm::SmallVector<SyncLaneFrontierExperiment, 32> experiments;
};

LaneFrontierAnalysisResult analyzeLaneFrontiers(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics = nullptr);
void printLaneFrontierAnalysis(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& analysis, llvm::raw_ostream& output);
void printReadyReleaseFrontierComparison(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& analysis, const SyncReadyReleasePlan& plan,
    llvm::raw_ostream& output);
llvm::StringRef stringifySyncLaneDemandKind(SyncLaneDemandKind kind);
llvm::StringRef stringifySyncLaneTopology(SyncLaneTopology topology);
llvm::StringRef stringifySyncFrontierPlacement(SyncFrontierPlacement placement);
llvm::StringRef stringifySyncLaneFrontierStatus(SyncLaneFrontierStatus status);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_LANEFRONTIERANALYSIS_H
