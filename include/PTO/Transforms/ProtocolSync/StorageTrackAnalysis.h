// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackAnalysis.h - Read-only storage tracks -----*- C++ -*-===//
//
// Storage tracks partition exact local byte ranges into non-overlapping atoms.
// Transition frontiers combine those atoms with execution lanes, structured
// control, and iteration facts. Every record is diagnostic and non-selectable.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKANALYSIS_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKANALYSIS_H

#include "PTO/Transforms/ProtocolSync/LanePatternAnalysis.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace mlir::pto::protocol_sync {

using SyncStorageTrackId = std::uint32_t;
using SyncStorageTransitionId = std::uint32_t;

enum class SyncStorageSlotBinding : std::uint8_t {
    Unslotted,
    Constant,
    AffineConditional,
    UnknownConditional,
};

enum class SyncStorageProjectionRejection : std::uint8_t {
    MissingFamily,
    NonPhysical,
    UnknownRange,
    InvalidInterval,
};

enum class SyncStorageTransitionKind : std::uint8_t {
    Ready,
    Release,
    Completion,
    Residual,
};

enum class SyncStorageTransitionOrigin : std::uint8_t {
    LaneFrontier,
    CompletionCut,
    RawAccessPair,
};

enum class SyncStorageTargetResult : std::uint8_t {
    Intrinsic,
    PipeBarrier,
    Event,
    UnsupportedTarget,
    UnsupportedMechanism,
    NotApplicable,
};

enum class SyncStorageLifecycleRejection : std::uint8_t {
    None,
    NotRecurring,
    UnsupportedControl,
    MultipleLoops,
    UnsupportedAccessShape,
    ProducerAfterConsumer,
    IncompleteTrackSet,
    UnknownCapacity,
    UnsupportedCapacity,
    UnknownSlotRelation,
    InvalidReuseDistance,
    UnresolvedLane,
};

enum class SyncStorageEDifferentialStatus : std::uint8_t {
    Match,
    Mismatch,
    IndependentOnly,
    EOnly,
    Neither,
    Unavailable,
};

struct SyncStorageTrackFamilyBinding {
    SyncStorageFamilyId family = kInvalidSyncId;
    std::optional<unsigned> physicalSlot;
};

struct SyncStorageTrackOccurrence {
    SyncAccessId access = kInvalidSyncId;
    SyncStorageFamilyId family = kInvalidSyncId;
    std::optional<unsigned> physicalSlot;
    std::optional<SyncSlotExpression> slot;
    SyncGenerationId generation = kInvalidSyncId;
    SyncPhaseId phase = kInvalidSyncId;
    SyncStageId stage = kInvalidSyncId;
    SyncExecutionLaneId executionLane = kInvalidSyncId;
    SyncProgramPointId before = kInvalidSyncId;
    SyncProgramPointId after = kInvalidSyncId;
    SyncAccessMode mode = SyncAccessMode::ReadWrite;
    SyncStorageSlotBinding slotBinding = SyncStorageSlotBinding::Unslotted;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
};

struct SyncStorageTrack {
    SyncStorageTrackId id = kInvalidSyncId;
    AddressSpace space = AddressSpace::Zero;
    std::uint64_t begin = 0;
    std::uint64_t size = 0;
    llvm::SmallVector<SyncStorageTrackFamilyBinding, 2> families;
    llvm::SmallVector<SyncStorageTrackOccurrence, 8> occurrences;
    bool uncertainAlias = false;
    bool multiplePhysicalCores = false;
};

struct SyncUnprojectedStorageAccess {
    SyncAccessId access = kInvalidSyncId;
    SyncStorageProjectionRejection reason = SyncStorageProjectionRejection::UnknownRange;
};

struct SyncStorageProjectionAudit {
    std::uint64_t exactAccesses = 0;
    std::uint64_t accessMaskMismatches = 0;
    std::uint64_t accessPairRelations = 0;
    std::uint64_t overlappingAccessPairs = 0;
    std::uint64_t disjointAccessPairs = 0;
    std::uint64_t overlapPairsMissingSharedTrack = 0;
    std::uint64_t disjointPairsSharingTrack = 0;
    std::uint64_t readReadOverlapPairs = 0;
    std::uint64_t accumulatorReadReadOverlapPairs = 0;
    std::uint64_t crossLaneReadReadOverlapPairs = 0;
    std::uint64_t overlapComponents = 0;
    std::uint64_t maximumAtomsPerComponent = 0;
    std::uint64_t maximumAtomsPerAccess = 0;

    bool isExact() const
    {
        return accessMaskMismatches == 0 && overlapPairsMissingSharedTrack == 0 && disjointPairsSharingTrack == 0;
    }
};

struct SyncStorageTransitionAudit {
    std::uint64_t rawPairs = 0;
    std::uint64_t pairMemberships = 0;
    std::uint64_t pairsCoveredOnce = 0;
    std::uint64_t pairsUncovered = 0;
    std::uint64_t pairsMultiplyCovered = 0;
    std::uint64_t invalidPairMemberships = 0;
    std::uint64_t trackMaskMismatches = 0;
    std::uint64_t linearFrontierMemberships = 0;
    std::uint64_t linearFrontierMismatches = 0;
    std::uint64_t frontierMembershipsNotLinear = 0;

    bool isExact() const
    {
        return pairsUncovered == 0 && pairsMultiplyCovered == 0 && invalidPairMemberships == 0 &&
               trackMaskMismatches == 0 && linearFrontierMismatches == 0;
    }
};

struct SyncStorageLifecycleReconstruction {
    unsigned component = 0;
    SyncStorageLifecycleRejection rejection = SyncStorageLifecycleRejection::NotRecurring;
    SyncStorageFamilyId family = kInvalidSyncId;
    SyncAccessId producerAccess = kInvalidSyncId;
    SyncAccessId consumerAccess = kInvalidSyncId;
    SyncPhaseId producerPhase = kInvalidSyncId;
    SyncPhaseId consumerPhase = kInvalidSyncId;
    SyncRegionId loop = kInvalidSyncId;
    SyncExecutionLaneId producerLane = kInvalidSyncId;
    SyncExecutionLaneId consumerLane = kInvalidSyncId;
    unsigned capacity = 0;
    unsigned reuseDistance = 0;
    SyncProgramPointId publication = kInvalidSyncId;
    SyncProgramPointId acquisition = kInvalidSyncId;
    SyncProgramPointId finalUse = kInvalidSyncId;
    SyncProgramPointId nextOverwrite = kInvalidSyncId;
    llvm::SmallVector<SyncStorageTrackId, 4> tracks;

    bool isReady() const { return rejection == SyncStorageLifecycleRejection::None; }
};

struct SyncStorageEDifferential {
    SyncStorageEDifferentialStatus status = SyncStorageEDifferentialStatus::Neither;
    SyncReadyReleaseRejection eRejection = SyncReadyReleaseRejection::None;
    bool capacityMatches = false;
    bool lanesMatch = false;
    bool loopMatches = false;
    bool phasesMatch = false;
    bool lifecycleMatches = false;
    bool physicalSlotsMatch = false;
};

struct SyncStorageTransitionFrontier {
    SyncStorageTransitionId id = kInvalidSyncId;
    SyncStorageTransitionKind kind = SyncStorageTransitionKind::Residual;
    SyncStorageTransitionOrigin origin = SyncStorageTransitionOrigin::RawAccessPair;
    SyncGenerationId generation = kInvalidSyncId;
    SyncExecutionLaneId sourceLane = kInvalidSyncId;
    SyncExecutionLaneId targetLane = kInvalidSyncId;
    SyncProgramFrontier sourceFrontier;
    SyncProgramFrontier targetFrontier;
    unsigned iterationDistance = 0;
    bool recurring = false;
    llvm::SmallVector<SyncStorageTrackId, 4> tracks;
    llvm::SmallVector<SyncLaneRawAccessPairId, 8> rawPairMembers;
    SyncLanePatternCost cost;
    SyncStorageTargetResult targetResult = SyncStorageTargetResult::NotApplicable;
    SyncCheckpointEStatus checkpointE = SyncCheckpointEStatus::NotApplicable;
    SyncReadyReleaseRejection checkpointERejection = SyncReadyReleaseRejection::None;
};

class StorageTrackAnalysisResult {
public:
    llvm::ArrayRef<SyncStorageTrack> getTracks() const { return tracks; }
    llvm::ArrayRef<SyncStorageTransitionFrontier> getTransitions() const { return transitions; }
    llvm::ArrayRef<SyncUnprojectedStorageAccess> getUnprojectedAccesses() const { return unprojectedAccesses; }
    llvm::ArrayRef<SyncStorageTrackId> getTracksForAccess(SyncAccessId access) const;
    const SyncStorageProjectionAudit& getProjectionAudit() const { return projectionAudit; }
    const SyncStorageTransitionAudit& getTransitionAudit() const { return transitionAudit; }
    llvm::ArrayRef<SyncStorageLifecycleReconstruction> getLifecycleReconstructions() const
    {
        return lifecycleReconstructions;
    }
    const SyncStorageEDifferential& getEDifferential() const { return eDifferential; }

private:
    friend StorageTrackAnalysisResult analyzeStorageTracks(
        const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
        const LaneFrontierAnalysisResult& laneFrontiers, const LanePatternAnalysisResult& lanePatterns,
        ProtocolSyncStatistics* statistics);
    llvm::SmallVector<SyncStorageTrack, 32> tracks;
    llvm::SmallVector<SyncStorageTransitionFrontier, 32> transitions;
    llvm::SmallVector<SyncUnprojectedStorageAccess, 8> unprojectedAccesses;
    llvm::SmallVector<llvm::SmallVector<SyncStorageTrackId, 2>, 64> accessToTracks;
    SyncStorageProjectionAudit projectionAudit;
    SyncStorageTransitionAudit transitionAudit;
    llvm::SmallVector<SyncStorageLifecycleReconstruction, 8> lifecycleReconstructions;
    SyncStorageEDifferential eDifferential;
};

StorageTrackAnalysisResult analyzeStorageTracks(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& laneFrontiers, const LanePatternAnalysisResult& lanePatterns,
    ProtocolSyncStatistics* statistics = nullptr);
void printStorageTrackAnalysis(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    const StorageTrackAnalysisResult& analysis, llvm::raw_ostream& output);
void printReadyReleaseStorageTrackComparison(
    const StructuredSyncIR& schedule, const StorageTrackAnalysisResult& analysis, const SyncReadyReleasePlan& plan,
    llvm::raw_ostream& output);

llvm::StringRef stringifySyncStorageSlotBinding(SyncStorageSlotBinding binding);
llvm::StringRef stringifySyncStorageProjectionRejection(SyncStorageProjectionRejection rejection);
llvm::StringRef stringifySyncStorageTransitionKind(SyncStorageTransitionKind kind);
llvm::StringRef stringifySyncStorageTransitionOrigin(SyncStorageTransitionOrigin origin);
llvm::StringRef stringifySyncStorageTargetResult(SyncStorageTargetResult result);
llvm::StringRef stringifySyncStorageLifecycleRejection(SyncStorageLifecycleRejection rejection);
llvm::StringRef stringifySyncStorageEDifferentialStatus(SyncStorageEDifferentialStatus status);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETRACKANALYSIS_H
