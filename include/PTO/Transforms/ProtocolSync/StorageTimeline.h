// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTimeline.h - Storage-generation timeline model --*- C++ -*-===//
//
// Timelines derive symbolic contents from immutable accesses. They carry only
// diagnostic evidence and contain no selected synchronization or event IDs.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINE_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINE_H

#include "PTO/Transforms/ProtocolSync/PipelineStages.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::pto::protocol_sync {

using SyncGenerationId = std::uint32_t;

enum class SyncGenerationKind : std::uint8_t {
    OneShot,
    LoopIteration,
};

enum class SyncTimelineRejection : std::uint8_t {
    None,
    NonPhysicalLocalStorage,
    UnknownRange,
    ConflictingPhysicalRange,
    UnknownCapacity,
    UnsupportedCapacity,
    IncompletePhysicalSlots,
    UnknownSlotExpression,
    PartialOverlap,
    MissingProducer,
    MissingConsumer,
    OrderedAccess,
    InPlaceAccess,
    InvalidStage,
    ProducerAfterConsumer,
    IncompatibleIterationDomain,
    MultipleGenerationsPerIteration,
    UnknownReuseDistance,
};

struct SyncGuardedProgramPoint {
    SyncProgramPointId point = kInvalidSyncId;
    llvm::SmallVector<SyncControlAtom, 2> guard;
};

struct SyncProgramFrontier {
    llvm::SmallVector<SyncGuardedProgramPoint, 2> points;
};

struct SyncNextOverwrite {
    SyncProgramFrontier frontier;
    unsigned iterationDistance = 0;
};

struct SyncGenerationTimeline {
    SyncGenerationId id = kInvalidSyncId;
    SyncStorageFamilyId family = kInvalidSyncId;
    llvm::SmallVector<SyncAccessId, 4> accesses;
    llvm::SmallVector<SyncByteInterval, 2> slice;
    std::optional<SyncSlotExpression> slot;
    SyncGenerationKind generationKind = SyncGenerationKind::OneShot;
    SyncRegionId carryingRegion = kInvalidSyncId;
    llvm::SmallVector<SyncStageId, 2> producers;
    llvm::SmallVector<SyncStageId, 4> consumers;
    SyncProgramFrontier publication;
    llvm::SmallVector<SyncProgramFrontier, 2> acquisitions;
    llvm::SmallVector<SyncProgramFrontier, 2> finalUses;
    std::optional<SyncNextOverwrite> nextOverwrite;
    std::optional<unsigned> reuseDistance;
    SyncTimelineRejection rejection = SyncTimelineRejection::None;

    bool isAdmitted() const { return rejection == SyncTimelineRejection::None; }
};

class StorageTimelineAnalysisResult {
public:
    llvm::ArrayRef<SyncGenerationTimeline> getTimelines() const { return timelines; }

private:
    friend StorageTimelineAnalysisResult analyzeStorageTimelines(
        const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
        ProtocolSyncStatistics* statistics);
    llvm::SmallVector<SyncGenerationTimeline, 16> timelines;
};

StorageTimelineAnalysisResult analyzeStorageTimelines(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ProtocolSyncStatistics* statistics = nullptr);
llvm::StringRef stringifySyncGenerationKind(SyncGenerationKind kind);
llvm::StringRef stringifySyncTimelineRejection(SyncTimelineRejection reason);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_STORAGETIMELINE_H
