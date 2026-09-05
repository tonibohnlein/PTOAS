// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackProjection.cpp - Atomic storage segments ------------===//

#include "StorageTrackInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct AccessInterval {
    SyncAccessId access = kInvalidSyncId;
    unsigned space = 0;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

void rejectProjection(
    SyncAccessId access, SyncStorageProjectionRejection reason, SmallVectorImpl<SyncUnprojectedStorageAccess>& rejected,
    ProtocolSyncStatistics* statistics)
{
    rejected.push_back({access, reason});
    if (statistics) {
        ++statistics->storageTrackAccessesUnprojected;
        ++statistics->storageProjectionRejections[stringifySyncStorageProjectionRejection(reason).str()];
    }
}

bool hasValidIntervals(const SyncAccess& access)
{
    if (access.storage.intervals.empty()) {
        return false;
    }
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    return llvm::all_of(access.storage.intervals, [&](const SyncByteInterval& interval) {
        return interval.size != 0 && interval.begin <= maximum - interval.size;
    });
}

SmallVector<AccessInterval, 64> collectIntervals(
    const StructuredSyncIR& schedule, SmallVectorImpl<SyncUnprojectedStorageAccess>& rejected,
    ProtocolSyncStatistics* statistics)
{
    SmallVector<AccessInterval, 64> intervals;
    for (const SyncAccess& access : schedule.getAccesses()) {
        const SyncStorageFamily* family = schedule.findStorageFamily(access.family);
        if (family && family->role != SyncStorageRole::LocalBuffer) {
            continue;
        }
        if (statistics) {
            ++statistics->storageTrackAccessesAttempted;
        }
        if (!family) {
            rejectProjection(access.id, SyncStorageProjectionRejection::MissingFamily, rejected, statistics);
            continue;
        }
        if (!access.storage.physical) {
            rejectProjection(access.id, SyncStorageProjectionRejection::NonPhysical, rejected, statistics);
            continue;
        }
        if (access.storage.unknownRange) {
            rejectProjection(access.id, SyncStorageProjectionRejection::UnknownRange, rejected, statistics);
            continue;
        }
        if (!hasValidIntervals(access)) {
            rejectProjection(access.id, SyncStorageProjectionRejection::InvalidInterval, rejected, statistics);
            continue;
        }
        if (statistics) {
            ++statistics->storageTrackAccessesProjected;
        }
        for (const SyncByteInterval& interval : access.storage.intervals) {
            intervals.push_back(
                {access.id, static_cast<unsigned>(access.storage.space), interval.begin,
                 interval.begin + interval.size});
        }
    }
    return intervals;
}

SmallVector<SyncGenerationId, 64> mapAccessesToGenerations(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines)
{
    SmallVector<SyncGenerationId, 64> generations(schedule.getAccesses().size(), kInvalidSyncId);
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        for (const SyncRawAccessEndpoint& endpoint : timeline.rawAccesses) {
            if (endpoint.access < generations.size()) {
                generations[endpoint.access] = timeline.id;
            }
        }
    }
    return generations;
}

std::optional<unsigned> findPhysicalSlot(const SyncStorageFamily& family, std::uint64_t begin, std::uint64_t end)
{
    if (!family.physicalSlotsComplete) {
        return std::nullopt;
    }
    for (auto [slot, interval] : llvm::enumerate(family.intervals)) {
        const std::uint64_t intervalEnd = interval.begin + interval.size;
        if (interval.begin <= begin && end <= intervalEnd) {
            return static_cast<unsigned>(slot);
        }
    }
    return std::nullopt;
}

void appendFamilyBinding(
    const StructuredSyncIR& schedule, SyncStorageTrack& track, SyncStorageFamilyId familyId, std::uint64_t end)
{
    const bool recorded = llvm::any_of(
        track.families, [&](const SyncStorageTrackFamilyBinding& binding) { return binding.family == familyId; });
    if (recorded) {
        return;
    }
    const SyncStorageFamily* family = schedule.findStorageFamily(familyId);
    track.families.push_back({familyId, family ? findPhysicalSlot(*family, track.begin, end) : std::nullopt});
}

SyncStorageTrackOccurrence makeOccurrence(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const LaneFrontierAnalysisResult& laneFrontiers, ArrayRef<SyncGenerationId> generations, const SyncAccess& access,
    std::optional<unsigned> physicalSlot)
{
    SyncStorageTrackOccurrence occurrence;
    occurrence.access = access.id;
    occurrence.family = access.family;
    occurrence.physicalSlot = physicalSlot;
    occurrence.slot = access.slot;
    occurrence.generation = access.id < generations.size() ? generations[access.id] : kInvalidSyncId;
    occurrence.phase = access.phase;
    const SyncPhase* phase = schedule.findPhase(access.phase);
    const SyncStage* stage = phase ? stages.findStageForPhase(phase->id) : nullptr;
    const SyncExecutionLane* lane = phase ? laneFrontiers.findLaneForPhase(phase->id) : nullptr;
    occurrence.stage = stage ? stage->id : kInvalidSyncId;
    occurrence.executionLane = lane ? lane->id : kInvalidSyncId;
    occurrence.before = phase ? phase->before : kInvalidSyncId;
    occurrence.after = phase ? phase->after : kInvalidSyncId;
    occurrence.mode = access.mode;
    occurrence.slotBinding = mlir::pto::protocol_sync::detail::classifyStorageSlotBinding(access);
    if (phase) {
        occurrence.guard = phase->guard;
        occurrence.iterationDomain = phase->iterationDomain;
    }
    return occurrence;
}

void finishTrackFacts(const StructuredSyncIR& schedule, SyncStorageTrack& track)
{
    std::optional<SyncPhysicalCore> firstCore;
    for (const SyncStorageTrackOccurrence& occurrence : track.occurrences) {
        const SyncAccess* access = schedule.findAccess(occurrence.access);
        const SyncStorageFamily* family = access ? schedule.findStorageFamily(access->family) : nullptr;
        track.uncertainAlias = track.uncertainAlias || (access && access->storage.aliasesUnknownRange) ||
                               (family && family->aliasesUnknownRange);
        const SyncPhase* phase = schedule.findPhase(occurrence.phase);
        if (!phase || phase->core == SyncPhysicalCore::Unknown) {
            continue;
        }
        if (!firstCore) {
            firstCore = phase->core;
        } else if (*firstCore != phase->core) {
            track.multiplePhysicalCores = true;
        }
    }
}

void appendTrack(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const LaneFrontierAnalysisResult& laneFrontiers, ArrayRef<SyncGenerationId> generations,
    ArrayRef<AccessInterval> active, std::uint64_t begin, std::uint64_t end, SmallVectorImpl<SyncStorageTrack>& tracks,
    MutableArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks)
{
    SyncStorageTrack track;
    track.id = static_cast<SyncStorageTrackId>(tracks.size());
    track.space = static_cast<AddressSpace>(active.front().space);
    track.begin = begin;
    track.size = end - begin;
    SmallVector<SyncAccessId, 8> accessIds;
    for (const AccessInterval& interval : active) {
        if (!llvm::is_contained(accessIds, interval.access)) {
            accessIds.push_back(interval.access);
        }
    }
    llvm::sort(accessIds);
    for (SyncAccessId accessId : accessIds) {
        const SyncAccess* access = schedule.findAccess(accessId);
        if (!access) {
            continue;
        }
        appendFamilyBinding(schedule, track, access->family, end);
        const SyncStorageFamily* family = schedule.findStorageFamily(access->family);
        const std::optional<unsigned> physicalSlot =
            family ? findPhysicalSlot(*family, track.begin, end) : std::nullopt;
        track.occurrences.push_back(
            makeOccurrence(schedule, stages, laneFrontiers, generations, *access, physicalSlot));
        if (accessId < accessToTracks.size()) {
            accessToTracks[accessId].push_back(track.id);
        }
    }
    finishTrackFacts(schedule, track);
    tracks.push_back(std::move(track));
}

void buildTracksForAddressSpace(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const LaneFrontierAnalysisResult& laneFrontiers, ArrayRef<SyncGenerationId> generations,
    ArrayRef<AccessInterval> intervals, SmallVectorImpl<SyncStorageTrack>& tracks,
    MutableArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks)
{
    SmallVector<std::uint64_t, 64> boundaries;
    for (const AccessInterval& interval : intervals) {
        boundaries.push_back(interval.begin);
        boundaries.push_back(interval.end);
    }
    llvm::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    for (auto adjacent : llvm::zip(boundaries, llvm::ArrayRef(boundaries).drop_front())) {
        const std::uint64_t begin = std::get<0>(adjacent);
        const std::uint64_t end = std::get<1>(adjacent);
        SmallVector<AccessInterval, 8> active;
        for (const AccessInterval& interval : intervals) {
            if (interval.begin <= begin && end <= interval.end) {
                active.push_back(interval);
            }
        }
        if (!active.empty()) {
            appendTrack(schedule, stages, laneFrontiers, generations, active, begin, end, tracks, accessToTracks);
        }
    }
}

void updateProjectionStatistics(const StorageTrackAnalysisResult& result, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    statistics->storageTracks = result.getTracks().size();
    for (const SyncStorageTrack& track : result.getTracks()) {
        statistics->storageTrackOccurrences += track.occurrences.size();
        statistics->storageTracksMultipleFamilies += track.families.size() > 1 ? 1 : 0;
        statistics->storageTracksUncertainAlias += track.uncertainAlias ? 1 : 0;
        statistics->storageTracksMultiplePhysicalCores += track.multiplePhysicalCores ? 1 : 0;
    }
}

} // namespace

ArrayRef<SyncStorageTrackId> StorageTrackAnalysisResult::getTracksForAccess(SyncAccessId access) const
{
    return access < accessToTracks.size() ? ArrayRef<SyncStorageTrackId>(accessToTracks[access]) :
                                            ArrayRef<SyncStorageTrackId>();
}

StorageTrackAnalysisResult mlir::pto::protocol_sync::analyzeStorageTracks(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const LaneFrontierAnalysisResult& laneFrontiers, const LanePatternAnalysisResult& lanePatterns,
    ProtocolSyncStatistics* statistics)
{
    StorageTrackAnalysisResult result;
    result.accessToTracks.resize(schedule.getAccesses().size());
    SmallVector<AccessInterval, 64> intervals = collectIntervals(schedule, result.unprojectedAccesses, statistics);
    SmallVector<SyncGenerationId, 64> generations = mapAccessesToGenerations(schedule, timelines);
    std::map<unsigned, SmallVector<AccessInterval, 32>> byAddressSpace;
    for (const AccessInterval& interval : intervals) {
        byAddressSpace[interval.space].push_back(interval);
    }
    for (const auto& entry : byAddressSpace) {
        buildTracksForAddressSpace(
            schedule, stages, laneFrontiers, generations, entry.second, result.tracks, result.accessToTracks);
    }
    updateProjectionStatistics(result, statistics);
    mlir::pto::protocol_sync::detail::auditStorageProjection(
        schedule, result.tracks, result.accessToTracks, result.projectionAudit, statistics);
    FailureOr<SyncReadyReleasePlan> checkpointEPlan =
        buildReadyReleaseProtocolPlan(schedule, stages, timelines, channels);
    const SyncReadyReleasePlan* checkpointEPlanPointer = succeeded(checkpointEPlan) ? &*checkpointEPlan : nullptr;
    mlir::pto::protocol_sync::detail::reconstructStorageLifecycles(
        schedule, laneFrontiers, result.tracks, result.accessToTracks, checkpointEPlanPointer,
        result.lifecycleReconstructions, result.eDifferential, statistics);
    mlir::pto::protocol_sync::detail::appendStorageTransitionFrontiers(
        schedule, timelines, checkpointEPlanPointer, laneFrontiers, lanePatterns, result.tracks, result.accessToTracks,
        result.transitions, statistics);
    mlir::pto::protocol_sync::detail::auditStorageTransitions(
        lanePatterns, result.tracks, result.accessToTracks, result.transitions, result.transitionAudit, statistics);
    return result;
}
