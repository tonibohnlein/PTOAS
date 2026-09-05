// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackAudit.cpp - Read-only track soundness audits --------===//

#include "StorageTrackInternal.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct ByteRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;

    bool operator==(const ByteRange& other) const { return begin == other.begin && end == other.end; }
};

bool hasExactLocalRange(const StructuredSyncIR& schedule, const SyncAccess& access)
{
    const SyncStorageFamily* family = schedule.findStorageFamily(access.family);
    if (!family || family->role != SyncStorageRole::LocalBuffer || !access.storage.physical ||
        access.storage.unknownRange || access.storage.aliasesUnknownRange || access.storage.intervals.empty()) {
        return false;
    }
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    return llvm::all_of(access.storage.intervals, [&](const SyncByteInterval& interval) {
        return interval.size != 0 && interval.begin <= maximum - interval.size;
    });
}

SmallVector<ByteRange, 4> canonicalizeRanges(SmallVector<ByteRange, 4> ranges)
{
    llvm::sort(ranges, [](const ByteRange& left, const ByteRange& right) {
        return left.begin < right.begin || (left.begin == right.begin && left.end < right.end);
    });
    SmallVector<ByteRange, 4> canonical;
    for (const ByteRange& range : ranges) {
        if (canonical.empty()) {
            canonical.push_back(range);
            continue;
        }
        const bool startsAfterCanonicalRange = canonical.back().end < range.begin;
        if (startsAfterCanonicalRange) {
            canonical.push_back(range);
            continue;
        }
        canonical.back().end = std::max(canonical.back().end, range.end);
    }
    return canonical;
}

SmallVector<ByteRange, 4> getAccessRanges(const SyncAccess& access)
{
    SmallVector<ByteRange, 4> ranges;
    for (const SyncByteInterval& interval : access.storage.intervals) {
        ranges.push_back({interval.begin, interval.begin + interval.size});
    }
    return canonicalizeRanges(std::move(ranges));
}

SmallVector<ByteRange, 4> getTrackRanges(
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SyncStorageTrackId> trackIds, AddressSpace expectedSpace,
    bool& validTrackIdsAndSpace)
{
    SmallVector<ByteRange, 4> ranges;
    validTrackIdsAndSpace = true;
    for (SyncStorageTrackId trackId : trackIds) {
        const bool trackIdIsValid = trackId < tracks.size();
        if (!trackIdIsValid) {
            validTrackIdsAndSpace = false;
            continue;
        }
        if (tracks[trackId].space != expectedSpace) {
            validTrackIdsAndSpace = false;
            continue;
        }
        const SyncStorageTrack& track = tracks[trackId];
        ranges.push_back({track.begin, track.begin + track.size});
    }
    return canonicalizeRanges(std::move(ranges));
}

bool exactRangesOverlap(const SyncAccess& first, const SyncAccess& second)
{
    if (first.storage.space != second.storage.space) {
        return false;
    }
    for (const SyncByteInterval& left : first.storage.intervals) {
        const std::uint64_t leftEnd = left.begin + left.size;
        for (const SyncByteInterval& right : second.storage.intervals) {
            const std::uint64_t rightEnd = right.begin + right.size;
            if (left.begin < rightEnd && right.begin < leftEnd) {
                return true;
            }
        }
    }
    return false;
}

void recordReadReadEffectPair(
    const StructuredSyncIR& schedule, const SyncAccess& first, const SyncAccess& second,
    SyncStorageProjectionAudit& audit)
{
    if (first.mode != SyncAccessMode::Read || second.mode != SyncAccessMode::Read) {
        return;
    }
    ++audit.readReadOverlapPairs;
    audit.accumulatorReadReadOverlapPairs += first.storage.space == AddressSpace::ACC ? 1 : 0;
    const SyncPhase* firstPhase = schedule.findPhase(first.phase);
    const SyncPhase* secondPhase = schedule.findPhase(second.phase);
    const bool crossLane =
        firstPhase && secondPhase && (firstPhase->core != secondPhase->core || firstPhase->pipe != secondPhase->pipe);
    audit.crossLaneReadReadOverlapPairs += crossLane ? 1 : 0;
}

bool shareTrack(ArrayRef<SyncStorageTrackId> first, ArrayRef<SyncStorageTrackId> second)
{
    return llvm::any_of(first, [&](SyncStorageTrackId track) { return llvm::is_contained(second, track); });
}

void auditComponents(
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    SyncStorageProjectionAudit& audit)
{
    SmallVector<mlir::pto::protocol_sync::detail::SyncStorageTrackComponent, 16> components =
        mlir::pto::protocol_sync::detail::buildStorageTrackComponents(tracks, accessToTracks);
    audit.overlapComponents = components.size();
    for (const auto& component : components) {
        audit.maximumAtomsPerComponent = std::max<std::uint64_t>(audit.maximumAtomsPerComponent, component.size());
    }
}

SmallVector<SyncStorageTrackId, 4> intersectTracks(
    ArrayRef<SyncStorageTrackId> first, ArrayRef<SyncStorageTrackId> second)
{
    SmallVector<SyncStorageTrackId, 4> common;
    for (SyncStorageTrackId track : first) {
        if (llvm::is_contained(second, track)) {
            common.push_back(track);
        }
    }
    return common;
}

bool frontierIsLinearPoint(const SyncProgramFrontier& frontier, SyncProgramPointId& point)
{
    const bool hasSinglePoint = frontier.points.size() == 1;
    if (!hasSinglePoint) {
        return false;
    }
    const bool hasEmptyGuard = frontier.points.front().guard.empty();
    if (!hasEmptyGuard) {
        return false;
    }
    point = frontier.points.front().point;
    return true;
}

bool linearFrontierCoversPair(const SyncStorageTransitionFrontier& transition, const SyncLaneRawAccessPair& pair)
{
    SyncProgramPointId source = kInvalidSyncId;
    SyncProgramPointId target = kInvalidSyncId;
    const bool hasLinearSource = frontierIsLinearPoint(transition.sourceFrontier, source);
    const bool hasLinearTarget = frontierIsLinearPoint(transition.targetFrontier, target);
    if (!hasLinearSource || !hasLinearTarget) {
        return false;
    }
    if (transition.kind == SyncStorageTransitionKind::Residual) {
        return source == pair.sourceAfter && target == pair.targetBefore;
    }
    if (transition.kind == SyncStorageTransitionKind::Completion) {
        return source == target && pair.sourceAfter <= source && source <= pair.targetBefore;
    }
    return pair.sourceAfter <= source && source <= target && target <= pair.targetBefore;
}

bool hasLinearFrontiers(const SyncStorageTransitionFrontier& transition)
{
    SyncProgramPointId ignored = kInvalidSyncId;
    return frontierIsLinearPoint(transition.sourceFrontier, ignored) &&
           frontierIsLinearPoint(transition.targetFrontier, ignored);
}

void updateProjectionStatistics(const SyncStorageProjectionAudit& audit, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    statistics->storageProjectionExactAccesses += audit.exactAccesses;
    statistics->storageProjectionAccessMaskMismatches += audit.accessMaskMismatches;
    statistics->storageProjectionPairRelations += audit.accessPairRelations;
    statistics->storageProjectionOverlapPairs += audit.overlappingAccessPairs;
    statistics->storageProjectionDisjointPairs += audit.disjointAccessPairs;
    statistics->storageProjectionOverlapPairsMissingTrack += audit.overlapPairsMissingSharedTrack;
    statistics->storageProjectionDisjointPairsSharingTrack += audit.disjointPairsSharingTrack;
    statistics->storageProjectionReadReadOverlapPairs += audit.readReadOverlapPairs;
    statistics->storageProjectionAccumulatorReadReadOverlapPairs += audit.accumulatorReadReadOverlapPairs;
    statistics->storageProjectionCrossLaneReadReadOverlapPairs += audit.crossLaneReadReadOverlapPairs;
    statistics->storageProjectionOverlapComponents += audit.overlapComponents;
    statistics->storageProjectionMaximumAtomsPerComponent =
        std::max(statistics->storageProjectionMaximumAtomsPerComponent, audit.maximumAtomsPerComponent);
    statistics->storageProjectionMaximumAtomsPerAccess =
        std::max(statistics->storageProjectionMaximumAtomsPerAccess, audit.maximumAtomsPerAccess);
}

void updateTransitionStatistics(const SyncStorageTransitionAudit& audit, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    statistics->storageTransitionPairMemberships += audit.pairMemberships;
    statistics->storageTransitionPairsCoveredOnce += audit.pairsCoveredOnce;
    statistics->storageTransitionPairsMultiplyCovered += audit.pairsMultiplyCovered;
    statistics->storageTransitionInvalidPairMemberships += audit.invalidPairMemberships;
    statistics->storageTransitionTrackMaskMismatches += audit.trackMaskMismatches;
    statistics->storageTransitionLinearFrontierMemberships += audit.linearFrontierMemberships;
    statistics->storageTransitionLinearFrontierMismatches += audit.linearFrontierMismatches;
    statistics->storageTransitionFrontierMembershipsNotLinear += audit.frontierMembershipsNotLinear;
}

} // namespace

SmallVector<mlir::pto::protocol_sync::detail::SyncStorageTrackComponent, 16>
mlir::pto::protocol_sync::detail::buildStorageTrackComponents(
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks)
{
    SmallVector<SyncStorageTrackComponent, 16> components;
    llvm::SmallBitVector visited(tracks.size());
    for (SyncStorageTrackId seed = 0; seed < tracks.size(); ++seed) {
        if (visited.test(seed)) {
            continue;
        }
        SyncStorageTrackComponent component;
        SmallVector<SyncStorageTrackId, 8> pending{seed};
        visited.set(seed);
        while (!pending.empty()) {
            const SyncStorageTrackId current = pending.pop_back_val();
            component.push_back(current);
            for (const SyncStorageTrackOccurrence& occurrence : tracks[current].occurrences) {
                if (occurrence.access >= accessToTracks.size()) {
                    continue;
                }
                for (SyncStorageTrackId neighbor : accessToTracks[occurrence.access]) {
                    const bool neighborIsValid = neighbor < tracks.size();
                    const bool neighborWasVisited = neighborIsValid && visited.test(neighbor);
                    if (neighborIsValid && !neighborWasVisited) {
                        visited.set(neighbor);
                        pending.push_back(neighbor);
                    }
                }
            }
        }
        llvm::sort(component);
        components.push_back(std::move(component));
    }
    return components;
}

void mlir::pto::protocol_sync::detail::auditStorageProjection(
    const StructuredSyncIR& schedule, ArrayRef<SyncStorageTrack> tracks,
    ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks, SyncStorageProjectionAudit& audit,
    ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncAccessId, 64> exactAccesses;
    for (const SyncAccess& access : schedule.getAccesses()) {
        if (!hasExactLocalRange(schedule, access)) {
            continue;
        }
        exactAccesses.push_back(access.id);
        ++audit.exactAccesses;
        ArrayRef<SyncStorageTrackId> trackIds =
            access.id < accessToTracks.size() ? accessToTracks[access.id] : ArrayRef<SyncStorageTrackId>();
        audit.maximumAtomsPerAccess = std::max<std::uint64_t>(audit.maximumAtomsPerAccess, trackIds.size());
        bool validTracks = false;
        const SmallVector<ByteRange, 4> accessRanges = getAccessRanges(access);
        const SmallVector<ByteRange, 4> trackRanges =
            getTrackRanges(tracks, trackIds, access.storage.space, validTracks);
        if (accessRanges != trackRanges || !validTracks) {
            ++audit.accessMaskMismatches;
        }
    }
    for (auto [targetIndex, targetId] : llvm::enumerate(exactAccesses)) {
        const SyncAccess* target = schedule.findAccess(targetId);
        if (!target) {
            continue;
        }
        for (SyncAccessId sourceId : ArrayRef(exactAccesses).take_front(targetIndex)) {
            const SyncAccess* source = schedule.findAccess(sourceId);
            if (!source) {
                continue;
            }
            ++audit.accessPairRelations;
            const bool overlaps = exactRangesOverlap(*source, *target);
            const bool shared = shareTrack(accessToTracks[sourceId], accessToTracks[targetId]);
            if (overlaps) {
                ++audit.overlappingAccessPairs;
                audit.overlapPairsMissingSharedTrack += shared ? 0 : 1;
                recordReadReadEffectPair(schedule, *source, *target, audit);
            } else {
                ++audit.disjointAccessPairs;
                audit.disjointPairsSharingTrack += shared ? 1 : 0;
            }
        }
    }
    auditComponents(tracks, accessToTracks, audit);
    updateProjectionStatistics(audit, statistics);
}

void mlir::pto::protocol_sync::detail::auditStorageTransitions(
    const LanePatternAnalysisResult& lanePatterns, ArrayRef<SyncStorageTrack> tracks,
    ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks, ArrayRef<SyncStorageTransitionFrontier> transitions,
    SyncStorageTransitionAudit& audit, ProtocolSyncStatistics* statistics)
{
    (void)tracks;
    audit.rawPairs = lanePatterns.getRawAccessPairs().size();
    SmallVector<unsigned, 32> membershipCounts(audit.rawPairs);
    for (const SyncStorageTransitionFrontier& transition : transitions) {
        for (SyncLaneRawAccessPairId pairId : transition.rawPairMembers) {
            ++audit.pairMemberships;
            if (pairId >= lanePatterns.getRawAccessPairs().size()) {
                ++audit.invalidPairMemberships;
                continue;
            }
            ++membershipCounts[pairId];
            const SyncLaneRawAccessPair& pair = lanePatterns.getRawAccessPairs()[pairId];
            const bool validAccesses =
                pair.sourceAccess < accessToTracks.size() && pair.targetAccess < accessToTracks.size();
            SmallVector<SyncStorageTrackId, 4> commonTracks;
            if (validAccesses) {
                commonTracks = intersectTracks(accessToTracks[pair.sourceAccess], accessToTracks[pair.targetAccess]);
            }
            const bool completeMask = !commonTracks.empty() && llvm::all_of(commonTracks, [&](SyncStorageTrackId id) {
                return llvm::is_contained(transition.tracks, id);
            });
            audit.trackMaskMismatches += completeMask ? 0 : 1;
            if (hasLinearFrontiers(transition)) {
                ++audit.linearFrontierMemberships;
                audit.linearFrontierMismatches += linearFrontierCoversPair(transition, pair) ? 0 : 1;
            } else {
                ++audit.frontierMembershipsNotLinear;
            }
        }
    }
    for (unsigned count : membershipCounts) {
        audit.pairsUncovered += count == 0 ? 1 : 0;
        audit.pairsCoveredOnce += count == 1 ? 1 : 0;
        audit.pairsMultiplyCovered += count > 1 ? 1 : 0;
    }
    updateTransitionStatistics(audit, statistics);
}
