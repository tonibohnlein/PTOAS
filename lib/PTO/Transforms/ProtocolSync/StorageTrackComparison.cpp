// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackComparison.cpp - Checkpoint E differential ---------===//

#include "PTO/Transforms/ProtocolSync/StorageTrackAnalysis.h"

#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

bool frontierIsPoint(const SyncProgramFrontier& frontier, SyncProgramPointId point)
{
    return frontier.points.size() == 1 && frontier.points.front().point == point &&
           frontier.points.front().guard.empty();
}

const SyncStorageTransitionFrontier* findUniqueTransition(
    const StorageTrackAnalysisResult& analysis, SyncGenerationId generation, SyncStorageTransitionKind kind)
{
    const SyncStorageTransitionFrontier* match = nullptr;
    for (const SyncStorageTransitionFrontier& transition : analysis.getTransitions()) {
        const bool isLifecycleTransition = transition.origin == SyncStorageTransitionOrigin::LaneFrontier &&
                                           transition.generation == generation && transition.kind == kind;
        if (!isLifecycleTransition) {
            continue;
        }
        if (match) {
            return nullptr;
        }
        match = &transition;
    }
    return match;
}

const SyncStorageFamily* findPlanFamily(const StructuredSyncIR& schedule, const SyncReadyReleasePlan& plan)
{
    const SyncPhase* producer = schedule.findPhase(plan.producerPhase);
    if (!producer) {
        return nullptr;
    }
    const SyncStorageFamily* match = nullptr;
    for (SyncAccessId accessId : producer->accesses) {
        const SyncAccess* access = schedule.findAccess(accessId);
        const SyncStorageFamily* family = access ? schedule.findStorageFamily(access->family) : nullptr;
        if (!access || !family || family->role != SyncStorageRole::LocalBuffer ||
            access->mode == SyncAccessMode::Read) {
            continue;
        }
        if (match && match->id != family->id) {
            return nullptr;
        }
        match = family;
    }
    return match;
}

bool sameTrackSet(const SyncStorageTransitionFrontier& first, const SyncStorageTransitionFrontier& second)
{
    return first.tracks.size() == second.tracks.size() && llvm::all_of(first.tracks, [&](SyncStorageTrackId track) {
               return llvm::is_contained(second.tracks, track);
           });
}

bool trackSlotsMatch(
    const StorageTrackAnalysisResult& analysis, const SyncStorageTransitionFrontier& transition,
    SyncStorageFamilyId family, unsigned capacity)
{
    const bool invalidCapacity = capacity == 0 || transition.tracks.size() != capacity;
    if (invalidCapacity) {
        return false;
    }
    SmallBitVector slots(capacity);
    for (SyncStorageTrackId trackId : transition.tracks) {
        if (trackId >= analysis.getTracks().size()) {
            return false;
        }
        const SyncStorageTrack& track = analysis.getTracks()[trackId];
        auto binding = llvm::find_if(
            track.families, [&](const SyncStorageTrackFamilyBinding& candidate) { return candidate.family == family; });
        if (binding == track.families.end()) {
            return false;
        }
        std::optional<unsigned> slot = binding->physicalSlot;
        if (!slot && capacity == 1) {
            slot = 0;
        }
        if (!slot || *slot >= capacity || slots.test(*slot)) {
            return false;
        }
        slots.set(*slot);
    }
    return slots.all();
}

} // namespace

void mlir::pto::protocol_sync::printReadyReleaseStorageTrackComparison(
    const StructuredSyncIR& schedule, const StorageTrackAnalysisResult& analysis, const SyncReadyReleasePlan& plan,
    raw_ostream& output)
{
    const SyncPhase* producer = schedule.findPhase(plan.producerPhase);
    const SyncPhase* consumer = schedule.findPhase(plan.consumerPhase);
    const SyncStorageFamily* family = findPlanFamily(schedule, plan);
    const SyncStorageTransitionFrontier* ready =
        findUniqueTransition(analysis, plan.generation, SyncStorageTransitionKind::Ready);
    const SyncStorageTransitionFrontier* release =
        findUniqueTransition(analysis, plan.generation, SyncStorageTransitionKind::Release);

    const bool directionsMatch =
        ready && release && ready->sourceLane == release->targetLane && ready->targetLane == release->sourceLane;
    const bool publicationMatches = producer && ready && frontierIsPoint(ready->sourceFrontier, producer->after);
    const bool acquisitionMatches = consumer && ready && frontierIsPoint(ready->targetFrontier, consumer->before);
    const bool finalUseMatches = consumer && release && frontierIsPoint(release->sourceFrontier, consumer->after);
    const bool nextOverwriteMatches = producer && release &&
                                      frontierIsPoint(release->targetFrontier, producer->before) &&
                                      release->iterationDistance == plan.capacity;
    const bool trackSetsMatch = ready && release && sameTrackSet(*ready, *release);
    const bool slotsMatch = family && ready && release &&
                            trackSlotsMatch(analysis, *ready, family->id, plan.capacity) &&
                            trackSlotsMatch(analysis, *release, family->id, plan.capacity);
    const bool matches = plan.status == SyncReadyReleasePlanStatus::Ready && directionsMatch && publicationMatches &&
                         acquisitionMatches && finalUseMatches && nextOverwriteMatches && trackSetsMatch && slotsMatch;

    output << "PROTOCOL-SYNC storage-ready-release-differential function=@" << schedule.getFunction().getSymName()
           << " status=" << (matches ? "match" : "mismatch") << " capacity=" << plan.capacity << '\n';
    output << "  directions=" << (directionsMatch ? "match" : "mismatch")
           << " publication=" << (publicationMatches ? "match" : "mismatch")
           << " acquisition=" << (acquisitionMatches ? "match" : "mismatch")
           << " final-use=" << (finalUseMatches ? "match" : "mismatch")
           << " next-overwrite=" << (nextOverwriteMatches ? "match" : "mismatch")
           << " track-set=" << (trackSetsMatch ? "match" : "mismatch")
           << " physical-slots=" << (slotsMatch ? "match" : "mismatch") << '\n';
}
