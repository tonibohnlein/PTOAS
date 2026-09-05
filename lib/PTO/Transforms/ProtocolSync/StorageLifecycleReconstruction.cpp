// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageLifecycleReconstruction.cpp - Independent lifecycle facts ===//

#include "StorageTrackInternal.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

SmallVector<SyncAccessId, 8> collectComponentAccesses(
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SyncStorageTrackId> component)
{
    SmallVector<SyncAccessId, 8> accesses;
    for (SyncStorageTrackId trackId : component) {
        if (trackId >= tracks.size()) {
            continue;
        }
        for (const SyncStorageTrackOccurrence& occurrence : tracks[trackId].occurrences) {
            if (!llvm::is_contained(accesses, occurrence.access)) {
                accesses.push_back(occurrence.access);
            }
        }
    }
    llvm::sort(accesses);
    return accesses;
}

bool findSingleLoop(
    const StructuredSyncIR& schedule, ArrayRef<SyncAccessId> accesses, SyncRegionId& loop,
    SyncStorageLifecycleRejection& rejection)
{
    bool sawRecurring = false;
    for (SyncAccessId accessId : accesses) {
        const SyncAccess* access = schedule.findAccess(accessId);
        const SyncPhase* phase = access ? schedule.findPhase(access->phase) : nullptr;
        if (!phase || !phase->guard.empty()) {
            rejection = SyncStorageLifecycleRejection::UnsupportedControl;
            return false;
        }
        if (phase->iterationDomain.loops.empty()) {
            continue;
        }
        const bool hasSingleLoop = phase->iterationDomain.loops.size() == 1;
        if (!hasSingleLoop) {
            rejection = SyncStorageLifecycleRejection::MultipleLoops;
            return false;
        }
        const SyncRegionId current = phase->iterationDomain.loops.front();
        if (sawRecurring && loop != current) {
            rejection = SyncStorageLifecycleRejection::MultipleLoops;
            return false;
        }
        loop = current;
        sawRecurring = true;
    }
    if (!sawRecurring) {
        rejection = SyncStorageLifecycleRejection::NotRecurring;
    }
    return sawRecurring;
}

bool classifyProducerConsumer(
    const StructuredSyncIR& schedule, ArrayRef<SyncAccessId> accesses, SyncAccessId& producer, SyncAccessId& consumer)
{
    const bool hasTwoAccesses = accesses.size() == 2;
    if (!hasTwoAccesses) {
        return false;
    }
    for (SyncAccessId accessId : accesses) {
        const SyncAccess* access = schedule.findAccess(accessId);
        if (!access) {
            return false;
        }
        if (access->mode == SyncAccessMode::Write && producer == kInvalidSyncId) {
            producer = accessId;
        } else if (access->mode == SyncAccessMode::Read && consumer == kInvalidSyncId) {
            consumer = accessId;
        } else {
            return false;
        }
    }
    return producer != kInvalidSyncId && consumer != kInvalidSyncId;
}

bool sameTrackSet(ArrayRef<SyncStorageTrackId> first, ArrayRef<SyncStorageTrackId> second)
{
    return first.size() == second.size() &&
           llvm::all_of(first, [&](SyncStorageTrackId track) { return llvm::is_contained(second, track); });
}

bool physicalSlotsComplete(
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SyncStorageTrackId> trackIds, SyncStorageFamilyId family,
    unsigned capacity)
{
    llvm::SmallBitVector slots(capacity);
    for (SyncStorageTrackId trackId : trackIds) {
        if (trackId >= tracks.size()) {
            return false;
        }
        auto binding = llvm::find_if(tracks[trackId].families, [&](const SyncStorageTrackFamilyBinding& candidate) {
            return candidate.family == family;
        });
        if (binding == tracks[trackId].families.end()) {
            return false;
        }
        std::optional<unsigned> slot = binding->physicalSlot;
        if (!slot && capacity == 1) {
            slot = 0;
        }
        if (!slot || *slot >= capacity) {
            return false;
        }
        slots.set(*slot);
    }
    return slots.all();
}

SyncStorageLifecycleReconstruction reconstructComponent(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    ArrayRef<SyncStorageTrackId> component, unsigned componentIndex)
{
    SyncStorageLifecycleReconstruction result;
    result.component = componentIndex;
    const SmallVector<SyncAccessId, 8> accesses = collectComponentAccesses(tracks, component);
    if (!findSingleLoop(schedule, accesses, result.loop, result.rejection)) {
        return result;
    }
    if (!classifyProducerConsumer(schedule, accesses, result.producerAccess, result.consumerAccess)) {
        result.rejection = SyncStorageLifecycleRejection::UnsupportedAccessShape;
        return result;
    }
    const SyncAccess* producer = schedule.findAccess(result.producerAccess);
    const SyncAccess* consumer = schedule.findAccess(result.consumerAccess);
    const SyncPhase* producerPhase = producer ? schedule.findPhase(producer->phase) : nullptr;
    const SyncPhase* consumerPhase = consumer ? schedule.findPhase(consumer->phase) : nullptr;
    if (!producer || !consumer || !producerPhase || !consumerPhase || producer->family != consumer->family) {
        result.rejection = SyncStorageLifecycleRejection::UnsupportedAccessShape;
        return result;
    }
    result.family = producer->family;
    result.producerPhase = producerPhase->id;
    result.consumerPhase = consumerPhase->id;
    const bool producerBeforeConsumer = producerPhase->operation && consumerPhase->operation &&
                                        producerPhase->operation->getBlock() == consumerPhase->operation->getBlock() &&
                                        producerPhase->operation->isBeforeInBlock(consumerPhase->operation);
    if (!producerBeforeConsumer) {
        result.rejection = SyncStorageLifecycleRejection::ProducerAfterConsumer;
        return result;
    }
    ArrayRef<SyncStorageTrackId> producerTracks = accessToTracks[result.producerAccess];
    ArrayRef<SyncStorageTrackId> consumerTracks = accessToTracks[result.consumerAccess];
    const bool accessesCoverSameTracks = sameTrackSet(producerTracks, consumerTracks);
    const bool accessesCoverComponent = sameTrackSet(producerTracks, component);
    if (!accessesCoverSameTracks || !accessesCoverComponent) {
        result.rejection = SyncStorageLifecycleRejection::IncompleteTrackSet;
        return result;
    }
    result.tracks.assign(producerTracks.begin(), producerTracks.end());
    const SyncStorageFamily* family = schedule.findStorageFamily(result.family);
    if (!family || !family->slotCount) {
        result.rejection = SyncStorageLifecycleRejection::UnknownCapacity;
        return result;
    }
    result.capacity = *family->slotCount;
    if (result.capacity != 1 && result.capacity != 2) {
        result.rejection = SyncStorageLifecycleRejection::UnsupportedCapacity;
        return result;
    }
    if (!physicalSlotsComplete(tracks, result.tracks, result.family, result.capacity)) {
        result.rejection = SyncStorageLifecycleRejection::IncompleteTrackSet;
        return result;
    }
    if (result.capacity == 1) {
        result.reuseDistance = 1;
    } else {
        if (!producer->slot || !consumer->slot ||
            compareSlotsAtDistance(*producer->slot, *consumer->slot, 0) != SyncSlotRelation::Same) {
            result.rejection = SyncStorageLifecycleRejection::UnknownSlotRelation;
            return result;
        }
        FailureOr<unsigned> reuse = findFirstPositiveReuseDistance(*producer->slot, result.capacity);
        const bool reuseFailed = failed(reuse);
        if (reuseFailed || *reuse != result.capacity) {
            result.rejection = SyncStorageLifecycleRejection::InvalidReuseDistance;
            return result;
        }
        result.reuseDistance = *reuse;
    }
    const SyncExecutionLane* producerLane = laneFrontiers.findLaneForPhase(result.producerPhase);
    const SyncExecutionLane* consumerLane = laneFrontiers.findLaneForPhase(result.consumerPhase);
    if (!producerLane || !consumerLane) {
        result.rejection = SyncStorageLifecycleRejection::UnresolvedLane;
        return result;
    }
    result.producerLane = producerLane->id;
    result.consumerLane = consumerLane->id;
    result.publication = producerPhase->after;
    result.acquisition = consumerPhase->before;
    result.finalUse = consumerPhase->after;
    result.nextOverwrite = producerPhase->before;
    result.rejection = SyncStorageLifecycleRejection::None;
    return result;
}

const SyncStorageLifecycleReconstruction* findUniqueReady(
    ArrayRef<SyncStorageLifecycleReconstruction> reconstructions, unsigned& readyCount)
{
    const SyncStorageLifecycleReconstruction* ready = nullptr;
    readyCount = 0;
    for (const SyncStorageLifecycleReconstruction& reconstruction : reconstructions) {
        if (!reconstruction.isReady()) {
            continue;
        }
        ++readyCount;
        if (!ready) {
            ready = &reconstruction;
        }
    }
    return readyCount == 1 ? ready : nullptr;
}

bool lanesMatchPlan(
    const LaneFrontierAnalysisResult& laneFrontiers, const SyncStorageLifecycleReconstruction& reconstruction,
    const SyncReadyReleasePlan& plan)
{
    const SyncExecutionLane* producer = laneFrontiers.findLane(reconstruction.producerLane);
    const SyncExecutionLane* consumer = laneFrontiers.findLane(reconstruction.consumerLane);
    return producer && consumer && producer->core == plan.core && consumer->core == plan.core &&
           producer->pipe == plan.producerPipe && consumer->pipe == plan.consumerPipe;
}

SyncStorageEDifferential compareWithE(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SyncStorageLifecycleReconstruction> reconstructions,
    const SyncReadyReleasePlan* plan)
{
    SyncStorageEDifferential differential;
    if (!plan) {
        differential.status = SyncStorageEDifferentialStatus::Unavailable;
        return differential;
    }
    unsigned readyCount = 0;
    const SyncStorageLifecycleReconstruction* reconstruction = findUniqueReady(reconstructions, readyCount);
    const bool eReady = plan->status == SyncReadyReleasePlanStatus::Ready;
    if (!eReady) {
        if (plan->status == SyncReadyReleasePlanStatus::Unsupported && !plan->rejections.empty()) {
            differential.eRejection = plan->rejections.front().reason;
        }
        differential.status =
            readyCount != 0 ? SyncStorageEDifferentialStatus::IndependentOnly : SyncStorageEDifferentialStatus::Neither;
        return differential;
    }
    if (readyCount > 1) {
        differential.status = SyncStorageEDifferentialStatus::Mismatch;
        return differential;
    }
    if (!reconstruction) {
        differential.status = SyncStorageEDifferentialStatus::EOnly;
        return differential;
    }
    differential.capacityMatches = reconstruction->capacity == plan->capacity;
    differential.lanesMatch = lanesMatchPlan(laneFrontiers, *reconstruction, *plan);
    differential.loopMatches = reconstruction->loop == plan->loopRegion;
    differential.phasesMatch =
        reconstruction->producerPhase == plan->producerPhase && reconstruction->consumerPhase == plan->consumerPhase;
    const SyncPhase* producer = schedule.findPhase(plan->producerPhase);
    const SyncPhase* consumer = schedule.findPhase(plan->consumerPhase);
    differential.lifecycleMatches =
        producer && consumer && reconstruction->publication == producer->after &&
        reconstruction->acquisition == consumer->before && reconstruction->finalUse == consumer->after &&
        reconstruction->nextOverwrite == producer->before && reconstruction->reuseDistance == plan->capacity;
    differential.physicalSlotsMatch =
        physicalSlotsComplete(tracks, reconstruction->tracks, reconstruction->family, reconstruction->capacity);
    const bool matches = differential.capacityMatches && differential.lanesMatch && differential.loopMatches &&
                         differential.phasesMatch && differential.lifecycleMatches && differential.physicalSlotsMatch;
    differential.status = matches ? SyncStorageEDifferentialStatus::Match : SyncStorageEDifferentialStatus::Mismatch;
    return differential;
}

void updateStatistics(
    ArrayRef<SyncStorageLifecycleReconstruction> reconstructions, const SyncStorageEDifferential& differential,
    ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    statistics->storageLifecycleComponentsAttempted += reconstructions.size();
    for (const SyncStorageLifecycleReconstruction& reconstruction : reconstructions) {
        if (reconstruction.isReady()) {
            ++statistics->storageLifecyclesReconstructed;
        } else {
            ++statistics
                  ->storageLifecycleRejections[stringifySyncStorageLifecycleRejection(reconstruction.rejection).str()];
        }
    }
    statistics->storageLifecycleEMatches += differential.status == SyncStorageEDifferentialStatus::Match ? 1 : 0;
    statistics->storageLifecycleEMismatches += differential.status == SyncStorageEDifferentialStatus::Mismatch ? 1 : 0;
    statistics->storageLifecycleIndependentOnly +=
        differential.status == SyncStorageEDifferentialStatus::IndependentOnly ? 1 : 0;
    statistics->storageLifecycleEOnly += differential.status == SyncStorageEDifferentialStatus::EOnly ? 1 : 0;
    if (differential.status == SyncStorageEDifferentialStatus::IndependentOnly) {
        ++statistics->storageLifecycleIndependentERejections[stringifySyncReadyReleaseRejection(differential.eRejection)
                                                                 .str()];
    }
}

} // namespace

void mlir::pto::protocol_sync::detail::reconstructStorageLifecycles(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    ArrayRef<SyncStorageTrack> tracks, ArrayRef<SmallVector<SyncStorageTrackId, 2>> accessToTracks,
    const SyncReadyReleasePlan* checkpointEPlan, SmallVectorImpl<SyncStorageLifecycleReconstruction>& reconstructions,
    SyncStorageEDifferential& differential, ProtocolSyncStatistics* statistics)
{
    SmallVector<SyncStorageTrackComponent, 16> components = buildStorageTrackComponents(tracks, accessToTracks);
    for (auto [index, component] : llvm::enumerate(components)) {
        SyncStorageLifecycleReconstruction reconstruction = reconstructComponent(
            schedule, laneFrontiers, tracks, accessToTracks, component, static_cast<unsigned>(index));
        if (reconstruction.rejection != SyncStorageLifecycleRejection::NotRecurring) {
            reconstructions.push_back(std::move(reconstruction));
        }
    }
    differential = compareWithE(schedule, laneFrontiers, tracks, reconstructions, checkpointEPlan);
    updateStatistics(reconstructions, differential, statistics);
}
