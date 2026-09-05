// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTimeline.cpp - Derive storage-generation timelines -------===//

#include "PTO/Transforms/ProtocolSync/StorageTimeline.h"

#include "StorageTimelineInternal.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace mlir::pto::protocol_sync::detail;

namespace {

void reject(SyncGenerationTimeline& timeline, SyncTimelineRejection reason, ProtocolSyncStatistics* statistics)
{
    timeline.rejection = reason;
    if (statistics) {
        ++statistics->generationTimelinesRejected;
        ++statistics->generationRejections[stringifySyncTimelineRejection(reason).str()];
    }
}

void retainRawAccesses(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const StorageAccessClass& accessClass,
    SyncGenerationTimeline& timeline, ProtocolSyncStatistics* statistics)
{
    for (SyncAccessId id : accessClass.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        const SyncPhase* phase = access ? schedule.findPhase(access->phase) : nullptr;
        const SyncStage* stage = phase ? stages.findStageForPhase(phase->id) : nullptr;
        SyncRawAccessEndpoint endpoint;
        endpoint.access = id;
        endpoint.phase = phase ? phase->id : kInvalidSyncId;
        endpoint.stage = stage ? stage->id : kInvalidSyncId;
        endpoint.before = phase ? phase->before : kInvalidSyncId;
        endpoint.after = phase ? phase->after : kInvalidSyncId;
        endpoint.mode = access ? access->mode : SyncAccessMode::ReadWrite;
        timeline.rawAccesses.push_back(endpoint);
        if (statistics) {
            ++statistics->rawAccessEndpointsRetained;
        }
    }
}

bool buildStageSets(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const StorageAccessClass& accessClass,
    SyncGenerationTimeline& timeline, SyncTimelineRejection& rejection)
{
    bool inPlace = accessClass.inPlace;
    llvm::SetVector<SyncStageId> producers;
    llvm::SetVector<SyncStageId> consumers;
    for (SyncAccessId id : accessClass.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        const SyncStage* stage = access ? stages.findStageForPhase(access->phase) : nullptr;
        if (!access || !stage) {
            rejection = SyncTimelineRejection::InvalidStage;
            return false;
        }
        if (access->mode == SyncAccessMode::Ordered) {
            rejection = SyncTimelineRejection::OrderedAccess;
            return false;
        }
        if (access->mode == SyncAccessMode::ReadWrite) {
            producers.insert(stage->id);
            consumers.insert(stage->id);
            inPlace = true;
        } else if (access->mode == SyncAccessMode::Write) {
            producers.insert(stage->id);
        } else if (access->mode == SyncAccessMode::Read) {
            consumers.insert(stage->id);
        }
    }
    for (SyncStageId producer : producers) {
        if (consumers.contains(producer)) {
            inPlace = true;
        }
    }
    timeline.producers.assign(producers.begin(), producers.end());
    timeline.consumers.assign(consumers.begin(), consumers.end());
    if (inPlace) {
        rejection = SyncTimelineRejection::InPlaceAccess;
        return false;
    }
    return true;
}

void appendFrontierPoint(SyncProgramFrontier& frontier, const SyncStage& stage, SyncProgramPointId point)
{
    frontier.points.push_back({point, stage.guard});
}

bool populateFrontiers(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncGenerationTimeline& timeline,
    SyncTimelineRejection& rejection)
{
    for (SyncStageId producerId : timeline.producers) {
        const SyncStage* producer = stages.findStage(producerId);
        const SyncPhase* phase =
            producer && !producer->phases.empty() ? schedule.findPhase(producer->phases.front()) : nullptr;
        if (!producer || !phase) {
            rejection = SyncTimelineRejection::InvalidStage;
            return false;
        }
        appendFrontierPoint(timeline.publication, *producer, phase->after);
    }
    for (SyncStageId consumerId : timeline.consumers) {
        const SyncStage* consumer = stages.findStage(consumerId);
        const SyncPhase* phase =
            consumer && !consumer->phases.empty() ? schedule.findPhase(consumer->phases.front()) : nullptr;
        if (!consumer || !phase) {
            rejection = SyncTimelineRejection::InvalidStage;
            return false;
        }
        SyncProgramFrontier acquisition;
        appendFrontierPoint(acquisition, *consumer, phase->before);
        timeline.acquisitions.push_back(std::move(acquisition));
        SyncProgramFrontier finalUse;
        appendFrontierPoint(finalUse, *consumer, phase->after);
        timeline.finalUses.push_back(std::move(finalUse));
    }
    return true;
}

bool deriveIteration(
    const PipelineStageAnalysisResult& stages, SyncGenerationTimeline& timeline, SyncTimelineRejection& rejection)
{
    const SyncStage* reference = stages.findStage(timeline.producers.front());
    if (!reference) {
        rejection = SyncTimelineRejection::InvalidStage;
        return false;
    }
    for (SyncStageId id : timeline.producers) {
        const SyncStage* stage = stages.findStage(id);
        if (!stage || stage->iterationDomain.loops != reference->iterationDomain.loops) {
            rejection = SyncTimelineRejection::IncompatibleIterationDomain;
            return false;
        }
    }
    for (SyncStageId id : timeline.consumers) {
        const SyncStage* stage = stages.findStage(id);
        if (!stage || stage->iterationDomain.loops != reference->iterationDomain.loops) {
            rejection = SyncTimelineRejection::IncompatibleIterationDomain;
            return false;
        }
    }
    if (reference->iterationDomain.loops.empty()) {
        return true;
    }
    timeline.generationKind = SyncGenerationKind::LoopIteration;
    timeline.carryingRegion = reference->iterationDomain.loops.back();
    return true;
}

bool deriveReuse(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncStorageFamily& family,
    SyncGenerationTimeline& timeline, SyncTimelineRejection& rejection)
{
    if (timeline.generationKind != SyncGenerationKind::LoopIteration) {
        return true;
    }
    if (*family.slotCount == 1 && !timeline.slot) {
        timeline.reuseDistance = 1;
    } else {
        FailureOr<unsigned> reuse = findFirstPositiveReuseDistance(*timeline.slot, *family.slotCount);
        if (failed(reuse)) {
            rejection = SyncTimelineRejection::UnknownReuseDistance;
            return false;
        }
        timeline.reuseDistance = *reuse;
    }
    SyncNextOverwrite next;
    next.iterationDistance = *timeline.reuseDistance;
    for (SyncStageId producerId : timeline.producers) {
        const SyncStage* producer = stages.findStage(producerId);
        const SyncPhase* phase =
            producer && !producer->phases.empty() ? schedule.findPhase(producer->phases.front()) : nullptr;
        if (!producer || !phase) {
            rejection = SyncTimelineRejection::InvalidStage;
            return false;
        }
        appendFrontierPoint(next.frontier, *producer, phase->before);
    }
    timeline.nextOverwrite = std::move(next);
    return true;
}

void linkStaticOverwrites(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    MutableArrayRef<SyncGenerationTimeline> timelines)
{
    for (auto [index, timeline] : llvm::enumerate(timelines.drop_back())) {
        SyncGenerationTimeline& next = timelines[index + 1];
        const bool sameStorageClass = timeline.family == next.family &&
                                      sameTimelineIntervals(timeline.slice, next.slice) &&
                                      sameTimelineSlotClass(timeline.slot, next.slot);
        const bool cannotLink = !timeline.isAdmitted() || !next.isAdmitted() || !sameStorageClass ||
                                next.producers.empty() || timeline.generationKind != SyncGenerationKind::OneShot;
        if (cannotLink) {
            continue;
        }
        SyncNextOverwrite overwrite;
        for (SyncStageId producerId : next.producers) {
            const SyncStage* producer = stages.findStage(producerId);
            const SyncPhase* phase =
                producer && !producer->phases.empty() ? schedule.findPhase(producer->phases.front()) : nullptr;
            if (!producer || !phase) {
                overwrite.frontier.points.clear();
                break;
            }
            appendFrontierPoint(overwrite.frontier, *producer, phase->before);
        }
        if (!overwrite.frontier.points.empty()) {
            timeline.nextOverwrite = std::move(overwrite);
        }
    }
}

} // namespace

StorageTimelineAnalysisResult mlir::pto::protocol_sync::analyzeStorageTimelines(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, ProtocolSyncStatistics* statistics)
{
    StorageTimelineAnalysisResult result;
    SmallVector<StorageAccessClass, 16> generations = buildGenerationAccessClasses(schedule, statistics);
    for (const StorageAccessClass& accessClass : generations) {
        SyncGenerationTimeline timeline;
        timeline.id = result.timelines.size();
        timeline.family = accessClass.family;
        timeline.accesses.assign(accessClass.accesses.begin(), accessClass.accesses.end());
        timeline.slice = accessClass.slice;
        timeline.slot = accessClass.slot;
        retainRawAccesses(schedule, stages, accessClass, timeline, statistics);
        if (statistics) {
            ++statistics->generationTimelinesAttempted;
        }
        const SyncStorageFamily* family = schedule.findStorageFamily(accessClass.family);
        SyncTimelineRejection rejection = SyncTimelineRejection::None;
        if (!family || family->role != SyncStorageRole::LocalBuffer || !family->physical) {
            rejection = SyncTimelineRejection::NonPhysicalLocalStorage;
        } else if (
            family->unknownRange || family->aliasesUnknownRange || accessClass.unknownRange ||
            accessClass.aliasesUnknownRange || accessClass.slice.empty()) {
            rejection = SyncTimelineRejection::UnknownRange;
        } else if (accessClass.conflictingPhysicalRange) {
            rejection = SyncTimelineRejection::ConflictingPhysicalRange;
        } else if (!family->slotCount) {
            rejection = SyncTimelineRejection::UnknownCapacity;
        } else if (*family->slotCount < 1 || *family->slotCount > 2) {
            rejection = SyncTimelineRejection::UnsupportedCapacity;
        } else if (*family->slotCount > 1 && !family->physicalSlotsComplete) {
            rejection = SyncTimelineRejection::IncompletePhysicalSlots;
        } else if (
            *family->slotCount > 1 && (!timeline.slot || timeline.slot->kind == SyncSlotExpressionKind::Unknown)) {
            rejection = SyncTimelineRejection::UnknownSlotExpression;
        } else if (accessClass.partialOverlap) {
            rejection = SyncTimelineRejection::PartialOverlap;
        }
        if (rejection == SyncTimelineRejection::None) {
            (void)buildStageSets(schedule, stages, accessClass, timeline, rejection);
        }
        if (rejection == SyncTimelineRejection::None && timeline.producers.empty()) {
            rejection = SyncTimelineRejection::MissingProducer;
        } else if (rejection == SyncTimelineRejection::None && timeline.consumers.empty()) {
            rejection = SyncTimelineRejection::MissingConsumer;
        } else if (
            rejection == SyncTimelineRejection::None &&
            *llvm::max_element(timeline.producers) >= *llvm::min_element(timeline.consumers)) {
            rejection = SyncTimelineRejection::ProducerAfterConsumer;
        }
        if (rejection == SyncTimelineRejection::None) {
            (void)deriveIteration(stages, timeline, rejection);
        }
        if (rejection == SyncTimelineRejection::None && timeline.generationKind == SyncGenerationKind::LoopIteration &&
            accessClass.multipleGenerations) {
            rejection = SyncTimelineRejection::MultipleGenerationsPerIteration;
        }
        if (rejection == SyncTimelineRejection::None) {
            (void)populateFrontiers(schedule, stages, timeline, rejection);
        }
        if (rejection == SyncTimelineRejection::None) {
            (void)deriveReuse(schedule, stages, *family, timeline, rejection);
        }
        if (rejection != SyncTimelineRejection::None) {
            reject(timeline, rejection, statistics);
        } else if (statistics) {
            ++statistics->generationTimelinesAdmitted;
        }
        result.timelines.push_back(std::move(timeline));
    }
    const bool hasMultipleTimelines = result.timelines.size() > 1;
    if (hasMultipleTimelines) {
        linkStaticOverwrites(schedule, stages, result.timelines);
    }
    return result;
}

StringRef mlir::pto::protocol_sync::stringifySyncGenerationKind(SyncGenerationKind kind)
{
    switch (kind) {
        case SyncGenerationKind::OneShot:
            return "one-shot";
        case SyncGenerationKind::LoopIteration:
            return "loop-iteration";
    }
    return "one-shot";
}

StringRef mlir::pto::protocol_sync::stringifySyncTimelineRejection(SyncTimelineRejection reason)
{
    switch (reason) {
        case SyncTimelineRejection::None:
            return "none";
        case SyncTimelineRejection::NonPhysicalLocalStorage:
            return "non-physical-local-storage";
        case SyncTimelineRejection::UnknownRange:
            return "unknown-range";
        case SyncTimelineRejection::ConflictingPhysicalRange:
            return "conflicting-physical-range";
        case SyncTimelineRejection::UnknownCapacity:
            return "unknown-capacity";
        case SyncTimelineRejection::UnsupportedCapacity:
            return "unsupported-capacity";
        case SyncTimelineRejection::IncompletePhysicalSlots:
            return "incomplete-physical-slots";
        case SyncTimelineRejection::UnknownSlotExpression:
            return "unknown-slot-expression";
        case SyncTimelineRejection::PartialOverlap:
            return "partial-overlap";
        case SyncTimelineRejection::MissingProducer:
            return "missing-producer";
        case SyncTimelineRejection::MissingConsumer:
            return "missing-consumer";
        case SyncTimelineRejection::OrderedAccess:
            return "ordered-access";
        case SyncTimelineRejection::InPlaceAccess:
            return "in-place-access";
        case SyncTimelineRejection::InvalidStage:
            return "invalid-stage";
        case SyncTimelineRejection::ProducerAfterConsumer:
            return "producer-after-consumer";
        case SyncTimelineRejection::IncompatibleIterationDomain:
            return "incompatible-iteration-domain";
        case SyncTimelineRejection::MultipleGenerationsPerIteration:
            return "multiple-generations-per-iteration";
        case SyncTimelineRejection::UnknownReuseDistance:
            return "unknown-reuse-distance";
    }
    return "unknown-range";
}
