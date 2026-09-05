// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- LanePatternCompletion.cpp - Raw same-lane completion cuts -------===//

#include "LanePatternInternal.h"

#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct RetainedEndpoint {
    SyncGenerationId generation = kInvalidSyncId;
    const SyncRawAccessEndpoint* endpoint = nullptr;
};

struct SameLanePairGroup {
    SyncExecutionLaneId lane = kInvalidSyncId;
    Block* block = nullptr;
    SmallVector<unsigned, 8> pairs;
};

bool reads(SyncAccessMode mode) { return mode == SyncAccessMode::Read || mode == SyncAccessMode::ReadWrite; }

bool writes(SyncAccessMode mode) { return mode == SyncAccessMode::Write || mode == SyncAccessMode::ReadWrite; }

std::optional<std::uint64_t> intervalEnd(const SyncByteInterval& interval)
{
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    if (interval.size == 0 || interval.begin > maximum - interval.size) {
        return std::nullopt;
    }
    return interval.begin + interval.size;
}

bool storageOverlaps(const SyncAccess& first, const SyncAccess& second)
{
    const SyncStorageProvenance& left = first.storage;
    const SyncStorageProvenance& right = second.storage;
    if (!left.physical || !right.physical || left.unknownRange || right.unknownRange || left.aliasesUnknownRange ||
        right.aliasesUnknownRange || left.space != right.space) {
        return false;
    }
    for (const SyncByteInterval& leftInterval : left.intervals) {
        const std::optional<std::uint64_t> leftEnd = intervalEnd(leftInterval);
        if (!leftEnd) {
            continue;
        }
        for (const SyncByteInterval& rightInterval : right.intervals) {
            const std::optional<std::uint64_t> rightEnd = intervalEnd(rightInterval);
            if (rightEnd && leftInterval.begin < *rightEnd && rightInterval.begin < *leftEnd) {
                return true;
            }
        }
    }
    return false;
}

std::optional<SyncLaneRawHazardKind> classifyHazard(const SyncAccess& source, const SyncAccess& target)
{
    const bool sourceReads = reads(source.mode);
    const bool sourceWrites = writes(source.mode);
    const bool targetReads = reads(target.mode);
    const bool targetWrites = writes(target.mode);
    if (sourceWrites && targetWrites) {
        return SyncLaneRawHazardKind::WriteAfterWrite;
    }
    if (sourceWrites && targetReads) {
        return SyncLaneRawHazardKind::ReadAfterWrite;
    }
    if (sourceReads && targetWrites) {
        return SyncLaneRawHazardKind::WriteAfterRead;
    }
    return std::nullopt;
}

bool isLinearBefore(const SyncPhase& source, const SyncPhase& target)
{
    return source.operation && target.operation && source.operation != target.operation &&
           source.operation->getBlock() == target.operation->getBlock() &&
           source.operation->isBeforeInBlock(target.operation);
}

const SyncPhase* findEndpointPhase(const StructuredSyncIR& schedule, const RetainedEndpoint& retained)
{
    return retained.endpoint ? schedule.findPhase(retained.endpoint->phase) : nullptr;
}

SmallVector<RetainedEndpoint, 64> collectEndpoints(const StorageTimelineAnalysisResult& timelines)
{
    SmallVector<RetainedEndpoint, 64> endpoints;
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        for (const SyncRawAccessEndpoint& endpoint : timeline.rawAccesses) {
            endpoints.push_back({timeline.id, &endpoint});
        }
    }
    llvm::sort(endpoints, [](const RetainedEndpoint& left, const RetainedEndpoint& right) {
        return left.endpoint->access < right.endpoint->access;
    });
    return endpoints;
}

SmallVector<SameLanePairGroup, 4> groupSameLanePairs(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, ArrayRef<SyncLaneRawAccessPair> pairs)
{
    SmallVector<SameLanePairGroup, 4> groups;
    for (auto [index, pair] : llvm::enumerate(pairs)) {
        if (pair.sourceLane == kInvalidSyncId || pair.sourceLane != pair.targetLane ||
            pair.completion != ProtocolSyncSameLaneCompletion::PipeBarrier) {
            continue;
        }
        const SyncStage* sourceStage = stages.findStage(pair.sourceStage);
        const SyncStage* targetStage = stages.findStage(pair.targetStage);
        const SyncPhase* source = schedule.findPhase(pair.sourcePhase);
        const SyncPhase* target = schedule.findPhase(pair.targetPhase);
        const bool unsupported = !sourceStage || !targetStage || !source || !target || !source->operation ||
                                 !target->operation || !source->guard.empty() || !target->guard.empty() ||
                                 !source->iterationDomain.loops.empty() || !target->iterationDomain.loops.empty() ||
                                 source->operation->getBlock() != target->operation->getBlock();
        if (unsupported) {
            continue;
        }
        Block* block = target->operation->getBlock();
        auto group = llvm::find_if(groups, [&](const SameLanePairGroup& current) {
            return current.lane == pair.sourceLane && current.block == block;
        });
        if (group == groups.end()) {
            groups.push_back({pair.sourceLane, block, {}});
            group = std::prev(groups.end());
        }
        group->pairs.push_back(static_cast<unsigned>(index));
    }
    return groups;
}

void appendUniqueGeneration(SmallVectorImpl<SyncGenerationId>& generations, SyncGenerationId generation)
{
    if (generation != kInvalidSyncId && !llvm::is_contained(generations, generation)) {
        generations.push_back(generation);
    }
}

} // namespace

SmallVector<SyncLaneRawAccessPair, 32> mlir::pto::protocol_sync::detail::buildLaneRawAccessPairs(
    const StructuredSyncIR& schedule, const StorageTimelineAnalysisResult& timelines,
    const LaneFrontierAnalysisResult& frontiers)
{
    SmallVector<SyncLaneRawAccessPair, 32> pairs;
    SmallVector<RetainedEndpoint, 64> endpoints = collectEndpoints(timelines);
    for (auto [targetIndex, targetRetained] : llvm::enumerate(endpoints)) {
        const SyncRawAccessEndpoint& targetEndpoint = *targetRetained.endpoint;
        const SyncAccess* targetAccess = schedule.findAccess(targetEndpoint.access);
        const SyncPhase* targetPhase = findEndpointPhase(schedule, targetRetained);
        if (!targetAccess || !targetPhase) {
            continue;
        }
        for (const RetainedEndpoint& sourceRetained : llvm::reverse(ArrayRef(endpoints).take_front(targetIndex))) {
            const SyncRawAccessEndpoint& sourceEndpoint = *sourceRetained.endpoint;
            const SyncAccess* sourceAccess = schedule.findAccess(sourceEndpoint.access);
            const SyncPhase* sourcePhase = findEndpointPhase(schedule, sourceRetained);
            const bool incompatible = !sourceAccess || !sourcePhase || !isLinearBefore(*sourcePhase, *targetPhase) ||
                                      !storageOverlaps(*sourceAccess, *targetAccess);
            if (incompatible) {
                continue;
            }
            const std::optional<SyncLaneRawHazardKind> hazard = classifyHazard(*sourceAccess, *targetAccess);
            if (!hazard) {
                continue;
            }
            const SyncExecutionLane* sourceLane = frontiers.findLaneForPhase(sourceEndpoint.phase);
            const SyncExecutionLane* targetLane = frontiers.findLaneForPhase(targetEndpoint.phase);
            SyncLaneRawAccessPair pair;
            pair.id = static_cast<SyncLaneRawAccessPairId>(pairs.size());
            pair.sourceAccess = sourceEndpoint.access;
            pair.targetAccess = targetEndpoint.access;
            pair.sourceGeneration = sourceRetained.generation;
            pair.targetGeneration = targetRetained.generation;
            pair.sourcePhase = sourceEndpoint.phase;
            pair.targetPhase = targetEndpoint.phase;
            pair.sourceStage = sourceEndpoint.stage;
            pair.targetStage = targetEndpoint.stage;
            pair.sourceLane = sourceLane ? sourceLane->id : kInvalidSyncId;
            pair.targetLane = targetLane ? targetLane->id : kInvalidSyncId;
            pair.sourceAfter = sourceEndpoint.after;
            pair.targetBefore = targetEndpoint.before;
            pair.hazard = *hazard;
            pairs.push_back(pair);
        }
    }
    return pairs;
}

void mlir::pto::protocol_sync::detail::appendSameLaneCompletionCuts(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    ArrayRef<SyncLaneRawAccessPair> rawPairs, SmallVectorImpl<SyncLanePatternCandidate>& candidates)
{
    for (SameLanePairGroup& group : groupSameLanePairs(schedule, stages, rawPairs)) {
        llvm::sort(group.pairs, [&](unsigned left, unsigned right) {
            return rawPairs[left].targetBefore < rawPairs[right].targetBefore;
        });
        llvm::SmallBitVector covered(group.pairs.size());
        for (unsigned candidateIndex = 0; candidateIndex < group.pairs.size(); ++candidateIndex) {
            if (covered.test(candidateIndex)) {
                continue;
            }
            const SyncLaneRawAccessPair& seed = rawPairs[group.pairs[candidateIndex]];
            const SyncProgramPointId cut = seed.targetBefore;
            SyncLanePatternCandidate candidate;
            candidate.kind = SyncLanePatternKind::SameLaneCompletionCut;
            candidate.referencePattern = SyncLaneReferencePattern::MultiDemandPipeBarrier;
            candidate.sourceLane = group.lane;
            candidate.targetLane = group.lane;
            candidate.firstSource.points.push_back({cut, {}});
            candidate.cost.steadyStateActions = 1;
            for (unsigned memberIndex = candidateIndex; memberIndex < group.pairs.size(); ++memberIndex) {
                const SyncLaneRawAccessPair& pair = rawPairs[group.pairs[memberIndex]];
                if (pair.sourceAfter > cut || cut > pair.targetBefore) {
                    continue;
                }
                covered.set(memberIndex);
                candidate.rawPairMembers.push_back(pair.id);
                appendUniqueGeneration(candidate.generations, pair.sourceGeneration);
                appendUniqueGeneration(candidate.generations, pair.targetGeneration);
            }
            candidates.push_back(std::move(candidate));
        }
    }
}
