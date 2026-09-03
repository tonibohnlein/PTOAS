// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ChannelAnalysis.cpp - Strict lifecycle channel diagnostics ------===//

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <iterator>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

void reject(SyncChannel& channel, SyncChannelRejection reason, ProtocolSyncStatistics* statistics)
{
    channel.rejection = reason;
    if (statistics) {
        ++statistics->channelCandidatesRejected;
        ++statistics->channelRejections[stringifySyncChannelRejection(reason).str()];
    }
}

bool hasUnsupportedControl(
    const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline)
{
    for (SyncStageId id : timeline.producers) {
        const SyncStage* stage = stages.findStage(id);
        if (!stage || !stage->guard.empty()) {
            return true;
        }
    }
    for (SyncStageId id : timeline.consumers) {
        const SyncStage* stage = stages.findStage(id);
        if (!stage || !stage->guard.empty()) {
            return true;
        }
    }
    return false;
}

bool hasUnknownRole(const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline)
{
    const SyncStage* producer = stages.findStage(timeline.producers.front());
    const SyncStage* consumer = stages.findStage(timeline.consumers.front());
    return !producer || !consumer || producer->role == SyncStageRole::Unknown ||
           consumer->role == SyncStageRole::Unknown;
}

bool crossesPhysicalCores(const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline)
{
    const SyncStage* producer = stages.findStage(timeline.producers.front());
    const SyncStage* consumer = stages.findStage(timeline.consumers.front());
    return !producer || !consumer || producer->core == SyncPhysicalCore::Unknown || consumer->core != producer->core;
}

using LegacyStorageIndex = llvm::DenseMap<Value, SmallVector<const BaseMemInfo*, 1>>;

struct LegacyPhaseInfo {
    const CompoundInstanceElement* phase = nullptr;
    SmallVector<unsigned, 2> loops;
    LegacyStorageIndex definitions;
    LegacyStorageIndex uses;
};

LegacyStorageIndex indexLegacyStorage(ArrayRef<const BaseMemInfo*> accesses)
{
    LegacyStorageIndex result;
    for (const BaseMemInfo* access : accesses) {
        if (access && access->rootBuffer) {
            result[access->rootBuffer].push_back(access);
        }
    }
    return result;
}

SmallVector<LegacyPhaseInfo, 32> indexLegacyPhases(const LegacySyncSnapshot& legacy)
{
    SmallVector<LegacyPhaseInfo, 32> result;
    SmallVector<unsigned, 2> loopStack;
    for (const std::unique_ptr<InstanceElement>& element : legacy.syncIR) {
        if (auto* loop = dyn_cast<LoopInstanceElement>(element.get())) {
            const bool isLoopBegin = loop->getLoopKind() == KindOfLoop::LOOP_BEGIN;
            if (isLoopBegin) {
                loopStack.push_back(loop->GetIndex());
            } else if (!loopStack.empty() && loopStack.back() == loop->beginId) {
                loopStack.pop_back();
            }
            continue;
        }
        if (auto* compound = dyn_cast<CompoundInstanceElement>(element.get())) {
            result.push_back(
                {compound, loopStack, indexLegacyStorage(compound->defVec), indexLegacyStorage(compound->useVec)});
        }
    }
    return result;
}

std::optional<unsigned> getLegacyMacroPhase(const CompoundInstanceElement& phase)
{
    return phase.macroOpInstanceId < 0 ?
               std::nullopt :
               std::optional<unsigned>(static_cast<unsigned>(phase.macroOpInstanceId));
}
const LegacyPhaseInfo* findLegacyPhase(ArrayRef<LegacyPhaseInfo> phases, const SyncPhase& phase)
{
    const LegacyPhaseInfo* result = nullptr;
    for (const LegacyPhaseInfo& candidate : phases) {
        if (!candidate.phase || candidate.phase->elementOp != phase.operation ||
            getLegacyMacroPhase(*candidate.phase) != phase.macroPhase) {
            continue;
        }
        // Duplicate matches mean the mapping is ambiguous; keep the oracle
        // unavailable rather than silently pairing the wrong dynamic phase.
        if (result) {
            return nullptr;
        }
        result = &candidate;
    }
    return result;
}

const SyncPhase* getOnlyPhase(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncStageId id)
{
    const SyncStage* stage = stages.findStage(id);
    const bool doesNotHaveOnePhase = !stage || stage->phases.size() != 1;
    if (doesNotHaveOnePhase) {
        return nullptr;
    }
    return schedule.findPhase(stage->phases.front());
}

bool matchesTimelineStorage(
    const BaseMemInfo& legacy, const SyncStorageFamily& family, const SyncGenerationTimeline& timeline)
{
    if (!legacy.hasKnownPhysicalAddresses || legacy.aliasesUnknownRange || legacy.rootBuffer != family.root ||
        legacy.scope != family.space || legacy.baseAddresses.size() != timeline.slice.size()) {
        return false;
    }
    return llvm::all_of(llvm::zip(legacy.baseAddresses, timeline.slice), [&](const auto& pair) {
        const auto& [address, interval] = pair;
        return address == interval.begin && legacy.allocateSize == interval.size;
    });
}

SmallVector<const BaseMemInfo*> filterTimelineStorage(
    const LegacyStorageIndex& accesses, const SyncStorageFamily& family, const SyncGenerationTimeline& timeline)
{
    SmallVector<const BaseMemInfo*> result;
    auto root = accesses.find(family.root);
    if (root == accesses.end()) {
        return result;
    }
    llvm::copy_if(root->second, std::back_inserter(result), [&](const BaseMemInfo* access) {
        return access && matchesTimelineStorage(*access, family, timeline);
    });
    return result;
}

bool hasLegacyForwardOrder(const LegacyPhaseInfo& producer, const LegacyPhaseInfo& consumer)
{
    return producer.phase && consumer.phase && producer.phase->GetIndex() < consumer.phase->GetIndex();
}

bool hasLegacyLoopBackedge(const LegacyPhaseInfo& producer, const LegacyPhaseInfo& consumer)
{
    return hasLegacyForwardOrder(producer, consumer) && !producer.loops.empty() && producer.loops == consumer.loops;
}

SyncDemandOracleStatus compareLegacyDependency(
    MemoryDependentAnalyzer& analyzer, const SmallVector<const BaseMemInfo*>& first,
    const SmallVector<const BaseMemInfo*>& second)
{
    DepBaseMemInfoPairVec dependencies;
    return analyzer.DepBetween(first, second, dependencies) ? SyncDemandOracleStatus::Match :
                                                              SyncDemandOracleStatus::Mismatch;
}

} // namespace

ChannelAnalysisResult mlir::pto::protocol_sync::analyzeChannels(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, ProtocolSyncStatistics* statistics)
{
    ChannelAnalysisResult result;
    for (const SyncGenerationTimeline& timeline : timelines.getTimelines()) {
        SyncChannel channel;
        channel.id = result.channels.size();
        channel.generation = timeline.id;
        const SyncStorageFamily* family = schedule.findStorageFamily(timeline.family);
        channel.capacity = family && family->slotCount ? *family->slotCount : 0;
        if (statistics) {
            ++statistics->channelCandidatesAttempted;
        }
        SyncChannelRejection rejection = SyncChannelRejection::None;
        if (!timeline.isAdmitted()) {
            rejection = SyncChannelRejection::TimelineRejected;
        } else if (!schedule.getFailures().empty()) {
            rejection = SyncChannelRejection::UnresolvedScheduleFailure;
        } else if (timeline.producers.size() != 1) {
            rejection = SyncChannelRejection::MultipleProducers;
        } else if (timeline.consumers.size() != 1) {
            rejection = SyncChannelRejection::MultipleConsumers;
        } else if (hasUnsupportedControl(stages, timeline)) {
            rejection = SyncChannelRejection::UnsupportedControlFlow;
        } else {
            const SyncStage* producer = stages.findStage(timeline.producers.front());
            const bool hasNestedLoop = producer && producer->iterationDomain.loops.size() > 1;
            if (hasNestedLoop) {
                rejection = SyncChannelRejection::NestedLoop;
            } else if (crossesPhysicalCores(stages, timeline)) {
                rejection = SyncChannelRejection::CrossCore;
            } else if (hasUnknownRole(stages, timeline)) {
                rejection = SyncChannelRejection::UnknownStageRole;
            } else if (timeline.generationKind == SyncGenerationKind::OneShot && timeline.nextOverwrite) {
                rejection = SyncChannelRejection::StaticOverwrite;
            } else if (
                timeline.generationKind == SyncGenerationKind::LoopIteration &&
                (!timeline.reuseDistance || *timeline.reuseDistance != channel.capacity)) {
                rejection = SyncChannelRejection::ReuseCapacityMismatch;
            }
        }
        channel.kind = timeline.generationKind == SyncGenerationKind::LoopIteration ? SyncChannelKind::ReadyRelease :
                                                                                      SyncChannelKind::OneShot;
        if (rejection != SyncChannelRejection::None) {
            reject(channel, rejection, statistics);
        } else if (statistics) {
            ++statistics->channelCandidatesAdmitted;
        }
        result.channels.push_back(std::move(channel));
    }
    return result;
}

void mlir::pto::protocol_sync::compareWithLegacyDemandOracle(
    const LegacySyncSnapshot& legacy, const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, ChannelAnalysisResult& channels)
{
    SmallVector<LegacyPhaseInfo, 32> legacyPhases = indexLegacyPhases(legacy);
    MemoryDependentAnalyzer analyzer;
    for (SyncChannel& channel : channels.getMutableChannels()) {
        const bool cannotCompare = !channel.isAdmitted() || channel.generation >= timelines.getTimelines().size();
        if (cannotCompare) {
            continue;
        }
        const SyncGenerationTimeline& timeline = timelines.getTimelines()[channel.generation];
        const SyncStorageFamily* family = schedule.findStorageFamily(timeline.family);
        const SyncPhase* producer = getOnlyPhase(schedule, stages, timeline.producers.front());
        const SyncPhase* consumer = getOnlyPhase(schedule, stages, timeline.consumers.front());
        const LegacyPhaseInfo* legacyProducer = producer ? findLegacyPhase(legacyPhases, *producer) : nullptr;
        const LegacyPhaseInfo* legacyConsumer = consumer ? findLegacyPhase(legacyPhases, *consumer) : nullptr;
        if (!family || !legacyProducer || !legacyConsumer) {
            continue;
        }
        SmallVector<const BaseMemInfo*> producerDefinitions =
            filterTimelineStorage(legacyProducer->definitions, *family, timeline);
        SmallVector<const BaseMemInfo*> consumerUses = filterTimelineStorage(legacyConsumer->uses, *family, timeline);
        channel.readyOracle = hasLegacyForwardOrder(*legacyProducer, *legacyConsumer) ?
                                  compareLegacyDependency(analyzer, consumerUses, producerDefinitions) :
                                  SyncDemandOracleStatus::Mismatch;
        if (channel.kind == SyncChannelKind::ReadyRelease) {
            channel.releaseOracle = hasLegacyLoopBackedge(*legacyProducer, *legacyConsumer) ?
                                        compareLegacyDependency(analyzer, producerDefinitions, consumerUses) :
                                        SyncDemandOracleStatus::Mismatch;
        } else {
            channel.releaseOracle = SyncDemandOracleStatus::NotApplicable;
        }
    }
}

StringRef mlir::pto::protocol_sync::stringifySyncChannelKind(SyncChannelKind kind)
{
    switch (kind) {
        case SyncChannelKind::OneShot:
            return "one-shot";
        case SyncChannelKind::ReadyRelease:
            return "ready-release";
    }
    return "one-shot";
}

StringRef mlir::pto::protocol_sync::stringifySyncChannelRejection(SyncChannelRejection reason)
{
    switch (reason) {
        case SyncChannelRejection::None:
            return "none";
        case SyncChannelRejection::TimelineRejected:
            return "timeline-rejected";
        case SyncChannelRejection::UnresolvedScheduleFailure:
            return "unresolved-schedule-failure";
        case SyncChannelRejection::MultipleProducers:
            return "multiple-producers";
        case SyncChannelRejection::MultipleConsumers:
            return "multiple-consumers";
        case SyncChannelRejection::UnsupportedControlFlow:
            return "unsupported-control-flow";
        case SyncChannelRejection::NestedLoop:
            return "nested-loop";
        case SyncChannelRejection::CrossCore:
            return "cross-core";
        case SyncChannelRejection::UnknownStageRole:
            return "unknown-stage-role";
        case SyncChannelRejection::StaticOverwrite:
            return "static-overwrite";
        case SyncChannelRejection::ReuseCapacityMismatch:
            return "reuse-capacity-mismatch";
    }
    return "timeline-rejected";
}

StringRef mlir::pto::protocol_sync::stringifySyncDemandOracleStatus(SyncDemandOracleStatus status)
{
    switch (status) {
        case SyncDemandOracleStatus::Match:
            return "match";
        case SyncDemandOracleStatus::Mismatch:
            return "mismatch";
        case SyncDemandOracleStatus::NotApplicable:
            return "not-applicable";
        case SyncDemandOracleStatus::Unavailable:
            return "unavailable";
    }
    return "unavailable";
}
