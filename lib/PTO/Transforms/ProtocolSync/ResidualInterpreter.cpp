// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ResidualInterpreter.cpp - Interpret selected synchronization -----===//

#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

struct GenerationState {
    SyncGenerationId current = kInvalidSyncId;
    llvm::SmallVector<SyncPhaseId, 2> availableTo;
    llvm::SmallVector<SyncPhaseId, 2> activeConsumers;
    bool produced = false;
    bool reclaimable = false;
};

struct TokenState {
    SyncChannelId channel = kInvalidSyncId;
    unsigned lane = 0;
    bool free = true;
    bool ready = false;
};

struct AbstractState {
    llvm::SmallVector<GenerationState, 8> generations;
    llvm::SmallVector<TokenState, 4> tokens;
};

class CompletionGraph {
public:
    explicit CompletionGraph(unsigned phaseCount)
    {
        sameIteration.reserve(phaseCount);
        directSameIteration.resize(phaseCount);
        for (unsigned phase = 0; phase < phaseCount; ++phase) {
            sameIteration.emplace_back(phaseCount);
            sameIteration.back().set(phase);
        }
    }

    LogicalResult add(const SyncSelectedCompletion& completion)
    {
        const bool invalidEndpoint =
            completion.source >= sameIteration.size() || completion.target >= sameIteration.size();
        const bool invalidRelation = completion.control != SyncControlRelation::MustExecute;
        if (invalidEndpoint || invalidRelation) {
            return failure();
        }
        if (completion.iteration.kind == SyncIterationRelationKind::SameIteration) {
            const bool invalidSameIteration = completion.source == completion.target ||
                                              completion.iteration.distance != 0 ||
                                              sameIteration[completion.source].test(completion.target);
            if (invalidSameIteration) {
                return failure();
            }
            sameIteration[completion.source].set(completion.target);
            directSameIteration[completion.source].push_back(completion.target);
            return success();
        }
        if (completion.iteration.kind != SyncIterationRelationKind::LoopCarried || completion.iteration.distance == 0 ||
            completion.iteration.carrier == kInvalidSyncId) {
            return failure();
        }
        const bool duplicate = llvm::any_of(loopCarried, [&](const SyncSelectedCompletion& current) {
            return current.source == completion.source && current.target == completion.target &&
                   current.iteration.distance == completion.iteration.distance &&
                   current.iteration.carrier == completion.iteration.carrier;
        });
        if (duplicate) {
            return failure();
        }
        loopCarried.push_back(completion);
        return success();
    }

    void finalize()
    {
        for (unsigned source = sameIteration.size(); source > 0; --source) {
            const unsigned phase = source - 1;
            for (SyncPhaseId target : directSameIteration[phase]) {
                sameIteration[phase] |= sameIteration[target];
            }
        }
    }

    bool covers(SyncPhaseId source, SyncPhaseId target, const SyncIterationRelation& relation) const
    {
        if (relation.kind == SyncIterationRelationKind::SameIteration && relation.distance == 0) {
            return reachesSameIteration(source, target);
        }
        if (relation.kind != SyncIterationRelationKind::LoopCarried || relation.distance == 0) {
            return false;
        }
        return llvm::any_of(loopCarried, [&](const SyncSelectedCompletion& edge) {
            return edge.iteration.distance == relation.distance && edge.iteration.carrier == relation.carrier &&
                   reachesSameIteration(source, edge.source) && reachesSameIteration(edge.target, target);
        });
    }

private:
    bool reachesSameIteration(SyncPhaseId source, SyncPhaseId target) const
    {
        const bool invalidEndpoint = source >= sameIteration.size() || target >= sameIteration.size();
        return !invalidEndpoint && sameIteration[source].test(target);
    }

    llvm::SmallVector<llvm::BitVector, 16> sameIteration;
    llvm::SmallVector<llvm::SmallVector<SyncPhaseId, 4>, 16> directSameIteration;
    llvm::SmallVector<SyncSelectedCompletion, 4> loopCarried;
};

bool checkedIncrement(std::uint64_t& value)
{
    if (value == std::numeric_limits<std::uint64_t>::max()) {
        return false;
    }
    ++value;
    return true;
}

const SyncPhase* phaseForStage(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, SyncStageId id)
{
    const SyncStage* stage = stages.findStage(id);
    return stage && stage->phases.size() == 1 ? schedule.findPhase(stage->phases.front()) : nullptr;
}

const SyncPhase* phaseForPoint(const StructuredSyncIR& schedule, SyncProgramPointId id)
{
    if (id >= schedule.getProgramPoints().size()) {
        return nullptr;
    }
    const SyncProgramPoint& point = schedule.getProgramPoints()[id];
    return point.id == id && point.phase != kInvalidSyncId ? schedule.findPhase(point.phase) : nullptr;
}

bool sameGuard(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return first.size() == second.size() && llvm::all_of(llvm::zip(first, second), [](const auto& pair) {
               const auto& [lhs, rhs] = pair;
               return lhs.choice == rhs.choice && lhs.arm == rhs.arm;
           });
}

std::optional<unsigned> guardArm(ArrayRef<SyncControlAtom> guard, SyncRegionId choice)
{
    for (const SyncControlAtom& atom : guard) {
        if (atom.choice == choice) {
            return atom.arm;
        }
    }
    return std::nullopt;
}

bool guardsCompatible(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    for (const SyncControlAtom& atom : first) {
        const std::optional<unsigned> other = guardArm(second, atom.choice);
        if (other && *other != atom.arm) {
            return false;
        }
    }
    return true;
}

bool guardImplies(ArrayRef<SyncControlAtom> premise, ArrayRef<SyncControlAtom> consequence)
{
    return llvm::all_of(consequence, [&](const SyncControlAtom& atom) {
        const std::optional<unsigned> selected = guardArm(premise, atom.choice);
        return selected && *selected == atom.arm;
    });
}

SyncControlRelation controlRelation(const SyncPhase* source, const SyncPhase* target)
{
    if (!source || !target || !sameGuard(source->guard, target->guard)) {
        return SyncControlRelation::Unknown;
    }
    return source->guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::SameGuard;
}

SyncRegionId innermostCommonLoop(ArrayRef<SyncRegionId> first, ArrayRef<SyncRegionId> second)
{
    const unsigned commonSize = std::min(first.size(), second.size());
    SyncRegionId carrier = kInvalidSyncId;
    for (unsigned index = 0; index < commonSize && first[index] == second[index]; ++index) {
        carrier = first[index];
    }
    return carrier;
}

SyncIterationRelation iterationRelation(const SyncPhase* source, const SyncPhase* target)
{
    if (!source || !target) {
        return {SyncIterationRelationKind::Unknown, 0};
    }
    if (source->iterationDomain.loops != target->iterationDomain.loops) {
        return {
            SyncIterationRelationKind::Unknown, 0,
            innermostCommonLoop(source->iterationDomain.loops, target->iterationDomain.loops)};
    }
    return {SyncIterationRelationKind::SameIteration, 0};
}

using ObligationKey = std::tuple<
    std::uint8_t, SyncPhaseId, SyncPhaseId, SyncGenerationId, SyncChannelId, std::uint8_t, std::uint8_t, unsigned,
    SyncRegionId>;

class ResidualAccumulator {
public:
    LogicalResult append(SyncResidualObligation obligation)
    {
        const ObligationKey key{
            static_cast<std::uint8_t>(obligation.kind),
            obligation.source,
            obligation.target,
            obligation.generation.value_or(kInvalidSyncId),
            obligation.channel.value_or(kInvalidSyncId),
            static_cast<std::uint8_t>(obligation.control),
            static_cast<std::uint8_t>(obligation.iteration.kind),
            obligation.iteration.distance,
            obligation.iteration.carrier};
        if (!seen.insert(key).second) {
            return success();
        }
        const bool idSpaceExhausted = result.obligations.size() >= kInvalidSyncId;
        if (idSpaceExhausted) {
            return failure();
        }
        obligation.id = static_cast<SyncObligationId>(result.obligations.size());
        result.obligations.push_back(std::move(obligation));
        return success();
    }

    SyncInterpretationResult result;

private:
    std::set<ObligationKey> seen;
};

SyncObligationKind rejectionKind(SyncTimelineRejection rejection)
{
    if (rejection == SyncTimelineRejection::OrderedAccess) {
        return SyncObligationKind::OrderedMemory;
    }
    if (rejection == SyncTimelineRejection::InPlaceAccess) {
        return SyncObligationKind::AccConflict;
    }
    return SyncObligationKind::UnknownAlias;
}

const SyncPhase* firstTimelinePhase(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    bool producer)
{
    ArrayRef<SyncStageId> candidates =
        producer ? ArrayRef<SyncStageId>(timeline.producers) : ArrayRef<SyncStageId>(timeline.consumers);
    for (SyncStageId stage : candidates) {
        if (const SyncPhase* phase = phaseForStage(schedule, stages, stage)) {
            return phase;
        }
    }
    return nullptr;
}

bool hasProducerRole(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Write || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

bool hasConsumerRole(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Read || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

const SyncPhase* firstTimelineAccessPhase(
    const StructuredSyncIR& schedule, const SyncGenerationTimeline& timeline, bool producer,
    const SyncPhase* excluded = nullptr)
{
    for (SyncAccessId id : timeline.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        const SyncPhase* phase = access ? schedule.findPhase(access->phase) : nullptr;
        const bool matchingRole = access && (producer ? hasProducerRole(access->mode) : hasConsumerRole(access->mode));
        if (phase && phase != excluded && matchingRole) {
            return phase;
        }
    }
    for (SyncAccessId id : timeline.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        const SyncPhase* phase = access ? schedule.findPhase(access->phase) : nullptr;
        if (phase && phase != excluded) {
            return phase;
        }
    }
    return nullptr;
}

LogicalResult appendRejectedTimeline(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    ResidualAccumulator& accumulator)
{
    const SyncPhase* source = firstTimelinePhase(schedule, stages, timeline, true);
    if (!source) {
        source = firstTimelineAccessPhase(schedule, timeline, true);
    }
    const SyncPhase* target = firstTimelinePhase(schedule, stages, timeline, false);
    if (!target) {
        target = firstTimelineAccessPhase(schedule, timeline, false, source);
    }
    if (!target) {
        target = source;
    }
    SyncResidualObligation obligation;
    obligation.kind = rejectionKind(timeline.rejection);
    obligation.source = source ? source->id : kInvalidSyncId;
    obligation.target = target ? target->id : kInvalidSyncId;
    obligation.generation = timeline.id;
    obligation.control = controlRelation(source, target);
    obligation.iteration = iterationRelation(source, target);
    obligation.detail = ("timeline rejected: " + stringifySyncTimelineRejection(timeline.rejection)).str();
    return accumulator.append(std::move(obligation));
}

const SyncSelectedProtocol* findSelectedProtocol(const SyncSelectedWorld& world, SyncChannelId channel)
{
    for (const SyncSelectedProtocol& selected : world.protocols) {
        if (selected.channel == channel) {
            return &selected;
        }
    }
    return nullptr;
}

LogicalResult validateSelectedProtocol(const SyncSelectedProtocol& selected, const SyncChannel& channel)
{
    const SyncSelectedProtocolKind expected = channel.kind == SyncChannelKind::ReadyRelease ?
                                                  SyncSelectedProtocolKind::ReadyRelease :
                                                  SyncSelectedProtocolKind::OneShotPublish;
    return channel.isAdmitted() && selected.kind == expected && selected.channel == channel.id &&
                   selected.generation == channel.generation && selected.capacity == channel.capacity ?
               success() :
               failure();
}

LogicalResult validateSelectedCompletion(const StructuredSyncIR& schedule, const SyncSelectedCompletion& completion)
{
    const SyncPhase* source = schedule.findPhase(completion.source);
    const SyncPhase* target = schedule.findPhase(completion.target);
    const bool invalidEndpoint = !source || !target;
    const bool invalidControl = !invalidEndpoint && controlRelation(source, target) != completion.control;
    if (invalidEndpoint || invalidControl) {
        return failure();
    }
    if (completion.iteration.kind == SyncIterationRelationKind::SameIteration) {
        const bool forward = source->id < target->id;
        const bool sameIteration = source->iterationDomain.loops == target->iterationDomain.loops;
        const bool validRelation = completion.iteration.distance == 0 && completion.iteration.carrier == kInvalidSyncId;
        return forward && sameIteration && validRelation ? success() : failure();
    }
    const bool carrierContainsSource = llvm::is_contained(source->iterationDomain.loops, completion.iteration.carrier);
    const bool carrierContainsTarget = llvm::is_contained(target->iterationDomain.loops, completion.iteration.carrier);
    const bool recurring = completion.iteration.kind == SyncIterationRelationKind::LoopCarried &&
                           completion.iteration.distance > 0 && !source->iterationDomain.loops.empty() &&
                           completion.iteration.carrier != kInvalidSyncId && carrierContainsSource &&
                           carrierContainsTarget;
    return recurring ? success() : failure();
}

LogicalResult validateWorld(
    const StructuredSyncIR& schedule, const ChannelAnalysisResult& channels, const SyncSelectedWorld& world,
    CompletionGraph& completionGraph, CompletionGraph& visibilityGraph)
{
    llvm::BitVector selectedChannels(channels.getChannels().size());
    for (const SyncSelectedProtocol& selected : world.protocols) {
        const bool invalidChannel =
            selected.channel >= channels.getChannels().size() || selectedChannels.test(selected.channel);
        if (invalidChannel) {
            return failure();
        }
        const SyncChannel& channel = channels.getChannels()[selected.channel];
        if (channel.id != selected.channel || failed(validateSelectedProtocol(selected, channel))) {
            return failure();
        }
        selectedChannels.set(selected.channel);
    }
    for (const SyncSelectedCompletion& completion : world.completions) {
        const bool invalidCompletion =
            failed(validateSelectedCompletion(schedule, completion)) || failed(completionGraph.add(completion));
        if (invalidCompletion) {
            return failure();
        }
    }
    for (const SyncSelectedVisibility& visibility : world.visibility) {
        const SyncSelectedCompletion effect{
            visibility.source, visibility.target, visibility.control, visibility.iteration};
        const bool invalidVisibility =
            failed(validateSelectedCompletion(schedule, effect)) || failed(visibilityGraph.add(effect));
        if (invalidVisibility) {
            return failure();
        }
    }
    llvm::BitVector exitPhases(schedule.getPhases().size());
    for (SyncPhaseId phase : world.exitCompletedPhases) {
        const bool invalidExit =
            phase >= schedule.getPhases().size() || exitPhases.test(phase) || !schedule.findPhase(phase);
        if (invalidExit) {
            return failure();
        }
        exitPhases.set(phase);
    }
    completionGraph.finalize();
    visibilityGraph.finalize();
    return success();
}

LogicalResult appendCompletionObligation(
    const SyncGenerationTimeline& timeline, const SyncChannel& channel, const SyncPhase& producer,
    const SyncPhase& consumer, ResidualAccumulator& accumulator)
{
    SyncResidualObligation obligation;
    obligation.kind = SyncObligationKind::Completion;
    obligation.source = producer.id;
    obligation.target = consumer.id;
    obligation.generation = timeline.id;
    obligation.channel = channel.id;
    obligation.control = controlRelation(&producer, &consumer);
    obligation.iteration = iterationRelation(&producer, &consumer);
    obligation.detail = "consumer cannot observe the selected storage generation";
    return accumulator.append(std::move(obligation));
}

LogicalResult evaluateAvailability(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    const SyncChannel& channel, const CompletionGraph& graph, GenerationState& state, ResidualAccumulator& accumulator)
{
    SyncInterpretationResult& result = accumulator.result;
    for (SyncStageId producerStage : timeline.producers) {
        const SyncPhase* producer = phaseForStage(schedule, stages, producerStage);
        if (!producer) {
            return failure();
        }
        state.produced = true;
        for (SyncStageId consumerStage : timeline.consumers) {
            const SyncPhase* consumer = phaseForStage(schedule, stages, consumerStage);
            if (!consumer) {
                return failure();
            }
            const SyncIterationRelation relation = iterationRelation(producer, consumer);
            if (graph.covers(producer->id, consumer->id, relation)) {
                state.availableTo.push_back(consumer->id);
            } else if (failed(appendCompletionObligation(timeline, channel, *producer, *consumer, accumulator))) {
                return failure();
            }
            state.activeConsumers.push_back(consumer->id);
            if (!checkedIncrement(result.transitions)) {
                return failure();
            }
        }
    }
    return success();
}

LogicalResult appendReclamationObligation(
    const SyncGenerationTimeline& timeline, const SyncChannel& channel, const SyncPhase* source,
    const SyncPhase* target, const SyncIterationRelation& relation, ResidualAccumulator& accumulator)
{
    SyncResidualObligation obligation;
    obligation.kind = SyncObligationKind::Reclamation;
    obligation.source = source ? source->id : kInvalidSyncId;
    obligation.target = target ? target->id : kInvalidSyncId;
    obligation.generation = timeline.id;
    obligation.channel = channel.id;
    obligation.control = controlRelation(source, target);
    obligation.iteration = relation;
    obligation.detail = "the physical slot is not reclaimed before its next overwrite";
    return accumulator.append(std::move(obligation));
}

LogicalResult evaluateReclamation(
    const StructuredSyncIR& schedule, const SyncGenerationTimeline& timeline, const SyncChannel& channel,
    const CompletionGraph& graph, GenerationState& state, ResidualAccumulator& accumulator)
{
    SyncInterpretationResult& result = accumulator.result;
    if (!timeline.nextOverwrite) {
        state.reclaimable = true;
        return success();
    }
    const unsigned distance = timeline.nextOverwrite->iterationDistance;
    const SyncIterationRelation relation =
        distance == 0 ?
            SyncIterationRelation{SyncIterationRelationKind::SameIteration, 0} :
            SyncIterationRelation{SyncIterationRelationKind::LoopCarried, distance, timeline.carryingRegion};
    bool complete = true;
    for (const SyncProgramFrontier& finalUse : timeline.finalUses) {
        for (const SyncGuardedProgramPoint& sourcePoint : finalUse.points) {
            const SyncPhase* source = phaseForPoint(schedule, sourcePoint.point);
            for (const SyncGuardedProgramPoint& targetPoint : timeline.nextOverwrite->frontier.points) {
                const SyncPhase* target = phaseForPoint(schedule, targetPoint.point);
                if (!source || !target || !graph.covers(source->id, target->id, relation)) {
                    complete = false;
                    if (failed(appendReclamationObligation(timeline, channel, source, target, relation, accumulator))) {
                        return failure();
                    }
                }
                if (!checkedIncrement(result.transitions)) {
                    return failure();
                }
            }
        }
    }
    state.reclaimable = complete;
    return success();
}

LogicalResult evaluateReadyReleaseTokens(
    const SyncSelectedProtocol& selected, AbstractState& state, ResidualAccumulator& accumulator)
{
    SyncInterpretationResult& result = accumulator.result;
    for (unsigned lane = 0; lane < selected.capacity; ++lane) {
        TokenState token{selected.channel, lane, true, false};
        token.free = false;
        token.ready = true;
        const bool invalidReadyTransfer = !checkedIncrement(result.transitions) || token.free || !token.ready;
        if (invalidReadyTransfer) {
            return failure();
        }
        token.ready = false;
        token.free = true;
        const bool invalidReleaseTransfer = !checkedIncrement(result.transitions) || !token.free || token.ready;
        if (invalidReleaseTransfer) {
            return failure();
        }
        state.tokens.push_back(token);
    }
    return success();
}

LogicalResult evaluateTimeline(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages, const SyncGenerationTimeline& timeline,
    const SyncChannel& channel, const SyncSelectedWorld& world, const CompletionGraph& graph, AbstractState& state,
    ResidualAccumulator& accumulator)
{
    SyncInterpretationResult& result = accumulator.result;
    if (!timeline.isAdmitted()) {
        return appendRejectedTimeline(schedule, stages, timeline, accumulator);
    }
    GenerationState generation;
    generation.current = timeline.id;
    const bool invalidEvaluation =
        !checkedIncrement(result.transitions) ||
        failed(evaluateAvailability(schedule, stages, timeline, channel, graph, generation, accumulator)) ||
        failed(evaluateReclamation(schedule, timeline, channel, graph, generation, accumulator));
    if (invalidEvaluation) {
        return failure();
    }
    const SyncSelectedProtocol* selected = findSelectedProtocol(world, channel.id);
    const bool selectedIncomplete = selected && (generation.availableTo.size() != timeline.consumers.size() ||
                                                 (timeline.nextOverwrite && !generation.reclaimable));
    if (selectedIncomplete) {
        return failure();
    }
    if (selected && selected->kind == SyncSelectedProtocolKind::ReadyRelease &&
        failed(evaluateReadyReleaseTokens(*selected, state, accumulator))) {
        return failure();
    }
    state.generations.push_back(std::move(generation));
    return success();
}

enum class AliasRelation : std::uint8_t { NoAlias, MayAlias, Unknown };

bool intervalsOverlap(ArrayRef<SyncByteInterval> first, ArrayRef<SyncByteInterval> second)
{
    for (const SyncByteInterval& lhs : first) {
        const bool invalidLeft = lhs.size == 0 || lhs.begin > std::numeric_limits<std::uint64_t>::max() - lhs.size;
        if (invalidLeft) {
            return true;
        }
        const std::uint64_t lhsEnd = lhs.begin + lhs.size;
        for (const SyncByteInterval& rhs : second) {
            const bool invalidRight = rhs.size == 0 || rhs.begin > std::numeric_limits<std::uint64_t>::max() - rhs.size;
            if (invalidRight) {
                return true;
            }
            const std::uint64_t rhsEnd = rhs.begin + rhs.size;
            if (lhs.begin < rhsEnd && rhs.begin < lhsEnd) {
                return true;
            }
        }
    }
    return false;
}

AliasRelation aliasRelation(const SyncAccess& first, const SyncAccess& second)
{
    if (first.storage.space != second.storage.space) {
        return AliasRelation::NoAlias;
    }
    if (first.family == second.family) {
        return AliasRelation::MayAlias;
    }
    if (first.storage.unknownRange || second.storage.unknownRange || first.storage.aliasesUnknownRange ||
        second.storage.aliasesUnknownRange) {
        return AliasRelation::Unknown;
    }
    if (!first.storage.physical || !second.storage.physical) {
        return AliasRelation::NoAlias;
    }
    return intervalsOverlap(first.storage.intervals, second.storage.intervals) ? AliasRelation::MayAlias :
                                                                                 AliasRelation::NoAlias;
}

bool writesMemory(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Write || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

bool readsMemory(SyncAccessMode mode)
{
    return mode == SyncAccessMode::Read || mode == SyncAccessMode::ReadWrite || mode == SyncAccessMode::Ordered;
}

LogicalResult appendMemoryObligation(
    const SyncAccess& sourceAccess, const SyncAccess& targetAccess, const SyncPhase& source, const SyncPhase& target,
    AliasRelation alias, const SyncIterationRelation& relation, ResidualAccumulator& accumulator)
{
    SyncResidualObligation obligation;
    if (alias == AliasRelation::Unknown) {
        obligation.kind = SyncObligationKind::UnknownAlias;
    } else if (sourceAccess.mode == SyncAccessMode::Ordered || targetAccess.mode == SyncAccessMode::Ordered) {
        obligation.kind = SyncObligationKind::OrderedMemory;
    } else {
        const bool publishesToReader = writesMemory(sourceAccess.mode) && readsMemory(targetAccess.mode);
        obligation.kind = publishesToReader ? SyncObligationKind::Visibility : SyncObligationKind::OrderedMemory;
    }
    obligation.source = source.id;
    obligation.target = target.id;
    obligation.control = controlRelation(&source, &target);
    obligation.iteration = relation;
    obligation.detail = alias == AliasRelation::Unknown ?
                            "global or unknown storage may alias without a selected ordering proof" :
                            "ordered memory or visibility is not established by the selected world";
    return accumulator.append(std::move(obligation));
}

bool memoryHazardCovered(
    const SyncAccess& source, const SyncAccess& target, AliasRelation alias, const CompletionGraph& completionGraph,
    const CompletionGraph& visibilityGraph, SyncPhaseId sourcePhase, SyncPhaseId targetPhase,
    const SyncIterationRelation& relation, const StructuredSyncIR& schedule)
{
    const bool hasOrderedEndpoint = source.mode == SyncAccessMode::Ordered || target.mode == SyncAccessMode::Ordered;
    if (alias == AliasRelation::Unknown || hasOrderedEndpoint) {
        return false;
    }
    const bool publishesToReader = writesMemory(source.mode) && readsMemory(target.mode);
    if (publishesToReader) {
        return visibilityGraph.covers(sourcePhase, targetPhase, relation);
    }
    const SyncPhase* sourcePhaseRecord = schedule.findPhase(sourcePhase);
    const SyncPhase* targetPhaseRecord = schedule.findPhase(targetPhase);
    const bool intrinsicallyOrdered = sourcePhaseRecord && targetPhaseRecord &&
                                      sourcePhaseRecord->core == targetPhaseRecord->core &&
                                      sourcePhaseRecord->pipe == targetPhaseRecord->pipe;
    if (intrinsicallyOrdered) {
        return true;
    }
    return completionGraph.covers(sourcePhase, targetPhase, relation);
}

LogicalResult evaluateMemoryPair(
    const StructuredSyncIR& schedule, const SyncAccess& sourceAccess, const SyncAccess& targetAccess,
    const CompletionGraph& completionGraph, const CompletionGraph& visibilityGraph,
    const SyncIterationRelation& relation, ResidualAccumulator& accumulator)
{
    const SyncPhase* source = schedule.findPhase(sourceAccess.phase);
    const SyncPhase* target = schedule.findPhase(targetAccess.phase);
    if (!source || !target) {
        return failure();
    }
    const AliasRelation alias = aliasRelation(sourceAccess, targetAccess);
    if (alias == AliasRelation::NoAlias) {
        return success();
    }
    const bool covered = memoryHazardCovered(
        sourceAccess, targetAccess, alias, completionGraph, visibilityGraph, source->id, target->id, relation,
        schedule);
    if (!covered &&
        failed(appendMemoryObligation(sourceAccess, targetAccess, *source, *target, alias, relation, accumulator))) {
        return failure();
    }
    return checkedIncrement(accumulator.result.transitions) ? success() : failure();
}

llvm::SmallVector<SyncIterationRelation, 2> carriedMemoryRelations(const SyncPhase& source, const SyncPhase& target)
{
    llvm::SmallVector<SyncIterationRelation, 2> relations;
    const unsigned commonSize = std::min(source.iterationDomain.loops.size(), target.iterationDomain.loops.size());
    for (unsigned index = 0;
         index < commonSize && source.iterationDomain.loops[index] == target.iterationDomain.loops[index]; ++index) {
        const SyncRegionId carrier = source.iterationDomain.loops[index];
        const bool exactCarrier =
            source.iterationDomain.loops.back() == carrier && target.iterationDomain.loops.back() == carrier;
        const bool mustExecute = exactCarrier && source.guard.empty() && target.guard.empty();
        relations.push_back(
            mustExecute ? SyncIterationRelation{SyncIterationRelationKind::LoopCarried, 1, carrier} :
                          SyncIterationRelation{SyncIterationRelationKind::Unknown, 0, carrier});
    }
    return relations;
}

LogicalResult evaluateMemoryHazards(
    const StructuredSyncIR& schedule, const CompletionGraph& completionGraph, const CompletionGraph& visibilityGraph,
    ResidualAccumulator& accumulator)
{
    for (auto [sourceIndex, sourceAccess] : llvm::enumerate(schedule.getAccesses())) {
        if (sourceAccess.visibility == SyncVisibilityClass::Local) {
            continue;
        }
        for (const SyncAccess& targetAccess : schedule.getAccesses().drop_front(sourceIndex + 1)) {
            if (targetAccess.visibility == SyncVisibilityClass::Local ||
                (!writesMemory(sourceAccess.mode) && !writesMemory(targetAccess.mode))) {
                continue;
            }
            const SyncPhase* source = schedule.findPhase(sourceAccess.phase);
            const SyncPhase* target = schedule.findPhase(targetAccess.phase);
            if (!source || !target) {
                return failure();
            }
            const bool incompatible = !guardsCompatible(source->guard, target->guard);
            if (source->id == target->id || source->id > target->id || incompatible) {
                continue;
            }
            const SyncIterationRelation relation = iterationRelation(source, target);
            if (failed(evaluateMemoryPair(
                    schedule, sourceAccess, targetAccess, completionGraph, visibilityGraph, relation, accumulator))) {
                return failure();
            }
        }
    }
    for (const SyncAccess& sourceAccess : schedule.getAccesses()) {
        if (sourceAccess.visibility == SyncVisibilityClass::Local) {
            continue;
        }
        const SyncPhase* source = schedule.findPhase(sourceAccess.phase);
        if (!source || source->iterationDomain.loops.empty()) {
            continue;
        }
        for (const SyncAccess& targetAccess : schedule.getAccesses()) {
            const bool noWrite = !writesMemory(sourceAccess.mode) && !writesMemory(targetAccess.mode);
            if (targetAccess.visibility == SyncVisibilityClass::Local || noWrite) {
                continue;
            }
            const SyncPhase* target = schedule.findPhase(targetAccess.phase);
            if (!target) {
                continue;
            }
            for (const SyncIterationRelation& relation : carriedMemoryRelations(*source, *target)) {
                const bool sourceDirectlyInCarrier = source->iterationDomain.loops.back() == relation.carrier;
                const bool killedInTargetIteration =
                    source->id < target->id && sourceDirectlyInCarrier && guardImplies(target->guard, source->guard);
                if (killedInTargetIteration) {
                    continue;
                }
                if (failed(evaluateMemoryPair(
                        schedule, sourceAccess, targetAccess, completionGraph, visibilityGraph, relation,
                        accumulator))) {
                    return failure();
                }
            }
        }
    }
    return success();
}

enum class SSATraceKind : std::uint8_t {
    Normal,
    LoopBackedge,
    LoopExit,
};

struct SSATraceItem {
    Value value;
    SSATraceKind kind = SSATraceKind::Normal;
    SyncRegionId carrier = kInvalidSyncId;
};

SyncRegionId findLoopRegion(const StructuredSyncIR& schedule, Operation* operation)
{
    for (const SyncRegion& region : schedule.getRegions()) {
        if (region.kind == SyncRegionKind::Loop && region.operation == operation) {
            return region.id;
        }
    }
    return kInvalidSyncId;
}

SyncIterationRelation ssaIterationRelation(
    const SyncPhase& source, const SyncPhase& target, SSATraceKind kind, SyncRegionId carrier)
{
    if (kind == SSATraceKind::Normal) {
        if (source.id < target.id) {
            return iterationRelation(&source, &target);
        }
        return {
            SyncIterationRelationKind::Unknown, 0,
            innermostCommonLoop(source.iterationDomain.loops, target.iterationDomain.loops)};
    }
    if (kind == SSATraceKind::LoopExit) {
        return {SyncIterationRelationKind::Unknown, 0, carrier};
    }
    const bool sourceInCarrier = llvm::is_contained(source.iterationDomain.loops, carrier);
    const bool targetInCarrier = llvm::is_contained(target.iterationDomain.loops, carrier);
    const bool exactCarrier = sourceInCarrier && targetInCarrier && !source.iterationDomain.loops.empty() &&
                              !target.iterationDomain.loops.empty() && source.iterationDomain.loops.back() == carrier &&
                              target.iterationDomain.loops.back() == carrier;
    const bool mustExecute = exactCarrier && source.guard.empty() && target.guard.empty();
    return mustExecute ? SyncIterationRelation{SyncIterationRelationKind::LoopCarried, 1, carrier} :
                         SyncIterationRelation{SyncIterationRelationKind::Unknown, 0, carrier};
}

LogicalResult appendSSACompletion(
    const SyncPhase& source, const SyncPhase& target, const CompletionGraph& graph, SSATraceKind kind,
    SyncRegionId carrier, ResidualAccumulator& accumulator)
{
    if (source.id == target.id) {
        return success();
    }
    const SyncIterationRelation relation = ssaIterationRelation(source, target, kind, carrier);
    if (!graph.covers(source.id, target.id, relation)) {
        SyncResidualObligation obligation;
        obligation.kind = SyncObligationKind::SSACompletion;
        obligation.source = source.id;
        obligation.target = target.id;
        obligation.control = controlRelation(&source, &target);
        obligation.iteration = relation;
        obligation.detail = "a physical SSA producer is not completed before its consumer";
        if (failed(accumulator.append(std::move(obligation)))) {
            return failure();
        }
    }
    return checkedIncrement(accumulator.result.transitions) ? success() : failure();
}

LogicalResult traceSSAProducers(
    const StructuredSyncIR& schedule, const llvm::DenseMap<Operation*, const SyncPhase*>& terminalPhaseByOperation,
    const SyncPhase& target, Value root, const CompletionGraph& graph, ResidualAccumulator& accumulator)
{
    llvm::SmallVector<SSATraceItem, 16> worklist{{root, SSATraceKind::Normal, kInvalidSyncId}};
    llvm::DenseSet<std::pair<Value, std::uint64_t>> visited;
    while (!worklist.empty()) {
        const SSATraceItem item = worklist.pop_back_val();
        const std::uint64_t encodedRelation =
            (static_cast<std::uint64_t>(item.carrier) << 8) | static_cast<std::uint8_t>(item.kind);
        if (!item.value || !visited.insert({item.value, encodedRelation}).second) {
            continue;
        }
        if (auto argument = dyn_cast<BlockArgument>(item.value)) {
            auto loop = dyn_cast_or_null<scf::ForOp>(argument.getParentRegion()->getParentOp());
            const bool isIterationArgument = loop && argument.getArgNumber() > 0;
            if (!isIterationArgument) {
                continue;
            }
            const unsigned index = argument.getArgNumber() - 1;
            const SyncRegionId carrier = findLoopRegion(schedule, loop);
            const bool invalidLoop = index >= loop.getInitArgs().size() || index >= loop.getYieldedValues().size() ||
                                     carrier == kInvalidSyncId;
            if (invalidLoop) {
                return failure();
            }
            worklist.push_back({loop.getInitArgs()[index], item.kind, item.carrier});
            worklist.push_back({loop.getYieldedValues()[index], SSATraceKind::LoopBackedge, carrier});
            continue;
        }

        Operation* definingOperation = item.value.getDefiningOp();
        if (!definingOperation) {
            continue;
        }
        auto source = terminalPhaseByOperation.find(definingOperation);
        if (source != terminalPhaseByOperation.end()) {
            if (failed(appendSSACompletion(*source->second, target, graph, item.kind, item.carrier, accumulator))) {
                return failure();
            }
            continue;
        }

        auto result = dyn_cast<OpResult>(item.value);
        if (auto choice = dyn_cast<scf::IfOp>(definingOperation)) {
            const bool invalidChoice = !result || !choice.elseBlock();
            if (invalidChoice) {
                return failure();
            }
            const unsigned index = result.getResultNumber();
            auto thenYield = dyn_cast<scf::YieldOp>(choice.thenBlock()->getTerminator());
            auto elseYield = dyn_cast<scf::YieldOp>(choice.elseBlock()->getTerminator());
            const bool invalidYields =
                !thenYield || !elseYield || index >= thenYield.getNumOperands() || index >= elseYield.getNumOperands();
            if (invalidYields) {
                return failure();
            }
            worklist.push_back({thenYield.getOperand(index), item.kind, item.carrier});
            worklist.push_back({elseYield.getOperand(index), item.kind, item.carrier});
            continue;
        }
        if (auto loop = dyn_cast<scf::ForOp>(definingOperation)) {
            const unsigned index = result ? result.getResultNumber() : loop.getNumResults();
            const SyncRegionId carrier = findLoopRegion(schedule, loop);
            const bool invalidLoop = index >= loop.getInitArgs().size() || index >= loop.getYieldedValues().size() ||
                                     carrier == kInvalidSyncId;
            if (invalidLoop) {
                return failure();
            }
            worklist.push_back({loop.getInitArgs()[index], item.kind, item.carrier});
            worklist.push_back({loop.getYieldedValues()[index], SSATraceKind::LoopExit, carrier});
            continue;
        }
        if (!isMemoryEffectFree(definingOperation)) {
            continue;
        }
        for (Value operand : definingOperation->getOperands()) {
            worklist.push_back({operand, item.kind, item.carrier});
        }
    }
    return success();
}

LogicalResult evaluateSSACompletions(
    const StructuredSyncIR& schedule, const CompletionGraph& graph, ResidualAccumulator& accumulator)
{
    llvm::DenseMap<Operation*, const SyncPhase*> terminalPhaseByOperation;
    for (const SyncPhase& phase : schedule.getPhases()) {
        if (phase.operation) {
            terminalPhaseByOperation[phase.operation] = &phase;
        }
    }
    for (const SyncPhase& target : schedule.getPhases()) {
        if (!target.operation) {
            return failure();
        }
        for (Value operand : target.operation->getOperands()) {
            if (failed(traceSSAProducers(schedule, terminalPhaseByOperation, target, operand, graph, accumulator))) {
                return failure();
            }
        }
    }
    return success();
}

std::pair<const SyncPhase*, const SyncPhase*> findNeighborPhases(const StructuredSyncIR& schedule, Operation* operation)
{
    const SyncPhase* before = nullptr;
    const SyncPhase* after = nullptr;
    if (!operation) {
        return {before, after};
    }
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool differentBlock = !phase.operation || phase.operation->getBlock() != operation->getBlock();
        if (differentBlock) {
            continue;
        }
        if (phase.operation->isBeforeInBlock(operation)) {
            before = &phase;
        } else if (operation->isBeforeInBlock(phase.operation)) {
            after = &phase;
            break;
        }
    }
    return {before, after};
}

LogicalResult evaluateOpaqueEffects(const StructuredSyncIR& schedule, ResidualAccumulator& accumulator)
{
    for (const SyncSemanticAction& action : schedule.getSemanticActions()) {
        const auto [source, target] = findNeighborPhases(schedule, action.operation);
        SyncResidualObligation obligation;
        obligation.kind = SyncObligationKind::OrderedMemory;
        obligation.source = source ? source->id : kInvalidSyncId;
        obligation.target = target ? target->id : kInvalidSyncId;
        obligation.control = controlRelation(source, target);
        obligation.iteration = iterationRelation(source, target);
        obligation.detail = "ordered semantic action is not modeled by a selected protocol";
        if (failed(accumulator.append(std::move(obligation)))) {
            return failure();
        }
    }
    for (const SyncFailure& scheduleFailure : schedule.getFailures()) {
        const auto [source, target] = findNeighborPhases(schedule, scheduleFailure.operation);
        SyncResidualObligation obligation;
        obligation.kind = SyncObligationKind::UnknownAlias;
        obligation.source = source ? source->id : kInvalidSyncId;
        obligation.target = target ? target->id : kInvalidSyncId;
        obligation.control = controlRelation(source, target);
        obligation.iteration = iterationRelation(source, target);
        obligation.detail = ("schedule failure: " + stringifySyncFailureReason(scheduleFailure.reason)).str();
        if (failed(accumulator.append(std::move(obligation)))) {
            return failure();
        }
    }
    return success();
}

std::uint32_t resourceKey(const SyncPhase& phase)
{
    return (static_cast<std::uint32_t>(phase.core) << 16) | static_cast<std::uint32_t>(phase.pipe);
}

bool loopDomainIsPrefix(ArrayRef<SyncRegionId> prefix, ArrayRef<SyncRegionId> domain)
{
    return prefix.size() <= domain.size() && std::equal(prefix.begin(), prefix.end(), domain.begin());
}

bool guardIsSatisfied(ArrayRef<SyncControlAtom> guard, ArrayRef<SyncControlAtom> assignment)
{
    return guardImplies(assignment, guard);
}

unsigned choiceArmCount(const StructuredSyncIR& schedule, SyncRegionId choice)
{
    const SyncRegion* region = schedule.findRegion(choice);
    if (!region || region->kind != SyncRegionKind::Choice) {
        return 0;
    }
    unsigned count = 0;
    for (const SyncRegionElement& element : region->elements) {
        const SyncRegion* child =
            element.kind == SyncRegionElement::Kind::ChildRegion ? schedule.findRegion(element.child) : nullptr;
        if (child && child->kind == SyncRegionKind::Alternative) {
            count = std::max(count, child->arm + 1);
        }
    }
    return count;
}

bool allGuardPathsCovered(
    const StructuredSyncIR& schedule, ArrayRef<const SyncPhase*> candidates, ArrayRef<SyncControlAtom> assignment)
{
    for (const SyncPhase* candidate : candidates) {
        if (candidate && guardIsSatisfied(candidate->guard, assignment)) {
            return true;
        }
    }
    SyncRegionId nextChoice = kInvalidSyncId;
    for (const SyncPhase* candidate : candidates) {
        if (!candidate || !guardsCompatible(candidate->guard, assignment)) {
            continue;
        }
        for (const SyncControlAtom& atom : candidate->guard) {
            const bool unassignedChoice = !guardArm(assignment, atom.choice);
            const bool earlierChoice = nextChoice == kInvalidSyncId || atom.choice < nextChoice;
            if (unassignedChoice && earlierChoice) {
                nextChoice = atom.choice;
            }
        }
    }
    if (nextChoice == kInvalidSyncId) {
        return false;
    }
    const unsigned arms = choiceArmCount(schedule, nextChoice);
    if (arms == 0) {
        return false;
    }
    for (unsigned arm = 0; arm < arms; ++arm) {
        llvm::SmallVector<SyncControlAtom, 4> nested(assignment.begin(), assignment.end());
        nested.push_back({nextChoice, arm});
        if (!allGuardPathsCovered(schedule, candidates, nested)) {
            return false;
        }
    }
    return true;
}

bool isTerminalPhase(const StructuredSyncIR& schedule, const SyncPhase& source)
{
    llvm::SmallVector<const SyncPhase*, 8> later;
    for (const SyncPhase& target : schedule.getPhases()) {
        const bool eligible = target.id > source.id && resourceKey(target) == resourceKey(source) &&
                              loopDomainIsPrefix(target.iterationDomain.loops, source.iterationDomain.loops) &&
                              guardsCompatible(source.guard, target.guard);
        if (eligible) {
            later.push_back(&target);
        }
    }
    return !allGuardPathsCovered(schedule, later, source.guard);
}

bool exitIsCovered(const SyncSelectedWorld& world, const CompletionGraph& graph, SyncPhaseId phase)
{
    return llvm::any_of(world.exitCompletedPhases, [&](SyncPhaseId completed) {
        return phase == completed || graph.covers(phase, completed, {SyncIterationRelationKind::SameIteration, 0});
    });
}

LogicalResult evaluateExitCompletion(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& world, const CompletionGraph& graph,
    ResidualAccumulator& accumulator)
{
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool skip = !isTerminalPhase(schedule, phase) || exitIsCovered(world, graph, phase.id);
        if (skip) {
            continue;
        }
        SyncResidualObligation obligation;
        obligation.kind = SyncObligationKind::ExitCompletion;
        obligation.source = phase.id;
        obligation.target = phase.id;
        obligation.control = phase.guard.empty() ? SyncControlRelation::MustExecute : SyncControlRelation::SameGuard;
        obligation.iteration = phase.iterationDomain.loops.empty() ?
                                   SyncIterationRelation{SyncIterationRelationKind::SameIteration, 0} :
                                   SyncIterationRelation{SyncIterationRelationKind::Unknown, 0};
        obligation.detail = "terminal physical work is not completed before function exit";
        if (failed(accumulator.append(std::move(obligation)))) {
            return failure();
        }
    }
    return success();
}

void addSaturating(std::uint64_t& target, std::uint64_t value)
{
    const bool wouldOverflow = value > std::numeric_limits<std::uint64_t>::max() - target;
    if (wouldOverflow) {
        target = std::numeric_limits<std::uint64_t>::max();
    } else {
        target += value;
    }
}

void updateStatistics(const SyncInterpretationResult& result, ProtocolSyncStatistics* statistics)
{
    if (!statistics) {
        return;
    }
    addSaturating(statistics->interpreterTransitions, result.transitions);
    statistics->interpreterPeakStates = std::max(statistics->interpreterPeakStates, result.peakStates);
    statistics->residualObligations = result.obligations.size();
    for (const SyncResidualObligation& obligation : result.obligations) {
        std::uint64_t& count =
            statistics->residualObligationsByKind[stringifySyncObligationKind(obligation.kind).str()];
        addSaturating(count, 1);
    }
}

} // namespace

FailureOr<SyncInterpretationResult> mlir::pto::protocol_sync::interpretSelectedWorld(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncSelectedWorld& world, ProtocolSyncStatistics* statistics)
{
    ResidualAccumulator accumulator;
    bool statisticsUpdated = false;
    auto statisticsGuard = llvm::make_scope_exit([&]() {
        if (!statisticsUpdated) {
            updateStatistics(accumulator.result, statistics);
        }
    });
    const bool invalidAnalysis =
        !schedule.isFrozen() || timelines.getTimelines().size() != channels.getChannels().size();
    if (invalidAnalysis) {
        return failure();
    }
    accumulator.result.peakStates = 1;
    CompletionGraph completionGraph(schedule.getPhases().size());
    CompletionGraph visibilityGraph(schedule.getPhases().size());
    if (failed(validateWorld(schedule, channels, world, completionGraph, visibilityGraph))) {
        return failure();
    }

    AbstractState state;
    for (auto [index, timeline] : llvm::enumerate(timelines.getTimelines())) {
        const SyncChannel& channel = channels.getChannels()[index];
        if (timeline.id != index || channel.id != index || channel.generation != timeline.id ||
            failed(evaluateTimeline(schedule, stages, timeline, channel, world, completionGraph, state, accumulator))) {
            return failure();
        }
    }
    const bool invalidResult = failed(evaluateMemoryHazards(schedule, completionGraph, visibilityGraph, accumulator)) ||
                               failed(evaluateSSACompletions(schedule, completionGraph, accumulator)) ||
                               failed(evaluateOpaqueEffects(schedule, accumulator)) ||
                               failed(evaluateExitCompletion(schedule, world, completionGraph, accumulator));
    if (invalidResult) {
        return failure();
    }
    updateStatistics(accumulator.result, statistics);
    statisticsUpdated = true;
    return std::move(accumulator.result);
}
