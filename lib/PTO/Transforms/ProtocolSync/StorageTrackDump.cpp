// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StorageTrackDump.cpp - Stable storage-track diagnostics --------===//

#include "PTO/Transforms/ProtocolSync/StorageTrackAnalysis.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

StringRef stringifyStorageSpace(AddressSpace space)
{
    switch (space) {
        case AddressSpace::GM:
            return "gm";
        case AddressSpace::MAT:
            return "mat";
        case AddressSpace::LEFT:
            return "left";
        case AddressSpace::RIGHT:
            return "right";
        case AddressSpace::ACC:
            return "acc";
        case AddressSpace::VEC:
            return "vec";
        case AddressSpace::BIAS:
            return "bias";
        case AddressSpace::SCALING:
            return "scaling";
        case AddressSpace::Zero:
            return "unknown";
    }
    return "unknown";
}

void printGuard(ArrayRef<SyncControlAtom> guard, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        guard, output, [&](const SyncControlAtom& atom) { output << '#' << atom.choice << ':' << atom.arm; });
    output << ']';
}

void printIds(ArrayRef<std::uint32_t> ids, StringRef prefix, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](std::uint32_t id) { output << prefix << id; });
    output << ']';
}

void printLaneId(SyncExecutionLaneId id, raw_ostream& output)
{
    if (id == kInvalidSyncId) {
        output << "unknown";
    } else {
        output << '#' << id;
    }
}

StringRef getCoreResourceName(SyncPhysicalCore core)
{
    switch (core) {
        case SyncPhysicalCore::Cube:
            return "AIC";
        case SyncPhysicalCore::Vector:
            return "AIV";
        case SyncPhysicalCore::Unknown:
            return "unknown";
    }
    return "unknown";
}

void printResource(
    const LaneFrontierAnalysisResult& laneFrontiers, SyncExecutionLaneId sourceId, SyncExecutionLaneId targetId,
    raw_ostream& output)
{
    const SyncExecutionLane* source = laneFrontiers.findLane(sourceId);
    const SyncExecutionLane* target = laneFrontiers.findLane(targetId);
    if (!source || !target) {
        output << "unknown";
        return;
    }
    output << getCoreResourceName(source->core) << ':' << stringifyPIPE(source->pipe) << "->"
           << getCoreResourceName(target->core) << ':' << stringifyPIPE(target->pipe);
}

void printFrontier(const SyncProgramFrontier& frontier, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(frontier.points, output, [&](const SyncGuardedProgramPoint& point) {
        output << "pp" << point.point << " guard=";
        printGuard(point.guard, output);
    });
    output << ']';
}

void printPointDescription(const StructuredSyncIR& schedule, SyncProgramPointId id, raw_ostream& output)
{
    if (id >= schedule.getProgramPoints().size()) {
        output << "unknown(pp" << id << ')';
        return;
    }
    const SyncProgramPoint& point = schedule.getProgramPoints()[id];
    Operation* operation = nullptr;
    bool before = false;
    switch (point.kind) {
        case SyncProgramPointKind::RegionEntry:
        case SyncProgramPointKind::RegionExit: {
            const SyncRegion* region = schedule.findRegion(point.region);
            operation = region ? region->operation : nullptr;
            before = point.kind == SyncProgramPointKind::RegionEntry;
            break;
        }
        case SyncProgramPointKind::SemanticActionBefore:
        case SyncProgramPointKind::SemanticActionAfter: {
            const SyncSemanticAction* action = schedule.findSemanticAction(point.action);
            operation = action ? action->operation : nullptr;
            before = point.kind == SyncProgramPointKind::SemanticActionBefore;
            break;
        }
        case SyncProgramPointKind::PhaseBefore:
        case SyncProgramPointKind::PhaseAfter: {
            const SyncPhase* phase = schedule.findPhase(point.phase);
            operation = phase ? phase->operation : nullptr;
            before = point.kind == SyncProgramPointKind::PhaseBefore;
            break;
        }
    }
    if (!operation) {
        output << "unknown(pp" << id << ')';
        return;
    }
    output << (before ? "before(" : "after(") << operation->getName() << ')';
}

void printReferenceEndpoint(const StructuredSyncIR& schedule, const SyncProgramFrontier& frontier, raw_ostream& output)
{
    const bool hasSinglePoint = frontier.points.size() == 1;
    if (!hasSinglePoint) {
        output << "multi-point";
        return;
    }
    printPointDescription(schedule, frontier.points.front().point, output);
}

StringRef getReferencePattern(SyncStorageTransitionKind kind)
{
    switch (kind) {
        case SyncStorageTransitionKind::Ready:
            return "storage-ready-frontier";
        case SyncStorageTransitionKind::Release:
            return "storage-release-frontier";
        case SyncStorageTransitionKind::Completion:
            return "multi-demand-pipe-barrier";
        case SyncStorageTransitionKind::Residual:
            return "raw-storage-hazard";
    }
    return "raw-storage-hazard";
}

StringRef getReferenceMechanism(const SyncStorageTransitionFrontier& transition)
{
    switch (transition.targetResult) {
        case SyncStorageTargetResult::Intrinsic:
            return "intrinsic";
        case SyncStorageTargetResult::PipeBarrier:
            return "barrier";
        case SyncStorageTargetResult::Event:
            return transition.recurring ? "recurring-event" : "event";
        case SyncStorageTargetResult::UnsupportedTarget:
        case SyncStorageTargetResult::UnsupportedMechanism:
        case SyncStorageTargetResult::NotApplicable:
            return "unresolved";
    }
    return "unresolved";
}

void printTrack(const SyncStorageTrack& track, raw_ostream& output)
{
    output << "  storage-track #" << track.id << " space=" << stringifyStorageSpace(track.space) << " range=["
           << track.begin << ',' << track.begin + track.size << ") families=[";
    llvm::interleaveComma(track.families, output, [&](const SyncStorageTrackFamilyBinding& binding) {
        output << '#' << binding.family << ":slot=";
        if (binding.physicalSlot) {
            output << *binding.physicalSlot;
        } else {
            output << "none";
        }
    });
    output << "] uncertain-alias=" << (track.uncertainAlias ? "yes" : "no")
           << " multiple-cores=" << (track.multiplePhysicalCores ? "yes" : "no") << '\n';
    for (const SyncStorageTrackOccurrence& occurrence : track.occurrences) {
        output << "    occurrence access=a#" << occurrence.access << " family=#" << occurrence.family
               << " physical-slot=";
        if (occurrence.physicalSlot) {
            output << *occurrence.physicalSlot;
        } else {
            output << "none";
        }
        output << " generation=#" << occurrence.generation << " phase=#" << occurrence.phase << " stage=#"
               << occurrence.stage << " execution-lane=";
        printLaneId(occurrence.executionLane, output);
        output << " before=pp" << occurrence.before << " after=pp" << occurrence.after
               << " mode=" << stringifySyncAccessMode(occurrence.mode)
               << " slot-binding=" << stringifySyncStorageSlotBinding(occurrence.slotBinding);
        if (occurrence.slot) {
            output << " slot-depth=" << occurrence.slot->depth << " slot-loop=#" << occurrence.slot->loop
                   << " slot-coefficient=" << occurrence.slot->coefficient << " slot-offset=" << occurrence.slot->offset
                   << " slot-modulus=" << occurrence.slot->modulus;
        }
        output << " membership=";
        if (!occurrence.slot) {
            output << "unconditional";
        } else if (occurrence.physicalSlot) {
            output << "slot(t)==" << *occurrence.physicalSlot;
        } else {
            output << "unresolved";
        }
        output << " guard=";
        printGuard(occurrence.guard, output);
        output << " loops=";
        printIds(occurrence.iterationDomain.loops, "#", output);
        output << '\n';
    }
}

void printTransition(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    const SyncStorageTransitionFrontier& transition, raw_ostream& output)
{
    output << "  storage-transition #" << transition.id
           << " kind=" << stringifySyncStorageTransitionKind(transition.kind)
           << " origin=" << stringifySyncStorageTransitionOrigin(transition.origin)
           << " old-pattern=" << getReferencePattern(transition.kind)
           << " old-mechanism=" << getReferenceMechanism(transition) << " lanes=";
    printLaneId(transition.sourceLane, output);
    output << "->";
    printLaneId(transition.targetLane, output);
    output << " resource=";
    printResource(laneFrontiers, transition.sourceLane, transition.targetLane, output);
    output << " generation=#" << transition.generation << " tracks=";
    printIds(transition.tracks, "t#", output);
    output << " raw-pair-members=";
    printIds(transition.rawPairMembers, "r#", output);
    output << '\n';
    output << "    source-frontier=";
    printFrontier(transition.sourceFrontier, output);
    output << " target-frontier=";
    printFrontier(transition.targetFrontier, output);
    output << " iteration-distance=" << transition.iterationDistance << '\n';
    output << "    reference-placement=";
    printReferenceEndpoint(schedule, transition.sourceFrontier, output);
    output << "->";
    printReferenceEndpoint(schedule, transition.targetFrontier, output);
    output << '\n';
    output << "    target-query=" << stringifySyncStorageTargetResult(transition.targetResult)
           << " checkpoint-e=" << stringifySyncCheckpointEStatus(transition.checkpointE)
           << " checkpoint-e-reason=" << stringifySyncReadyReleaseRejection(transition.checkpointERejection)
           << " cost-logical=" << transition.cost.logicalCandidates
           << " cost-steady-actions=" << transition.cost.steadyStateActions << " selectable=no\n";
}

void printProjectionAudit(const SyncStorageProjectionAudit& audit, raw_ostream& output)
{
    output << "  projection-audit status=" << (audit.isExact() ? "exact" : "mismatch")
           << " exact-accesses=" << audit.exactAccesses << " access-mask-mismatches=" << audit.accessMaskMismatches
           << " pair-relations=" << audit.accessPairRelations << " overlap-pairs=" << audit.overlappingAccessPairs
           << " disjoint-pairs=" << audit.disjointAccessPairs
           << " overlap-missing-track=" << audit.overlapPairsMissingSharedTrack
           << " disjoint-sharing-track=" << audit.disjointPairsSharingTrack
           << " read-read-overlap=" << audit.readReadOverlapPairs
           << " accumulator-read-read=" << audit.accumulatorReadReadOverlapPairs
           << " cross-lane-read-read=" << audit.crossLaneReadReadOverlapPairs
           << " read-read-effect-query=unavailable components=" << audit.overlapComponents
           << " max-atoms-per-component=" << audit.maximumAtomsPerComponent
           << " max-atoms-per-access=" << audit.maximumAtomsPerAccess << '\n';
}

void printTransitionAudit(const SyncStorageTransitionAudit& audit, raw_ostream& output)
{
    output << "  transition-audit status=" << (audit.isExact() ? "exact" : "mismatch")
           << " raw-pairs=" << audit.rawPairs << " memberships=" << audit.pairMemberships
           << " covered-once=" << audit.pairsCoveredOnce << " uncovered=" << audit.pairsUncovered
           << " multiply-covered=" << audit.pairsMultiplyCovered
           << " invalid-memberships=" << audit.invalidPairMemberships
           << " track-mask-mismatches=" << audit.trackMaskMismatches
           << " linear-frontier-memberships=" << audit.linearFrontierMemberships
           << " linear-frontier-mismatches=" << audit.linearFrontierMismatches
           << " non-linear-frontier-memberships=" << audit.frontierMembershipsNotLinear << '\n';
}

void printLifecycleReconstruction(const SyncStorageLifecycleReconstruction& reconstruction, raw_ostream& output)
{
    output << "  independent-lifecycle component=#" << reconstruction.component
           << " status=" << (reconstruction.isReady() ? "ready" : "rejected")
           << " reason=" << stringifySyncStorageLifecycleRejection(reconstruction.rejection) << " family=#"
           << reconstruction.family << " producer=a#" << reconstruction.producerAccess << " consumer=a#"
           << reconstruction.consumerAccess << " loop=#" << reconstruction.loop << " lanes=";
    printLaneId(reconstruction.producerLane, output);
    output << "->";
    printLaneId(reconstruction.consumerLane, output);
    output << " capacity=" << reconstruction.capacity << " reuse-distance=" << reconstruction.reuseDistance
           << " tracks=";
    printIds(reconstruction.tracks, "t#", output);
    output << " publication=pp" << reconstruction.publication << " acquisition=pp" << reconstruction.acquisition
           << " final-use=pp" << reconstruction.finalUse << " next-overwrite=pp" << reconstruction.nextOverwrite
           << " selectable=no\n";
}

void printEDifferential(const SyncStorageEDifferential& differential, raw_ostream& output)
{
    output << "  independent-e-differential status=" << stringifySyncStorageEDifferentialStatus(differential.status)
           << " e-rejection=" << stringifySyncReadyReleaseRejection(differential.eRejection)
           << " capacity=" << (differential.capacityMatches ? "match" : "mismatch")
           << " lanes=" << (differential.lanesMatch ? "match" : "mismatch")
           << " loop=" << (differential.loopMatches ? "match" : "mismatch")
           << " phases=" << (differential.phasesMatch ? "match" : "mismatch")
           << " lifecycle=" << (differential.lifecycleMatches ? "match" : "mismatch")
           << " physical-slots=" << (differential.physicalSlotsMatch ? "match" : "mismatch") << '\n';
}

} // namespace

void mlir::pto::protocol_sync::printStorageTrackAnalysis(
    const StructuredSyncIR& schedule, const LaneFrontierAnalysisResult& laneFrontiers,
    const StorageTrackAnalysisResult& analysis, raw_ostream& output)
{
    output << "PROTOCOL-SYNC storage-tracks function=@" << schedule.getFunction().getSymName() << '\n';
    output << "  semantics=diagnostic-only storage-order=atomic-byte-segments "
              "lane-order=structured-partial selectable=no\n";
    printProjectionAudit(analysis.getProjectionAudit(), output);
    printTransitionAudit(analysis.getTransitionAudit(), output);
    for (const SyncStorageLifecycleReconstruction& reconstruction : analysis.getLifecycleReconstructions()) {
        printLifecycleReconstruction(reconstruction, output);
    }
    printEDifferential(analysis.getEDifferential(), output);
    for (const SyncStorageTrack& track : analysis.getTracks()) {
        printTrack(track, output);
    }
    for (const SyncUnprojectedStorageAccess& rejected : analysis.getUnprojectedAccesses()) {
        output << "  unprojected-access a#" << rejected.access
               << " reason=" << stringifySyncStorageProjectionRejection(rejected.reason) << '\n';
    }
    for (const SyncStorageTransitionFrontier& transition : analysis.getTransitions()) {
        printTransition(schedule, laneFrontiers, transition, output);
    }
    output << "PROTOCOL-SYNC storage-tracks-end function=@" << schedule.getFunction().getSymName() << '\n';
}
