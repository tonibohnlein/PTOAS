// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredFrontierDomain.cpp - Control and occurrence qualification --===//
// A bounded vector-core region language: sequences, result-free choices and
// one top-level positive-step loop. Participation is occurrence-sensitive;
// opposing loop arms are exclusive only within the same iteration.
#include "PTO/Transforms/ProtocolSync/StructuredFrontier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
bool implies(ArrayRef<SyncControlAtom> a, ArrayRef<SyncControlAtom> b)
{
    return llvm::all_of(b, [&](const SyncControlAtom& required) {
        return llvm::any_of(
            a, [&](const SyncControlAtom& atom) { return atom.choice == required.choice && atom.arm == required.arm; });
    });
}
bool ordinarySummary(const SyncOpSummary& summary, bool fixed)
{
    const bool provider =
        summary.provider == SyncSummaryProvider::Structural || summary.provider == SyncSummaryProvider::Pipeline;
    const bool synchronization = fixed && isa_and_nonnull<SetFlagOp, WaitFlagOp, BarrierOp>(summary.operation);
    // Explicit set/wait/barrier summaries advertise fixed supply. The concrete
    // checker must consume these actual instructions, not reject their provider
    // marker as though it were an opaque macro protocol.
    return summary.eventReservations.empty() && !summary.queue &&
           (synchronization || (provider && summary.suppliedProtocols.empty()));
}
} // namespace

bool mlir::pto::protocol_sync::supportsStructuredFrontier(const StructuredSyncIR& schedule, bool allowFixedSync)
{
    const bool valid = schedule.isFrozen() && schedule.getFailures().empty() &&
                       llvm::hasSingleElement(schedule.getFunction().getBody()) && !schedule.getPhases().empty() &&
                       schedule.getPhases().size() <= 128 && schedule.getAccesses().size() <= 1024;
    if (!valid) {
        return false;
    }
    bool choice = false;
    unsigned loops = 0;
    for (const SyncRegion& region : schedule.getRegions()) {
        const bool unsupportedRegion = region.guard.size() > 4 || region.kind == SyncRegionKind::PhysicalSection;
        if (unsupportedRegion) {
            return false;
        }
        if (region.kind == SyncRegionKind::Choice) {
            auto op = dyn_cast_or_null<scf::IfOp>(region.operation);
            const bool unsupportedChoice = !op || op.getNumResults() != 0;
            if (unsupportedChoice) {
                return false;
            }
            choice = true;
        }
        if (region.kind == SyncRegionKind::Loop) {
            auto op = dyn_cast_or_null<scf::ForOp>(region.operation);
            const bool ordinary = op && ++loops == 1 && op.getNumRegionIterArgs() == 0 &&
                                  op->getBlock() == &schedule.getFunction().getBody().front();
            if (!ordinary) {
                return false;
            }
            IntegerAttr step;
            const bool positive = matchPattern(op.getStep(), m_Constant(&step)) && step.getValue().isStrictlyPositive();
            if (!positive) {
                return false;
            }
        }
    }
    if (!choice) {
        return false;
    } // Preserve the existing straight-line/loop alternatives.
    for (const SyncPhase& phase : schedule.getPhases()) {
        const bool ordinary =
            phase.operation && !phase.macroPhase && phase.core == SyncPhysicalCore::Vector &&
            phase.completion == SyncCompletionKind::PhaseEnd &&
            (phase.pipe == PIPE::PIPE_V || phase.pipe == PIPE::PIPE_MTE2 || phase.pipe == PIPE::PIPE_MTE3);
        if (!ordinary) {
            return false;
        }
    }
    return llvm::all_of(schedule.getSummaries(), [&](const SyncOpSummary& summary) {
        return ordinarySummary(summary, allowFixedSync);
    });
}

SyncParticipationRelation mlir::pto::protocol_sync::classifySyncParticipation(
    const SyncPhase& source, const SyncPhase& target, const SyncIterationRelation& relation)
{
    if (relation.kind == SyncIterationRelationKind::Unknown) {
        return SyncParticipationRelation::Unknown;
    }
    if (relation.kind == SyncIterationRelationKind::LoopCarriedAny ||
        relation.kind == SyncIterationRelationKind::LoopCarried) {
        return SyncParticipationRelation::SameFeasiblePath;
    }
    for (const auto& a : source.guard) {
        for (const auto& b : target.guard) {
            if (a.choice == b.choice && a.arm != b.arm) {
                return SyncParticipationRelation::MutuallyExclusive;
            }
        }
    }
    const bool unconditional = source.guard.empty() && target.guard.empty();
    if (unconditional) {
        return SyncParticipationRelation::MustCoexecute;
    }
    const bool sourceImpliesTarget = implies(source.guard, target.guard);
    const bool targetImpliesSource = implies(target.guard, source.guard);
    if (sourceImpliesTarget && !targetImpliesSource) {
        return SyncParticipationRelation::SourceMayBeAbsent;
    }
    if (targetImpliesSource && !sourceImpliesTarget) {
        return SyncParticipationRelation::TargetMayBeAbsent;
    }
    return SyncParticipationRelation::SameFeasiblePath;
}

SyncIterationRelation mlir::pto::protocol_sync::structuredIterationRelation(
    const SyncPhase& source, const SyncPhase& target)
{
    if (source.iterationDomain.loops == target.iterationDomain.loops && source.id < target.id) {
        return {SyncIterationRelationKind::SameIteration, 0};
    }
    const bool entry =
        source.id < target.id && source.iterationDomain.loops.empty() && target.iterationDomain.loops.size() == 1;
    if (entry) {
        return {SyncIterationRelationKind::LoopEntry, 0, target.iterationDomain.loops.front()};
    }
    const bool exit =
        source.id < target.id && target.iterationDomain.loops.empty() && source.iterationDomain.loops.size() == 1;
    if (exit) {
        return {SyncIterationRelationKind::LoopExit, 0, source.iterationDomain.loops.front()};
    }
    return {};
}

bool mlir::pto::protocol_sync::structuredFrontierOrders(
    const StructuredSyncIR& schedule, ArrayRef<SyncPhaseId> acknowledged, SyncPhaseId source, SyncPhaseId target,
    const SyncIterationRelation& relation)
{
    const SyncPhase* a = schedule.findPhase(source);
    const SyncPhase* b = schedule.findPhase(target);
    const bool endpoints =
        a && b && llvm::is_contained(acknowledged, source) && llvm::is_contained(acknowledged, target);
    if (!endpoints) {
        return false;
    }
    const auto participation = classifySyncParticipation(*a, *b, relation);
    if (participation == SyncParticipationRelation::Unknown ||
        participation == SyncParticipationRelation::MutuallyExclusive) {
        return false;
    }
    if (relation.kind == SyncIterationRelationKind::LoopCarriedAny ||
        relation.kind == SyncIterationRelationKind::LoopCarried) {
        const bool distance =
            relation.kind == SyncIterationRelationKind::LoopCarriedAny ? relation.distance == 0 : relation.distance > 0;
        return distance && relation.carrier != kInvalidSyncId &&
               a->iterationDomain.loops == SmallVector<SyncRegionId, 2>{relation.carrier} &&
               b->iterationDomain.loops == a->iterationDomain.loops;
    }
    if (relation.kind == SyncIterationRelationKind::LoopBypass) {
        const auto* carrier = schedule.findRegion(relation.carrier);
        return carrier && carrier->operation && a->iterationDomain.loops.empty() && b->iterationDomain.loops.empty() &&
               relation.distance == 0 && a->operation->getBlock() == carrier->operation->getBlock() &&
               b->operation->getBlock() == carrier->operation->getBlock() &&
               a->operation->isBeforeInBlock(carrier->operation) && carrier->operation->isBeforeInBlock(b->operation);
    }
    const auto expected = structuredIterationRelation(*a, *b);
    return expected.kind == relation.kind && expected.kind != SyncIterationRelationKind::Unknown &&
           expected.distance == relation.distance && expected.carrier == relation.carrier;
}
