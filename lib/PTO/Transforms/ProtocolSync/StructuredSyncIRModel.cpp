// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredSyncIRModel.cpp - Immutable schedule invariants --------===//

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto::protocol_sync;

const SyncRegion* StructuredSyncIR::findRegion(SyncRegionId id) const
{
    return id < regions.size() ? &regions[id] : nullptr;
}

const SyncPhase* StructuredSyncIR::findPhase(SyncPhaseId id) const
{
    return id < phases.size() ? &phases[id] : nullptr;
}

const SyncAccess* StructuredSyncIR::findAccess(SyncAccessId id) const
{
    return id < accesses.size() ? &accesses[id] : nullptr;
}

LogicalResult StructuredSyncIR::freeze()
{
    if (frozen) {
        return failure();
    }
    llvm::SmallVector<unsigned, 64> phaseReferences(phases.size(), 0);
    llvm::SmallVector<unsigned, 64> accessReferences(accesses.size(), 0);
    for (auto [index, region] : llvm::enumerate(regions)) {
        const bool invalidId = region.id != index;
        const bool invalidParent = region.parent != kInvalidSyncId && region.parent >= regions.size();
        const bool invalidEntry = region.entry >= points.size();
        const bool invalidExit = region.exit >= points.size();
        if (invalidId || invalidParent || invalidEntry || invalidExit) {
            return failure();
        }
        const SyncProgramPoint& entry = points[region.entry];
        const SyncProgramPoint& exit = points[region.exit];
        const bool invalidEntryPoint = entry.kind != SyncProgramPointKind::RegionEntry || entry.region != region.id;
        const bool invalidExitPoint = exit.kind != SyncProgramPointKind::RegionExit || exit.region != region.id;
        if (invalidEntryPoint || invalidExitPoint) {
            return failure();
        }
        for (auto [elementIndex, element] : llvm::enumerate(region.elements)) {
            if (element.order != elementIndex) {
                return failure();
            }
            const bool invalidPhase = element.kind == SyncRegionElement::Kind::Phase && element.phase >= phases.size();
            const bool invalidChild =
                element.kind == SyncRegionElement::Kind::ChildRegion && element.child >= regions.size();
            if (invalidPhase || invalidChild) {
                return failure();
            }
            const bool wrongPhaseOwner =
                element.kind == SyncRegionElement::Kind::Phase && phases[element.phase].region != region.id;
            const bool wrongChildOwner =
                element.kind == SyncRegionElement::Kind::ChildRegion && regions[element.child].parent != region.id;
            if (wrongPhaseOwner || wrongChildOwner) {
                return failure();
            }
            if (element.kind == SyncRegionElement::Kind::Phase) {
                ++phaseReferences[element.phase];
            }
        }
    }
    for (auto [index, phase] : llvm::enumerate(phases)) {
        const bool invalidId = phase.id != index;
        const bool invalidSummary = phase.summary >= summaries.size();
        const bool invalidRegion = phase.region >= regions.size();
        const bool invalidBefore = phase.before >= points.size();
        const bool invalidAfter = phase.after >= points.size();
        if (invalidId || invalidSummary || invalidRegion || invalidBefore || invalidAfter) {
            return failure();
        }
        const bool wrongOperation = summaries[phase.summary].operation != phase.operation;
        const SyncProgramPoint& before = points[phase.before];
        const SyncProgramPoint& after = points[phase.after];
        const bool wrongBefore = before.kind != SyncProgramPointKind::PhaseBefore || before.phase != phase.id ||
                                 before.region != phase.region;
        const bool wrongAfter =
            after.kind != SyncProgramPointKind::PhaseAfter || after.phase != phase.id || after.region != phase.region;
        if (wrongOperation || wrongBefore || wrongAfter) {
            return failure();
        }
        for (SyncAccessId access : phase.accesses) {
            if (access >= accesses.size()) {
                return failure();
            }
            ++accessReferences[access];
            if (accesses[access].phase != phase.id) {
                return failure();
            }
        }
    }
    for (auto [index, access] : llvm::enumerate(accesses)) {
        const bool invalidId = access.id != index;
        const bool invalidPhase = access.phase >= phases.size();
        if (invalidId || invalidPhase) {
            return failure();
        }
        if (!access.value) {
            return failure();
        }
        if (accessReferences[index] != 1) {
            return failure();
        }
    }
    for (unsigned phaseReferenceCount : phaseReferences) {
        if (phaseReferenceCount != 1) {
            return failure();
        }
    }
    for (auto [index, point] : llvm::enumerate(points)) {
        const bool invalidId = point.id != index;
        const bool invalidRegion = point.region >= regions.size();
        const bool isRegionPoint =
            point.kind == SyncProgramPointKind::RegionEntry || point.kind == SyncProgramPointKind::RegionExit;
        const bool invalidPhase = isRegionPoint ? point.phase != kInvalidSyncId : point.phase >= phases.size();
        if (invalidId || invalidRegion || invalidPhase) {
            return failure();
        }
    }
    frozen = true;
    return success();
}

StringRef protocol_sync::stringifySyncRegionKind(SyncRegionKind kind)
{
    switch (kind) {
        case SyncRegionKind::Function:
            return "function";
        case SyncRegionKind::Sequence:
            return "sequence";
        case SyncRegionKind::Choice:
            return "choice";
        case SyncRegionKind::Alternative:
            return "alternative";
        case SyncRegionKind::Loop:
            return "loop";
        case SyncRegionKind::PhysicalSection:
            return "physical-section";
    }
    return "sequence";
}

StringRef protocol_sync::stringifySyncCardinality(SyncCardinality cardinality)
{
    switch (cardinality) {
        case SyncCardinality::ExactlyOnce:
            return "exactly-once";
        case SyncCardinality::ZeroOrOne:
            return "zero-or-one";
        case SyncCardinality::ZeroOrMore:
            return "zero-or-more";
        case SyncCardinality::OneOrMore:
            return "one-or-more";
    }
    return "exactly-once";
}
