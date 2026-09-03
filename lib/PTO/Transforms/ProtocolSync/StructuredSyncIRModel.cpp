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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

using namespace mlir;
using namespace mlir::pto::protocol_sync;

namespace {

bool hasValidCompleteSlots(const SyncStorageFamily& family)
{
    if (!family.slotCount || *family.slotCount < 2) {
        return true;
    }
    const bool incompleteSlots = !family.physicalSlotsComplete || family.intervals.size() != *family.slotCount;
    if (incompleteSlots) {
        return false;
    }
    for (auto [index, interval] : llvm::enumerate(family.intervals)) {
        const bool invalidInterval =
            interval.size == 0 || interval.begin > std::numeric_limits<std::uint64_t>::max() - interval.size;
        if (invalidInterval) {
            return false;
        }
        const std::uint64_t intervalEnd = interval.begin + interval.size;
        for (const SyncByteInterval& previous : llvm::ArrayRef(family.intervals).take_front(index)) {
            const std::uint64_t previousEnd = previous.begin + previous.size;
            if (interval.begin < previousEnd && previous.begin < intervalEnd) {
                return false;
            }
        }
    }
    return true;
}

bool hasValidCanonicalSlot(
    const SyncSlotExpression& slot, const SyncStorageFamily& family, const SyncPhase& phase,
    ArrayRef<SyncRegion> regions)
{
    if (slot.kind == SyncSlotExpressionKind::Unknown) {
        return true;
    }
    if (slot.depth == 0 || slot.modulus != slot.depth || !family.slotCount || *family.slotCount != slot.depth ||
        slot.offset < 0 || static_cast<std::uint64_t>(slot.offset) >= slot.modulus) {
        return false;
    }
    if (slot.kind == SyncSlotExpressionKind::Constant) {
        return !slot.induction && slot.loop == kInvalidSyncId && slot.coefficient == 0;
    }
    const bool invalidAffineForm = !slot.induction || slot.loop >= regions.size() ||
                                   !llvm::is_contained(phase.iterationDomain.loops, slot.loop) ||
                                   slot.coefficient < 0 || static_cast<std::uint64_t>(slot.coefficient) >= slot.modulus;
    if (invalidAffineForm) {
        return false;
    }
    const SyncRegion& loopRegion = regions[slot.loop];
    auto loop = dyn_cast_or_null<scf::ForOp>(loopRegion.operation);
    return loopRegion.kind == SyncRegionKind::Loop && loop && loop.getInductionVar() == slot.induction;
}

} // namespace

const SyncRegion* StructuredSyncIR::findRegion(SyncRegionId id) const
{
    return id < regions.size() ? &regions[id] : nullptr;
}

const SyncPhase* StructuredSyncIR::findPhase(SyncPhaseId id) const
{
    return id < phases.size() ? &phases[id] : nullptr;
}

const SyncStorageFamily* StructuredSyncIR::findStorageFamily(SyncStorageFamilyId id) const
{
    return id < storageFamilies.size() ? &storageFamilies[id] : nullptr;
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
        const bool invalidFamily = access.family >= storageFamilies.size();
        if (invalidId || invalidPhase || invalidFamily) {
            return failure();
        }
        if (!access.value) {
            return failure();
        }
        const SyncStorageFamily& family = storageFamilies[access.family];
        if (family.root != access.storage.root || family.space != access.storage.space) {
            return failure();
        }
        if (access.slot && access.slot->depth != 0 && (!family.slotCount || *family.slotCount != access.slot->depth)) {
            return failure();
        }
        if (access.slot && !hasValidCanonicalSlot(*access.slot, family, phases[access.phase], regions)) {
            return failure();
        }
        if (accessReferences[index] != 1) {
            return failure();
        }
    }
    for (auto [index, family] : llvm::enumerate(storageFamilies)) {
        const bool invalidFamily = family.id != index || !family.root || (family.slotCount && *family.slotCount == 0) ||
                                   family.capacityConflict || (family.physical && !hasValidCompleteSlots(family));
        if (invalidFamily) {
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

StringRef mlir::pto::protocol_sync::stringifySyncRegionKind(SyncRegionKind kind)
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

StringRef mlir::pto::protocol_sync::stringifySyncCardinality(SyncCardinality cardinality)
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

StringRef mlir::pto::protocol_sync::stringifySyncStorageRole(SyncStorageRole role)
{
    switch (role) {
        case SyncStorageRole::LocalBuffer:
            return "local-buffer";
        case SyncStorageRole::GlobalBuffer:
            return "global-buffer";
        case SyncStorageRole::Unknown:
            return "unknown";
    }
    return "unknown";
}
