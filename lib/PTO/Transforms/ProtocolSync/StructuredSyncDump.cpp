// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredSyncDump.cpp - Deterministic schedule dump -------------===//

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"

#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

void printValue(mlir::Value value, AsmState& state, raw_ostream& output)
{
    if (!value) {
        output << "none";
        return;
    }
    value.printAsOperand(output, state);
}

void printIds(ArrayRef<SyncRegionId> ids, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](SyncRegionId id) { output << '#' << id; });
    output << ']';
}

void printGuard(ArrayRef<SyncControlAtom> guard, raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(
        guard, output, [&](const SyncControlAtom& atom) { output << '#' << atom.choice << ':' << atom.arm; });
    output << ']';
}

void printIntervals(ArrayRef<SyncByteInterval> intervals, raw_ostream& output)
{
    if (intervals.empty()) {
        output << "unknown";
        return;
    }
    output << '[';
    llvm::interleaveComma(
        intervals, output, [&](const SyncByteInterval& interval) { output << interval.begin << '+' << interval.size; });
    output << ']';
}

void printStorageFamily(const SyncStorageFamily& family, AsmState& state, raw_ostream& output)
{
    output << "  storage-family #" << family.id << " root=";
    printValue(family.root, state, output);
    output << " role=" << stringifySyncStorageRole(family.role) << " space=" << static_cast<unsigned>(family.space)
           << " physical=" << (family.physical ? "yes" : "no")
           << " aliases-unknown=" << (family.aliasesUnknownRange ? "yes" : "no") << " slot-count=";
    if (family.slotCount) {
        output << *family.slotCount;
    } else {
        output << "unknown";
    }
    output << " range=";
    printIntervals(family.intervals, output);
    output << " unknown-range=" << (family.unknownRange ? "yes" : "no")
           << " complete-slots=" << (family.physicalSlotsComplete ? "yes" : "no") << '\n';
}

void printSlotExpression(const SyncSlotExpression& slot, AsmState& state, raw_ostream& output)
{
    output << " expr=" << stringifySyncSlotExpressionKind(slot.kind);
    if (slot.kind == SyncSlotExpressionKind::Constant) {
        output << "(offset=" << slot.offset << ",modulus=" << slot.modulus << ')';
    } else if (slot.kind == SyncSlotExpressionKind::AffineModulo) {
        output << "(loop=#" << slot.loop << ",induction=";
        printValue(slot.induction, state, output);
        output << ",coefficient=" << slot.coefficient << ",offset=" << slot.offset << ",modulus=" << slot.modulus
               << ')';
    }
    output << " distance-1=" << stringifySyncSlotRelation(compareSlotsAtDistance(slot, slot, 1))
           << " distance-2=" << stringifySyncSlotRelation(compareSlotsAtDistance(slot, slot, 2)) << " reuse=";
    const unsigned searchLimit = slot.depth != 0 ? slot.depth : slot.modulus;
    FailureOr<unsigned> reuse = findFirstPositiveReuseDistance(slot, searchLimit);
    if (succeeded(reuse)) {
        output << *reuse;
    } else {
        output << "unknown";
    }
}

void printSummary(const SyncOpSummary& summary, unsigned id, AsmState& state, raw_ostream& output)
{
    output << "  summary #" << id << " op=" << summary.operation->getName().getStringRef()
           << " provider=" << stringifySyncSummaryProvider(summary.provider) << " phases=" << summary.phases.size()
           << " completion=" << stringifySyncCompletionKind(summary.completion.kind) << " supplies=[";
    llvm::interleaveComma(
        summary.suppliedProtocols, output, [&](const SyncInternalProtocol& protocol) { output << protocol.kind; });
    output << ']';
    if (summary.queue) {
        output << " queue=" << stringifySyncQueueRole(summary.queue->role) << " handle=";
        printValue(summary.queue->handle, state, output);
        output << " depth=";
        if (summary.queue->depth) {
            output << *summary.queue->depth;
        } else {
            output << "unknown";
        }
        output << " direction=" << static_cast<unsigned>(summary.queue->directionMask) << " local-slots=";
        if (summary.queue->localSlotCount) {
            output << *summary.queue->localSlotCount;
        } else {
            output << "unknown";
        }
        output << " flag-base=";
        if (summary.queue->flagBase) {
            output << *summary.queue->flagBase;
        } else {
            output << "unknown";
        }
    }
    output << '\n';
    for (const SyncEventReservation& reservation : summary.eventReservations) {
        output << "    reservation " << stringifyPIPE(reservation.source) << "->" << stringifyPIPE(reservation.target)
               << " ids=[";
        llvm::interleaveComma(reservation.eventIds, output);
        output << "]\n";
    }
}

void printRegion(const SyncRegion& region, raw_ostream& output)
{
    output << "  region #" << region.id << " kind=" << stringifySyncRegionKind(region.kind) << " parent=";
    if (region.parent == kInvalidSyncId) {
        output << "none";
    } else {
        output << '#' << region.parent;
    }
    output << " cardinality=" << stringifySyncCardinality(region.cardinality) << " arm=" << region.arm << " guard=";
    printGuard(region.guard, output);
    output << " loops=";
    printIds(region.iterationDomain.loops, output);
    output << " entry=pp" << region.entry << " exit=pp" << region.exit << " elements=[";
    llvm::interleaveComma(region.elements, output, [&](const SyncRegionElement& element) {
        if (element.kind == SyncRegionElement::Kind::Phase) {
            output << "phase#" << element.phase;
        } else {
            output << "region#" << element.child;
        }
    });
    output << "]\n";
}

void printPhase(const StructuredSyncIR& schedule, const SyncPhase& phase, AsmState& state, raw_ostream& output)
{
    output << "  phase #" << phase.id << " region=#" << phase.region
           << " op=" << phase.operation->getName().getStringRef() << " macro=";
    if (phase.macroPhase) {
        output << *phase.macroPhase;
    } else {
        output << "none";
    }
    output << " core=" << stringifySyncPhysicalCore(phase.core) << " pipe=" << stringifyPIPE(phase.pipe)
           << " completion=" << stringifySyncCompletionKind(phase.completion) << " guard=";
    printGuard(phase.guard, output);
    output << " loops=";
    printIds(phase.iterationDomain.loops, output);
    output << " before=pp" << phase.before << " after=pp" << phase.after << '\n';
    for (SyncAccessId id : phase.accesses) {
        const SyncAccess* access = schedule.findAccess(id);
        if (!access) {
            output << "    access invalid-id=" << id << '\n';
            continue;
        }
        output << "    access #" << access->id << ' ' << stringifySyncAccessMode(access->mode) << " value=";
        printValue(access->value, state, output);
        output << " root=";
        printValue(access->storage.root, state, output);
        output << " space=" << static_cast<unsigned>(access->storage.space)
               << " visibility=" << stringifySyncVisibility(access->visibility)
               << " physical=" << (access->storage.physical ? "yes" : "no")
               << " aliases-unknown=" << (access->storage.aliasesUnknownRange ? "yes" : "no") << " range=";
        printIntervals(access->storage.intervals, output);
        output << " family=#" << access->family << " slot=";
        if (access->slot) {
            printValue(access->slot->selector, state, output);
            output << '/' << access->slot->depth;
            printSlotExpression(*access->slot, state, output);
        } else {
            output << "none";
        }
        output << '\n';
    }
}

} // namespace

void mlir::pto::protocol_sync::printStructuredSyncIR(const StructuredSyncIR& schedule, raw_ostream& output)
{
    output << "PROTOCOL-SYNC schedule function=@" << schedule.getFunction().getSymName()
           << " frozen=" << (schedule.isFrozen() ? "yes" : "no") << '\n';
    AsmState state(schedule.getFunction());
    unsigned physicalLocalFamilies = 0;
    unsigned logicalStorageFamilies = 0;
    for (const SyncStorageFamily& family : schedule.getStorageFamilies()) {
        const bool completePhysicalIdentity =
            !family.slotCount || *family.slotCount < 2 || family.physicalSlotsComplete;
        const bool hasPhysicalLocalIdentity = family.role == SyncStorageRole::LocalBuffer && family.physical &&
                                              !family.intervals.empty() && completePhysicalIdentity;
        if (hasPhysicalLocalIdentity) {
            ++physicalLocalFamilies;
        }
        if (family.slotCount) {
            ++logicalStorageFamilies;
        }
    }
    llvm::DenseSet<Value> queueHandles;
    for (const SyncOpSummary& summary : schedule.getSummaries()) {
        if (summary.queue && summary.queue->handle && summary.queue->depth) {
            queueHandles.insert(summary.queue->handle);
        }
    }
    output << "  preservation-point=post-plan-pre-buffer-select physical-local-families=" << physicalLocalFamilies
           << " logical-storage-families=" << logicalStorageFamilies << " queue-identities=" << queueHandles.size()
           << '\n';
    for (const auto& [id, summary] : llvm::enumerate(schedule.getSummaries())) {
        printSummary(summary, id, state, output);
    }
    for (const SyncStorageFamily& family : schedule.getStorageFamilies()) {
        printStorageFamily(family, state, output);
    }
    for (const SyncRegion& region : schedule.getRegions()) {
        printRegion(region, output);
    }
    for (const SyncPhase& phase : schedule.getPhases()) {
        printPhase(schedule, phase, state, output);
    }
    for (const SyncFailure& failure : schedule.getFailures()) {
        output << "  rejected reason=" << stringifySyncFailureReason(failure.reason)
               << " op=" << failure.operation->getName().getStringRef() << " detail=\"" << failure.detail << "\"\n";
    }
    output << "PROTOCOL-SYNC schedule-end function=@" << schedule.getFunction().getSymName() << '\n';
}
