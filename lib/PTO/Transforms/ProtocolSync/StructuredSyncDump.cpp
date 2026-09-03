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
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;
using namespace llvm;

namespace {

void printValue(Value value, AsmState& state, raw_ostream& output)
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
        if (access->storage.intervals.empty()) {
            output << "unknown";
        } else {
            output << '[';
            llvm::interleaveComma(access->storage.intervals, output, [&](const SyncByteInterval& interval) {
                output << interval.begin << '+' << interval.size;
            });
            output << ']';
        }
        output << " slot=";
        if (access->slot) {
            printValue(access->slot->selector, state, output);
            output << '/' << access->slot->depth;
        } else {
            output << "none";
        }
        output << '\n';
    }
}

} // namespace

void protocol_sync::printStructuredSyncIR(const StructuredSyncIR& schedule, raw_ostream& output)
{
    output << "PROTOCOL-SYNC schedule function=@" << schedule.getFunction().getSymName()
           << " frozen=" << (schedule.isFrozen() ? "yes" : "no") << '\n';
    AsmState state(schedule.getFunction());
    for (const auto& [id, summary] : llvm::enumerate(schedule.getSummaries())) {
        printSummary(summary, id, state, output);
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
