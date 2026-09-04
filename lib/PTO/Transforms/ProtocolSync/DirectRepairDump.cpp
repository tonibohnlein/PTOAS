// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DirectRepairDump.cpp - Stable direct-repair diagnostics ---------===//

#include "PTO/Transforms/ProtocolSync/DirectRepair.h"

#include "PTO/Support/CodeConstants.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

void printIdList(ArrayRef<SyncObligationId> ids, llvm::raw_ostream& output)
{
    output << '[';
    llvm::interleaveComma(ids, output, [&](SyncObligationId id) { output << '#' << id; });
    output << ']';
}

} // namespace

void mlir::pto::protocol_sync::printDirectRepairPlan(
    func::FuncOp function, const SyncDirectRepairPlan& plan, llvm::raw_ostream& output)
{
    output << "PROTOCOL-SYNC direct-repair function=@" << function.getSymName()
           << " status=" << stringifySyncDirectRepairPlanStatus(plan.status) << " obligations=" << plan.obligationCount
           << " candidates=" << plan.candidates.size() << " uncovered=" << plan.uncoveredObligations.size() << '\n';
    for (const SyncDirectRepairCandidate& candidate : plan.candidates) {
        output << "  candidate #" << candidate.id << " kind=" << stringifySyncDirectRepairKind(candidate.kind)
               << " core=" << stringifySyncPhysicalCore(candidate.core)
               << " source-pipe=" << stringifyPIPE(candidate.sourcePipe)
               << " target-pipe=" << stringifyPIPE(candidate.targetPipe) << " source=";
        if (candidate.sourcePhase == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << candidate.sourcePhase;
        }
        output << " target=";
        if (candidate.targetPhase == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << candidate.targetPhase;
        }
        output << " control=" << stringifySyncControlRelation(candidate.control)
               << " iteration=" << stringifySyncIterationRelationKind(candidate.iteration.kind)
               << " distance=" << candidate.iteration.distance << " carrier=";
        if (candidate.iteration.carrier == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << candidate.iteration.carrier;
        }
        output << " event=";
        if (candidate.eventId) {
            output << *candidate.eventId;
        } else {
            output << "unallocated";
        }
        output << " tail-section=";
        if (candidate.tailSectionOperation) {
            output << candidate.tailSectionOperation->getName();
        } else {
            output << "none";
        }
        output << " covers=";
        printIdList(candidate.obligations, output);
        output << '\n';
    }
    for (const SyncDirectRepairPlanRejection& rejection : plan.rejections) {
        output << "  rejection obligation=";
        if (rejection.obligation == kInvalidSyncId) {
            output << "none";
        } else {
            output << '#' << rejection.obligation;
        }
        output << " reason=" << stringifySyncDirectRepairRejection(rejection.reason) << " detail=\"" << rejection.detail
               << "\"\n";
    }
    output << "PROTOCOL-SYNC direct-repair-end function=@" << function.getSymName() << '\n';
}

StringRef mlir::pto::protocol_sync::stringifySyncDirectRepairKind(SyncDirectRepairKind kind)
{
    switch (kind) {
        case SyncDirectRepairKind::PipeBarrier:
            return "pipe-barrier";
        case SyncDirectRepairKind::DirectedEvent:
            return "directed-event";
        case SyncDirectRepairKind::ExitBarrier:
            return "exit-barrier";
    }
    return "unknown";
}

StringRef mlir::pto::protocol_sync::stringifySyncDirectRepairPlanStatus(SyncDirectRepairPlanStatus status)
{
    switch (status) {
        case SyncDirectRepairPlanStatus::Empty:
            return "empty";
        case SyncDirectRepairPlanStatus::Ready:
            return "ready";
        case SyncDirectRepairPlanStatus::Partial:
            return "partial";
        case SyncDirectRepairPlanStatus::Unsupported:
            return "unsupported";
        case SyncDirectRepairPlanStatus::ResourceInfeasible:
            return "resource-infeasible";
    }
    return "unsupported";
}

StringRef mlir::pto::protocol_sync::stringifySyncDirectRepairRejection(SyncDirectRepairRejection rejection)
{
    switch (rejection) {
        case SyncDirectRepairRejection::None:
            return "none";
        case SyncDirectRepairRejection::UnsupportedTarget:
            return "unsupported-target";
        case SyncDirectRepairRejection::UnsupportedObligation:
            return "unsupported-obligation";
        case SyncDirectRepairRejection::InvalidEndpoint:
            return "invalid-endpoint";
        case SyncDirectRepairRejection::UnsupportedControl:
            return "unsupported-control";
        case SyncDirectRepairRejection::UnsupportedRecurrence:
            return "unsupported-recurrence";
        case SyncDirectRepairRejection::UnsupportedStageShape:
            return "unsupported-stage-shape";
        case SyncDirectRepairRejection::MixedPhysicalCores:
            return "mixed-physical-cores";
        case SyncDirectRepairRejection::UnorderedEndpoints:
            return "unordered-endpoints";
        case SyncDirectRepairRejection::UnsupportedBarrier:
            return "unsupported-barrier";
        case SyncDirectRepairRejection::UnsupportedEventDirection:
            return "unsupported-event-direction";
        case SyncDirectRepairRejection::EventCapacity:
            return "event-capacity";
        case SyncDirectRepairRejection::InternalInvariant:
            return "internal-invariant";
    }
    return "internal-invariant";
}
