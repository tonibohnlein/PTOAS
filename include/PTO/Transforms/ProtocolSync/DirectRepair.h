// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DirectRepair.h - Targeted ProtocolSync residual repair -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_DIRECTREPAIR_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_DIRECTREPAIR_H

#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

using SyncDirectCandidateId = std::uint32_t;

enum class SyncDirectRepairKind : std::uint8_t {
    PipeBarrier,
    DirectedEvent,
    ExitBarrier,
};

enum class SyncDirectRepairPlanStatus : std::uint8_t {
    Empty,
    Ready,
    Partial,
    Unsupported,
    ResourceInfeasible,
};

enum class SyncDirectRepairRejection : std::uint8_t {
    None,
    UnsupportedTarget,
    UnsupportedObligation,
    InvalidEndpoint,
    UnsupportedControl,
    UnsupportedRecurrence,
    UnsupportedStageShape,
    MixedPhysicalCores,
    UnorderedEndpoints,
    UnsupportedBarrier,
    UnsupportedEventDirection,
    EventCapacity,
    InternalInvariant,
};

/// One indivisible physical recipe. A shared candidate covers every listed
/// obligation; selection and reverse deletion must therefore add or remove the
/// complete record rather than any individual action.
struct SyncDirectRepairCandidate {
    SyncDirectCandidateId id = kInvalidSyncId;
    SyncDirectRepairKind kind = SyncDirectRepairKind::PipeBarrier;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    SyncPhaseId sourcePhase = kInvalidSyncId;
    SyncPhaseId targetPhase = kInvalidSyncId;
    Operation* sourceOperation = nullptr;
    Operation* targetOperation = nullptr;
    Operation* tailSectionOperation = nullptr;
    SyncControlRelation control = SyncControlRelation::Unknown;
    SyncIterationRelation iteration;
    llvm::SmallVector<SyncObligationId, 4> obligations;
    std::optional<unsigned> eventId;
};

struct SyncDirectRepairPlanRejection {
    SyncObligationId obligation = kInvalidSyncId;
    SyncDirectRepairRejection reason = SyncDirectRepairRejection::None;
    std::string detail;
};

struct SyncDirectRepairPlan {
    SyncDirectRepairPlanStatus status = SyncDirectRepairPlanStatus::Empty;
    std::uint32_t obligationCount = 0;
    llvm::SmallVector<SyncDirectRepairCandidate, 8> candidates;
    llvm::SmallVector<SyncObligationId, 8> uncoveredObligations;
    llvm::SmallVector<SyncDirectRepairPlanRejection, 8> rejections;

    bool isComplete() const
    {
        return status == SyncDirectRepairPlanStatus::Empty || status == SyncDirectRepairPlanStatus::Ready;
    }
};

FailureOr<SyncDirectRepairPlan> buildDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncResidualObligation> obligations, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult verifyDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
LogicalResult applyDirectRepairCandidates(
    const SyncDirectRepairPlan& plan, llvm::ArrayRef<SyncResidualObligation> obligations,
    llvm::ArrayRef<SyncDirectCandidateId> selected, SyncSelectedWorld& world);
LogicalResult allocateDirectRepairEvents(
    const StructuredSyncIR& schedule, SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult allocateDirectRepairEvents(
    const ProtocolSyncTarget& target, llvm::ArrayRef<SyncEventReservation> reservations,
    SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics = nullptr);

/// Low-level emitter for a function inside caller-owned disposable staging IR.
/// The caller must verify and discard the complete staging module on failure.
LogicalResult materializeDirectRepairPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncDirectRepairPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);

LogicalResult verifyDirectRepairMaterialization(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncResidualObligation> obligations, func::FuncOp clone, const IRMapping& mapping,
    const SyncDirectRepairPlan& plan, ProtocolSyncStatistics* statistics = nullptr);
/// Clone the containing module and commit this function's body only after the
/// plan, concrete placement, and staged module all verify successfully.
LogicalResult materializeAndVerifyDirectRepairPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
/// Mutate and verify a function inside a disposable whole-module staging
/// clone. This helper performs no rollback; its caller must discard the entire
/// staging module on failure.
LogicalResult materializeAndVerifyDirectRepairPlanInDisposableModule(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    llvm::ArrayRef<SyncResidualObligation> obligations, const SyncDirectRepairPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);

void printDirectRepairPlan(func::FuncOp function, const SyncDirectRepairPlan& plan, llvm::raw_ostream& output);
llvm::StringRef stringifySyncDirectRepairKind(SyncDirectRepairKind kind);
llvm::StringRef stringifySyncDirectRepairPlanStatus(SyncDirectRepairPlanStatus status);
llvm::StringRef stringifySyncDirectRepairRejection(SyncDirectRepairRejection rejection);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_DIRECTREPAIR_H
