// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- OneShotProtocol.h - Atomic Checkpoint D protocols ---------*- C++ -*-===//
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPROTOCOL_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPROTOCOL_H

#include "PTO/Transforms/ProtocolSync/ChannelProtocolIR.h"
#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

using SyncOneShotProtocolId = std::uint32_t;

enum class SyncOneShotProtocolKind : std::uint8_t {
    IntrinsicOrder,
    PipeBarrier,
    DirectedEvent,
};
enum class SyncOneShotPlanStatus : std::uint8_t { Empty, Ready, Unsupported };
enum class SyncOneShotRejection : std::uint8_t {
    None,
    UnsupportedTarget,
    ExistingSynchronization,
    ScheduleFailure,
    SemanticAction,
    UnsupportedControlFlow,
    UnsupportedStageShape,
    MixedPhysicalCores,
    MixedPhysicalSections,
    UnorderedEndpoints,
    UnsupportedBarrier,
    UnsupportedEventDirection,
    UnsupportedVisibility,
    NonOneShotChannel,
    UnverifiedChannel,
    IncompleteChannelSet,
    EventCapacity,
    InternalInvariant,
};

/// One atomic completion boundary between adjacent exact physical phases.
/// IntrinsicOrder has no emitted action; it records the documented same-scalar
/// completion edge in the concrete plan.
struct SyncOneShotProtocol {
    SyncOneShotProtocolId id = kInvalidSyncId;
    SyncOneShotProtocolKind kind = SyncOneShotProtocolKind::PipeBarrier;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    SyncPhaseId sourcePhase = kInvalidSyncId;
    SyncPhaseId targetPhase = kInvalidSyncId;
    Operation* sourceOperation = nullptr;
    Operation* targetOperation = nullptr;
    llvm::SmallVector<SyncChannelId, 2> channels;
    std::optional<unsigned> eventId;
};

struct SyncOneShotPlanRejection {
    SyncChannelId channel = kInvalidSyncId;
    SyncOneShotRejection reason = SyncOneShotRejection::None;
    std::string detail;
};

/// Checkpoint D intentionally accepts only a completely linear, once-only,
/// single-core physical phase chain. Every adjacent pair is completion ordered,
/// so the concrete plan is obligation-complete without relying on the future
/// residual-obligation interpreter.
struct SyncOneShotPlan {
    SyncOneShotPlanStatus status = SyncOneShotPlanStatus::Empty;
    SyncPhysicalCore functionCore = SyncPhysicalCore::Unknown;

    /// Exact lexical order of physical phases certified by the planner.
    llvm::SmallVector<SyncPhaseId, 8> phaseOrder;

    /// Tail drain is mandatory whenever phaseOrder is non-empty. When
    /// tailSectionOperation is set, PIPE_ALL is emitted before that section's
    /// terminator. Otherwise it is emitted before the function return.
    bool emitTailBarrier = false;
    Operation* tailSectionOperation = nullptr;

    llvm::SmallVector<SyncOneShotProtocol, 8> protocols;
    llvm::SmallVector<SyncOneShotPlanRejection, 4> rejections;
};

FailureOr<SyncOneShotPlan> buildOneShotProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    ProtocolSyncStatistics* statistics = nullptr);
LogicalResult allocateOneShotProtocolEvents(
    const StructuredSyncIR& schedule, SyncOneShotPlan& plan, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult materializeAndVerifyOneShotProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, const SyncOneShotPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
void printOneShotProtocolPlan(func::FuncOp function, const SyncOneShotPlan& plan, llvm::raw_ostream& output);

llvm::StringRef stringifySyncOneShotProtocolKind(SyncOneShotProtocolKind kind);
llvm::StringRef stringifySyncOneShotPlanStatus(SyncOneShotPlanStatus status);
llvm::StringRef stringifySyncOneShotRejection(SyncOneShotRejection rejection);

} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPROTOCOL_H
