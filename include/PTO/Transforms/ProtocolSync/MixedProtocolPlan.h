// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- MixedProtocolPlan.h - Complete protocol/direct selection -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_MIXEDPROTOCOLPLAN_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_MIXEDPROTOCOLPLAN_H

#include "PTO/Transforms/ProtocolSync/DirectRepair.h"
#include "PTO/Transforms/ProtocolSync/OneShotPublish.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

enum class SyncMixedPlanStatus : std::uint8_t {
    Empty,
    Ready,
    Unsupported,
    ResourceInfeasible,
};

enum class SyncMixedPlanRejection : std::uint8_t {
    None,
    UnsupportedTarget,
    IncompleteDirectRepair,
    EventCapacity,
    InternalInvariant,
};

struct SyncMixedPlanFailure {
    SyncMixedPlanRejection reason = SyncMixedPlanRejection::None;
    std::string detail;
};

enum class SyncMixedWorldKind : std::uint8_t {
    DirectOnly,
    OneShotPublish,
    ReadyRelease,
    CombinedProtocols,
};

struct SyncMixedWorldCost {
    std::uint64_t generatedEventPairs = 0;
    std::uint64_t targetedBarriers = 0;
    std::uint64_t fixedExitDrains = 0;
    std::uint64_t eventPressure = 0;
    std::uint64_t staticActions = 0;
};

/// A complete Checkpoint-F world. Protocol plans and direct recipes remain
/// indivisible candidates: selection and reverse deletion never expose their
/// individual materialized actions.
struct SyncMixedProtocolPlan {
    SyncMixedPlanStatus status = SyncMixedPlanStatus::Empty;
    std::optional<SyncOneShotPublishPlan> oneShot;
    std::optional<SyncReadyReleasePlan> readyRelease;
    llvm::SmallVector<SyncResidualObligation, 16> directObligations;
    SyncDirectRepairPlan directRepair;
    SyncSelectedWorld selectedWorld;
    std::uint64_t initialResidualCount = 0;
    std::uint64_t candidateCountBeforeDeletion = 0;
    std::uint64_t reverseDeletionAttempts = 0;
    std::uint64_t reverseDeletionRemoved = 0;
    std::uint64_t completeWorldsAttempted = 0;
    std::uint64_t completeWorldsFeasible = 0;
    SyncMixedWorldKind selectedWorldKind = SyncMixedWorldKind::DirectOnly;
    SyncMixedWorldCost selectedCost;
    bool protocolsEnabled = true;
    llvm::SmallVector<SyncMixedPlanFailure, 2> failures;

    bool hasProtocol() const { return oneShot.has_value() || readyRelease.has_value(); }
    bool isComplete() const { return status == SyncMixedPlanStatus::Empty || status == SyncMixedPlanStatus::Ready; }
};

FailureOr<SyncMixedProtocolPlan> buildMixedProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, bool enableProtocols = true,
    ProtocolSyncStatistics* statistics = nullptr);
/// Select and reverse-delete an individually well-formed, possibly
/// overcomplete candidate pool, then record the exact complete world. Event
/// IDs must not have been allocated yet.
LogicalResult selectMixedProtocolCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels, SyncMixedProtocolPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);
LogicalResult verifyMixedProtocolPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics = nullptr);
LogicalResult allocateMixedProtocolEvents(
    const StructuredSyncIR& schedule, SyncMixedProtocolPlan& plan, ProtocolSyncStatistics* statistics = nullptr);
/// Materialize and independently verify every selected atomic candidate in a
/// disposable whole-module clone. The caller owns module-level rollback.
LogicalResult materializeAndVerifyMixedProtocolPlanInDisposableModule(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncMixedProtocolPlan& plan, const SyncSemanticContext& context,
    ProtocolSyncStatistics* statistics = nullptr);

void printMixedProtocolPlan(func::FuncOp function, const SyncMixedProtocolPlan& plan, llvm::raw_ostream& output);
llvm::StringRef stringifySyncMixedPlanStatus(SyncMixedPlanStatus status);
llvm::StringRef stringifySyncMixedPlanRejection(SyncMixedPlanRejection rejection);
llvm::StringRef stringifySyncMixedWorldKind(SyncMixedWorldKind kind);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_MIXEDPROTOCOLPLAN_H
