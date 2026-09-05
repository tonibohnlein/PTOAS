// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- OneShotPublish.h - Channel-scoped one-shot protocols ---*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPUBLISH_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPUBLISH_H

#include "PTO/Transforms/ProtocolSync/ProtocolSyncTarget.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "mlir/IR/IRMapping.h"

#include <cstdint>
#include <optional>

namespace mlir::pto::protocol_sync {

using SyncOneShotPublishId = std::uint32_t;

enum class SyncOneShotPublishKind : std::uint8_t {
    IntrinsicOrder,
    PipeBarrier,
    DirectedEvent,
};

/// One indivisible publication/acquisition recipe. Channels are grouped only
/// when they have the same exact physical source and target frontiers.
struct SyncOneShotPublishCandidate {
    SyncOneShotPublishId id = kInvalidSyncId;
    SyncOneShotPublishKind kind = SyncOneShotPublishKind::IntrinsicOrder;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE sourcePipe = PIPE::PIPE_UNASSIGNED;
    PIPE targetPipe = PIPE::PIPE_UNASSIGNED;
    SyncPhaseId sourcePhase = kInvalidSyncId;
    SyncPhaseId targetPhase = kInvalidSyncId;
    Operation* sourceOperation = nullptr;
    Operation* targetOperation = nullptr;
    llvm::SmallVector<SyncChannelId, 2> channels;
    llvm::SmallVector<SyncGenerationId, 2> generations;
    std::optional<unsigned> eventId;
};

struct SyncOneShotPublishPlan {
    llvm::SmallVector<SyncOneShotPublishCandidate, 4> candidates;
};

FailureOr<SyncOneShotPublishPlan> buildOneShotPublishCandidates(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels);

LogicalResult appendOneShotPublishSelectedWorld(
    const SyncOneShotPublishPlan& plan, const ChannelAnalysisResult& channels, SyncSelectedWorld& world);

LogicalResult verifyOneShotPublishPlan(
    const StructuredSyncIR& schedule, const PipelineStageAnalysisResult& stages,
    const StorageTimelineAnalysisResult& timelines, const ChannelAnalysisResult& channels,
    const SyncOneShotPublishPlan& plan);

LogicalResult materializeOneShotPublishPlan(
    func::FuncOp clone, const IRMapping& mapping, const SyncOneShotPublishPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);

LogicalResult verifyOneShotPublishMaterialization(
    const StructuredSyncIR& schedule, func::FuncOp clone, const IRMapping& mapping, const SyncOneShotPublishPlan& plan,
    ProtocolSyncStatistics* statistics = nullptr);

llvm::StringRef stringifySyncOneShotPublishKind(SyncOneShotPublishKind kind);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_ONESHOTPUBLISH_H
