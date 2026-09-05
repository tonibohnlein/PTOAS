// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredFrontier.h - Balanced path-local completion interfaces ----===//
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STRUCTUREDFRONTIER_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STRUCTUREDFRONTIER_H
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/IR/IRMapping.h"

namespace mlir::pto::protocol_sync {

struct SyncStructuredPhaseRepair {
    SyncPhaseId phase = kInvalidSyncId;
    Operation* operation = nullptr;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    unsigned acquireId = 0;
    unsigned releaseId = 0;
};
struct SyncStructuredFrontierPlan {
    llvm::SmallVector<SyncStructuredPhaseRepair, 16> phases;
    llvm::SmallVector<SyncResidualObligation, 16> requirements;
};

bool supportsStructuredFrontier(const StructuredSyncIR& schedule, bool allowFixedSync);
SyncParticipationRelation classifySyncParticipation(
    const SyncPhase& source, const SyncPhase& target, const SyncIterationRelation& relation);
SyncIterationRelation structuredIterationRelation(const SyncPhase& source, const SyncPhase& target);
bool structuredFrontierOrders(
    const StructuredSyncIR& schedule, ArrayRef<SyncPhaseId> acknowledged, SyncPhaseId source, SyncPhaseId target,
    const SyncIterationRelation& relation);
FailureOr<std::optional<SyncStructuredFrontierPlan>> buildStructuredFrontierPlan(const StructuredSyncIR& schedule);
LogicalResult materializeStructuredFrontier(
    func::FuncOp clone, const IRMapping& mapping, const SyncStructuredFrontierPlan& plan);
/// Reconstruct per-phase balanced entry/exit handoffs, including token lifetimes.
/// No selected recipe, planner tags, or requirement coverage is consulted.
FailureOr<SyncSelectedWorld> reconstructStructuredFrontier(const StructuredSyncIR& schedule);
} // namespace mlir::pto::protocol_sync
#endif
