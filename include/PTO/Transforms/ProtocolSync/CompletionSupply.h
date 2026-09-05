// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CompletionSupply.h - Explicit completion-only frontier summaries ----===//
// Logical supply, not hardware certification: callers must validate selected
// mechanisms or reconstruct concrete synchronization before trusting the input.
// This first version summarizes only explicit forward completion paths in one
// unguarded, non-recurring block. It invents neither pipe issue ordering nor
// visibility. The independent concrete scoreboard deliberately does not use it.

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_COMPLETIONSUPPLY_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_COMPLETIONSUPPLY_H

#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "llvm/ADT/BitVector.h"

#include <cstddef>

namespace mlir::pto::protocol_sync {

enum class SyncCompletionSupplyStatus { Unsupported, Complete, LimitExceeded };

struct SyncCompletionFrontier {
    SyncPhaseId target = kInvalidSyncId;
    SyncProgramPointId before = kInvalidSyncId;
    SyncRegionId region = kInvalidSyncId;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE lane = PIPE::PIPE_UNASSIGNED;
    /// Completed phase identities, not issue positions or contiguous prefixes.
    llvm::BitVector completed;
};

class SyncCompletionSupply {
public:
    SyncCompletionSupplyStatus getStatus() const { return status; }
    const SyncCompletionFrontier* getBefore(SyncPhaseId target) const;
    bool covers(SyncPhaseId source, SyncPhaseId target, const SyncIterationRelation& relation) const;
    /// Indices into the input world's completions, in source-to-target order.
    /// Empty means no nonempty certified path; self-reachability is not supply.
    llvm::SmallVector<unsigned, 4> witness(SyncPhaseId source, SyncPhaseId target) const;

private:
    friend FailureOr<SyncCompletionSupply> buildCompletionSupply(
        const StructuredSyncIR&, const SyncSelectedWorld&, std::size_t);
    SyncCompletionSupplyStatus status = SyncCompletionSupplyStatus::Unsupported;
    llvm::SmallVector<SyncCompletionFrontier, 16> frontiers;
    llvm::SmallVector<SyncSelectedCompletion, 16> edges;
    llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 16> incoming;
};

/// No partial summaries escape a limit. Unsupported/limited results leave the
/// established interpreter in charge. Requirements and placement are unchanged.
FailureOr<SyncCompletionSupply> buildCompletionSupply(
    const StructuredSyncIR& schedule, const SyncSelectedWorld& world, std::size_t maximumMemberships = 1048576);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_COMPLETIONSUPPLY_H
