// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- RecurringEventLifetime.h - Unconditional event-action induction -----===//
#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_RECURRINGEVENTLIFETIME_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_RECURRINGEVENTLIFETIME_H

#include "llvm/ADT/ArrayRef.h"
#include <cstdint>

namespace mlir::pto::protocol_sync {

/// One complete logical channel, with a distinct key from all other channels.
/// Orders describe actual event instructions, not physical phase completion.
/// Each lane executes its actions in increasing order once per iteration.
struct SyncRecurringEventChannel {
    unsigned setLane = 0;
    unsigned waitLane = 0;
    unsigned setOrder = 0;
    unsigned waitOrder = 0;
    /// set(i) supplies wait(i + distance); only zero and one are supported.
    unsigned distance = 0;
    bool primed = false;
    bool drained = false;
};

enum class SyncRecurringEventStatus : std::uint8_t {
    Proven,
    InvalidContract,
    ZeroDistanceCycle,
    UnprovedRearm,
    AnalysisLimit,
};

struct SyncRecurringEventProof {
    SyncRecurringEventStatus status = SyncRecurringEventStatus::InvalidContract;
    unsigned channel = 0;
};

/// Prove arbitrary-trip token safety for one unconditional, fixed body.
/// Distance-one channels require one prime before all body actions on their
/// source lane and one drain after all body actions on their destination lane;
/// distance-zero channels start and finish empty. The caller must reconstruct
/// those boundaries, unique concrete keys, target directions, and participation.
/// Every lane must have the same trip count, and all blocking synchronization
/// must be accounted for: no hidden macro, queue, collective, or fixed supply.
/// Assumes eventual instruction completion; proves no event-induced deadlock
/// or rearm, not memory coverage, visibility, or cross-channel ID reuse.
/// An analysis limit supplies no proof and is never resource scarcity.
SyncRecurringEventProof proveRecurringEventLifetimes(llvm::ArrayRef<SyncRecurringEventChannel> channels);

} // namespace mlir::pto::protocol_sync
#endif // PTO_TRANSFORMS_PROTOCOLSYNC_RECURRINGEVENTLIFETIME_H
