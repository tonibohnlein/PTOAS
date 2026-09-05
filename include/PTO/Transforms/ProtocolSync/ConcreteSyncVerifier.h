// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ConcreteSyncVerifier.h - Verify emitted synchronization -*- C++ -*-===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_CONCRETESYNCVERIFIER_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_CONCRETESYNCVERIFIER_H

#include "PTO/Transforms/ProtocolSync/SyncSemantics.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::pto::protocol_sync {

/// Import only the fixed synchronization that the concrete interpreter can
/// certify. Failure leaves opaque-effect protection in place. Occupied IDs are
/// conservatively reserved for planning; this does not enable new ID reuse.
FailureOr<SyncSelectedWorld> reconstructFixedSyncSupply(
    const StructuredSyncIR& schedule, llvm::SmallVectorImpl<SyncEventReservation>* occupiedEvents = nullptr);

/// Additional instruction-ordered check for ordinary, non-recurring vector UB
/// accesses within one block. Reconstructs event payloads and per-pipe knowledge
/// directly, without selected completions, atoms, timelines, or requirements.
/// Other domains and region boundaries still require the existing verifier;
/// success here alone is not a complete function certificate.
LogicalResult verifyConcreteLocalScoreboard(
    const StructuredSyncIR& schedule, SyncAccessId* uncoveredSource = nullptr, SyncAccessId* uncoveredTarget = nullptr);

/// Rebuild provenance as well as the schedule for a mutated or cloned function.
LogicalResult verifyFreshConcreteSyncSemantics(func::FuncOp function, ProtocolSyncStatistics* statistics = nullptr);

/// Rebuild semantic facts from the staged function, reconstruct synchronization
/// effects from concrete PTO operations without consulting planner candidates
/// or diagnostic tags, and require the resulting world to discharge every
/// memory, generation, visibility, SSA, and exit obligation.
/// When requested, report the first failed reconstruction/verification stage.
/// A stage rejection is not a race witness or an obligation coverage percentage.
LogicalResult verifyConcreteSyncSemantics(
    const SyncSemanticContext& context, func::FuncOp function, ProtocolSyncStatistics* statistics = nullptr,
    llvm::StringRef* firstFailedStage = nullptr);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_CONCRETESYNCVERIFIER_H
