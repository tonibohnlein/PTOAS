// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- CanonicalSyncAnalysis.h - Lean MLIR graph adapter ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

enum class CanonicalSyncGmAliasPolicy : std::uint8_t {
  MayAlias,
  DistinctArgumentsNoAlias,
  /// Caller-provided contract that every pair of GM accesses is disjoint,
  /// including accesses rooted at the same argument and across iterations.
  AllAccessesNoAlias,
};

constexpr std::size_t kCanonicalSyncMaximumPeriodicRecurrenceStates = 16;
constexpr std::size_t kCanonicalSyncMaximumRecurrenceWitnessStates = 1U << 18;

struct CanonicalSyncAnalysisOptions {
  CanonicalSyncGmAliasPolicy gmAliasPolicy =
      CanonicalSyncGmAliasPolicy::MayAlias;
  std::size_t maximumNodes = 1U << 16;
  std::size_t maximumScopes = 1U << 14;
  std::size_t maximumControls = 1U << 14;
  std::size_t maximumStorageAccesses = 1U << 20;
  /// Maximum undirected node-conflict entries retained by the storage index.
  std::size_t maximumStorageConflictEdges = 1U << 20;
  std::size_t maximumPairInspections = 1U << 24;
  /// Maximum reachable joint states retained while correlating periodic
  /// controls for one loop recurrence. Exceeding this correctness-critical
  /// bound fails analysis before the graph is frozen.
  std::size_t maximumPeriodicRecurrenceStates =
      kCanonicalSyncMaximumPeriodicRecurrenceStates;
  /// Maximum compact witness/phase records retained while finding the first
  /// recurrence edge for each reachable phase class of one node pair.
  std::size_t maximumRecurrenceWitnessStates =
      kCanonicalSyncMaximumRecurrenceWitnessStates;
  /// Basic ownership discovery is a bounded optional synthesis analysis. If
  /// any limit is reached, discovery stops without retaining partial
  /// certificates; subsequent catalog construction either proves a cover
  /// from the remaining mechanisms or fails closed.
  std::size_t maximumBasicOwnershipInspections = 1U << 28;
  std::size_t maximumBasicOwnershipCertificates = 1U << 10;
  std::size_t maximumBasicOwnershipSlots = 1U << 14;
  std::size_t maximumBasicOwnershipPaths = 1U << 14;
  std::size_t maximumBasicOwnershipUses = 1U << 16;
  std::size_t maximumBasicOwnershipNodeReferences = 1U << 20;
  std::size_t maximumBasicOwnershipAccessIncidences = 1U << 20;
};

constexpr std::size_t kCanonicalSyncBasicOwnershipKindCount = 3;

struct CanonicalSyncOwnershipDiscoveryStatistics {
  std::size_t inspections = 0;
  std::array<std::size_t, kCanonicalSyncBasicOwnershipKindCount>
      certificatesByKind{};
  std::size_t slots = 0;
  std::size_t paths = 0;
  std::size_t uses = 0;
  std::size_t nodeReferences = 0;
  std::size_t accessIncidences = 0;
  bool truncated = false;
};

struct CanonicalSyncNodeBinding {
  Operation *operation = nullptr;
  int macroPhase = -1;
};

struct CanonicalSyncScopeBinding {
  Operation *owner = nullptr;
  Region *region = nullptr;
};

struct CanonicalSyncControlBinding {
  Operation *owner = nullptr;
};

using CanonicalSyncEventReservations =
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<unsigned>>;

struct CanonicalSyncTargetCapabilities {
  /// An A3 PIPE_MTE1 set issued after the final producer of an exact L0
  /// ownership use certifies completion of every earlier MTE1 producer in
  /// that use. The verified recipe may therefore use one ready event lane
  /// instead of one lane per producer.
  bool mte1L0ReadySetCompletesPrefix = false;

  /// An A3 PIPE_M set at an exhaustive branch join releases the exact L0
  /// operand slots consumed by whichever branch alternative executed. This
  /// is a narrow ownership-lifecycle contract, not generic PIPE_M completion
  /// ordering.
  bool mL0AlternativeJoinSetCompletes = false;

  /// A PIPE_MTE1 set issued after a structured scope certifies completion of
  /// every earlier MTE1 operation issued by that scope. This is target
  /// evidence, not a consequence of ordinary MTE1 issue order.
  bool mte1ScopeExitSetCompletesPrefix = false;

  /// An A3 PIPE_M -> PIPE_FIX event at the accumulator boundary orders the
  /// completed accumulator result before the single FIX consumer. This is a
  /// narrow target contract for the verified accumulator lifecycle; it does
  /// not make PIPE_M a generic direct-completion source.
  bool mToFixAccumulatorBoundaryCompletes = false;
};

/// One authoritative synchronization graph plus the minimal MLIR side tables
/// needed to materialize graph anchors. The side tables contain no copied
/// dependency, mechanism, cost, or selection state. Their raw MLIR bindings
/// remain valid only while the function's operation and region structure is
/// unchanged.
class CanonicalSyncProgram {
public:
  CanonicalSyncProgram() = default;
  CanonicalSyncProgram(CanonicalSyncProgram &&) = default;
  CanonicalSyncProgram &operator=(CanonicalSyncProgram &&) = default;
  CanonicalSyncProgram(const CanonicalSyncProgram &) = delete;
  CanonicalSyncProgram &operator=(const CanonicalSyncProgram &) = delete;
  CanonicalSyncProgram(
      func::FuncOp function, SyncCoverGraph graph,
      std::vector<CanonicalSyncNodeBinding> nodeBindings,
      std::vector<CanonicalSyncScopeBinding> scopeBindings,
      std::vector<CanonicalSyncControlBinding> controlBindings,
      std::vector<AddressSpace> storageSpaces,
      CanonicalSyncTargetCapabilities targetCapabilities,
      CanonicalSyncOwnershipDiscoveryStatistics ownershipDiscoveryStatistics,
      CanonicalSyncEventReservations eventReservations)
      : function_(function), graph_(std::move(graph)),
        nodeBindings_(std::move(nodeBindings)),
        scopeBindings_(std::move(scopeBindings)),
        controlBindings_(std::move(controlBindings)),
        storageSpaces_(std::move(storageSpaces)),
        targetCapabilities_(targetCapabilities),
        ownershipDiscoveryStatistics_(ownershipDiscoveryStatistics),
        eventReservations_(std::move(eventReservations)) {}

  SyncCoverGraph &getGraph() { return graph_; }
  const SyncCoverGraph &getGraph() const { return graph_; }
  func::FuncOp getFunction() const { return function_; }
  const std::vector<CanonicalSyncNodeBinding> &getNodeBindings() const {
    return nodeBindings_;
  }
  const std::vector<CanonicalSyncScopeBinding> &getScopeBindings() const {
    return scopeBindings_;
  }
  const std::vector<CanonicalSyncControlBinding> &getControlBindings() const {
    return controlBindings_;
  }
  const std::vector<AddressSpace> &getStorageSpaces() const {
    return storageSpaces_;
  }
  const CanonicalSyncTargetCapabilities &getTargetCapabilities() const {
    return targetCapabilities_;
  }
  const CanonicalSyncOwnershipDiscoveryStatistics &
  getOwnershipDiscoveryStatistics() const {
    return ownershipDiscoveryStatistics_;
  }
  const CanonicalSyncEventReservations &getEventReservations() const {
    return eventReservations_;
  }

private:
  func::FuncOp function_;
  SyncCoverGraph graph_;
  std::vector<CanonicalSyncNodeBinding> nodeBindings_;
  std::vector<CanonicalSyncScopeBinding> scopeBindings_;
  std::vector<CanonicalSyncControlBinding> controlBindings_;
  std::vector<AddressSpace> storageSpaces_;
  CanonicalSyncTargetCapabilities targetCapabilities_;
  CanonicalSyncOwnershipDiscoveryStatistics ownershipDiscoveryStatistics_;
  CanonicalSyncEventReservations eventReservations_;
};

FailureOr<CanonicalSyncProgram>
buildCanonicalSyncProgram(func::FuncOp function,
                          const CanonicalSyncAnalysisOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCANALYSIS_H
