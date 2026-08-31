// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverProtocol.h - Event lifecycle certificates -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverProtocolChannelId = std::size_t;

/// Target-owned facts consumed by the target-neutral lifecycle verifier. The
/// provider must populate the exact directed HardEvent table and the usable ID
/// set from one versioned device specification.
struct SyncCoverProtocolTargetContract {
  struct EventCapability {
    std::uint32_t sourceResource = 0;
    std::uint32_t targetResource = 0;
    SyncCoverOrderingRequirementMask suppliedRequirements = 0;

    bool operator<(const EventCapability &other) const;
    bool operator==(const EventCapability &other) const;
  };

  struct RearmFact {
    std::uint32_t evidence = 0;
    std::uint32_t fromWaitResource = 0;
    SyncCoverAnchor fromWaitAnchor;
    SyncCoverGuard fromWaitGuard;
    std::uint32_t toSetResource = 0;
    SyncCoverAnchor toSetAnchor;
    SyncCoverGuard toSetGuard;
    SyncCoverScopeId loopScope = 0;
    unsigned iterationDistance = 0;
    std::size_t width = 0;

    bool operator<(const RearmFact &other) const;
    bool operator==(const RearmFact &other) const;
  };

  /// Every entry is copied from one versioned, core-specific device profile.
  /// The requirement mask prevents an ordinary HardEvent from being promoted
  /// into cache visibility or a hardware-special ordering certificate.
  std::vector<EventCapability> eventCapabilities;
  std::vector<unsigned> compilerUsableEventIds;
  /// Exact provider-owned Wait-to-Set facts. An evidence number alone never
  /// certifies a different resource, guard, anchor, scope, width, or distance.
  std::vector<RearmFact> certifiedRearmFacts;

  bool supportsEvent(std::uint32_t source, std::uint32_t target,
                     SyncCoverOrderingRequirementMask requirements) const;
  const RearmFact *findRearmFact(std::uint32_t evidence) const;
};

enum class SyncCoverEventProtocolKind : std::uint8_t {
  SingleShot,
  ProvenNoOverlap,
  RoundTrip,
  RotatingLanes,
};

/// SameIteration pairs one body Set with one body Wait in the same dynamic
/// iteration. LoopCarry primes each lane at entry, consumes before reuse,
/// produces after release, and drains every lane at exit.
enum class SyncCoverEventChannelFlow : std::uint8_t {
  SingleShot,
  SameIteration,
  LoopCarry,
};

struct SyncCoverProtocolLoopSchedule {
  SyncCoverScopeId scope = 0;
  bool mayExecuteZeroTimes = true;
  /// If present, phases are the graph-owned reachable phases of this control.
  /// If absent, the loop has one unconditional phase. Callers cannot invent a
  /// transition relation or phase guards.
  std::optional<SyncCoverControlId> phaseControl;
  /// Lane selected by each declared graph phase. The vector has one element
  /// for an unconditional loop or exactly matches the authoritative relation.
  std::vector<std::size_t> laneByPhase;
};

struct SyncCoverEventChannel {
  SyncCoverProtocolChannelId id = 0;
  SyncCoverEventChannelFlow flow = SyncCoverEventChannelFlow::SingleShot;
  SyncCoverCutPoint set;
  SyncCoverCutPoint wait;
  std::size_t width = 1;
  /// SameIteration and SingleShot use zero. LoopCarry uses the exact dynamic
  /// iteration displacement between a body Set and its consuming body Wait.
  unsigned distance = 0;
  SyncCoverOrderingRequirementMask suppliedRequirements = 0;
  /// Empty means every reachable phase. Otherwise the sorted unique phase set
  /// controls both body actions.
  std::vector<std::size_t> activePhases;
  bool exportsCompletionAtExit = false;
};

/// An externally certified order can rearm a one-way channel. It is not
/// inferred from a demand that still needs synchronization. Nonzero evidence
/// identifies the target/provider contract that established the relation.
struct SyncCoverProtocolRearmProof {
  SyncCoverProtocolChannelId fromWaitChannel = 0;
  SyncCoverProtocolChannelId toSetChannel = 0;
  unsigned iterationDistance = 0;
  std::uint32_t evidence = 0;
};

struct SyncCoverEventProtocol {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverEventProtocolKind kind = SyncCoverEventProtocolKind::SingleShot;
  std::optional<SyncCoverProtocolLoopSchedule> loop;
  std::vector<SyncCoverEventChannel> channels;
  std::vector<SyncCoverProtocolRearmProof> rearmProofs;
};

struct SyncCoverProtocolLimits {
  std::size_t maximumGraphNodes = 1U << 16;
  std::size_t maximumGraphEdges = 1U << 18;
  std::size_t maximumGraphDemands = 1U << 16;
  std::size_t maximumGraphScopes = 1U << 16;
  std::size_t maximumGraphRegions = 1U << 17;
  std::size_t maximumGraphControls = 1U << 16;
  std::size_t maximumGraphStorageDomains = 1U << 16;
  std::size_t maximumGraphStorageAccesses = 1U << 18;
  std::size_t maximumGraphStorageWitnesses = 1U << 18;
  std::size_t maximumTargetCapabilities = 256;
  std::size_t maximumTargetEventIds = 64;
  std::size_t maximumTargetRearmFacts = 1U << 12;
  std::size_t maximumChannels = 256;
  std::size_t maximumChannelLaneIncidences = 1U << 16;
  std::size_t maximumProtocols = 1U << 14;
  std::size_t maximumWorlds = 256;
  std::size_t maximumWorldMechanismIncidences = 1U << 18;
  std::size_t maximumResultRows = 1U << 16;
  std::size_t maximumResultWords = 1U << 22;
  std::size_t maximumGuardLiterals = 1U << 18;
  std::size_t maximumPhaseIncidences = 1U << 18;
  std::size_t maximumReachablePhases = 1U << 12;
  std::size_t maximumTripCounts = 1U << 12;
  std::size_t maximumDynamicActions = 1U << 14;
  std::size_t maximumTotalDynamicActions = 1U << 20;
  std::size_t maximumLaneInitializationWork = 1U << 20;
  std::size_t maximumAutomatonEdges = 1U << 18;
  std::size_t maximumRearmProofs = 1U << 12;
  std::size_t maximumRearmProofLaneIncidences = 1U << 16;
  std::size_t maximumRearmLookupWork = 1U << 20;
  std::size_t maximumRearmQueries = 1U << 18;
  std::size_t maximumReachabilityWords = 1U << 22;
  std::size_t maximumReachabilityWork = 1U << 24;
  std::size_t maximumCoverageStates = 1U << 22;
  std::size_t maximumCoverageTransitions = 1U << 24;
  std::size_t maximumExitExports = 1U << 16;
  std::size_t maximumExitExportGuardLiterals = 1U << 18;
  std::size_t maximumLifecycleSccs = 1U << 12;
  std::size_t maximumLifecycleVertices = 1U << 16;
  std::size_t maximumLifecycleEdges = 1U << 18;
  std::size_t maximumStorageWitnessIncidences = 1U << 20;
  std::size_t maximumLifecycleDomainIncidences = 1U << 20;
  std::size_t maximumReservations = 1U << 14;
  std::size_t maximumReservationDomains = 1U << 12;
  std::size_t maximumReservationIdIncidences = 1U << 16;
  std::size_t maximumChannelRequests = 1U << 16;
  std::size_t maximumAllocatedEventIds = 1U << 16;
};

enum class SyncCoverProtocolError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidTargetContract,
  InvalidProtocol,
  InvalidTokenLifecycle,
  LimitExceeded,
  WorkLimitExceeded,
  ResourceInfeasible,
};

struct SyncCoverProtocolExitExport {
  SyncCoverProtocolChannelId channel = 0;
  SyncCoverGuard guard;
  bool availableOnZeroTrip = false;
  bool availableOnNonzeroTrip = false;
};

struct SyncCoverProtocolStatistics {
  std::size_t reachablePhases = 0;
  std::size_t tripCountsChecked = 0;
  std::size_t maximumDynamicActions = 0;
  std::size_t automatonEdges = 0;
  std::size_t rearmQueries = 0;
  std::size_t rearmProofLaneIncidences = 0;
  std::size_t rearmLookupWork = 0;
  std::size_t totalDynamicActions = 0;
  std::size_t laneInitializationWork = 0;
  std::size_t reachabilityWork = 0;
  std::size_t coverageTransitions = 0;
};

struct SyncCoverProtocolVerificationResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  SyncCoverProtocolStatistics statistics;
  std::vector<SyncCoverProtocolExitExport> exitExports;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolVerificationResult verifySyncCoverEventProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// One exact-world coverage result for lifecycle-verified direct protocol
/// rectangles. This does not trust or import legacy supply bindings.
struct SyncCoverProtocolCoverageResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverDemandSet> coveredByWorld;
  SyncCoverProtocolStatistics statistics;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolCoverageResult computeSyncCoverProtocolExactWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// A generic storage-lifecycle SCC. It is a proposal/certificate input, not a
/// selectable mechanism: later synthesis must still build and verify a full
/// physical event recipe.
struct SyncCoverLifecycleScc {
  SyncCoverScopeId loopScope = 0;
  std::vector<SyncCoverNodeId> nodes;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverStorageDomainId> storageDomains;
  unsigned maximumDistance = 0;
};

struct SyncCoverLifecycleSccResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverLifecycleScc> components;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverLifecycleSccResult discoverSyncCoverLifecycleSccs(
    const SyncCoverGraph &graph, SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

struct SyncCoverProtocolEventReservation {
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::vector<unsigned> eventIds;
};

struct SyncCoverProtocolChannelAllocation {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverProtocolChannelId channel = 0;
  std::vector<unsigned> eventIds;
};

struct SyncCoverProtocolAllocationResult {
  SyncCoverProtocolError error = SyncCoverProtocolError::None;
  std::vector<SyncCoverProtocolChannelAllocation> channels;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverProtocolError::None;
  }
};

SyncCoverProtocolAllocationResult allocateSyncCoverProtocolEventIds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverProtocolEventReservation> &reservations = {},
    SyncCoverProtocolLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOL_H
