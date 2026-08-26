// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSync.h - Canonical synchronization planning ------*- C++
//-*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace mlir {
namespace pto {

enum class CanonicalDependencyKind : std::uint8_t {
  SSA,
  MemoryRAW,
  MemoryWAR,
  MemoryWAW,
  LoopCarriedSSA,
};

enum class CanonicalGMAliasPolicy : std::uint8_t {
  MayAlias,
  DistinctArgumentsNoAlias,
  AllAccessesNoAlias,
};

enum class CanonicalOwnershipKind : std::uint8_t {
  L0Operand,
  L1Tile,
  L0Accumulator,
};

enum class CanonicalOwnershipProtocolKind : std::uint8_t {
  RoundTrip,
  AlternatingPrefetch,
  BoundaryGuardedRoundTrip,
  HierarchicalOuterCarry,
};

enum class CanonicalEventBundleKind : std::uint8_t {
  Standalone,
  Ownership,
  CompositeOwnership,
};

enum class CanonicalSelectionMechanismKind : std::uint8_t {
  Barrier,
  EventBundle,
  SlotProtocol,
};

struct CanonicalPhysicalSlot {
  AddressSpace space = AddressSpace::Zero;
  std::uint64_t address = 0;
  std::uint64_t size = 0;

  bool operator<(const CanonicalPhysicalSlot &other) const {
    return std::tie(space, address, size) <
           std::tie(other.space, other.address, other.size);
  }

  bool operator==(const CanonicalPhysicalSlot &other) const {
    return space == other.space && address == other.address &&
           size == other.size;
  }
};

struct CanonicalMemoryAccess {
  Value base;
  Value root;
  AddressSpace space = AddressSpace::Zero;
  SmallVector<std::uint64_t, 2> addresses;
  std::uint64_t size = 0;
  bool knownPhysical = false;
  bool unknownRange = false;
  bool reads = false;
  bool writes = false;
};

struct CanonicalSyncNode {
  std::size_t id = 0;
  Operation *operation = nullptr;
  PipelineType pipe = PipelineType::PIPE_UNASSIGNED;
  int macroPhase = -1;
  std::size_t order = 0;
  std::uint64_t computeWeight = 0;
  std::uint64_t transferWeight = 0;
  SmallVector<CanonicalMemoryAccess, 4> accesses;
};

enum class CanonicalStorageProvenance : std::uint8_t {
  NotApplicable,
  Complete,
  Incomplete,
};

struct CanonicalMemoryHazardWitness {
  std::size_t sourceAccess = 0;
  std::size_t targetAccess = 0;
  unsigned sourceAddressOrdinal = 0;
  unsigned targetAddressOrdinal = 0;
  std::uint64_t overlapBegin = 0;
  std::uint64_t overlapEnd = 0;

  bool operator<(const CanonicalMemoryHazardWitness &other) const {
    return std::tie(sourceAccess, targetAccess, sourceAddressOrdinal,
                    targetAddressOrdinal, overlapBegin, overlapEnd) <
           std::tie(other.sourceAccess, other.targetAccess,
                    other.sourceAddressOrdinal, other.targetAddressOrdinal,
                    other.overlapBegin, other.overlapEnd);
  }
  bool operator==(const CanonicalMemoryHazardWitness &other) const {
    return sourceAccess == other.sourceAccess &&
           targetAccess == other.targetAccess &&
           sourceAddressOrdinal == other.sourceAddressOrdinal &&
           targetAddressOrdinal == other.targetAddressOrdinal &&
           overlapBegin == other.overlapBegin && overlapEnd == other.overlapEnd;
  }
};

struct CanonicalDependency {
  std::size_t source = 0;
  std::size_t target = 0;
  CanonicalDependencyKind kind = CanonicalDependencyKind::SSA;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
  bool active = true;
  bool possible = true;
  bool retained = true;
  CanonicalStorageProvenance storageProvenance =
      CanonicalStorageProvenance::NotApplicable;
  SmallVector<CanonicalMemoryHazardWitness, 2> storageWitnesses;
};

struct CanonicalAnchor {
  Operation *operation = nullptr;
  bool before = true;
};

struct CanonicalOwnershipLane {
  unsigned id = 0;
  SmallVector<CanonicalPhysicalSlot, 2> slots;
};

struct CanonicalOwnershipUse {
  unsigned lane = 0;
  unsigned producerLane = 0;
  SmallVector<std::size_t, 2> producers;
  SmallVector<std::size_t, 2> consumers;
  CanonicalAnchor writeAcquireAnchor;
  CanonicalAnchor readyAnchor;
  CanonicalAnchor readAcquireAnchor;
  CanonicalAnchor releaseAnchor;
};

struct CanonicalOwnershipPath {
  Region *region = nullptr;
  SmallVector<CanonicalOwnershipUse, 8> uses;
};

struct CanonicalOwnershipCycle {
  std::size_t id = 0;
  CanonicalOwnershipKind kind = CanonicalOwnershipKind::L0Operand;
  CanonicalOwnershipProtocolKind protocol =
      CanonicalOwnershipProtocolKind::RoundTrip;
  Operation *loop = nullptr;
  PipelineType producerPipe = PipelineType::PIPE_UNASSIGNED;
  PipelineType consumerPipe = PipelineType::PIPE_UNASSIGNED;
  SmallVector<CanonicalOwnershipLane, 2> lanes;
  SmallVector<CanonicalOwnershipPath, 2> paths;
  SmallVector<std::size_t, 2> initialProducers;
  CanonicalAnchor initialWriteAcquireAnchor;
  CanonicalAnchor initialReadyAnchor;
  unsigned initialReadyLane = 0;
  SmallVector<unsigned, 2> initiallyFreeLanes;
};

struct CanonicalBarrier {
  std::size_t id = 0;
  PipelineType pipe = PipelineType::PIPE_UNASSIGNED;
  CanonicalAnchor anchor;
  SmallVector<std::size_t, 2> anchorNodes;
  Operation *recurrenceLoop = nullptr;
  std::size_t recurrenceScope = 0;
  SmallVector<std::size_t, 3> requirements;
};

enum class CanonicalEventActionKind : std::uint8_t { Set, Wait };

enum class CanonicalEventActionPhase : std::uint8_t {
  Straight,
  Prime,
  Body,
  Condition,
  Drain,
};

enum class CanonicalEventExecutionGuardKind : std::uint8_t {
  None,
  LoopNonEmpty,
  LoopEmpty,
  NotFirstIteration,
  HasSuccessor,
};

enum class CanonicalEventLaneKind : std::uint8_t { Static, Dynamic, All };

enum class CanonicalEventTraceKind : std::uint8_t {
  Straight,
  Prime,
  Cycle,
  Final,
};

enum class CanonicalOwnershipEventRole : std::uint8_t {
  None,
  Ready,
  Release,
};

struct CanonicalEventLane {
  CanonicalEventLaneKind kind = CanonicalEventLaneKind::Static;
  unsigned index = 0;
  Value selector;
};

struct CanonicalEventExecutionGuard {
  CanonicalEventExecutionGuardKind kind =
      CanonicalEventExecutionGuardKind::None;
  Operation *loop = nullptr;
};

struct CanonicalEventAction {
  CanonicalEventActionKind kind = CanonicalEventActionKind::Set;
  CanonicalEventActionPhase phase = CanonicalEventActionPhase::Straight;
  CanonicalAnchor anchor;
  CanonicalEventLane lane;
  CanonicalEventExecutionGuard guard;
};

struct CanonicalEventCompletion {
  std::size_t source = 0;
  std::size_t target = 0;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
  unsigned setAction = 0;
  unsigned waitAction = 0;
};

struct CanonicalEventTrace {
  CanonicalEventTraceKind kind = CanonicalEventTraceKind::Straight;
  SmallVector<unsigned, 8> actions;
  Region *controlRegion = nullptr;
  CanonicalEventExecutionGuard guard;
  bool hasExplicitTokenState = false;
  SmallVector<unsigned, 8> initialTokens;
  SmallVector<unsigned, 8> expectedTokens;
};

struct CanonicalEvent {
  std::size_t source = 0;
  std::size_t target = 0;
  PipelineType sourcePipe = PipelineType::PIPE_UNASSIGNED;
  PipelineType targetPipe = PipelineType::PIPE_UNASSIGNED;
  CanonicalAnchor setAnchor;
  CanonicalAnchor waitAnchor;
  Operation *recurrenceLoop = nullptr;
  Operation *forwardDrainLoop = nullptr;
  Operation *scopeLoop = nullptr;
  Operation *resourceScopeLoop = nullptr;
  unsigned iterationDistance = 0;
  Value setSlot;
  Value waitSlot;
  unsigned width = 1;
  SmallVector<unsigned, 2> eventIds;
  std::size_t intervalBegin = 0;
  std::size_t intervalEnd = 0;
  SmallVector<CanonicalEventAction, 8> actions;
  SmallVector<CanonicalEventCompletion, 4> completions;
  SmallVector<CanonicalEventTrace, 4> traces;
  std::size_t ownershipCycle = 0;
  CanonicalOwnershipProtocolKind ownershipProtocolKind =
      CanonicalOwnershipProtocolKind::RoundTrip;
  CanonicalOwnershipEventRole ownershipRole = CanonicalOwnershipEventRole::None;
  bool ownershipProtocol = false;
};

struct CanonicalEventDomain {
  PipelineType sourcePipe = PipelineType::PIPE_UNASSIGNED;
  PipelineType targetPipe = PipelineType::PIPE_UNASSIGNED;
  std::size_t originalEventCount = 0;
  unsigned eventCount = 0;
  unsigned availableIds = 0;
  unsigned originalColorCount = 0;
  unsigned colorCount = 0;
  std::size_t serializationCost = 0;
  std::uint64_t originalCriticalPathWeight = 0;
  std::uint64_t criticalPathWeight = 0;
  SmallVector<unsigned, 2> reservedIds;
};

struct CanonicalSelectionMechanismRef {
  CanonicalSelectionMechanismKind kind =
      CanonicalSelectionMechanismKind::Barrier;
  std::size_t id = 0;

  bool operator<(const CanonicalSelectionMechanismRef &other) const {
    return std::tie(kind, id) < std::tie(other.kind, other.id);
  }

  bool operator==(const CanonicalSelectionMechanismRef &other) const {
    return kind == other.kind && id == other.id;
  }
};

struct CanonicalSyncCoveringSelectedProvider {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
};

struct CanonicalSyncCoveringSelectedBarrier {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
  CanonicalBarrier barrier;
};

struct CanonicalSyncCoveringSelectedEventBundle {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
  std::size_t bundleId = 0;
  CanonicalEventBundleKind kind = CanonicalEventBundleKind::Standalone;
  std::vector<CanonicalEvent> events;
};

struct CanonicalSyncCoveringResourceAllocation {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
  std::size_t resourceUse = 0;
  SyncCoverResourceDomainId domain = 0;
  SyncCoverResourceKind kind = SyncCoverResourceKind::EventId;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::vector<unsigned> ids;
};

struct CanonicalSyncCoveringSelectedResourceUse {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
  std::size_t resourceUse = 0;
  SyncCoverResourceDomainId domain = 0;
  SyncCoverResourceKind kind = SyncCoverResourceKind::EventId;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::uint64_t poolIdentity = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::size_t width = 0;
  SyncCoverTimelineInterval lifetime;
  std::optional<std::size_t> materializationEventIndex;
};

/// Emission recipe for one selected verified slot protocol. The event remains
/// unallocated until materialization binds its single resource use.
struct CanonicalSyncCoveringSelectedSlotProtocol {
  SyncCoverMechanismId mechanism = 0;
  CanonicalSelectionMechanismRef provider;
  std::size_t resourceUse = 0;
  CanonicalEvent event;
};

enum class CanonicalSyncCoveringAllocationError : std::uint8_t {
  None,
  SelectionNotReady,
  InvalidProvider,
  InvalidDomain,
  InvalidResourceUse,
  InvalidAllocation,
  UnsupportedResourceKind,
  InvalidPressure,
  ConflictingAssignment,
};

struct CanonicalSyncCoveringAllocationValidation {
  CanonicalSyncCoveringAllocationError error =
      CanonicalSyncCoveringAllocationError::None;
  std::optional<SyncCoverMechanismId> mechanism;
  std::optional<std::size_t> resourceUse;
  std::optional<SyncCoverResourceDomainId> domain;
  std::optional<unsigned> physicalId;

  explicit operator bool() const {
    return error == CanonicalSyncCoveringAllocationError::None;
  }
};

/// Stable diagnostic copy of a discovered exact-range lifecycle. Keeping the
/// analyzer's private representation out of this public plan header avoids
/// rebuilding every CanonicalSync translation unit when factories evolve.
struct CanonicalSyncCoveringSlotLifecycle {
  std::size_t id = 0;
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;
  std::vector<std::size_t> ready;
  std::vector<std::size_t> release;
  std::vector<SyncCoverStorageAccessId> managedAccesses;
  bool hasUnrepresentedAccesses = false;
  bool requiresPathSensitiveProof = false;
};

/// Direct-cover translation, selection diagnostics, and emission handoff.
struct CanonicalSyncCoveringSnapshot {
  std::size_t scopes = 0;
  std::size_t controls = 0;
  std::size_t nodes = 0;
  std::size_t fixedEdges = 0;
  std::size_t recurrenceCarryEdges = 0;
  std::size_t conservativeDemands = 0;
  std::size_t activeDemands = 0;
  std::size_t intrinsicallySatisfiedDemands = 0;
  std::size_t storageDomains = 0;
  std::size_t storageAccesses = 0;
  std::size_t storageWitnesses = 0;
  std::size_t slotLifecycleCandidates = 0;
  std::size_t pathSensitiveSlotLifecycles = 0;
  std::size_t partialSlotOpportunities = 0;
  bool slotLifecycleDiscoveryTruncated = false;
  std::size_t slotProtocolCandidates = 0;
  std::size_t pathSensitiveSlotProtocolLifecycles = 0;
  std::size_t accessOpenSlotProtocolLifecycles = 0;
  std::size_t unsupportedEffectSlotProtocolLifecycles = 0;
  std::size_t unsupportedDistanceSlotProtocolReleases = 0;
  std::size_t nonBoundarySlotProtocolReleases = 0;
  std::size_t slotProtocolEvaluations = 0;
  bool slotProtocolGenerationTruncated = false;
  std::size_t unmaterializableSlotProtocolCandidates = 0;
  std::size_t resourceDomainCount = 0;
  std::size_t barrierCandidates = 0;
  std::size_t eventBundleCandidates = 0;
  std::size_t slotProtocolMechanismCandidates = 0;
  std::size_t candidateMechanisms = 0;
  std::size_t generatedColumnCandidates = 0;
  std::size_t generatedColumns = 0;
  bool columnGenerationTruncated = false;
  std::size_t selectedMechanisms = 0;
  std::size_t solverComponents = 0;
  std::size_t solverEvaluations = 0;
  std::size_t redundancyEvaluations = 0;
  std::size_t oracleRedundancyChecks = 0;
  std::size_t rescuedComponents = 0;
  std::vector<std::size_t> demandsWithoutEventColumn;
  SyncCoverSelectionError selectionError = SyncCoverSelectionError::None;
  bool selectionAttempted = false;
  bool searchTruncated = false;
  bool optimalityProven = false;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierActionProfile;
  std::vector<CanonicalSyncCoveringSelectedProvider> selectedProviders;
  std::vector<CanonicalSyncCoveringSelectedBarrier> selectedBarriers;
  std::vector<CanonicalSyncCoveringSelectedEventBundle>
      selectedEventBundles;
  std::vector<CanonicalSyncCoveringSelectedResourceUse> selectedResourceUses;
  std::vector<CanonicalSyncCoveringSelectedSlotProtocol> selectedSlotProtocols;
  std::vector<CanonicalSyncCoveringResourceAllocation> selectedAllocations;
  std::vector<SyncCoverResourceDomain> resourceDomainDetails;
  SyncCoverResourceSelection selectedResources;
  SyncCoverCoverageStatistics coverageStatistics;
  SyncCoverCoverageStatistics finalVerificationStatistics;
  std::vector<SyncCoverScope> scopeDetails;
  std::vector<SyncCoverControl> controlDetails;
  std::vector<SyncCoverNode> nodeDetails;
  std::vector<SyncCoverEdge> edgeDetails;
  std::vector<SyncCoverDemand> demandDetails;
  std::vector<std::size_t> activeDemandIds;
  std::vector<CanonicalSyncCoveringSlotLifecycle> slotLifecycleDetails;
};

/// Configuration for CanonicalSync plan construction. Callers set the event
/// budget explicitly; zero therefore remains an invalid, fail-closed default.
struct CanonicalSyncBuildOptions {
  unsigned eventIdMax = 0;
  CanonicalGMAliasPolicy gmAliasPolicy = CanonicalGMAliasPolicy::MayAlias;
};

CanonicalSyncCoveringAllocationValidation
validateCanonicalSyncCoveringAllocation(
    const CanonicalSyncCoveringSnapshot &snapshot);

bool canonicalSyncCoveringResourceUseMatches(
    const CanonicalSyncCoveringSelectedResourceUse &selected,
    const SyncCoverResourceUse &live,
    const SyncCoverTimelineInterval &liveLifetime);

class CanonicalSyncPlanBuilder;

class CanonicalSyncPlan {
public:
  ArrayRef<CanonicalSyncNode> getNodes() const { return nodes_; }
  ArrayRef<SyncGraphEdge> getFixedEdges() const { return fixedEdges_; }
  ArrayRef<CanonicalDependency> getDependencies() const {
    return dependencies_;
  }
  ArrayRef<CanonicalDependency> getCompletionRequirements() const {
    return completionRequirements_;
  }
  ArrayRef<CanonicalDependency> getConservativeCompletionRequirements() const {
    return conservativeCompletionRequirements_;
  }
  ArrayRef<CanonicalBarrier> getBarriers() const { return barriers_; }
  ArrayRef<CanonicalEvent> getEvents() const { return events_; }
  ArrayRef<CanonicalEventDomain> getDomains() const { return domains_; }
  ArrayRef<CanonicalOwnershipCycle> getOwnershipCycles() const {
    return ownershipCycles_;
  }
  const std::optional<CanonicalSyncCoveringSnapshot> &
  getCoveringSnapshot() const {
    return coveringSnapshot_;
  }
  CanonicalGMAliasPolicy getGMAliasPolicy() const { return gmAliasPolicy_; }

private:
  friend class CanonicalSyncPlanBuilder;
  friend FailureOr<CanonicalSyncPlan>
  buildCanonicalSyncPlan(func::FuncOp, const CanonicalSyncBuildOptions &);

  std::vector<CanonicalSyncNode> nodes_;
  std::vector<SyncGraphEdge> fixedEdges_;
  std::vector<CanonicalDependency> dependencies_;
  std::vector<CanonicalDependency> conservativeCompletionRequirements_;
  std::vector<CanonicalDependency> completionRequirements_;
  std::vector<CanonicalBarrier> barriers_;
  std::vector<CanonicalEvent> events_;
  std::vector<CanonicalEventDomain> domains_;
  std::vector<CanonicalOwnershipCycle> ownershipCycles_;
  std::optional<CanonicalSyncCoveringSnapshot> coveringSnapshot_;
  CanonicalGMAliasPolicy gmAliasPolicy_ = CanonicalGMAliasPolicy::MayAlias;
};

FailureOr<CanonicalSyncPlan>
buildCanonicalSyncPlan(func::FuncOp func,
                       const CanonicalSyncBuildOptions &options);
LogicalResult emitCanonicalSyncPlan(func::FuncOp func,
                                    const CanonicalSyncPlan &plan);

StringRef stringifyCanonicalDependencyKind(CanonicalDependencyKind kind);
StringRef stringifyCanonicalGMAliasPolicy(CanonicalGMAliasPolicy policy);
StringRef stringifyCanonicalEventBundleKind(CanonicalEventBundleKind kind);
StringRef
stringifyCanonicalSelectionMechanismKind(CanonicalSelectionMechanismKind kind);
StringRef stringifyCanonicalOwnershipKind(CanonicalOwnershipKind kind);
void printCanonicalSyncPlan(llvm::raw_ostream &os, func::FuncOp func,
                            const CanonicalSyncPlan &plan, StringRef view);
void printCanonicalSyncPlanDot(llvm::raw_ostream &os, func::FuncOp func,
                               const CanonicalSyncPlan &plan, StringRef view);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
