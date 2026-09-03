// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncModel.h - Immutable synchronization model ---*- C++ -*-===//
//
// CanonicalSync separates hardware obligations from ways of satisfying them.
// Builders may append records until freeze(); every later phase receives the
// same immutable, stably numbered program.
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCMODEL_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCMODEL_H

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir {
namespace pto {

using CanonicalRegionId = std::uint32_t;
using CanonicalPhaseId = std::uint32_t;
using CanonicalAccessId = std::uint32_t;
using CanonicalFenceEffectId = std::uint32_t;
using CanonicalDemandId = std::uint32_t;
using CanonicalMechanismId = std::uint32_t;
using CanonicalSetCoverCandidateId = std::uint32_t;
using CanonicalOwnershipChannelId = std::uint32_t;
using CanonicalStorageGenerationId = std::uint32_t;

inline constexpr std::uint32_t kInvalidCanonicalSyncId =
    std::numeric_limits<std::uint32_t>::max();
enum class CanonicalCore : std::uint8_t { AIC, AIV };
enum class CanonicalRegionKind : std::uint8_t {
  Function,
  Sequence,
  Choice,
  Loop,
  Transparent,
};
enum class CanonicalCardinality : std::uint8_t {
  ExactlyOnce,
  ZeroOrOne,
  ZeroOrMore,
  OneOrMore,
};
enum class CanonicalAccessMode : std::uint8_t { Read, Write, ReadWrite };
enum class CanonicalDemandKind : std::uint8_t {
  Raw,
  War,
  Waw,
  OrderedMemory,
  HardwareAccReadConflict,
  SsaCompletion,
  ExitCompletion,
  Visibility,
};
enum class CanonicalRequirement : std::uint8_t { Completion, Visibility };
enum class CanonicalGmAliasPolicy : std::uint8_t {
  Conservative,
  DistinctRootsUnsafe,
};
enum class CanonicalOwnershipPlanning : std::uint8_t {
  Disabled,
  Diagnostic,
  ReadyRelease2,
};

struct CanonicalSyncStatistics {
  std::uint64_t regions = 0;
  std::uint64_t phases = 0;
  std::uint64_t accesses = 0;
  std::uint64_t fenceEffects = 0;
  std::uint64_t demands = 0;
  std::uint64_t fixedCoveredDemands = 0;
  std::uint64_t mechanisms = 0;
  std::uint64_t pipeBarrierCandidates = 0;
  std::uint64_t sharedEventFrontiers = 0;
  std::uint64_t sharedEventFrontierMembers = 0;
  std::uint64_t selectedSharedEventFrontiers = 0;
  std::uint64_t multiDemandPipeBarrierCandidates = 0;
  std::uint64_t multiDemandPipeBarrierCoveredDemands = 0;
  std::uint64_t selectedMultiDemandPipeBarriers = 0;
  std::uint64_t selectedPipeBarriers = 0;
  std::uint64_t ownershipChannels = 0;
  std::uint64_t storageGenerations = 0;
  std::uint64_t ownershipReadyEdges = 0;
  std::uint64_t ownershipReleaseEdges = 0;
  std::uint64_t depthTwoOwnershipChannels = 0;
  std::uint64_t slotTrackedOwnershipChannels = 0;
  std::uint64_t readyRelease2Candidates = 0;
  std::uint64_t selectedReadyRelease2 = 0;
  std::uint64_t readyRelease2AllocationFallbacks = 0;
  std::uint64_t coverageWorlds = 0;
  std::uint64_t coverUniverse = 0;
  std::uint64_t coverCandidates = 0;
  std::uint64_t selectedMechanisms = 0;
  std::uint64_t aliasPairTests = 0;
  std::uint64_t aliasCandidatePairs = 0;
  std::uint64_t localIntervalRecords = 0;
  std::uint64_t sparseIncidenceEntries = 0;
  std::uint64_t greedyHeapPops = 0;
  std::uint64_t greedyIncidenceVisits = 0;
  std::uint64_t precomputedPrefixEntries = 0;
  std::uint64_t ssaTraceVisits = 0;
  std::uint64_t mechanismInternKeyTests = 0;
  std::uint64_t mechanismPrefixPhaseTests = 0;
  std::uint64_t coverageFactKeyTests = 0;
  std::uint64_t coverageTransferKeyTests = 0;
  std::uint64_t coverageTransferComposeTests = 0;
  std::uint64_t coveragePropagationFactTests = 0;
  std::uint64_t coverageBoundaryPhaseTests = 0;
  std::uint64_t coverageRegionSummaries = 0;
  std::uint64_t coverageSummaryFacts = 0;
  std::uint64_t coverageSummaryTransfers = 0;
  std::uint64_t coverageOracleWorlds = 0;
  std::uint64_t coverageOracleSkippedWorlds = 0;
  std::uint64_t coverageOracleStateOperations = 0;
  std::uint64_t coverageOracleMechanismTests = 0;
  std::uint64_t coverageOracleDemandTests = 0;
  std::uint64_t coverageOracleSourceInstanceTests = 0;
  std::uint64_t structureUs = 0;
  std::uint64_t demandsUs = 0;
  std::uint64_t mechanismsUs = 0;
  std::uint64_t coverageUs = 0;
  std::uint64_t setCoverBuildUs = 0;
  std::uint64_t selectionUs = 0;
  std::uint64_t allocationUs = 0;
  std::uint64_t freezeUs = 0;
  std::uint64_t materializeVerifyUs = 0;
  std::uint64_t verifierLoopTransfers = 0;
  std::uint64_t maxVerifierLoopStates = 0;
};
enum class CanonicalVisibilityDirection : std::uint8_t {
  ScalarToNonScalar,
  NonScalarToScalar,
  Mte3ToMte2Gm,
};
enum class CanonicalIterationRelation : std::uint8_t {
  Same,
  AnyPositive,
  Any,
};
enum class CanonicalCacheMaintenance : std::uint8_t {
  None,
  CleanSource,
  InvalidateTarget,
};
enum class CanonicalMechanismKind : std::uint8_t {
  IntrinsicOrder,
  PipeBarrier,
  Event,
  CrossCoreEvent,
  RecurringEvent,
  ReadyRelease2,
  VisibilityFence,
  FixedFence,
  TailBarrier,
};
enum class CanonicalMechanismSynthesis : std::uint8_t {
  None,
  SharedEventFrontier,
  StorageOwnershipProtocol,
};
enum class CanonicalProgramPointPosition : std::uint8_t { Before, After };

struct CanonicalPhysicalResource {
  CanonicalCore core = CanonicalCore::AIV;
  PIPE pipe = PIPE::PIPE_UNASSIGNED;

  bool operator==(const CanonicalPhysicalResource &other) const {
    return core == other.core && pipe == other.pipe;
  }
  bool operator!=(const CanonicalPhysicalResource &other) const {
    return !(*this == other);
  }
  bool operator<(const CanonicalPhysicalResource &other) const;
};

struct CanonicalControlAtom {
  CanonicalRegionId choice = kInvalidCanonicalSyncId;
  unsigned arm = 0;

  bool operator==(const CanonicalControlAtom &other) const {
    return choice == other.choice && arm == other.arm;
  }
};

struct CanonicalLoopDistance {
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  CanonicalIterationRelation relation = CanonicalIterationRelation::Same;

  bool operator==(const CanonicalLoopDistance &other) const {
    return loop == other.loop && relation == other.relation;
  }
};

struct CanonicalProgramPoint {
  Operation *operation = nullptr;
  CanonicalProgramPointPosition position =
      CanonicalProgramPointPosition::Before;

  bool operator==(const CanonicalProgramPoint &other) const {
    return operation == other.operation && position == other.position;
  }
};

struct CanonicalRegion {
  CanonicalRegionId id = kInvalidCanonicalSyncId;
  CanonicalRegionId parent = kInvalidCanonicalSyncId;
  CanonicalRegionKind kind = CanonicalRegionKind::Sequence;
  CanonicalCardinality cardinality = CanonicalCardinality::ExactlyOnce;
  Operation *operation = nullptr;
  unsigned depth = 0;
  unsigned arm = 0;
};

struct CanonicalPhase {
  CanonicalPhaseId id = kInvalidCanonicalSyncId;
  CanonicalRegionId region = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource resource;
  Operation *operation = nullptr;
  unsigned sourceOrder = 0;
  std::optional<unsigned> macroPhase;
  llvm::SmallVector<CanonicalControlAtom, 2> controlPath;
  llvm::SmallVector<CanonicalRegionId, 2> loopPath;
};

struct CanonicalByteInterval {
  std::uint64_t begin = 0;
  std::uint64_t size = 0;

  std::optional<std::uint64_t> end() const;
};

struct CanonicalAccess {
  CanonicalAccessId id = kInvalidCanonicalSyncId;
  CanonicalPhaseId phase = kInvalidCanonicalSyncId;
  CanonicalAccessMode mode = CanonicalAccessMode::Read;
  AddressSpace space = AddressSpace::Zero;
  bool unknownSpace = false;
  bool ordered = false;
  Value value;
  Value aliasRoot;
  std::optional<int64_t> addressByteOffset;
  std::optional<int64_t> addressByteSize;
  llvm::SmallVector<CanonicalByteInterval, 2> intervals;
  bool physical = false;
  bool unknownRange = true;
  Value slotExpression;
  std::string provenance;
};

struct CanonicalFenceEffect {
  CanonicalFenceEffectId id = kInvalidCanonicalSyncId;
  Operation *operation = nullptr;
  CanonicalRegionId region = kInvalidCanonicalSyncId;
  FenceScope scope = FenceScope::GM;
  llvm::SmallVector<CanonicalPhysicalResource, 8> drainedResources;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  llvm::SmallVector<CanonicalRegionId, 2> loopPath;
};

struct CanonicalVisibilityRequirement {
  CanonicalVisibilityDirection direction =
      CanonicalVisibilityDirection::ScalarToNonScalar;
  FenceScope scope = FenceScope::GM;
  CanonicalCacheMaintenance cacheMaintenance = CanonicalCacheMaintenance::None;

  bool operator==(const CanonicalVisibilityRequirement &other) const {
    return direction == other.direction && scope == other.scope &&
           cacheMaintenance == other.cacheMaintenance;
  }
};

struct CanonicalDemandCause {
  CanonicalAccessId sourceAccess = kInvalidCanonicalSyncId;
  CanonicalAccessId targetAccess = kInvalidCanonicalSyncId;
  std::string provenance;
};

struct CanonicalDemand {
  CanonicalDemandId id = kInvalidCanonicalSyncId;
  CanonicalPhaseId source = kInvalidCanonicalSyncId;
  CanonicalPhaseId target = kInvalidCanonicalSyncId;
  CanonicalRegionId owner = kInvalidCanonicalSyncId;
  CanonicalDemandKind kind = CanonicalDemandKind::Raw;
  CanonicalRequirement requirement = CanonicalRequirement::Completion;
  std::optional<CanonicalVisibilityRequirement> visibility;
  llvm::SmallVector<CanonicalLoopDistance, 2> iterationDistance;
  llvm::SmallVector<CanonicalControlAtom, 2> sourceGuard;
  llvm::SmallVector<CanonicalControlAtom, 2> targetGuard;
  llvm::SmallVector<CanonicalDemandCause, 1> causes;
};

struct CanonicalOwnershipEdge {
  CanonicalDemandId demand = kInvalidCanonicalSyncId;
  CanonicalAccessId sourceAccess = kInvalidCanonicalSyncId;
  CanonicalAccessId targetAccess = kInvalidCanonicalSyncId;
};

/// One exact symbolic generation family for the initial protocol-first
/// migration. The two physical slots are selected by `(iv + offset) mod 2`.
/// These are storage and schedule facts discovered before demands exist; they
/// are not themselves a synchronization proof.
struct CanonicalStorageGeneration {
  CanonicalStorageGenerationId id = kInvalidCanonicalSyncId;
  Value storage;
  AddressSpace space = AddressSpace::Zero;
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  CanonicalAccessId producerAccess = kInvalidCanonicalSyncId;
  CanonicalAccessId consumerAccess = kInvalidCanonicalSyncId;
  CanonicalProgramPoint writeAcquire;
  CanonicalProgramPoint ready;
  CanonicalProgramPoint readAcquire;
  CanonicalProgramPoint lastUse;
  Value slotExpression;
  std::uint32_t staticDepth = 0;
  std::uint32_t slotOffset = 0;
  std::uint32_t reuseDistance = 0;
  std::uint32_t witnessHorizon = 0;
};

/// A diagnostic storage-generation cycle. A same-generation RAW edge
/// publishes a local buffer from producer to consumer; a positive-distance
/// WAR edge returns that same logical storage family to the producer. This is
/// proposal evidence for a future atomic ReadyRelease<N> mechanism, not a
/// correctness proof or a selectable mechanism by itself.
struct CanonicalOwnershipChannel {
  CanonicalOwnershipChannelId id = kInvalidCanonicalSyncId;
  CanonicalStorageGenerationId generation = kInvalidCanonicalSyncId;
  Value storage;
  AddressSpace space = AddressSpace::Zero;
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  std::uint32_t staticDepth = 1;
  bool slotTracked = false;
  bool readyRelease2Eligible = false;
  llvm::SmallVector<CanonicalOwnershipEdge, 4> readyEdges;
  llvm::SmallVector<CanonicalOwnershipEdge, 4> releaseEdges;
};

struct CanonicalMechanism {
  CanonicalMechanismId id = kInvalidCanonicalSyncId;
  CanonicalMechanismKind kind = CanonicalMechanismKind::PipeBarrier;
  CanonicalMechanismSynthesis synthesis = CanonicalMechanismSynthesis::None;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  CanonicalProgramPoint sourcePoint;
  CanonicalProgramPoint targetPoint;
  llvm::SmallVector<CanonicalDemandId, 2> origins;
  llvm::SmallVector<Operation *, 2> cacheMaintenance;
  std::optional<CanonicalCacheMaintenance> generatedCacheMaintenance;
  std::optional<CanonicalFenceEffectId> fenceEffect;
  CanonicalRegionId actionRegion = kInvalidCanonicalSyncId;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<CanonicalRegionId> recurrenceLoop;
  /// The forward ready channel is also carried at the loop boundary. This is
  /// used when source and target are in mutually exclusive control arms and
  /// no legal same-iteration ready cut exists.
  bool boundaryRecurring = false;
  /// The ready and reverse-release body actions execute under the same
  /// repeated control guard. Skipped iterations leave the primed release token
  /// untouched instead of resetting it at the loop header and latch.
  bool guardedRecurring = false;
  std::optional<CanonicalOwnershipChannelId> ownershipChannel;
  std::optional<unsigned> eventId;
  std::optional<unsigned> releaseEventId;
  llvm::SmallVector<unsigned, 2> readyEventIds;
  llvm::SmallVector<unsigned, 2> ownershipReleaseEventIds;
};

struct CanonicalCompletionTransfer {
  CanonicalPhaseId phase = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource resource;
  CanonicalProgramPoint availableAt;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  llvm::SmallVector<CanonicalRegionId, 2> requiredLoops;
};

struct CanonicalBoundaryTransfer {
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  CanonicalProgramPoint sourcePoint;
  CanonicalProgramPoint targetPoint;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  llvm::SmallVector<CanonicalRegionId, 2> requiredLoops;
};

struct CanonicalRegionSummary {
  CanonicalRegionId region = kInvalidCanonicalSyncId;
  llvm::SmallVector<CanonicalRegionId, 0> children;
  llvm::SmallVector<CanonicalCompletionTransfer, 0> completions;
  llvm::SmallVector<CanonicalBoundaryTransfer, 0> transfers;
};

struct CanonicalCoverageWorld {
  std::string name;
  llvm::SmallVector<CanonicalMechanismId, 8> mechanisms;
  llvm::SmallVector<CanonicalDemandId, 8> covered;
  llvm::SmallVector<CanonicalRegionSummary, 0> summaries;
  llvm::SmallVector<CanonicalDemandId, 8> differentialDisagreements;
  bool flattenedOracleMatched = false;
  bool unrolledOracleAvailable = false;
  bool unrolledOracleExhaustive = false;
  bool unrolledOracleMatched = false;
  bool setCoverCandidate = false;
};

struct CanonicalSetCoverCandidate {
  CanonicalSetCoverCandidateId id = kInvalidCanonicalSyncId;
  llvm::SmallVector<CanonicalMechanismId, 2> mechanisms;
  llvm::SmallVector<CanonicalDemandId, 4> directOrigins;
  llvm::SmallVector<CanonicalDemandId, 4> additionalCoverage;
  llvm::SmallVector<CanonicalDemandId, 8> coveredDemands;
  std::uint64_t weight = 0;
};

struct CanonicalSetCoverInstance {
  llvm::SmallVector<CanonicalMechanismId, 4> baseline;
  llvm::SmallVector<CanonicalDemandId, 8> universe;
  llvm::SmallVector<CanonicalSetCoverCandidate, 8> candidates;
  llvm::SmallVector<llvm::SmallVector<CanonicalSetCoverCandidateId, 2>, 0>
      providersByDemand;
};

/// A physical event group used only after ordinary allocation exhausts a
/// directed domain. Coalesced groups widen compatible cuts; serialized groups
/// use a forward ready and reverse release handshake. The members keep the
/// logical coverage established by their singleton mechanisms.
enum class CanonicalScarcityEventKind : std::uint8_t {
  Coalesced,
  Serialized,
};

struct CanonicalScarcityEventGroup {
  CanonicalScarcityEventKind kind = CanonicalScarcityEventKind::Coalesced;
  llvm::SmallVector<CanonicalMechanismId, 4> members;
  CanonicalPhysicalResource source;
  CanonicalPhysicalResource target;
  CanonicalProgramPoint sourcePoint;
  CanonicalProgramPoint targetPoint;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<CanonicalRegionId> recurrenceLoop;
  unsigned eventId = 0;
  std::optional<unsigned> releaseEventId;
};

/// A lifecycle that lets non-nested recurring protocols reuse a reverse
/// release token. The token is primed before the complete structured frontier,
/// circulated by every executed loop body, and drained afterward. A singleton
/// pool also prevents an inner protocol's prime/drain from repeating inside an
/// outer loop.
struct CanonicalRecurringReleasePool {
  llvm::SmallVector<CanonicalRegionId, 4> recurrenceLoops;
  CanonicalPhysicalResource releaseSource;
  CanonicalPhysicalResource releaseTarget;
  CanonicalProgramPoint primePoint;
  CanonicalProgramPoint drainPoint;
  unsigned releaseEventId = 0;
};

struct CanonicalSetCoverSolution {
  llvm::SmallVector<CanonicalSetCoverCandidateId, 8> greedyCandidates;
  llvm::SmallVector<CanonicalMechanismId, 8> mechanisms;
  llvm::SmallVector<CanonicalMechanismId, 8> reverseDeleted;
  llvm::SmallVector<CanonicalScarcityEventGroup, 2> scarcityEventGroups;
  llvm::SmallVector<CanonicalRecurringReleasePool, 2> recurringReleasePools;
  std::uint64_t weight = 0;
  bool coverageVerified = false;
};

class CanonicalSyncProgram;
LogicalResult buildCanonicalDirectMechanisms(
    CanonicalSyncProgram &program, bool enableSharedEventFrontiers,
    CanonicalOwnershipPlanning ownershipPlanning);
LogicalResult evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
LogicalResult buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);
LogicalResult solveCanonicalSyncSetCover(CanonicalSyncProgram &program);

class CanonicalSyncProgram {
public:
  explicit CanonicalSyncProgram(func::FuncOp function,
                                CanonicalGmAliasPolicy gmAliasPolicy =
                                    CanonicalGmAliasPolicy::Conservative,
                                CanonicalSyncStatistics *statistics = nullptr)
      : function(function), gmAliasPolicy(gmAliasPolicy),
        statistics(statistics) {}

  CanonicalRegionId appendRegion(CanonicalRegion region);
  CanonicalPhaseId appendPhase(CanonicalPhase phase);
  CanonicalAccessId appendAccess(CanonicalAccess access);
  CanonicalFenceEffectId appendFenceEffect(CanonicalFenceEffect effect);
  CanonicalStorageGenerationId
  appendStorageGeneration(CanonicalStorageGeneration generation);
  CanonicalDemandId appendDemand(CanonicalDemand demand);
  CanonicalOwnershipChannelId
  appendOwnershipChannel(CanonicalOwnershipChannel channel);
  void retainDemands(const llvm::BitVector &retained);
  void appendDemandCause(CanonicalDemandId demand, CanonicalDemandCause cause);
  CanonicalMechanismId appendMechanism(CanonicalMechanism mechanism);
  void appendMechanismOrigin(CanonicalMechanismId mechanism,
                             CanonicalDemandId demand);
  void appendMechanismCacheMaintenance(CanonicalMechanismId mechanism,
                                       llvm::ArrayRef<Operation *> actions);
  void setMechanismEventId(CanonicalMechanismId mechanism, unsigned eventId);
  void setMechanismReleaseEventId(CanonicalMechanismId mechanism,
                                  unsigned eventId);
  void setMechanismOwnershipEventIds(CanonicalMechanismId mechanism,
                                     llvm::ArrayRef<unsigned> ready,
                                     llvm::ArrayRef<unsigned> release);
  void disableMechanismForSelection(CanonicalMechanismId mechanism);
  bool isMechanismDisabled(CanonicalMechanismId mechanism) const;
  void clearSetCoverSolution();
  void setScarcityEventPlan(
      llvm::SmallVector<CanonicalScarcityEventGroup, 2> groups,
      llvm::SmallVector<CanonicalRecurringReleasePool, 2> releasePools);
  void setDirectMechanism(CanonicalDemandId demand,
                          CanonicalMechanismId mechanism);

  LogicalResult freezeGraph();
  LogicalResult freeze();
  bool isGraphFrozen() const { return graphFrozen; }
  bool isFrozen() const { return frozen; }
  func::FuncOp getFunction() const { return function; }
  CanonicalGmAliasPolicy getGmAliasPolicy() const { return gmAliasPolicy; }
  CanonicalSyncStatistics *getStatistics() const { return statistics; }
  llvm::ArrayRef<CanonicalRegion> getRegions() const { return regions; }
  llvm::ArrayRef<CanonicalRegionId>
  getRegionChildren(CanonicalRegionId id) const {
    return regionChildren[id];
  }
  llvm::ArrayRef<CanonicalPhase> getPhases() const { return phases; }
  llvm::ArrayRef<CanonicalAccess> getAccesses() const { return accesses; }
  llvm::ArrayRef<CanonicalFenceEffect> getFenceEffects() const {
    return fenceEffects;
  }
  llvm::ArrayRef<CanonicalDemand> getDemands() const { return demands; }
  llvm::ArrayRef<CanonicalStorageGeneration> getStorageGenerations() const {
    return storageGenerations;
  }
  llvm::ArrayRef<CanonicalOwnershipChannel> getOwnershipChannels() const {
    return ownershipChannels;
  }
  llvm::ArrayRef<CanonicalMechanism> getMechanisms() const {
    return mechanisms;
  }
  llvm::ArrayRef<CanonicalCoverageWorld> getCoverageWorlds() const {
    return coverageWorlds;
  }
  llvm::ArrayRef<CanonicalMechanismId> getDirectMechanisms() const {
    return directMechanisms;
  }
  const std::optional<CanonicalSetCoverInstance> &getSetCoverInstance() const {
    return setCoverInstance;
  }
  const std::optional<CanonicalSetCoverSolution> &getSetCoverSolution() const {
    return setCoverSolution;
  }

  const CanonicalRegion &getRegion(CanonicalRegionId id) const;
  const CanonicalPhase &getPhase(CanonicalPhaseId id) const;
  const CanonicalAccess &getAccess(CanonicalAccessId id) const;
  const CanonicalFenceEffect &getFenceEffect(CanonicalFenceEffectId id) const;
  const CanonicalDemand &getDemand(CanonicalDemandId id) const;
  const CanonicalStorageGeneration &
  getStorageGeneration(CanonicalStorageGenerationId id) const;
  const CanonicalMechanism &getMechanism(CanonicalMechanismId id) const;
  llvm::ArrayRef<CanonicalRegionId>
  getMechanismExecutionLoops(CanonicalMechanismId id) const {
    return mechanismExecutionLoops[id];
  }
  llvm::ArrayRef<CanonicalPhaseId>
  getMechanismSourcePrefix(CanonicalMechanismId id) const {
    return mechanismSourcePrefixes[id];
  }

private:
  friend LogicalResult
  buildCanonicalDirectMechanisms(CanonicalSyncProgram &program,
                                 bool enableSharedEventFrontiers,
                                 CanonicalOwnershipPlanning ownershipPlanning);
  friend LogicalResult
  evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
  friend LogicalResult
  buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);
  friend LogicalResult
  solveCanonicalSyncSetCover(CanonicalSyncProgram &program);

  void appendCoverageWorld(CanonicalCoverageWorld world);
  void setSetCoverInstance(CanonicalSetCoverInstance instance);
  void setSetCoverSolution(CanonicalSetCoverSolution solution);

  func::FuncOp function;
  CanonicalGmAliasPolicy gmAliasPolicy = CanonicalGmAliasPolicy::Conservative;
  CanonicalSyncStatistics *statistics = nullptr;
  llvm::SmallVector<CanonicalRegion> regions;
  llvm::SmallVector<llvm::SmallVector<CanonicalRegionId, 4>, 0> regionChildren;
  llvm::SmallVector<CanonicalPhase> phases;
  llvm::SmallVector<CanonicalAccess> accesses;
  llvm::SmallVector<CanonicalFenceEffect> fenceEffects;
  llvm::SmallVector<CanonicalDemand> demands;
  llvm::SmallVector<CanonicalStorageGeneration> storageGenerations;
  llvm::SmallVector<CanonicalOwnershipChannel> ownershipChannels;
  llvm::SmallVector<CanonicalMechanism> mechanisms;
  llvm::SmallVector<llvm::SmallVector<CanonicalRegionId, 2>, 0>
      mechanismExecutionLoops;
  llvm::SmallVector<llvm::SmallVector<CanonicalPhaseId, 8>, 0>
      mechanismSourcePrefixes;
  llvm::SmallVector<CanonicalMechanismId> directMechanisms;
  llvm::SmallVector<CanonicalCoverageWorld> coverageWorlds;
  std::optional<CanonicalSetCoverInstance> setCoverInstance;
  std::optional<CanonicalSetCoverSolution> setCoverSolution;
  llvm::BitVector disabledMechanisms;
  bool buildingMechanisms = false;
  bool mechanismCatalogComplete = false;
  bool coverageCatalogComplete = false;
  bool graphFrozen = false;
  bool frozen = false;
};

llvm::StringRef stringifyCanonicalCore(CanonicalCore core);
llvm::StringRef stringifyCanonicalRegionKind(CanonicalRegionKind kind);
llvm::StringRef stringifyCanonicalAccessMode(CanonicalAccessMode mode);
llvm::StringRef stringifyCanonicalDemandKind(CanonicalDemandKind kind);
llvm::StringRef stringifyCanonicalGmAliasPolicy(CanonicalGmAliasPolicy policy);
llvm::StringRef
stringifyCanonicalOwnershipPlanning(CanonicalOwnershipPlanning mode);
llvm::StringRef
stringifyCanonicalVisibilityDirection(CanonicalVisibilityDirection direction);
llvm::StringRef
stringifyCanonicalCacheMaintenance(CanonicalCacheMaintenance maintenance);
llvm::StringRef stringifyCanonicalMechanismKind(CanonicalMechanismKind kind);
llvm::StringRef
stringifyCanonicalMechanismSynthesis(CanonicalMechanismSynthesis synthesis);
void printCanonicalSyncProgram(const CanonicalSyncProgram &program,
                               llvm::raw_ostream &os);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCMODEL_H
