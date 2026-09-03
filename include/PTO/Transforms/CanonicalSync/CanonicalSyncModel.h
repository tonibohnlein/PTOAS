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
using CanonicalStructuralProposalId = std::uint32_t;
using CanonicalSetCoverCandidateId = std::uint32_t;
using CanonicalStorageGenerationId = std::uint32_t;
using CanonicalOwnershipProtocolId = std::uint32_t;

enum class CanonicalStructuralCoverFamily : std::uint32_t {
  Level = 1U << 0,
  Transitive = 1U << 1,
  Connector = 1U << 2,
  Semantic = 1U << 3,
  Storage = 1U << 4,
};
using CanonicalStructuralCoverFamilies = std::uint32_t;
inline constexpr CanonicalStructuralCoverFamilies
    kAllCanonicalStructuralCoverFamilies =
        static_cast<CanonicalStructuralCoverFamilies>(
            CanonicalStructuralCoverFamily::Level) |
        static_cast<CanonicalStructuralCoverFamilies>(
            CanonicalStructuralCoverFamily::Transitive) |
        static_cast<CanonicalStructuralCoverFamilies>(
            CanonicalStructuralCoverFamily::Connector) |
        static_cast<CanonicalStructuralCoverFamilies>(
            CanonicalStructuralCoverFamily::Semantic) |
        static_cast<CanonicalStructuralCoverFamilies>(
            CanonicalStructuralCoverFamily::Storage);

inline bool
hasCanonicalStructuralCoverFamily(CanonicalStructuralCoverFamilies families,
                                  CanonicalStructuralCoverFamily family) {
  return (families & static_cast<CanonicalStructuralCoverFamilies>(family)) !=
         0U;
}

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

struct CanonicalSyncStatistics {
  std::uint64_t regions = 0;
  std::uint64_t phases = 0;
  std::uint64_t accesses = 0;
  std::uint64_t fenceEffects = 0;
  std::uint64_t demands = 0;
  std::uint64_t fixedCoveredDemands = 0;
  std::uint64_t mechanisms = 0;
  std::uint64_t coverageWorlds = 0;
  std::uint64_t coverUniverse = 0;
  std::uint64_t coverCandidates = 0;
  std::uint64_t storageGenerations = 0;
  std::uint64_t structuralProposals = 0;
  std::uint64_t admittedStructuralProposals = 0;
  std::uint64_t structuralMechanismMemberships = 0;
  std::uint64_t structuralAdditionalCoverageRows = 0;
  std::uint64_t structuralSetCoverCandidates = 0;
  std::uint64_t selectedMechanisms = 0;
  std::uint64_t aliasPairTests = 0;
  std::uint64_t aliasCandidatePairs = 0;
  std::uint64_t localIntervalRecords = 0;
  std::uint64_t sparseIncidenceEntries = 0;
  std::uint64_t greedyHeapPops = 0;
  std::uint64_t greedyIncidenceVisits = 0;
  std::uint64_t precomputedPrefixEntries = 0;
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
  PeriodicOwnership,
  VisibilityFence,
  FixedFence,
  TailBarrier,
};
enum class CanonicalStructuralProposalKind : std::uint8_t {
  LevelBoundary,
  LevelBoundaryMinusOne,
  SemanticLevelBoundary,
  RegionTransitiveBasis,
  ConnectorNeighborhood,
  StorageLifecycle,
  StorageLifecycleMinusOne,
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

struct CanonicalMechanism {
  CanonicalMechanismId id = kInvalidCanonicalSyncId;
  CanonicalMechanismKind kind = CanonicalMechanismKind::PipeBarrier;
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
  std::optional<CanonicalOwnershipProtocolId> ownershipProtocol;
  std::optional<unsigned> eventId;
  std::optional<unsigned> releaseEventId;
};

struct CanonicalOwnershipSlot {
  Value root;
  CanonicalByteInterval interval;
  Value slotExpression;
  unsigned lane = 0;
  unsigned reuseDistance = 0;
};

/// One circulating ready/release token pair.  A token lane may serialize a
/// sequence of disjoint physical slots when their ownership stages form one
/// total order.  Keeping token lanes separate from storage slots is essential:
/// buffer depth and event pressure are related, but they are not identical.
struct CanonicalOwnershipLane {
  std::optional<unsigned> readyEventId;
  std::optional<unsigned> releaseEventId;
};

/// One statically described generation of an exact physical storage slot.
///
/// This is an IR/storage fact, not a synchronization proof.  It records the
/// producer completion frontier, first and last consumer frontiers, and the
/// next overwrite of the same slot.  A hardware protocol may later map one or
/// more generations to event-token lanes, but coverage is still established
/// independently from the materialized cuts.
struct CanonicalStorageGeneration {
  CanonicalStorageGenerationId id = kInvalidCanonicalSyncId;
  CanonicalRegionId recurrenceLoop = kInvalidCanonicalSyncId;
  /// Stable semantic identity shared by the physical slots of one logical
  /// producer/consumer buffer family.
  std::string familyKey;
  unsigned familyDepth = 0;
  unsigned slot = 0;
  unsigned stageOrdinal = 0;
  Value root;
  CanonicalByteInterval interval;
  Value slotExpression;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  CanonicalProgramPoint writeAcquire;
  CanonicalProgramPoint ready;
  CanonicalProgramPoint readAcquire;
  CanonicalProgramPoint lastUse;
  CanonicalProgramPoint nextOverwrite;
  llvm::SmallVector<CanonicalPhaseId, 2> producers;
  llvm::SmallVector<CanonicalPhaseId, 4> consumers;
  llvm::SmallVector<CanonicalControlAtom, 2> producerGuard;
  llvm::SmallVector<CanonicalControlAtom, 2> consumerGuard;
  llvm::SmallVector<unsigned, 4> producerResidues;
  llvm::SmallVector<unsigned, 4> consumerResidues;
  unsigned period = 0;
  unsigned readyDistance = 0;
  unsigned nextOverwriteDistance = 0;
  bool initialProducer = false;
};

struct CanonicalOwnershipStage {
  CanonicalStorageGenerationId generation = kInvalidCanonicalSyncId;
  unsigned slot = 0;
  unsigned lane = 0;
  /// An initial producer executes before recurrenceLoop and seeds the ready
  /// token instead of consuming a release credit.
  bool initialProducer = false;
  CanonicalIterationRelation readyRelation = CanonicalIterationRelation::Same;
  unsigned readyDistance = 0;
  unsigned releaseDistance = 0;
  CanonicalProgramPoint writeAcquire;
  CanonicalProgramPoint ready;
  CanonicalProgramPoint readAcquire;
  CanonicalProgramPoint release;
  llvm::SmallVector<CanonicalPhaseId, 2> producers;
  llvm::SmallVector<CanonicalPhaseId, 4> consumers;
  llvm::SmallVector<CanonicalControlAtom, 2> producerGuard;
  llvm::SmallVector<CanonicalControlAtom, 2> consumerGuard;
};

enum class CanonicalOwnershipWitnessKind : std::uint8_t {
  Ready,
  Release,
};

struct CanonicalOwnershipWitnessEdge {
  CanonicalOwnershipWitnessKind kind = CanonicalOwnershipWitnessKind::Ready;
  unsigned slot = 0;
  unsigned lane = 0;
  CanonicalPhaseId source = kInvalidCanonicalSyncId;
  CanonicalPhaseId target = kInvalidCanonicalSyncId;
  unsigned sourceIteration = 0;
  unsigned targetIteration = 0;
};

/// A hardware-level ready/release ring synthesized from one normalized storage
/// family. Physical slots and event-token lanes are distinct: sequential slots
/// may circulate one token lane, while genuinely overlapping modulo slots use
/// separate lanes. Every steady-state producer consumes a release token before
/// writing, transfers readiness to the consumer, and returns the release token
/// after the last read. A certified preheader producer may seed readiness
/// directly. Priming and draining happen outside recurrenceLoop, so zero-trip
/// execution is balanced.
struct CanonicalOwnershipProtocol {
  CanonicalOwnershipProtocolId id = kInvalidCanonicalSyncId;
  CanonicalMechanismId mechanism = kInvalidCanonicalSyncId;
  CanonicalRegionId owner = kInvalidCanonicalSyncId;
  CanonicalRegionId recurrenceLoop = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  std::string familyKey;
  /// Number of exact physical storage slots managed by this protocol.
  unsigned depth = 0;
  /// Repetition period of the stage/control schedule in recurrenceLoop.
  unsigned period = 0;
  /// Maximum proven distance to the next use of any managed slot.
  unsigned reuseDistance = 0;
  unsigned witnessHorizon = 0;
  llvm::SmallVector<Value, 2> roots;
  llvm::SmallVector<CanonicalOwnershipSlot, 2> slots;
  llvm::SmallVector<CanonicalOwnershipLane, 2> lanes;
  llvm::SmallVector<CanonicalOwnershipStage, 8> stages;
  llvm::SmallVector<CanonicalOwnershipWitnessEdge, 16> witnessEdges;
  llvm::SmallVector<CanonicalMechanismId, 8> parentMechanisms;
  llvm::SmallVector<CanonicalDemandId, 16> witnessDemands;
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
  std::optional<CanonicalStructuralProposalId> structuralProposal;
};

/// One bounded graph/IR-derived suggestion for mechanisms worth grounding
/// together. A proposal does not assert coverage: the ordinary coverage
/// oracles populate groundedCoverage and additionalCoverage before the group
/// may become a set-cover candidate.
struct CanonicalStructuralProposal {
  CanonicalStructuralProposalId id = kInvalidCanonicalSyncId;
  CanonicalStructuralProposalKind kind =
      CanonicalStructuralProposalKind::LevelBoundary;
  CanonicalRegionId owner = kInvalidCanonicalSyncId;
  unsigned level = 0;
  std::string semanticKey;
  llvm::SmallVector<CanonicalMechanismId, 8> mechanisms;
  llvm::SmallVector<CanonicalDemandId, 8> crossingDemands;
  llvm::SmallVector<CanonicalDemandId, 8> singletonUnionCoverage;
  llvm::SmallVector<CanonicalDemandId, 8> groundedCoverage;
  llvm::SmallVector<CanonicalDemandId, 8> additionalCoverage;
  bool admitted = false;
};

struct CanonicalSetCoverCandidate {
  CanonicalSetCoverCandidateId id = kInvalidCanonicalSyncId;
  llvm::SmallVector<CanonicalMechanismId, 2> mechanisms;
  llvm::SmallVector<CanonicalDemandId, 4> directOrigins;
  llvm::SmallVector<CanonicalDemandId, 4> additionalCoverage;
  llvm::SmallVector<CanonicalDemandId, 8> coveredDemands;
  std::uint64_t weight = 0;
  std::optional<CanonicalStructuralProposalId> structuralProposal;
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

struct CanonicalSetCoverSolution {
  llvm::SmallVector<CanonicalSetCoverCandidateId, 8> greedyCandidates;
  llvm::SmallVector<CanonicalMechanismId, 8> mechanisms;
  llvm::SmallVector<CanonicalMechanismId, 8> reverseDeleted;
  llvm::SmallVector<CanonicalScarcityEventGroup, 2> scarcityEventGroups;
  std::uint64_t weight = 0;
  bool coverageVerified = false;
};

class CanonicalSyncProgram;
LogicalResult buildCanonicalDirectMechanisms(CanonicalSyncProgram &program);
LogicalResult proposeCanonicalSyncStructuralGroups(
    CanonicalSyncProgram &program,
    CanonicalStructuralCoverFamilies enabledFamilies);
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
  CanonicalDemandId appendDemand(CanonicalDemand demand);
  void retainDemands(const llvm::BitVector &retained);
  void appendDemandCause(CanonicalDemandId demand, CanonicalDemandCause cause);
  CanonicalMechanismId appendMechanism(CanonicalMechanism mechanism);
  CanonicalMechanismId
  appendOwnershipProtocol(CanonicalOwnershipProtocol protocol,
                          CanonicalMechanism mechanism);
  CanonicalStorageGenerationId
  appendStorageGeneration(CanonicalStorageGeneration generation);
  void appendMechanismOrigin(CanonicalMechanismId mechanism,
                             CanonicalDemandId demand);
  void appendMechanismCacheMaintenance(CanonicalMechanismId mechanism,
                                       llvm::ArrayRef<Operation *> actions);
  void setMechanismEventId(CanonicalMechanismId mechanism, unsigned eventId);
  void setMechanismReleaseEventId(CanonicalMechanismId mechanism,
                                  unsigned eventId);
  void setOwnershipLaneEventIds(CanonicalOwnershipProtocolId protocol,
                                unsigned lane, unsigned readyEventId,
                                unsigned releaseEventId);
  void setScarcityEventGroups(
      llvm::SmallVector<CanonicalScarcityEventGroup, 2> groups);
  void setDirectMechanism(CanonicalDemandId demand,
                          CanonicalMechanismId mechanism);
  CanonicalStructuralProposalId
  appendStructuralProposal(CanonicalStructuralProposal proposal);

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
  llvm::ArrayRef<CanonicalMechanism> getMechanisms() const {
    return mechanisms;
  }
  llvm::ArrayRef<CanonicalCoverageWorld> getCoverageWorlds() const {
    return coverageWorlds;
  }
  llvm::ArrayRef<CanonicalStructuralProposal> getStructuralProposals() const {
    return structuralProposals;
  }
  llvm::ArrayRef<CanonicalStorageGeneration> getStorageGenerations() const {
    return storageGenerations;
  }
  llvm::ArrayRef<CanonicalOwnershipProtocol> getOwnershipProtocols() const {
    return ownershipProtocols;
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
  const CanonicalMechanism &getMechanism(CanonicalMechanismId id) const;
  const CanonicalStorageGeneration &
  getStorageGeneration(CanonicalStorageGenerationId id) const;
  const CanonicalOwnershipProtocol &
  getOwnershipProtocol(CanonicalOwnershipProtocolId id) const;
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
  buildCanonicalDirectMechanisms(CanonicalSyncProgram &program);
  friend LogicalResult proposeCanonicalSyncStructuralGroups(
      CanonicalSyncProgram &program,
      CanonicalStructuralCoverFamilies enabledFamilies);
  friend LogicalResult
  evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
  friend LogicalResult
  buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);
  friend LogicalResult
  solveCanonicalSyncSetCover(CanonicalSyncProgram &program);

  void appendCoverageWorld(CanonicalCoverageWorld world);
  CanonicalMechanismId appendMechanismRecord(CanonicalMechanism mechanism);
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
  llvm::SmallVector<CanonicalMechanism> mechanisms;
  llvm::SmallVector<CanonicalStorageGeneration, 0> storageGenerations;
  llvm::SmallVector<CanonicalOwnershipProtocol, 0> ownershipProtocols;
  llvm::SmallVector<llvm::SmallVector<CanonicalRegionId, 2>, 0>
      mechanismExecutionLoops;
  llvm::SmallVector<llvm::SmallVector<CanonicalPhaseId, 8>, 0>
      mechanismSourcePrefixes;
  llvm::SmallVector<CanonicalMechanismId> directMechanisms;
  llvm::SmallVector<CanonicalCoverageWorld> coverageWorlds;
  llvm::SmallVector<CanonicalStructuralProposal, 0> structuralProposals;
  std::optional<CanonicalSetCoverInstance> setCoverInstance;
  std::optional<CanonicalSetCoverSolution> setCoverSolution;
  bool buildingMechanisms = false;
  bool mechanismCatalogComplete = false;
  bool structuralProposalCatalogComplete = false;
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
stringifyCanonicalVisibilityDirection(CanonicalVisibilityDirection direction);
llvm::StringRef
stringifyCanonicalCacheMaintenance(CanonicalCacheMaintenance maintenance);
llvm::StringRef stringifyCanonicalMechanismKind(CanonicalMechanismKind kind);
llvm::StringRef
stringifyCanonicalStructuralProposalKind(CanonicalStructuralProposalKind kind);
bool canonicalOwnershipProtocolCoversDemand(
    const CanonicalSyncProgram &program,
    const CanonicalOwnershipProtocol &protocol, const CanonicalDemand &demand);
void printCanonicalSyncProgram(const CanonicalSyncProgram &program,
                               llvm::raw_ostream &os);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCMODEL_H
