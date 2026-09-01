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
using CanonicalDemandId = std::uint32_t;
using CanonicalMechanismId = std::uint32_t;
using CanonicalSetCoverCandidateId = std::uint32_t;

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
  FixedFence,
  TailBarrier,
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
  CanonicalRegionId actionRegion = kInvalidCanonicalSyncId;
  llvm::SmallVector<CanonicalControlAtom, 2> guard;
  std::optional<unsigned> eventId;
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
  std::uint64_t weight = 0;
};

struct CanonicalSetCoverInstance {
  llvm::SmallVector<CanonicalMechanismId, 4> baseline;
  llvm::SmallVector<CanonicalDemandId, 8> universe;
  llvm::SmallVector<CanonicalSetCoverCandidate, 8> candidates;
};

class CanonicalSyncProgram;
LogicalResult buildCanonicalDirectMechanisms(CanonicalSyncProgram &program);
LogicalResult evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
LogicalResult buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);

class CanonicalSyncProgram {
public:
  explicit CanonicalSyncProgram(func::FuncOp function) : function(function) {}

  CanonicalRegionId appendRegion(CanonicalRegion region);
  CanonicalPhaseId appendPhase(CanonicalPhase phase);
  CanonicalAccessId appendAccess(CanonicalAccess access);
  CanonicalDemandId appendDemand(CanonicalDemand demand);
  void appendDemandCause(CanonicalDemandId demand, CanonicalDemandCause cause);
  CanonicalMechanismId appendMechanism(CanonicalMechanism mechanism);
  void appendMechanismOrigin(CanonicalMechanismId mechanism,
                             CanonicalDemandId demand);
  void setMechanismEventId(CanonicalMechanismId mechanism, unsigned eventId);
  void setDirectMechanism(CanonicalDemandId demand,
                          CanonicalMechanismId mechanism);

  LogicalResult freezeGraph();
  LogicalResult freeze();
  bool isGraphFrozen() const { return graphFrozen; }
  bool isFrozen() const { return frozen; }
  func::FuncOp getFunction() const { return function; }
  llvm::ArrayRef<CanonicalRegion> getRegions() const { return regions; }
  llvm::ArrayRef<CanonicalPhase> getPhases() const { return phases; }
  llvm::ArrayRef<CanonicalAccess> getAccesses() const { return accesses; }
  llvm::ArrayRef<CanonicalDemand> getDemands() const { return demands; }
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

  const CanonicalRegion &getRegion(CanonicalRegionId id) const;
  const CanonicalPhase &getPhase(CanonicalPhaseId id) const;
  const CanonicalAccess &getAccess(CanonicalAccessId id) const;
  const CanonicalDemand &getDemand(CanonicalDemandId id) const;
  const CanonicalMechanism &getMechanism(CanonicalMechanismId id) const;

private:
  friend LogicalResult
  buildCanonicalDirectMechanisms(CanonicalSyncProgram &program);
  friend LogicalResult
  evaluateCanonicalSyncCoverage(CanonicalSyncProgram &program);
  friend LogicalResult
  buildCanonicalSyncSetCoverInstance(CanonicalSyncProgram &program);

  void appendCoverageWorld(CanonicalCoverageWorld world);
  void setSetCoverInstance(CanonicalSetCoverInstance instance);

  func::FuncOp function;
  llvm::SmallVector<CanonicalRegion> regions;
  llvm::SmallVector<CanonicalPhase> phases;
  llvm::SmallVector<CanonicalAccess> accesses;
  llvm::SmallVector<CanonicalDemand> demands;
  llvm::SmallVector<CanonicalMechanism> mechanisms;
  llvm::SmallVector<CanonicalMechanismId> directMechanisms;
  llvm::SmallVector<CanonicalCoverageWorld> coverageWorlds;
  std::optional<CanonicalSetCoverInstance> setCoverInstance;
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
llvm::StringRef
stringifyCanonicalVisibilityDirection(CanonicalVisibilityDirection direction);
llvm::StringRef
stringifyCanonicalCacheMaintenance(CanonicalCacheMaintenance maintenance);
llvm::StringRef stringifyCanonicalMechanismKind(CanonicalMechanismKind kind);
void printCanonicalSyncProgram(const CanonicalSyncProgram &program,
                               llvm::raw_ostream &os);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCMODEL_H
