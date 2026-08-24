// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- CanonicalSync.h - Canonical synchronization planning ------*- C++
//-*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
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

enum class CanonicalOwnershipKind : std::uint8_t {
  L0Operand,
  L1Tile,
  L0Accumulator,
};

enum class CanonicalOwnershipProtocolKind : std::uint8_t {
  RoundTrip,
  AlternatingPrefetch,
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

struct CanonicalDependency {
  std::size_t source = 0;
  std::size_t target = 0;
  CanonicalDependencyKind kind = CanonicalDependencyKind::SSA;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
  bool active = true;
  bool possible = true;
  bool retained = true;
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
  CanonicalAnchor initialReadyAnchor;
  unsigned initialReadyLane = 0;
  SmallVector<unsigned, 2> initiallyFreeLanes;
};

struct CanonicalRecurrenceScope {
  std::size_t id = 0;
  Operation *operation = nullptr;
  std::size_t operationOrder = 0;
  std::size_t parentScope = 0;
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

struct CanonicalEventAction {
  CanonicalEventActionKind kind = CanonicalEventActionKind::Set;
  CanonicalEventActionPhase phase = CanonicalEventActionPhase::Straight;
  CanonicalAnchor anchor;
  CanonicalEventLane lane;
  Operation *nonEmptyLoopGuard = nullptr;
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
  std::size_t protocolBundle = 0;
  std::size_t ownershipCycle = 0;
  CanonicalOwnershipEventRole ownershipRole =
      CanonicalOwnershipEventRole::None;
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
  ArrayRef<CanonicalRecurrenceScope> getRecurrenceScopes() const {
    return recurrenceScopes_;
  }
  ArrayRef<CanonicalBarrier> getBarriers() const { return barriers_; }
  ArrayRef<CanonicalEvent> getEvents() const { return events_; }
  ArrayRef<CanonicalEventDomain> getDomains() const { return domains_; }
  ArrayRef<CanonicalOwnershipCycle> getOwnershipCycles() const {
    return ownershipCycles_;
  }
  bool usedInfeasibleBootstrap() const { return usedInfeasibleBootstrap_; }

private:
  friend class CanonicalSyncPlanBuilder;
  friend FailureOr<CanonicalSyncPlan> buildCanonicalSyncPlan(func::FuncOp,
                                                             unsigned);

  std::vector<CanonicalSyncNode> nodes_;
  std::vector<SyncGraphEdge> fixedEdges_;
  std::vector<CanonicalDependency> dependencies_;
  std::vector<CanonicalDependency> conservativeCompletionRequirements_;
  std::vector<CanonicalDependency> completionRequirements_;
  std::vector<CanonicalRecurrenceScope> recurrenceScopes_;
  std::vector<CanonicalBarrier> barriers_;
  std::vector<CanonicalEvent> events_;
  std::vector<CanonicalEventDomain> domains_;
  std::vector<CanonicalOwnershipCycle> ownershipCycles_;
  bool usedInfeasibleBootstrap_ = false;
};

FailureOr<CanonicalSyncPlan> buildCanonicalSyncPlan(func::FuncOp func,
                                                    unsigned eventIdMax);
LogicalResult emitCanonicalSyncPlan(func::FuncOp func,
                                    const CanonicalSyncPlan &plan);

StringRef stringifyCanonicalDependencyKind(CanonicalDependencyKind kind);
StringRef stringifyCanonicalOwnershipKind(CanonicalOwnershipKind kind);
void printCanonicalSyncPlan(llvm::raw_ostream &os, func::FuncOp func,
                            const CanonicalSyncPlan &plan, StringRef view);
void printCanonicalSyncPlanDot(llvm::raw_ostream &os, func::FuncOp func,
                               const CanonicalSyncPlan &plan, StringRef view);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
