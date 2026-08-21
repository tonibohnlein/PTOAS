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
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
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
  SmallVector<CanonicalMemoryAccess, 4> accesses;
};

struct CanonicalDependency {
  std::size_t source = 0;
  std::size_t target = 0;
  CanonicalDependencyKind kind = CanonicalDependencyKind::SSA;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
  bool retained = true;
};

struct CanonicalAnchor {
  Operation *operation = nullptr;
  bool before = true;
};

struct CanonicalBarrier {
  PipelineType pipe = PipelineType::PIPE_UNASSIGNED;
  CanonicalAnchor anchor;
  Operation *recurrenceLoop = nullptr;
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
  Value setSlot;
  Value waitSlot;
  unsigned width = 1;
  SmallVector<unsigned, 2> eventIds;
  std::size_t intervalBegin = 0;
  std::size_t intervalEnd = 0;
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
  ArrayRef<CanonicalBarrier> getBarriers() const { return barriers_; }
  ArrayRef<CanonicalEvent> getEvents() const { return events_; }
  ArrayRef<CanonicalEventDomain> getDomains() const { return domains_; }

private:
  friend class CanonicalSyncPlanBuilder;
  friend FailureOr<CanonicalSyncPlan> buildCanonicalSyncPlan(func::FuncOp,
                                                             unsigned);

  std::vector<CanonicalSyncNode> nodes_;
  std::vector<SyncGraphEdge> fixedEdges_;
  std::vector<CanonicalDependency> dependencies_;
  std::vector<CanonicalBarrier> barriers_;
  std::vector<CanonicalEvent> events_;
  std::vector<CanonicalEventDomain> domains_;
};

FailureOr<CanonicalSyncPlan> buildCanonicalSyncPlan(func::FuncOp func,
                                                    unsigned eventIdMax);
LogicalResult emitCanonicalSyncPlan(func::FuncOp func,
                                    const CanonicalSyncPlan &plan);

StringRef stringifyCanonicalDependencyKind(CanonicalDependencyKind kind);
void printCanonicalSyncPlan(llvm::raw_ostream &os, func::FuncOp func,
                            const CanonicalSyncPlan &plan, StringRef view);
void printCanonicalSyncPlanDot(llvm::raw_ostream &os, func::FuncOp func,
                               const CanonicalSyncPlan &plan, StringRef view);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
