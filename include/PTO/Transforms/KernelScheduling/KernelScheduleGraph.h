// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- KernelScheduleGraph.h - Kernel scheduling graph ----------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_KERNELSCHEDULEGRAPH_H
#define MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_KERNELSCHEDULEGRAPH_H

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/KernelScheduling/ScheduleGraph.h"
#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

enum class ScheduleNodeKind : std::uint8_t { Compute, Transfer };

enum class ScheduleDependencyKind : std::uint8_t {
  SSA,
  MemoryRAW,
  MemoryWAR,
  MemoryWAW,
  /// A placement-induced hazard between distinct logical allocations. These
  /// are not emitted by the base SSA graph; reuse-penalty analysis adds them
  /// for the complete DSA placement being evaluated.
  PlacementReuseRAW,
  PlacementReuseWAR,
  PlacementReuseWAW,
  Control,
  LoopCarriedSSA,
};

struct KernelScheduleNode {
  ScheduleGraph::VertexIdx id = 0;
  Operation *operation = nullptr;
  PIPE pipe = PIPE::PIPE_UNASSIGNED;
  ScheduleNodeKind kind = ScheduleNodeKind::Compute;
  ScheduleGraph::VertexIdx originalOrder = 0;
  ScheduleGraph::VertexIdx block = 0;
  unsigned loopDepth = 0;
  /// PyPTO's source-level access identity, when carried by the PTO location.
  /// It is the stable join key for external DSA reuse-candidate provenance.
  std::optional<unsigned> pyptoAccessOrder;
  /// Exact durations are supplied by a pinned PTO-ISA formula table. The
  /// legacy structural graph intentionally retains unit weights.
  uint64_t durationCycles = 1;
  bool hasExactDuration = false;
  /// Present exactly when PTO-ISA supplied the node duration.
  std::optional<PTOISADurationSignature> durationSignature;
};

struct KernelScheduleGraphBuildOptions {
  /// Null keeps the existing structural graph and its unit node weights.
  const PTOISADurationTable *durationTable = nullptr;
  /// Required for latency scoring: do not silently use a unit/family fallback.
  bool requireExactDurations = false;
};

struct KernelScheduleDependency {
  ScheduleGraph::VertexIdx source = 0;
  ScheduleGraph::VertexIdx target = 0;
  ScheduleDependencyKind kind = ScheduleDependencyKind::SSA;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
  /// Delay after the source completes before the target can start. Structural
  /// SSA/control edges are zero; placement-induced reuse edges carry the
  /// globally calibrated synchronization delay used by the penalty model.
  uint64_t latencyCycles = 0;
  /// DSA analysis evidence that selected this placement-induced dependency.
  std::string provenance;
};

/// Owns the structural scheduling DAG and the richer dependency metadata.
/// Positive-distance recurrence dependencies are metadata only: adding them to
/// the adjacency graph would turn the per-iteration computational DAG cyclic.
class KernelScheduleGraph {
public:
  using VertexIdx = ScheduleGraph::VertexIdx;

  VertexIdx addNode(Operation *operation, PIPE pipe, ScheduleNodeKind kind,
                    VertexIdx originalOrder, VertexIdx block,
                    unsigned loopDepth, uint64_t durationCycles = 1,
                    bool hasExactDuration = false,
                    std::optional<PTOISADurationSignature> durationSignature = std::nullopt,
                    std::optional<unsigned> pyptoAccessOrder = std::nullopt);
  void addDependency(VertexIdx source, VertexIdx target,
                     ScheduleDependencyKind kind,
                     unsigned iterationDistance = 0,
                     Operation *recurrenceLoop = nullptr,
                     uint64_t latencyCycles = 0,
                     llvm::StringRef provenance = {});

  const ScheduleGraph &getGraph() const { return graph_; }
  ArrayRef<KernelScheduleNode> getNodes() const { return nodes_; }
  ArrayRef<KernelScheduleDependency> getDependencies() const {
    return dependencies_;
  }
  const KernelScheduleNode &getNode(VertexIdx id) const { return nodes_[id]; }
  bool isAcyclic() const { return graph_.IsAcyclic(); }
  /// Longest zero-distance path using both operation and dependency latency.
  /// Positive-distance recurrences describe a loop initiation interval and are
  /// intentionally excluded from this per-iteration DAG score.
  FailureOr<uint64_t> getLongestPathCycles() const;

private:
  ScheduleGraph graph_;
  std::vector<KernelScheduleNode> nodes_;
  std::vector<KernelScheduleDependency> dependencies_;
};

FailureOr<KernelScheduleGraph>
buildKernelScheduleGraph(func::FuncOp func,
                         KernelScheduleGraphBuildOptions options = {});

StringRef stringifyScheduleNodeKind(ScheduleNodeKind kind);
StringRef stringifyScheduleDependencyKind(ScheduleDependencyKind kind);
void printKernelScheduleGraph(llvm::raw_ostream &os, func::FuncOp func,
                              const KernelScheduleGraph &graph);
void printKernelScheduleGraphDot(llvm::raw_ostream &os, func::FuncOp func,
                                 const KernelScheduleGraph &graph);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_KERNELSCHEDULEGRAPH_H
