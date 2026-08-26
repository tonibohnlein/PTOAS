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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

enum class ScheduleNodeKind : std::uint8_t { Compute, Transfer };

enum class ScheduleDependencyKind : std::uint8_t {
  SSA,
  MemoryRAW,
  MemoryWAR,
  MemoryWAW,
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
};

struct KernelScheduleDependency {
  ScheduleGraph::VertexIdx source = 0;
  ScheduleGraph::VertexIdx target = 0;
  ScheduleDependencyKind kind = ScheduleDependencyKind::SSA;
  unsigned iterationDistance = 0;
  Operation *recurrenceLoop = nullptr;
};

/// Owns the structural scheduling DAG and the richer dependency metadata.
/// Positive-distance recurrence dependencies are metadata only: adding them to
/// the adjacency graph would turn the per-iteration computational DAG cyclic.
class KernelScheduleGraph {
public:
  using VertexIdx = ScheduleGraph::VertexIdx;

  VertexIdx addNode(Operation *operation, PIPE pipe, ScheduleNodeKind kind,
                    VertexIdx originalOrder, VertexIdx block,
                    unsigned loopDepth);
  void addDependency(VertexIdx source, VertexIdx target,
                     ScheduleDependencyKind kind,
                     unsigned iterationDistance = 0,
                     Operation *recurrenceLoop = nullptr);

  const ScheduleGraph &getGraph() const { return graph_; }
  ArrayRef<KernelScheduleNode> getNodes() const { return nodes_; }
  ArrayRef<KernelScheduleDependency> getDependencies() const {
    return dependencies_;
  }
  const KernelScheduleNode &getNode(VertexIdx id) const { return nodes_[id]; }
  bool isAcyclic() const { return graph_.IsAcyclic(); }

private:
  ScheduleGraph graph_;
  std::vector<KernelScheduleNode> nodes_;
  std::vector<KernelScheduleDependency> dependencies_;
};

FailureOr<KernelScheduleGraph> buildKernelScheduleGraph(func::FuncOp func);

StringRef stringifyScheduleNodeKind(ScheduleNodeKind kind);
StringRef stringifyScheduleDependencyKind(ScheduleDependencyKind kind);
void printKernelScheduleGraph(llvm::raw_ostream &os, func::FuncOp func,
                              const KernelScheduleGraph &graph);
void printKernelScheduleGraphDot(llvm::raw_ostream &os, func::FuncOp func,
                                 const KernelScheduleGraph &graph);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_KERNELSCHEDULEGRAPH_H
