// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"

#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <tuple>

using namespace mlir;

namespace mlir {
namespace pto {

KernelScheduleGraph::VertexIdx
KernelScheduleGraph::addNode(Operation *operation, PIPE pipe,
                             ScheduleNodeKind kind, VertexIdx originalOrder,
                             VertexIdx block, unsigned loopDepth) {
  const VertexIdx id = graph_.AddVertex(
      /*workWeight=*/1, /*commWeight=*/0, /*memWeight=*/0,
      static_cast<ScheduleGraph::VertexTypeType>(pipe));
  nodes_.push_back(
      {id, operation, pipe, kind, originalOrder, block, loopDepth});
  return id;
}

void KernelScheduleGraph::addDependency(VertexIdx source, VertexIdx target,
                                        ScheduleDependencyKind kind,
                                        unsigned iterationDistance,
                                        Operation *recurrenceLoop) {
  const bool invalidSource = source >= nodes_.size();
  const bool invalidTarget = target >= nodes_.size();
  if (invalidSource || invalidTarget) {
    return;
  }
  if (iterationDistance == 0 && source == target) {
    return;
  }
  const bool duplicate = llvm::any_of(
      dependencies_, [&](const KernelScheduleDependency &dependency) {
        return dependency.source == source && dependency.target == target &&
               dependency.kind == kind &&
               dependency.iterationDistance == iterationDistance &&
               dependency.recurrenceLoop == recurrenceLoop;
      });
  if (duplicate) {
    return;
  }

  dependencies_.push_back(
      {source, target, kind, iterationDistance, recurrenceLoop});
  if (iterationDistance == 0) {
    graph_.AddEdge(source, target);
  }
}

StringRef stringifyScheduleNodeKind(ScheduleNodeKind kind) {
  switch (kind) {
  case ScheduleNodeKind::Compute:
    return "compute";
  case ScheduleNodeKind::Transfer:
    return "transfer";
  }
  return "unknown";
}

StringRef stringifyScheduleDependencyKind(ScheduleDependencyKind kind) {
  switch (kind) {
  case ScheduleDependencyKind::SSA:
    return "ssa";
  case ScheduleDependencyKind::MemoryRAW:
    return "memory-raw";
  case ScheduleDependencyKind::MemoryWAR:
    return "memory-war";
  case ScheduleDependencyKind::MemoryWAW:
    return "memory-waw";
  case ScheduleDependencyKind::Control:
    return "control";
  case ScheduleDependencyKind::LoopCarriedSSA:
    return "loop-carried-ssa";
  }
  return "unknown";
}

static unsigned getLoopDepth(Operation *loop) {
  unsigned depth = 1;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<LoopLikeOpInterface>(parent)) {
      ++depth;
    }
  }
  return depth;
}

static SmallVector<const KernelScheduleDependency *>
getSortedDependencies(const KernelScheduleGraph &graph) {
  SmallVector<const KernelScheduleDependency *> sorted;
  for (const KernelScheduleDependency &dependency : graph.getDependencies()) {
    sorted.push_back(&dependency);
  }
  llvm::sort(sorted, [](const KernelScheduleDependency *lhs,
                        const KernelScheduleDependency *rhs) {
    const unsigned lhsLoopDepth =
        lhs->recurrenceLoop ? getLoopDepth(lhs->recurrenceLoop) : 0;
    const unsigned rhsLoopDepth =
        rhs->recurrenceLoop ? getLoopDepth(rhs->recurrenceLoop) : 0;
    return std::tie(lhs->source, lhs->target, lhs->iterationDistance, lhs->kind,
                    lhsLoopDepth) < std::tie(rhs->source, rhs->target,
                                             rhs->iterationDistance, rhs->kind,
                                             rhsLoopDepth);
  });
  return sorted;
}

void printKernelScheduleGraph(llvm::raw_ostream &os, func::FuncOp func,
                              const KernelScheduleGraph &graph) {
  os << "KernelScheduleGraph @" << func.getSymName()
     << " nodes=" << graph.getGraph().NumVertices()
     << " dag_edges=" << graph.getGraph().NumEdges()
     << " dependencies=" << graph.getDependencies().size() << "\n";
  for (const KernelScheduleNode &node : graph.getNodes()) {
    os << "  node[" << node.id
       << "] op=" << node.operation->getName().getStringRef()
       << " pipe=" << stringifyPIPE(node.pipe)
       << " kind=" << stringifyScheduleNodeKind(node.kind)
       << " order=" << node.originalOrder << " block=" << node.block
       << " loop_depth=" << node.loopDepth << "\n";
  }
  for (const KernelScheduleDependency *dependency :
       getSortedDependencies(graph)) {
    os << "  edge " << dependency->source << " -> " << dependency->target
       << " kind=" << stringifyScheduleDependencyKind(dependency->kind)
       << " distance=" << dependency->iterationDistance;
    if (dependency->recurrenceLoop) {
      os << " recurrence_loop_depth="
         << getLoopDepth(dependency->recurrenceLoop);
    }
    os << "\n";
  }
}

static void printDotQuoted(llvm::raw_ostream &os, StringRef value) {
  os << '"';
  for (char character : value) {
    if (character == '\n') {
      os << "\\n";
      continue;
    }
    if (character == '"' || character == '\\') {
      os << '\\';
    }
    os << character;
  }
  os << '"';
}

static StringRef getDependencyColor(ScheduleDependencyKind kind) {
  switch (kind) {
  case ScheduleDependencyKind::SSA:
    return "#4C566A";
  case ScheduleDependencyKind::MemoryRAW:
    return "#2E7D32";
  case ScheduleDependencyKind::MemoryWAR:
    return "#B26A00";
  case ScheduleDependencyKind::MemoryWAW:
    return "#9C2F45";
  case ScheduleDependencyKind::Control:
    return "#6B5CA5";
  case ScheduleDependencyKind::LoopCarriedSSA:
    return "#007C91";
  }
  return "#4C566A";
}

void printKernelScheduleGraphDot(llvm::raw_ostream &os, func::FuncOp func,
                                 const KernelScheduleGraph &graph) {
  os << "digraph ";
  printDotQuoted(os, func.getSymName());
  os << " {\n"
        "  rankdir=LR;\n"
        "  graph [bgcolor=\"white\", pad=\"0.2\", nodesep=\"0.35\", "
        "ranksep=\"0.7\"];\n"
        "  node [fontname=\"DejaVu Sans\", fontsize=10, shape=box, "
        "style=\"filled,rounded\", color=\"#59636E\"];\n"
        "  edge [fontname=\"DejaVu Sans\", fontsize=8, arrowsize=0.7];\n";
  for (const KernelScheduleNode &node : graph.getNodes()) {
    os << "  n" << node.id << " [label=";
    const std::string label = (llvm::Twine(node.id) + ": " +
                               node.operation->getName().getStringRef() + "\n" +
                               stringifyPIPE(node.pipe) + " / " +
                               stringifyScheduleNodeKind(node.kind))
                                  .str();
    printDotQuoted(os, label);
    os << ", fillcolor=\""
       << (node.kind == ScheduleNodeKind::Compute ? "#FDE7C8" : "#DCEBFA")
       << "\"];\n";
  }
  for (const KernelScheduleDependency *dependency :
       getSortedDependencies(graph)) {
    os << "  n" << dependency->source << " -> n" << dependency->target
       << " [label=";
    std::string label = stringifyScheduleDependencyKind(dependency->kind).str();
    if (dependency->iterationDistance != 0) {
      label +=
          (llvm::Twine(", d=") + llvm::Twine(dependency->iterationDistance))
              .str();
      if (dependency->recurrenceLoop) {
        label += (llvm::Twine(", loop-depth=") +
                  llvm::Twine(getLoopDepth(dependency->recurrenceLoop)))
                     .str();
      }
    }
    printDotQuoted(os, label);
    os << ", color=\"" << getDependencyColor(dependency->kind)
       << "\", fontcolor=\"" << getDependencyColor(dependency->kind) << '"';
    if (dependency->iterationDistance != 0) {
      os << ", style=dashed";
    }
    os << "];\n";
  }
  os << "}\n";
}

} // namespace pto
} // namespace mlir
