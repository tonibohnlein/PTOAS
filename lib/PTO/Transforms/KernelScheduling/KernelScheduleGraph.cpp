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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>

using namespace mlir;

namespace mlir {
namespace pto {

namespace {

FailureOr<uint64_t> addCycles(uint64_t lhs, uint64_t rhs)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return failure();
    }
    return lhs + rhs;
}

FailureOr<std::vector<KernelScheduleGraph::VertexIdx>> getTopologicalOrder(const ScheduleGraph& graph)
{
    std::vector<KernelScheduleGraph::VertexIdx> indegrees;
    std::vector<KernelScheduleGraph::VertexIdx> ready;
    indegrees.reserve(graph.NumVertices());
    ready.reserve(graph.NumVertices());
    for (KernelScheduleGraph::VertexIdx node : graph.Vertices()) {
        indegrees.push_back(graph.InDegree(node));
        if (indegrees.back() == 0) {
            ready.push_back(node);
        }
    }
    for (std::size_t index = 0; index < ready.size(); ++index) {
        for (KernelScheduleGraph::VertexIdx target : graph.Children(ready[index])) {
            if (--indegrees[target] == 0) {
                ready.push_back(target);
            }
        }
    }
    if (ready.size() != graph.NumVertices()) {
        return failure();
    }
    return ready;
}

bool isResolvedDurationEvidenceClass(llvm::StringRef evidenceClass)
{
    return llvm::StringSwitch<bool>(evidenceClass)
        .Cases("calibrated_signature", "calibrated_compatible_encoding", true)
        .Cases("calibrated_operation_family", "calibrated_formula_signature", true)
        .Cases("calibrated_instruction_model", "pinned_analytical_model", true)
        .Cases("pinned_perf_sim_approximation", "pinned_formula_shape_approximation", true)
        .Case("pinned_formula_family_approximation", true)
        .Default(false);
}

using EdgeLatencyMap =
    llvm::DenseMap<std::pair<KernelScheduleGraph::VertexIdx, KernelScheduleGraph::VertexIdx>, uint64_t>;

EdgeLatencyMap getZeroDistanceEdgeLatencies(ArrayRef<KernelScheduleDependency> dependencies)
{
    EdgeLatencyMap result;
    for (const KernelScheduleDependency& dependency : dependencies) {
        if (dependency.iterationDistance != 0) {
            continue;
        }
        auto key = std::make_pair(dependency.source, dependency.target);
        auto [it, inserted] = result.try_emplace(key, dependency.latencyCycles);
        if (!inserted) {
            it->second = std::max(it->second, dependency.latencyCycles);
        }
    }
    return result;
}

} // namespace

KernelScheduleGraph::VertexIdx KernelScheduleGraph::addNode(
    Operation* operation, PIPE pipe, ScheduleNodeKind kind, VertexIdx originalOrder, VertexIdx block,
    unsigned loopDepth, uint64_t durationCycles, bool hasResolvedDuration,
    std::optional<PTOISADurationSignature> durationSignature, std::optional<unsigned> pyptoAccessOrder)
{
    const VertexIdx id = graph_.AddVertex(
        /*workWeight=*/durationCycles, /*commWeight=*/0, /*memWeight=*/0,
        static_cast<ScheduleGraph::VertexTypeType>(pipe));
    nodes_.push_back(
        {id,
         operation,
         pipe,
         kind,
         originalOrder,
         block,
         loopDepth,
         pyptoAccessOrder,
         durationCycles,
         hasResolvedDuration,
         std::move(durationSignature),
         {},
         hasResolvedDuration ? "calibrated_formula_signature" : "unresolved"});
    return id;
}

LogicalResult KernelScheduleGraph::setNodeDuration(
    VertexIdx id, uint64_t durationCycles, llvm::StringRef provenance, llvm::StringRef evidenceClass)
{
    if (id >= nodes_.size() || provenance.empty() || !isResolvedDurationEvidenceClass(evidenceClass) ||
        !graph_.SetVertexWorkWeight(id, durationCycles)) {
        return failure();
    }
    KernelScheduleNode& node = nodes_[id];
    node.durationCycles = durationCycles;
    node.hasResolvedDuration = true;
    node.durationSignature.reset();
    node.durationProvenance = provenance.str();
    node.durationEvidenceClass = evidenceClass.str();
    return success();
}

void KernelScheduleGraph::addDependency(VertexIdx source, VertexIdx target,
                                        ScheduleDependencyKind kind,
                                        unsigned iterationDistance,
                                        Operation *recurrenceLoop,
                                        uint64_t latencyCycles,
                                        llvm::StringRef provenance) {
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
               dependency.recurrenceLoop == recurrenceLoop &&
               dependency.latencyCycles == latencyCycles &&
               dependency.provenance == provenance;
      });
  if (duplicate) {
    return;
  }

  dependencies_.push_back(
      {source, target, kind, iterationDistance, recurrenceLoop,
       latencyCycles, provenance.str()});
  if (iterationDistance == 0) {
    graph_.AddEdge(source, target);
  }
}

FailureOr<uint64_t> KernelScheduleGraph::getLongestPathCycles() const {
  if (graph_.NumVertices() == 0) {
    return uint64_t{0};
  }
  FailureOr<std::vector<VertexIdx>> order = getTopologicalOrder(graph_);
  if (failed(order)) {
      return failure();
  }
  std::vector<uint64_t> distances(graph_.NumVertices(), 0);
  const EdgeLatencyMap edgeLatencies = getZeroDistanceEdgeLatencies(dependencies_);
  for (VertexIdx source : *order) {
      distances[source] = std::max(distances[source], graph_.VertexWorkWeight(source));
      for (VertexIdx target : graph_.Children(source)) {
          const auto latency = edgeLatencies.find(std::make_pair(source, target));
          const uint64_t edgeLatency = latency == edgeLatencies.end() ? 0 : latency->second;
          FailureOr<uint64_t> withEdge = addCycles(distances[source], edgeLatency);
          if (failed(withEdge)) {
              return failure();
          }
          FailureOr<uint64_t> withTarget = addCycles(*withEdge, graph_.VertexWorkWeight(target));
          if (failed(withTarget)) {
              return failure();
          }
          distances[target] = std::max(distances[target], *withTarget);
      }
  }
  return *std::max_element(distances.begin(), distances.end());
}

FailureOr<uint64_t> KernelScheduleGraph::getRecurrenceInitiationIntervalCycles() const
{
    FailureOr<std::vector<VertexIdx>> order = getTopologicalOrder(graph_);
    if (failed(order)) {
        return failure();
    }
    const EdgeLatencyMap edgeLatencies = getZeroDistanceEdgeLatencies(dependencies_);
    std::map<VertexIdx, std::vector<std::optional<uint64_t>>> returnPathsByTarget;
    uint64_t recurrenceII = 0;
    for (const KernelScheduleDependency& recurrence : dependencies_) {
        if (recurrence.iterationDistance == 0) {
            continue;
        }
        auto [cached, inserted] = returnPathsByTarget.try_emplace(recurrence.target);
        if (inserted) {
            std::vector<std::optional<uint64_t>>& distances = cached->second;
            distances.resize(graph_.NumVertices());
            distances[recurrence.target] = graph_.VertexWorkWeight(recurrence.target);
            for (VertexIdx source : *order) {
                if (!distances[source]) {
                    continue;
                }
                for (VertexIdx target : graph_.Children(source)) {
                    const auto latency = edgeLatencies.find(std::make_pair(source, target));
                    const uint64_t edgeLatency = latency == edgeLatencies.end() ? 0 : latency->second;
                    FailureOr<uint64_t> withEdge = addCycles(*distances[source], edgeLatency);
                    if (failed(withEdge)) {
                        return failure();
                    }
                    FailureOr<uint64_t> withTarget = addCycles(*withEdge, graph_.VertexWorkWeight(target));
                    if (failed(withTarget)) {
                        return failure();
                    }
                    if (!distances[target] || *withTarget > *distances[target]) {
                        distances[target] = *withTarget;
                    }
                }
            }
        }
        const std::vector<std::optional<uint64_t>>& distances = cached->second;
        if (!distances[recurrence.source]) {
            continue;
        }
        FailureOr<uint64_t> cycle = addCycles(*distances[recurrence.source], recurrence.latencyCycles);
        if (failed(cycle)) {
            return failure();
        }
        const uint64_t distance = recurrence.iterationDistance;
        const uint64_t bound = *cycle / distance + (*cycle % distance != 0);
        recurrenceII = std::max(recurrenceII, bound);
    }
    return recurrenceII;
}

FailureOr<uint64_t> KernelScheduleGraph::getLatencyLowerBoundCycles() const
{
    FailureOr<uint64_t> longestPath = getLongestPathCycles();
    FailureOr<uint64_t> recurrenceII = getRecurrenceInitiationIntervalCycles();
    if (failed(longestPath) || failed(recurrenceII)) {
        return failure();
    }
    return std::max(*longestPath, *recurrenceII);
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
  case ScheduleDependencyKind::PlacementReuseRAW:
    return "placement-reuse-raw";
  case ScheduleDependencyKind::PlacementReuseWAR:
    return "placement-reuse-war";
  case ScheduleDependencyKind::PlacementReuseWAW:
    return "placement-reuse-waw";
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
      os << "  node[" << node.id << "] op=" << node.operation->getName().getStringRef()
         << " pipe=" << stringifyPIPE(node.pipe) << " kind=" << stringifyScheduleNodeKind(node.kind)
         << " order=" << node.originalOrder << " block=" << node.block << " loop_depth=" << node.loopDepth
         << " duration_cycles=" << node.durationCycles
         << " duration_resolved=" << (node.hasResolvedDuration ? "true" : "false")
         << " duration_evidence_class=" << node.durationEvidenceClass;
      if (node.pyptoAccessOrder) {
          os << " pypto_access_order=" << *node.pyptoAccessOrder;
      }
    if (node.durationSignature) {
      os << " duration_signature=" << node.durationSignature->opcode << ":"
         << node.durationSignature->dtype << ":"
         << node.durationSignature->rows << "x"
         << node.durationSignature->cols;
    }
    if (!node.durationProvenance.empty()) {
      os << " duration_provenance=" << node.durationProvenance;
    }
    os << "\n";
  }
  for (const KernelScheduleDependency *dependency :
       getSortedDependencies(graph)) {
    os << "  edge " << dependency->source << " -> " << dependency->target
       << " kind=" << stringifyScheduleDependencyKind(dependency->kind)
       << " distance=" << dependency->iterationDistance
       << " latency_cycles=" << dependency->latencyCycles;
    if (!dependency->provenance.empty()) {
      os << " provenance=" << dependency->provenance;
    }
    if (dependency->recurrenceLoop) {
      os << " recurrence_loop_depth="
         << getLoopDepth(dependency->recurrenceLoop);
    }
    os << "\n";
  }
  FailureOr<uint64_t> longestPath = graph.getLongestPathCycles();
  if (succeeded(longestPath)) {
      os << "  longest_path_cycles=" << *longestPath << "\n";
  } else {
      os << "  longest_path_cycles=unavailable_cycle\n";
  }
  FailureOr<uint64_t> recurrenceII = graph.getRecurrenceInitiationIntervalCycles();
  if (succeeded(recurrenceII)) {
      os << "  recurrence_ii_cycles=" << *recurrenceII << "\n";
  } else {
      os << "  recurrence_ii_cycles=unavailable_cycle\n";
  }
  if (succeeded(longestPath) && succeeded(recurrenceII)) {
      os << "  latency_lower_bound_cycles=" << std::max(*longestPath, *recurrenceII) << "\n";
  } else {
      os << "  latency_lower_bound_cycles=unavailable_cycle\n";
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
  case ScheduleDependencyKind::PlacementReuseRAW:
    return "#00796B";
  case ScheduleDependencyKind::PlacementReuseWAR:
    return "#E65100";
  case ScheduleDependencyKind::PlacementReuseWAW:
    return "#AD1457";
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
    if (dependency->latencyCycles != 0) {
      label += (llvm::Twine(", latency=") +
                llvm::Twine(dependency->latencyCycles))
                   .str();
    }
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
