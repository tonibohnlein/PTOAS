// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/ScheduleGraph.h"
#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"

#include <iostream>
#include <string_view>

namespace {

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "ScheduleGraphTest failure: " << message << '\n';
  }
  return condition;
}

bool testEmptyGraph() {
  mlir::pto::ScheduleGraph graph;
  return check(graph.NumVertices() == 0, "empty vertex count") &&
         check(graph.NumEdges() == 0, "empty edge count") &&
         check(graph.NumVertexTypes() == 0, "empty type count") &&
         check(graph.IsAcyclic(), "empty graph is acyclic");
}

bool testVertexPropertiesAndEdges() {
  mlir::pto::ScheduleGraph graph;
  const auto source = graph.AddVertex(11, 12, 13, 2);
  const auto middle = graph.AddVertex(21, 22, 23, 5);
  const auto target = graph.AddVertex(31, 32, 33, 1);

  bool passed = true;
  passed &=
      check(source == 0 && middle == 1 && target == 2, "stable vertex ids");
  passed &= check(graph.NumVertexTypes() == 6, "type range includes gaps");
  passed &= check(graph.VertexWorkWeight(middle) == 21, "work weight");
  passed &= check(graph.VertexCommWeight(middle) == 22, "communication weight");
  passed &= check(graph.VertexMemWeight(middle) == 23, "memory weight");
  passed &= check(graph.VertexType(middle) == 5, "vertex type");
  passed &= check(graph.AddEdge(source, middle), "first edge insertion");
  passed &= check(graph.AddEdge(middle, target), "second edge insertion");
  passed &= check(graph.NumEdges() == 2, "edge count");
  passed &= check(graph.OutDegree(source) == 1, "source out degree");
  passed &= check(graph.InDegree(target) == 1, "target in degree");
  passed &= check(graph.Children(source).front() == middle, "child adjacency");
  passed &= check(graph.Parents(target).front() == middle, "parent adjacency");
  passed &= check(graph.IsAcyclic(), "chain is acyclic");
  return passed;
}

bool testInvalidAndDuplicateEdges() {
  mlir::pto::ScheduleGraph graph;
  const auto source = graph.AddVertex(1, 0, 0);
  const auto target = graph.AddVertex(1, 0, 0);

  bool passed = true;
  passed &= check(!graph.AddEdge(source, source), "self edge rejected");
  passed &= check(!graph.AddEdge(2, target), "invalid source rejected");
  passed &= check(!graph.AddEdge(source, 2), "invalid target rejected");
  passed &= check(graph.AddEdge(source, target), "valid edge accepted");
  passed &= check(!graph.AddEdge(source, target), "duplicate edge rejected");
  passed &= check(graph.NumEdges() == 1, "rejections do not change edge count");
  return passed;
}

bool testCycleDetection() {
  mlir::pto::ScheduleGraph graph;
  const auto first = graph.AddVertex(1, 0, 0);
  const auto second = graph.AddVertex(1, 0, 0);
  const auto third = graph.AddVertex(1, 0, 0);
  graph.AddEdge(first, second);
  graph.AddEdge(second, third);
  graph.AddEdge(third, first);
  return check(!graph.IsAcyclic(), "three-node cycle is detected");
}

bool testCompletePlacementLongestPath() {
  mlir::pto::KernelScheduleGraph graph;
  const auto first = graph.addNode(
      nullptr, mlir::pto::PIPE::PIPE_V, mlir::pto::ScheduleNodeKind::Compute,
      0, 0, 0, /*durationCycles=*/10, /*hasExactDuration=*/true);
  const auto second = graph.addNode(
      nullptr, mlir::pto::PIPE::PIPE_MTE2,
      mlir::pto::ScheduleNodeKind::Transfer, 1, 0, 0,
      /*durationCycles=*/20, /*hasExactDuration=*/true);
  const auto independent = graph.addNode(
      nullptr, mlir::pto::PIPE::PIPE_V, mlir::pto::ScheduleNodeKind::Compute,
      2, 0, 0, /*durationCycles=*/40, /*hasExactDuration=*/true);
  // This models a selected address reuse: a synchronization delay belongs to
  // the edge, and the whole placement is scored once through the DAG.
  graph.addDependency(first, second,
                      mlir::pto::ScheduleDependencyKind::PlacementReuseRAW,
                      0, nullptr, /*latencyCycles=*/17);
  // A recurrence contributes to a loop-II model, not the per-iteration path.
  graph.addDependency(second, first,
                      mlir::pto::ScheduleDependencyKind::MemoryWAR,
                      /*iterationDistance=*/1, nullptr,
                      /*latencyCycles=*/999);
  auto longestPath = graph.getLongestPathCycles();
  return check(succeeded(longestPath), "weighted graph remains acyclic") &&
         check(*longestPath == 47,
               "node and selected reuse-edge weights form the longest path") &&
         check(graph.getGraph().VertexWorkWeight(second) == 20,
               "node duration is exposed to DAG consumers") &&
         check(graph.getDependencies().back().latencyCycles == 999,
               "recurrence latency remains explicit metadata");
}

} // namespace

int main() {
  const bool passed = testEmptyGraph() && testVertexPropertiesAndEdges() &&
                      testInvalidAndDuplicateEdges() && testCycleDetection() &&
                      testCompletePlacementLongestPath();
  return passed ? 0 : 1;
}
