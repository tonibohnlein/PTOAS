// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"

#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using mlir::pto::CompletionRequirement;
using mlir::pto::IntervalColoring;
using mlir::pto::SyncConflictGraph;
using mlir::pto::SyncGraphEdge;
using mlir::pto::SyncGraphEdgeKind;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncAlgorithmsTest failure: " << message << '\n';
  }
  return condition;
}

SyncConflictGraph makeGraph(std::size_t vertices,
                            const std::vector<std::pair<int, int>> &edges) {
  SyncConflictGraph graph;
  for (std::size_t vertex = 0; vertex < vertices; ++vertex) {
    graph.addVertex();
  }
  for (auto [first, second] : edges) {
    graph.addEdge(static_cast<std::size_t>(first),
                  static_cast<std::size_t>(second));
  }
  return graph;
}

bool testIntervalRecognitionAndOptimalColoring() {
  SyncConflictGraph path = makeGraph(4, {{0, 1}, {1, 2}, {2, 3}});
  IntervalColoring pathColoring = mlir::pto::colorIntervalGraph(path);
  bool passed =
      check(pathColoring.isInterval, "path is interval") &&
      check(pathColoring.coloring.colorCount == 2, "path needs two colors") &&
      check(mlir::pto::isValidColoring(path, pathColoring.coloring),
            "path coloring is valid");

  SyncConflictGraph clique =
      makeGraph(4, {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}});
  IntervalColoring cliqueColoring = mlir::pto::colorIntervalGraph(clique);
  passed &= check(cliqueColoring.isInterval, "clique is interval");
  passed &= check(cliqueColoring.coloring.colorCount == 4,
                  "four-clique needs four colors");
  return passed;
}

bool testNonIntervalCertificates() {
  SyncConflictGraph cycle = makeGraph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}});
  bool passed = check(!mlir::pto::colorIntervalGraph(cycle).isInterval,
                      "induced four-cycle is not interval");

  SyncConflictGraph threeSun = makeGraph(
      6,
      {{0, 1}, {1, 2}, {2, 0}, {3, 0}, {3, 1}, {4, 1}, {4, 2}, {5, 2}, {5, 0}});
  passed &= check(!mlir::pto::colorIntervalGraph(threeSun).isInterval,
                  "three-sun is chordal but not interval");
  return passed;
}

bool testHeuristicColorings() {
  SyncConflictGraph graph = makeGraph(
      6, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}, {3, 4}, {3, 5}, {4, 5}});
  const auto dsatur = mlir::pto::colorDsatur(graph);
  const auto smallestLast = mlir::pto::colorSmallestLast(graph);
  bool passed = check(mlir::pto::isValidColoring(graph, dsatur),
                      "DSATUR coloring is valid") &&
                check(mlir::pto::isValidColoring(graph, smallestLast),
                      "smallest-last coloring is valid");

  SyncConflictGraph witness = makeGraph(
      6,
      {{0, 1}, {0, 2}, {0, 4}, {1, 4}, {1, 5}, {2, 3}, {2, 5}, {3, 4}, {3, 5}});
  const auto witnessDsatur = mlir::pto::colorDsatur(witness);
  const auto witnessSmallestLast = mlir::pto::colorSmallestLast(witness);
  passed &= check(witnessDsatur.colorCount == 3,
                  "DSATUR uses three colors on the witness");
  passed &= check(witnessSmallestLast.colorCount == 4,
                  "smallest-last uses four colors on the witness");
  return passed;
}

bool testCompletionQualifiedReduction() {
  const std::vector<SyncGraphEdge> issueEdges = {
      {0, 1, SyncGraphEdgeKind::IssueOrder}};
  const std::vector<CompletionRequirement> requirements = {{1, 2}, {0, 2}};
  const std::vector<bool> keep =
      mlir::pto::reduceCompletionRequirements(3, issueEdges, requirements);
  bool passed = check(keep.size() == 2 && keep[0] && !keep[1],
                      "completion path removes the long requirement");

  const std::vector<SyncGraphEdge> issueOnly = {
      {0, 1, SyncGraphEdgeKind::IssueOrder},
      {1, 2, SyncGraphEdgeKind::IssueOrder}};
  const std::vector<bool> keepIssueOnly =
      mlir::pto::reduceCompletionRequirements(3, issueOnly, {{0, 2}});
  passed &= check(keepIssueOnly.size() == 1 && keepIssueOnly[0],
                  "issue-only path cannot satisfy completion");

  const std::vector<SyncGraphEdge> hardwarePath = {
      {0, 1, SyncGraphEdgeKind::HardwareCompletion},
      {1, 2, SyncGraphEdgeKind::IssueOrder}};
  const std::vector<bool> keepHardware =
      mlir::pto::reduceCompletionRequirements(3, hardwarePath, {{0, 2}});
  passed &= check(keepHardware.size() == 1 && !keepHardware[0],
                  "hardware completion path satisfies requirement");
  return passed;
}

bool testColoringBoundariesAndDeterminism() {
  SyncConflictGraph empty = makeGraph(0, {});
  const auto emptyColoring = mlir::pto::colorIntervalGraph(empty);
  bool passed = check(emptyColoring.isInterval, "empty graph is interval") &&
                check(emptyColoring.coloring.colorCount == 0,
                      "empty graph needs no colors") &&
                check(mlir::pto::isValidColoring(empty, emptyColoring.coloring),
                      "empty coloring is valid");

  SyncConflictGraph clique =
      makeGraph(8, {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7},
                    {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {2, 3},
                    {2, 4}, {2, 5}, {2, 6}, {2, 7}, {3, 4}, {3, 5}, {3, 6},
                    {3, 7}, {4, 5}, {4, 6}, {4, 7}, {5, 6}, {5, 7}, {6, 7}});
  const auto cliqueColoring = mlir::pto::colorIntervalGraph(clique);
  passed &= check(cliqueColoring.isInterval, "eight-clique is interval");
  passed &= check(cliqueColoring.coloring.colorCount == 8,
                  "eight-clique reaches the hardware color bound");

  SyncConflictGraph witness = makeGraph(
      6,
      {{0, 1}, {0, 2}, {0, 4}, {1, 4}, {1, 5}, {2, 3}, {2, 5}, {3, 4}, {3, 5}});
  const auto firstDsatur = mlir::pto::colorDsatur(witness);
  const auto secondDsatur = mlir::pto::colorDsatur(witness);
  const auto firstSmallestLast = mlir::pto::colorSmallestLast(witness);
  const auto secondSmallestLast = mlir::pto::colorSmallestLast(witness);
  passed &= check(firstDsatur.colorCount == secondDsatur.colorCount &&
                      firstDsatur.colors == secondDsatur.colors,
                  "DSATUR is deterministic");
  passed &=
      check(firstSmallestLast.colorCount == secondSmallestLast.colorCount &&
                firstSmallestLast.colors == secondSmallestLast.colors,
            "smallest-last is deterministic");
  return passed;
}

bool testInvalidCompletionRequirements() {
  const std::vector<CompletionRequirement> requirements = {
      {4, 5}, {0, 3}, {2, 2}, {3, 1}};
  const std::vector<bool> keep =
      mlir::pto::reduceCompletionRequirements(4, {}, requirements);
  return check(keep.size() == 4 && !keep[0] && keep[1] && !keep[2] && !keep[3],
               "invalid completion requirements are rejected safely");
}

} // namespace

int main() {
  const bool passed = testIntervalRecognitionAndOptimalColoring() &&
                      testNonIntervalCertificates() &&
                      testHeuristicColorings() &&
                      testCompletionQualifiedReduction() &&
                      testColoringBoundariesAndDeterminism() &&
                      testInvalidCompletionRequirements();
  return passed ? 0 : 1;
}
