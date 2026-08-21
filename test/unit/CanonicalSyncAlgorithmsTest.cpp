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

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using mlir::pto::CompletionRequirement;
using mlir::pto::SyncColoring;
using mlir::pto::SyncGraphEdge;
using mlir::pto::SyncGraphEdgeKind;
using mlir::pto::SyncInterval;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncAlgorithmsTest failure: " << message << '\n';
  }
  return condition;
}

bool intervalsOverlap(const SyncInterval &first, const SyncInterval &second) {
  return !(first.end < second.begin || second.end < first.begin);
}

bool isValidColoring(const std::vector<SyncInterval> &intervals,
                     const SyncColoring &coloring) {
  if (coloring.colors.size() != intervals.size()) {
    return false;
  }
  for (std::size_t first = 0; first < intervals.size(); ++first) {
    if (coloring.colors[first] >= coloring.colorCount) {
      return false;
    }
    for (std::size_t second = first + 1; second < intervals.size(); ++second) {
      if (intervalsOverlap(intervals[first], intervals[second]) &&
          coloring.colors[first] == coloring.colors[second]) {
        return false;
      }
    }
  }
  return true;
}

bool testOptimalIntervalColoring() {
  const std::vector<SyncInterval> intervals = {
      {0, 2}, {2, 4}, {3, 5}, {6, 7}, {8, 9}};
  const SyncColoring coloring = mlir::pto::colorSyncIntervals(intervals);
  bool passed =
      check(coloring.colorCount == 2,
            "maximum overlap determines the color count") &&
      check(isValidColoring(intervals, coloring), "interval coloring is valid");
  passed &= check(coloring.colors == std::vector<unsigned>({0, 1, 0, 0, 0}),
                  "expired colors are reused deterministically");

  const std::vector<SyncInterval> inclusive = {{4, 4}, {4, 4}};
  const SyncColoring inclusiveColoring =
      mlir::pto::colorSyncIntervals(inclusive);
  passed &= check(inclusiveColoring.colorCount == 2,
                  "inclusive endpoint overlap requires distinct colors");
  return passed;
}

bool testMaximumIntervalClique() {
  const std::vector<SyncInterval> intervals = {
      {0, 3}, {1, 4}, {2, 2}, {4, 6}, {7, 8}};
  bool passed = check(mlir::pto::findMaximumIntervalClique(intervals) ==
                          std::vector<std::size_t>({0, 1, 2}),
                      "maximum interval clique is found deterministically");

  const std::vector<SyncInterval> tied = {{0, 1}, {1, 2}, {3, 4}, {4, 5}};
  passed &= check(mlir::pto::findMaximumIntervalClique(tied) ==
                      std::vector<std::size_t>({0, 1}),
                  "earliest lexicographic maximum clique breaks ties");
  passed &= check(mlir::pto::findMaximumIntervalClique({}).empty(),
                  "empty intervals have an empty maximum clique");
  return passed;
}

bool testWeightedCriticalPath() {
  const std::vector<std::uint64_t> weights = {2 + 3, 4 + 0, 1 + 6, 3 + 1};
  const std::vector<SyncGraphEdge> edges = {
      {0, 1, SyncGraphEdgeKind::IssueOrder},
      {0, 2, SyncGraphEdgeKind::HardwareCompletion},
      {1, 3, SyncGraphEdgeKind::IssueOrder},
      {2, 3, SyncGraphEdgeKind::HardwareCompletion},
      {0, 2, SyncGraphEdgeKind::IssueOrder}};
  const std::optional<std::uint64_t> criticalPath =
      mlir::pto::calculateWeightedCriticalPath(weights, edges);
  bool passed =
      check(criticalPath && *criticalPath == 16,
            "one scalar vertex weight may sum multiple cost components");

  const std::optional<std::uint64_t> disconnected =
      mlir::pto::calculateWeightedCriticalPath({2, 9, 4}, {});
  passed &= check(disconnected && *disconnected == 9,
                  "a disconnected DAG uses its heaviest path");

  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  const std::optional<std::uint64_t> saturated =
      mlir::pto::calculateWeightedCriticalPath(
          {maximum - 1, 4}, {{0, 1, SyncGraphEdgeKind::IssueOrder}});
  passed &= check(saturated && *saturated == maximum,
                  "critical-path weight addition saturates");

  const std::optional<std::uint64_t> cyclic =
      mlir::pto::calculateWeightedCriticalPath(
          {1, 1}, {{0, 1, SyncGraphEdgeKind::IssueOrder},
                   {1, 0, SyncGraphEdgeKind::IssueOrder}});
  passed &= check(!cyclic, "a cyclic graph is rejected");

  const std::optional<std::uint64_t> invalid =
      mlir::pto::calculateWeightedCriticalPath(
          {1}, {{0, 1, SyncGraphEdgeKind::IssueOrder}});
  passed &= check(!invalid, "an invalid edge endpoint is rejected");
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
  const std::vector<SyncInterval> empty;
  const SyncColoring emptyColoring = mlir::pto::colorSyncIntervals(empty);
  bool passed = check(emptyColoring.colorCount == 0,
                      "empty interval set needs no colors") &&
                check(isValidColoring(empty, emptyColoring),
                      "empty interval coloring is valid");

  const std::vector<SyncInterval> clique(8, {1, 5});
  const SyncColoring cliqueColoring = mlir::pto::colorSyncIntervals(clique);
  passed &= check(cliqueColoring.colorCount == 8,
                  "eight overlapping intervals reach the hardware bound");
  passed &= check(isValidColoring(clique, cliqueColoring),
                  "eight-interval clique coloring is valid");

  const std::vector<SyncInterval> unordered = {{5, 7}, {0, 1}, {2, 4}};
  const SyncColoring first = mlir::pto::colorSyncIntervals(unordered);
  const SyncColoring second = mlir::pto::colorSyncIntervals(unordered);
  passed &=
      check(first.colorCount == 1, "disjoint intervals reuse one color") &&
      check(first.colorCount == second.colorCount &&
                first.colors == second.colors,
            "interval coloring is deterministic");
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

bool testScopedCompletionReduction() {
  const std::vector<SyncGraphEdge> conditionalPath = {
      {0, 1, SyncGraphEdgeKind::HardwareCompletion},
      {1, 2, SyncGraphEdgeKind::IssueOrder}};
  const std::vector<CompletionRequirement> requirements = {{0, 2}};
  const auto excludesConditionalVertex = [](std::size_t, std::size_t vertex) {
    return vertex != 1;
  };
  const std::vector<bool> keep = mlir::pto::reduceCompletionRequirements(
      3, conditionalPath, requirements, excludesConditionalVertex);
  bool passed =
      check(keep.size() == 1 && keep[0],
            "a conditional completion cannot satisfy an outer requirement");

  const auto includesAllVertices = [](std::size_t, std::size_t) {
    return true;
  };
  const std::vector<bool> remove = mlir::pto::reduceCompletionRequirements(
      3, conditionalPath, requirements, includesAllVertices);
  passed &= check(remove.size() == 1 && !remove[0],
                  "an available completion path satisfies the requirement");

  const auto excludesSourceVertex = [](std::size_t, std::size_t vertex) {
    return vertex != 0;
  };
  const std::vector<bool> unavailableSource =
      mlir::pto::reduceCompletionRequirements(3, conditionalPath, requirements,
                                              excludesSourceVertex);
  passed &= check(unavailableSource.size() == 1 && unavailableSource[0],
                  "an unavailable source cannot start a coverage path");
  return passed;
}

bool testDenseCompletionReduction() {
  constexpr std::size_t vertexCount = 96;
  std::vector<SyncGraphEdge> issueEdges;
  issueEdges.reserve(vertexCount - 1);
  for (std::size_t vertex = 0; vertex + 1 < vertexCount; ++vertex) {
    issueEdges.push_back({vertex, vertex + 1, SyncGraphEdgeKind::IssueOrder});
  }

  std::vector<CompletionRequirement> requirements;
  for (std::size_t source = 0; source < vertexCount; ++source) {
    for (std::size_t target = source + 1; target < vertexCount; ++target) {
      requirements.push_back({source, target});
    }
  }

  const std::vector<bool> keep = mlir::pto::reduceCompletionRequirements(
      vertexCount, issueEdges, requirements);
  return check(std::count(keep.begin(), keep.end(), true) == vertexCount - 1,
               "dense reduction retains only adjacent completion requirements");
}

bool testCompletionCoverage() {
  const std::vector<SyncGraphEdge> edges = {
      {0, 1, SyncGraphEdgeKind::IssueOrder},
      {1, 2, SyncGraphEdgeKind::HardwareCompletion},
      {2, 3, SyncGraphEdgeKind::IssueOrder}};
  const std::vector<CompletionRequirement> requirements = {
      {0, 2}, {1, 3}, {0, 3}, {0, 1}};
  const std::vector<bool> covered =
      mlir::pto::getCompletionRequirementCoverage(4, edges, requirements);
  bool passed =
      check(covered == std::vector<bool>({true, true, true, false}),
            "coverage distinguishes completion paths from issue-only paths");

  const auto excludesMergedProducer = [](std::size_t, std::size_t vertex) {
    return vertex != 1;
  };
  const std::vector<bool> scoped = mlir::pto::getCompletionRequirementCoverage(
      4, edges, {{0, 3}}, excludesMergedProducer);
  passed &= check(scoped == std::vector<bool>({false}),
                  "coverage respects structured execution filters");
  return passed;
}

} // namespace

int main() {
  const bool passed =
      testOptimalIntervalColoring() && testMaximumIntervalClique() &&
      testWeightedCriticalPath() && testCompletionQualifiedReduction() &&
      testColoringBoundariesAndDeterminism() &&
      testInvalidCompletionRequirements() && testScopedCompletionReduction() &&
      testDenseCompletionReduction() && testCompletionCoverage();
  return passed ? 0 : 1;
}
