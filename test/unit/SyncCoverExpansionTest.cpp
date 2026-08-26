// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <string_view>

using namespace mlir::pto;

namespace {

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverEdge edge(SyncCoverNodeId source, SyncCoverNodeId target,
                   SyncCoverScopeId scope = 0, unsigned distance = 0) {
  SyncCoverEdge result;
  result.source = source;
  result.target = target;
  result.kind = SyncCoverEdgeKind::CompletionPreservingIssueOrder;
  result.scope = scope;
  result.distance = distance;
  return result;
}

SyncCoverDemand demand(SyncCoverNodeId source, SyncCoverNodeId target,
                       SyncCoverScopeId scope, unsigned distance) {
  SyncCoverDemand result;
  result.source = source;
  result.target = target;
  result.scope = scope;
  result.distance = distance;
  return result;
}

bool testExactScopeArenas() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true),
      passed, "add outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{2, 18}, true),
      passed, "add inner loop");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, outer, 3), passed, "add first node");
  const SyncCoverNodeId second =
      takeIndex(graph.addNode(2, 1, inner, 4), passed, "add second node");
  passed &= check(static_cast<bool>(graph.addEdge(edge(first, second))),
                  "add fixed edge");
  passed &= check(static_cast<bool>(
                      graph.addEdge(edge(second, first, outer, 2))),
                  "add outer recurrence");
  passed &= check(static_cast<bool>(
                      graph.addEdge(edge(second, second, inner, 3))),
                  "add inner recurrence");
  passed &= check(static_cast<bool>(
                      graph.addDemand(demand(second, first, outer, 2))),
                  "add outer demand");
  passed &= check(static_cast<bool>(
                      graph.addDemand(demand(second, second, inner, 1))),
                  "add inner demand");
  passed &= check(static_cast<bool>(graph.freezeStructure()),
                  "freeze expansion structure");

  SyncCoverExpandedProgram expansion(graph);
  passed &= check(static_cast<bool>(expansion), "expansion succeeds");
  passed &= check(expansion.getArenaCount() == 3,
                  "base and exact recurrence arenas are built once");
  passed &= check(expansion.getBaseArena().getHorizon() == 0 &&
                      expansion.getBaseArena()
                              .getStructuralEdges()
                              .getEdges()
                              .size() == 1,
                  "base arena contains only distance-zero structure");

  const SyncCoverExpandedArena *outerArena =
      expansion.getRecurrenceArena(outer);
  const SyncCoverExpandedArena *innerArena =
      expansion.getRecurrenceArena(inner);
  passed &= check(outerArena && outerArena->getHorizon() == 2,
                  "outer horizon follows its maximum demand distance");
  passed &= check(innerArena && innerArena->getHorizon() == 1,
                  "irrelevant longer edges do not inflate the horizon");
  if (!outerArena || !innerArena) {
    return false;
  }

  const auto countGraphEdge = [](const SyncCoverExpandedArena &arena,
                                 std::size_t graphEdge) {
    const auto &edges = arena.getStructuralEdges().getEdges();
    return std::count_if(
        edges.begin(), edges.end(),
        [&](const SyncCoverExpandedEdge &candidate) {
          return candidate.graphEdge == graphEdge;
        });
  };
  passed &= check(countGraphEdge(*outerArena, 0) == 3 &&
                      countGraphEdge(*outerArena, 1) == 1 &&
                      countGraphEdge(*outerArena, 2) == 0,
                  "outer arena excludes inner recurrence edges");
  passed &= check(countGraphEdge(*innerArena, 0) == 2 &&
                      countGraphEdge(*innerArena, 1) == 0 &&
                      countGraphEdge(*innerArena, 2) == 0,
                  "inner arena keeps boundary edges but excludes other-scope "
                  "and out-of-window recurrences");
  passed &= check(outerArena->getNodeCount() == 2 &&
                      innerArena->getNodeCount() == 2,
                  "recurrence arenas include subtree and ancestor boundaries");

  for (std::size_t node = 0; node < outerArena->getVirtualNodeCount(); ++node) {
    for (const SyncCoverExpandedEdge &expanded :
         outerArena->getStructuralEdges().getOutgoingEdges(node)) {
      passed &= check(expanded.source == node,
                      "CSR adjacency owns exactly the source's edges");
    }
  }
  const std::size_t innerTail = innerArena->getVirtualNodeCount() - 1;
  passed &= check(
      innerArena->getStructuralEdges().getOutgoingEdges(innerTail).begin() ==
          innerArena->getStructuralEdges().getOutgoingEdges(innerTail).end(),
      "empty CSR ranges remain valid");
  const auto invalidRange = outerArena->getStructuralEdges().getOutgoingEdges(
      std::numeric_limits<std::size_t>::max());
  passed &= check(invalidRange.begin() == invalidRange.end(),
                  "maximum invalid CSR index cannot wrap into the arena");

  const SyncCoverExpandedProgram edgeLimited(
      graph, SyncCoverExpansionLimits{100, 3});
  passed &= check(!edgeLimited &&
                      edgeLimited.getError() ==
                          SyncCoverExpansionError::ExpansionLimitExceeded,
                  "aggregate edge limit fails before edge allocation");

  SyncCoverGraph other = graph;
  passed &= check(other.getGeneration() == graph.getGeneration(),
                  "copied graph preserves the generation counter");
  passed &= check(!expansion.isStructuralCurrent(other) &&
                      expansion.refreshMechanismOverlay(other) ==
                          SyncCoverExpansionError::InvalidGraph,
                  "same-generation graph cannot replace the expansion owner");
  return passed;
}

bool testExpansionLimit() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 4}, true),
      passed, "add bounded loop");
  const SyncCoverNodeId node =
      takeIndex(graph.addNode(1, 1, loop, 1), passed, "add bounded node");
  passed &= check(static_cast<bool>(graph.addDemand(demand(
                      node, node, loop,
                      std::numeric_limits<unsigned>::max()))),
                  "add excessive-distance demand");
  passed &= check(static_cast<bool>(graph.freezeStructure()),
                  "freeze excessive expansion structure");
  const SyncCoverExpandedProgram expansion(graph);
  passed &= check(!expansion &&
                      expansion.getError() ==
                          SyncCoverExpansionError::ExpansionLimitExceeded,
                  "excessive expansion fails before allocation");
  return passed;
}

bool testRequiresFrozenStructure() {
  SyncCoverGraph graph;
  const SyncCoverExpandedProgram expansion(graph);
  return check(!expansion &&
                   expansion.getError() ==
                       SyncCoverExpansionError::InvalidGraph,
               "expansion rejects a mutable structural graph");
}

} // namespace

int main() {
  const bool passed = testExactScopeArenas() && testExpansionLimit() &&
                      testRequiresFrozenStructure();
  std::cout << (passed ? "SyncCoverExpansionTest PASS"
                       : "SyncCoverExpansionTest FAIL")
            << std::endl;
  return passed ? 0 : 1;
}
