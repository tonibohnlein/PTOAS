// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverCoverageTest failure: " << message << '\n';
  }
  return condition;
}

bool check(const SyncCoverGraphResult &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverDemand makeDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                           SyncCoverScopeId scope = 0, unsigned distance = 0) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  return demand;
}

SyncCoverEdge makeEdge(SyncCoverNodeId source, SyncCoverNodeId target,
                       SyncCoverEdgeKind kind, SyncCoverScopeId scope = 0,
                       unsigned distance = 0) {
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = kind;
  edge.scope = scope;
  edge.distance = distance;
  return edge;
}

bool testOneSupplyCoversSeveralDemands() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add first target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add second target");
  passed &= check(graph.addEdge(makeEdge(
                      firstTarget, secondTarget,
                      SyncCoverEdgeKind::NonCompletionPreservingIssueOrder)),
                  "add target issue order");
  passed &= check(graph.addDemand(makeDemand(source, firstTarget)),
                  "add first demand");
  passed &= check(graph.addDemand(makeDemand(source, secondTarget)),
                  "add second demand");
  passed &= check(graph.freezeStructure(), "freeze multi-cover graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph, expansion,
      {{0,
        makeEdge(source, firstTarget, SyncCoverEdgeKind::CompletionSupply)}});
  passed &= check(coverage && coverage.coversAll(),
                  "one event covers both ordered targets");
  passed &=
      check(coverage.covered.count() == 2, "both demand bits are present");
  return passed;
}

bool testTwoSuppliesCompose() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add composed source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add composed middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add composed target");
  passed &= check(graph.addDemand(makeDemand(source, middle)),
                  "add first direct demand");
  passed &= check(graph.addDemand(makeDemand(middle, target)),
                  "add second direct demand");
  passed &=
      check(graph.addDemand(makeDemand(source, target)), "add composed demand");
  passed &= check(graph.freezeStructure(), "freeze composed graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCompletionSupply first{
      0, makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply)};
  const SyncCoverCompletionSupply second{
      1, makeEdge(middle, target, SyncCoverEdgeKind::CompletionSupply)};
  const SyncCoverCoverageResult firstOnly =
      computeSyncCoverCoverage(graph, expansion, {first});
  passed &= check(firstOnly && firstOnly.covered.contains(0) &&
                      !firstOnly.covered.contains(2),
                  "one leg cannot cover the composed demand");
  const SyncCoverCoverageResult both =
      computeSyncCoverCoverage(graph, expansion, {first, second});
  passed &= check(both && both.coversAll(),
                  "two selected legs activate composed coverage");
  return passed;
}

bool testIssueOrderDoesNotCreatePrefixCompletion() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add prefix source");
  const SyncCoverNodeId marker =
      takeIndex(graph.addNode(1, 1, 0, 1), passed, "add prefix marker");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add prefix target");
  passed &= check(
      graph.addEdge(makeEdge(
          source, marker, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
      "add pre-supply issue order");
  passed &=
      check(graph.addDemand(makeDemand(source, target)), "add prefix demand");
  passed &= check(graph.freezeStructure(), "freeze prefix graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph, expansion,
      {{0, makeEdge(marker, target, SyncCoverEdgeKind::CompletionSupply)}});
  passed &= check(coverage && !coverage.coversAll(),
                  "later set does not complete an earlier source");
  return passed;
}

bool testGuardedRecurrenceAndCompactCarry() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true),
                passed, "add recurrence loop");
  const SyncCoverControlId control =
      takeIndex(graph.addControl(2, loop), passed, "add loop control");
  passed &= check(graph.setResourceRecurrenceCarryKind(
                      1, SyncCoverEdgeKind::CompletionPreservingIssueOrder),
                  "set compact carry kind");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, {{{control, 0}}}), passed,
                "add guarded source");
  const SyncCoverNodeId marker =
      takeIndex(graph.addNode(1, 1, loop, 1, {{{control, 0}}}), passed,
                "add guarded marker");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 2, {{{control, 1}}}), passed,
                "add guarded target");
  passed &= check(graph.addEdge(makeEdge(source, marker,
                                         SyncCoverEdgeKind::CompletionSupply)),
                  "add fixed completion edge");
  passed &= check(graph.addDemand(makeDemand(source, target, loop, 1)),
                  "add guarded recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze recurrence graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverExpansionStatistics statistics = expansion.getStatistics();
  passed &= check(expansion && statistics.arenaCount == 2 &&
                      statistics.compactCarryEdges != 0,
                  "recurrence builds one compact d+1 arena");
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0,
       makeEdge(marker, target, SyncCoverEdgeKind::CompletionSupply, loop, 1)}};
  const SyncCoverCoverageResult coverage =
      computeSyncCoverCoverage(graph, expansion, supplies);
  const SyncCoverSingletonCoverageResult singleton =
      computeSyncCoverSingletonCoverage(graph, expansion, 1, supplies);
  passed &= check(coverage && coverage.coversAll() && singleton &&
                      singleton.mechanisms[0] == coverage.covered,
                  "guarded compact recurrence matches singleton propagation");
  return passed;
}

bool testCompactCarryKindsAndResourceIsolation() {
  bool passed = true;
  for (SyncCoverEdgeKind carryKind :
       {SyncCoverEdgeKind::CompletionPreservingIssueOrder,
        SyncCoverEdgeKind::NonCompletionPreservingIssueOrder}) {
    SyncCoverGraph graph;
    const SyncCoverScopeId loop = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true), passed,
        "add compact-carry loop");
    passed &= check(graph.setResourceRecurrenceCarryKind(1, carryKind),
                    "set compact-carry issue kind");
    const SyncCoverNodeId source = takeIndex(
        graph.addNode(2, 1, loop, 0), passed, "add compact-carry source");
    const SyncCoverNodeId marker = takeIndex(
        graph.addNode(1, 1, loop, 1), passed, "add compact-carry marker");
    const SyncCoverNodeId sameResourceTarget =
        takeIndex(graph.addNode(1, 1, loop, 2), passed,
                  "add same-resource recurrence target");
    const SyncCoverNodeId otherResourceTarget =
        takeIndex(graph.addNode(2, 1, loop, 3), passed,
                  "add other-resource recurrence target");
    passed &=
        check(graph.addDemand(makeDemand(source, sameResourceTarget, loop, 1)),
              "add compact-carry recurrence demand");
    passed &=
        check(graph.addDemand(makeDemand(source, otherResourceTarget, loop, 1)),
              "add resource-isolation recurrence demand");
    passed &= check(graph.freezeStructure(), "freeze compact-carry graph");

    SyncCoverExpandedProgram expansion(graph);
    const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
        graph, expansion,
        {{0, makeEdge(source, marker, SyncCoverEdgeKind::CompletionSupply)}});
    passed &= check(coverage && coverage.covered.contains(0),
                    "established completion crosses compact carry");
    passed &= check(!coverage.covered.contains(1),
                    "compact carry remains within one issue resource");
  }
  return passed;
}

bool testOptionalIntermediateAvailability() {
  bool passed = true;
  for (bool mustExecute : {false, true}) {
    SyncCoverGraph graph;
    const SyncCoverScopeId intermediateScope = takeIndex(
        graph.addScope(0, mustExecute, SyncCoverTimelineInterval{0, 10}),
        passed, "add intermediate scope");
    const SyncCoverNodeId source =
        takeIndex(graph.addNode(1, 1, 0, 0), passed, "add availability source");
    const SyncCoverNodeId intermediate =
        takeIndex(graph.addNode(2, 1, intermediateScope, 1), passed,
                  "add availability intermediate");
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(2, 1, 0, 2), passed, "add availability target");
    passed &= check(graph.addEdge(makeEdge(
                        intermediate, target,
                        SyncCoverEdgeKind::NonCompletionPreservingIssueOrder)),
                    "add intermediate issue order");
    passed &= check(graph.addDemand(makeDemand(source, target)),
                    "add availability demand");
    passed &= check(graph.freezeStructure(), "freeze availability graph");

    SyncCoverExpandedProgram expansion(graph);
    const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
        graph, expansion,
        {{0, makeEdge(source, intermediate,
                      SyncCoverEdgeKind::CompletionSupply)}});
    passed &= check(coverage.coversAll() == mustExecute,
                    "only a must-execute intermediate proves coverage");
  }
  return passed;
}

bool testExpansionOwnershipAndMaximumHorizon() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true),
                passed, "add maximum-horizon loop");
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, loop, 0), passed,
                                           "add maximum-horizon source");
  const SyncCoverNodeId firstTarget = takeIndex(
      graph.addNode(2, 1, loop, 1), passed, "add shorter-distance target");
  const SyncCoverNodeId secondTarget = takeIndex(
      graph.addNode(2, 1, loop, 2), passed, "add maximum-distance target");
  passed &= check(graph.addDemand(makeDemand(source, firstTarget, loop, 1)),
                  "add shorter-distance demand");
  passed &= check(graph.addDemand(makeDemand(source, secondTarget, loop, 3)),
                  "add maximum-distance demand");
  passed &= check(graph.freezeStructure(), "freeze maximum-horizon graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverExpandedArena *shorter =
      expansion.getArena(graph.getDemands()[0]);
  const SyncCoverExpandedArena *maximum =
      expansion.getArena(graph.getDemands()[1]);
  passed &= check(shorter && shorter == maximum && maximum->getHorizon() == 3,
                  "one maximum-distance arena serves shorter demands");

  SyncCoverGraph moved = std::move(graph);
  SyncCoverGraph unrelated;
  passed &=
      check(expansion.isForGraph(moved) && !expansion.isForGraph(unrelated),
            "expansion identity follows graph moves only");
  return passed;
}

bool testUnavailableArenaIsReported() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true),
                passed, "add limited loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0), passed, "add limited source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 1), passed, "add limited target");
  passed &= check(graph.addDemand(makeDemand(source, target, loop, 8)),
                  "add long recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze limited graph");

  SyncCoverExpansionLimits limits;
  limits.maximumArenaNodes = 4;
  limits.maximumArenaEdges = 16;
  limits.maximumTotalNodes = 8;
  limits.maximumTotalEdges = 32;
  SyncCoverExpandedProgram expansion(graph, limits);
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph, expansion,
      {{0, makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, loop,
                    8)}});
  passed &= check(
      coverage.error == SyncCoverCoverageError::ExpansionUnavailable &&
          coverage.unavailableDemands == std::vector<SyncCoverDemandId>{0},
      "unavailable recurrence arena fails closed per demand");
  return passed;
}

bool testBaseLimitFailsClosed() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add base-limit source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add base-limit target");
  passed &= check(graph.addDemand(makeDemand(source, target)),
                  "add base-limit demand");
  passed &= check(graph.freezeStructure(), "freeze base-limit graph");

  SyncCoverExpansionLimits limits;
  limits.maximumArenaNodes = 1;
  limits.maximumArenaEdges = 8;
  limits.maximumTotalNodes = 1;
  limits.maximumTotalEdges = 8;
  SyncCoverExpandedProgram expansion(graph, limits);
  const SyncCoverCoverageResult coverage =
      computeSyncCoverCoverage(graph, expansion, {});
  passed &= check(
      coverage.error == SyncCoverCoverageError::ExpansionUnavailable &&
          coverage.unavailableDemands == std::vector<SyncCoverDemandId>{0},
      "base-arena exhaustion is reported as unavailable");
  return passed;
}

bool testInvalidSupplyFailsClosed() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add invalid source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add invalid target");
  passed &= check(graph.addDemand(makeDemand(source, target)),
                  "add invalid-supply demand");
  passed &= check(graph.freezeStructure(), "freeze invalid-supply graph");
  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph, expansion,
      {{0, makeEdge(target, source, SyncCoverEdgeKind::CompletionSupply)}});
  passed &= check(coverage.error == SyncCoverCoverageError::InvalidSupply,
                  "backward zero-distance supply is rejected");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testOneSupplyCoversSeveralDemands();
  passed &= testTwoSuppliesCompose();
  passed &= testIssueOrderDoesNotCreatePrefixCompletion();
  passed &= testGuardedRecurrenceAndCompactCarry();
  passed &= testCompactCarryKindsAndResourceIsolation();
  passed &= testOptionalIntermediateAvailability();
  passed &= testExpansionOwnershipAndMaximumHorizon();
  passed &= testUnavailableArenaIsReported();
  passed &= testBaseLimitFailsClosed();
  passed &= testInvalidSupplyFailsClosed();
  return passed ? 0 : 1;
}
