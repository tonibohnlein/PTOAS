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

#include <algorithm>
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

bool testDemandQualifiedSupplyDoesNotEscape() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add qualified source");
  const SyncCoverNodeId firstTarget = takeIndex(
      graph.addNode(2, 1, 0, 1), passed, "add qualified first target");
  const SyncCoverNodeId secondTarget = takeIndex(
      graph.addNode(2, 1, 0, 2), passed, "add qualified second target");
  passed &= check(graph.addEdge(makeEdge(
                      firstTarget, secondTarget,
                      SyncCoverEdgeKind::NonCompletionPreservingIssueOrder)),
                  "add qualified target issue order");
  passed &= check(graph.addDemand(makeDemand(source, firstTarget)),
                  "add qualified demand");
  passed &= check(graph.addDemand(makeDemand(source, secondTarget)),
                  "add unrelated demand");
  passed &= check(graph.freezeStructure(), "freeze qualified-supply graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverEdge edge =
      makeEdge(source, firstTarget, SyncCoverEdgeKind::CompletionSupply);
  const SyncCoverCoverageResult restricted =
      computeSyncCoverCoverage(graph, expansion, {{0, edge, {0}}});
  passed &= check(restricted && restricted.covered.contains(0),
                  "qualified supply covers its attested demand");
  passed &= check(!restricted.covered.contains(1),
                  "qualified supply cannot escape to another demand");

  const SyncCoverCoverageResult unrestricted =
      computeSyncCoverCoverage(graph, expansion, {{0, edge}});
  passed &= check(unrestricted && unrestricted.coversAll(),
                  "ordinary completion supply remains reusable");
  return passed;
}

bool testDistanceZeroSupplyDoesNotCoverRecurrence() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 20}, true),
                passed, "add distance-qualified loop");
  passed &= check(graph.setResourceRecurrenceCarryKind(
                      2, SyncCoverEdgeKind::CompletionPreservingIssueOrder),
                  "set distance-qualified target carry");
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, loop, 0), passed,
                                           "add distance-qualified source");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, loop, 1), passed,
                                           "add distance-qualified target");
  passed &= check(graph.addDemand(makeDemand(source, target, loop, 0)),
                  "add distance-zero demand");
  passed &= check(graph.addDemand(makeDemand(source, target, loop, 1)),
                  "add recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze distance-qualified graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverEdge edge =
      makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, loop);
  const SyncCoverCompletionSupply distanceZeroOnly{
      0, edge, {}, false, SyncCoverSupplyApplicability::DistanceZeroOnly};
  const SyncCoverCoverageResult qualified =
      computeSyncCoverCoverage(graph, expansion, {distanceZeroOnly});
  passed &= check(qualified && qualified.covered.contains(0),
                  "distance-zero supply covers a distance-zero row");
  passed &= check(!qualified.covered.contains(1),
                  "distance-zero supply cannot cross a recurrence carry");
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

bool testCertifiedFrontierCreatesPrefixCompletion() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {}, std::nullopt, true), passed,
                "add certified prefix source");
  const SyncCoverNodeId frontier =
      takeIndex(graph.addNode(1, 1, 0, 1, {}, {}, std::nullopt, true), passed,
                "add certified prefix frontier");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, 0, 2), passed,
                                           "add certified prefix target");
  passed &= check(
      graph.addEdge(makeEdge(source, frontier,
                             SyncCoverEdgeKind::CertifiedCompletionFrontier)),
      "add certified prefix edge");
  passed &= check(graph.addCompletionDominance(source, frontier),
                  "register certified prefix dominance");
  passed &= check(graph.addDemand(makeDemand(source, target)),
                  "add certified prefix demand");
  passed &= check(graph.freezeStructure(), "freeze certified prefix graph");

  SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
      graph, expansion,
      {{0, makeEdge(frontier, target, SyncCoverEdgeKind::CompletionSupply)}});
  passed &= check(coverage && coverage.coversAll(),
                  "later certified set completes the earlier source");
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

struct CoverageSnapshot {
  SyncCoverCoverageResult full;
  SyncCoverSingletonCoverageResult singleton;
  SyncCoverPairCoverageResult pair;
};

CoverageSnapshot getTwoLegCoverage(const SyncCoverGraph &graph,
                                   const SyncCoverExpandedProgram &expansion,
                                   SyncCoverNodeId source,
                                   SyncCoverNodeId middle,
                                   SyncCoverNodeId completionTarget,
                                   SyncCoverScopeId scope) {
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0, makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply, scope)},
      {1, makeEdge(middle, completionTarget,
                   SyncCoverEdgeKind::CompletionSupply, scope)}};
  return {computeSyncCoverCoverage(graph, expansion, supplies),
          computeSyncCoverSingletonCoverage(graph, expansion, 2, supplies),
          computeSyncCoverPairCoverage(graph, expansion, 2, supplies, {{0, 1}},
                                       {0})};
}

bool snapshotsMatch(const CoverageSnapshot &left,
                    const CoverageSnapshot &right) {
  return left.full && right.full && left.singleton && right.singleton &&
         left.pair && right.pair && left.full.covered == right.full.covered &&
         left.singleton.baseline == right.singleton.baseline &&
         left.singleton.mechanisms == right.singleton.mechanisms &&
         left.pair.pairs == right.pair.pairs;
}

bool testLoopSummaryMatchesExplicitUnrollings() {
  bool passed = true;
  SyncCoverGraph summarized;
  const SyncCoverScopeId outer = takeIndex(
      summarized.addScope(0, true, SyncCoverTimelineInterval{0, 40}, true),
      passed, "add summarized outer loop");
  const SyncCoverScopeId inner = takeIndex(
      summarized.addScope(outer, false, SyncCoverTimelineInterval{4, 24}, true),
      passed, "add optional summarized inner loop");
  const SyncCoverControlId control =
      takeIndex(summarized.addControl(2, inner), passed,
                "add summarized periodic control");
  SyncCoverControlPhaseRelation relation;
  relation.loopScope = inner;
  relation.initialPhase = 0;
  relation.nextPhase = {2, 3, 0, 1};
  relation.activeAlternative = {0, 1, 0, 1};
  passed &= check(summarized.setControlPhaseRelation(control, relation),
                  "add summarized periodic phase relation");
  const SyncCoverNodeId source = takeIndex(summarized.addNode(0, 1, outer, 0),
                                           passed, "add summarized source");
  const SyncCoverNodeId middle = takeIndex(summarized.addNode(2, 1, outer, 1),
                                           passed, "add summarized middle");
  const SyncCoverNodeId beforeLoop =
      takeIndex(summarized.addNode(1, 1, outer, 2), passed,
                "add summarized pre-loop completion point");
  const SyncCoverNodeId firstInner =
      takeIndex(summarized.addNode(1, 1, inner, 4, {{{control, 0}}}), passed,
                "add first reachable guarded inner operation");
  const SyncCoverNodeId secondInner =
      takeIndex(summarized.addNode(1, 1, inner, 5, {{{control, 0}}}), passed,
                "add second reachable guarded inner operation");
  const SyncCoverNodeId target =
      takeIndex(summarized.addNode(1, 1, outer, 15), passed,
                "add summarized post-loop target");
  passed &= check(summarized.addEdge(makeEdge(
                      beforeLoop, firstInner,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "enter summarized inner loop");
  passed &= check(summarized.addEdge(makeEdge(
                      firstInner, secondInner,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "order summarized inner operations");
  passed &= check(summarized.addEdge(makeEdge(
                      secondInner, target,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "exit summarized inner loop");
  passed &= check(summarized.addEdge(makeEdge(
                      beforeLoop, target,
                      SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                  "preserve explicit zero-trip path");
  passed &= check(summarized.addDemand(makeDemand(source, target, outer)),
                  "add summarized demand");
  passed &=
      check(summarized.freezeStructure(), "freeze summarized nested graph");

  const SyncCoverExpandedProgram summarizedExpansion(summarized);
  const SyncCoverLoopSummary *summary =
      summarizedExpansion.getLoopSummary(inner);
  const SyncCoverExpandedArena *outerArena =
      summarizedExpansion.getRecurrenceArena(outer);
  const bool summaryContract =
      summary && summary->parentLoop == outer && summary->zeroTripPossible &&
      summary->entry.kind == SyncCoverAnchorKind::ScopeEntry &&
      summary->entry.position == 4 &&
      summary->exit.kind == SyncCoverAnchorKind::ScopeExit &&
      summary->exit.position == 24 &&
      summary->resources == std::vector<std::uint32_t>({1}) &&
      summary->carryResources.empty() &&
      summary->periodicControls.size() == 1 &&
      summary->periodicControls[0].control == control &&
      summary->periodicControls[0].initialPhase == 0 &&
      summary->periodicControls[0].nextPhase ==
          std::vector<std::size_t>({2, 3, 0, 1}) &&
      summary->periodicControls[0].activeAlternative ==
          std::vector<unsigned>({0, 1, 0, 1}) &&
      summary->periodicControls[0].reachablePhases ==
          std::vector<std::size_t>({0, 2}) &&
      summary->completionTransfers.size() == 1 &&
      summary->completionTransfers[0].resource == 1 &&
      summary->completionTransfers[0].hasNonZeroPath &&
      summary->completionTransfers[0].hasZeroTripPath;
  passed &= check(summarizedExpansion && summaryContract,
                  "build bottom-up loop summary contract");
  const bool excludesChildBody =
      outerArena &&
      std::find(outerArena->getOperationNodes().begin(),
                outerArena->getOperationNodes().end(),
                firstInner) == outerArena->getOperationNodes().end() &&
      std::find(outerArena->getOperationNodes().begin(),
                outerArena->getOperationNodes().end(),
                secondInner) == outerArena->getOperationNodes().end();
  passed &= check(excludesChildBody &&
                      outerArena->getLoopBoundary(
                          inner, 1, SyncCoverLoopBoundaryKind::Entry, 0) &&
                      outerArena->getLoopBoundary(
                          inner, 1, SyncCoverLoopBoundaryKind::Exit, 0),
                  "replace the child body with its transfer interface");
  const SyncCoverExpansionStatistics statistics =
      summarizedExpansion.getStatistics();
  passed &=
      check(statistics.loopSummaryNodes != 0 && statistics.zeroTripEdges != 0,
            "materialize explicit zero-trip summary transfers");

  const CoverageSnapshot summarizedCoverage = getTwoLegCoverage(
      summarized, summarizedExpansion, source, middle, beforeLoop, outer);
  passed &= check(summarizedCoverage.full &&
                      summarizedCoverage.full.covered.contains(0) &&
                      summarizedCoverage.singleton &&
                      summarizedCoverage.singleton.mechanisms[0].empty() &&
                      summarizedCoverage.singleton.mechanisms[1].empty() &&
                      summarizedCoverage.pair &&
                      summarizedCoverage.pair.pairs[0].contains(0),
                  "two-leg pair completion crosses the child summary");

  for (unsigned iterations = 0; iterations <= 2; ++iterations) {
    SyncCoverGraph unrolled;
    const SyncCoverScopeId unrolledOuter = takeIndex(
        unrolled.addScope(0, true, SyncCoverTimelineInterval{0, 40}, true),
        passed, "add explicitly unrolled outer loop");
    const SyncCoverNodeId unrolledSource =
        takeIndex(unrolled.addNode(0, 1, unrolledOuter, 0), passed,
                  "add explicitly unrolled source");
    const SyncCoverNodeId unrolledMiddle =
        takeIndex(unrolled.addNode(2, 1, unrolledOuter, 1), passed,
                  "add explicitly unrolled middle");
    const SyncCoverNodeId unrolledBefore =
        takeIndex(unrolled.addNode(1, 1, unrolledOuter, 2), passed,
                  "add explicitly unrolled pre-loop point");
    SyncCoverNodeId previous = unrolledBefore;
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
      const SyncCoverNodeId body =
          takeIndex(unrolled.addNode(1, 1, unrolledOuter, 3 + iteration),
                    passed, "add explicit loop iteration");
      passed &= check(unrolled.addEdge(makeEdge(
                          previous, body,
                          SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                      "order explicit loop iteration");
      previous = body;
    }
    const SyncCoverNodeId unrolledTarget =
        takeIndex(unrolled.addNode(1, 1, unrolledOuter, 10), passed,
                  "add explicitly unrolled target");
    passed &= check(unrolled.addEdge(makeEdge(
                        previous, unrolledTarget,
                        SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                    "exit explicit loop iterations");
    passed &= check(unrolled.addDemand(makeDemand(
                        unrolledSource, unrolledTarget, unrolledOuter)),
                    "add explicitly unrolled demand");
    passed &= check(unrolled.freezeStructure(), "freeze explicit unrolling");
    const SyncCoverExpandedProgram unrolledExpansion(unrolled);
    const CoverageSnapshot unrolledCoverage =
        getTwoLegCoverage(unrolled, unrolledExpansion, unrolledSource,
                          unrolledMiddle, unrolledBefore, unrolledOuter);
    passed &= check(snapshotsMatch(summarizedCoverage, unrolledCoverage),
                    "summary exactly matches zero/one/two iteration coverage");
  }
  return passed;
}

bool testChildPortsPreserveEndpointIdentity() {
  bool passed = true;
  const auto build = [&](bool summarized) {
    SyncCoverGraph graph;
    const SyncCoverScopeId outer = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 50}, true), passed,
        "add endpoint-order outer loop");
    SyncCoverScopeId child = outer;
    if (summarized) {
      child = takeIndex(
          graph.addScope(outer, true, SyncCoverTimelineInterval{2, 16}, true),
          passed, "add endpoint-order child loop");
    }
    const SyncCoverNodeId source = takeIndex(
        graph.addNode(0, 1, outer, 0), passed, "add endpoint-order source");
    const SyncCoverNodeId earlySource = takeIndex(
        graph.addNode(1, 1, child, 2), passed, "add early child source");
    const SyncCoverNodeId lateSource = takeIndex(
        graph.addNode(1, 1, child, 3), passed, "add late child source");
    const SyncCoverNodeId earlyTarget = takeIndex(
        graph.addNode(2, 1, child, 4), passed, "add early child target");
    const SyncCoverNodeId lateTarget = takeIndex(
        graph.addNode(2, 1, child, 6), passed, "add late child target");
    const SyncCoverNodeId outerTarget =
        takeIndex(graph.addNode(2, 1, outer, 10), passed,
                  "add endpoint-order outer target");
    passed &= check(
        graph.addEdge(makeEdge(source, lateTarget,
                               SyncCoverEdgeKind::CompletionSupply, outer)),
        "add fixed late-target completion");
    passed &= check(
        graph.addEdge(makeEdge(earlySource, earlyTarget,
                               SyncCoverEdgeKind::CompletionSupply, child)),
        "add fixed early-source completion");
    passed &= check(graph.addEdge(makeEdge(
                        earlyTarget, outerTarget,
                        SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                    "route early child target outward");
    passed &= check(graph.addDemand(makeDemand(source, earlyTarget, outer)),
                    "add earlier-target parent demand");
    passed &= check(graph.addDemand(makeDemand(lateSource, outerTarget, outer)),
                    "add later-source parent demand");
    passed &= check(graph.addDemand(makeDemand(source, lateTarget, outer)),
                    "expose late target port");
    passed &=
        check(graph.addDemand(makeDemand(earlySource, earlyTarget, child)),
              "expose early source port");
    passed &= check(graph.freezeStructure(), "freeze endpoint-order graph");

    const SyncCoverExpandedProgram expansion(graph);
    if (summarized) {
      const SyncCoverExpandedArena *arena = expansion.getRecurrenceArena(outer);
      const std::optional<std::size_t> earlySourcePort =
          arena ? arena->getLoopPort(child, earlySource, 0) : std::nullopt;
      const std::optional<std::size_t> lateSourcePort =
          arena ? arena->getLoopPort(child, lateSource, 0) : std::nullopt;
      const std::optional<std::size_t> earlyTargetPort =
          arena ? arena->getLoopPort(child, earlyTarget, 0) : std::nullopt;
      const std::optional<std::size_t> lateTargetPort =
          arena ? arena->getLoopPort(child, lateTarget, 0) : std::nullopt;
      passed &=
          check(earlySourcePort && lateSourcePort && earlyTargetPort &&
                    lateTargetPort && *earlySourcePort != *lateSourcePort &&
                    *earlyTargetPort != *lateTargetPort,
                "preserve ordered same-resource endpoint ports");
    }
    const std::vector<SyncCoverCompletionSupply> supplies = {
        {0, makeEdge(source, lateTarget, SyncCoverEdgeKind::CompletionSupply,
                     outer)},
        {1, makeEdge(earlySource, earlyTarget,
                     SyncCoverEdgeKind::CompletionSupply, child)}};
    const std::vector<SyncCoverDemandId> active{0, 1};
    return CoverageSnapshot{
        computeSyncCoverCoverage(graph, expansion, supplies, active),
        computeSyncCoverSingletonCoverage(graph, expansion, 2, supplies,
                                          active),
        computeSyncCoverPairCoverage(graph, expansion, 2, supplies, {{0, 1}},
                                     active)};
  };

  const CoverageSnapshot summarized = build(true);
  const CoverageSnapshot unrolled = build(false);
  passed &= check(snapshotsMatch(summarized, unrolled),
                  "endpoint-sensitive summary matches explicit child body");
  passed &=
      check(summarized.full && summarized.full.covered.empty() &&
                summarized.singleton && summarized.singleton.baseline.empty() &&
                summarized.singleton.mechanisms[0].empty() &&
                summarized.singleton.mechanisms[1].empty() && summarized.pair &&
                summarized.pair.pairs[0].empty(),
            "late waits and early sets cannot cover reversed endpoints");
  return passed;
}

bool testChildLocalPairExportsThroughPorts() {
  bool passed = true;
  const auto build = [&](bool summarized) {
    SyncCoverGraph graph;
    const SyncCoverScopeId outer = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 40}, true), passed,
        "add child-pair outer loop");
    SyncCoverScopeId child = outer;
    if (summarized) {
      child = takeIndex(
          graph.addScope(outer, true, SyncCoverTimelineInterval{2, 20}, true),
          passed, "add child-pair summarized loop");
    }
    const SyncCoverNodeId source =
        takeIndex(graph.addNode(1, 1, child, 2, {}, {2}), passed,
                  "add child-pair source");
    const SyncCoverNodeId middle =
        takeIndex(graph.addNode(2, 1, child, 4, {}, {3}), passed,
                  "add child-pair middle");
    const SyncCoverNodeId childTarget = takeIndex(
        graph.addNode(3, 1, child, 6), passed, "add child-pair target");
    const SyncCoverNodeId outerTarget = takeIndex(
        graph.addNode(3, 1, outer, 12), passed, "add child-pair outer target");
    passed &= check(graph.addEdge(makeEdge(
                        childTarget, outerTarget,
                        SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
                    "export child-pair completion");
    passed &= check(graph.addDemand(makeDemand(source, outerTarget, outer)),
                    "add child-pair parent demand");
    passed &= check(graph.addDemand(makeDemand(source, middle, child)),
                    "add child-pair first local demand");
    passed &= check(graph.addDemand(makeDemand(middle, childTarget, child)),
                    "add child-pair second local demand");
    passed &= check(graph.freezeStructure(), "freeze child-pair graph");
    const SyncCoverExpandedProgram expansion(graph);
    return getTwoLegCoverage(graph, expansion, source, middle, childTarget,
                             child);
  };

  const CoverageSnapshot summarized = build(true);
  const CoverageSnapshot unrolled = build(false);
  passed &= check(snapshotsMatch(summarized, unrolled),
                  "child-local pair summary matches explicit body");
  passed &= check(summarized.full && summarized.full.covered.contains(0) &&
                      summarized.singleton &&
                      !summarized.singleton.mechanisms[0].contains(0) &&
                      !summarized.singleton.mechanisms[1].contains(0) &&
                      summarized.pair && summarized.pair.pairs[0].contains(0),
                  "child-local pair completion exports through exact ports");
  return passed;
}

bool testChildFixedPathClosureMatchesExplicitBody() {
  bool passed = true;
  const auto build = [&](bool summarized) {
    SyncCoverGraph graph;
    const SyncCoverScopeId outer = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 50}, true), passed,
        "add fixed-path outer loop");
    SyncCoverScopeId child = outer;
    if (summarized) {
      child = takeIndex(
          graph.addScope(outer, true, SyncCoverTimelineInterval{2, 20}, true),
          passed, "add fixed-path summarized loop");
    }
    const SyncCoverNodeId source = takeIndex(graph.addNode(0, 1, outer, 0),
                                             passed, "add fixed-path source");
    const SyncCoverNodeId first = takeIndex(
        graph.addNode(1, 1, child, 2), passed, "add fixed-path first port");
    const SyncCoverNodeId bridge = takeIndex(graph.addNode(1, 1, child, 4),
                                             passed, "add fixed-path bridge");
    const SyncCoverNodeId last =
        takeIndex(graph.addNode(1, 1, child, 6, {}, {2}), passed,
                  "add fixed-path last port");
    const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, outer, 12),
                                             passed, "add fixed-path target");
    passed &= check(
        graph.addEdge(makeEdge(
            first, bridge, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
        "enter unexposed fixed-path bridge");
    passed &= check(
        graph.addEdge(makeEdge(
            bridge, last, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
        "exit unexposed fixed-path bridge");
    passed &= check(graph.addDemand(makeDemand(source, target, outer)),
                    "add fixed-path parent demand");
    passed &= check(graph.addDemand(makeDemand(source, first, outer)),
                    "expose fixed-path entry port");
    passed &= check(graph.freezeStructure(), "freeze fixed-path graph");

    const SyncCoverExpandedProgram expansion(graph);
    if (summarized) {
      const SyncCoverExpandedArena *arena = expansion.getRecurrenceArena(outer);
      passed &= check(arena && arena->getLoopPort(child, bridge, 0),
                      "retain the internal node needed by a fixed path");
    }
    const std::vector<SyncCoverCompletionSupply> supplies = {
        {0,
         makeEdge(source, first, SyncCoverEdgeKind::CompletionSupply, outer)},
        {1,
         makeEdge(last, target, SyncCoverEdgeKind::CompletionSupply, outer)}};
    const std::vector<SyncCoverDemandId> active{0};
    return CoverageSnapshot{
        computeSyncCoverCoverage(graph, expansion, supplies, active),
        computeSyncCoverSingletonCoverage(graph, expansion, 2, supplies,
                                          active),
        computeSyncCoverPairCoverage(graph, expansion, 2, supplies, {{0, 1}},
                                     active)};
  };

  const CoverageSnapshot summarized = build(true);
  const CoverageSnapshot unrolled = build(false);
  passed &= check(snapshotsMatch(summarized, unrolled),
                  "fixed-path summary matches explicit child body");
  passed &= check(summarized.full && summarized.full.covered.contains(0) &&
                      summarized.singleton &&
                      summarized.singleton.mechanisms[0].empty() &&
                      summarized.singleton.mechanisms[1].empty() &&
                      summarized.pair && summarized.pair.pairs[0].contains(0),
                  "two supplies compose through an internal fixed-path node");
  return passed;
}

bool testNestedExportCrossesInactiveMiddleSummary() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 80}, true),
                passed, "add export outer loop");
  const SyncCoverScopeId middle = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{5, 60}, true),
      passed, "add inactive must-execute middle loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(middle, true, SyncCoverTimelineInterval{10, 50}, true),
      passed, "add export inner loop");
  const SyncCoverNodeId wait = takeIndex(graph.addNode(2, 1, inner, 12), passed,
                                         "add export inner wait");
  const SyncCoverNodeId set =
      takeIndex(graph.addNode(1, 1, inner, 20), passed, "add export inner set");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, outer, 35),
                                           passed, "add export outer target");
  passed &= check(
      graph.addEdge(makeEdge(
          wait, target, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
      "route inactive-middle exit to target");
  passed &= check(graph.addDemand(makeDemand(set, target, outer)),
                  "add parent export demand");
  passed &= check(graph.addDemand(makeDemand(set, wait, inner, 2)),
                  "add distance-two inner recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze inactive-middle graph");

  const std::vector<SyncCoverCompletionSupply> protocol = {
      {0,
       makeEdge(set, wait, SyncCoverEdgeKind::CompletionSupply, inner, 2),
       {},
       true}};
  const SyncCoverExpandedProgram parentOnlyExpansion(
      graph, std::vector<std::size_t>{0});
  const SyncCoverCoverageResult parentOnly =
      computeSyncCoverCoverage(graph, parentOnlyExpansion, protocol, {0});
  const SyncCoverExpandedProgram completeExpansion(graph);
  const SyncCoverCoverageResult complete =
      computeSyncCoverCoverage(graph, completeExpansion, protocol, {0});
  const SyncCoverExpandedArena *outerArena =
      parentOnlyExpansion.getRecurrenceArena(outer);
  const bool immediateInterfaceOnly =
      outerArena &&
      outerArena->getLoopBoundary(middle, 2, SyncCoverLoopBoundaryKind::Exit,
                                  0) &&
      !outerArena->getLoopBoundary(inner, 2, SyncCoverLoopBoundaryKind::Exit,
                                   0) &&
      outerArena->getOperationNodes() == std::vector<SyncCoverNodeId>({target});
  passed &= check(parentOnlyExpansion && completeExpansion && parentOnly &&
                      complete && parentOnly.covered == complete.covered &&
                      parentOnly.covered.contains(0),
                  "distance-two export is independent of active child rows");
  passed &= check(immediateInterfaceOnly,
                  "parent instantiates only the inactive middle interface");
  return passed;
}

bool testGuardedProtocolRemainsLocal() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 40}, true),
                passed, "add guarded-export outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{4, 24}, true),
      passed, "add guarded-export inner loop");
  const SyncCoverControlId control = takeIndex(
      graph.addControl(2, inner), passed, "add guarded-export control");
  passed &=
      check(graph.setControlPhaseRelation(control, {inner, 0, {1, 0}, {0, 1}}),
            "add guarded-export phase relation");
  const SyncCoverGuard guard{{{control, 0}}};
  const SyncCoverNodeId wait = takeIndex(graph.addNode(2, 1, inner, 6, guard),
                                         passed, "add guarded-export wait");
  const SyncCoverNodeId set = takeIndex(graph.addNode(1, 1, inner, 10, guard),
                                        passed, "add guarded-export set");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, outer, 15), passed,
                "add guarded-export parent target");
  passed &= check(
      graph.addEdge(makeEdge(
          wait, target, SyncCoverEdgeKind::CompletionPreservingIssueOrder)),
      "route guarded child to parent target");
  passed &= check(graph.addDemand(makeDemand(set, target, outer)),
                  "add guarded parent demand");
  SyncCoverDemand localDemand = makeDemand(set, wait, inner, 1);
  localDemand.sourceGuard = guard;
  localDemand.targetGuard = guard;
  passed &= check(graph.addDemand(localDemand), "add guarded local demand");
  passed &= check(graph.freezeStructure(), "freeze guarded-export graph");

  SyncCoverEdge supply =
      makeEdge(set, wait, SyncCoverEdgeKind::CompletionSupply, inner, 1);
  supply.sourceGuard = guard;
  supply.targetGuard = guard;
  const SyncCoverExpandedProgram expansion(graph);
  const SyncCoverCoverageResult coverage =
      computeSyncCoverCoverage(graph, expansion, {{0, supply, {}, true}});
  passed &= check(coverage && !coverage.covered.contains(0) &&
                      coverage.covered.contains(1),
                  "guarded export is ignored by parents but remains local");
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

bool testPairWorkspaceLimitReturnsNoPartialCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add pair-limit source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add pair-limit middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add pair-limit target");
  passed &= check(graph.addDemand(makeDemand(source, target)),
                  "add pair-limit demand");
  passed &= check(graph.freezeStructure(), "freeze pair-limit graph");

  const SyncCoverExpandedProgram expansion(graph);
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0, makeEdge(source, middle, SyncCoverEdgeKind::CompletionSupply)},
      {1, makeEdge(middle, target, SyncCoverEdgeKind::CompletionSupply)}};
  SyncCoverCoverageLimits limits;
  limits.maximumWorkspaceWords = 0;
  const SyncCoverPairCoverageResult result = computeSyncCoverPairCoverage(
      graph, expansion, 2, supplies, {{0, 1}}, {0}, limits);
  passed &= check(result.error == SyncCoverCoverageError::LimitExceeded &&
                      result.pairs.empty() && result.unavailableDemands.empty(),
                  "pair workspace exhaustion discards the entire batch");
  return passed;
}

bool testPairMechanismRowsAreBoundedWithoutDemands() {
  bool passed = true;
  SyncCoverGraph graph;
  passed &= check(graph.freezeStructure(), "freeze zero-demand pair graph");
  const SyncCoverExpandedProgram expansion(graph);
  SyncCoverCoverageLimits limits;
  limits.maximumResultRows = 0;
  limits.maximumResultWords = 0;
  limits.maximumMechanismRows = 0;
  const SyncCoverPairCoverageResult rejected =
      computeSyncCoverPairCoverage(graph, expansion, 1, {}, {}, {}, limits);
  const SyncCoverPairCoverageResult empty =
      computeSyncCoverPairCoverage(graph, expansion, 0, {}, {}, {}, limits);
  passed &= check(rejected.error == SyncCoverCoverageError::LimitExceeded &&
                      rejected.pairs.empty(),
                  "bound pair mechanism rows before zero-demand allocation");
  passed &= check(empty && empty.pairs.empty(),
                  "accept an actually empty zero-row pair query");
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

bool testHierarchicalMetadataHonorsBaseNodeLimit() {
  bool passed = true;
  SyncCoverGraph graph;
  passed &= check(graph.setResourceRecurrenceCarryKind(
                      1, SyncCoverEdgeKind::CompletionPreservingIssueOrder),
                  "set deeply nested carry resource");
  SyncCoverScopeId deepest = 0;
  constexpr unsigned nestingDepth = 32;
  for (unsigned depth = 0; depth < nestingDepth; ++depth) {
    deepest = takeIndex(
        graph.addScope(deepest, true, SyncCoverTimelineInterval{0, 256}, true),
        passed, "add deeply nested loop");
  }

  std::vector<SyncCoverNodeId> ports;
  constexpr unsigned endpointCount = 64;
  for (unsigned order = 0; order < endpointCount; ++order) {
    ports.push_back(takeIndex(graph.addNode(1, 1, deepest, order), passed,
                              "add deeply nested endpoint"));
  }
  for (unsigned endpoint = 0; endpoint < endpointCount; endpoint += 2) {
    passed &= check(graph.addDemand(makeDemand(ports[endpoint],
                                               ports[endpoint + 1], deepest)),
                    "add deeply nested demand");
  }
  passed &= check(graph.freezeStructure(), "freeze deeply nested graph");

  SyncCoverExpansionLimits limits;
  limits.maximumArenaNodes = 3;
  limits.maximumArenaEdges = 32;
  limits.maximumTotalNodes = 3;
  limits.maximumTotalEdges = 32;
  const SyncCoverExpandedProgram first(graph, limits);
  const SyncCoverExpandedProgram second(graph, limits);
  std::size_t storedPorts = 0;
  std::size_t storedResources = 0;
  std::size_t storedCarryResources = 0;
  std::size_t storedTransfers = 0;
  for (const SyncCoverLoopSummary &summary : first.getLoopSummaries()) {
    storedPorts += summary.completionPorts.size();
    storedResources += summary.resources.size();
    storedCarryResources += summary.carryResources.size();
    storedTransfers += summary.completionTransfers.size();
  }
  passed &= check(storedPorts == ports.size(),
                  "store each hierarchical port in only its owning summary");
  passed &= check(storedResources == 1 && storedCarryResources == 1 &&
                      storedTransfers == 1,
                  "store same-resource metadata only in its owning summary");
  passed &= check(
      first.getError() == SyncCoverExpansionError::BaseLimitExceeded &&
          second.getError() == first.getError(),
      "deep hierarchical ports fail deterministically at the base node limit");
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

bool testBoundedCoverageMetersWholeQuery() {
  bool passed = true;
  SyncCoverGraph graph;
  SyncCoverGuard targetGuard;
  SyncCoverGuard sourceGuard;
  constexpr std::size_t guardCount = 8;
  for (std::size_t index = 0; index < guardCount; ++index) {
    const SyncCoverControlId control = takeIndex(
        graph.addControl(2), passed, "add bounded-query guard control");
    targetGuard.literals.push_back({control, 0});
  }
  for (std::size_t index = 0; index < guardCount; ++index) {
    const SyncCoverControlId control = takeIndex(
        graph.addControl(2), passed, "add bounded-query guard control");
    sourceGuard.literals.push_back({control, 0});
  }

  SyncCoverScopeId outer = 0;
  SyncCoverScopeId deepest = 0;
  constexpr std::size_t loopDepth = 8;
  for (std::size_t depth = 0; depth < loopDepth; ++depth) {
    deepest = takeIndex(
        graph.addScope(deepest, true, SyncCoverTimelineInterval{0, 200}, true),
        passed, "add bounded-query nested loop");
    if (depth == 0) {
      outer = deepest;
    }
  }

  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, deepest, 1, sourceGuard), passed,
                "add bounded-query source");
  constexpr std::size_t targetCount = 16;
  std::vector<SyncCoverNodeId> targets;
  std::vector<SyncCoverDemandId> activeDemands;
  std::vector<SyncCoverCompletionSupply> supplies;
  targets.reserve(targetCount);
  activeDemands.reserve(targetCount);
  for (std::size_t index = 0; index < targetCount; ++index) {
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(2, 1, deepest, index + 2, targetGuard), passed,
                  "add bounded-query target");
    targets.push_back(target);
    passed &= check(graph.addDemand(makeDemand(source, target, outer)),
                    "add bounded-query demand");
    activeDemands.push_back(index);
  }
  for (std::size_t index = 0; index < targetCount; ++index) {
    supplies.push_back({index,
                        makeEdge(source, targets[index],
                                 SyncCoverEdgeKind::CompletionSupply, outer),
                        activeDemands});
  }
  passed &= check(graph.freezeStructure(), "freeze bounded-query graph");
  const SyncCoverExpandedProgram expansion(graph);

  SyncCoverCoverageWorkBudget referenceBudget;
  const SyncCoverCoverageResult reference = computeSyncCoverCoverage(
      graph, expansion, supplies, activeDemands, &referenceBudget);
  const SyncCoverExpandedArena *arena =
      expansion.getArena(graph.getDemands().front());
  const std::size_t scopes = graph.getScopes().size();
  const std::size_t projectionReservation =
      arena ? 6 * scopes * scopes + 4 * arena->getVirtualNodeCount() : 0;
  const std::size_t contextShiftWork =
      sourceGuard.literals.size() * targetGuard.literals.size() * targetCount;
  passed &= check(reference && reference.coversAll() &&
                      referenceBudget.workUnits >=
                          projectionReservation + contextShiftWork,
                  "meter projection ancestry and middle context insertion");

  SyncCoverCoverageWorkBudget exactBudget(referenceBudget.workUnits);
  const SyncCoverCoverageResult exact = computeSyncCoverCoverage(
      graph, expansion, supplies, activeDemands, &exactBudget);
  passed &= check(exact && exact.coversAll() && !exactBudget.exhausted &&
                      exactBudget.workUnits == referenceBudget.workUnits,
                  "accept adversarial coverage at its exact work bound");

  SyncCoverCoverageWorkBudget belowBudget(referenceBudget.workUnits - 1);
  const SyncCoverCoverageResult below = computeSyncCoverCoverage(
      graph, expansion, supplies, activeDemands, &belowBudget);
  passed &= check(below.error == SyncCoverCoverageError::WorkLimitExceeded &&
                      belowBudget.exhausted &&
                      belowBudget.workUnits == belowBudget.maximumWorkUnits,
                  "stop before exceeding the adversarial coverage work bound");

  SyncCoverCoverageWorkBudget tinyBudget(4);
  const SyncCoverCoverageResult tiny = computeSyncCoverCoverage(
      graph, expansion, supplies, activeDemands, &tinyBudget);
  passed &= check(tiny.error == SyncCoverCoverageError::WorkLimitExceeded &&
                      tinyBudget.exhausted && tinyBudget.workUnits <= 4,
                  "reject large metadata before a tiny budget is exceeded");
  return passed;
}

bool testDeepProjectionReservationFailsEarly() {
  bool passed = true;
  SyncCoverGraph graph;
  SyncCoverScopeId outer = 0;
  SyncCoverScopeId deepest = 0;
  constexpr std::size_t loopDepth = 32;
  for (std::size_t depth = 0; depth < loopDepth; ++depth) {
    deepest = takeIndex(
        graph.addScope(deepest, true, SyncCoverTimelineInterval{0, 20}, true),
        passed, "add isolated projection loop");
    if (depth == 0) {
      outer = deepest;
    }
  }
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, deepest, 1), passed,
                "add isolated projection source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, deepest, 2), passed,
                "add isolated projection target");
  passed &= check(graph.addDemand(makeDemand(source, target, outer)),
                  "add isolated projection demand");
  passed &= check(graph.freezeStructure(), "freeze isolated projection graph");
  const SyncCoverExpandedProgram expansion(graph);
  const SyncCoverExpandedArena *arena =
      expansion.getArena(graph.getDemands().front());
  if (!check(arena != nullptr, "build isolated projection arena")) {
    return false;
  }
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0,
       makeEdge(source, target, SyncCoverEdgeKind::CompletionSupply, outer)}};
  const std::vector<SyncCoverDemandId> active{0};
  const std::size_t scopes = graph.getScopes().size();
  const std::size_t completionReservation =
      6 * scopes * scopes + 4 * arena->getVirtualNodeCount();

  SyncCoverCoverageWorkBudget referenceBudget;
  const SyncCoverCoverageResult reference = computeSyncCoverCoverage(
      graph, expansion, supplies, active, &referenceBudget);
  passed &= check(reference && reference.coversAll() &&
                      referenceBudget.workUnits >= completionReservation,
                  "reserve the analytic deep-projection work bound");

  SyncCoverCoverageWorkBudget insufficient(completionReservation - 1);
  const SyncCoverCoverageResult limited = computeSyncCoverCoverage(
      graph, expansion, supplies, active, &insufficient);
  passed &= check(limited.error == SyncCoverCoverageError::WorkLimitExceeded &&
                      insufficient.exhausted &&
                      insufficient.workUnits <= insufficient.maximumWorkUnits,
                  "stop before an underfunded deep projection");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testOneSupplyCoversSeveralDemands();
  passed &= testTwoSuppliesCompose();
  passed &= testDemandQualifiedSupplyDoesNotEscape();
  passed &= testDistanceZeroSupplyDoesNotCoverRecurrence();
  passed &= testIssueOrderDoesNotCreatePrefixCompletion();
  passed &= testCertifiedFrontierCreatesPrefixCompletion();
  passed &= testGuardedRecurrenceAndCompactCarry();
  passed &= testCompactCarryKindsAndResourceIsolation();
  passed &= testOptionalIntermediateAvailability();
  passed &= testLoopSummaryMatchesExplicitUnrollings();
  passed &= testChildPortsPreserveEndpointIdentity();
  passed &= testChildLocalPairExportsThroughPorts();
  passed &= testChildFixedPathClosureMatchesExplicitBody();
  passed &= testNestedExportCrossesInactiveMiddleSummary();
  passed &= testGuardedProtocolRemainsLocal();
  passed &= testExpansionOwnershipAndMaximumHorizon();
  passed &= testUnavailableArenaIsReported();
  passed &= testPairWorkspaceLimitReturnsNoPartialCoverage();
  passed &= testPairMechanismRowsAreBoundedWithoutDemands();
  passed &= testBaseLimitFailsClosed();
  passed &= testHierarchicalMetadataHonorsBaseNodeLimit();
  passed &= testInvalidSupplyFailsClosed();
  passed &= testBoundedCoverageMetersWholeQuery();
  passed &= testDeepProjectionReservationFailsEarly();
  return passed ? 0 : 1;
}
