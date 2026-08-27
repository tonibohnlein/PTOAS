// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncSelectionTest failure: " << message << '\n';
  }
  return condition;
}

template <typename Result>
bool check(const Result &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

template <typename Result>
std::size_t takeIndex(const Result &result, bool &passed,
                      std::string_view message) {
  passed &=
      check(static_cast<bool>(result) && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverDemand demand(SyncCoverNodeId source, SyncCoverNodeId target,
                       SyncCoverScopeId scope = 0, unsigned distance = 0) {
  SyncCoverDemand result;
  result.source = source;
  result.target = target;
  result.scope = scope;
  result.distance = distance;
  return result;
}

SyncCoverEdge supply(SyncCoverNodeId source, SyncCoverNodeId target,
                     SyncCoverScopeId scope = 0, unsigned distance = 0) {
  SyncCoverEdge result;
  result.source = source;
  result.target = target;
  result.scope = scope;
  result.distance = distance;
  result.kind = SyncCoverEdgeKind::CompletionSupply;
  return result;
}

SyncCoverAnchor before(SyncCoverNodeId node) {
  return {SyncCoverAnchorKind::BeforeNode, node, 0, 0};
}

SyncCoverAnchor after(SyncCoverNodeId node) {
  return {SyncCoverAnchorKind::AfterNode, node, 0, 0};
}

CanonicalSyncMechanismDescriptor event(CanonicalSyncEventDomainId domain,
                                       std::uint32_t sourceResource,
                                       std::uint32_t targetResource,
                                       SyncCoverNodeId source,
                                       SyncCoverNodeId target) {
  CanonicalSyncMechanismDescriptor result;
  result.eventUses.push_back({domain, 1, std::nullopt});
  result.actions.push_back({CanonicalSyncActionKind::EventSet,
                            sourceResource,
                            after(source),
                            0,
                            0,
                            {}});
  result.actions.push_back({CanonicalSyncActionKind::EventWait,
                            targetResource,
                            before(target),
                            0,
                            0,
                            {}});
  result.supplies.push_back({supply(source, target), 0, std::nullopt,
                             std::nullopt, std::nullopt,
                             CanonicalSyncSupplyProof::DirectAction});
  return result;
}

CanonicalSyncMechanismDescriptor
protocol(CanonicalSyncEventDomainId domain, std::uint32_t sourceResource,
         std::uint32_t targetResource, SyncCoverNodeId source,
         SyncCoverNodeId target, SyncCoverScopeId loop, std::size_t width,
         unsigned distance = 0) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back({domain, width, loop});
  for (std::size_t lane = 0; lane < width; ++lane) {
    result.actions.push_back({CanonicalSyncActionKind::EventSet,
                              sourceResource,
                              after(source),
                              0,
                              lane,
                              {}});
  }
  for (std::size_t lane = 0; lane < width; ++lane) {
    result.actions.push_back({CanonicalSyncActionKind::EventWait,
                              targetResource,
                              before(target),
                              0,
                              lane,
                              {}});
  }
  result.supplies.push_back({supply(source, target, loop, distance), 0,
                             std::nullopt, 0, width,
                             CanonicalSyncSupplyProof::VerifiedProtocol});
  return result;
}

CanonicalSyncMechanismDescriptor barrier(std::uint32_t actionResource,
                                         std::vector<std::uint32_t> resources,
                                         SyncCoverNodeId source,
                                         SyncCoverNodeId target) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Barrier;
  result.actions.push_back(
      {CanonicalSyncActionKind::Barrier, actionResource, before(target),
       std::nullopt, 0, std::move(resources), CanonicalSyncBarrierKind::All});
  result.supplies.push_back({supply(source, target), std::nullopt, 0,
                             std::nullopt, std::nullopt,
                             CanonicalSyncSupplyProof::DirectAction});
  return result;
}

std::vector<SyncCoverDemandId> allDemands(const SyncCoverGraph &graph) {
  std::vector<SyncCoverDemandId> result(graph.getDemands().size());
  std::iota(result.begin(), result.end(), 0);
  return result;
}

bool testBatchedSingletonCoverageMatchesIndependentQueries() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first source");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add first target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add second target");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget)),
                  "add first demand");
  passed &= check(graph.addDemand(demand(secondSource, secondTarget)),
                  "add second demand");
  passed &= check(graph.freezeStructure(), "freeze singleton graph");
  const SyncCoverExpandedProgram expansion(graph);
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0, supply(firstSource, firstTarget)},
      {1, supply(secondSource, secondTarget)},
      {2, supply(firstSource, firstTarget)},
      {2, supply(secondSource, secondTarget)},
  };
  const SyncCoverSingletonCoverageResult batched =
      computeSyncCoverSingletonCoverage(graph, expansion, 3, supplies);
  passed &= check(static_cast<bool>(batched), "batch singleton coverage");
  for (std::size_t mechanism = 0; mechanism < 3; ++mechanism) {
    std::vector<SyncCoverCompletionSupply> selected;
    for (const auto &candidate : supplies) {
      if (candidate.mechanism == mechanism) {
        selected.push_back(candidate);
      }
    }
    const SyncCoverCoverageResult independent =
        computeSyncCoverCoverage(graph, expansion, selected);
    passed &= check(independent &&
                        independent.covered == batched.mechanisms[mechanism],
                    "batched singleton equals independent coverage");
  }
  return passed;
}

bool testBatchedSingletonCoverageHandlesRecurrence() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add recurrence scope");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, loop, 1, {}, {2}), passed, "add recurrence source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 2), passed, "add recurrence target");
  passed &= check(graph.addDemand(demand(source, target, loop, 1)),
                  "add recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze recurrence graph");
  const SyncCoverExpandedProgram expansion(graph);
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0, supply(source, target, loop, 1)}};
  const SyncCoverSingletonCoverageResult batched =
      computeSyncCoverSingletonCoverage(graph, expansion, 1, supplies);
  const SyncCoverCoverageResult independent =
      computeSyncCoverCoverage(graph, expansion, supplies);
  passed &= check(batched && independent &&
                      batched.mechanisms[0] == independent.covered &&
                      batched.mechanisms[0].contains(0),
                  "batched recurrence coverage matches independent query");
  return passed;
}

bool testBatchedSingletonCoverageHandlesFixedCompletionPrefix() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}),
                                           passed, "add prefix source");
  const SyncCoverNodeId marker = takeIndex(graph.addNode(1, 1, 0, 1, {}, {2}),
                                           passed, "add prefix marker");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add prefix target");
  passed &= check(graph.addEdge(supply(source, marker)),
                  "add fixed completion prefix");
  passed &=
      check(graph.addDemand(demand(source, target)), "add prefixed demand");
  passed &= check(graph.freezeStructure(), "freeze prefixed graph");
  const SyncCoverExpandedProgram expansion(graph);
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {0, supply(marker, target)}};
  const SyncCoverSingletonCoverageResult batched =
      computeSyncCoverSingletonCoverage(graph, expansion, 1, supplies);
  const SyncCoverCoverageResult independent =
      computeSyncCoverCoverage(graph, expansion, supplies);
  passed &= check(batched && independent && independent.covered.contains(0) &&
                      batched.mechanisms[0] == independent.covered,
                  "fixed completion prefix reaches a later mechanism supply");
  return passed;
}

bool testBatchedSingletonCoverageRejectsOversizedResult() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add limit source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add limit target");
  passed &= check(graph.addDemand(demand(source, target)), "add limit demand");
  passed &= check(graph.freezeStructure(), "freeze limit graph");
  const SyncCoverExpandedProgram expansion(graph);
  constexpr std::size_t tooManyMechanisms = (1U << 24) + 1;
  const SyncCoverSingletonCoverageResult result =
      computeSyncCoverSingletonCoverage(graph, expansion, tooManyMechanisms,
                                        {});
  passed &= check(result.error == SyncCoverCoverageError::LimitExceeded &&
                      result.mechanisms.empty(),
                  "oversized singleton result fails before allocation");
  SyncCoverGraph emptyGraph;
  passed &= check(emptyGraph.freezeStructure(), "freeze empty limit graph");
  const SyncCoverExpandedProgram emptyExpansion(emptyGraph);
  constexpr std::size_t tooManyRows = (1U << 20) + 1;
  const SyncCoverSingletonCoverageResult emptyResult =
      computeSyncCoverSingletonCoverage(emptyGraph, emptyExpansion, tooManyRows,
                                        {});
  passed &= check(emptyResult.error == SyncCoverCoverageError::LimitExceeded &&
                      emptyResult.mechanisms.empty(),
                  "zero-demand singleton rows remain bounded");
  return passed;
}

bool testFixedCompletionNeedsNoSelectedMechanism() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add baseline source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add baseline target");
  passed &= check(graph.addEdge(supply(source, target)),
                  "add baseline completion edge");
  passed &=
      check(graph.addDemand(demand(source, target)), "add baseline demand");
  passed &= check(graph.freezeStructure(), "freeze baseline graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.freeze(), "fixed-covered problem freezes");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(problem, selection);
  passed &= check(selection && selection.mechanisms.empty() && verified,
                  "fixed completion requires no synchronization mechanism");
  return passed;
}

bool testReverseDeletionPreservesBaselineCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId baselineSource =
      takeIndex(graph.addNode(3, 1, 0, 0), passed, "add mixed baseline source");
  const SyncCoverNodeId baselineTarget =
      takeIndex(graph.addNode(4, 1, 0, 1), passed, "add mixed baseline target");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 2, {}, {2}), passed, "add mixed source");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add mixed first target");
  const SyncCoverNodeId second =
      takeIndex(graph.addNode(2, 1, 0, 4), passed, "add mixed second target");
  passed &= check(graph.addEdge(supply(baselineSource, baselineTarget)),
                  "add mixed baseline edge");
  SyncCoverEdge targetOrder = supply(first, second);
  targetOrder.kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  passed &= check(graph.addEdge(targetOrder), "add mixed target order");
  passed &= check(graph.addDemand(demand(baselineSource, baselineTarget)),
                  "add mixed baseline demand");
  passed &=
      check(graph.addDemand(demand(source, second)), "add mixed broad demand");
  passed &=
      check(graph.addDemand(demand(source, first)), "add mixed narrow demand");
  passed &= check(graph.freezeStructure(), "freeze mixed graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 2, {}}), "add mixed domain");
  const CanonicalSyncMechanismId eventId =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, second)), passed,
                "add mixed event");
  const CanonicalSyncMechanismId barrierId = takeIndex(
      problem.internMechanism(barrier(2, {1, 2, 3, 4}, source, first)), passed,
      "add mixed barrier");
  passed &= check(problem.freeze(), "freeze mixed problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection && selection.statistics.deletionEvaluations != 0 &&
                      selection.mechanisms ==
                          std::vector<CanonicalSyncMechanismId>{barrierId} &&
                      !std::binary_search(selection.mechanisms.begin(),
                                          selection.mechanisms.end(), eventId),
                  "reverse deletion retains baseline while removing event");
  return passed;
}

bool testInactiveRecurrenceDoesNotBuildAnArena() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add inactive loop");
  const SyncCoverNodeId activeSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add active source");
  const SyncCoverNodeId activeTarget =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add active target");
  const SyncCoverNodeId inactiveSource =
      takeIndex(graph.addNode(3, 1, loop, 2), passed, "add inactive source");
  const SyncCoverNodeId inactiveTarget =
      takeIndex(graph.addNode(4, 1, loop, 3), passed, "add inactive target");
  passed &= check(graph.addDemand(demand(activeSource, activeTarget)),
                  "add active demand");
  passed &=
      check(graph.addDemand(demand(inactiveSource, inactiveTarget, loop, 8)),
            "add inactive recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze inactive graph");
  SyncCoverExpansionLimits expansionLimits;
  expansionLimits.maximumArenaNodes = 10;
  expansionLimits.maximumArenaEdges = 64;
  expansionLimits.maximumTotalNodes = 20;
  expansionLimits.maximumTotalEdges = 128;
  const SyncCoverExpandedProgram allExpansion(graph, expansionLimits);
  passed &= check(!allExpansion.getArena(graph.getDemands()[1]),
                  "all-demand expansion rejects the large recurrence arena");

  CanonicalSyncPatternProblem::Limits limits;
  CanonicalSyncPatternProblem problem(graph, {0}, limits, expansionLimits);
  passed &=
      check(problem.addEventDomain({0, 1, 2, 2, {}}), "add active-only domain");
  passed &=
      check(problem.internMechanism(event(0, 1, 2, activeSource, activeTarget)),
            "add active-only event");
  passed &= check(problem.freeze(), "inactive recurrence cannot poison freeze");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection && verifyCanonicalSyncSelection(problem, selection),
                  "active-only selection verifies");
  return passed;
}

bool testRoundTripPatternActivatesJointCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add round loop");
  const SyncCoverNodeId early = takeIndex(graph.addNode(2, 1, loop, 0, {}, {1}),
                                          passed, "add round early");
  const SyncCoverNodeId late = takeIndex(graph.addNode(1, 1, loop, 1, {}, {2}),
                                         passed, "add round late");
  passed &= check(graph.addDemand(demand(late, late, loop, 1)),
                  "add composed recurrence demand");
  passed &= check(graph.freezeStructure(), "freeze round graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 8, {}}), "add carried domain");
  passed &=
      check(problem.addEventDomain({1, 2, 1, 8, {}}), "add closing domain");
  const auto verifier = [](const CanonicalSyncMechanismDescriptor &) {
    return true;
  };
  const CanonicalSyncMechanismId carried =
      takeIndex(problem.internVerifiedProtocol(
                    protocol(0, 1, 2, late, early, loop, 1, 1), verifier),
                passed, "add carried mechanism");
  const CanonicalSyncMechanismId blockedClosing =
      takeIndex(problem.internMechanism(event(1, 2, 1, early, late)), passed,
                "add blocked closing mechanism");
  const CanonicalSyncMechanismId closing =
      takeIndex(problem.internVerifiedProtocol(
                    protocol(1, 2, 1, early, late, loop, 1), verifier),
                passed, "add alternate closing mechanism");
  passed &= check(problem.addConflict(carried, blockedClosing),
                  "block first round-trip pair");
  CanonicalSyncRoundTripOptions oneEvaluation;
  oneEvaluation.maximumEvaluations = 1;
  const CanonicalSyncProblemResult capped =
      addCanonicalSyncRoundTripPatterns(problem, oneEvaluation);
  passed &= check(capped && capped.index == 0,
                  "infeasible pair consumes the evaluation cap");
  CanonicalSyncRoundTripOptions twoEvaluations;
  twoEvaluations.maximumEvaluations = 2;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncRoundTripPatterns(problem, twoEvaluations);
  passed &= check(generated && generated.index == 1,
                  "generate one bounded round-trip pattern");
  passed &= check(problem.freeze(), "freeze round-trip problem");
  passed &=
      check(!problem.getPatterns()[carried].coverage.contains(0) &&
                !problem.getPatterns()[blockedClosing].coverage.contains(0) &&
                !problem.getPatterns()[closing].coverage.contains(0) &&
                problem.getPatterns().back().coverage.contains(0),
            "only complete round trip covers composed demand");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(
      selection && selection.mechanisms ==
                       std::vector<CanonicalSyncMechanismId>{carried, closing},
      "shared round-trip members are selected once");
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(problem, selection);
  passed &= check(static_cast<bool>(verified),
                  "fresh final coverage accepts round trip");
  CanonicalSyncGreedyOptions bounded;
  bounded.maximumWorkUnits = 1;
  passed &= check(selectCanonicalSyncPatterns(problem, bounded).error ==
                      CanonicalSyncSelectionError::WorkLimitExceeded,
                  "global work bound covers construction and search scans");
  return passed;
}

bool testScarcityUsesBarrierFallback() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add scarce source one");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add scarce source two");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add scarce target one");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add scarce target two");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget)),
                  "add scarce demand one");
  passed &= check(graph.addDemand(demand(secondSource, secondTarget)),
                  "add scarce demand two");
  passed &= check(graph.freezeStructure(), "freeze scarce graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 1, {}}), "add scarce domain");
  const CanonicalSyncMechanismId firstEvent = takeIndex(
      problem.internMechanism(event(0, 1, 2, firstSource, firstTarget)), passed,
      "add first scarce event");
  const CanonicalSyncMechanismId secondEvent = takeIndex(
      problem.internMechanism(event(0, 1, 2, secondSource, secondTarget)),
      passed, "add second scarce event");
  const CanonicalSyncMechanismId fallback = takeIndex(
      problem.internMechanism(barrier(2, {1, 2}, secondSource, secondTarget)),
      passed, "add scarcity barrier");
  passed &= check(problem.freeze(), "freeze scarcity problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection && selection.mechanisms ==
                                   std::vector<CanonicalSyncMechanismId>{
                                       firstEvent, fallback},
                  "greedy uses barrier when overlapping event is infeasible");
  passed &=
      check(!std::binary_search(selection.mechanisms.begin(),
                                selection.mechanisms.end(), secondEvent) &&
                selection.allocation.domains[0].required == 1,
            "scarce event remains unselected and allocation is exact");
  return passed;
}

bool testReservationsAndFinalValidation() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}),
                                           passed, "add reserve source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add reserve target");
  passed &=
      check(graph.addDemand(demand(source, target)), "add reserve demand");
  passed &= check(graph.freezeStructure(), "freeze reserve graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 4, {0, 2, 9}}),
                  "add reserved domain");
  const CanonicalSyncMechanismId mechanism =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, target)), passed,
                "add reserved event");
  passed &= check(problem.freeze(), "freeze reserved problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &=
      check(selection && selection.mechanisms ==
                             std::vector<CanonicalSyncMechanismId>{mechanism},
            "select reserved event");
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(problem, selection);
  passed &= check(verified && verified.allocation.domains[0].available == 2 &&
                      verified.allocation.domains[0].uses[0].ids ==
                          std::vector<unsigned>{1},
                  "allocation skips in-range reservations only");
  return passed;
}

bool testAllocatorWidthsReuseAndConflicts() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add allocator loop");
  std::vector<SyncCoverNodeId> sources;
  std::vector<SyncCoverNodeId> targets;
  for (std::size_t index = 0; index < 3; ++index) {
    sources.push_back(takeIndex(graph.addNode(1, 1, loop, index * 2, {}, {2}),
                                passed, "add allocator source"));
    targets.push_back(takeIndex(graph.addNode(2, 1, loop, index * 2 + 1),
                                passed, "add allocator target"));
    passed &=
        check(graph.addDemand(demand(sources.back(), targets.back(), loop)),
              "add allocator demand");
  }
  passed &= check(graph.freezeStructure(), "freeze allocator graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 2, {}}), "add allocator domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, sources[0], targets[0])),
                passed, "add allocator first event");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(event(0, 1, 2, sources[1], targets[1])),
                passed, "add allocator second event");
  const CanonicalSyncMechanismId wide = takeIndex(
      problem.internVerifiedProtocol(
          protocol(0, 1, 2, sources[2], targets[2], loop, 2),
          [](const CanonicalSyncMechanismDescriptor &descriptor) {
            return descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
                   descriptor.eventUses.size() == 1 &&
                   descriptor.eventUses[0].width == 2;
          }),
      passed, "add allocator wide event");
  const CanonicalSyncResourceAllocation reused =
      allocateCanonicalSyncEvents(problem, {first, second});
  passed &= check(
      reused.valid && reused.feasible && reused.domains[0].required == 1 &&
          reused.domains[0].uses[0].ids == std::vector<unsigned>{0} &&
          reused.domains[0].uses[1].ids == std::vector<unsigned>{0},
      "nonoverlapping lifetimes reuse one physical ID");
  const CanonicalSyncResourceAllocation widened =
      allocateCanonicalSyncEvents(problem, {wide});
  passed &= check(
      widened.valid && widened.feasible && widened.domains[0].required == 2 &&
          widened.domains[0].uses[0].ids == std::vector<unsigned>{0, 1},
      "weighted interval receives distinct lanes");
  passed &= check(problem.addConflict(first, second), "add allocator conflict");
  const CanonicalSyncResourceAllocation conflicting =
      allocateCanonicalSyncEvents(problem, {first, second});
  passed &= check(!conflicting.valid && !conflicting.feasible,
                  "explicit mechanism conflicts fail closed");
  return passed;
}

bool testVerifiedProtocolTrustBoundary() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add protocol loop");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, loop, 1, {}, {2}), passed, "add protocol source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 2), passed, "add protocol target");
  passed &= check(graph.addDemand(demand(source, target, loop, 1)),
                  "add protocol demand");
  passed &= check(graph.freezeStructure(), "freeze protocol graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 2, {}}), "add protocol domain");
  const auto verifier = [=](const CanonicalSyncMechanismDescriptor &candidate) {
    return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
           candidate.eventUses.size() == 1 &&
           candidate.eventUses[0].recurrenceScope == loop &&
           candidate.supplies.size() == 1 && candidate.actions.size() == 2 &&
           candidate.actions[0].anchor.kind == SyncCoverAnchorKind::AfterNode &&
           candidate.actions[0].anchor.node == source &&
           candidate.actions[1].anchor.kind ==
               SyncCoverAnchorKind::BeforeNode &&
           candidate.actions[1].anchor.node == target &&
           candidate.supplies[0].edge.source == source &&
           candidate.supplies[0].edge.target == target &&
           candidate.supplies[0].edge.distance == 1;
  };
  const CanonicalSyncProblemResult admitted = problem.internVerifiedProtocol(
      protocol(0, 1, 2, source, target, loop, 1, 1), verifier);
  passed &= check(admitted, "verified recurrence protocol is admitted");

  CanonicalSyncMechanismDescriptor rejected =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  rejected.supplies[0].edge.distance = 0;
  passed &= check(problem.internVerifiedProtocol(rejected, verifier).error ==
                      CanonicalSyncProblemError::UnverifiedProtocol,
                  "independent verifier rejects a tampered distance");
  CanonicalSyncMechanismDescriptor badAnchor =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  badAnchor.actions[0].anchor = before(source);
  passed &= check(problem.internVerifiedProtocol(badAnchor, verifier).error ==
                      CanonicalSyncProblemError::UnverifiedProtocol,
                  "independent verifier rejects a tampered anchor");
  CanonicalSyncMechanismDescriptor missingAction =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  missingAction.actions.pop_back();
  passed &=
      check(problem.internVerifiedProtocol(missingAction, verifier).error ==
                CanonicalSyncProblemError::InvalidMechanism,
            "common admission rejects a missing protocol action");
  CanonicalSyncMechanismDescriptor badLane =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  badLane.actions[0].eventLane = 1;
  passed &= check(problem.internVerifiedProtocol(badLane, verifier).error ==
                      CanonicalSyncProblemError::InvalidMechanism,
                  "common admission rejects an out-of-range protocol lane");
  CanonicalSyncMechanismDescriptor badScope =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  badScope.eventUses[0].recurrenceScope = 0;
  passed &= check(problem.internVerifiedProtocol(badScope, verifier).error ==
                      CanonicalSyncProblemError::InvalidMechanism,
                  "common admission rejects a non-loop recurrence scope");
  return passed;
}

bool testFailClosedConstruction() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add fail source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add fail target");
  passed &= check(graph.addDemand(demand(source, target)), "add fail demand");
  passed &= check(graph.freezeStructure(), "freeze fail graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}), "add fail domain");
  CanonicalSyncMechanismDescriptor protocol = event(0, 1, 2, source, target);
  protocol.kind = CanonicalSyncMechanismKind::Protocol;
  protocol.eventUses[0].recurrenceScope = 0;
  protocol.supplies[0].proof = CanonicalSyncSupplyProof::VerifiedProtocol;
  protocol.supplies[0].produceAction = 0;
  protocol.supplies[0].consumeAction = 1;
  passed &= check(problem.internMechanism(protocol).error ==
                      CanonicalSyncProblemError::UnverifiedProtocol,
                  "protocol cannot bypass its verifier");
  passed &= check(
      problem.internMechanism(barrier(99, {1, 2}, source, target)).error ==
          CanonicalSyncProblemError::InvalidMechanism,
      "barrier action must execute on the target resource");
  CanonicalSyncMechanismDescriptor wideEvent = event(0, 1, 2, source, target);
  wideEvent.eventUses[0].width = 2;
  passed &= check(problem.internMechanism(wideEvent).error ==
                      CanonicalSyncProblemError::InvalidMechanism,
                  "direct events cannot leave allocated lanes unmaterialized");
  passed &= check(problem.freeze().error ==
                      CanonicalSyncProblemError::UncoverableDemand,
                  "missing fallback fails before search");

  CanonicalSyncPatternProblem::Limits limits;
  limits.maximumIncidences = 1;
  CanonicalSyncPatternProblem limited(graph, allDemands(graph), limits);
  passed &=
      check(limited.addEventDomain({0, 1, 2, 8, {}}), "add incidence domain");
  passed &= check(limited.internMechanism(event(0, 1, 2, source, target)),
                  "add incidence event");
  passed &= check(limited.internMechanism(barrier(2, {1, 2}, source, target)),
                  "add incidence barrier");
  passed &=
      check(limited.freeze().error == CanonicalSyncProblemError::LimitExceeded,
            "incidence limit fails during construction");

  CanonicalSyncPatternProblem::Limits patternLimits;
  patternLimits.maximumPatterns = 2;
  CanonicalSyncPatternProblem patternLimited(graph, allDemands(graph),
                                             patternLimits);
  passed &= check(patternLimited.addEventDomain({0, 1, 2, 8, {}}),
                  "add pattern-limit domain");
  const CanonicalSyncMechanismId first =
      takeIndex(patternLimited.internMechanism(event(0, 1, 2, source, target)),
                passed, "add pattern-limit event");
  const CanonicalSyncMechanismId second = takeIndex(
      patternLimited.internMechanism(barrier(2, {1, 2}, source, target)),
      passed, "add pattern-limit barrier");
  passed &= check(patternLimited
                          .addPattern({CanonicalSyncPatternKind::PipelineScope,
                                       {first, second}})
                          .error == CanonicalSyncProblemError::LimitExceeded,
                  "optional patterns cannot exceed the aggregate limit");
  return passed;
}

} // namespace

int main() {
  const bool passed =
      testBatchedSingletonCoverageMatchesIndependentQueries() &&
      testBatchedSingletonCoverageHandlesRecurrence() &&
      testBatchedSingletonCoverageHandlesFixedCompletionPrefix() &&
      testBatchedSingletonCoverageRejectsOversizedResult() &&
      testFixedCompletionNeedsNoSelectedMechanism() &&
      testReverseDeletionPreservesBaselineCoverage() &&
      testInactiveRecurrenceDoesNotBuildAnArena() &&
      testRoundTripPatternActivatesJointCoverage() &&
      testScarcityUsesBarrierFallback() &&
      testReservationsAndFinalValidation() &&
      testAllocatorWidthsReuseAndConflicts() &&
      testVerifiedProtocolTrustBoundary() && testFailClosedConstruction();
  return passed ? 0 : 1;
}
