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
#include <iterator>
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
eventInScope(CanonicalSyncEventDomainId domain, std::uint32_t sourceResource,
             std::uint32_t targetResource, SyncCoverNodeId source,
             SyncCoverNodeId target, SyncCoverScopeId scope) {
  CanonicalSyncMechanismDescriptor result =
      event(domain, sourceResource, targetResource, source, target);
  result.supplies.front().edge.scope = scope;
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
  result.selectionTier = CanonicalSyncSelectionTier::PipeAllRescue;
  result.actions.push_back(
      {CanonicalSyncActionKind::Barrier, actionResource, before(target),
       std::nullopt, 0, std::move(resources), CanonicalSyncBarrierKind::All});
  result.supplies.push_back({supply(source, target), std::nullopt, 0,
                             std::nullopt, std::nullopt,
                             CanonicalSyncSupplyProof::DirectAction});
  return result;
}

CanonicalSyncMechanismDescriptor targetedBarrier(std::uint32_t resource,
                                                 SyncCoverNodeId source,
                                                 SyncCoverNodeId target,
                                                 SyncCoverScopeId scope = 0) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Barrier;
  result.actions.push_back({CanonicalSyncActionKind::Barrier,
                            resource,
                            before(target),
                            std::nullopt,
                            0,
                            {resource},
                            CanonicalSyncBarrierKind::Targeted});
  result.supplies.push_back({supply(source, target, scope), std::nullopt, 0,
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
  takeIndex(graph.addNode(5, 1, 0, 5), passed, "add mixed idle resource");
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
      problem.internMechanism(barrier(2, {1, 2, 3, 4, 5}, source, first)),
      passed, "add mixed barrier");
  passed &= check(problem.freeze(), "freeze mixed problem");
  CanonicalSyncGreedyOptions options;
  options.maximumTier = CanonicalSyncSelectionTier::PipeAllRescue;
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(problem, options);
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

bool testDirectPairDiscoversJointCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add pair source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed, "add pair middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add pair target");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add pair-composed demand");
  passed &= check(graph.freezeStructure(), "freeze direct-pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 8, {}}), "add first pair domain");
  passed &=
      check(problem.addEventDomain({1, 2, 3, 8, {}}), "add second pair domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, middle)), passed,
                "add first pair event");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(event(1, 2, 3, middle, target)), passed,
                "add second pair event");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 1,
                  "retain one direct pair with extra coverage");
  passed &= check(problem.freeze(), "freeze direct-pair problem");
  passed &= check(!problem.getPatterns()[first].coverage.contains(0) &&
                      !problem.getPatterns()[second].coverage.contains(0) &&
                      problem.getPatterns().back().kind ==
                          CanonicalSyncPatternKind::DirectPair &&
                      problem.getPatterns().back().coverage.contains(0) &&
                      problem.getPatterns().back().extraCoverageCount == 1,
                  "store only the pair's exact extra coverage");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection &&
                      selection.mechanisms ==
                          std::vector<CanonicalSyncMechanismId>{first, second},
                  "select both shared pair members exactly once");
  CanonicalSyncGreedyOptions fixedOptions;
  fixedOptions.strategy = CanonicalSyncSelectionStrategy::FixedCover;
  passed &= check(selectCanonicalSyncPatterns(problem, fixedOptions),
                  "fixed-cover selects the retained pair column");
  CanonicalSyncGreedyOptions singletonOptions;
  singletonOptions.strategy =
      CanonicalSyncSelectionStrategy::ActionAwareSingleton;
  passed &= check(
      selectCanonicalSyncPatterns(problem, singletonOptions).error ==
          CanonicalSyncSelectionError::NoCoveringPattern,
      "singleton action selection cannot activate two-member-only coverage");
  return passed;
}

bool testPairPreparationLimitKeepsSingletonCorrectness() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed,
                "add optional-pair source");
  const SyncCoverNodeId middle = takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}),
                                           passed, "add optional-pair middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add optional-pair target");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add optional-pair demand");
  passed &= check(graph.freezeStructure(), "freeze optional-pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 3, 8, {}}),
                  "add direct singleton domain");
  passed &= check(problem.addEventDomain({1, 1, 2, 8, {}}),
                  "add optional-pair first domain");
  passed &= check(problem.addEventDomain({2, 2, 3, 8, {}}),
                  "add optional-pair second domain");
  const CanonicalSyncMechanismId direct =
      takeIndex(problem.internMechanism(event(0, 1, 3, source, target)), passed,
                "add covering direct singleton");
  passed &= check(problem.internMechanism(event(1, 1, 2, source, middle)),
                  "add optional-pair first member");
  passed &= check(problem.internMechanism(event(2, 2, 3, middle, target)),
                  "add optional-pair second member");

  CanonicalSyncDirectPairOptions options;
  options.pairCoverageLimits.maximumWorkspaceWords = 0;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem, options);
  passed &= check(generated && generated.index == 0 &&
                      problem.wasPatternGenerationTruncated() &&
                      problem.getPatternStatistics().directPairProposals == 1 &&
                      problem.getPatternStatistics().directPairEvaluations == 0,
                  "truncate an optional pair scope at its workspace bound");
  passed &= check(problem.freeze(),
                  "freeze singleton-valid problem after pair truncation");
  passed &= check(problem.getPatternStatistics().directPairProposals == 1 &&
                      problem.getPatternStatistics().directPairEvaluations == 0,
                  "preserve pair truncation statistics through freeze");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &=
      check(selection && selection.mechanisms ==
                             std::vector<CanonicalSyncMechanismId>{direct},
            "retain singleton correctness after optional pair truncation");
  return passed;
}

bool testPairOwnerUsesEverySupplyScope() {
  bool passed = true;
  const auto run = [&](bool leftNodesFirst) {
    SyncCoverGraph graph;
    const SyncCoverScopeId left = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 100}, true),
        passed, "add multi-supply left loop");
    const SyncCoverScopeId right = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{0, 100}, true),
        passed, "add multi-supply right loop");
    SyncCoverNodeId leftSource = 0;
    SyncCoverNodeId leftMiddle = 0;
    SyncCoverNodeId leftTarget = 0;
    SyncCoverNodeId rightSource = 0;
    SyncCoverNodeId rightTarget = 0;
    unsigned order = 0;
    const auto addLeft = [&]() {
      leftSource = takeIndex(graph.addNode(1, 1, left, order++, {}, {2}),
                             passed, "add multi-supply left source");
      leftMiddle = takeIndex(graph.addNode(2, 1, left, order++, {}, {3}),
                             passed, "add multi-supply left middle");
      leftTarget = takeIndex(graph.addNode(3, 1, left, order++), passed,
                             "add multi-supply left target");
    };
    const auto addRight = [&]() {
      rightSource = takeIndex(graph.addNode(7, 1, right, order++, {}, {8}),
                              passed, "add multi-supply right source");
      rightTarget = takeIndex(graph.addNode(8, 1, right, order++), passed,
                              "add multi-supply right target");
    };
    if (leftNodesFirst) {
      addLeft();
      addRight();
    } else {
      addRight();
      addLeft();
    }
    const SyncCoverNodeId rootSource =
        takeIndex(graph.addNode(4, 1, 0, order++, {}, {5}), passed,
                  "add multi-supply root source");
    const SyncCoverNodeId rootMiddle =
        takeIndex(graph.addNode(5, 1, 0, order++, {}, {6}), passed,
                  "add multi-supply root middle");
    const SyncCoverNodeId rootTarget =
        takeIndex(graph.addNode(6, 1, 0, order++), passed,
                  "add multi-supply root target");
    passed &= check(graph.freezeStructure(), "freeze multi-supply graph");

    CanonicalSyncPatternProblem problem(graph, allDemands(graph));
    passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                    "add multi-supply left domain");
    passed &= check(problem.addEventDomain({1, 7, 8, 8, {}}),
                    "add multi-supply right domain");
    passed &= check(problem.addEventDomain({2, 2, 3, 8, {}}),
                    "add multi-supply connector domain");
    passed &= check(problem.addEventDomain({3, 4, 5, 8, {}}),
                    "add first root connector domain");
    passed &= check(problem.addEventDomain({4, 5, 6, 8, {}}),
                    "add second root connector domain");
    CanonicalSyncMechanismDescriptor multi =
        protocol(0, 1, 2, leftSource, leftMiddle, left, 1);
    CanonicalSyncMechanismDescriptor rightProtocol =
        protocol(1, 7, 8, rightSource, rightTarget, right, 1);
    rightProtocol.actions[0].eventUse = 1;
    rightProtocol.actions[1].eventUse = 1;
    rightProtocol.supplies.front().eventUse = 1;
    rightProtocol.supplies.front().produceAction = 2;
    rightProtocol.supplies.front().consumeAction = 3;
    multi.eventUses.push_back(std::move(rightProtocol.eventUses.front()));
    multi.actions.insert(multi.actions.end(),
                         std::make_move_iterator(rightProtocol.actions.begin()),
                         std::make_move_iterator(rightProtocol.actions.end()));
    multi.supplies.push_back(std::move(rightProtocol.supplies.front()));
    passed &= check(
        problem.internVerifiedProtocol(std::move(multi),
                                       [](const auto &candidate) {
                                         return candidate.supplies.size() == 2;
                                       }),
        "add multi-supply mechanism");
    passed &= check(problem.internMechanism(
                        eventInScope(2, 2, 3, leftMiddle, leftTarget, left)),
                    "add multi-supply connector");
    passed &=
        check(problem.internMechanism(event(3, 4, 5, rootSource, rootMiddle)),
              "add root connector first member");
    passed &=
        check(problem.internMechanism(event(4, 5, 6, rootMiddle, rootTarget)),
              "add root connector second member");
    CanonicalSyncDirectPairOptions options;
    options.maximumEvaluationsPerScope = 1;
    const CanonicalSyncProblemResult generated =
        addCanonicalSyncDirectPairPatterns(problem, options);
    const CanonicalSyncPatternStatistics &statistics =
        problem.getPatternStatistics();
    passed &= check(generated && generated.index == 0 &&
                        problem.wasPatternGenerationTruncated() &&
                        statistics.directPairProposals == 2 &&
                        statistics.directPairEvaluations == 0,
                    "own a multi-supply pair at the LCA of every supply");
  };
  run(true);
  run(false);
  return passed;
}

bool testOwnerPairBatchesTruncateAtomicallyAndContinue() {
  bool passed = true;
  const auto run = [&](bool coverageRowLimit) {
    SyncCoverGraph graph;
    const SyncCoverScopeId child =
        takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{12, 30}),
                  passed, "add atomic-pair child scope");
    unsigned order = 0;
    const auto addNode =
        [&](std::uint32_t resource, SyncCoverScopeId scope,
            std::string_view message,
            std::vector<std::uint32_t> completionTargets = {}) {
          return takeIndex(graph.addNode(resource, 1, scope, order++, {},
                                         std::move(completionTargets)),
                           passed, message);
        };
    const SyncCoverNodeId firstSource =
        addNode(1, 0, "add first root-chain source", {2});
    const SyncCoverNodeId firstMiddle =
        addNode(2, 0, "add first root-chain middle", {3});
    const SyncCoverNodeId firstTarget =
        addNode(3, 0, "add first root-chain target");
    const SyncCoverNodeId secondSource =
        addNode(4, 0, "add second root-chain source", {5});
    const SyncCoverNodeId secondMiddle =
        addNode(5, 0, "add second root-chain middle", {6});
    const SyncCoverNodeId secondTarget =
        addNode(6, 0, "add second root-chain target");
    const SyncCoverNodeId childSource =
        addNode(7, child, "add child-chain source", {8});
    const SyncCoverNodeId childMiddle =
        addNode(8, child, "add child-chain middle", {9});
    const SyncCoverNodeId childTarget =
        addNode(9, child, "add child-chain target");
    passed &= check(graph.freezeStructure(), "freeze atomic-pair graph");

    CanonicalSyncPatternProblem::Limits problemLimits;
    if (!coverageRowLimit) {
      problemLimits.maximumPatternProposals = 1;
    }
    CanonicalSyncPatternProblem problem(graph, allDemands(graph),
                                        problemLimits);
    for (const CanonicalSyncEventDomain &domain :
         std::vector<CanonicalSyncEventDomain>{{0, 1, 2, 8, {}},
                                               {1, 2, 3, 8, {}},
                                               {2, 4, 5, 8, {}},
                                               {3, 5, 6, 8, {}},
                                               {4, 7, 8, 8, {}},
                                               {5, 8, 9, 8, {}}}) {
      passed &=
          check(problem.addEventDomain(domain), "add atomic-pair event domain");
    }
    passed &=
        check(problem.internMechanism(event(0, 1, 2, firstSource, firstMiddle)),
              "add first root-chain member");
    passed &=
        check(problem.internMechanism(event(1, 2, 3, firstMiddle, firstTarget)),
              "add first root-chain connector");
    passed &= check(
        problem.internMechanism(event(2, 4, 5, secondSource, secondMiddle)),
        "add second root-chain member");
    passed &= check(
        problem.internMechanism(event(3, 5, 6, secondMiddle, secondTarget)),
        "add second root-chain connector");
    passed &= check(problem.internMechanism(
                        eventInScope(4, 7, 8, childSource, childMiddle, child)),
                    "add child-chain member");
    passed &= check(problem.internMechanism(
                        eventInScope(5, 8, 9, childMiddle, childTarget, child)),
                    "add child-chain connector");
    CanonicalSyncDirectPairOptions options;
    if (coverageRowLimit) {
      options.pairCoverageLimits.maximumResultRows = 1;
    }
    const CanonicalSyncProblemResult generated =
        addCanonicalSyncDirectPairPatterns(problem, options);
    const std::size_t expectedEvaluations = coverageRowLimit ? 1 : 3;
    passed &=
        check(generated && generated.index == 0 &&
                  problem.wasPatternGenerationTruncated() &&
                  problem.getPatternStatistics().directPairProposals == 3 &&
                  problem.getPatternStatistics().directPairEvaluations ==
                      expectedEvaluations,
              "truncate one owner batch and evaluate the later scope");
    passed &=
        check(problem.freeze(), "freeze after atomic owner-pair truncation");
    passed &= check(problem.getPatternStatistics()
                            .get(CanonicalSyncPatternKind::DirectPair)
                            .patterns == 1,
                    "commit no partial rows from the oversized owner batch");
  };
  run(true);
  run(false);
  return passed;
}

bool testOwnerPairCoverageWordLimitIsAtomic() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed, "add word-limit source");
  const SyncCoverNodeId middle = takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}),
                                           passed, "add word-limit middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add word-limit target");
  const SyncCoverNodeId otherSource = takeIndex(
      graph.addNode(4, 1, 0, 3, {}, {5}), passed, "add zero-extra source");
  const SyncCoverNodeId otherMiddle = takeIndex(
      graph.addNode(5, 1, 0, 4, {}, {6}), passed, "add zero-extra middle");
  const SyncCoverNodeId otherTarget =
      takeIndex(graph.addNode(6, 1, 0, 5), passed, "add zero-extra target");
  passed &=
      check(graph.addDemand(demand(source, target)), "add word-limit demand");
  passed &= check(graph.freezeStructure(), "freeze word-limit graph");

  CanonicalSyncPatternProblem::Limits limits;
  limits.maximumCoverageWords = 0;
  CanonicalSyncPatternProblem problem(graph, allDemands(graph), limits);
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add word-limit first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add word-limit second domain");
  passed &= check(problem.addEventDomain({2, 1, 3, 8, {}}),
                  "add word-limit direct domain");
  passed &= check(problem.addEventDomain({3, 4, 5, 8, {}}),
                  "add zero-extra first domain");
  passed &= check(problem.addEventDomain({4, 5, 6, 8, {}}),
                  "add zero-extra second domain");
  passed &= check(problem.internMechanism(event(0, 1, 2, source, middle)),
                  "add word-limit first pair member");
  passed &= check(problem.internMechanism(event(1, 2, 3, middle, target)),
                  "add word-limit second pair member");
  passed &=
      check(problem.internMechanism(event(3, 4, 5, otherSource, otherMiddle)),
            "add zero-extra first pair member");
  passed &=
      check(problem.internMechanism(event(4, 5, 6, otherMiddle, otherTarget)),
            "add zero-extra second pair member");
  passed &= check(problem.internMechanism(event(2, 1, 3, source, target)),
                  "add word-limit covering singleton");

  SyncCoverDemandSet retainedJoint(1);
  retainedJoint.insert(0);
  SyncCoverDemandSet emptyJoint(1);
  const std::vector<SyncCoverDemandSet> singletonRows(
      problem.getMechanisms().size(), SyncCoverDemandSet(1));
  const CanonicalSyncProblemResult rejected =
      problem.addDirectPairBatch({{0, 1}}, {retainedJoint}, singletonRows);
  const CanonicalSyncProblemResult continued =
      problem.addDirectPairBatch({{2, 3}}, {emptyJoint}, singletonRows);
  passed &=
      check(rejected && rejected.index == 0 && continued &&
                continued.index == 0 && problem.wasPatternGenerationTruncated(),
            "discard a whole retained row at the coverage-word bound");
  passed &=
      check(problem.freeze(), "freeze after coverage-word batch truncation");
  passed &= check(problem.getPatternStatistics()
                          .get(CanonicalSyncPatternKind::DirectPair)
                          .patterns == 1,
                  "continue with a later zero-extra owner batch");
  return passed;
}

bool testSiblingAndBarrierPairsComposeAtTheirLca() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId left =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 7}),
                passed, "add left sibling scope");
  const SyncCoverScopeId right =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{8, 15}),
                passed, "add right sibling scope");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, left, 0, {}, {2}), passed, "add sibling source");
  const SyncCoverNodeId leftMiddle = takeIndex(
      graph.addNode(2, 1, left, 1), passed, "add sibling left middle");
  const SyncCoverNodeId rightMiddle =
      takeIndex(graph.addNode(2, 1, right, 4, {}, {3}), passed,
                "add sibling right middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, right, 5), passed, "add sibling target");
  passed &= check(
      graph.addEdge({leftMiddle, rightMiddle,
                     SyncCoverEdgeKind::CompletionPreservingIssueOrder, 0}),
      "connect sibling boundary terminals");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add sibling-owned parent demand");
  passed &= check(graph.freezeStructure(), "freeze sibling graph");

  CanonicalSyncPatternProblem siblingProblem(graph, allDemands(graph));
  passed &= check(siblingProblem.addEventDomain({0, 1, 2, 8, {}}),
                  "add sibling first event domain");
  passed &= check(siblingProblem.addEventDomain({1, 2, 3, 8, {}}),
                  "add sibling second event domain");
  CanonicalSyncMechanismDescriptor firstDescriptor =
      event(0, 1, 2, source, leftMiddle);
  firstDescriptor.supplies.front().edge.scope = left;
  CanonicalSyncMechanismDescriptor secondDescriptor =
      event(1, 2, 3, rightMiddle, target);
  secondDescriptor.supplies.front().edge.scope = right;
  passed &= check(siblingProblem.internMechanism(std::move(firstDescriptor)),
                  "add left sibling mechanism");
  passed &= check(siblingProblem.internMechanism(std::move(secondDescriptor)),
                  "add right sibling mechanism");
  const CanonicalSyncProblemResult siblingPairs =
      addCanonicalSyncDirectPairPatterns(siblingProblem);
  passed &= check(siblingPairs && siblingPairs.index == 1,
                  "retain a sibling pair at the parent LCA");

  SyncCoverGraph barrierGraph;
  const SyncCoverNodeId barrierSource =
      takeIndex(barrierGraph.addNode(1, 1, 0, 0, {}, {2}), passed,
                "add barrier-pair source");
  const SyncCoverNodeId barrierMiddle = takeIndex(
      barrierGraph.addNode(2, 1, 0, 1), passed, "add barrier-pair middle");
  const SyncCoverNodeId barrierTarget = takeIndex(
      barrierGraph.addNode(2, 1, 0, 2), passed, "add barrier-pair target");
  passed &= check(barrierGraph.addDemand(demand(barrierSource, barrierTarget)),
                  "add event-barrier pair demand");
  passed &= check(barrierGraph.freezeStructure(), "freeze event-barrier graph");
  CanonicalSyncPatternProblem barrierProblem(barrierGraph,
                                             allDemands(barrierGraph));
  passed &= check(barrierProblem.addEventDomain({0, 1, 2, 8, {}}),
                  "add event-barrier domain");
  passed &= check(barrierProblem.internMechanism(
                      event(0, 1, 2, barrierSource, barrierMiddle)),
                  "add event member");
  passed &= check(barrierProblem.internMechanism(
                      targetedBarrier(2, barrierMiddle, barrierTarget)),
                  "add targeted-barrier member");
  const CanonicalSyncProblemResult barrierPairs =
      addCanonicalSyncDirectPairPatterns(barrierProblem);
  return passed && check(barrierPairs && barrierPairs.index == 1,
                         "retain event plus targeted-barrier transitivity");
}

bool testNestedPairExtendsToParentDemand() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId inner =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add nested pair scope");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, inner, 0, {}, {2}), passed, "add nested pair source");
  const SyncCoverNodeId middle = takeIndex(
      graph.addNode(2, 1, inner, 1, {}, {3}), passed, "add nested pair middle");
  const SyncCoverNodeId innerTarget = takeIndex(
      graph.addNode(3, 1, inner, 2), passed, "add nested pair inner target");
  const SyncCoverNodeId outerTarget = takeIndex(
      graph.addNode(3, 1, 0, 3), passed, "add nested pair outer target");
  passed &= check(
      graph.addEdge({innerTarget, outerTarget,
                     SyncCoverEdgeKind::CompletionPreservingIssueOrder, 0}),
      "add nested-to-parent completion path");
  passed &= check(graph.addDemand(demand(source, outerTarget)),
                  "add parent-level pair demand");
  passed &= check(graph.freezeStructure(), "freeze nested pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add nested first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add nested second domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, middle)), passed,
                "add nested first event");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(event(1, 2, 3, middle, innerTarget)),
                passed, "add nested second event");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 1,
                  "extend one inner pair onto a parent demand");
  passed &= check(problem.freeze(), "freeze nested pair problem");
  const CanonicalSyncPattern &pair = problem.getPatterns().back();
  passed &= check(pair.kind == CanonicalSyncPatternKind::DirectPair &&
                      pair.coverage.contains(0) && pair.extraCoverageCount == 1,
                  "retain exact parent-level extra coverage");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return passed && check(selection && selection.mechanisms ==
                                          std::vector<CanonicalSyncMechanismId>{
                                              first, second},
                         "select the nested pair globally");
}

bool testDirectPairComposesAcrossRecurrenceArena() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add recurrence-pair loop");
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, 0, {}, {2}), passed,
                "add recurrence-pair producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, 1, {}, {1}), passed,
                "add recurrence-pair consumer");
  passed &= check(graph.addDemand(demand(producer, producer, loop, 1)),
                  "add recurrence-pair reuse demand");
  passed &= check(graph.freezeStructure(), "freeze recurrence-pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add recurrence-pair forward domain");
  passed &= check(problem.addEventDomain({1, 2, 1, 8, {}}),
                  "add recurrence-pair carried domain");
  const CanonicalSyncMechanismId forward =
      takeIndex(problem.internMechanism(event(0, 1, 2, producer, consumer)),
                passed, "add recurrence-pair forward event");
  CanonicalSyncMechanismDescriptor carriedDescriptor =
      protocol(1, 2, 1, consumer, producer, loop, 1, 1);
  const CanonicalSyncMechanismId carried = takeIndex(
      problem.internVerifiedProtocol(
          carriedDescriptor,
          [&](const CanonicalSyncMechanismDescriptor &candidate) {
            return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
                   candidate.supplies.size() == 1 &&
                   candidate.supplies.front().edge.distance == 1;
          }),
      passed, "add recurrence-pair carried event");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 1,
                  "retain recurrence-arena pair synergy");
  passed &= check(problem.freeze(), "freeze recurrence-pair problem");
  const CanonicalSyncPattern &pair = problem.getPatterns().back();
  passed &= check(pair.kind == CanonicalSyncPatternKind::DirectPair &&
                      pair.coverage.contains(0) && pair.extraCoverageCount == 1,
                  "derive reuse coverage from ready and carried events");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return passed && check(selection && selection.mechanisms ==
                                          std::vector<CanonicalSyncMechanismId>{
                                              forward, carried},
                         "select recurrence pair members globally");
}

bool testPipeAllRequiresRescueTier() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add rescue source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add rescue target");
  passed &= check(graph.addDemand(demand(source, target)), "add rescue demand");
  passed &= check(graph.freezeStructure(), "freeze rescue graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  const CanonicalSyncMechanismId rescue =
      takeIndex(problem.internMechanism(barrier(2, {1, 2}, source, target)),
                passed, "add PIPE_ALL rescue");
  passed &= check(problem.freeze(), "freeze rescue problem");

  CanonicalSyncGreedyOptions precise;
  precise.maximumTier = CanonicalSyncSelectionTier::Precise;
  passed &= check(selectCanonicalSyncPatterns(problem, precise).error ==
                      CanonicalSyncSelectionError::NoCoveringPattern,
                  "exclude PIPE_ALL from precise selection");
  CanonicalSyncGreedyOptions enabled;
  enabled.maximumTier = CanonicalSyncSelectionTier::PipeAllRescue;
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(problem, enabled);
  passed &=
      check(selection && selection.mechanisms ==
                             std::vector<CanonicalSyncMechanismId>{rescue},
            "admit PIPE_ALL only in the rescue tier");
  return passed;
}

bool testPackagingPatternHasNoExtraCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add package source one");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add package target one");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 2, {}, {2}), passed, "add package source two");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add package target two");
  const SyncCoverNodeId thirdSource = takeIndex(
      graph.addNode(1, 1, 0, 4, {}, {2}), passed, "add package source three");
  const SyncCoverNodeId thirdTarget =
      takeIndex(graph.addNode(2, 1, 0, 5), passed, "add package target three");
  const SyncCoverNodeId baselineSource =
      takeIndex(graph.addNode(3, 1, 0, 6, {}, {4}), passed,
                "add package baseline source");
  const SyncCoverNodeId baselineTarget = takeIndex(
      graph.addNode(4, 1, 0, 7), passed, "add package baseline target");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget)),
                  "add package demand one");
  passed &= check(graph.addDemand(demand(secondSource, secondTarget)),
                  "add package demand two");
  passed &= check(graph.addDemand(demand(thirdSource, thirdTarget)),
                  "add package demand three");
  passed &= check(graph.addEdge(supply(baselineSource, baselineTarget)),
                  "add package baseline edge");
  passed &= check(graph.addDemand(demand(baselineSource, baselineTarget)),
                  "add package baseline demand");
  passed &= check(graph.freezeStructure(), "freeze package graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 2, {}}), "add package domain");
  passed &= check(problem.addEventDomain({1, 3, 4, 1, {}}),
                  "add package baseline domain");
  const CanonicalSyncMechanismId first = takeIndex(
      problem.internMechanism(event(0, 1, 2, firstSource, firstTarget)), passed,
      "add package event one");
  const CanonicalSyncMechanismId second = takeIndex(
      problem.internMechanism(event(0, 1, 2, secondSource, secondTarget)),
      passed, "add package event two");
  const CanonicalSyncMechanismId third = takeIndex(
      problem.internMechanism(event(0, 1, 2, thirdSource, thirdTarget)), passed,
      "add package event three");
  const CanonicalSyncMechanismId baseline = takeIndex(
      problem.internMechanism(event(1, 3, 4, baselineSource, baselineTarget)),
      passed, "add event for fixed-covered package demand");
  passed &=
      check(problem.addPattern(
                {CanonicalSyncPatternKind::ScarcityFrontier, {first, second}}),
            "add package-only pattern");
  passed &=
      check(problem.addPattern(
                {CanonicalSyncPatternKind::ScarcityFrontier, {first, third}}),
            "add overlapping package-only pattern");
  passed &= check(problem.freeze(), "freeze package-only problem");
  const CanonicalSyncPatternKindStatistics &statistics =
      problem.getPatternStatistics().get(
          CanonicalSyncPatternKind::ScarcityFrontier);
  passed &= check(!problem.getPatterns()[baseline].coverage.contains(3),
                  "remove fixed coverage from singleton mechanism rows");
  passed &=
      check(problem.getPatterns().size() == problem.getMechanisms().size(),
            "drop package-only patterns from the selectable catalog");
  passed &= check(statistics.patterns == 2 &&
                      statistics.jointCoverageIncidences == 4 &&
                      statistics.singletonCoverageIncidences == 4 &&
                      statistics.extraCoverageIncidences == 0 &&
                      statistics.patternsWithExtraCoverage == 0,
                  "count overlapping package coverage as incidences");
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
  passed &= check(
      selection.error == CanonicalSyncSelectionError::ResourceInfeasible &&
          selection.mechanisms ==
              std::vector<CanonicalSyncMechanismId>{firstEvent, secondEvent} &&
          selection.allocation.domains[0].required == 2 &&
          selection.allocation.domains[0].available == 1 &&
          selection.allocation.domains[0].liveMechanisms ==
              std::vector<CanonicalSyncMechanismId>{firstEvent, secondEvent},
      "normal cover reports its exact post-selection event-pressure core");
  CanonicalSyncGreedyOptions fallbackOptions;
  fallbackOptions.maximumTier = CanonicalSyncSelectionTier::PipeAllRescue;
  const CanonicalSyncSelection repaired =
      selectCanonicalSyncPatterns(problem, fallbackOptions);
  passed &= check(repaired && repaired.mechanisms ==
                                  std::vector<CanonicalSyncMechanismId>{
                                      firstEvent, fallback},
                  "an explicitly enabled fallback repairs event pressure");
  return passed;
}

bool testOptionalPipelineScarcityFallsBack() {
  bool passed = true;
  SyncCoverGraph graph;
  std::vector<SyncCoverNodeId> sources;
  std::vector<SyncCoverNodeId> targets;
  for (std::size_t index = 0; index < 9; ++index) {
    sources.push_back(takeIndex(graph.addNode(1, 1, 0, index, {}, {2}), passed,
                                "add pipeline scarcity source"));
  }
  for (std::size_t index = 0; index < 9; ++index) {
    targets.push_back(takeIndex(graph.addNode(2, 1, 0, 9 + index), passed,
                                "add pipeline scarcity target"));
    passed &= check(graph.addDemand(demand(sources[index], targets[index])),
                    "add pipeline scarcity demand");
  }
  passed &= check(graph.freezeStructure(), "freeze pipeline scarcity graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add pipeline scarcity domain");
  std::vector<CanonicalSyncMechanismId> events;
  for (std::size_t index = 0; index < 9; ++index) {
    events.push_back(takeIndex(
        problem.internMechanism(event(0, 1, 2, sources[index], targets[index])),
        passed, "add pipeline scarcity event"));
    passed &= check(problem.internMechanism(
                        barrier(2, {1, 2}, sources[index], targets[index])),
                    "add pipeline scarcity fallback");
  }
  const CanonicalSyncProblemResult optional = addCanonicalSyncFeasiblePattern(
      problem, {CanonicalSyncPatternKind::ScarcityFrontier, events});
  passed &= check(optional && !optional.index,
                  "drop a coverage-free optional pipeline independently of "
                  "event coloring");
  CanonicalSyncPatternProblem::Limits memberLimits;
  memberLimits.maximumMembersPerPattern = 8;
  CanonicalSyncPatternProblem memberLimited(graph, allDemands(graph),
                                            memberLimits);
  passed &= check(memberLimited.addEventDomain({0, 1, 2, 8, {}}),
                  "add member-limited domain");
  std::vector<CanonicalSyncMechanismId> limitedEvents;
  for (std::size_t index = 0; index < 9; ++index) {
    limitedEvents.push_back(
        takeIndex(memberLimited.internMechanism(
                      event(0, 1, 2, sources[index], targets[index])),
                  passed, "add member-limited event"));
  }
  const CanonicalSyncProblemResult memberCapped =
      addCanonicalSyncFeasiblePattern(
          memberLimited,
          {CanonicalSyncPatternKind::ScarcityFrontier, limitedEvents});
  passed &= check(memberCapped && !memberCapped.index,
                  "skip an oversized optional pipeline");
  passed &= check(problem.freeze(), "freeze pipeline fallback problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection.error ==
                          CanonicalSyncSelectionError::ResourceInfeasible &&
                      selection.allocation.domains[0].required == 9 &&
                      selection.allocation.domains[0].available == 8,
                  "post-cover allocation diagnoses the oversized event family");
  CanonicalSyncGreedyOptions enabled;
  enabled.maximumTier = CanonicalSyncSelectionTier::PipeAllRescue;
  const CanonicalSyncSelection repaired =
      selectCanonicalSyncPatterns(problem, enabled);
  passed &= check(repaired && repaired.allocation.feasible,
                  "fallback-enabled re-cover is event feasible");
  passed &=
      check(static_cast<bool>(verifyCanonicalSyncSelection(problem, repaired)),
            "fresh finalization accepts the repaired scarce pipeline");
  return passed;
}

bool testReservationsAndFinalValidation() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId inactiveSource = takeIndex(
      graph.addNode(3, 1, 0, 0), passed, "add inactive reserve source");
  const SyncCoverNodeId inactiveTarget = takeIndex(
      graph.addNode(4, 1, 0, 1), passed, "add inactive reserve target");
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, 0, 2, {}, {2}),
                                           passed, "add reserve source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add reserve target");
  passed &= check(graph.addDemand(demand(inactiveSource, inactiveTarget)),
                  "add inactive reserve demand");
  passed &=
      check(graph.addDemand(demand(source, target)), "add reserve demand");
  passed &= check(graph.freezeStructure(), "freeze reserve graph");
  CanonicalSyncPatternProblem problem(graph, {1});
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
  CanonicalSyncSelection incomplete = selection;
  incomplete.mechanisms.clear();
  const CanonicalSyncVerifiedPlan rejected =
      verifyCanonicalSyncSelection(problem, incomplete);
  passed &= check(
      rejected.error == CanonicalSyncSelectionError::FinalValidationFailed &&
          rejected.firstUncoveredDemand == 1,
      "finalization rejects missing selected IDs with graph demand");
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

  CanonicalSyncMechanismDescriptor undrainedExport =
      protocol(0, 1, 2, source, target, loop, 1, 1);
  undrainedExport.supplies[0].completionExport =
      CanonicalSyncSupplyExport::ScopeExitAfterDrain;
  passed &=
      check(problem.internVerifiedProtocol(undrainedExport, verifier).error ==
                CanonicalSyncProblemError::InvalidMechanism,
            "undrained verified protocol cannot certify scope exit");

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

bool testHierarchicalProtocolLifetime() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 63}, true),
                passed, "add hierarchical outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{8, 47}, true),
      passed, "add hierarchical inner loop");
  const SyncCoverScopeId sibling = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{48, 55}, true),
      passed, "add hierarchical sibling loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, inner, 9, {}, {2}), passed,
                "add hierarchical source");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, inner, 10),
                                           passed, "add hierarchical target");
  passed &= check(graph.addDemand(demand(source, target, inner, 1)),
                  "add hierarchical demand");
  passed &= check(graph.freezeStructure(), "freeze hierarchical graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 2, {}}),
                  "add hierarchical domain");

  CanonicalSyncMechanismDescriptor descriptor =
      protocol(0, 1, 2, source, target, inner, 1, 1);
  descriptor.eventUses[0].lifetimeScope = outer;
  const auto verifier = [=](const CanonicalSyncMechanismDescriptor &candidate) {
    return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
           candidate.eventUses.size() == 1 &&
           candidate.eventUses[0].recurrenceScope == inner &&
           candidate.eventUses[0].lifetimeScope == outer;
  };
  const CanonicalSyncProblemResult admitted =
      problem.internVerifiedProtocol(descriptor, verifier);
  passed &= check(admitted && admitted.index,
                  "admit a verified wider hierarchical lifetime");
  if (admitted.index) {
    const CanonicalSyncMechanism &mechanism =
        problem.getMechanisms()[*admitted.index];
    passed &= check(mechanism.eventLifetimes.size() == 1 &&
                        mechanism.eventLifetimes[0].begin == 0 &&
                        mechanism.eventLifetimes[0].end == 63,
                    "color the event over the complete outer loop");
  }

  CanonicalSyncMechanismDescriptor unrelated = descriptor;
  unrelated.eventUses[0].lifetimeScope = sibling;
  passed &= check(problem.internVerifiedProtocol(unrelated, verifier).error ==
                      CanonicalSyncProblemError::InvalidMechanism,
                  "reject a lifetime outside the recurrence ancestry");
  CanonicalSyncMechanismDescriptor nonLoop = descriptor;
  nonLoop.eventUses[0].lifetimeScope = 0;
  passed &= check(problem.internVerifiedProtocol(nonLoop, verifier).error ==
                      CanonicalSyncProblemError::InvalidMechanism,
                  "reject a non-loop lifetime scope");
  return passed;
}

bool testFreezeRetryCommitsFreshDerivedState() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add retry first source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add retry first target");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(3, 1, 0, 2, {}, {4}), passed, "add retry second source");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(4, 1, 0, 3), passed, "add retry second target");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget)),
                  "add retry first demand");
  passed &= check(graph.addDemand(demand(secondSource, secondTarget)),
                  "add retry second demand");
  passed &= check(graph.freezeStructure(), "freeze retry graph");

  CanonicalSyncPatternProblem::Limits limits;
  limits.maximumIncidences = 2;
  CanonicalSyncPatternProblem problem(graph, allDemands(graph), limits);
  passed &=
      check(problem.addEventDomain({0, 1, 2, 8, {}}), "add retry first domain");
  passed &= check(problem.addEventDomain({1, 3, 4, 8, {}}),
                  "add retry second domain");
  passed &=
      check(problem.internMechanism(event(0, 1, 2, firstSource, firstTarget)),
            "add retry first mechanism");
  passed &= check(problem.freeze().error ==
                      CanonicalSyncProblemError::UncoverableDemand,
                  "first freeze reports the uncovered demand");
  passed &= check(!problem.isFrozen(), "failed freeze remains mutable");
  passed &=
      check(problem.internMechanism(event(1, 3, 4, secondSource, secondTarget)),
            "add retry covering mechanism");
  passed &=
      check(problem.freeze(), "retry freeze at the exact incidence limit");
  passed &= check(problem.getDemandPatterns().size() == 2 &&
                      problem.getDemandPatterns()[0].size() == 1 &&
                      problem.getDemandPatterns()[1].size() == 1,
                  "retry commits only freshly prepared incidences");
  return passed;
}

bool testOptionalPairReservesSingletonIncidences() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed,
                "add incidence-reservation source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed,
                "add incidence-reservation middle");
  const SyncCoverNodeId target = takeIndex(graph.addNode(3, 1, 0, 2), passed,
                                           "add incidence-reservation target");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add incidence-reservation demand");
  passed &=
      check(graph.freezeStructure(), "freeze incidence-reservation graph");

  CanonicalSyncPatternProblem::Limits limits;
  limits.maximumIncidences = 1;
  CanonicalSyncPatternProblem problem(graph, allDemands(graph), limits);
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add incidence-reservation first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add incidence-reservation second domain");
  passed &= check(problem.addEventDomain({2, 1, 3, 8, {}}),
                  "add incidence-reservation direct domain");
  passed &= check(problem.internMechanism(event(0, 1, 2, source, middle)),
                  "add incidence-reservation first pair member");
  passed &= check(problem.internMechanism(event(1, 2, 3, middle, target)),
                  "add incidence-reservation second pair member");
  passed &= check(problem.internMechanism(event(2, 1, 3, source, target)),
                  "add incidence-reservation covering singleton");

  SyncCoverDemandSet jointCoverage(1);
  jointCoverage.insert(0);
  std::vector<SyncCoverDemandSet> singletonCoverage(
      problem.getMechanisms().size(), SyncCoverDemandSet(1));
  singletonCoverage[2].insert(0);
  const CanonicalSyncProblemResult generated =
      problem.addDirectPairBatch({{0, 1}}, {jointCoverage}, singletonCoverage);
  passed &= check(generated && generated.index == 0 &&
                      problem.wasPatternGenerationTruncated(),
                  "truncate the optional pair after reserving singleton rows");
  passed &= check(problem.freeze(),
                  "singleton-valid problem freezes at its incidence limit");
  passed &= check(problem.getDemandPatterns().size() == 1 &&
                      problem.getDemandPatterns()[0].size() == 1,
                  "optional pair does not consume singleton incidence supply");
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
  passed &=
      check(patternLimited
                    .addPattern({CanonicalSyncPatternKind::ScarcityFrontier,
                                 {first, second}})
                    .error == CanonicalSyncProblemError::None,
            "zero-extra patterns do not consume retained capacity");
  passed &= check(patternLimited.freeze(),
                  "freeze a problem after dropping its optional pattern");
  passed &= check(patternLimited.getPatterns().size() == 2,
                  "retain only singleton patterns at the aggregate limit");

  CanonicalSyncPatternProblem::Limits proposalLimits;
  proposalLimits.maximumPatternProposals = 0;
  CanonicalSyncPatternProblem proposalLimited(graph, allDemands(graph),
                                              proposalLimits);
  passed &= check(proposalLimited.addEventDomain({0, 1, 2, 8, {}}),
                  "add proposal-limit domain");
  const CanonicalSyncMechanismId proposalFirst =
      takeIndex(proposalLimited.internMechanism(event(0, 1, 2, source, target)),
                passed, "add proposal-limit event");
  const CanonicalSyncMechanismId proposalSecond = takeIndex(
      proposalLimited.internMechanism(barrier(2, {1, 2}, source, target)),
      passed, "add proposal-limit barrier");
  const CanonicalSyncProblemResult proposal = addCanonicalSyncFeasiblePattern(
      proposalLimited, {CanonicalSyncPatternKind::ScarcityFrontier,
                        {proposalFirst, proposalSecond}});
  passed &= check(proposal && !proposal.index &&
                      proposalLimited.wasPatternGenerationTruncated(),
                  "report bounded optional-pattern generation");
  passed &= check(proposalLimited.freeze(),
                  "freeze a proposal-limited fallback problem");
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
      testDirectPairDiscoversJointCoverage() &&
      testPairPreparationLimitKeepsSingletonCorrectness() &&
      testPairOwnerUsesEverySupplyScope() &&
      testOwnerPairBatchesTruncateAtomicallyAndContinue() &&
      testOwnerPairCoverageWordLimitIsAtomic() &&
      testSiblingAndBarrierPairsComposeAtTheirLca() &&
      testNestedPairExtendsToParentDemand() &&
      testDirectPairComposesAcrossRecurrenceArena() &&
      testPipeAllRequiresRescueTier() &&
      testPackagingPatternHasNoExtraCoverage() &&
      testScarcityUsesBarrierFallback() &&
      testOptionalPipelineScarcityFallsBack() &&
      testReservationsAndFinalValidation() &&
      testAllocatorWidthsReuseAndConflicts() &&
      testVerifiedProtocolTrustBoundary() &&
      testHierarchicalProtocolLifetime() &&
      testFreezeRetryCommitsFreshDerivedState() &&
      testOptionalPairReservesSingletonIncidences() &&
      testFailClosedConstruction();
  return passed ? 0 : 1;
}
