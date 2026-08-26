// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverSolverTest failure: " << message << '\n';
  }
  return condition;
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

std::size_t takeIndex(const SyncCoverMechanismResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
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
  result.kind = SyncCoverEdgeKind::CompletionSupply;
  result.scope = scope;
  result.distance = distance;
  return result;
}

std::size_t appendEventUse(SyncCoverMechanismDescriptor &descriptor,
                           SyncCoverResourceDomainId domain,
                           std::uint32_t sourceResource,
                           std::uint32_t targetResource, SyncCoverNodeId source,
                           SyncCoverNodeId target, SyncCoverScopeId scope = 0,
                           unsigned distance = 0) {
  const std::size_t edge = descriptor.supplyEdges.size();
  const std::size_t produce = descriptor.actions.size();
  descriptor.actions.push_back({SyncCoverResourceActionKind::Produce,
                                sourceResource,
                                {SyncCoverAnchorKind::AfterNode, source, 0}});
  const std::size_t consume = descriptor.actions.size();
  descriptor.actions.push_back({SyncCoverResourceActionKind::Consume,
                                targetResource,
                                {SyncCoverAnchorKind::BeforeNode, target, 0}});
  const std::size_t use = descriptor.resourceUses.size();
  descriptor.resourceUses.push_back(
      {domain, scope, distance, 1, {produce, consume}, {edge}});
  descriptor.supplyEdges.push_back(supply(source, target, scope, distance));
  descriptor.supplyBindings.push_back({edge, use, produce, consume});
  return use;
}

SyncCoverMechanismId
addEvent(SyncCoverMechanismUniverse &universe, SyncCoverResourceDomainId domain,
         std::uint32_t sourceResource, std::uint32_t targetResource,
         SyncCoverNodeId source, SyncCoverNodeId target, bool &passed) {
  SyncCoverMechanismDescriptor descriptor;
  appendEventUse(descriptor, domain, sourceResource, targetResource, source,
                 target);
  return takeIndex(universe.addMechanism(descriptor), passed,
                   "add event mechanism");
}

std::optional<std::vector<SyncCoverMechanismId>>
findBruteForceOptimum(const SyncCoverMechanismUniverse &universe) {
  const std::size_t mechanismCount = universe.getMechanisms().size();
  if (mechanismCount >= 20) {
    return std::nullopt;
  }
  SyncCoverCoverageOracle coverage(universe.getGraph());
  const SyncCoverSelectionEvaluator evaluator(universe);
  std::optional<std::vector<SyncCoverMechanismId>> best;
  std::optional<SyncCoverStructuralCost> bestCost;
  const std::size_t subsetCount = std::size_t{1} << mechanismCount;
  for (std::size_t mask = 0; mask < subsetCount; ++mask) {
    std::vector<SyncCoverMechanismId> selected;
    for (std::size_t mechanism = 0; mechanism < mechanismCount; ++mechanism) {
      const bool selectedByMask = (mask & (std::size_t{1} << mechanism)) != 0;
      if (selectedByMask) {
        selected.push_back(mechanism);
      }
    }
    const SyncCoverSelectionEvaluation evaluation =
        evaluator.evaluate(selected);
    if (!evaluation || !evaluation.resources.resourceFeasible) {
      continue;
    }
    const std::vector<SyncCoverCoverageResult> demands =
        coverage.checkAll(selected);
    const bool covered =
        std::all_of(demands.begin(), demands.end(), [](const auto &item) {
          return static_cast<bool>(item) && item.covered;
        });
    if (covered && (!bestCost ||
                    syncCoverStructuralCostLess(evaluation.cost, *bestCost))) {
      best = selected;
      bestCost = evaluation.cost;
    }
  }
  return best;
}

bool testCutGuidedExactSelection() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed, "add first");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed, "add middle");
  const SyncCoverNodeId last =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add last");
  const SyncCoverDemandId demandId =
      takeIndex(graph.addDemand(demand(first, last)), passed, "add demand");
  SyncCoverMechanismUniverse universe(graph);
  const auto firstDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add first domain");
  const auto secondDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 3, 8),
      passed, "add second domain");
  const auto directDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 3, 8),
      passed, "add direct domain");
  const SyncCoverMechanismId firstHalf =
      addEvent(universe, firstDomain, 1, 2, first, middle, passed);
  const SyncCoverMechanismId secondHalf =
      addEvent(universe, secondDomain, 2, 3, middle, last, passed);
  const SyncCoverMechanismId direct =
      addEvent(universe, directDomain, 1, 3, first, last, passed);

  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe, {demandId});
  passed &= check(result && result.mechanisms == std::vector{direct},
                  "exact search prefers the cheaper direct cover");
  passed &= check(findBruteForceOptimum(universe) == result.mechanisms,
                  "exact solver matches exhaustive subset enumeration");
  const bool hasDirectAllocation =
      result.resources && result.resources.domains.size() > directDomain;
  passed &= check(hasDirectAllocation,
                  "final verification returns resource domains");
  if (hasDirectAllocation) {
    const SyncCoverDomainFeasibility &directAllocation =
        result.resources.domains[directDomain];
    passed &= check(directAllocation.allocations.size() == 1 &&
                        directAllocation.allocations.front().owner.mechanism ==
                            direct &&
                        directAllocation.allocations.front().ids ==
                            std::vector<unsigned>{0},
                    "final verification preserves the exact physical ID");
  }
  passed &= check(result.coverageStatistics.graphValidations == 1 &&
                      result.coverageStatistics.demandPreparations == 1 &&
                      result.coverageStatistics.groundingQueries == 1 &&
                      result.coverageStatistics.coverageQueries == 0 &&
                      result.finalVerificationStatistics.graphValidations ==
                          1 &&
                      result.finalVerificationStatistics.demandPreparations ==
                          1 &&
                      result.finalVerificationStatistics.coverageQueries == 2 &&
                      result.oracleRedundancyChecks == 1,
                  "solver batches incidence once and shares one independent "
                  "post-search oracle between the oracle-checked redundancy "
                  "pass and final verification");

  SyncCoverSolverOptions greedyOptions;
  greedyOptions.exactMechanismThreshold = 0;
  const SyncCoverSelectionResult greedyResult =
      solveSyncCoverSelection(universe, {demandId}, greedyOptions);
  passed &= check(greedyResult && !greedyResult.optimalityProven &&
                      !greedyResult.truncation &&
                      findBruteForceOptimum(universe) ==
                          greedyResult.mechanisms,
                  "grounded greedy search matches exhaustive enumeration on "
                  "the "
                  "representative component");

  SyncCoverGraph chainGraph;
  const SyncCoverNodeId chainSource = takeIndex(
      chainGraph.addNode(1, 1, 0, 0, {}, {2}), passed, "add chain source");
  const SyncCoverNodeId chainMiddle = takeIndex(
      chainGraph.addNode(2, 1, 0, 1, {}, {3}), passed, "add chain middle");
  const SyncCoverNodeId chainTarget =
      takeIndex(chainGraph.addNode(3, 1, 0, 2), passed, "add chain target");
  const SyncCoverDemandId chainDemand =
      takeIndex(chainGraph.addDemand(demand(chainSource, chainTarget)), passed,
                "add chain demand");
  SyncCoverMechanismUniverse chainUniverse(chainGraph);
  const auto chainFirstDomain = takeIndex(
      chainUniverse.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add chain first domain");
  const auto chainSecondDomain = takeIndex(
      chainUniverse.addResourceDomain(SyncCoverResourceKind::EventId, 2, 3, 8),
      passed, "add chain second domain");
  const SyncCoverMechanismId chainFirst = addEvent(
      chainUniverse, chainFirstDomain, 1, 2, chainSource, chainMiddle, passed);
  const SyncCoverMechanismId chainSecond = addEvent(
      chainUniverse, chainSecondDomain, 2, 3, chainMiddle, chainTarget, passed);
  const SyncCoverSelectionResult chain =
      solveSyncCoverSelection(chainUniverse, {chainDemand});
  passed &= check(
      chain && chain.mechanisms ==
                   std::vector<SyncCoverMechanismId>{chainFirst,
                                                     chainSecond} &&
          chain.missingFactoryDemands.empty() &&
          chain.coverageStatistics.groundingQueries == 2 &&
          chain.coverageStatistics.coverageQueries == 0,
      "depth-two pricing grounds a missing composite path before search");
  passed &= check(chainFirst == 0 && chainSecond == 1,
                  "chain mechanism identities remain deterministic");
  passed &= check(firstHalf == 0 && secondHalf == 1,
                  "mechanism identities remain deterministic");
  return passed;
}

bool testAtomicProtocolAndScarcity() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first source");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 4), passed, "add first target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 5), passed, "add second target");
  const SyncCoverDemandId firstDemand =
      takeIndex(graph.addDemand(demand(firstSource, firstTarget)), passed,
                "add first demand");
  const SyncCoverDemandId secondDemand =
      takeIndex(graph.addDemand(demand(secondSource, secondTarget)), passed,
                "add second demand");
  SyncCoverMechanismUniverse universe(graph);
  const auto domain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 1),
      passed, "add scarce domain");
  const SyncCoverMechanismId first =
      addEvent(universe, domain, 1, 2, firstSource, firstTarget, passed);
  const SyncCoverMechanismId second =
      addEvent(universe, domain, 1, 2, secondSource, secondTarget, passed);

  SyncCoverMechanismDescriptor protocol;
  protocol.kind = SyncCoverMechanismKind::VerifiedProtocol;
  appendEventUse(protocol, domain, 1, 2, firstSource, firstTarget);
  protocol.supplyEdges.push_back(supply(secondSource, secondTarget));
  protocol.resourceUses.front().supplyEdges.push_back(1);
  protocol.supplyBindings.push_back({1, 0, 0, 1});
  const SyncCoverMechanismId protocolId = takeIndex(
      universe.addVerifiedProtocol(protocol, [](const auto &) { return true; }),
      passed, "add atomic shared protocol");

  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe, {firstDemand, secondDemand});
  passed &= check(result && result.mechanisms == std::vector{protocolId},
                  "atomic protocol replaces an over-budget event pair");
  passed &= check(
      !universe.evaluateResourceSelection({first, second}).resourceFeasible,
      "independent events exceed the shared event budget");
  return passed;
}

bool testComponentsConflictsAndFailure() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add source a");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add target a");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(1, 1, 0, 3, {}, {2}), passed, "add source b");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 4), passed, "add target b");
  takeIndex(graph.addDemand(demand(firstSource, firstTarget)), passed,
            "add demand a");
  takeIndex(graph.addDemand(demand(secondSource, secondTarget)), passed,
            "add demand b");
  SyncCoverMechanismUniverse universe(graph);
  const auto firstDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add domain a");
  const SyncCoverMechanismId first =
      addEvent(universe, firstDomain, 1, 2, firstSource, firstTarget, passed);
  const SyncCoverMechanismId second =
      addEvent(universe, firstDomain, 1, 2, secondSource, secondTarget, passed);
  const SyncCoverSelectionResult independent =
      solveSyncCoverSelection(universe, {0, 1});
  passed &= check(independent && independent.components.size() == 2,
                  "disjoint lifetimes in one domain form separate components");
  passed &=
      check(findBruteForceOptimum(universe) == independent.mechanisms,
            "independent component composition matches global enumeration");

  passed &= check(static_cast<bool>(universe.addConflict(first, second)),
                  "add conflict");
  const SyncCoverSelectionResult conflicting =
      solveSyncCoverSelection(universe, {0, 1});
  passed &= check(!conflicting &&
                      conflicting.error ==
                          SyncCoverSelectionError::SearchIncomplete &&
                      conflicting.components.size() == 1,
                  "conflicts join components and fail closed without "
                  "overclaiming infeasibility");
  return passed;
}

bool testBarrierSelection() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add barrier source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(1, 1, 0, 1), passed, "add barrier target");
  const SyncCoverDemandId demandId = takeIndex(
      graph.addDemand(demand(source, target)), passed, "add barrier demand");
  SyncCoverMechanismUniverse universe(graph);
  SyncCoverMechanismDescriptor descriptor;
  descriptor.kind = SyncCoverMechanismKind::Barrier;
  descriptor.barrier = SyncCoverBarrierPlacement{1, target, 0};
  descriptor.supplyEdges.push_back(supply(source, target));
  const SyncCoverMechanismId barrier = takeIndex(
      universe.addMechanism(descriptor), passed, "add barrier candidate");
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe, {demandId});
  passed &=
      check(result && result.mechanisms == std::vector{barrier} &&
                result.cost.barrierActionProfile == std::vector<std::size_t>{1},
            "direct covering selects a same-resource barrier");
  SyncCoverSolverOptions beamOptions;
  beamOptions.exactMechanismThreshold = 0;
  const SyncCoverSelectionResult beam =
      solveSyncCoverSelection(universe, {demandId}, beamOptions);
  passed &= check(beam && beam.mechanisms == std::vector{barrier},
                  "bounded covering also selects a same-resource barrier");
  return passed;
}

bool testTruncationRescuedByBarrierFallback() {
  bool passed = true;
  SyncCoverGraph graph;
  // Interleaved orders make the two event lifetimes overlap in one domain,
  // fusing both demands into a single component whose greedy search needs
  // two rounds; a one-evaluation budget then truncates after the first.
  const SyncCoverNodeId firstSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first rescue source");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second rescue source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add first rescue target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, 0, 3), passed, "add second rescue target");
  const SyncCoverDemandId firstDemand =
      takeIndex(graph.addDemand(demand(firstSource, firstTarget)), passed,
                "add first rescue demand");
  const SyncCoverDemandId secondDemand =
      takeIndex(graph.addDemand(demand(secondSource, secondTarget)), passed,
                "add second rescue demand");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add rescue domain");
  addEvent(universe, domain, 1, 2, firstSource, firstTarget, passed);
  addEvent(universe, domain, 1, 2, secondSource, secondTarget, passed);
  SyncCoverMechanismDescriptor descriptor;
  descriptor.kind = SyncCoverMechanismKind::Barrier;
  SyncCoverBarrierPlacement placement{1, firstTarget, 0};
  placement.drainsAllResources = true;
  descriptor.barrier = placement;
  descriptor.supplyEdges.push_back(supply(firstSource, firstTarget));
  descriptor.supplyEdges.push_back(supply(secondSource, secondTarget));
  const SyncCoverMechanismId barrier = takeIndex(
      universe.addMechanism(descriptor), passed, "add rescue barrier");

  SyncCoverSolverOptions truncated;
  truncated.exactMechanismThreshold = 0;
  truncated.evaluationLimit = 1;
  const SyncCoverSelectionResult rescued = solveSyncCoverSelection(
      universe, {firstDemand, secondDemand}, truncated);
  passed &= check(static_cast<bool>(rescued),
                  "budget-one truncation is rescued, not failed");
  passed &= check(rescued.mechanisms == std::vector{barrier},
                  "the rescue selects the all-barrier column");
  passed &= check(rescued.rescuedComponents == 1 &&
                      rescued.truncation.evaluationLimit &&
                      !rescued.optimalityProven,
                  "the rescue is reported and demotes optimality");
  return passed;
}

bool testDeclaredCoverageSelection() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first source");
  const SyncCoverNodeId secondSource = takeIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second source");
  const SyncCoverNodeId firstBridge = takeIndex(
      graph.addNode(2, 1, 0, 2, {}, {1}), passed, "add first bridge");
  const SyncCoverNodeId secondBridge = takeIndex(
      graph.addNode(2, 1, 0, 3, {}, {1}), passed, "add second bridge");
  const SyncCoverNodeId firstTarget = takeIndex(
      graph.addNode(1, 1, 0, 4), passed, "add first target");
  const SyncCoverNodeId secondTarget = takeIndex(
      graph.addNode(1, 1, 0, 5), passed, "add second target");
  const SyncCoverDemandId firstDemand = takeIndex(
      graph.addDemand(demand(firstSource, firstTarget)), passed,
      "add first same-pipe demand");
  const SyncCoverDemandId secondDemand = takeIndex(
      graph.addDemand(demand(secondSource, secondTarget)), passed,
      "add second same-pipe demand");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId forwardDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add forward domain");
  const SyncCoverResourceDomainId reverseDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 1, 8),
      passed, "add reverse domain");
  const SyncCoverMechanismId firstForward = addEvent(
      universe, forwardDomain, 1, 2, firstSource, firstBridge, passed);
  const SyncCoverMechanismId firstReverse = addEvent(
      universe, reverseDomain, 2, 1, firstBridge, firstTarget, passed);
  const SyncCoverMechanismId secondForward = addEvent(
      universe, forwardDomain, 1, 2, secondSource, secondBridge, passed);
  const SyncCoverMechanismId secondReverse = addEvent(
      universe, reverseDomain, 2, 1, secondBridge, secondTarget, passed);

  SyncCoverMechanismDescriptor barrierDescriptor;
  barrierDescriptor.kind = SyncCoverMechanismKind::Barrier;
  barrierDescriptor.barrier = SyncCoverBarrierPlacement{1, firstTarget, 0};
  barrierDescriptor.supplyEdges = {
      supply(firstSource, firstTarget), supply(secondSource, secondTarget)};
  const SyncCoverMechanismId barrier = takeIndex(
      universe.addMechanism(barrierDescriptor), passed,
      "add broad barrier candidate");

  SyncCoverSolverOptions options;
  options.exactMechanismThreshold = 0;
  const SyncCoverSelectionResult result = solveSyncCoverSelection(
      universe, {firstDemand, secondDemand}, options);
  passed &= check(result &&
                      result.mechanisms ==
                          std::vector<SyncCoverMechanismId>{
                              firstForward, firstReverse, secondForward,
                              secondReverse},
                  "bounded selection consumes priced event-path columns");
  passed &= check(result &&
                      result.cost.barrierActionProfile ==
                          std::vector<std::size_t>{0},
                  "barrier-first cost prefers fine-grained round trips");
  passed &= check(firstForward == 0 && firstReverse == 1 &&
                      secondForward == 2 && secondReverse == 3,
                  "event candidates retain deterministic identities");
  passed &= check(barrier == 4, "broad barrier remains a deterministic option");
  return passed;
}

bool testBeamSeedsAndRedundancy() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add target");
  takeIndex(graph.addDemand(demand(source, target)), passed, "add demand");
  SyncCoverMechanismUniverse universe(graph);
  const auto domain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add domain");
  const SyncCoverMechanismId first =
      addEvent(universe, domain, 1, 2, source, target, passed);
  addEvent(universe, domain, 1, 2, source, target, passed);
  SyncCoverSolverOptions options;
  options.exactMechanismThreshold = 0;
  options.evaluationLimit = 1;
  const SyncCoverSelectionResult firstRun =
      solveSyncCoverSelection(universe, {0}, options);
  const SyncCoverSelectionResult secondRun =
      solveSyncCoverSelection(universe, {0}, options);
  passed &= check(firstRun && firstRun.mechanisms == std::vector{first},
                  "bounded greedy covers within a one-evaluation limit");
  passed &= check(!firstRun.truncation.evaluationLimit &&
                      !firstRun.optimalityProven,
                  "large-component greedy selection does not claim "
                  "optimality");
  passed &= check(firstRun.mechanisms == secondRun.mechanisms &&
                      firstRun.evaluations == secondRun.evaluations,
                  "bounded greedy selection is deterministic");
  return passed;
}

bool testRecurrenceAndInputValidation() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, false, SyncCoverTimelineInterval{0, 5}, true),
                passed, "add loop");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, loop, 1), passed, "add loop target");
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, loop, 2, {}, {2}), passed, "add loop source");
  takeIndex(graph.addDemand(demand(source, target, loop, 1)), passed,
            "add recurrence demand");
  SyncCoverMechanismUniverse universe(graph);
  const auto domain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add recurrence domain");
  const auto protocol = makeSyncCoverUnitRecurrenceEvent(
      universe.getResourceDomains()[domain], source, target, loop);
  passed &= check(protocol.has_value(), "build unit recurrence protocol");
  const SyncCoverMechanismId protocolId = takeIndex(
      universe.addVerifiedProtocol(
          protocol.value_or(SyncCoverMechanismDescriptor{}),
          [&](const auto &candidate) {
            return verifySyncCoverUnitRecurrenceEvent(universe, candidate);
          }),
      passed, "add verified recurrence protocol");
  passed &= check(solveSyncCoverSelection(universe, {0}).mechanisms ==
                      std::vector{protocolId},
                  "verified recurrence protocol participates in covering");
  passed &= check(solveSyncCoverSelection(universe, {}).mechanisms.empty(),
                  "an empty active demand set needs no synchronization");
  passed &= check(solveSyncCoverSelection(universe, {1}).error ==
                      SyncCoverSelectionError::InvalidDemand,
                  "unknown active demands are rejected");
  SyncCoverSolverOptions invalid;
  invalid.evaluationLimit = 0;
  passed &= check(solveSyncCoverSelection(universe, {0}, invalid).error ==
                      SyncCoverSelectionError::InvalidOptions,
                  "invalid search bounds are rejected");
  invalid = {};
  invalid.exactMechanismThreshold =
      SyncCoverSolverOptions::maximumExactMechanismThreshold + 1;
  passed &= check(solveSyncCoverSelection(universe, {0}, invalid).error ==
                      SyncCoverSelectionError::InvalidOptions,
                  "unsafe exact-search thresholds are rejected");
  return passed;
}

bool testBoundedSearchDiagnostics() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add target");
  takeIndex(graph.addDemand(demand(source, target)), passed, "add demand");
  SyncCoverMechanismUniverse universe(graph);
  const auto domain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add domain");
  addEvent(universe, domain, 1, 2, source, target, passed);

  SyncCoverSolverOptions options;
  options.exactMechanismThreshold = 0;
  options.evaluationLimit = 1;
  const SyncCoverSelectionResult incomplete =
      solveSyncCoverSelection(universe, {0}, options);
  passed &= check(incomplete &&
                      incomplete.mechanisms ==
                          std::vector<SyncCoverMechanismId>{0} &&
                      !incomplete.truncation.evaluationLimit &&
                      !incomplete.optimalityProven,
                  "a complete large-component greedy result is retained");
  passed &= check(static_cast<bool>(incomplete.cost),
                  "a successful truncated search returns its valid cost");

  options = {};
  options.evaluationLimit = 1;
  const SyncCoverSelectionResult exactIncomplete =
      solveSyncCoverSelection(universe, {0}, options);
  passed &= check(exactIncomplete &&
                      exactIncomplete.mechanisms ==
                          std::vector<SyncCoverMechanismId>{0} &&
                      exactIncomplete.truncation.evaluationLimit &&
                      !exactIncomplete.optimalityProven,
                  "exact search keeps the greedy incumbent at its budget");
  return passed;
}

bool testManyDemandCache() {
  constexpr std::size_t kDemandCount = 128;
  bool passed = true;
  SyncCoverGraph graph;
  std::vector<std::pair<SyncCoverNodeId, SyncCoverNodeId>> endpoints;
  for (std::size_t index = 0; index < kDemandCount; ++index) {
    const std::uint32_t sourceResource =
        static_cast<std::uint32_t>(index * 2 + 1);
    const std::uint32_t targetResource = sourceResource + 1;
    const SyncCoverNodeId source = takeIndex(
        graph.addNode(sourceResource, 1, 0, index * 2, {}, {targetResource}),
        passed, "add cache source");
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(targetResource, 1, 0, index * 2 + 1), passed,
                  "add cache target");
    takeIndex(graph.addDemand(demand(source, target)), passed,
              "add cache demand");
    endpoints.emplace_back(source, target);
  }
  SyncCoverMechanismUniverse universe(graph);
  for (std::size_t index = 0; index < kDemandCount; ++index) {
    const std::uint32_t sourceResource =
        static_cast<std::uint32_t>(index * 2 + 1);
    const std::uint32_t targetResource = sourceResource + 1;
    const auto domain =
        takeIndex(universe.addResourceDomain(SyncCoverResourceKind::EventId,
                                             sourceResource, targetResource, 8),
                  passed, "add cache domain");
    addEvent(universe, domain, sourceResource, targetResource,
             endpoints[index].first, endpoints[index].second, passed);
  }
  std::vector<SyncCoverDemandId> demands(kDemandCount);
  for (std::size_t index = 0; index < kDemandCount; ++index) {
    demands[index] = index;
  }
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe, demands);
  passed &=
      check(result && result.components.size() == kDemandCount &&
                result.coverageStatistics.graphValidations == 1 &&
                result.coverageStatistics.demandPreparations == 1 &&
                result.coverageStatistics.groundingQueries == kDemandCount &&
                result.coverageStatistics.coverageQueries == 0 &&
                result.finalVerificationStatistics.demandPreparations ==
                    1,
            "incidence and final verification each share one execution "
            "context across many demands");
  return passed;
}

bool testMissingFactoryDemand() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add target");
  takeIndex(graph.addDemand(demand(source, target)), passed, "add demand");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverSelectionResult result =
      solveSyncCoverSelection(universe, {0});
  passed &= check(!result &&
                      result.error ==
                          SyncCoverSelectionError::SearchIncomplete &&
                      !result.optimalityProven &&
                      result.missingFactoryDemands ==
                          std::vector<SyncCoverDemandId>{0},
                  "missing factory coverage is not called infeasible");
  return passed;
}

} // namespace

int main() {
  const bool passed =
      testCutGuidedExactSelection() &&
      testAtomicProtocolAndScarcity() && testComponentsConflictsAndFailure() &&
      testBarrierSelection() && testTruncationRescuedByBarrierFallback() &&
      testDeclaredCoverageSelection() &&
      testBeamSeedsAndRedundancy() && testRecurrenceAndInputValidation() &&
      testBoundedSearchDiagnostics() && testManyDemandCache() &&
      testMissingFactoryDemand();
  return passed ? 0 : 1;
}
