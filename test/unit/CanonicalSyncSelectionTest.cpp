// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
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

template <typename Predicate>
CanonicalSyncProtocolVerifier testProtocolVerifier(Predicate predicate) {
  return [predicate = std::move(predicate)](
             const CanonicalSyncMechanismDescriptor &descriptor,
             SyncCoverCoverageWorkBudget &work) {
    const std::size_t supplies = descriptor.supplies.size();
    const bool squareOverflows =
        supplies != 0 &&
        supplies > std::numeric_limits<std::size_t>::max() / supplies;
    if (squareOverflows) {
      work.exhausted = true;
      return CanonicalSyncProblemError::LimitExceeded;
    }
    const std::size_t square = supplies * supplies;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const bool fixedOverflows =
        descriptor.eventUses.size() > maximum - 1 ||
        descriptor.actions.size() > maximum - 1 - descriptor.eventUses.size();
    const std::size_t fixed = fixedOverflows ? 0
                                             : 1 + descriptor.eventUses.size() +
                                                   descriptor.actions.size();
    if (fixedOverflows ||
        square > std::numeric_limits<std::size_t>::max() - fixed ||
        !work.consume(fixed + square)) {
      work.exhausted = true;
      return CanonicalSyncProblemError::LimitExceeded;
    }
    return predicate(descriptor)
               ? CanonicalSyncProblemError::None
               : CanonicalSyncProblemError::UnverifiedProtocol;
  };
}

struct CopyCountedVerifierCapture {
  CopyCountedVerifierCapture(std::shared_ptr<std::size_t> copyCount,
                             std::size_t width)
      : copyCount(std::move(copyCount)), payload(width, 7) {}

  CopyCountedVerifierCapture(const CopyCountedVerifierCapture &other)
      : copyCount(other.copyCount), payload(other.payload) {
    ++*copyCount;
  }

  CopyCountedVerifierCapture(CopyCountedVerifierCapture &&) noexcept = default;

  std::shared_ptr<std::size_t> copyCount;
  std::vector<std::size_t> payload;
};

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
mixedTargetLocalEvent(CanonicalSyncEventDomainId domain,
                      std::uint32_t sourceResource,
                      std::uint32_t targetResource, SyncCoverNodeId source,
                      SyncCoverNodeId target, SyncCoverScopeId loop,
                      SyncCoverDemandId distanceZeroDemand,
                      SyncCoverDemandId positiveDistanceDemand) {
  CanonicalSyncMechanismDescriptor result;
  result.eventUses.push_back({domain, 1, std::nullopt});
  result.actions.push_back({CanonicalSyncActionKind::EventSet,
                            sourceResource,
                            before(target),
                            0,
                            0,
                            {}});
  result.actions.push_back({CanonicalSyncActionKind::EventWait,
                            targetResource,
                            before(target),
                            0,
                            0,
                            {}});
  CanonicalSyncSupplyBinding distanceZero;
  distanceZero.edge = supply(source, target, loop);
  distanceZero.eventUse = 0;
  distanceZero.proof = CanonicalSyncSupplyProof::TargetLocalFenceAction;
  distanceZero.attestedDemand = distanceZeroDemand;
  distanceZero.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
  result.supplies.push_back(std::move(distanceZero));
  CanonicalSyncSupplyBinding positiveDistance;
  positiveDistance.edge = supply(source, target, loop, 1);
  positiveDistance.eventUse = 0;
  positiveDistance.proof = CanonicalSyncSupplyProof::TargetLocalFenceAction;
  positiveDistance.allowedDemands = {positiveDistanceDemand};
  positiveDistance.attestedDemand = positiveDistanceDemand;
  result.supplies.push_back(std::move(positiveDistance));
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

CanonicalSyncMechanismDescriptor sourceLocalBarrier(
    std::uint32_t resource, SyncCoverNodeId source,
    const std::vector<std::pair<SyncCoverNodeId, SyncCoverDemandId>> &targets) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Barrier;
  result.actions.push_back({CanonicalSyncActionKind::Barrier,
                            resource,
                            after(source),
                            std::nullopt,
                            0,
                            {resource},
                            CanonicalSyncBarrierKind::Targeted});
  for (const auto &[target, demandId] : targets) {
    CanonicalSyncSupplyBinding binding;
    binding.edge = supply(source, target);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::SourceLocalPipeDrainAction;
    binding.attestedDemand = demandId;
    binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    result.supplies.push_back(std::move(binding));
  }
  return result;
}

CanonicalSyncMechanismDescriptor
targetLocalPipeDrain(std::uint32_t sourceResource, SyncCoverNodeId source,
                     SyncCoverNodeId target, SyncCoverDemandId demandId) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Barrier;
  result.actions.push_back({CanonicalSyncActionKind::Barrier,
                            sourceResource,
                            before(target),
                            std::nullopt,
                            0,
                            {sourceResource},
                            CanonicalSyncBarrierKind::Targeted});
  CanonicalSyncSupplyBinding binding;
  binding.edge = supply(source, target);
  binding.barrierAction = 0;
  binding.proof = CanonicalSyncSupplyProof::TargetLocalPipeDrainAction;
  binding.attestedDemand = demandId;
  binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
  result.supplies.push_back(std::move(binding));
  return result;
}

CanonicalSyncMechanismDescriptor sourcePrefixPipeDrain(
    const SyncCoverGraph &graph, std::uint32_t resource, SyncCoverNodeId cut,
    const std::vector<std::pair<SyncCoverNodeId, SyncCoverDemandId>> &targets) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Barrier;
  result.actions.push_back({CanonicalSyncActionKind::Barrier,
                            resource,
                            after(cut),
                            std::nullopt,
                            0,
                            {resource},
                            CanonicalSyncBarrierKind::Targeted});
  for (const auto &[target, demandId] : targets) {
    const SyncCoverDemand &demandDescription = graph.getDemands()[demandId];
    CanonicalSyncSupplyBinding binding;
    binding.edge = supply(demandDescription.source, target,
                          demandDescription.scope, demandDescription.distance);
    binding.barrierAction = 0;
    binding.proof = CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
    binding.attestedDemand = demandId;
    if (binding.edge.distance == 0) {
      binding.applicability = SyncCoverSupplyApplicability::DistanceZeroOnly;
    } else {
      binding.allowedDemands = {demandId};
    }
    result.supplies.push_back(std::move(binding));
  }
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
  SyncCoverCoverageLimits baselineLimits;
  baselineLimits.maximumResultWords = 1;
  const SyncCoverSingletonCoverageResult baselineLimited =
      computeSyncCoverSingletonCoverage(
          graph, expansion, 1, {{0, supply(source, target)}}, baselineLimits);
  passed &=
      check(baselineLimited.error == SyncCoverCoverageError::LimitExceeded &&
                baselineLimited.baseline.size() == 0 &&
                baselineLimited.mechanisms.empty(),
            "count the singleton baseline row before allocation");
  SyncCoverCoverageLimits totalLimits;
  totalLimits.maximumResultWords = 2;
  totalLimits.maximumTotalWords = 2;
  const SyncCoverSingletonCoverageResult totalLimited =
      computeSyncCoverSingletonCoverage(
          graph, expansion, 1, {{0, supply(source, target)}}, totalLimits);
  passed &= check(totalLimited.error == SyncCoverCoverageError::LimitExceeded &&
                      totalLimited.baseline.size() == 0 &&
                      totalLimited.mechanisms.empty(),
                  "bound simultaneous singleton result and workspace words");
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
  const std::vector<SyncCoverCompletionSupply> supplies = {
      {first, supply(source, middle)}, {second, supply(middle, target)}};
  SyncCoverCoverageLimits limitedPairWords;
  limitedPairWords.maximumTotalWords = 3;
  const SyncCoverPairCoverageResult limitedPair = computeSyncCoverPairCoverage(
      graph, problem.getExpansion(), problem.getMechanisms().size(), supplies,
      {{first, second}}, problem.getDemands(), limitedPairWords);
  passed &= check(limitedPair.error == SyncCoverCoverageError::LimitExceeded &&
                      limitedPair.pairs.empty(),
                  "bound simultaneous pair result and workspace words");
  limitedPairWords.maximumTotalWords = 4;
  const SyncCoverPairCoverageResult exactPair = computeSyncCoverPairCoverage(
      graph, problem.getExpansion(), problem.getMechanisms().size(), supplies,
      {{first, second}}, problem.getDemands(), limitedPairWords);
  passed &= check(exactPair && exactPair.pairs.size() == 1 &&
                      exactPair.pairs[0].contains(0),
                  "admit pair result and workspace at the exact word bound");
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

bool testFixedCoverUsesFrozenCompositeColumn() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}),
                                           passed, "add fixed-column source");
  const SyncCoverNodeId firstMiddle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed,
                "add fixed-column first middle");
  const SyncCoverNodeId secondMiddle =
      takeIndex(graph.addNode(3, 1, 0, 2, {}, {4}), passed,
                "add fixed-column second middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(4, 1, 0, 3), passed, "add fixed-column target");
  passed &= check(graph.addDemand(demand(source, secondMiddle)),
                  "add fixed-column pair demand");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add fixed-column triple demand");
  passed &= check(graph.freezeStructure(), "freeze fixed-column graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add fixed-column first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add fixed-column second domain");
  passed &= check(problem.addEventDomain({2, 3, 4, 8, {}}),
                  "add fixed-column third domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, firstMiddle)),
                passed, "add fixed-column first event");
  const CanonicalSyncMechanismId second = takeIndex(
      problem.internMechanism(event(1, 2, 3, firstMiddle, secondMiddle)),
      passed, "add fixed-column second event");
  const CanonicalSyncMechanismId third =
      takeIndex(problem.internMechanism(event(2, 3, 4, secondMiddle, target)),
                passed, "add fixed-column third event");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 1,
                  "retain the fixed-column direct pair");
  passed &= check(problem.addPattern({CanonicalSyncPatternKind::RepairFrontier,
                                      {first, second, third}}),
                  "add the fixed-column three-member pattern");
  passed &= check(problem.freeze(), "freeze fixed-column problem");

  CanonicalSyncGreedyOptions options;
  options.strategy = CanonicalSyncSelectionStrategy::FixedCover;
  const CanonicalSyncSelection firstSelection =
      selectCanonicalSyncPatterns(problem, options);
  const CanonicalSyncSelection secondSelection =
      selectCanonicalSyncPatterns(problem, options);
  const std::vector<CanonicalSyncMechanismId> expected{first, second, third};
  return passed &&
         check(firstSelection && firstSelection.mechanisms == expected &&
                   firstSelection.selectionOrder == expected,
               "select the complete frozen composite column in one move") &&
         check(secondSelection && secondSelection.mechanisms == expected &&
                   secondSelection.selectionOrder == expected &&
                   secondSelection.statistics.patternEvaluations ==
                       firstSelection.statistics.patternEvaluations &&
                   secondSelection.statistics.workUnits ==
                       firstSelection.statistics.workUnits,
               "repeat the fixed-column streaming choice deterministically");
}

bool testStreamingGreedySelectionIsDeterministic() {
  bool passed = true;
  constexpr std::size_t candidateCount = 16;
  SyncCoverGraph graph;
  std::vector<SyncCoverNodeId> sources;
  std::vector<SyncCoverNodeId> targets;
  sources.reserve(candidateCount);
  targets.reserve(candidateCount);
  for (std::size_t candidate = 0; candidate < candidateCount; ++candidate) {
    const std::uint32_t sourceResource =
        static_cast<std::uint32_t>(2 * candidate + 1);
    const std::uint32_t targetResource = sourceResource + 1;
    sources.push_back(
        takeIndex(graph.addNode(sourceResource, 1, 0, 2 * candidate, {},
                                {targetResource}),
                  passed, "add streaming source"));
    targets.push_back(
        takeIndex(graph.addNode(targetResource, 1, 0, 2 * candidate + 1),
                  passed, "add streaming target"));
    passed &= check(graph.addDemand(demand(sources.back(), targets.back())),
                    "add streaming demand");
  }
  passed &= check(graph.freezeStructure(), "freeze streaming graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  for (std::size_t candidate = 0; candidate < candidateCount; ++candidate) {
    const std::uint32_t sourceResource =
        static_cast<std::uint32_t>(2 * candidate + 1);
    const std::uint32_t targetResource = sourceResource + 1;
    passed &= check(problem.addEventDomain(
                        {candidate, sourceResource, targetResource, 1, {}}),
                    "add streaming event domain");
    passed &= check(
        problem.internMechanism(event(candidate, sourceResource, targetResource,
                                      sources[candidate], targets[candidate])),
        "add streaming event");
  }
  passed &= check(problem.freeze(), "freeze streaming problem");

  CanonicalSyncGreedyOptions options;
  options.strategy = CanonicalSyncSelectionStrategy::ActionAwareSingleton;
  const CanonicalSyncSelection first =
      selectCanonicalSyncPatterns(problem, options);
  const CanonicalSyncSelection second =
      selectCanonicalSyncPatterns(problem, options);
  std::vector<CanonicalSyncMechanismId> expected(candidateCount);
  std::iota(expected.begin(), expected.end(), 0);
  return passed &&
         check(first && first.mechanisms == expected &&
                   first.selectionOrder == expected,
               "stream every required singleton in stable order") &&
         check(second && second.mechanisms == first.mechanisms &&
                   second.selectionOrder == first.selectionOrder &&
                   second.statistics.patternEvaluations ==
                       first.statistics.patternEvaluations &&
                   second.statistics.workUnits == first.statistics.workUnits,
               "repeat the streaming singleton scan deterministically");
}

bool testDirectPairSkipsConflictingMechanisms() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed,
                "add conflicting-pair source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed,
                "add conflicting-pair middle");
  const SyncCoverNodeId target = takeIndex(graph.addNode(3, 1, 0, 2), passed,
                                           "add conflicting-pair target");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add conflicting-pair demand");
  passed &= check(graph.freezeStructure(), "freeze conflicting-pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add conflicting-pair first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add conflicting-pair second domain");
  passed &= check(problem.addEventDomain({2, 1, 3, 8, {}}),
                  "add conflicting-pair covering domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, middle)), passed,
                "add conflicting-pair first event");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(event(1, 2, 3, middle, target)), passed,
                "add conflicting-pair second event");
  passed &= check(problem.internMechanism(event(2, 1, 3, source, target)),
                  "add conflicting-pair covering event");
  passed &=
      check(problem.addConflict(first, second), "record direct-pair conflict");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 0,
                  "skip a conflicting direct-pair proposal");
  return passed &&
         check(problem.freeze(),
               "conflicting optional pair does not invalidate the problem");
}

bool testConflictingPairDoesNotConsumeProposalCapacity() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed,
                "add capacity-conflict source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}, std::nullopt, true), passed,
                "add capacity-conflict middle");
  const SyncCoverNodeId target = takeIndex(graph.addNode(3, 1, 0, 2), passed,
                                           "add capacity-conflict target");
  passed &= check(graph.addDemand(demand(source, target)),
                  "add capacity-conflict transitive demand");
  const SyncCoverDemandId directDemand =
      takeIndex(graph.addDemand(demand(middle, target)), passed,
                "add capacity-conflict directly attested demand");
  passed &= check(graph.freezeStructure(), "freeze capacity-conflict graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add capacity-conflict first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add capacity-conflict second domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, middle)), passed,
                "add capacity-conflict first event");
  const CanonicalSyncMechanismId conflicting =
      takeIndex(problem.internMechanism(event(1, 2, 3, middle, target)), passed,
                "add capacity-conflict second event");
  CanonicalSyncMechanismDescriptor targetFence;
  targetFence.eventUses.push_back({1, 1, std::nullopt});
  targetFence.actions.push_back(
      {CanonicalSyncActionKind::EventSet, 2, before(target), 0, 0, {}});
  targetFence.actions.push_back(
      {CanonicalSyncActionKind::EventWait, 3, before(target), 0, 0, {}});
  CanonicalSyncSupplyBinding targetFenceSupply;
  targetFenceSupply.edge = supply(middle, target);
  targetFenceSupply.eventUse = 0;
  targetFenceSupply.proof = CanonicalSyncSupplyProof::TargetLocalFenceAction;
  targetFenceSupply.attestedDemand = directDemand;
  targetFenceSupply.applicability =
      SyncCoverSupplyApplicability::DistanceZeroOnly;
  targetFence.supplies.push_back(std::move(targetFenceSupply));
  const CanonicalSyncMechanismId valid =
      takeIndex(problem.internMechanism(std::move(targetFence)), passed,
                "add capacity-conflict valid target fence");
  passed &= check(problem.addConflict(first, conflicting),
                  "record capacity-conflict pair");

  CanonicalSyncDirectPairOptions options;
  options.maximumEvaluationsPerScope = 1;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem, options);
  const CanonicalSyncPatternStatistics &statistics =
      problem.getPatternStatistics();
  passed &= check(generated && generated.index == 1 &&
                      !problem.wasPatternGenerationTruncated() &&
                      statistics.directPairProposals == 1 &&
                      statistics.directPairEvaluations == 1,
                  "exclude conflicts before consuming proposal capacity");
  passed &= check(problem.freeze(), "freeze capacity-conflict problem");
  if (!passed) {
    return false;
  }
  const CanonicalSyncPattern &pair = problem.getPatterns().back();
  return check(pair.kind == CanonicalSyncPatternKind::DirectPair &&
                   pair.members ==
                       std::vector<CanonicalSyncMechanismId>{first, valid} &&
                   pair.extraCoverageCount == 1 && pair.coverage.contains(0),
               "retain the later valid pair within a one-proposal budget");
}

bool testDirectPairTraversesFixedCompletionSupply() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed,
                "add fixed-connector pair source");
  const SyncCoverNodeId firstMiddle =
      takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed,
                "add fixed-connector first middle");
  const SyncCoverNodeId secondMiddle =
      takeIndex(graph.addNode(3, 1, 0, 2, {}, {4}), passed,
                "add fixed-connector second middle");
  const SyncCoverNodeId target = takeIndex(graph.addNode(4, 1, 0, 3), passed,
                                           "add fixed-connector pair target");
  passed &= check(graph.addEdge(supply(firstMiddle, secondMiddle)),
                  "add fixed completion connector") &&
            check(graph.addDemand(demand(source, target)),
                  "add fixed-connector pair demand") &&
            check(graph.freezeStructure(), "freeze fixed-connector pair graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add fixed-connector first domain") &&
            check(problem.addEventDomain({1, 3, 4, 8, {}}),
                  "add fixed-connector second domain");
  const CanonicalSyncMechanismId first =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, firstMiddle)),
                passed, "add fixed-connector first member");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(event(1, 3, 4, secondMiddle, target)),
                passed, "add fixed-connector second member");
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem);
  passed &= check(generated && generated.index == 1,
                  "retain pair joined by fixed completion supply") &&
            check(problem.freeze(), "freeze fixed-connector pair problem");
  if (!passed) {
    return false;
  }
  const CanonicalSyncPattern &pair = problem.getPatterns().back();
  return check(!problem.getPatterns()[first].coverage.contains(0) &&
                   !problem.getPatterns()[second].coverage.contains(0) &&
                   pair.kind == CanonicalSyncPatternKind::DirectPair &&
                   pair.members ==
                       std::vector<CanonicalSyncMechanismId>{first, second} &&
                   pair.coverage.contains(0) && pair.extraCoverageCount == 1,
               "fixed completion supply contributes only to the joint cover");
}

bool testDirectPairIndexesUnrestrictedBindingOfMixedMechanism() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add mixed-binding loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, {}, {2}), passed,
                "add mixed-binding source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, loop, 1, {}, {3}, std::nullopt, true),
                passed, "add mixed-binding middle");
  const SyncCoverNodeId target = takeIndex(graph.addNode(3, 1, loop, 2), passed,
                                           "add mixed-binding target");
  passed &= check(graph.addDemand(demand(source, target, loop)),
                  "add mixed-binding pair demand");
  const SyncCoverDemandId distanceZeroDemand = graph.getDemands().size();
  passed &= check(graph.addDemand(demand(middle, target, loop)),
                  "add mixed-binding distance-zero attestation");
  const SyncCoverDemandId positiveDistanceDemand = graph.getDemands().size();
  passed &= check(graph.addDemand(demand(middle, target, loop, 1)),
                  "add mixed-binding positive-distance attestation");
  const SyncCoverDemandId unrelatedPositiveDemand = graph.getDemands().size();
  passed &= check(graph.addDemand(demand(middle, target, loop, 2)),
                  "add mixed-binding unrelated positive-distance row");
  passed &= check(graph.freezeStructure(), "freeze mixed-binding graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add mixed-binding first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add mixed-binding second domain");
  const CanonicalSyncMechanismId first = takeIndex(
      problem.internMechanism(eventInScope(0, 1, 2, source, middle, loop)),
      passed, "add mixed-binding first event");
  const CanonicalSyncMechanismId second =
      takeIndex(problem.internMechanism(mixedTargetLocalEvent(
                    1, 2, 3, middle, target, loop, distanceZeroDemand,
                    positiveDistanceDemand)),
                passed, "add mixed target-local event");
  CanonicalSyncDirectPairOptions options;
  options.maximumConnectorIndexEntries = 4;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem, options);
  passed &= check(generated && generated.index == 1 &&
                      !problem.wasPatternGenerationTruncated(),
                  "index only the unrestricted binding of a mixed mechanism");
  CanonicalSyncMechanismDescriptor unrelatedCover =
      mixedTargetLocalEvent(1, 2, 3, middle, target, loop, distanceZeroDemand,
                            unrelatedPositiveDemand);
  unrelatedCover.supplies.erase(unrelatedCover.supplies.begin());
  unrelatedCover.supplies.front().edge.distance = 2;
  passed &= check(problem.internMechanism(std::move(unrelatedCover)),
                  "cover the unrelated recurrence row independently");
  if (!check(problem.freeze(), "freeze mixed-binding problem")) {
    return false;
  }
  const CanonicalSyncPattern &pair = problem.getPatterns().back();
  return passed &&
         check(pair.kind == CanonicalSyncPatternKind::DirectPair &&
                   pair.members ==
                       std::vector<CanonicalSyncMechanismId>{first, second} &&
                   pair.coverage.contains(0) &&
                   !pair.coverage.contains(unrelatedPositiveDemand),
               "retain pair coverage from a mixed mechanism's unrestricted "
               "binding without leaking it into recurrence rows");
}

bool testConnectorIndexRetainsDistinctEndpointNodes() {
  bool passed = true;
  SyncCoverGraph graph;
  CanonicalSyncMechanismDescriptor first;
  CanonicalSyncMechanismDescriptor second;
  SyncCoverNodeId demandedSource = 0;
  SyncCoverNodeId demandedTarget = 0;
  constexpr std::size_t packageWidth = 16;
  for (std::size_t index = 0; index < packageWidth; ++index) {
    const SyncCoverNodeId source =
        takeIndex(graph.addNode(1, 1, 0, index * 3, {}, {2}), passed,
                  "add duplicate-connector source");
    const SyncCoverNodeId middle =
        takeIndex(graph.addNode(2, 1, 0, index * 3 + 1, {}, {3}), passed,
                  "add duplicate-connector middle");
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(3, 1, 0, index * 3 + 2), passed,
                  "add duplicate-connector target");
    if (index == 0) {
      demandedSource = source;
      demandedTarget = target;
    }
    const auto appendEvent =
        [&](CanonicalSyncMechanismDescriptor &descriptor,
            CanonicalSyncEventDomainId domain, SyncCoverNodeId producer,
            SyncCoverNodeId consumer, std::uint32_t producerResource,
            std::uint32_t consumerResource) {
          const std::size_t use = descriptor.eventUses.size();
          descriptor.eventUses.push_back({domain, 1, std::nullopt});
          descriptor.actions.push_back({CanonicalSyncActionKind::EventSet,
                                        producerResource,
                                        after(producer),
                                        use,
                                        0,
                                        {}});
          descriptor.actions.push_back({CanonicalSyncActionKind::EventWait,
                                        consumerResource,
                                        before(consumer),
                                        use,
                                        0,
                                        {}});
          CanonicalSyncSupplyBinding binding;
          binding.edge = supply(producer, consumer);
          binding.eventUse = use;
          descriptor.supplies.push_back(std::move(binding));
        };
    appendEvent(first, 0, source, middle, 1, 2);
    appendEvent(second, 1, middle, target, 2, 3);
  }
  passed &= check(graph.addDemand(demand(demandedSource, demandedTarget)),
                  "add duplicate-connector demand");
  passed &= check(graph.freezeStructure(), "freeze duplicate-connector graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add duplicate-connector first domain");
  passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                  "add duplicate-connector second domain");
  passed &= check(problem.internMechanism(std::move(first)),
                  "add duplicate first connector");
  passed &= check(problem.internMechanism(std::move(second)),
                  "add duplicate second connector");

  CanonicalSyncDirectPairOptions options;
  options.maximumConnectorInspections = packageWidth * 2 + 1;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem, options);
  passed &= check(
      generated && generated.index == 1 &&
          !problem.wasPatternGenerationTruncated() &&
          problem.getPatternStatistics().directPairConnectorInspections ==
              packageWidth * 2 + 1,
      "join only structurally connected endpoint nodes within the exact work "
      "bound");
  passed &= check(problem.freeze(), "freeze duplicate-connector problem");
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
  options.maximumPreparationWords = 7;
  const CanonicalSyncProblemResult generated =
      addCanonicalSyncDirectPairPatterns(problem, options);
  passed &= check(generated && generated.index == 0 &&
                      problem.wasPatternGenerationTruncated() &&
                      problem.getPatternStatistics().directPairProposals == 1 &&
                      problem.getPatternStatistics().directPairEvaluations == 0,
                  "truncate optional pairs at the aggregate preparation bound");
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

  CanonicalSyncPatternProblem connectorLimited(graph, allDemands(graph));
  passed &= check(connectorLimited.addEventDomain({0, 1, 3, 8, {}}),
                  "add connector-limit direct domain");
  passed &= check(connectorLimited.addEventDomain({1, 1, 2, 8, {}}),
                  "add connector-limit first domain");
  passed &= check(connectorLimited.addEventDomain({2, 2, 3, 8, {}}),
                  "add connector-limit second domain");
  passed &=
      check(connectorLimited.internMechanism(event(0, 1, 3, source, target)),
            "add connector-limit covering singleton");
  passed &=
      check(connectorLimited.internMechanism(event(1, 1, 2, source, middle)),
            "add connector-limit first pair member");
  passed &=
      check(connectorLimited.internMechanism(event(2, 2, 3, middle, target)),
            "add connector-limit second pair member");
  CanonicalSyncDirectPairOptions connectorOptions;
  connectorOptions.maximumConnectorInspections = 1;
  const CanonicalSyncProblemResult connectorGeneration =
      addCanonicalSyncDirectPairPatterns(connectorLimited, connectorOptions);
  passed &=
      check(connectorGeneration && connectorGeneration.index == 0 &&
                connectorLimited.wasPatternGenerationTruncated() &&
                connectorLimited.getPatternStatistics()
                        .directPairConnectorInspections == 1,
            "truncate connector discovery at its explicit inspection bound");
  passed &= check(connectorLimited.freeze(),
                  "freeze singleton-valid connector-limited problem");
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
        problem.internVerifiedProtocol(
            std::move(multi), testProtocolVerifier([](const auto &candidate) {
              return candidate.supplies.size() == 2;
            })),
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
                    "own every pair at the supply LCA and atomically skip an "
                    "oversized owner batch");
  };
  run(true);
  run(false);
  return passed;
}

bool testPairOwnerExact4096Boundary() {
  bool passed = true;
  const auto run = [&](std::size_t pairCount, bool expectTruncated) {
    SyncCoverGraph graph;
    const SyncCoverNodeId source = takeIndex(
        graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add 4096-boundary source");
    const SyncCoverNodeId middle =
        takeIndex(graph.addNode(2, 1, 0, 1, {}, {3}), passed,
                  "add 4096-boundary connector");
    std::vector<SyncCoverNodeId> targets;
    targets.reserve(pairCount);
    for (std::size_t index = 0; index < pairCount; ++index) {
      targets.push_back(takeIndex(graph.addNode(3, 1, 0, index + 2), passed,
                                  "add 4096-boundary target"));
    }
    passed &= check(graph.freezeStructure(), "freeze 4096-boundary graph");
    CanonicalSyncPatternProblem problem(graph, allDemands(graph));
    passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                    "add 4096-boundary first domain");
    passed &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                    "add 4096-boundary second domain");
    passed &= check(problem.internMechanism(event(0, 1, 2, source, middle)),
                    "add 4096-boundary first member");
    for (SyncCoverNodeId target : targets) {
      passed &= check(problem.internMechanism(event(1, 2, 3, middle, target)),
                      "add 4096-boundary successor");
    }
    CanonicalSyncDirectPairOptions options;
    options.maximumEvaluationsPerScope = 4096;
    const CanonicalSyncProblemResult generated =
        addCanonicalSyncDirectPairPatterns(problem, options);
    const CanonicalSyncPatternStatistics &statistics =
        problem.getPatternStatistics();
    passed &= check(
        generated && generated.index == 0 &&
            problem.wasPatternGenerationTruncated() == expectTruncated &&
            statistics.directPairProposals == pairCount &&
            statistics.directPairEvaluations ==
                (expectTruncated ? 0 : pairCount),
        expectTruncated ? "skip an owner atomically at proposal 4097"
                        : "evaluate the complete owner batch at proposal 4096");
  };
  run(4096, false);
  run(4097, true);
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
          testProtocolVerifier(
              [&](const CanonicalSyncMechanismDescriptor &candidate) {
                return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
                       candidate.supplies.size() == 1 &&
                       candidate.supplies.front().edge.distance == 1;
              })),
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

bool testStructuralCostSeparatesBarrierAndEventActions() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed,
                "add barrier-priority source");
  for (std::size_t order = 1; order < 7; ++order) {
    takeIndex(graph.addNode(3, 1, 0, order), passed,
              "add barrier-priority independent work");
  }
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, 0, 7), passed,
                                           "add barrier-priority target");
  const SyncCoverDemandId demandId =
      takeIndex(graph.addDemand(demand(source, target)), passed,
                "add barrier-priority demand");
  passed &= check(graph.setBlockingTargetedBarrierResources({1}),
                  "enable barrier-priority source drain") &&
            check(graph.freezeStructure(), "freeze barrier-priority graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                  "add barrier-priority event domain");
  const CanonicalSyncMechanismId barrierId =
      takeIndex(problem.internMechanism(
                    sourceLocalBarrier(1, source, {{target, demandId}})),
                passed, "add barrier-priority targeted drain");
  const CanonicalSyncMechanismId eventId =
      takeIndex(problem.internMechanism(event(0, 1, 2, source, target)), passed,
                "add barrier-priority event");
  passed &= check(problem.freeze(), "freeze barrier-priority problem");

  const std::optional<CanonicalSyncStructuralCost> cost =
      computeCanonicalSyncStructuralCost(problem, {barrierId, eventId});
  if (!check(cost.has_value(), "compute checked structural cost")) {
    return false;
  }
  const std::uint64_t barriers =
      std::accumulate(cost->barrierActionProfile.begin(),
                      cost->barrierActionProfile.end(), std::uint64_t{0});
  const std::uint64_t events =
      std::accumulate(cost->eventActionProfile.begin(),
                      cost->eventActionProfile.end(), std::uint64_t{0});
  const std::uint64_t actions = std::accumulate(
      cost->actionProfile.begin(), cost->actionProfile.end(), std::uint64_t{0});
  CanonicalSyncGreedyOptions actionFirst;
  actionFirst.objective = CanonicalSyncSelectionObjective::ActionFirst;
  const CanonicalSyncSelection actionSelection =
      selectCanonicalSyncPatterns(problem, actionFirst);
  CanonicalSyncGreedyOptions serializationFirst;
  serializationFirst.objective =
      CanonicalSyncSelectionObjective::SerializationFirst;
  const CanonicalSyncSelection serializationSelection =
      selectCanonicalSyncPatterns(problem, serializationFirst);
  return passed &&
         check(barriers == 1 && events == 2 && actions == 3,
               "report separate and aggregate action profiles") &&
         check(actionSelection &&
                   actionSelection.mechanisms ==
                       std::vector<CanonicalSyncMechanismId>{barrierId},
               "action-first objective prefers one broad barrier") &&
         check(serializationSelection &&
                   serializationSelection.mechanisms ==
                       std::vector<CanonicalSyncMechanismId>{eventId},
               "serialization-first objective prefers the tight event");
}

bool testStructuralCostOverflowFailsClosed() {
  bool passed = true;
  SyncCoverGraph graph;
  const std::uint64_t heavyWeight =
      std::numeric_limits<std::uint64_t>::max() / 4;
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, heavyWeight, 0, 0, {}, {2}), passed,
                "add first checked-cost source");
  const SyncCoverNodeId firstTarget = takeIndex(
      graph.addNode(2, 1, 0, 1), passed, "add first checked-cost target");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(1, heavyWeight, 0, 2, {}, {2}), passed,
                "add second checked-cost source");
  const SyncCoverNodeId secondTarget = takeIndex(
      graph.addNode(2, 1, 0, 3), passed, "add second checked-cost target");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget)),
                  "add first checked-cost demand") &&
            check(graph.addDemand(demand(secondSource, secondTarget)),
                  "add second checked-cost demand") &&
            check(graph.freezeStructure(), "freeze checked-cost graph");
  if (!passed) {
    return false;
  }

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &=
      check(problem.addEventDomain({0, 1, 2, 8, {}}),
            "add checked-cost event domain") &&
      check(problem.internMechanism(event(0, 1, 2, firstSource, firstTarget)),
            "add first individually representable cost") &&
      check(problem.internMechanism(event(0, 1, 2, secondSource, secondTarget)),
            "add second individually representable cost") &&
      check(problem.freeze(), "freeze checked-cost problem");
  if (!passed) {
    return false;
  }
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return check(selection.error ==
                       CanonicalSyncSelectionError::ArithmeticOverflow &&
                   selection.statistics.arithmeticOverflow,
               "reject aggregate cost overflow without saturation") &&
         check(!computeCanonicalSyncStructuralCost(problem, {0, 1}),
               "reject direct checked-cost aggregation overflow");
}

bool testPipeAllFallbackProblem() {
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

  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &=
      check(selection && selection.mechanisms ==
                             std::vector<CanonicalSyncMechanismId>{rescue},
            "select PIPE_ALL from a fallback-only problem");
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
  passed &= check(problem.addPattern({CanonicalSyncPatternKind::RepairFrontier,
                                      {first, second}}),
                  "add package-only pattern");
  passed &= check(problem.addPattern({CanonicalSyncPatternKind::RepairFrontier,
                                      {first, third}}),
                  "add overlapping package-only pattern");
  passed &= check(problem.freeze(), "freeze package-only problem");
  const CanonicalSyncPatternKindStatistics &statistics =
      problem.getPatternStatistics().get(
          CanonicalSyncPatternKind::RepairFrontier);
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

bool testSeparateFallbackRepairsEventPressure() {
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
  passed &= check(problem.freeze(), "freeze precise scarcity problem");
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

  CanonicalSyncPatternProblem fallbackProblem(graph, allDemands(graph));
  passed &= check(fallbackProblem.addEventDomain({0, 1, 2, 1, {}}),
                  "add fallback scarcity domain");
  const CanonicalSyncMechanismId fallbackFirstEvent = takeIndex(
      fallbackProblem.internMechanism(event(0, 1, 2, firstSource, firstTarget)),
      passed, "add first fallback event");
  passed &= check(fallbackFirstEvent == firstEvent,
                  "preserve precise mechanism IDs in the fallback problem");
  passed &= check(fallbackProblem.internMechanism(
                      event(0, 1, 2, secondSource, secondTarget)),
                  "add second fallback event");
  const CanonicalSyncMechanismId fallback =
      takeIndex(fallbackProblem.internMechanism(
                    barrier(2, {1, 2}, secondSource, secondTarget)),
                passed, "add scarcity fallback barrier");
  passed &= check(fallbackProblem.freeze(), "freeze scarcity fallback problem");
  const CanonicalSyncSelection repaired =
      selectCanonicalSyncPatterns(fallbackProblem);
  passed &= check(repaired && repaired.mechanisms ==
                                  std::vector<CanonicalSyncMechanismId>{
                                      firstEvent, fallback},
                  "an explicitly enabled fallback repairs event pressure");
  return passed;
}

bool testOptionalPipelineFallback() {
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
  }
  const CanonicalSyncProblemResult optional = addCanonicalSyncFeasiblePattern(
      problem, {CanonicalSyncPatternKind::RepairFrontier, events});
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
          {CanonicalSyncPatternKind::RepairFrontier, limitedEvents});
  passed &= check(memberCapped && !memberCapped.index,
                  "skip an oversized optional pipeline");
  passed &= check(problem.freeze(), "freeze precise pipeline problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  passed &= check(selection.error ==
                          CanonicalSyncSelectionError::ResourceInfeasible &&
                      selection.allocation.domains[0].required == 9 &&
                      selection.allocation.domains[0].available == 8,
                  "post-cover allocation diagnoses the oversized event family");

  CanonicalSyncPatternProblem fallbackProblem(graph, allDemands(graph));
  passed &= check(fallbackProblem.addEventDomain({0, 1, 2, 8, {}}),
                  "add pipeline fallback domain");
  for (std::size_t index = 0; index < 9; ++index) {
    passed &= check(fallbackProblem.internMechanism(
                        event(0, 1, 2, sources[index], targets[index])),
                    "add pipeline fallback event");
    passed &= check(fallbackProblem.internMechanism(
                        barrier(2, {1, 2}, sources[index], targets[index])),
                    "add pipeline scarcity fallback");
  }
  passed &= check(fallbackProblem.freeze(), "freeze pipeline fallback problem");
  const CanonicalSyncSelection repaired =
      selectCanonicalSyncPatterns(fallbackProblem);
  passed &= check(repaired && repaired.allocation.feasible,
                  "fallback-enabled re-cover is event feasible");
  passed &= check(static_cast<bool>(
                      verifyCanonicalSyncSelection(fallbackProblem, repaired)),
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
  passed &= check(problem.verifyMechanism(mechanism),
                  "freshly revalidate immutable mechanism data");
  passed &= check(problem.getMechanismSignature(mechanism) != 0 &&
                      problem.getMechanismSignature(mechanism) ==
                          problem.getMechanismSignature(mechanism),
                  "derive a stable immutable mechanism signature");
  std::size_t lowerVerificationBound = 1;
  std::size_t upperVerificationBound = 1U << 20;
  while (lowerVerificationBound < upperVerificationBound) {
    const std::size_t middle =
        lowerVerificationBound +
        (upperVerificationBound - lowerVerificationBound) / 2;
    SyncCoverCoverageWorkBudget work(middle);
    if (verifyCanonicalSyncSelection(problem, selection, &work)) {
      upperVerificationBound = middle;
    } else {
      lowerVerificationBound = middle + 1;
    }
  }
  SyncCoverCoverageWorkBudget exactWork(lowerVerificationBound);
  const CanonicalSyncVerifiedPlan exact =
      verifyCanonicalSyncSelection(problem, selection, &exactWork);
  SyncCoverCoverageWorkBudget belowWork(lowerVerificationBound - 1);
  const CanonicalSyncVerifiedPlan below =
      verifyCanonicalSyncSelection(problem, selection, &belowWork);
  passed &= check(exact && exactWork.workUnits == lowerVerificationBound,
                  "accept fresh verification at its exact work bound");
  passed &=
      check(below.error == CanonicalSyncSelectionError::WorkLimitExceeded &&
                belowWork.exhausted,
            "reject fresh verification one unit below its exact work bound");
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
  const CanonicalSyncMechanismId wide =
      takeIndex(problem.internVerifiedProtocol(
                    protocol(0, 1, 2, sources[2], targets[2], loop, 2),
                    testProtocolVerifier(
                        [](const CanonicalSyncMechanismDescriptor &descriptor) {
                          return descriptor.kind ==
                                     CanonicalSyncMechanismKind::Protocol &&
                                 descriptor.eventUses.size() == 1 &&
                                 descriptor.eventUses[0].width == 2;
                        })),
                passed, "add allocator wide event");
  const CanonicalSyncResourceAllocation reused =
      allocateCanonicalSyncEvents(problem, {first, second});
  passed &= check(
      reused.valid && reused.feasible && reused.domains[0].required == 2 &&
          reused.domains[0].uses[0].ids == std::vector<unsigned>{0} &&
          reused.domains[0].uses[1].ids == std::vector<unsigned>{1},
      "lexically disjoint event uses retain distinct hardware channels");
  CanonicalSyncPatternProblem oneIdProblem(graph, allDemands(graph));
  passed &= check(oneIdProblem.addEventDomain({0, 1, 2, 1, {}}),
                  "add one-ID allocator domain");
  const CanonicalSyncMechanismId oneIdFirst = takeIndex(
      oneIdProblem.internMechanism(event(0, 1, 2, sources[0], targets[0])),
      passed, "add first one-ID event");
  const CanonicalSyncMechanismId oneIdSecond = takeIndex(
      oneIdProblem.internMechanism(event(0, 1, 2, sources[1], targets[1])),
      passed, "add second one-ID event");
  const CanonicalSyncResourceAllocation oneIdAllocation =
      allocateCanonicalSyncEvents(oneIdProblem, {oneIdFirst, oneIdSecond});
  passed &= check(oneIdAllocation.valid && !oneIdAllocation.feasible &&
                      oneIdAllocation.domains[0].required == 2 &&
                      oneIdAllocation.domains[0].available == 1,
                  "independent event channels fail closed at ID exhaustion");
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
  const auto protocolPredicate =
      [=](const CanonicalSyncMechanismDescriptor &candidate) {
        return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
               candidate.eventUses.size() == 1 &&
               candidate.eventUses[0].recurrenceScope == loop &&
               candidate.supplies.size() == 1 &&
               candidate.actions.size() == 2 &&
               candidate.actions[0].anchor.kind ==
                   SyncCoverAnchorKind::AfterNode &&
               candidate.actions[0].anchor.node == source &&
               candidate.actions[1].anchor.kind ==
                   SyncCoverAnchorKind::BeforeNode &&
               candidate.actions[1].anchor.node == target &&
               candidate.supplies[0].edge.source == source &&
               candidate.supplies[0].edge.target == target &&
               candidate.supplies[0].edge.distance == 1;
      };
  const auto verifier = testProtocolVerifier(protocolPredicate);
  const CanonicalSyncProblemResult admitted = problem.internVerifiedProtocol(
      protocol(0, 1, 2, source, target, loop, 1, 1), verifier);
  passed &= check(admitted, "verified recurrence protocol is admitted");

  CanonicalSyncPatternProblem freshProblem(graph, allDemands(graph));
  passed &= check(freshProblem.addEventDomain({0, 1, 2, 2, {}}),
                  "add fresh-verification protocol domain");
  bool certificateStillValid = true;
  const CanonicalSyncProblemResult freshlyAdmitted =
      freshProblem.internVerifiedProtocol(
          protocol(0, 1, 2, source, target, loop, 1, 1),
          testProtocolVerifier(
              [&](const CanonicalSyncMechanismDescriptor &candidate) {
                return certificateStillValid && protocolPredicate(candidate);
              }));
  passed &= check(freshlyAdmitted && freshlyAdmitted.index,
                  "admit a protocol with a persistent verifier");
  if (freshlyAdmitted.index) {
    certificateStillValid = false;
    passed &=
        check(freshProblem.verifyMechanism(*freshlyAdmitted.index).error ==
                  CanonicalSyncProblemError::UnverifiedProtocol,
              "fresh verification re-runs the persistent protocol verifier");
  }

  CanonicalSyncPatternProblem meteredProblem(graph, allDemands(graph));
  passed &= check(meteredProblem.addEventDomain({0, 1, 2, 2, {}}),
                  "add metered protocol domain");
  bool verifierBodyRan = false;
  constexpr std::size_t callbackWork = 7;
  const CanonicalSyncProtocolVerifier meteredVerifier =
      [&](const CanonicalSyncMechanismDescriptor &candidate,
          SyncCoverCoverageWorkBudget &work) {
        if (!work.consume(callbackWork)) {
          return CanonicalSyncProblemError::LimitExceeded;
        }
        verifierBodyRan = true;
        return protocolPredicate(candidate)
                   ? CanonicalSyncProblemError::None
                   : CanonicalSyncProblemError::UnverifiedProtocol;
      };
  const CanonicalSyncProblemResult meteredAdmission =
      meteredProblem.internVerifiedProtocol(
          protocol(0, 1, 2, source, target, loop, 1, 1), meteredVerifier);
  passed &= check(meteredAdmission && meteredAdmission.index,
                  "admit an explicitly metered protocol verifier");
  if (meteredAdmission.index) {
    verifierBodyRan = false;
    SyncCoverCoverageWorkBudget shortCallbackWork(callbackWork - 1);
    passed &=
        check(meteredProblem
                          .verifyMechanism(*meteredAdmission.index,
                                           &shortCallbackWork)
                          .error == CanonicalSyncProblemError::LimitExceeded &&
                  !verifierBodyRan,
              "reject one-less callback work before verifier predicate work");

    SyncCoverCoverageWorkBudget measuredWork;
    verifierBodyRan = false;
    const CanonicalSyncProblemResult measured =
        meteredProblem.verifyMechanism(*meteredAdmission.index, &measuredWork);
    passed &= check(measured && verifierBodyRan && measuredWork.workUnits != 0,
                    "measure complete protocol fresh-verification work");
    SyncCoverCoverageWorkBudget exactWork(measuredWork.workUnits);
    const CanonicalSyncProblemResult exact =
        meteredProblem.verifyMechanism(*meteredAdmission.index, &exactWork);
    passed &= check(exact && exactWork.workUnits == measuredWork.workUnits,
                    "accept the exact fresh-verification work bound");
    SyncCoverCoverageWorkBudget oneLessWork(measuredWork.workUnits - 1);
    passed &= check(
        meteredProblem.verifyMechanism(*meteredAdmission.index, &oneLessWork)
                .error == CanonicalSyncProblemError::LimitExceeded,
        "reject one-less complete fresh-verification work");

    SyncCoverGraph wideGraph;
    const SyncCoverScopeId wideLoop = takeIndex(
        wideGraph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
        passed, "add wide-completion protocol loop");
    std::vector<std::uint32_t> completionTargets(128);
    std::iota(completionTargets.begin(), completionTargets.end(), 2);
    const SyncCoverNodeId wideSource =
        takeIndex(wideGraph.addNode(1, 1, wideLoop, 1, {}, completionTargets),
                  passed, "add wide-completion protocol source");
    const SyncCoverNodeId wideTarget =
        takeIndex(wideGraph.addNode(2, 1, wideLoop, 2), passed,
                  "add wide-completion protocol target");
    passed &=
        check(wideGraph.addDemand(demand(wideSource, wideTarget, wideLoop, 1)),
              "add wide-completion protocol demand");
    passed &= check(wideGraph.freezeStructure(),
                    "freeze wide-completion protocol graph");
    CanonicalSyncPatternProblem wideProblem(wideGraph, allDemands(wideGraph));
    passed &= check(wideProblem.addEventDomain({0, 1, 2, 2, {}}),
                    "add wide-completion protocol domain");
    const CanonicalSyncProblemResult wideAdmission =
        wideProblem.internVerifiedProtocol(
            protocol(0, 1, 2, wideSource, wideTarget, wideLoop, 1, 1),
            [](const CanonicalSyncMechanismDescriptor &candidate,
               SyncCoverCoverageWorkBudget &work) {
              if (!work.consume(callbackWork)) {
                return CanonicalSyncProblemError::LimitExceeded;
              }
              return candidate.kind == CanonicalSyncMechanismKind::Protocol
                         ? CanonicalSyncProblemError::None
                         : CanonicalSyncProblemError::UnverifiedProtocol;
            });
    if (wideAdmission.index) {
      SyncCoverCoverageWorkBudget wideWork;
      const CanonicalSyncProblemResult wideVerified =
          wideProblem.verifyMechanism(*wideAdmission.index, &wideWork);
      passed &= check(
          wideVerified && wideWork.workUnits >= measuredWork.workUnits + 127,
          "charge the complete source completion-target lookup dimension");
      SyncCoverCoverageWorkBudget wideOneLess(wideWork.workUnits - 1);
      passed &= check(
          wideProblem.verifyMechanism(*wideAdmission.index, &wideOneLess)
                  .error == CanonicalSyncProblemError::LimitExceeded,
          "reject wide completion-target verification at one less work unit");
    } else {
      passed &= check(false, "admit wide-completion protocol");
    }
  }

  const auto rejectsBrokenVerifier =
      [&](CanonicalSyncProblemError returned, bool exhausts,
          CanonicalSyncProblemError expected, std::string_view message) {
        CanonicalSyncPatternProblem boundaryProblem(graph, allDemands(graph));
        if (!boundaryProblem.addEventDomain({0, 1, 2, 2, {}})) {
          return check(false, "add broken-verifier protocol domain");
        }
        const CanonicalSyncProblemResult result =
            boundaryProblem.internVerifiedProtocol(
                protocol(0, 1, 2, source, target, loop, 1, 1),
                [=](const CanonicalSyncMechanismDescriptor &,
                    SyncCoverCoverageWorkBudget &work) {
                  work.exhausted = exhausts;
                  return returned;
                });
        return check(result.error == expected &&
                         boundaryProblem.getMechanisms().empty(),
                     message);
      };
  passed &= rejectsBrokenVerifier(
      CanonicalSyncProblemError::None, true,
      CanonicalSyncProblemError::LimitExceeded,
      "normalize exhausted verifier success to a limit failure");
  passed &= rejectsBrokenVerifier(
      CanonicalSyncProblemError::UnverifiedProtocol, true,
      CanonicalSyncProblemError::LimitExceeded,
      "normalize exhausted semantic rejection to a limit failure");
  passed &= rejectsBrokenVerifier(
      CanonicalSyncProblemError::InvalidGraph, false,
      CanonicalSyncProblemError::UnverifiedProtocol,
      "normalize an out-of-contract verifier error to semantic rejection");

  CanonicalSyncPatternProblem freshBoundaryProblem(graph, allDemands(graph));
  passed &= check(freshBoundaryProblem.addEventDomain({0, 1, 2, 2, {}}),
                  "add fresh broken-verifier domain");
  bool exhaustFreshVerifier = false;
  const CanonicalSyncProblemResult freshBoundaryAdmission =
      freshBoundaryProblem.internVerifiedProtocol(
          protocol(0, 1, 2, source, target, loop, 1, 1),
          [&](const CanonicalSyncMechanismDescriptor &,
              SyncCoverCoverageWorkBudget &work) {
            work.exhausted = exhaustFreshVerifier;
            return CanonicalSyncProblemError::None;
          });
  if (freshBoundaryAdmission.index) {
    exhaustFreshVerifier = true;
    passed &= check(
        freshBoundaryProblem.verifyMechanism(*freshBoundaryAdmission.index)
                .error == CanonicalSyncProblemError::LimitExceeded,
        "normalize fresh verifier exhaustion before common validation");
  } else {
    passed &= check(false, "admit fresh broken-verifier protocol");
  }

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

bool testProtocolVerifierStorageSharesOpaqueCapture() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add shared-verifier loop");
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 1, {}, {2}), passed,
                "add shared-verifier source");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, loop, 2), passed,
                                           "add shared-verifier target");
  passed &= check(graph.addDemand(demand(source, target, loop, 1)),
                  "add shared-verifier demand") &&
            check(graph.freezeStructure(), "freeze shared-verifier graph");

  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  passed &= check(problem.addEventDomain({0, 1, 2, 2, {}}),
                  "add shared-verifier event domain");
  const auto copyCount = std::make_shared<std::size_t>(0);
  CanonicalSyncProtocolVerifier verifier =
      [capture = CopyCountedVerifierCapture(copyCount, 1U << 12)](
          const CanonicalSyncMechanismDescriptor &candidate,
          SyncCoverCoverageWorkBudget &work) {
        if (!work.consume()) {
          return CanonicalSyncProblemError::LimitExceeded;
        }
        const bool valid =
            capture.payload.size() == (1U << 12) &&
            capture.payload.front() == 7 &&
            candidate.kind == CanonicalSyncMechanismKind::Protocol;
        return valid ? CanonicalSyncProblemError::None
                     : CanonicalSyncProblemError::UnverifiedProtocol;
      };
  const std::size_t copiesAfterCallbackFormation = *copyCount;
  const CanonicalSyncProblemResult admitted = problem.internVerifiedProtocol(
      protocol(0, 1, 2, source, target, loop, 1, 1), std::move(verifier));
  passed &= check(admitted && *copyCount == copiesAfterCallbackFormation,
                  "move opaque verifier capture into immutable storage") &&
            check(problem.freeze(), "freeze shared-verifier problem");
  if (!passed) {
    return false;
  }

  SyncCoverCoverageWorkBudget cloneWork;
  std::unique_ptr<CanonicalSyncPatternProblem> clone =
      problem.cloneMutableRepairPrefix(&cloneWork);
  passed &= check(clone && *copyCount == copiesAfterCallbackFormation,
                  "clone only the shared verifier handle");
  if (clone && admitted.index) {
    passed &= check(clone->verifyMechanism(*admitted.index) &&
                        *copyCount == copiesAfterCallbackFormation,
                    "invoke a shared cloned verifier without target copies");
  }
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
  const auto verifier = testProtocolVerifier(
      [=](const CanonicalSyncMechanismDescriptor &candidate) {
        return candidate.kind == CanonicalSyncMechanismKind::Protocol &&
               candidate.eventUses.size() == 1 &&
               candidate.eventUses[0].recurrenceScope == inner &&
               candidate.eventUses[0].lifetimeScope == outer;
      });
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

bool testRepairProtocolAdmissionUsesSharedBudget() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add repair-budget loop");
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, loop, 1, {}, {2}), passed,
                "add first repair-budget source");
  const SyncCoverNodeId firstTarget = takeIndex(
      graph.addNode(2, 1, loop, 2), passed, "add first repair-budget target");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(3, 1, loop, 3, {}, {4}), passed,
                "add second repair-budget source");
  const SyncCoverNodeId secondTarget = takeIndex(
      graph.addNode(4, 1, loop, 4), passed, "add second repair-budget target");
  passed &= check(graph.addDemand(demand(firstSource, firstTarget, loop, 1)),
                  "add first repair-budget demand") &&
            check(graph.addDemand(demand(secondSource, secondTarget, loop, 1)),
                  "add second repair-budget demand") &&
            check(graph.freezeStructure(), "freeze repair-budget graph");
  constexpr std::size_t callbackWork = 11;
  const auto makeVerifier = [] {
    return CanonicalSyncProtocolVerifier{
        [](const CanonicalSyncMechanismDescriptor &candidate,
           SyncCoverCoverageWorkBudget &work) {
          if (!work.consume(callbackWork)) {
            return CanonicalSyncProblemError::LimitExceeded;
          }
          return candidate.kind == CanonicalSyncMechanismKind::Protocol
                     ? CanonicalSyncProblemError::None
                     : CanonicalSyncProblemError::UnverifiedProtocol;
        }};
  };
  CanonicalSyncPatternProblem precise(graph, {});
  passed &= check(precise.addEventDomain({0, 1, 2, 2, {}}),
                  "add first repair-budget event domain") &&
            check(precise.addEventDomain({1, 3, 4, 2, {}}),
                  "add second repair-budget event domain");
  const CanonicalSyncMechanismDescriptor prefixDescriptor =
      protocol(0, 1, 2, firstSource, firstTarget, loop, 1, 1);
  const CanonicalSyncMechanismDescriptor appendDescriptor =
      protocol(1, 3, 4, secondSource, secondTarget, loop, 1, 1);
  passed &=
      check(precise.internVerifiedProtocol(prefixDescriptor, makeVerifier()),
            "add nonempty repair catalog prefix") &&
      check(precise.freeze(), "freeze nonempty precise repair prefix");
  if (!passed) {
    return false;
  }
  const std::uint64_t prefixSignature = precise.getMechanismSignature(0);

  const auto verifyAdmissionBoundary =
      [&](const CanonicalSyncMechanismDescriptor &descriptor,
          std::size_t expectedSize, std::string_view exactMessage,
          std::string_view belowMessage) {
        SyncCoverCoverageWorkBudget referenceWork;
        std::unique_ptr<CanonicalSyncPatternProblem> reference =
            precise.cloneMutableRepairPrefix(&referenceWork);
        const std::size_t referencePrefixWork = referenceWork.workUnits;
        const CanonicalSyncProblemResult referenceAdmission =
            reference->internVerifiedProtocol(descriptor, makeVerifier());
        const std::size_t admissionWork =
            referenceWork.workUnits - referencePrefixWork;
        if (!check(referenceAdmission && admissionWork >= callbackWork &&
                       reference->getMechanisms().size() == expectedSize,
                   "measure nonempty repair catalog admission work")) {
          return false;
        }

        SyncCoverCoverageWorkBudget exactWork;
        std::unique_ptr<CanonicalSyncPatternProblem> exact =
            precise.cloneMutableRepairPrefix(&exactWork);
        const std::size_t exactPrefixWork = exactWork.workUnits;
        exactWork.maximumWorkUnits = exactPrefixWork + admissionWork;
        const CanonicalSyncProblemResult exactAdmission =
            exact->internVerifiedProtocol(descriptor, makeVerifier());
        if (!check(exactAdmission &&
                       exactWork.workUnits == exactWork.maximumWorkUnits &&
                       exact->getMechanisms().size() == expectedSize,
                   exactMessage)) {
          return false;
        }

        SyncCoverCoverageWorkBudget oneLessWork;
        std::unique_ptr<CanonicalSyncPatternProblem> oneLess =
            precise.cloneMutableRepairPrefix(&oneLessWork);
        const std::size_t oneLessPrefixWork = oneLessWork.workUnits;
        oneLessWork.maximumWorkUnits = oneLessPrefixWork + admissionWork - 1;
        const CanonicalSyncProblemResult rejected =
            oneLess->internVerifiedProtocol(descriptor, makeVerifier());
        return check(rejected.error ==
                             CanonicalSyncProblemError::LimitExceeded &&
                         oneLess->getMechanisms().size() == 1 &&
                         oneLess->getMechanismSignature(0) == prefixSignature,
                     belowMessage);
      };

  return verifyAdmissionBoundary(
             prefixDescriptor, 1,
             "resolve a duplicate collision bucket at the exact shared bound",
             "reject duplicate lookup one below without prefix mutation") &&
         verifyAdmissionBoundary(
             appendDescriptor, 2,
             "append every catalog vector at the exact shared bound",
             "reject catalog growth one below without prefix mutation");
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
  passed &= check(patternLimited
                          .addPattern({CanonicalSyncPatternKind::RepairFrontier,
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
      proposalLimited, {CanonicalSyncPatternKind::RepairFrontier,
                        {proposalFirst, proposalSecond}});
  passed &= check(proposal && !proposal.index &&
                      proposalLimited.wasPatternGenerationTruncated(),
                  "report bounded optional-pattern generation");
  passed &= check(proposalLimited.freeze(),
                  "freeze a proposal-limited fallback problem");
  return passed;
}

bool testSelectionBasisRetainsFullObligationVerification() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add basis first node");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(1, 1, 0, 1), passed, "add basis middle node");
  const SyncCoverNodeId last =
      takeIndex(graph.addNode(1, 1, 0, 2), passed, "add basis last node");
  SyncCoverEdge firstIssue = supply(first, middle);
  firstIssue.kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  SyncCoverEdge secondIssue = supply(middle, last);
  secondIssue.kind = SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
  passed &=
      check(graph.addEdge(firstIssue), "add first basis issue edge") &&
      check(graph.addEdge(secondIssue), "add second basis issue edge") &&
      check(graph.addDemand(demand(first, middle)), "add first basis demand") &&
      check(graph.addDemand(demand(middle, last)), "add second basis demand") &&
      check(graph.addDemand(demand(first, last)),
            "add redundant full-universe demand") &&
      check(graph.setBlockingTargetedBarrierResources({1}),
            "enable basis targeted barriers") &&
      check(graph.freezeStructure(), "freeze basis graph");
  if (!passed) {
    return false;
  }

  const std::vector<SyncCoverDemandId> obligations{0, 1, 2};
  CanonicalSyncPatternProblem reduced(graph, obligations,
                                      std::vector<SyncCoverDemandId>{0, 1},
                                      CanonicalSyncPatternProblem::Limits{});
  passed &= check(reduced.getObligationDemands() == obligations,
                  "retain the complete obligation universe") &&
            check(reduced.getDemands() == std::vector<SyncCoverDemandId>{0, 1},
                  "retain the smaller immutable selection basis") &&
            check(reduced.internMechanism(targetedBarrier(1, first, middle)),
                  "add first basis barrier") &&
            check(reduced.internMechanism(targetedBarrier(1, middle, last)),
                  "add second basis barrier") &&
            check(reduced.freeze(), "freeze reduced-basis problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(reduced);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(reduced, selection);
  passed &= check(selection && verified,
                  "verify all three obligations from two basis rows");

  CanonicalSyncPatternProblem differentBasis(
      graph, obligations, obligations, CanonicalSyncPatternProblem::Limits{});
  passed &=
      check(differentBasis.internMechanism(targetedBarrier(1, first, middle)),
            "add first different-basis barrier") &&
      check(differentBasis.internMechanism(targetedBarrier(1, middle, last)),
            "add second different-basis barrier") &&
      check(!reduced.hasSameCandidatePrefix(differentBasis),
            "reject an otherwise-identical different demand basis");

  CanonicalSyncPatternProblem invalid(graph, obligations,
                                      std::vector<SyncCoverDemandId>{0},
                                      CanonicalSyncPatternProblem::Limits{});
  passed &= check(invalid.internMechanism(targetedBarrier(1, first, middle)),
                  "add incomplete-basis barrier") &&
            check(invalid.freeze(), "freeze incomplete selection basis");
  const CanonicalSyncSelection incomplete =
      selectCanonicalSyncPatterns(invalid);
  const CanonicalSyncVerifiedPlan rejected =
      verifyCanonicalSyncSelection(invalid, incomplete);
  return passed &&
         check(incomplete && !rejected && rejected.firstUncoveredDemand,
               "reject a greedy-complete basis that misses an obligation");
}

bool testRecurrenceBasisLemmasPreserveExactDistance() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add recurrence-basis loop");
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, loop, 0, {}, {2}), passed,
                "add recurrence-basis first node");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, loop, 1, {}, {3}), passed,
                "add recurrence-basis middle node");
  const SyncCoverNodeId last = takeIndex(graph.addNode(3, 1, loop, 2), passed,
                                         "add recurrence-basis last node");
  passed &= check(graph.addDemand(demand(first, middle, loop, 1)),
                  "add first distance-one basis lemma") &&
            check(graph.addDemand(demand(middle, last, loop, 1)),
                  "add second distance-one basis lemma") &&
            check(graph.addDemand(demand(first, last, loop, 2)),
                  "add implied distance-two obligation") &&
            check(graph.addDemand(demand(first, last, loop, 1)),
                  "add non-implied distance-one obligation") &&
            check(graph.freezeStructure(), "freeze recurrence-basis graph");
  if (!passed) {
    return false;
  }
  const auto admitDistanceOne =
      testProtocolVerifier([](const auto &descriptor) {
        return descriptor.supplies.size() == 1 &&
               descriptor.supplies.front().edge.distance == 1;
      });
  const auto addBasisProtocols = [&](CanonicalSyncPatternProblem &problem) {
    bool added = true;
    added &= check(problem.addEventDomain({0, 1, 2, 8, {}}),
                   "add first recurrence-basis domain");
    added &= check(problem.addEventDomain({1, 2, 3, 8, {}}),
                   "add second recurrence-basis domain");
    added &= check(
        problem.internVerifiedProtocol(
            protocol(0, 1, 2, first, middle, loop, 1, 1), admitDistanceOne),
        "add first recurrence-basis protocol");
    added &= check(
        problem.internVerifiedProtocol(
            protocol(1, 2, 3, middle, last, loop, 1, 1), admitDistanceOne),
        "add second recurrence-basis protocol");
    return added;
  };

  const std::vector<SyncCoverDemandId> exactObligations{0, 1, 2};
  const std::vector<SyncCoverDemandId> basis{0, 1};
  CanonicalSyncPatternProblem exact(graph, exactObligations, basis,
                                    CanonicalSyncPatternProblem::Limits{});
  passed &= addBasisProtocols(exact) &&
            check(exact.freeze(), "freeze exact recurrence-basis problem");
  const CanonicalSyncSelection exactSelection =
      selectCanonicalSyncPatterns(exact);
  passed &= check(exactSelection &&
                      verifyCanonicalSyncSelection(exact, exactSelection),
                  "prove d2 from two proved d1 recurrence lemmas");

  const std::vector<SyncCoverDemandId> wrongDistanceObligations{0, 1, 2, 3};
  CanonicalSyncPatternProblem wrongDistance(
      graph, wrongDistanceObligations, basis,
      CanonicalSyncPatternProblem::Limits{});
  passed &= addBasisProtocols(wrongDistance) &&
            check(wrongDistance.freeze(),
                  "freeze wrong-distance recurrence-basis problem");
  const CanonicalSyncSelection wrongSelection =
      selectCanonicalSyncPatterns(wrongDistance);
  const CanonicalSyncVerifiedPlan rejected =
      verifyCanonicalSyncSelection(wrongDistance, wrongSelection);
  return passed &&
         check(wrongSelection && !rejected &&
                   rejected.firstUncoveredDemand ==
                       std::optional<SyncCoverDemandId>(3),
               "do not use d1 plus d1 to prove a distance-one obligation");
}

bool testSourceLocalBarrierCoversFanout() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 2, 0, 0), passed, "add source-local source");
  std::vector<SyncCoverNodeId> targets;
  std::vector<std::pair<SyncCoverNodeId, SyncCoverDemandId>> bindings;
  for (std::size_t index = 0; index < 4; ++index) {
    const std::uint32_t targetResource = index % 2 == 0 ? 2 : 3;
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(targetResource, 1, 0, index + 1), passed,
                  "add source-local fanout target");
    const SyncCoverDemandId demandId =
        takeIndex(graph.addDemand(demand(source, target)), passed,
                  "add source-local fanout demand");
    targets.push_back(target);
    bindings.push_back({target, demandId});
  }
  passed &= check(graph.setBlockingTargetedBarrierResources({1}),
                  "enable source-local pipe drain") &&
            check(graph.freezeStructure(), "freeze source-local graph");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  for (std::size_t index = 0; index < targets.size(); ++index) {
    passed &= check(problem.internMechanism(
                        targetLocalPipeDrain(1, source, targets[index], index)),
                    "add target-local fanout alternative");
  }
  const CanonicalSyncMechanismId sourceLocal = takeIndex(
      problem.internMechanism(sourceLocalBarrier(1, source, bindings)), passed,
      "add one source-local fanout barrier");
  passed &= check(problem.freeze(), "freeze source-local fanout problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return passed &&
         check(selection &&
                   selection.mechanisms ==
                       std::vector<CanonicalSyncMechanismId>{sourceLocal},
               "select one source-local barrier over four target barriers") &&
         check(verifyCanonicalSyncSelection(problem, selection),
               "freshly verify every source-local fanout row");
}

bool testSourceLocalBarrierReconcilesReducedBasis() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add reduced source");
  const SyncCoverNodeId middle =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add reduced middle");
  const SyncCoverNodeId last =
      takeIndex(graph.addNode(2, 1, 0, 2), passed, "add reduced last");
  passed &=
      check(graph.addDemand(demand(first, middle)),
            "add reduced source demand") &&
      check(graph.addDemand(demand(middle, last)), "add reduced tail demand") &&
      check(graph.addDemand(demand(first, last)),
            "add reduced transitive obligation") &&
      check(graph.setBlockingTargetedBarrierResources({1, 2}),
            "enable reduced source drain") &&
      check(graph.freezeStructure(), "freeze reduced source graph");
  if (!passed) {
    return false;
  }
  const std::vector<SyncCoverDemandId> obligations{0, 1, 2};
  CanonicalSyncPatternProblem problem(graph, obligations,
                                      std::vector<SyncCoverDemandId>{0, 1},
                                      CanonicalSyncPatternProblem::Limits{});
  const CanonicalSyncMechanismId sourceLocal =
      takeIndex(problem.internMechanism(
                    sourceLocalBarrier(1, first, {{middle, 0}, {last, 2}})),
                passed, "add reduced-basis source drain");
  const CanonicalSyncMechanismId tail =
      takeIndex(problem.internMechanism(targetedBarrier(2, middle, last)),
                passed, "add reduced-basis tail barrier");
  passed &= check(problem.freeze(), "freeze reduced source problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(problem, selection);
  return passed &&
         check(selection &&
                   selection.mechanisms ==
                       std::vector<CanonicalSyncMechanismId>{sourceLocal, tail},
               "select the source drain on the reduced universe") &&
         check(verified,
               "freshly verify the omitted transitive source obligation");
}

bool testSourceLocalBarrierRejectsUnsupportedAndSameMacro() {
  bool passed = true;
  SyncCoverGraph unsupported;
  const SyncCoverNodeId unsupportedSource = takeIndex(
      unsupported.addNode(1, 1, 0, 0), passed, "add unsupported source");
  const SyncCoverNodeId unsupportedTarget = takeIndex(
      unsupported.addNode(2, 1, 0, 1), passed, "add unsupported target");
  passed &=
      check(unsupported.addDemand(demand(unsupportedSource, unsupportedTarget)),
            "add unsupported demand") &&
      check(unsupported.setBlockingTargetedBarrierResources({2}),
            "exclude unsupported source pipe") &&
      check(unsupported.freezeStructure(), "freeze unsupported source graph");
  CanonicalSyncPatternProblem unsupportedProblem(unsupported,
                                                 allDemands(unsupported));
  passed &= check(unsupportedProblem
                          .internMechanism(sourceLocalBarrier(
                              1, unsupportedSource, {{unsupportedTarget, 0}}))
                          .error == CanonicalSyncProblemError::InvalidMechanism,
                  "reject a source drain without the blocking capability");

  SyncCoverGraph macro;
  const SyncCoverNodeId entry =
      takeIndex(macro.addNode(1, 1, 0, 0), passed, "add macro entry");
  const SyncCoverNodeId phase = takeIndex(
      macro.addNode(2, 1, 0, 1, {}, {}, entry), passed, "add macro phase");
  passed &=
      check(macro.setPhysicalExit(entry, phase), "set macro entry exit") &&
      check(macro.setPhysicalExit(phase, phase), "set macro phase exit") &&
      check(macro.addDemand(demand(entry, phase)), "add same-macro demand") &&
      check(macro.setBlockingTargetedBarrierResources({1}),
            "enable macro source pipe") &&
      check(macro.freezeStructure(), "freeze same-macro graph");
  CanonicalSyncPatternProblem macroProblem(macro, allDemands(macro));
  return passed &&
         check(macroProblem
                       .internMechanism(
                           sourceLocalBarrier(1, entry, {{phase, 0}}))
                       .error == CanonicalSyncProblemError::InvalidMechanism,
               "reject a source barrier placed after its same macro target");
}

bool testSourcePrefixBarrierConsolidatesIssuedSources() {
  bool passed = true;
  SyncCoverGraph graph;
  std::vector<SyncCoverNodeId> sources;
  for (std::size_t index = 0; index < 4; ++index) {
    sources.push_back(
        takeIndex(graph.addNode(1, 1, 0, index), passed, "add prefix source"));
  }
  std::vector<std::pair<SyncCoverNodeId, SyncCoverDemandId>> bindings;
  for (std::size_t index = 0; index < sources.size(); ++index) {
    const SyncCoverNodeId target =
        takeIndex(graph.addNode(2, 1, 0, sources.size() + index), passed,
                  "add prefix target");
    const SyncCoverDemandId demandId =
        takeIndex(graph.addDemand(demand(sources[index], target)), passed,
                  "add prefix demand");
    bindings.push_back({target, demandId});
  }
  passed &= check(graph.setBlockingTargetedBarrierResources({1}),
                  "enable prefix source drain") &&
            check(graph.setBlockingTargetedBarrierPrefix(
                      1, sources.back(), {sources[0], sources[1], sources[2]}),
                  "certify the issued source prefix") &&
            check(graph.freezeStructure(), "freeze source-prefix graph");
  if (!passed) {
    return false;
  }
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  for (const auto &[target, demandId] : bindings) {
    const SyncCoverNodeId source = graph.getDemands()[demandId].source;
    passed &= check(problem.internMechanism(
                        sourceLocalBarrier(1, source, {{target, demandId}})),
                    "add source-local prefix alternative");
  }
  const CanonicalSyncMechanismId prefix =
      takeIndex(problem.internMechanism(
                    sourcePrefixPipeDrain(graph, 1, sources.back(), bindings)),
                passed, "add one source-prefix barrier");
  passed &= check(problem.freeze(), "freeze source-prefix problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return passed &&
         check(selection && selection.mechanisms ==
                                std::vector<CanonicalSyncMechanismId>{prefix},
               "select one prefix drain over four source drains") &&
         check(verifyCanonicalSyncSelection(problem, selection),
               "freshly verify every issued-prefix obligation");
}

bool testSourcePrefixFinalVerificationWorkIsBounded() {
  bool passed = true;
  SyncCoverGraph graph;
  std::vector<std::uint32_t> resources;
  std::vector<SyncCoverNodeId> sources;
  std::vector<SyncCoverNodeId> cuts;
  std::vector<std::vector<std::pair<SyncCoverNodeId, SyncCoverDemandId>>>
      bindings;
  passed &= check(graph.setBlockingTargetedBarrierResources({1, 2, 3}),
                  "enable bounded-verification source pipes");
  for (std::uint32_t resource = 1; resource <= 3; ++resource) {
    const SyncCoverNodeId source =
        takeIndex(graph.addNode(resource, 1, 0, (resource - 1) * 2), passed,
                  "add bounded-verification prefix source");
    const SyncCoverNodeId cut =
        takeIndex(graph.addNode(resource, 1, 0, (resource - 1) * 2 + 1), passed,
                  "add bounded-verification prefix cut");
    passed &=
        check(graph.setBlockingTargetedBarrierPrefix(resource, cut, {source}),
              "certify bounded-verification issued prefix");
    resources.push_back(resource);
    sources.push_back(source);
    cuts.push_back(cut);
  }
  for (std::size_t index = 0; index < resources.size(); ++index) {
    const SyncCoverNodeId target = takeIndex(
        graph.addNode(resources[index] + 10, 1, 0, 10 + resources[index]),
        passed, "add bounded-verification prefix target");
    const SyncCoverDemandId sourceDemand =
        takeIndex(graph.addDemand(demand(sources[index], target)), passed,
                  "add bounded-verification prefix demand");
    const SyncCoverDemandId cutDemand =
        takeIndex(graph.addDemand(demand(cuts[index], target)), passed,
                  "add bounded-verification cut demand");
    bindings.push_back({{target, sourceDemand}, {target, cutDemand}});
  }
  passed &= check(graph.freezeStructure(),
                  "freeze bounded-verification source-prefix graph");
  if (!passed) {
    return false;
  }
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  for (std::size_t index = 0; index < resources.size(); ++index) {
    passed &= check(problem.internMechanism(sourcePrefixPipeDrain(
                        graph, resources[index], cuts[index], bindings[index])),
                    "add bounded-verification source-prefix drain");
  }
  passed &= check(problem.freeze(),
                  "freeze bounded-verification source-prefix problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  SyncCoverCoverageWorkBudget referenceWork(1U << 20);
  const CanonicalSyncVerifiedPlan reference =
      verifyCanonicalSyncSelection(problem, selection, &referenceWork);
  if (!check(reference && selection.mechanisms.size() == resources.size(),
             "select and verify several source-prefix drains")) {
    return false;
  }
  SyncCoverCoverageWorkBudget exactWork(referenceWork.workUnits);
  const CanonicalSyncVerifiedPlan exact =
      verifyCanonicalSyncSelection(problem, selection, &exactWork);
  SyncCoverCoverageWorkBudget belowWork(referenceWork.workUnits - 1);
  const CanonicalSyncVerifiedPlan below =
      verifyCanonicalSyncSelection(problem, selection, &belowWork);
  return check(exact && exactWork.workUnits == referenceWork.workUnits,
               "accept source-prefix verification at its exact work bound") &&
         check(below.error == CanonicalSyncSelectionError::WorkLimitExceeded &&
                   belowWork.exhausted,
               "reject source-prefix verification one unit below its bound");
}

bool testDeepScopeLcaVerificationWorkIsBounded() {
  bool passed = true;
  const auto measure =
      [&](std::size_t depth,
          std::string_view label) -> std::optional<std::size_t> {
    SyncCoverGraph graph;
    SyncCoverScopeId sourceScope = 0;
    SyncCoverScopeId targetScope = 0;
    for (std::size_t index = 0; index < depth; ++index) {
      sourceScope = takeIndex(graph.addScope(sourceScope, true), passed,
                              "add deep-LCA source scope");
      targetScope = takeIndex(graph.addScope(targetScope, true), passed,
                              "add deep-LCA target scope");
    }
    const SyncCoverNodeId source = takeIndex(
        graph.addNode(1, 1, sourceScope, 0), passed, "add deep-LCA source");
    const SyncCoverNodeId target = takeIndex(
        graph.addNode(2, 1, targetScope, 1), passed, "add deep-LCA target");
    const SyncCoverDemandId demandId = takeIndex(
        graph.addDemand(demand(source, target)), passed, "add deep-LCA demand");
    passed &= check(graph.setBlockingTargetedBarrierResources({1}),
                    "enable deep-LCA source drain") &&
              check(graph.freezeStructure(), "freeze deep-LCA graph");
    if (!passed) {
      return std::nullopt;
    }
    CanonicalSyncPatternProblem problem(graph, allDemands(graph));
    const CanonicalSyncMechanismId mechanism =
        takeIndex(problem.internMechanism(
                      sourceLocalBarrier(1, source, {{target, demandId}})),
                  passed, "add deep-LCA source drain");
    if (!passed) {
      return std::nullopt;
    }
    SyncCoverCoverageWorkBudget measured;
    const CanonicalSyncProblemResult verified =
        problem.verifyMechanism(mechanism, &measured);
    SyncCoverCoverageWorkBudget exact(measured.workUnits);
    SyncCoverCoverageWorkBudget oneLess(measured.workUnits - 1);
    const bool exactBoundary =
        verified && problem.verifyMechanism(mechanism, &exact) &&
        exact.workUnits == measured.workUnits &&
        problem.verifyMechanism(mechanism, &oneLess).error ==
            CanonicalSyncProblemError::LimitExceeded;
    passed &= check(exactBoundary, label);
    return exactBoundary ? std::optional<std::size_t>(measured.workUnits)
                         : std::nullopt;
  };

  const std::optional<std::size_t> shallow =
      measure(1, "bound shallow source-drain LCA verification exactly");
  const std::optional<std::size_t> deep =
      measure(32, "bound deep source-drain LCA verification exactly");
  return passed &&
         check(shallow && deep && *deep > *shallow,
               "charge linear parent-chain LCA growth in fresh verification");
}

bool testSourcePrefixBarrierRejectsInvalidCertificates() {
  bool passed = true;
  SyncCoverGraph missing;
  const SyncCoverNodeId missingSource = takeIndex(
      missing.addNode(1, 1, 0, 0), passed, "add missing-prefix source");
  const SyncCoverNodeId certifiedSource = takeIndex(
      missing.addNode(1, 1, 0, 1), passed, "add certified prefix source");
  const SyncCoverNodeId missingCut =
      takeIndex(missing.addNode(1, 1, 0, 2), passed, "add missing-prefix cut");
  const SyncCoverNodeId missingTarget = takeIndex(
      missing.addNode(2, 1, 0, 3), passed, "add missing-prefix target");
  passed &= check(missing.addDemand(demand(missingSource, missingTarget)),
                  "add missing-prefix demand") &&
            check(missing.setBlockingTargetedBarrierResources({1}),
                  "enable missing-prefix source pipe") &&
            check(missing.setBlockingTargetedBarrierPrefix(1, missingCut,
                                                           {certifiedSource}),
                  "certify a prefix that omits the demanded source") &&
            check(missing.freezeStructure(), "freeze missing-prefix graph");
  CanonicalSyncPatternProblem missingProblem(missing, allDemands(missing));
  passed &=
      check(missingProblem
                    .internMechanism(sourcePrefixPipeDrain(
                        missing, 1, missingCut, {{missingTarget, 0}}))
                    .error == CanonicalSyncProblemError::InvalidMechanism,
            "reject a source-prefix drain whose certificate omits the source");

  SyncCoverGraph scoped;
  const SyncCoverScopeId child =
      takeIndex(scoped.addScope(0, true, SyncCoverTimelineInterval{2, 3}),
                passed, "add source-prefix child scope");
  const SyncCoverNodeId scopedSource =
      takeIndex(scoped.addNode(1, 1, 0, 0), passed, "add outer prefix source");
  const SyncCoverNodeId scopedCut =
      takeIndex(scoped.addNode(1, 1, child, 1), passed, "add child prefix cut");
  const SyncCoverNodeId scopedTarget =
      takeIndex(scoped.addNode(2, 1, 0, 2), passed, "add outer prefix target");
  passed &= check(scoped.addDemand(demand(scopedSource, scopedTarget)),
                  "add cross-scope prefix demand") &&
            check(scoped.setBlockingTargetedBarrierResources({1}),
                  "enable scoped source pipe") &&
            check(scoped.setBlockingTargetedBarrierPrefix(1, scopedCut,
                                                          {scopedSource}),
                  "certify compatible cross-scope issued history") &&
            check(scoped.freezeStructure(), "freeze scoped prefix graph");
  CanonicalSyncPatternProblem scopedProblem(scoped, allDemands(scoped));
  passed &= check(scopedProblem
                          .internMechanism(sourcePrefixPipeDrain(
                              scoped, 1, scopedCut, {{scopedTarget, 0}}))
                          .error == CanonicalSyncProblemError::InvalidMechanism,
                  "reject a source-prefix drain across distinct action scopes");

  SyncCoverGraph unsupported;
  const SyncCoverNodeId unsupportedSource = takeIndex(
      unsupported.addNode(1, 1, 0, 0), passed, "add unsupported prefix source");
  const SyncCoverNodeId unsupportedTarget = takeIndex(
      unsupported.addNode(2, 1, 0, 1), passed, "add unsupported prefix target");
  passed &=
      check(unsupported.addDemand(demand(unsupportedSource, unsupportedTarget)),
            "add unsupported prefix demand") &&
      check(unsupported.setBlockingTargetedBarrierResources({2}),
            "exclude unsupported prefix source pipe") &&
      check(unsupported.freezeStructure(), "freeze unsupported prefix graph");
  CanonicalSyncPatternProblem unsupportedProblem(unsupported,
                                                 allDemands(unsupported));
  return passed &&
         check(unsupportedProblem
                       .internMechanism(sourcePrefixPipeDrain(
                           unsupported, 1, unsupportedSource,
                           {{unsupportedTarget, 0}}))
                       .error == CanonicalSyncProblemError::InvalidMechanism,
               "reject a source-prefix drain without blocking capability");
}

bool testSourcePrefixBarrierKeepsDistanceQualifiers() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true),
                passed, "add mixed-distance prefix loop");
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, loop, 1), passed,
                                           "add mixed-distance source");
  const SyncCoverNodeId cut =
      takeIndex(graph.addNode(1, 1, loop, 2, {}, {2}), passed,
                "add mixed-distance completion-capable cut");
  const SyncCoverNodeId target = takeIndex(graph.addNode(2, 1, loop, 3), passed,
                                           "add mixed-distance target");
  const SyncCoverDemandId sameIteration =
      takeIndex(graph.addDemand(demand(source, target, loop, 0)), passed,
                "add distance-zero prefix demand");
  const SyncCoverDemandId nextIteration =
      takeIndex(graph.addDemand(demand(source, target, loop, 1)), passed,
                "add distance-one prefix demand");
  passed &=
      check(graph.setBlockingTargetedBarrierResources({1}),
            "enable mixed-distance source pipe") &&
      check(graph.setBlockingTargetedBarrierPrefix(1, cut, {source}),
            "certify mixed-distance issued prefix") &&
      check(graph.freezeStructure(), "freeze mixed-distance prefix graph");
  if (!passed) {
    return false;
  }
  CanonicalSyncPatternProblem sameOnly(graph, allDemands(graph));
  passed &= check(sameOnly.internMechanism(sourcePrefixPipeDrain(
                      graph, 1, cut, {{target, sameIteration}})),
                  "admit the distance-zero source-prefix qualifier");
  CanonicalSyncPatternProblem nextOnly(graph, allDemands(graph));
  passed &= check(nextOnly.internMechanism(sourcePrefixPipeDrain(
                      graph, 1, cut, {{target, nextIteration}})),
                  "admit the distance-one source-prefix qualifier");
  CanonicalSyncPatternProblem problem(graph, allDemands(graph));
  const CanonicalSyncMechanismId prefix = takeIndex(
      problem.internMechanism(sourcePrefixPipeDrain(
          graph, 1, cut, {{target, sameIteration}, {target, nextIteration}})),
      passed, "add mixed-distance source-prefix barrier");
  passed &= check(problem.freeze(), "freeze mixed-distance prefix problem");
  const CanonicalSyncSelection selection = selectCanonicalSyncPatterns(problem);
  return passed &&
         check(selection && selection.mechanisms ==
                                std::vector<CanonicalSyncMechanismId>{prefix},
               "select one action with independent distance qualifiers") &&
         check(verifyCanonicalSyncSelection(problem, selection),
               "freshly verify both source-prefix distance classes");
}

bool testMechanismOriginInterningIsBoundedAndDiagnosticOnly() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(graph.addNode(1, 1, 0, 0, {}, {2}),
                                           passed, "add provenance source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add provenance target");
  passed &=
      check(graph.addDemand(demand(source, target)), "add provenance demand") &&
      check(graph.freezeStructure(), "freeze provenance graph");
  if (!passed) {
    return false;
  }

  CanonicalSyncPatternProblem merged(graph, allDemands(graph));
  passed &= check(merged.addEventDomain({0, 1, 2, 8, {}}),
                  "add merged provenance domain");
  const CanonicalSyncProblemResult direct = merged.internMechanism(
      event(0, 1, 2, source, target),
      CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent);
  const CanonicalSyncProblemResult frontier = merged.internMechanism(
      event(0, 1, 2, source, target),
      CanonicalSyncMechanismOrigin::CompletionFrontierEvent);
  const CanonicalSyncMechanismOriginMask expectedMask =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::CompletionFrontierEvent);
  passed &= check(direct && frontier && direct.index == frontier.index &&
                      merged.getMechanisms().size() == 1 &&
                      merged.getMechanisms().front().originMask == expectedMask,
                  "merge explicit origins on one interned descriptor");

  const CanonicalSyncProblemResult invalid = merged.internMechanism(
      event(0, 1, 2, source, target), static_cast<CanonicalSyncMechanismOrigin>(
                                          kCanonicalSyncMechanismOriginCount));
  passed &= check(invalid.error == CanonicalSyncProblemError::InvalidMechanism,
                  "reject an out-of-range mechanism origin");

  CanonicalSyncPatternProblem directOnly(graph, allDemands(graph));
  CanonicalSyncPatternProblem frontierOnly(graph, allDemands(graph));
  passed &= check(directOnly.addEventDomain({0, 1, 2, 8, {}}),
                  "add direct-only provenance domain") &&
            check(frontierOnly.addEventDomain({0, 1, 2, 8, {}}),
                  "add frontier-only provenance domain");
  const CanonicalSyncMechanismId directId =
      takeIndex(directOnly.internMechanism(
                    event(0, 1, 2, source, target),
                    CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent),
                passed, "add direct-only provenance mechanism");
  const CanonicalSyncMechanismId frontierId =
      takeIndex(frontierOnly.internMechanism(
                    event(0, 1, 2, source, target),
                    CanonicalSyncMechanismOrigin::CompletionFrontierEvent),
                passed, "add frontier-only provenance mechanism");
  if (!passed) {
    return false;
  }
  const CanonicalSyncMechanism &directMechanism =
      directOnly.getMechanisms()[directId];
  const CanonicalSyncMechanism &frontierMechanism =
      frontierOnly.getMechanisms()[frontierId];
  return check(directOnly.getMechanismSignature(directId) ==
                   frontierOnly.getMechanismSignature(frontierId),
               "exclude provenance from the recipe signature") &&
         check(directMechanism.cost.barrierActions ==
                       frontierMechanism.cost.barrierActions &&
                   directMechanism.cost.eventActions ==
                       frontierMechanism.cost.eventActions &&
                   directMechanism.cost.serializationBreadth ==
                       frontierMechanism.cost.serializationBreadth,
               "exclude provenance from structural cost") &&
         check(!directOnly.hasSameCandidatePrefix(frontierOnly),
               "include provenance in precise-repair prefix comparison");
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
      testFixedCoverUsesFrozenCompositeColumn() &&
      testStreamingGreedySelectionIsDeterministic() &&
      testDirectPairSkipsConflictingMechanisms() &&
      testConflictingPairDoesNotConsumeProposalCapacity() &&
      testDirectPairTraversesFixedCompletionSupply() &&
      testDirectPairIndexesUnrestrictedBindingOfMixedMechanism() &&
      testConnectorIndexRetainsDistinctEndpointNodes() &&
      testPairPreparationLimitKeepsSingletonCorrectness() &&
      testPairOwnerUsesEverySupplyScope() && testPairOwnerExact4096Boundary() &&
      testOwnerPairBatchesTruncateAtomicallyAndContinue() &&
      testOwnerPairCoverageWordLimitIsAtomic() &&
      testSiblingAndBarrierPairsComposeAtTheirLca() &&
      testNestedPairExtendsToParentDemand() &&
      testDirectPairComposesAcrossRecurrenceArena() &&
      testStructuralCostSeparatesBarrierAndEventActions() &&
      testStructuralCostOverflowFailsClosed() && testPipeAllFallbackProblem() &&
      testPackagingPatternHasNoExtraCoverage() &&
      testSeparateFallbackRepairsEventPressure() &&
      testOptionalPipelineFallback() && testReservationsAndFinalValidation() &&
      testAllocatorWidthsReuseAndConflicts() &&
      testVerifiedProtocolTrustBoundary() &&
      testProtocolVerifierStorageSharesOpaqueCapture() &&
      testRepairProtocolAdmissionUsesSharedBudget() &&
      testHierarchicalProtocolLifetime() &&
      testFreezeRetryCommitsFreshDerivedState() &&
      testOptionalPairReservesSingletonIncidences() &&
      testSelectionBasisRetainsFullObligationVerification() &&
      testRecurrenceBasisLemmasPreserveExactDistance() &&
      testSourceLocalBarrierCoversFanout() &&
      testSourceLocalBarrierReconcilesReducedBasis() &&
      testSourceLocalBarrierRejectsUnsupportedAndSameMacro() &&
      testSourcePrefixBarrierConsolidatesIssuedSources() &&
      testSourcePrefixFinalVerificationWorkIsBounded() &&
      testDeepScopeLcaVerificationWorkIsBounded() &&
      testSourcePrefixBarrierRejectsInvalidCertificates() &&
      testSourcePrefixBarrierKeepsDistanceQualifiers() &&
      testMechanismOriginInterningIsBoundedAndDiagnosticOnly() &&
      testFailClosedConstruction();
  return passed ? 0 : 1;
}
