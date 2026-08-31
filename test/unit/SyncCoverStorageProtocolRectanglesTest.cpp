// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolRectangles.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolRectanglesTest failure: " << message
              << '\n';
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

SyncCoverDemandId addDemand(SyncCoverGraph &graph, SyncCoverNodeId source,
                            SyncCoverNodeId target, SyncCoverScopeId scope,
                            unsigned distance, SyncCoverDemandKind kind,
                            SyncCoverStorageAccessId sourceAccess,
                            SyncCoverStorageAccessId targetAccess,
                            bool &passed) {
  const SyncCoverStorageWitnessId witness =
      takeIndex(graph.addStorageWitness(sourceAccess, targetAccess), passed,
                "add protocol-rectangle witness");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  return takeIndex(graph.addDemand(std::move(demand)), passed,
                   "add protocol-rectangle demand");
}

struct RectangleInputs {
  SyncCoverGraph graph;
  SyncCoverStorageLifecycleIndex lifecycle;
  SyncCoverStorageProtocolAutomatonIndex automata;
  SyncCoverStorageProtocolFrontierIndex frontiers;
  bool valid = false;
};

RectangleInputs buildPrefixCertificateProtocol(bool addCompletionFact = false) {
  RectangleInputs result;
  bool passed = true;
  constexpr std::uint32_t kProducerResource = 1;
  constexpr std::uint32_t kConsumerResource = 2;
  constexpr std::uint32_t kUnusedResource = 3;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
      passed, "add protocol-rectangle loop");
  const SyncCoverStorageDomainId domain = takeIndex(
      result.graph.addStorageDomain(SyncCoverStorageDomainRole::L0Left), passed,
      "add protocol-rectangle domain");
  const SyncCoverNodeId firstProducer =
      takeIndex(result.graph.addNode(kProducerResource, 1, loop, 2, {},
                                     {kConsumerResource}),
                passed, "add first protocol-rectangle producer");
  const SyncCoverNodeId secondProducer = takeIndex(
      result.graph.addNode(
          kProducerResource, 1, loop, 3, {}, {kConsumerResource},
          addCompletionFact ? std::optional<SyncCoverNodeId>(firstProducer)
                            : std::nullopt,
          true),
      passed, "add second protocol-rectangle producer");
  if (addCompletionFact) {
    passed &= check(result.graph.setPhysicalExit(firstProducer, secondProducer),
                    "set multi-demand protocol-rectangle physical exit");
  }
  const SyncCoverNodeId consumer =
      takeIndex(result.graph.addNode(kConsumerResource, 1, loop, 6, {},
                                     {kProducerResource}),
                passed, "add protocol-rectangle consumer");
  passed &= check(
      result.graph.addEdge({firstProducer,
                            secondProducer,
                            SyncCoverEdgeKind::CertifiedCompletionFrontier,
                            loop,
                            0,
                            {},
                            {}}),
      "add certified producer completion frontier");
  passed &=
      check(result.graph.addCompletionDominance(firstProducer, secondProducer),
            "add producer completion dominance");
  const SyncCoverStorageAccessId firstWrite =
      takeIndex(result.graph.addStorageAccess(firstProducer, domain, 9, {0, 64},
                                              SyncCoverStorageAccessMode::Write,
                                              std::nullopt, true),
                passed, "add first protocol-rectangle write");
  const SyncCoverStorageAccessId secondWrite =
      takeIndex(result.graph.addStorageAccess(
                    secondProducer, domain, 9, {0, 64},
                    SyncCoverStorageAccessMode::Write, std::nullopt, true),
                passed, "add second protocol-rectangle write");
  const SyncCoverStorageAccessId read =
      takeIndex(result.graph.addStorageAccess(consumer, domain, 9, {0, 64},
                                              SyncCoverStorageAccessMode::Read,
                                              std::nullopt, true),
                passed, "add protocol-rectangle read");
  const SyncCoverDemandId firstReady =
      addDemand(result.graph, firstProducer, consumer, loop, 0,
                SyncCoverDemandKind::MemoryRAW, firstWrite, read, passed);
  const SyncCoverDemandId secondReady =
      addDemand(result.graph, secondProducer, consumer, loop, 0,
                SyncCoverDemandKind::MemoryRAW, secondWrite, read, passed);
  (void)addDemand(result.graph, consumer, firstProducer, loop, 1,
                  SyncCoverDemandKind::MemoryWAR, read, firstWrite, passed);
  (void)addDemand(result.graph, consumer, secondProducer, loop, 1,
                  SyncCoverDemandKind::MemoryWAR, read, secondWrite, passed);
  passed &= check(result.graph.setTargetCompletionResources(
                      {kProducerResource, kConsumerResource, kUnusedResource}),
                  "set protocol-rectangle target resources");
  if (!addCompletionFact) {
    passed &= check(result.graph.addTargetCompletionCertificate(
                        SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix,
                        firstProducer, consumer, kProducerResource,
                        kConsumerResource, {domain}, {firstReady}),
                    "add first protocol-rectangle certificate");
    passed &= check(result.graph.addTargetCompletionCertificate(
                        SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix,
                        secondProducer, consumer, kProducerResource,
                        kConsumerResource, {domain}, {firstReady, secondReady}),
                    "add prefix protocol-rectangle certificate");
  }
  if (addCompletionFact) {
    passed &= check(result.graph.addCompletionCutFact(
                        secondProducer, kProducerResource, kConsumerResource,
                        {domain}, {firstReady, secondReady}),
                    "add multi-demand protocol-rectangle completion fact");
  }
  passed &=
      check(result.graph.freezeStructure(), "freeze protocol-rectangle graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  const SyncCoverStorageProtocolGroupIndex groups =
      buildSyncCoverStorageProtocolGroupIndex(result.graph, result.lifecycle,
                                              seeds);
  result.automata = buildSyncCoverStorageProtocolAutomatonIndex(
      result.graph, result.lifecycle, seeds, groups);
  result.frontiers = buildSyncCoverStorageProtocolFrontierIndex(
      result.graph, result.lifecycle, result.automata);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.frontiers.isComplete();
  return result;
}

RectangleInputs buildPartialDomainFactProtocol() {
  RectangleInputs result;
  bool passed = true;
  constexpr std::uint32_t kProducerResource = 1;
  constexpr std::uint32_t kConsumerResource = 2;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
      passed, "add partial-fact loop");
  const SyncCoverControlId control = takeIndex(
      result.graph.addControl(2, loop), passed, "add partial-fact control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
      "set partial-fact phase relation");
  const SyncCoverGuard producerGuard{{{control, 0}}};
  const SyncCoverScopeId producerScope = takeIndex(
      result.graph.addScope(loop, false, SyncCoverTimelineInterval{0, 16},
                            false, producerGuard),
      passed, "add partial-fact producer scope");
  const SyncCoverNodeId producer =
      takeIndex(result.graph.addNode(kProducerResource, 1, producerScope, 2,
                                     producerGuard),
                passed, "add partial-fact producer");
  const SyncCoverNodeId consumer =
      takeIndex(result.graph.addNode(kConsumerResource, 1, loop, 4, {},
                                     {kProducerResource}),
                passed, "add partial-fact consumer");
  const SyncCoverStorageDomainId admittedDomain =
      takeIndex(result.graph.addStorageDomain(), passed,
                "add admitted partial-fact domain");
  const SyncCoverStorageDomainId otherDomain = takeIndex(
      result.graph.addStorageDomain(), passed, "add other partial-fact domain");
  const auto addAccess = [&](SyncCoverNodeId node,
                             SyncCoverStorageDomainId domain,
                             SyncCoverStorageAccessMode mode,
                             std::string_view label) {
    return takeIndex(result.graph.addStorageAccess(node, domain, 7, {0, 64},
                                                   mode, std::nullopt, true),
                     passed, label);
  };
  const SyncCoverStorageAccessId admittedWrite =
      addAccess(producer, admittedDomain, SyncCoverStorageAccessMode::Write,
                "add admitted partial-fact write");
  const SyncCoverStorageAccessId admittedRead =
      addAccess(consumer, admittedDomain, SyncCoverStorageAccessMode::Read,
                "add admitted partial-fact read");
  const SyncCoverStorageAccessId otherWrite =
      addAccess(producer, otherDomain, SyncCoverStorageAccessMode::Write,
                "add other partial-fact write");
  const SyncCoverStorageAccessId otherRead =
      addAccess(consumer, otherDomain, SyncCoverStorageAccessMode::Read,
                "add other partial-fact read");
  const SyncCoverStorageWitnessId admittedReady =
      takeIndex(result.graph.addStorageWitness(admittedWrite, admittedRead),
                passed, "add admitted partial-fact ready witness");
  const SyncCoverStorageWitnessId otherReady =
      takeIndex(result.graph.addStorageWitness(otherWrite, otherRead), passed,
                "add other partial-fact ready witness");
  SyncCoverDemand ready;
  ready.source = producer;
  ready.target = consumer;
  ready.scope = loop;
  ready.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  ready.storageWitnesses = {admittedReady, otherReady};
  const SyncCoverDemandId readyDemand =
      takeIndex(result.graph.addDemand(std::move(ready)), passed,
                "add partial-domain ready demand");
  const SyncCoverStorageWitnessId admittedReuse =
      takeIndex(result.graph.addStorageWitness(admittedRead, admittedWrite),
                passed, "add admitted partial-fact reuse witness");
  const SyncCoverStorageWitnessId otherReuse =
      takeIndex(result.graph.addStorageWitness(otherRead, otherWrite), passed,
                "add other partial-fact reuse witness");
  SyncCoverDemand reuse;
  reuse.source = consumer;
  reuse.target = producer;
  reuse.scope = loop;
  reuse.distance = 1;
  reuse.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
  reuse.storageWitnesses = {admittedReuse, otherReuse};
  passed &= check(result.graph.addDemand(std::move(reuse)),
                  "add partial-domain reuse demand");
  passed &= check(result.graph.addCompletionCutFact(
                      producer, kProducerResource, kConsumerResource,
                      {admittedDomain}, {readyDemand}),
                  "add partial-domain completion-cut fact");
  passed &=
      check(result.graph.freezeStructure(), "freeze partial-domain fact graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  const SyncCoverStorageProtocolGroupIndex groups =
      buildSyncCoverStorageProtocolGroupIndex(result.graph, result.lifecycle,
                                              seeds);
  result.automata = buildSyncCoverStorageProtocolAutomatonIndex(
      result.graph, result.lifecycle, seeds, groups);
  result.frontiers = buildSyncCoverStorageProtocolFrontierIndex(
      result.graph, result.lifecycle, result.automata);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.frontiers.isComplete();
  return result;
}

SyncCoverStorageProtocolRectangleLimits exactRectangleLimits(
    const SyncCoverStorageProtocolRectangleStatistics &statistics) {
  SyncCoverStorageProtocolRectangleLimits limits;
  limits.maximumWorkUnits = statistics.workUnits;
  limits.maximumFrontierInspections = statistics.frontierInspections;
  limits.maximumRectangles = statistics.rectangles;
  limits.maximumFrontierIncidences = statistics.frontierIncidences;
  return limits;
}

bool isTransactionalLimit(const SyncCoverStorageProtocolRectangleIndex &index,
                          std::string_view message) {
  return check(index.getError() ==
                       SyncCoverStorageProtocolRectangleError::LimitExceeded &&
                   index.getRectangles().empty() &&
                   index.getFrontierIncidences().empty() &&
                   index.getStatistics().truncated,
               message);
}

bool testCompactRectanglesAndGrounding() {
  RectangleInputs inputs = buildPrefixCertificateProtocol();
  if (!check(inputs.valid, "build protocol-rectangle inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolRectangleIndex rectangles =
      buildSyncCoverStorageProtocolRectangleIndex(inputs.graph, inputs.automata,
                                                  inputs.frontiers);
  const SyncCoverStorageProtocolRectangleStatistics &statistics =
      rectangles.getStatistics();
  bool passed = check(
      rectangles.isComplete() && statistics.plans == 1 &&
          statistics.frontierInspections ==
              inputs.frontiers.getFrontiers().size() &&
          statistics.rectangles < statistics.frontierInspections &&
          statistics.mergedRectangles == 2 &&
          statistics.maximumRectangleFrontiers == 2 &&
          statistics.frontierIncidences == statistics.frontierInspections,
      "factor direct and certificate alternatives at equal endpoint cuts");
  if (!passed) {
    return false;
  }

  std::vector<SyncCoverDemandId> demands(inputs.graph.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < demands.size(); ++demand) {
    demands[demand] = demand;
  }
  const SyncCoverExpandedProgram expansion(inputs.graph, demands);
  const SyncCoverStorageProtocolRectangleGrounding grounding =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands);
  const SyncCoverStorageProtocolRectangleGroundingStatistics
      &groundingStatistics = grounding.getStatistics();
  passed &= check(
      grounding.isComplete() &&
          groundingStatistics.evaluatedRectangles == statistics.rectangles &&
          groundingStatistics.rectanglesWithCoverage != 0 &&
          groundingStatistics.rectanglesCoveringMultipleRows != 0 &&
          groundingStatistics.maximumCoverageRows == 2 &&
          groundingStatistics.admittedDemandIncidences != 0,
      "ground a certified prefix rectangle over two ready obligations");
  passed &=
      check(!grounding.getDetails().empty() &&
                grounding.getDetails().front().coverageRows == 2 &&
                grounding.getDetails().front().frontierCount == 2,
            "retain compact rectangle provenance in ranked grounding details");

  SyncCoverStorageProtocolRectangleLimits rectangleLimits =
      exactRectangleLimits(statistics);
  passed &= check(
      buildSyncCoverStorageProtocolRectangleIndex(
          inputs.graph, inputs.automata, inputs.frontiers, rectangleLimits)
          .isComplete(),
      "accept protocol rectangles at every exact bound");
  --rectangleLimits.maximumWorkUnits;
  passed &= isTransactionalLimit(
      buildSyncCoverStorageProtocolRectangleIndex(
          inputs.graph, inputs.automata, inputs.frontiers, rectangleLimits),
      "enforce protocol-rectangle work transactionally");
  rectangleLimits = exactRectangleLimits(statistics);
  --rectangleLimits.maximumRectangles;
  passed &= isTransactionalLimit(
      buildSyncCoverStorageProtocolRectangleIndex(
          inputs.graph, inputs.automata, inputs.frontiers, rectangleLimits),
      "enforce the protocol-rectangle count transactionally");
  rectangleLimits = exactRectangleLimits(statistics);
  --rectangleLimits.maximumFrontierInspections;
  passed &= isTransactionalLimit(
      buildSyncCoverStorageProtocolRectangleIndex(
          inputs.graph, inputs.automata, inputs.frontiers, rectangleLimits),
      "enforce frontier inspection bounds transactionally");
  rectangleLimits = exactRectangleLimits(statistics);
  --rectangleLimits.maximumFrontierIncidences;
  passed &= isTransactionalLimit(
      buildSyncCoverStorageProtocolRectangleIndex(
          inputs.graph, inputs.automata, inputs.frontiers, rectangleLimits),
      "enforce frontier incidence bounds transactionally");

  SyncCoverStorageProtocolRectangleGroundingLimits groundingLimits;
  groundingLimits.maximumWorkUnits = groundingStatistics.workUnits;
  groundingLimits.maximumAdmittedDemandIncidences =
      groundingStatistics.admittedDemandIncidences;
  groundingLimits.maximumDetails = grounding.getDetails().size();
  passed &= check(groundSyncCoverStorageProtocolRectangles(
                      inputs.graph, expansion, inputs.automata,
                      inputs.frontiers, rectangles, demands, groundingLimits)
                      .isComplete(),
                  "accept protocol grounding at every exact bound");
  --groundingLimits.maximumWorkUnits;
  const SyncCoverStorageProtocolRectangleGrounding workLimited =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands, groundingLimits);
  passed &= check(workLimited.getError() ==
                          SyncCoverStorageProtocolRectangleGroundingError::
                              WorkLimitExceeded &&
                      workLimited.getStatistics().truncated,
                  "bound batched protocol grounding work");
  groundingLimits.maximumWorkUnits = groundingStatistics.workUnits;
  --groundingLimits.maximumAdmittedDemandIncidences;
  const SyncCoverStorageProtocolRectangleGrounding incidenceLimited =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands, groundingLimits);
  passed &= check(incidenceLimited.getError() ==
                          SyncCoverStorageProtocolRectangleGroundingError::
                              IncidenceLimitExceeded &&
                      incidenceLimited.getStatistics().truncated,
                  "bound protocol grounding demand incidences");

  SyncCoverStorageProtocolRectangleGroundingLimits singleBatchLimits;
  singleBatchLimits.maximumBatchRectangles = 1;
  const SyncCoverStorageProtocolRectangleGrounding singleBatches =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands, singleBatchLimits);
  passed &= check(singleBatches.isComplete() &&
                      singleBatches.getStatistics().coverageBatches ==
                          statistics.rectangles,
                  "ground bounded single-rectangle batches deterministically");
  singleBatchLimits.maximumCoverageBatches = statistics.rectangles - 1;
  const SyncCoverStorageProtocolRectangleGrounding batchLimited =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands, singleBatchLimits);
  passed &= check(batchLimited.getError() ==
                          SyncCoverStorageProtocolRectangleGroundingError::
                              CoverageLimitExceeded &&
                      batchLimited.getStatistics().truncated,
                  "bound the number of protocol-grounding batches");

  SyncCoverStorageProtocolRectangleGroundingLimits memoryLimited;
  memoryLimited.coverageLimits.maximumTotalWords = 1;
  const SyncCoverStorageProtocolRectangleGrounding coverageLimited =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands, memoryLimited);
  passed &= check(coverageLimited.getError() ==
                          SyncCoverStorageProtocolRectangleGroundingError::
                              CoverageLimitExceeded &&
                      coverageLimited.getStatistics().truncated,
                  "bound batched protocol-grounding coverage memory");
  return passed;
}

bool testProviderFactDoesNotCoverPartialDomainRow() {
  RectangleInputs inputs = buildPartialDomainFactProtocol();
  if (!check(inputs.valid, "build partial-domain fact inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolRectangleIndex rectangles =
      buildSyncCoverStorageProtocolRectangleIndex(inputs.graph, inputs.automata,
                                                  inputs.frontiers);
  const bool rectanglesAvailable =
      rectangles.isComplete() && !rectangles.getRectangles().empty();
  if (!check(rectanglesAvailable, "build partial-domain fact rectangles")) {
    return false;
  }
  bool foundFactRectangle = false;
  for (const SyncCoverStorageProtocolRectangle &rectangle :
       rectangles.getRectangles()) {
    for (std::size_t offset = 0; offset < rectangle.frontierCount; ++offset) {
      const SyncCoverStorageProtocolFrontierId frontier =
          rectangles.getFrontierIncidences()[rectangle.frontierBegin + offset];
      foundFactRectangle |= inputs.frontiers.getFrontiers()[frontier]
                                .completionCutFact.has_value();
    }
  }
  std::vector<SyncCoverDemandId> demands(inputs.graph.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < demands.size(); ++demand) {
    demands[demand] = demand;
  }
  const SyncCoverExpandedProgram expansion(inputs.graph, demands);
  const SyncCoverStorageProtocolRectangleGrounding grounding =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands);
  return check(
      foundFactRectangle && grounding.isComplete() &&
          grounding.getStatistics().evaluatedRectangles ==
              rectangles.getRectangles().size() &&
          grounding.getStatistics().rectanglesWithCoverage <
              rectangles.getRectangles().size(),
      "do not credit a fact that admits only one witness of a canonical row");
}

bool testProviderFactRetainsAllAuthorizedDemands() {
  RectangleInputs inputs = buildPrefixCertificateProtocol(true);
  if (!check(inputs.valid, "build multi-demand fact inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolRectangleIndex rectangles =
      buildSyncCoverStorageProtocolRectangleIndex(inputs.graph, inputs.automata,
                                                  inputs.frontiers);
  if (!check(rectangles.isComplete(), "build multi-demand fact rectangles")) {
    return false;
  }
  std::vector<bool> factRectangles(rectangles.getRectangles().size(), false);
  for (const SyncCoverStorageProtocolRectangle &rectangle :
       rectangles.getRectangles()) {
    for (std::size_t offset = 0; offset < rectangle.frontierCount; ++offset) {
      const SyncCoverStorageProtocolFrontierId frontier =
          rectangles.getFrontierIncidences()[rectangle.frontierBegin + offset];
      factRectangles[rectangle.id] = factRectangles[rectangle.id] ||
                                     inputs.frontiers.getFrontiers()[frontier]
                                         .completionCutFact.has_value();
    }
  }
  std::vector<SyncCoverDemandId> demands(inputs.graph.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < demands.size(); ++demand) {
    demands[demand] = demand;
  }
  const SyncCoverExpandedProgram expansion(inputs.graph, demands);
  const SyncCoverStorageProtocolRectangleGrounding grounding =
      groundSyncCoverStorageProtocolRectangles(
          inputs.graph, expansion, inputs.automata, inputs.frontiers,
          rectangles, demands);
  bool foundMultiDemandFact = false;
  for (const SyncCoverStorageProtocolRectangleGroundingDetail &detail :
       grounding.getDetails()) {
    foundMultiDemandFact |= factRectangles[detail.rectangle] &&
                            detail.admittedDemands >= 2 &&
                            detail.coverageRows >= 2;
  }
  return check(foundMultiDemandFact,
               "ground every exact demand authorized by one provider fact");
}

bool testMixedGraphIndexesFailClosed() {
  RectangleInputs first = buildPrefixCertificateProtocol();
  RectangleInputs second = buildPartialDomainFactProtocol();
  if (!check(first.valid && second.valid, "build mixed-index inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolRectangleIndex secondRectangles =
      buildSyncCoverStorageProtocolRectangleIndex(second.graph, second.automata,
                                                  second.frontiers);
  const SyncCoverStorageProtocolFrontierIndex mixedFrontiers =
      buildSyncCoverStorageProtocolFrontierIndex(first.graph, first.lifecycle,
                                                 second.automata);
  const SyncCoverStorageProtocolRectangleIndex mixedAutomata =
      buildSyncCoverStorageProtocolRectangleIndex(first.graph, second.automata,
                                                  first.frontiers);
  const SyncCoverStorageProtocolRectangleIndex mixedConstruction =
      buildSyncCoverStorageProtocolRectangleIndex(first.graph, first.automata,
                                                  second.frontiers);
  std::vector<SyncCoverDemandId> demands(first.graph.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < demands.size(); ++demand) {
    demands[demand] = demand;
  }
  const SyncCoverExpandedProgram expansion(first.graph, demands);
  const SyncCoverStorageProtocolRectangleGrounding mixedGrounding =
      groundSyncCoverStorageProtocolRectangles(first.graph, expansion,
                                               first.automata, first.frontiers,
                                               secondRectangles, demands);
  return check(mixedFrontiers.getError() ==
                   SyncCoverStorageProtocolFrontierError::InvalidGraph,
               "reject automata from a different frozen graph") &&
         check(mixedAutomata.getError() ==
                   SyncCoverStorageProtocolRectangleError::InvalidGraph,
               "reject rectangle automata from a different frozen graph") &&
         check(mixedConstruction.getError() ==
                   SyncCoverStorageProtocolRectangleError::InvalidGraph,
               "reject frontier provenance from a different frozen graph") &&
         check(
             mixedGrounding.getError() ==
                 SyncCoverStorageProtocolRectangleGroundingError::InvalidGraph,
             "reject rectangle provenance from a different frozen graph");
}

} // namespace

int main() {
  return testCompactRectanglesAndGrounding() &&
                 testProviderFactDoesNotCoverPartialDomainRow() &&
                 testProviderFactRetainsAllAuthorizedDemands() &&
                 testMixedGraphIndexesFailClosed()
             ? 0
             : 1;
}
