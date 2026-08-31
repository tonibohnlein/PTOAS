// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolFrontiers.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverStorageCuts.h"

#include <algorithm>
#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolFrontiersTest failure: " << message
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

SyncCoverDemandId addDemandWithId(SyncCoverGraph &graph, SyncCoverNodeId source,
                                  SyncCoverNodeId target,
                                  SyncCoverScopeId scope, unsigned distance,
                                  SyncCoverDemandKind kind,
                                  SyncCoverStorageAccessId sourceAccess,
                                  SyncCoverStorageAccessId targetAccess,
                                  bool &passed) {
  const SyncCoverDemandId demandId = graph.getDemands().size();
  const SyncCoverStorageWitnessId witness =
      takeIndex(graph.addStorageWitness(sourceAccess, targetAccess), passed,
                "add protocol-frontier witness");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  passed &=
      check(graph.addDemand(std::move(demand)), "add protocol-frontier demand");
  return demandId;
}

void addDemand(SyncCoverGraph &graph, SyncCoverNodeId source,
               SyncCoverNodeId target, SyncCoverScopeId scope,
               unsigned distance, SyncCoverDemandKind kind,
               SyncCoverStorageAccessId sourceAccess,
               SyncCoverStorageAccessId targetAccess, bool &passed) {
  (void)addDemandWithId(graph, source, target, scope, distance, kind,
                        sourceAccess, targetAccess, passed);
}

struct ProtocolFrontierInputs {
  SyncCoverGraph graph;
  SyncCoverStorageLifecycleIndex lifecycle;
  SyncCoverStorageProtocolAutomatonIndex automata;
  SyncCoverStorageCutIndex directCuts;
  bool valid = false;
};

ProtocolFrontierInputs buildGuardedSiblingProtocols(
    bool completionCapable, std::uint32_t firstResource = 1,
    unsigned protocolCount = 1, unsigned nestingDepth = 1,
    unsigned controlCount = 1) {
  ProtocolFrontierInputs result;
  bool passed = true;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add protocol-frontier loop");
  SyncCoverGuard activeGuard;
  for (unsigned controlOrdinal = 0; controlOrdinal < controlCount;
       ++controlOrdinal) {
    const SyncCoverControlId control =
        takeIndex(result.graph.addControl(2, loop), passed,
                  "add protocol-frontier control");
    passed &= check(result.graph.setControlPhaseRelation(
                        control, {loop, 0, {1, 0}, {0, 1}}),
                    "set protocol-frontier phase relation");
    activeGuard.literals.push_back({control, 0});
  }
  SyncCoverScopeId producerScope = loop;
  SyncCoverScopeId consumerScope = loop;
  for (unsigned depth = 0; depth < nestingDepth; ++depth) {
    producerScope =
        takeIndex(result.graph.addScope(producerScope, false,
                                        SyncCoverTimelineInterval{0, 16},
                                        depth != 0, activeGuard),
                  passed, "add guarded producer scope");
    consumerScope =
        takeIndex(result.graph.addScope(consumerScope, false,
                                        SyncCoverTimelineInterval{16, 32},
                                        depth != 0, activeGuard),
                  passed, "add guarded consumer scope");
  }

  std::vector<SyncCoverStorageDomainId> domains;
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverStorageAccessId> writes;
  domains.reserve(protocolCount);
  producers.reserve(protocolCount);
  writes.reserve(protocolCount);
  for (unsigned protocol = 0; protocol < protocolCount; ++protocol) {
    const std::uint32_t producerResource = firstResource + 2 * protocol;
    const std::uint32_t consumerResource = producerResource + 1;
    const SyncCoverStorageDomainId domain =
        takeIndex(result.graph.addStorageDomain(), passed,
                  "add protocol-frontier domain");
    const std::vector<std::uint32_t> completionTargets =
        completionCapable ? std::vector<std::uint32_t>{consumerResource}
                          : std::vector<std::uint32_t>{};
    domains.push_back(domain);
    const SyncCoverNodeId producer = takeIndex(
        result.graph.addNode(producerResource, 1, producerScope, 2 + protocol,
                             activeGuard, completionTargets),
        passed, "add guarded protocol producer");
    producers.push_back(producer);
    const SyncCoverStorageAccessId write =
        takeIndex(result.graph.addStorageAccess(
                      producer, domain, 40 + protocol, {0, 64},
                      SyncCoverStorageAccessMode::Write, std::nullopt, true),
                  passed, "add guarded protocol write");
    writes.push_back(write);
  }
  for (unsigned protocol = 0; protocol < protocolCount; ++protocol) {
    const std::uint32_t producerResource = firstResource + 2 * protocol;
    const std::uint32_t consumerResource = producerResource + 1;
    const SyncCoverNodeId consumer =
        takeIndex(result.graph.addNode(consumerResource, 1, consumerScope,
                                       8 + protocolCount + protocol,
                                       activeGuard, {producerResource}),
                  passed, "add guarded protocol consumer");
    const SyncCoverStorageAccessId read =
        takeIndex(result.graph.addStorageAccess(
                      consumer, domains[protocol], 40 + protocol, {0, 64},
                      SyncCoverStorageAccessMode::Read, std::nullopt, true),
                  passed, "add guarded protocol read");
    addDemand(result.graph, producers[protocol], consumer, loop, 0,
              SyncCoverDemandKind::MemoryRAW, writes[protocol], read, passed);
    addDemand(result.graph, consumer, producers[protocol], loop, 2,
              SyncCoverDemandKind::MemoryWAR, read, writes[protocol], passed);
  }

  passed &=
      check(result.graph.freezeStructure(), "freeze protocol-frontier graph");
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
  result.directCuts =
      buildSyncCoverStorageCutIndex(result.graph, result.lifecycle);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.directCuts.isComplete();
  return result;
}

ProtocolFrontierInputs
buildCertificateBackedProtocol(unsigned producerCount = 2) {
  ProtocolFrontierInputs result;
  bool passed = true;
  constexpr std::uint32_t kProducerResource = 1;
  constexpr std::uint32_t kConsumerResource = 2;
  constexpr std::uint32_t kUnusedResource = 3;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
      passed, "add certificate-backed loop");
  const SyncCoverStorageDomainId domain = takeIndex(
      result.graph.addStorageDomain(SyncCoverStorageDomainRole::L0Left), passed,
      "add certificate-backed domain");
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverStorageAccessId> writes;
  producers.reserve(producerCount);
  writes.reserve(producerCount);
  for (unsigned producer = 0; producer < producerCount; ++producer) {
    const SyncCoverNodeId node = takeIndex(
        result.graph.addNode(kProducerResource, 1, loop, 2 + producer), passed,
        "add certificate-backed producer");
    producers.push_back(node);
    writes.push_back(takeIndex(
        result.graph.addStorageAccess(node, domain, 9, {0, 64},
                                      SyncCoverStorageAccessMode::Write,
                                      std::nullopt, true),
        passed, "add certificate-backed write"));
  }
  const SyncCoverNodeId consumer = takeIndex(
      result.graph.addNode(kConsumerResource, 1, loop, 4 + producerCount, {},
                           {kProducerResource}),
      passed, "add certificate-backed consumer");
  const SyncCoverStorageAccessId read =
      takeIndex(result.graph.addStorageAccess(consumer, domain, 9, {0, 64},
                                              SyncCoverStorageAccessMode::Read,
                                              std::nullopt, true),
                passed, "add certificate-backed read");
  std::vector<SyncCoverDemandId> readyDemands;
  readyDemands.reserve(producerCount);
  for (unsigned producer = 0; producer < producerCount; ++producer) {
    readyDemands.push_back(addDemandWithId(
        result.graph, producers[producer], consumer, loop, 0,
        SyncCoverDemandKind::MemoryRAW, writes[producer], read, passed));
    addDemand(result.graph, consumer, producers[producer], loop, 1,
              SyncCoverDemandKind::MemoryWAR, read, writes[producer], passed);
  }
  passed &= check(result.graph.setTargetCompletionResources(
                      {kProducerResource, kConsumerResource, kUnusedResource}),
                  "set certificate-backed target resources");
  for (unsigned producer = 0; producer < producerCount; ++producer) {
    std::vector<SyncCoverDemandId> prefix(readyDemands.begin(),
                                          readyDemands.begin() + producer + 1);
    passed &= check(result.graph.addTargetCompletionCertificate(
                        SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix,
                        producers[producer], consumer, kProducerResource,
                        kConsumerResource, {domain}, std::move(prefix)),
                    "add prefix certificate-backed completion cut");
  }
  passed &= check(result.graph.freezeStructure(),
                  "freeze certificate-backed protocol graph");
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
  result.directCuts =
      buildSyncCoverStorageCutIndex(result.graph, result.lifecycle);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.directCuts.isComplete();
  return result;
}

ProtocolFrontierInputs buildCompletionCutFactBackedProtocol() {
  ProtocolFrontierInputs result;
  bool passed = true;
  constexpr unsigned kProtocolCount = 2;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
      passed, "add completion-cut-fact loop");
  const SyncCoverControlId control =
      takeIndex(result.graph.addControl(2, loop), passed,
                "add completion-cut-fact control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
      "set completion-cut-fact phase relation");
  const SyncCoverGuard producerGuard{{{control, 0}}};
  const SyncCoverScopeId producerScope = takeIndex(
      result.graph.addScope(loop, false, SyncCoverTimelineInterval{0, 16},
                            false, producerGuard),
      passed, "add completion-cut-fact guarded scope");
  for (unsigned protocol = 0; protocol < kProtocolCount; ++protocol) {
    const std::uint32_t producerResource = 1 + 2 * protocol;
    const std::uint32_t consumerResource = producerResource + 1;
    const SyncCoverStorageDomainId domain =
        takeIndex(result.graph.addStorageDomain(), passed,
                  "add completion-cut-fact domain");
    const SyncCoverNodeId producer = takeIndex(
        result.graph.addNode(producerResource, 1, producerScope,
                             2 + 4 * protocol,
                             producerGuard),
        passed, "add completion-cut-fact producer");
    const SyncCoverNodeId consumer = takeIndex(
        result.graph.addNode(consumerResource, 1, loop,
                             4 + 4 * protocol, {},
                             {producerResource}),
        passed, "add completion-cut-fact consumer");
    const SyncCoverStorageAccessId write = takeIndex(
        result.graph.addStorageAccess(producer, domain, 7 + protocol, {0, 64},
                                      SyncCoverStorageAccessMode::Write,
                                      std::nullopt, true),
        passed, "add completion-cut-fact write");
    const SyncCoverStorageAccessId read = takeIndex(
        result.graph.addStorageAccess(consumer, domain, 7 + protocol, {0, 64},
                                      SyncCoverStorageAccessMode::Read,
                                      std::nullopt, true),
        passed, "add completion-cut-fact read");
    SyncCoverDemandId ready = 0;
    if (protocol == 0) {
      const SyncCoverStorageDomainId unauthorizedDomain =
          takeIndex(result.graph.addStorageDomain(), passed,
                    "add unauthorized completion-cut-fact domain");
      const SyncCoverStorageAccessId unauthorizedWrite = takeIndex(
          result.graph.addStorageAccess(producer, unauthorizedDomain, 17,
                                        {0, 64},
                                        SyncCoverStorageAccessMode::Write,
                                        std::nullopt, true),
          passed, "add unauthorized completion-cut-fact write");
      const SyncCoverStorageAccessId unauthorizedRead = takeIndex(
          result.graph.addStorageAccess(consumer, unauthorizedDomain, 17,
                                        {0, 64},
                                        SyncCoverStorageAccessMode::Read,
                                        std::nullopt, true),
          passed, "add unauthorized completion-cut-fact read");
      const SyncCoverStorageWitnessId readyWitness = takeIndex(
          result.graph.addStorageWitness(write, read), passed,
          "add authorized completion-cut-fact ready witness");
      const SyncCoverStorageWitnessId unauthorizedReadyWitness = takeIndex(
          result.graph.addStorageWitness(unauthorizedWrite, unauthorizedRead),
          passed, "add unauthorized completion-cut-fact ready witness");
      SyncCoverDemand readyDemand;
      readyDemand.source = producer;
      readyDemand.target = consumer;
      readyDemand.scope = loop;
      readyDemand.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
      readyDemand.storageWitnesses = {readyWitness,
                                      unauthorizedReadyWitness};
      ready = takeIndex(result.graph.addDemand(std::move(readyDemand)), passed,
                        "add multi-domain ready demand");
      const SyncCoverStorageWitnessId reuseWitness = takeIndex(
          result.graph.addStorageWitness(read, write), passed,
          "add authorized completion-cut-fact reuse witness");
      const SyncCoverStorageWitnessId unauthorizedReuseWitness = takeIndex(
          result.graph.addStorageWitness(unauthorizedRead, unauthorizedWrite),
          passed, "add unauthorized completion-cut-fact reuse witness");
      SyncCoverDemand reuseDemand;
      reuseDemand.source = consumer;
      reuseDemand.target = producer;
      reuseDemand.scope = loop;
      reuseDemand.distance = 1;
      reuseDemand.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
      reuseDemand.storageWitnesses = {reuseWitness,
                                      unauthorizedReuseWitness};
      passed &= check(result.graph.addDemand(std::move(reuseDemand)),
                      "add multi-domain reuse demand");
    } else {
      ready = addDemandWithId(result.graph, producer, consumer, loop, 0,
                              SyncCoverDemandKind::MemoryRAW, write, read,
                              passed);
      addDemand(result.graph, consumer, producer, loop, 1,
                SyncCoverDemandKind::MemoryWAR, read, write, passed);
    }
    passed &= check(result.graph.addCompletionCutFact(
                        producer, producerResource, consumerResource, {domain},
                        {ready}),
                    "add guarded provider completion-cut fact");
  }
  passed &=
      check(result.graph.freezeStructure(), "freeze completion-cut-fact graph");
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
  result.directCuts =
      buildSyncCoverStorageCutIndex(result.graph, result.lifecycle);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.directCuts.isComplete();
  return result;
}

bool checkTransactionalLimit(const SyncCoverStorageProtocolFrontierIndex &index,
                             std::string_view message) {
  return check(index.getError() ==
                       SyncCoverStorageProtocolFrontierError::LimitExceeded &&
                   index.getFrontiers().empty() && index.getPlans().empty() &&
                   index.getStatistics().truncated,
               message);
}

SyncCoverStorageProtocolFrontierLimits
exactLimits(const SyncCoverStorageProtocolFrontierStatistics &statistics) {
  SyncCoverStorageProtocolFrontierLimits limits;
  limits.maximumWorkUnits = statistics.workUnits;
  limits.maximumPlans = statistics.plans;
  limits.maximumFrontiers = statistics.frontiers;
  limits.maximumTransferInspections = statistics.transferInspections;
  limits.maximumStatePairInspections = statistics.statePairInspections;
  limits.maximumPlanFrontierIncidences = statistics.planFrontierIncidences;
  limits.maximumCertificateDemandIncidences =
      statistics.certificateDemandIncidences == 0
          ? 1
          : statistics.certificateDemandIncidences;
  limits.maximumCompletionCutFactDemandIncidences =
      statistics.completionCutFactDemandIncidences == 0
          ? 1
          : statistics.completionCutFactDemandIncidences;
  return limits;
}

bool testGuardedSiblingFrontiersAndBounds() {
  ProtocolFrontierInputs inputs = buildGuardedSiblingProtocols(true, 1, 2);
  if (!check(inputs.valid, "build guarded sibling protocol inputs")) {
    return false;
  }
  if (!check(inputs.directCuts.getRectangles().empty(),
             "ordinary balanced direct cuts reject guarded sibling scopes")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const SyncCoverStorageProtocolFrontierStatistics &statistics =
      index.getStatistics();
  const bool expectedPlans =
      index.isComplete() && index.getPlans().size() == 2 &&
      index.getFrontiers().size() == 4 && statistics.readyFrontiers == 2 &&
      statistics.reuseFrontiers == 2 &&
      statistics.missingCompletionFrontierAutomata == 0;
  if (!check(expectedPlans,
             "retain guarded sibling ready and release endpoint frontiers")) {
    return false;
  }
  bool passed = true;
  for (const SyncCoverStorageProtocolFrontierPlan &plan : index.getPlans()) {
    passed &= check(plan.laneCount == 2 && plan.frontiers.size() == 2 &&
                        plan.readyFrontiers == 1 && plan.reuseFrontiers == 1,
                    "retain one two-lane ready/release frontier plan");
  }
  for (const SyncCoverStorageProtocolFrontier &frontier :
       index.getFrontiers()) {
    passed &=
        check(frontier.transfer.has_value() && frontier.edge.has_value() &&
                  !frontier.completionCutFact.has_value() &&
                  !frontier.completionCertificate.has_value(),
              "retain exact direct-transfer frontier provenance");
  }

  SyncCoverStorageProtocolFrontierLimits exact = exactLimits(statistics);
  passed &= check(buildSyncCoverStorageProtocolFrontierIndex(
                      inputs.graph, inputs.lifecycle, inputs.automata, exact)
                      .isComplete(),
                  "accept protocol frontiers at every exact bound");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "reject one work unit below the exact bound");
  exact = exactLimits(statistics);
  --exact.maximumPlans;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "enforce the plan bound transactionally");
  exact = exactLimits(statistics);
  --exact.maximumFrontiers;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "enforce the frontier bound transactionally");
  exact = exactLimits(statistics);
  --exact.maximumTransferInspections;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "enforce the transfer-inspection bound transactionally");
  exact = exactLimits(statistics);
  --exact.maximumStatePairInspections;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "enforce the state-pair bound transactionally");
  exact = exactLimits(statistics);
  --exact.maximumPlanFrontierIncidences;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, exact),
      "enforce the plan-frontier incidence bound transactionally");
  return passed;
}

bool testCompletionCutFactFrontierAndBound() {
  ProtocolFrontierInputs inputs = buildCompletionCutFactBackedProtocol();
  if (!check(inputs.valid, "build completion-cut-fact protocol inputs")) {
    return false;
  }
  if (!check(inputs.directCuts.getRectangles().empty(),
             "standalone direct cuts reject the guarded accumulator shape")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const SyncCoverStorageProtocolFrontierStatistics &statistics =
      index.getStatistics();
  const auto factFrontier =
      std::find_if(index.getFrontiers().begin(), index.getFrontiers().end(),
                   [](const SyncCoverStorageProtocolFrontier &frontier) {
                     return frontier.completionCutFact.has_value();
                   });
  const bool retainedFact =
      index.isComplete() && index.getPlans().size() == 2 &&
      statistics.completionCutFactDemandIncidences == 2 &&
      statistics.completionCutFactFrontiers == 2 &&
      factFrontier != index.getFrontiers().end() &&
      factFrontier->transfer.has_value() && factFrontier->edge.has_value() &&
      !factFrontier->completionCertificate.has_value();
  if (!check(retainedFact,
             "retain exact demand-linked provider completion-cut frontier")) {
    return false;
  }
  SyncCoverStorageProtocolFrontierLimits limits = exactLimits(statistics);
  if (!check(buildSyncCoverStorageProtocolFrontierIndex(
                 inputs.graph, inputs.lifecycle, inputs.automata, limits)
                 .isComplete(),
             "accept completion-cut-fact work at every exact bound")) {
    return false;
  }
  --limits.maximumWorkUnits;
  if (!checkTransactionalLimit(
          buildSyncCoverStorageProtocolFrontierIndex(
              inputs.graph, inputs.lifecycle, inputs.automata, limits),
          "reject completion-cut-fact work below the exact work bound")) {
    return false;
  }
  limits = exactLimits(statistics);
  --limits.maximumCompletionCutFactDemandIncidences;
  return checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, limits),
      "enforce the completion-cut-fact incidence bound transactionally");
}

bool testCertificateFrontierAndBound() {
  ProtocolFrontierInputs inputs = buildCertificateBackedProtocol();
  if (!check(inputs.valid, "build certificate-backed protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const SyncCoverStorageProtocolFrontierStatistics &statistics =
      index.getStatistics();
  const bool expectedCertificateFrontier =
      index.isComplete() && index.getPlans().size() == 1 &&
      statistics.certificateDemandIncidences == 3 &&
      statistics.certificateFrontiers == 2 && statistics.readyFrontiers == 2 &&
      statistics.reuseFrontiers >= 1;
  if (!check(expectedCertificateFrontier,
             "retain one certificate-backed ready frontier")) {
    return false;
  }
  const auto certificateFrontier =
      std::find_if(index.getFrontiers().begin(), index.getFrontiers().end(),
                   [](const SyncCoverStorageProtocolFrontier &frontier) {
                     return frontier.completionCertificate.has_value();
                   });
  const bool exactCertificateProvenance =
      certificateFrontier != index.getFrontiers().end() &&
      !certificateFrontier->transfer.has_value() &&
      !certificateFrontier->edge.has_value() &&
      certificateFrontier->kind == SyncCoverStorageProtocolFrontierKind::Ready;
  if (!check(exactCertificateProvenance,
             "retain exact certificate-only frontier provenance")) {
    return false;
  }
  const std::size_t certificateAlternatives =
      std::count_if(index.getFrontiers().begin(), index.getFrontiers().end(),
                    [](const SyncCoverStorageProtocolFrontier &frontier) {
                      return frontier.completionCertificate.has_value();
                    });
  if (!check(certificateAlternatives == 2,
             "retain narrow and prefix certificate alternatives")) {
    return false;
  }
  SyncCoverStorageProtocolFrontierLimits limits = exactLimits(statistics);
  --limits.maximumCertificateDemandIncidences;
  return checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, limits),
      "enforce the certificate-demand incidence bound transactionally");
}

bool testDeepScopeAndLongGuardWorkIsBounded() {
  ProtocolFrontierInputs inputs =
      buildGuardedSiblingProtocols(true, 1, 1, 12, 8);
  if (!check(inputs.valid, "build deep-scope long-guard protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const bool retainedPlan = index.isComplete() && index.getPlans().size() == 1;
  if (!check(retainedPlan, "retain deep-scope long-guard frontier plan")) {
    return false;
  }
  SyncCoverStorageProtocolFrontierLimits limits =
      exactLimits(index.getStatistics());
  if (!check(buildSyncCoverStorageProtocolFrontierIndex(
                 inputs.graph, inputs.lifecycle, inputs.automata, limits)
                 .isComplete(),
             "accept deep-scope long-guard work at the exact bound")) {
    return false;
  }
  --limits.maximumWorkUnits;
  return checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, limits),
      "reject deep-scope long-guard work one unit below the exact bound");
}

bool testManyCertificateAlternativesAreBounded() {
  constexpr unsigned kProducerCount = 8;
  ProtocolFrontierInputs inputs =
      buildCertificateBackedProtocol(kProducerCount);
  if (!check(inputs.valid, "build many-certificate protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const SyncCoverStorageProtocolFrontierStatistics &statistics =
      index.getStatistics();
  constexpr std::size_t kCertificateDemandIncidences =
      kProducerCount * (kProducerCount + 1) / 2;
  const bool retainedAlternatives =
      index.isComplete() && index.getPlans().size() == 1 &&
      statistics.certificateFrontiers == kProducerCount &&
      statistics.certificateDemandIncidences == kCertificateDemandIncidences;
  if (!check(retainedAlternatives,
             "retain every bounded certificate-prefix alternative")) {
    return false;
  }
  SyncCoverStorageProtocolFrontierLimits limits = exactLimits(statistics);
  if (!check(buildSyncCoverStorageProtocolFrontierIndex(
                 inputs.graph, inputs.lifecycle, inputs.automata, limits)
                 .isComplete(),
             "accept many-certificate work at the exact bound")) {
    return false;
  }
  --limits.maximumWorkUnits;
  return checkTransactionalLimit(
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata, limits),
      "reject many-certificate work one unit below the exact bound");
}

bool testMissingCompletionRejectsOnlyTheAutomaton() {
  ProtocolFrontierInputs inputs = buildGuardedSiblingProtocols(false);
  if (!check(inputs.valid, "build completion-incapable frontier inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(inputs.graph, inputs.lifecycle,
                                                 inputs.automata);
  const SyncCoverStorageProtocolFrontierStatistics &statistics =
      index.getStatistics();
  return check(index.isComplete() && index.getPlans().empty() &&
                   index.getFrontiers().empty() &&
                   statistics.ineligibleAutomata == 1 &&
                   statistics.missingCompletionFrontierAutomata == 1,
               "reject only an automaton with an unauthorized endpoint cut");
}

bool testMixedIndexesAreRejected() {
  ProtocolFrontierInputs first = buildGuardedSiblingProtocols(true, 1);
  ProtocolFrontierInputs second = buildGuardedSiblingProtocols(true, 5);
  if (!check(first.valid && second.valid,
             "build frontier inputs with coincident numeric IDs")) {
    return false;
  }
  const SyncCoverStorageProtocolFrontierIndex index =
      buildSyncCoverStorageProtocolFrontierIndex(first.graph, first.lifecycle,
                                                 second.automata);
  return check(index.getError() ==
                       SyncCoverStorageProtocolFrontierError::InvalidGraph &&
                   index.getPlans().empty() && index.getFrontiers().empty(),
               "reject automata from a different lifecycle instance");
}

} // namespace

int main() {
  return testGuardedSiblingFrontiersAndBounds() &&
                 testCompletionCutFactFrontierAndBound() &&
                 testCertificateFrontierAndBound() &&
                 testDeepScopeAndLongGuardWorkIsBounded() &&
                 testManyCertificateAlternativesAreBounded() &&
                 testMissingCompletionRejectsOnlyTheAutomaton() &&
                 testMixedIndexesAreRejected()
             ? 0
             : 1;
}
