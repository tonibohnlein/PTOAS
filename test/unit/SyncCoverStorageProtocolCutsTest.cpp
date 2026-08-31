// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolCuts.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolCutsTest failure: " << message
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

void addDemand(SyncCoverGraph &graph, SyncCoverNodeId source,
               SyncCoverNodeId target, SyncCoverScopeId scope,
               unsigned distance, SyncCoverDemandKind kind,
               SyncCoverStorageAccessId sourceAccess,
               SyncCoverStorageAccessId targetAccess, bool &passed) {
  const SyncCoverStorageWitnessId witness =
      takeIndex(graph.addStorageWitness(sourceAccess, targetAccess), passed,
                "add protocol-cut witness");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  passed &=
      check(graph.addDemand(std::move(demand)), "add protocol-cut demand");
}

struct ProtocolCutInputs {
  SyncCoverGraph graph;
  SyncCoverStorageLifecycleIndex lifecycle;
  SyncCoverStorageProtocolAutomatonIndex automata;
  SyncCoverStorageCutIndex cuts;
  bool valid = false;
};

ProtocolCutInputs buildTwoPhaseProtocol(bool completionCapable,
                                        std::uint32_t producerResource = 1,
                                        std::uint32_t consumerResource = 2,
                                        unsigned protocolCount = 1) {
  ProtocolCutInputs result;
  bool passed = true;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add protocol-cut loop");
  const SyncCoverControlId control = takeIndex(
      result.graph.addControl(2, loop), passed, "add protocol-cut control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
      "set protocol-cut phase relation");
  for (unsigned protocol = 0; protocol < protocolCount; ++protocol) {
    const std::uint32_t protocolProducerResource =
        producerResource + 2 * protocol;
    const std::uint32_t protocolConsumerResource =
        consumerResource + 2 * protocol;
    const SyncCoverStorageDomainId domain =
        takeIndex(result.graph.addStorageDomain(), passed,
                  "add protocol-cut storage domain");
    SyncCoverNodeId producers[2];
    SyncCoverNodeId consumers[2];
    SyncCoverStorageAccessId writes[2];
    SyncCoverStorageAccessId reads[2];
    for (unsigned alternative = 0; alternative < 2; ++alternative) {
      const SyncCoverGuard guard{{{control, alternative}}};
      const std::vector<std::uint32_t> completionTargets =
          completionCapable
              ? std::vector<std::uint32_t>{protocolConsumerResource}
              : std::vector<std::uint32_t>{};
      const std::size_t order = 4 * protocol + 2 * alternative;
      producers[alternative] =
          takeIndex(result.graph.addNode(protocolProducerResource, 1, loop,
                                         order, guard, completionTargets),
                    passed, "add protocol-cut producer");
      consumers[alternative] = takeIndex(
          result.graph.addNode(protocolConsumerResource, 1, loop, order + 1,
                               guard, {protocolProducerResource}),
          passed, "add protocol-cut consumer");
      writes[alternative] =
          takeIndex(result.graph.addStorageAccess(
                        producers[alternative], domain, 20 + protocol, {0, 64},
                        SyncCoverStorageAccessMode::Write, std::nullopt, true),
                    passed, "add protocol-cut write");
      reads[alternative] =
          takeIndex(result.graph.addStorageAccess(
                        consumers[alternative], domain, 20 + protocol, {0, 64},
                        SyncCoverStorageAccessMode::Read, std::nullopt, true),
                    passed, "add protocol-cut read");
      addDemand(result.graph, producers[alternative], consumers[alternative],
                loop, 0, SyncCoverDemandKind::MemoryRAW, writes[alternative],
                reads[alternative], passed);
    }
    for (unsigned alternative = 0; alternative < 2; ++alternative) {
      const unsigned successor = 1 - alternative;
      addDemand(result.graph, consumers[alternative], producers[successor],
                loop, 1, SyncCoverDemandKind::MemoryWAR, reads[alternative],
                writes[successor], passed);
    }
  }
  passed &= check(result.graph.freezeStructure(), "freeze protocol-cut graph");
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
  result.cuts = buildSyncCoverStorageCutIndex(result.graph, result.lifecycle);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.cuts.isComplete();
  return result;
}

ProtocolCutInputs buildDistanceZeroReleaseProtocol() {
  ProtocolCutInputs result;
  bool passed = true;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add distance-zero-release loop");
  const SyncCoverStorageDomainId domain =
      takeIndex(result.graph.addStorageDomain(), passed,
                "add distance-zero-release storage domain");
  const SyncCoverNodeId producer =
      takeIndex(result.graph.addNode(1, 1, loop, 0, {}, {2}), passed,
                "add distance-zero-release producer");
  const SyncCoverNodeId consumer =
      takeIndex(result.graph.addNode(2, 1, loop, 1, {}, {1}), passed,
                "add distance-zero-release consumer");
  const SyncCoverNodeId overwriter =
      takeIndex(result.graph.addNode(1, 1, loop, 2, {}, {1}), passed,
                "add distance-zero-release overwriter");
  const SyncCoverStorageAccessId write =
      takeIndex(result.graph.addStorageAccess(producer, domain, 30, {0, 64},
                                              SyncCoverStorageAccessMode::Write,
                                              std::nullopt, true),
                passed, "add distance-zero-release write");
  const SyncCoverStorageAccessId read =
      takeIndex(result.graph.addStorageAccess(consumer, domain, 30, {0, 64},
                                              SyncCoverStorageAccessMode::Read,
                                              std::nullopt, true),
                passed, "add distance-zero-release read");
  const SyncCoverStorageAccessId overwrite =
      takeIndex(result.graph.addStorageAccess(overwriter, domain, 30, {0, 64},
                                              SyncCoverStorageAccessMode::Write,
                                              std::nullopt, true),
                passed, "add distance-zero-release overwrite");
  addDemand(result.graph, producer, consumer, loop, 0,
            SyncCoverDemandKind::MemoryRAW, write, read, passed);
  addDemand(result.graph, consumer, overwriter, loop, 0,
            SyncCoverDemandKind::MemoryWAR, read, overwrite, passed);
  addDemand(result.graph, overwriter, producer, loop, 1,
            SyncCoverDemandKind::MemoryWAW, overwrite, write, passed);
  passed &= check(result.graph.freezeStructure(),
                  "freeze distance-zero-release graph");
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
  result.cuts = buildSyncCoverStorageCutIndex(result.graph, result.lifecycle);
  result.valid = result.lifecycle.isComplete() && seeds.isComplete() &&
                 groups.isComplete() && result.automata.isComplete() &&
                 result.cuts.isComplete();
  return result;
}

bool checkTransactionalLimit(const SyncCoverStorageProtocolCutPlanIndex &index,
                             std::string_view message) {
  return check(index.getError() ==
                       SyncCoverStorageProtocolCutPlanError::LimitExceeded &&
                   index.getPlans().empty() && index.getStatistics().truncated,
               message);
}

bool testReadyRectangleAssociationAndBounds() {
  ProtocolCutInputs inputs = buildTwoPhaseProtocol(true);
  if (!check(inputs.valid, "build protocol-cut inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlanIndex index =
      buildSyncCoverStorageProtocolCutPlanIndex(inputs.graph, inputs.lifecycle,
                                                inputs.automata, inputs.cuts);
  const bool onePlan = index.isComplete() && index.getPlans().size() == 1;
  if (!check(onePlan, "build one protocol cut plan")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlan &plan = index.getPlans().front();
  const SyncCoverStorageProtocolCutPlanStatistics &statistics =
      index.getStatistics();
  bool passed =
      check(plan.laneCount == 1 && plan.directReadyTransfers == 2 &&
                plan.recurrenceReleaseTransfers == 2 &&
                plan.readyRectangles.size() == 2 &&
                statistics.eligibleAutomata == 1 &&
                statistics.ineligibleAutomata == 0 && statistics.plans == 1 &&
                statistics.transferInspections == 4 &&
                statistics.readyRectangleIncidences == 2,
            "associate each phase-ready transfer with one exact rectangle");

  SyncCoverStorageProtocolCutPlanLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumPlans = statistics.plans;
  exact.maximumRectangleEdgeIncidences = 2;
  exact.maximumTransferInspections = statistics.transferInspections;
  exact.maximumReadyRectangleIncidences = statistics.readyRectangleIncidences;
  passed &= check(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, exact)
          .isComplete(),
      "accept protocol cut plans at exact bounds");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, exact),
      "reject protocol cut plans one work unit below the exact bound");
  exact.maximumWorkUnits = statistics.workUnits;
  --exact.maximumReadyRectangleIncidences;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, exact),
      "enforce the ready-rectangle incidence bound transactionally");
  exact.maximumReadyRectangleIncidences = statistics.readyRectangleIncidences;
  --exact.maximumRectangleEdgeIncidences;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, exact),
      "enforce the rectangle-edge incidence bound transactionally");
  exact.maximumRectangleEdgeIncidences = 2;
  --exact.maximumTransferInspections;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, exact),
      "enforce the transfer-inspection bound transactionally");
  return passed;
}

bool testPlanCountBoundIsTransactional() {
  ProtocolCutInputs inputs = buildTwoPhaseProtocol(true, 1, 2, 2);
  if (!check(inputs.valid, "build two eligible protocol-cut inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlanIndex index =
      buildSyncCoverStorageProtocolCutPlanIndex(inputs.graph, inputs.lifecycle,
                                                inputs.automata, inputs.cuts);
  const bool builtTwoPlans = index.isComplete() && index.getPlans().size() == 2;
  if (!check(builtTwoPlans, "build two protocol cut plans")) {
    return false;
  }
  SyncCoverStorageProtocolCutPlanLimits limits;
  limits.maximumPlans = 1;
  return checkTransactionalLimit(
      buildSyncCoverStorageProtocolCutPlanIndex(
          inputs.graph, inputs.lifecycle, inputs.automata, inputs.cuts, limits),
      "enforce the plan-count bound transactionally");
}

bool testMixedIndexesAreRejected() {
  ProtocolCutInputs first = buildTwoPhaseProtocol(true, 1, 2);
  ProtocolCutInputs second = buildTwoPhaseProtocol(true, 3, 4);
  if (!check(first.valid && second.valid,
             "build protocol inputs with coincident numeric IDs")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlanIndex index =
      buildSyncCoverStorageProtocolCutPlanIndex(first.graph, first.lifecycle,
                                                second.automata, first.cuts);
  return check(index.getError() ==
                       SyncCoverStorageProtocolCutPlanError::InvalidGraph &&
                   index.getPlans().empty(),
               "reject automata built from a different lifecycle index");
}

bool testDistanceZeroReleaseDoesNotCloseTheProtocol() {
  ProtocolCutInputs inputs = buildDistanceZeroReleaseProtocol();
  if (!check(inputs.valid, "build distance-zero-release protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlanIndex index =
      buildSyncCoverStorageProtocolCutPlanIndex(inputs.graph, inputs.lifecycle,
                                                inputs.automata, inputs.cuts);
  const SyncCoverStorageProtocolCutPlanStatistics &statistics =
      index.getStatistics();
  return check(index.isComplete() && index.getPlans().empty() &&
                   statistics.eligibleAutomata == 0 &&
                   statistics.ineligibleAutomata == 1 &&
                   statistics.missingReleaseAutomata == 1,
               "require a positive-distance release transfer");
}

bool testMissingReadyCutRejectsOnlyTheProposal() {
  ProtocolCutInputs inputs = buildTwoPhaseProtocol(false);
  if (!check(inputs.valid, "build completion-incapable protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolCutPlanIndex index =
      buildSyncCoverStorageProtocolCutPlanIndex(inputs.graph, inputs.lifecycle,
                                                inputs.automata, inputs.cuts);
  const SyncCoverStorageProtocolCutPlanStatistics &statistics =
      index.getStatistics();
  return check(index.isComplete() && index.getPlans().empty() &&
                   statistics.eligibleAutomata == 0 &&
                   statistics.ineligibleAutomata == 1 &&
                   statistics.missingReadyCutAutomata == 1,
               "reject only an automaton with no target-authorized ready cut");
}

} // namespace

int main() {
  return testReadyRectangleAssociationAndBounds() &&
                 testPlanCountBoundIsTransactional() &&
                 testMixedIndexesAreRejected() &&
                 testDistanceZeroReleaseDoesNotCloseTheProtocol() &&
                 testMissingReadyCutRejectsOnlyTheProposal()
             ? 0
             : 1;
}
