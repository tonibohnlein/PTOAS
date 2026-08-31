// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolSeeds.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolSeedsTest failure: " << message
              << '\n';
  }
  return condition;
}

bool check(const SyncCoverGraphResult &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

bool checkTransactionalLimit(const SyncCoverStorageProtocolSeedIndex &index,
                             std::string_view message) {
  return check(
      index.getError() == SyncCoverStorageProtocolSeedError::LimitExceeded &&
          index.getSeeds().empty() && index.getStatistics().truncated,
      message);
}

std::size_t takeIndex(const SyncCoverGraphResult &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

void addRoundTrip(SyncCoverGraph &graph, SyncCoverScopeId scope,
                  SyncCoverStorageDomainId domain,
                  SyncCoverStorageAccessFamilyId family, unsigned order,
                  bool &passed) {
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, scope, order, {}, {2}), passed,
                "add protocol-seed producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, scope, order + 1, {}, {1}), passed,
                "add protocol-seed consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(producer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add protocol-seed write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(consumer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add protocol-seed read");
  const SyncCoverStorageWitnessId readyWitness =
      takeIndex(graph.addStorageWitness(write, read), passed,
                "add protocol-seed ready witness");
  SyncCoverDemand ready;
  ready.source = producer;
  ready.target = consumer;
  ready.scope = scope;
  ready.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  ready.storageWitnesses = {readyWitness};
  passed &= check(graph.addDemand(std::move(ready)),
                  "add protocol-seed ready demand");

  const SyncCoverStorageWitnessId releaseWitness =
      takeIndex(graph.addStorageWitness(read, write), passed,
                "add protocol-seed release witness");
  SyncCoverDemand release;
  release.source = consumer;
  release.target = producer;
  release.scope = scope;
  release.distance = 1;
  release.provenanceKinds = {SyncCoverDemandKind::MemoryWAR};
  release.storageWitnesses = {releaseWitness};
  passed &= check(graph.addDemand(std::move(release)),
                  "add protocol-seed release demand");
}

bool testProtocolSeeds() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
                passed, "add protocol-seed outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{8, 40}, true),
      passed, "add protocol-seed inner loop");
  const SyncCoverStorageDomainId firstDomain = takeIndex(
      graph.addStorageDomain(), passed, "add first protocol-seed domain");
  const SyncCoverStorageDomainId secondDomain = takeIndex(
      graph.addStorageDomain(), passed, "add second protocol-seed domain");
  addRoundTrip(graph, outer, firstDomain, 11, 0, passed);
  addRoundTrip(graph, inner, firstDomain, 11, 8, passed);
  addRoundTrip(graph, outer, secondDomain, 12, 16, passed);
  passed &= check(graph.freezeStructure(), "freeze protocol-seed graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  const bool completeIndexes = lifecycle.isComplete() && seeds.isComplete();
  if (!check(completeIndexes, "build complete protocol seeds")) {
    return false;
  }
  const SyncCoverStorageProtocolSeedStatistics &statistics =
      seeds.getStatistics();
  passed &= check(
      statistics.seeds == 2 && statistics.readyReleaseSeeds == 2 &&
          statistics.componentIncidences == 3 &&
          statistics.slotIncidences == 2 && statistics.sccIncidences == 3 &&
          statistics.demandIncidences == 6 &&
          statistics.maximumSeedComponents == 2 &&
          statistics.maximumSeedSlots == 1 && statistics.maximumSeedSccs == 2,
      "report merged nested lifecycle-owner statistics");
  passed &=
      check(seeds.getSeeds().size() == 2 && seeds.getSeeds()[0].family == 11 &&
                seeds.getSeeds()[0].owningScope == outer &&
                seeds.getSeeds()[0].components.size() == 2 &&
                seeds.getSeeds()[0].slots.size() == 1 &&
                seeds.getSeeds()[0].readyReleaseSccs.size() == 2 &&
                seeds.getSeeds()[0].demands.size() == 4 &&
                seeds.getSeeds()[0].maximumDistance == 1,
            "merge one exact owner across nested recurrence scopes");

  SyncCoverStorageProtocolSeedLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumSeeds = statistics.seeds;
  exact.maximumComponentIncidences = statistics.componentIncidences;
  exact.maximumSlotIncidences = statistics.slotIncidences;
  exact.maximumSccIncidences = statistics.sccIncidences;
  exact.maximumDemandIncidences = statistics.demandIncidences;
  passed &=
      check(buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, exact)
                .isComplete(),
            "accept protocol seeds at their exact bounds");
  SyncCoverStorageProtocolSeedLimits oneLessWork = exact;
  --oneLessWork.maximumWorkUnits;
  const SyncCoverStorageProtocolSeedIndex oneLess =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, oneLessWork);
  passed &= checkTransactionalLimit(
      oneLess, "publish no partial protocol seeds below the work bound");

  const auto checkOneLessIncidence =
      [&](SyncCoverStorageProtocolSeedLimits oneLessLimit,
          std::size_t SyncCoverStorageProtocolSeedLimits::*field,
          std::string_view message) {
        --(oneLessLimit.*field);
        return checkTransactionalLimit(buildSyncCoverStorageProtocolSeedIndex(
                                           graph, lifecycle, oneLessLimit),
                                       message);
      };
  passed &= checkOneLessIncidence(
      exact, &SyncCoverStorageProtocolSeedLimits::maximumSeeds,
      "enforce the seed bound before retaining a new group");
  passed &= checkOneLessIncidence(
      exact, &SyncCoverStorageProtocolSeedLimits::maximumComponentIncidences,
      "enforce the component-incidence bound before scratch allocation");
  passed &= checkOneLessIncidence(
      exact, &SyncCoverStorageProtocolSeedLimits::maximumSlotIncidences,
      "enforce the slot-incidence bound before insertion");
  passed &= checkOneLessIncidence(
      exact, &SyncCoverStorageProtocolSeedLimits::maximumSccIncidences,
      "enforce the SCC-incidence bound before insertion");
  passed &= checkOneLessIncidence(
      exact, &SyncCoverStorageProtocolSeedLimits::maximumDemandIncidences,
      "enforce the demand-incidence bound before insertion");

  SyncCoverStorageProtocolSeedLimits slotLimited = exact;
  slotLimited.maximumSlotIncidences = 1;
  const SyncCoverStorageProtocolSeedIndex slotLimitFailure =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, slotLimited);
  passed &= checkTransactionalLimit(
      slotLimitFailure, "classify slot-incidence exhaustion as truncation");
  SyncCoverStorageProtocolSeedLimits slotWorkLimited = exact;
  slotWorkLimited.maximumWorkUnits =
      slotLimitFailure.getStatistics().workUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, slotWorkLimited),
      "classify work exhaustion during slot insertion as truncation");

  SyncCoverStorageProtocolSeedLimits tinyWork = exact;
  tinyWork.maximumWorkUnits = lifecycle.getComponents().size() - 1;
  const SyncCoverStorageProtocolSeedIndex preflightFailure =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, tinyWork);
  passed &= checkTransactionalLimit(
      preflightFailure,
      "reject a tiny work budget before proportional scratch initialization");
  passed &= check(preflightFailure.getStatistics().workUnits == 0,
                  "retain zero charged work when scratch preflight fails");

  SyncCoverStorageLifecycleLimits lifecycleLimit;
  lifecycleLimit.maximumEdges = 1;
  const SyncCoverStorageLifecycleIndex incomplete =
      buildSyncCoverStorageLifecycleIndex(graph, lifecycleLimit);
  passed &= check(
      buildSyncCoverStorageProtocolSeedIndex(graph, incomplete).getError() ==
          SyncCoverStorageProtocolSeedError::IncompleteLifecycleIndex,
      "reject an incomplete lifecycle input");
  return passed;
}

bool testNestedAndSiblingOwnerChains() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverStorageDomainId nestedDomain =
      takeIndex(graph.addStorageDomain(), passed, "add nested owner domain");
  SyncCoverScopeId parent = 0;
  constexpr unsigned nestedOwners = 8;
  for (unsigned depth = 1; depth <= nestedOwners; ++depth) {
    const unsigned begin = depth * 4;
    parent = takeIndex(
        graph.addScope(parent, true,
                       SyncCoverTimelineInterval{begin, 256 - begin}, true),
        passed, "add nested protocol owner");
    addRoundTrip(graph, parent, nestedDomain, 21, begin, passed);
  }

  const SyncCoverStorageDomainId siblingDomain =
      takeIndex(graph.addStorageDomain(), passed, "add sibling owner domain");
  const SyncCoverScopeId firstSibling = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{600, 664}, true),
      passed, "add first sibling owner");
  const SyncCoverScopeId secondSibling = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{680, 744}, true),
      passed, "add second sibling owner");
  addRoundTrip(graph, firstSibling, siblingDomain, 22, 300, passed);
  addRoundTrip(graph, secondSibling, siblingDomain, 22, 340, passed);
  passed &= check(graph.freezeStructure(), "freeze owner-chain graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  const bool expectedGroups =
      seeds.isComplete() && seeds.getSeeds().size() == 3 &&
      seeds.getSeeds()[0].family == 21 &&
      seeds.getSeeds()[0].components.size() == nestedOwners &&
      seeds.getSeeds()[1].family == 22 &&
      seeds.getSeeds()[1].components.size() == 1 &&
      seeds.getSeeds()[2].family == 22 &&
      seeds.getSeeds()[2].components.size() == 1;
  passed &= check(
      expectedGroups,
      "merge a deep owner chain but keep unconnected siblings separate");
  if (!passed) {
    return false;
  }

  SyncCoverStorageProtocolSeedLimits exact;
  exact.maximumWorkUnits = seeds.getStatistics().workUnits;
  exact.maximumSeeds = seeds.getStatistics().seeds;
  exact.maximumComponentIncidences =
      seeds.getStatistics().componentIncidences;
  exact.maximumSlotIncidences = seeds.getStatistics().slotIncidences;
  exact.maximumSccIncidences = seeds.getStatistics().sccIncidences;
  exact.maximumDemandIncidences = seeds.getStatistics().demandIncidences;
  passed &= check(
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, exact)
          .isComplete(),
      "accept a deep owner chain at the complete work bound");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, exact),
      "reject a deep owner chain one unit below complete work");
  return passed;
}

} // namespace

int main() {
  return testProtocolSeeds() && testNestedAndSiblingOwnerChains() ? 0 : 1;
}
