// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolGroups.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolGroupsTest failure: " << message
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
                "add protocol-group witness");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  passed &=
      check(graph.addDemand(std::move(demand)), "add protocol-group demand");
}

void addStableSeed(SyncCoverGraph &graph, SyncCoverScopeId loop,
                   SyncCoverControlId control, SyncCoverStorageDomainId domain,
                   SyncCoverStorageAccessFamilyId family,
                   SyncCoverStorageInterval extent, unsigned order,
                   bool &passed) {
  SyncCoverNodeId producers[2];
  SyncCoverNodeId consumers[2];
  SyncCoverStorageAccessId writes[2];
  SyncCoverStorageAccessId reads[2];
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    const SyncCoverGuard guard{{{control, alternative}}};
    producers[alternative] = takeIndex(
        graph.addNode(1, 1, loop, order + 2 * alternative, guard, {2}), passed,
        "add stable protocol producer");
    consumers[alternative] = takeIndex(
        graph.addNode(2, 1, loop, order + 2 * alternative + 1, guard, {1}),
        passed, "add stable protocol consumer");
    writes[alternative] =
        takeIndex(graph.addStorageAccess(
                      producers[alternative], domain, family, extent,
                      SyncCoverStorageAccessMode::Write, std::nullopt, true),
                  passed, "add stable protocol write");
    reads[alternative] =
        takeIndex(graph.addStorageAccess(
                      consumers[alternative], domain, family, extent,
                      SyncCoverStorageAccessMode::Read, std::nullopt, true),
                  passed, "add stable protocol read");
    addDemand(graph, producers[alternative], consumers[alternative], loop, 0,
              SyncCoverDemandKind::MemoryRAW, writes[alternative],
              reads[alternative], passed);
  }
  addDemand(graph, consumers[0], producers[1], loop, 1,
            SyncCoverDemandKind::MemoryWAR, reads[0], writes[1], passed);
  addDemand(graph, consumers[1], producers[0], loop, 1,
            SyncCoverDemandKind::MemoryWAR, reads[1], writes[0], passed);
}

void addRotatingSeed(SyncCoverGraph &graph, SyncCoverScopeId loop,
                     SyncCoverControlId control,
                     SyncCoverStorageDomainId domain,
                     SyncCoverStorageAccessFamilyId family,
                     unsigned producerAlternative, unsigned order,
                     bool &passed) {
  const unsigned consumerAlternative = 1 - producerAlternative;
  const SyncCoverGuard producerGuard{{{control, producerAlternative}}};
  const SyncCoverGuard consumerGuard{{{control, consumerAlternative}}};
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, order, producerGuard, {2}), passed,
                "add rotating protocol producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, order + 1, consumerGuard, {1}),
                passed, "add rotating protocol consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(producer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add rotating protocol write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(consumer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add rotating protocol read");
  addDemand(graph, producer, consumer, loop, 1, SyncCoverDemandKind::MemoryRAW,
            write, read, passed);
  addDemand(graph, consumer, producer, loop, 1, SyncCoverDemandKind::MemoryWAR,
            read, write, passed);
}

void addPhasePartialRoundTrip(SyncCoverGraph &graph, SyncCoverScopeId loop,
                              const std::vector<SyncCoverControlId> &controls,
                              unsigned alternative,
                              SyncCoverStorageDomainId domain,
                              SyncCoverStorageAccessFamilyId family,
                              SyncCoverStorageInterval extent, unsigned order,
                              bool &passed) {
  SyncCoverGuard guard;
  for (SyncCoverControlId control : controls) {
    guard.literals.push_back({control, alternative});
  }
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, order, guard, {2}), passed,
                "add phase-partial producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, order + 1, guard, {1}), passed,
                "add phase-partial consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(producer, domain, family, extent,
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add phase-partial write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(consumer, domain, family, extent,
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add phase-partial read");
  addDemand(graph, producer, consumer, loop, 0, SyncCoverDemandKind::MemoryRAW,
            write, read, passed);
  addDemand(graph, consumer, producer, loop, 1, SyncCoverDemandKind::MemoryWAR,
            read, write, passed);
}

SyncCoverGuard makeGuard(const std::vector<SyncCoverControlId> &controls,
                         const std::vector<unsigned> &alternatives) {
  SyncCoverGuard guard;
  for (std::size_t index = 0; index < controls.size(); ++index) {
    guard.literals.push_back({controls[index], alternatives[index]});
  }
  return guard;
}

void addPeriodicCycleSeed(SyncCoverGraph &graph, SyncCoverScopeId loop,
                          const std::vector<SyncCoverControlId> &controls,
                          const std::vector<std::vector<unsigned>> &states,
                          SyncCoverStorageDomainId domain,
                          SyncCoverStorageAccessFamilyId family, unsigned order,
                          bool &passed) {
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverNodeId> consumers;
  std::vector<SyncCoverStorageAccessId> writes;
  std::vector<SyncCoverStorageAccessId> reads;
  for (std::size_t state = 0; state < states.size(); ++state) {
    const SyncCoverGuard guard = makeGuard(controls, states[state]);
    const SyncCoverNodeId producer =
        takeIndex(graph.addNode(1, 1, loop, order + 2 * state, guard, {2}),
                  passed, "add periodic-cycle producer");
    const SyncCoverNodeId consumer =
        takeIndex(graph.addNode(2, 1, loop, order + 2 * state + 1, guard, {1}),
                  passed, "add periodic-cycle consumer");
    producers.push_back(producer);
    consumers.push_back(consumer);
    writes.push_back(
        takeIndex(graph.addStorageAccess(producer, domain, family, {0, 64},
                                         SyncCoverStorageAccessMode::Write,
                                         std::nullopt, true),
                  passed, "add periodic-cycle write"));
    reads.push_back(
        takeIndex(graph.addStorageAccess(consumer, domain, family, {0, 64},
                                         SyncCoverStorageAccessMode::Read,
                                         std::nullopt, true),
                  passed, "add periodic-cycle read"));
    addDemand(graph, producer, consumer, loop, 0,
              SyncCoverDemandKind::MemoryRAW, writes.back(), reads.back(),
              passed);
  }
  for (std::size_t state = 0; state < states.size(); ++state) {
    const std::size_t successor = (state + 1) % states.size();
    addDemand(graph, consumers[state], producers[successor], loop, 1,
              SyncCoverDemandKind::MemoryWAR, reads[state], writes[successor],
              passed);
  }
}

void addUnguardedDistanceOnlySeed(SyncCoverGraph &graph, SyncCoverScopeId loop,
                                  SyncCoverStorageDomainId domain,
                                  SyncCoverStorageAccessFamilyId family,
                                  unsigned order, bool &passed) {
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, order, {}, {2}), passed,
                "add distance-only producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, order + 1, {}, {1}), passed,
                "add distance-only consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(producer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add distance-only write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(consumer, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add distance-only read");
  addDemand(graph, producer, consumer, loop, 1, SyncCoverDemandKind::MemoryRAW,
            write, read, passed);
  addDemand(graph, consumer, producer, loop, 1, SyncCoverDemandKind::MemoryWAR,
            read, write, passed);
}

bool checkTransactionalLimit(const SyncCoverStorageProtocolGroupIndex &index,
                             std::string_view message) {
  return check(index.getError() ==
                       SyncCoverStorageProtocolGroupError::LimitExceeded &&
                   index.getGroups().empty() && index.getStatistics().truncated,
               message);
}

bool testPhaseAwareGroupsAndBounds() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 128}, true), passed,
      "add protocol-group loop");
  const SyncCoverControlId control = takeIndex(
      graph.addControl(2, loop), passed, "add protocol-group periodic control");
  passed &=
      check(graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
            "set protocol-group phase relation");
  const SyncCoverStorageDomainId domains[] = {
      takeIndex(graph.addStorageDomain(), passed, "add stable domain zero"),
      takeIndex(graph.addStorageDomain(), passed, "add stable domain one"),
      takeIndex(graph.addStorageDomain(), passed, "add rotating domain zero"),
      takeIndex(graph.addStorageDomain(), passed, "add rotating domain one"),
  };
  addStableSeed(graph, loop, control, domains[0], 10, {0, 64}, 0, passed);
  addStableSeed(graph, loop, control, domains[1], 11, {0, 64}, 8, passed);
  addRotatingSeed(graph, loop, control, domains[2], 12, 0, 16, passed);
  addRotatingSeed(graph, loop, control, domains[3], 13, 1, 24, passed);
  passed &= check(graph.freezeStructure(), "freeze protocol-group graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  const SyncCoverStorageProtocolGroupIndex groups =
      buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, seeds);
  const bool indexesComplete =
      lifecycle.isComplete() && seeds.isComplete() && groups.isComplete();
  if (!check(indexesComplete, "build complete protocol groups")) {
    return false;
  }
  const auto &descriptions = groups.getGroups();
  const SyncCoverStorageProtocolGroupStatistics &statistics =
      groups.getStatistics();
  passed &= check(
      descriptions.size() == 2 && statistics.eligibleSeeds == 4 &&
          statistics.ineligibleSeeds == 0 && statistics.stableSeeds == 2 &&
          statistics.phaseRotatingSeeds == 2 &&
          statistics.seedIncidences == 4 && statistics.controlIncidences == 2 &&
          statistics.demandIncidences == 12 && statistics.slotIncidences == 4 &&
          statistics.jointStateIncidences == 8 &&
          statistics.maximumGroupSeeds == 2,
      "partition stable and rotating seeds with exact statistics");
  passed &=
      check(descriptions[0].behavior ==
                    SyncCoverStorageProtocolBehavior::StableRoundTrip &&
                descriptions[0].seeds ==
                    std::vector<SyncCoverStorageProtocolSeedId>{0, 1} &&
                descriptions[0].periodicControls ==
                    std::vector<SyncCoverControlId>{control} &&
                descriptions[1].behavior ==
                    SyncCoverStorageProtocolBehavior::PhaseRotatingRoundTrip &&
                descriptions[1].seeds ==
                    std::vector<SyncCoverStorageProtocolSeedId>{2, 3},
            "derive groups without storage-family or syntax-specific rules");

  SyncCoverStorageProtocolGroupLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumGroups = statistics.groups;
  exact.maximumSeedIncidences = statistics.seedIncidences;
  exact.maximumControlIncidences = statistics.controlIncidences;
  exact.maximumDemandIncidences = statistics.demandIncidences;
  exact.maximumSlotIncidences = statistics.slotIncidences;
  exact.maximumJointStateIncidences = statistics.jointStateIncidences;
  exact.maximumReachablePhases = 2;
  passed &= check(
      buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, seeds, exact)
          .isComplete(),
      "accept protocol groups at exact bounds");
  SyncCoverStorageProtocolGroupLimits oneLessWork = exact;
  --oneLessWork.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, seeds,
                                              oneLessWork),
      "reject protocol groups one work unit below the exact bound");
  const auto checkOneLess =
      [&](SyncCoverStorageProtocolGroupLimits oneLess,
          std::size_t SyncCoverStorageProtocolGroupLimits::*field,
          std::string_view message) {
        --(oneLess.*field);
        return checkTransactionalLimit(buildSyncCoverStorageProtocolGroupIndex(
                                           graph, lifecycle, seeds, oneLess),
                                       message);
      };
  passed &=
      checkOneLess(exact, &SyncCoverStorageProtocolGroupLimits::maximumGroups,
                   "enforce protocol-group count bound transactionally");
  passed &= checkOneLess(
      exact, &SyncCoverStorageProtocolGroupLimits::maximumSeedIncidences,
      "enforce protocol-group seed-incidence bound transactionally");
  passed &= checkOneLess(
      exact, &SyncCoverStorageProtocolGroupLimits::maximumControlIncidences,
      "enforce protocol-group control-incidence bound transactionally");
  passed &= checkOneLess(
      exact, &SyncCoverStorageProtocolGroupLimits::maximumDemandIncidences,
      "enforce protocol-group demand-incidence bound transactionally");
  passed &= checkOneLess(
      exact, &SyncCoverStorageProtocolGroupLimits::maximumSlotIncidences,
      "enforce protocol-group slot-incidence bound transactionally");
  passed &= checkOneLess(
      exact, &SyncCoverStorageProtocolGroupLimits::maximumJointStateIncidences,
      "enforce protocol-group joint-state bound transactionally");
  SyncCoverStorageProtocolGroupLimits onePhase = exact;
  onePhase.maximumReachablePhases = 1;
  passed &= checkTransactionalLimit(buildSyncCoverStorageProtocolGroupIndex(
                                        graph, lifecycle, seeds, onePhase),
                                    "bound reachable phase enumeration");

  SyncCoverStorageProtocolSeedLimits seedLimit;
  seedLimit.maximumSeeds = 1;
  const SyncCoverStorageProtocolSeedIndex incompleteSeeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle, seedLimit);
  passed &= check(
      buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, incompleteSeeds)
              .getError() ==
          SyncCoverStorageProtocolGroupError::IncompleteSeedIndex,
      "reject an incomplete protocol-seed input");
  return passed;
}

bool testJointOrbitClassification() {
  bool passed = true;
  SyncCoverGraph correlated;
  const SyncCoverScopeId loop = takeIndex(
      correlated.addScope(0, true, SyncCoverTimelineInterval{0, 128}, true),
      passed, "add correlated loop");
  const SyncCoverControlId first = takeIndex(
      correlated.addControl(2, loop), passed, "add first correlated control");
  const SyncCoverControlId second = takeIndex(
      correlated.addControl(2, loop), passed, "add second correlated control");
  passed &= check(
      correlated.setControlPhaseRelation(first, {loop, 0, {1, 0}, {0, 1}}),
      "set first correlated relation");
  passed &= check(
      correlated.setControlPhaseRelation(second, {loop, 0, {1, 0}, {1, 0}}),
      "set anti-phased correlated relation");
  const SyncCoverStorageDomainId domain =
      takeIndex(correlated.addStorageDomain(), passed, "add correlated domain");
  addPeriodicCycleSeed(correlated, loop, {first, second}, {{0, 0}, {1, 1}},
                       domain, 20, 0, passed);
  passed &= check(correlated.freezeStructure(), "freeze correlated graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex correlatedLifecycle =
      buildSyncCoverStorageLifecycleIndex(correlated);
  const SyncCoverStorageProtocolSeedIndex correlatedSeeds =
      buildSyncCoverStorageProtocolSeedIndex(correlated, correlatedLifecycle);
  const SyncCoverStorageProtocolGroupIndex correlatedGroups =
      buildSyncCoverStorageProtocolGroupIndex(correlated, correlatedLifecycle,
                                              correlatedSeeds);
  const bool antiPhaseDetected =
      correlatedGroups.isComplete() &&
      correlatedGroups.getGroups().size() == 1 &&
      correlatedGroups.getGroups().front().behavior ==
          SyncCoverStorageProtocolBehavior::PhaseRotatingRoundTrip;
  passed &= check(antiPhaseDetected,
                  "preserve correlation across anti-phased controls");

  SyncCoverGraph lcmGraph;
  const SyncCoverScopeId lcmLoop = takeIndex(
      lcmGraph.addScope(0, true, SyncCoverTimelineInterval{0, 128}, true),
      passed, "add joint-period loop");
  const SyncCoverControlId periodTwo = takeIndex(
      lcmGraph.addControl(2, lcmLoop), passed, "add period-two control");
  const SyncCoverControlId periodThree = takeIndex(
      lcmGraph.addControl(3, lcmLoop), passed, "add period-three control");
  passed &= check(
      lcmGraph.setControlPhaseRelation(periodTwo, {lcmLoop, 0, {1, 0}, {0, 1}}),
      "set period-two relation");
  passed &= check(lcmGraph.setControlPhaseRelation(
                      periodThree, {lcmLoop, 0, {1, 2, 0}, {0, 1, 2}}),
                  "set period-three relation");
  const SyncCoverStorageDomainId lcmDomain =
      takeIndex(lcmGraph.addStorageDomain(), passed, "add joint-period domain");
  const std::vector<std::vector<unsigned>> jointStates = {
      {0, 0}, {1, 1}, {0, 2}, {1, 0}, {0, 1}, {1, 2}};
  addPeriodicCycleSeed(lcmGraph, lcmLoop, {periodTwo, periodThree}, jointStates,
                       lcmDomain, 21, 0, passed);
  passed &= check(lcmGraph.freezeStructure(), "freeze joint-period graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex lcmLifecycle =
      buildSyncCoverStorageLifecycleIndex(lcmGraph);
  const SyncCoverStorageProtocolSeedIndex lcmSeeds =
      buildSyncCoverStorageProtocolSeedIndex(lcmGraph, lcmLifecycle);
  SyncCoverStorageProtocolGroupLimits sixPhases;
  sixPhases.maximumReachablePhases = 6;
  const SyncCoverStorageProtocolGroupIndex exactLcm =
      buildSyncCoverStorageProtocolGroupIndex(lcmGraph, lcmLifecycle, lcmSeeds,
                                              sixPhases);
  passed &= check(exactLcm.isComplete() &&
                      exactLcm.getGroups().front().behavior ==
                          SyncCoverStorageProtocolBehavior::StableRoundTrip,
                  "classify the complete six-state joint orbit");
  SyncCoverStorageProtocolGroupLimits fivePhases = sixPhases;
  fivePhases.maximumReachablePhases = 5;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolGroupIndex(lcmGraph, lcmLifecycle, lcmSeeds,
                                              fivePhases),
      "reject a joint orbit one phase below its LCM bound");

  SyncCoverGraph distanceOnly;
  const SyncCoverScopeId distanceLoop = takeIndex(
      distanceOnly.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
      passed, "add distance-only loop");
  const SyncCoverStorageDomainId distanceDomain = takeIndex(
      distanceOnly.addStorageDomain(), passed, "add distance-only domain");
  addUnguardedDistanceOnlySeed(distanceOnly, distanceLoop, distanceDomain, 22,
                               0, passed);
  passed &= check(distanceOnly.freezeStructure(), "freeze distance-only graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex distanceLifecycle =
      buildSyncCoverStorageLifecycleIndex(distanceOnly);
  const SyncCoverStorageProtocolSeedIndex distanceSeeds =
      buildSyncCoverStorageProtocolSeedIndex(distanceOnly, distanceLifecycle);
  const SyncCoverStorageProtocolGroupIndex distanceGroups =
      buildSyncCoverStorageProtocolGroupIndex(distanceOnly, distanceLifecycle,
                                              distanceSeeds);
  return passed &&
         check(distanceGroups.isComplete() &&
                   distanceGroups.getGroups().size() == 1 &&
                   distanceGroups.getGroups().front().behavior ==
                       SyncCoverStorageProtocolBehavior::PhaseRotatingRoundTrip,
               "do not call an unguarded distance-only lifecycle stable");
}

bool testGlobalMaskNormalizationAndWorkBounds() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 256}, true), passed,
      "add global-mask outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, true, SyncCoverTimelineInterval{0, 256}, true),
      passed, "add global-mask inner loop");
  const SyncCoverControlId control =
      takeIndex(graph.addControl(2, outer), passed, "add global-mask control");
  passed &=
      check(graph.setControlPhaseRelation(control, {outer, 0, {1, 0}, {0, 1}}),
            "set global-mask phase relation");
  const SyncCoverStorageDomainId inPhaseDomain = takeIndex(
      graph.addStorageDomain(), passed, "add in-phase lifecycle domain");
  const SyncCoverStorageDomainId antiPhaseDomain = takeIndex(
      graph.addStorageDomain(), passed, "add anti-phase lifecycle domain");
  const SyncCoverStorageDomainId shiftedDomain = takeIndex(
      graph.addStorageDomain(), passed, "add shifted lifecycle domain");
  addPhasePartialRoundTrip(graph, outer, {control}, 0, inPhaseDomain, 50,
                           {0, 64}, 0, passed);
  addPhasePartialRoundTrip(graph, inner, {control}, 0, inPhaseDomain, 50,
                           {0, 64}, 2, passed);
  addPhasePartialRoundTrip(graph, outer, {control}, 0, antiPhaseDomain, 51,
                           {0, 64}, 4, passed);
  addPhasePartialRoundTrip(graph, inner, {control}, 1, antiPhaseDomain, 51,
                           {0, 64}, 6, passed);
  addPhasePartialRoundTrip(graph, outer, {control}, 1, shiftedDomain, 52,
                           {0, 64}, 8, passed);
  addPhasePartialRoundTrip(graph, inner, {control}, 1, shiftedDomain, 52,
                           {0, 64}, 10, passed);
  passed &= check(graph.freezeStructure(), "freeze global-mask graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  const SyncCoverStorageProtocolGroupIndex groups =
      buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, seeds);
  passed &= check(
      groups.isComplete() && groups.getGroups().size() == 2 &&
          groups.getGroups()[0].seeds ==
              std::vector<SyncCoverStorageProtocolSeedId>{0, 2} &&
          groups.getGroups()[1].seeds ==
              std::vector<SyncCoverStorageProtocolSeedId>{1},
      "preserve relative SCC phases while accepting one global phase shift");

  SyncCoverGraph bounded;
  const SyncCoverScopeId boundedOuter = takeIndex(
      bounded.addScope(0, true, SyncCoverTimelineInterval{0, 512}, true),
      passed, "add work-bound outer loop");
  std::vector<SyncCoverControlId> controls;
  std::vector<std::size_t> nextPhase(16);
  std::vector<unsigned> activeAlternative(16);
  for (std::size_t phase = 0; phase < 16; ++phase) {
    nextPhase[phase] = (phase + 1) % 16;
    activeAlternative[phase] = phase % 2;
  }
  for (unsigned index = 0; index < 8; ++index) {
    const SyncCoverControlId current = takeIndex(
        bounded.addControl(2, boundedOuter), passed, "add work-bound control");
    passed &=
        check(bounded.setControlPhaseRelation(
                  current, {boundedOuter, 0, nextPhase, activeAlternative}),
              "set work-bound phase relation");
    controls.push_back(current);
  }
  const SyncCoverStorageDomainId boundedDomain =
      takeIndex(bounded.addStorageDomain(), passed, "add work-bound domain");
  SyncCoverScopeId scope = boundedOuter;
  for (unsigned scc = 0; scc < 4; ++scc) {
    if (scc != 0) {
      scope =
          takeIndex(bounded.addScope(scope, true,
                                     SyncCoverTimelineInterval{0, 512}, true),
                    passed, "add work-bound nested loop");
    }
    addPhasePartialRoundTrip(bounded, scope, controls, 0, boundedDomain, 60,
                             {0, 64}, 32 + 2 * scc, passed);
  }
  passed &= check(bounded.freezeStructure(), "freeze work-bound graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex boundedLifecycle =
      buildSyncCoverStorageLifecycleIndex(bounded);
  const SyncCoverStorageProtocolSeedIndex boundedSeeds =
      buildSyncCoverStorageProtocolSeedIndex(bounded, boundedLifecycle);
  SyncCoverStorageProtocolGroupLimits phaseLimit;
  phaseLimit.maximumReachablePhases = 16;
  const SyncCoverStorageProtocolGroupIndex boundedGroups =
      buildSyncCoverStorageProtocolGroupIndex(bounded, boundedLifecycle,
                                              boundedSeeds, phaseLimit);
  if (!check(boundedGroups.isComplete(),
             "build long-period many-SCC protocol group")) {
    return false;
  }
  const SyncCoverStorageProtocolGroupStatistics &statistics =
      boundedGroups.getStatistics();
  SyncCoverStorageProtocolGroupLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumGroups = statistics.groups;
  exact.maximumSeedIncidences = statistics.seedIncidences;
  exact.maximumControlIncidences = statistics.controlIncidences;
  exact.maximumDemandIncidences = statistics.demandIncidences;
  exact.maximumSlotIncidences = statistics.slotIncidences;
  exact.maximumJointStateIncidences = statistics.jointStateIncidences;
  exact.maximumReachablePhases = 16;
  passed &= check(buildSyncCoverStorageProtocolGroupIndex(
                      bounded, boundedLifecycle, boundedSeeds, exact)
                      .isComplete(),
                  "accept long guards and global mask rotation at exact work");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolGroupIndex(bounded, boundedLifecycle,
                                              boundedSeeds, exact),
      "bound long-guard matching and global mask rotation work");

  SyncCoverGraph deep;
  const SyncCoverScopeId deepOuter = takeIndex(
      deep.addScope(0, true, SyncCoverTimelineInterval{0, 1024}, true), passed,
      "add deep ancestry outer loop");
  SyncCoverScopeId deepScope = deepOuter;
  for (unsigned depth = 0; depth < 32; ++depth) {
    deepScope =
        takeIndex(deep.addScope(deepScope, true,
                                SyncCoverTimelineInterval{0, 1024}, true),
                  passed, "add deep ancestry loop");
  }
  const SyncCoverControlId deepControl = takeIndex(
      deep.addControl(2, deepScope), passed, "add deep ancestry control");
  passed &= check(
      deep.setControlPhaseRelation(deepControl, {deepScope, 0, {1, 0}, {0, 1}}),
      "set deep ancestry phase relation");
  const SyncCoverStorageDomainId deepDomain =
      takeIndex(deep.addStorageDomain(), passed, "add deep ancestry domain");
  addUnguardedDistanceOnlySeed(deep, deepOuter, deepDomain, 70, 0, passed);
  addPhasePartialRoundTrip(deep, deepScope, {deepControl}, 0, deepDomain, 70,
                           {0, 64}, 2, passed);
  passed &= check(deep.freezeStructure(), "freeze deep ancestry graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex deepLifecycle =
      buildSyncCoverStorageLifecycleIndex(deep);
  const SyncCoverStorageProtocolSeedIndex deepSeeds =
      buildSyncCoverStorageProtocolSeedIndex(deep, deepLifecycle);
  const SyncCoverStorageProtocolGroupIndex deepGroups =
      buildSyncCoverStorageProtocolGroupIndex(deep, deepLifecycle, deepSeeds);
  if (!check(deepGroups.isComplete(), "build deep ancestry protocol group")) {
    return false;
  }
  const SyncCoverStorageProtocolGroupStatistics &deepStatistics =
      deepGroups.getStatistics();
  SyncCoverStorageProtocolGroupLimits deepExact;
  deepExact.maximumWorkUnits = deepStatistics.workUnits;
  deepExact.maximumGroups = deepStatistics.groups;
  deepExact.maximumSeedIncidences = deepStatistics.seedIncidences;
  deepExact.maximumControlIncidences = deepStatistics.controlIncidences;
  deepExact.maximumDemandIncidences = deepStatistics.demandIncidences;
  deepExact.maximumSlotIncidences = deepStatistics.slotIncidences;
  deepExact.maximumJointStateIncidences = deepStatistics.jointStateIncidences;
  deepExact.maximumReachablePhases = 2;
  passed &= check(buildSyncCoverStorageProtocolGroupIndex(deep, deepLifecycle,
                                                          deepSeeds, deepExact)
                      .isComplete(),
                  "accept deep scope ancestry at its exact work bound");
  --deepExact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolGroupIndex(deep, deepLifecycle, deepSeeds,
                                              deepExact),
      "meter deep scope ancestry before containment queries");
  return passed;
}

SyncCoverStorageProtocolGroupIndex
buildSlotPartition(SyncCoverStorageInterval firstExtent,
                   SyncCoverStorageInterval secondExtent,
                   const SyncCoverStorageProtocolGroupLimits &limits = {}) {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
                passed, "add slot-partition loop");
  const SyncCoverControlId control = takeIndex(
      graph.addControl(2, loop), passed, "add slot-partition control");
  passed &=
      check(graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
            "set slot-partition phase relation");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add slot-partition domain");
  addStableSeed(graph, loop, control, domain, 30, firstExtent, 0, passed);
  addStableSeed(graph, loop, control, domain, 31, secondExtent, 8, passed);
  passed &= check(graph.freezeStructure(), "freeze slot-partition graph");
  if (!passed) {
    return {};
  }
  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  return buildSyncCoverStorageProtocolGroupIndex(graph, lifecycle, seeds,
                                                 limits);
}

bool testSlotPartitioningAndMixedSccs() {
  bool passed = true;
  const SyncCoverStorageProtocolGroupIndex overlapping =
      buildSlotPartition({0, 64}, {32, 96});
  const SyncCoverStorageProtocolGroupIndex touching =
      buildSlotPartition({0, 64}, {64, 128});
  passed &=
      check(overlapping.isComplete() && overlapping.getGroups().size() == 2,
            "separate overlapping exact slots");
  passed &= check(touching.isComplete() && touching.getGroups().size() == 1 &&
                      touching.getGroups().front().seeds.size() == 2,
                  "group touching but disjoint exact slots");

  SyncCoverGraph internallyOverlapping;
  const SyncCoverScopeId overlapOuter =
      takeIndex(internallyOverlapping.addScope(
                    0, true, SyncCoverTimelineInterval{0, 128}, true),
                passed, "add internal-overlap outer loop");
  const SyncCoverScopeId overlapInner = takeIndex(
      internallyOverlapping.addScope(overlapOuter, true,
                                     SyncCoverTimelineInterval{8, 120}, true),
      passed, "add internal-overlap inner loop");
  const SyncCoverControlId overlapControl =
      takeIndex(internallyOverlapping.addControl(2, overlapOuter), passed,
                "add internal-overlap control");
  passed &= check(internallyOverlapping.setControlPhaseRelation(
                      overlapControl, {overlapOuter, 0, {1, 0}, {0, 1}}),
                  "set internal-overlap phase relation");
  const SyncCoverStorageDomainId overlapDomain =
      takeIndex(internallyOverlapping.addStorageDomain(), passed,
                "add internal-overlap domain");
  addPhasePartialRoundTrip(internallyOverlapping, overlapOuter,
                           {overlapControl}, 0, overlapDomain, 32, {0, 64}, 0,
                           passed);
  addPhasePartialRoundTrip(internallyOverlapping, overlapInner,
                           {overlapControl}, 0, overlapDomain, 32, {32, 96}, 8,
                           passed);
  passed &= check(internallyOverlapping.freezeStructure(),
                  "freeze internal-overlap graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex overlapLifecycle =
      buildSyncCoverStorageLifecycleIndex(internallyOverlapping);
  const SyncCoverStorageProtocolSeedIndex overlapSeeds =
      buildSyncCoverStorageProtocolSeedIndex(internallyOverlapping,
                                             overlapLifecycle);
  const SyncCoverStorageProtocolGroupIndex overlapGroups =
      buildSyncCoverStorageProtocolGroupIndex(internallyOverlapping,
                                              overlapLifecycle, overlapSeeds);
  passed &=
      check(overlapGroups.isComplete() && overlapGroups.getGroups().empty() &&
                overlapGroups.getStatistics().ineligibleSeeds == 1,
            "reject a seed with internally overlapping exact slots");
  if (!passed) {
    return false;
  }
  std::vector<SyncCoverStorageProtocolGroup> reportGroups(3);
  reportGroups[0].seeds = {0, 1};
  reportGroups[0].behaviorSignature = {0, 1};
  reportGroups[1].seeds = {2, 3};
  reportGroups[1].behaviorSignature = {2, 3};
  reportGroups[2].seeds = {4};
  reportGroups[2].behaviorSignature = {4};
  const SyncCoverStorageProtocolGroupPrefix exactReportPrefix =
      boundedSyncCoverStorageProtocolGroupPrefix(reportGroups, 3, 5, 5);
  const SyncCoverStorageProtocolGroupPrefix seedLimitedReportPrefix =
      boundedSyncCoverStorageProtocolGroupPrefix(reportGroups, 3, 4, 5);
  const SyncCoverStorageProtocolGroupPrefix groupLimitedReportPrefix =
      boundedSyncCoverStorageProtocolGroupPrefix(reportGroups, 2, 5, 5);
  const SyncCoverStorageProtocolGroupPrefix signatureLimitedReportPrefix =
      boundedSyncCoverStorageProtocolGroupPrefix(reportGroups, 3, 5, 4);
  std::vector<SyncCoverStorageProtocolGroup> oversizedSignatureGroup(1);
  oversizedSignatureGroup[0].seeds = {0};
  oversizedSignatureGroup[0].behaviorSignature = {0, 1, 2, 3, 4, 5};
  const SyncCoverStorageProtocolGroupPrefix oversizedSignatureReportPrefix =
      boundedSyncCoverStorageProtocolGroupPrefix(oversizedSignatureGroup, 1, 1,
                                                 5);
  passed &= check(
      exactReportPrefix.retainedGroups == 3 &&
          exactReportPrefix.retainedSeedIncidences == 5 &&
          exactReportPrefix.retainedBehaviorSignatureEntries == 5 &&
          !exactReportPrefix.truncated &&
          seedLimitedReportPrefix.retainedGroups == 2 &&
          seedLimitedReportPrefix.retainedSeedIncidences == 4 &&
          seedLimitedReportPrefix.truncated &&
          groupLimitedReportPrefix.retainedGroups == 2 &&
          groupLimitedReportPrefix.truncated &&
          signatureLimitedReportPrefix.retainedGroups == 2 &&
          signatureLimitedReportPrefix.retainedBehaviorSignatureEntries == 4 &&
          signatureLimitedReportPrefix.truncated &&
          oversizedSignatureReportPrefix.retainedGroups == 0 &&
          oversizedSignatureReportPrefix.truncated,
      "bound aggregate report detail copying without partial groups");
  SyncCoverStorageProtocolGroupLimits exact;
  const auto &overlapStatistics = overlapping.getStatistics();
  exact.maximumWorkUnits = overlapStatistics.workUnits;
  exact.maximumGroups = overlapStatistics.groups;
  exact.maximumSeedIncidences = overlapStatistics.seedIncidences;
  exact.maximumControlIncidences = overlapStatistics.controlIncidences;
  exact.maximumDemandIncidences = overlapStatistics.demandIncidences;
  exact.maximumSlotIncidences = overlapStatistics.slotIncidences;
  exact.maximumJointStateIncidences = overlapStatistics.jointStateIncidences;
  exact.maximumReachablePhases = 2;
  passed &= check(buildSlotPartition({0, 64}, {32, 96}, exact).isComplete(),
                  "accept an overlap scan at its exact work bound");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSlotPartition({0, 64}, {32, 96}, exact),
      "stop transactionally when overlap-scan work is exhausted");

  SyncCoverGraph mixed;
  const SyncCoverScopeId outer =
      takeIndex(mixed.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
                passed, "add mixed outer loop");
  const SyncCoverScopeId inner = takeIndex(
      mixed.addScope(outer, true, SyncCoverTimelineInterval{8, 48}, true),
      passed, "add mixed inner loop");
  const SyncCoverControlId control =
      takeIndex(mixed.addControl(2, outer), passed, "add mixed control");
  passed &=
      check(mixed.setControlPhaseRelation(control, {outer, 0, {1, 0}, {0, 1}}),
            "set mixed phase relation");
  const SyncCoverStorageDomainId domain =
      takeIndex(mixed.addStorageDomain(), passed, "add mixed domain");
  addStableSeed(mixed, outer, control, domain, 40, {0, 64}, 0, passed);
  addUnguardedDistanceOnlySeed(mixed, inner, domain, 40, 16, passed);
  passed &= check(mixed.freezeStructure(), "freeze mixed-SCC graph");
  if (!passed) {
    return false;
  }
  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(mixed);
  const SyncCoverStorageProtocolSeedIndex seeds =
      buildSyncCoverStorageProtocolSeedIndex(mixed, lifecycle);
  const SyncCoverStorageProtocolGroupIndex groups =
      buildSyncCoverStorageProtocolGroupIndex(mixed, lifecycle, seeds);
  return check(groups.isComplete() && groups.getGroups().size() == 1 &&
                   groups.getGroups().front().behavior ==
                       SyncCoverStorageProtocolBehavior::PhaseRotatingRoundTrip,
               "let a rotating child SCC prevent stable classification");
}

} // namespace

int main() {
  return testPhaseAwareGroupsAndBounds() && testJointOrbitClassification() &&
                 testGlobalMaskNormalizationAndWorkBounds() &&
                 testSlotPartitioningAndMixedSccs()
             ? 0
             : 1;
}
