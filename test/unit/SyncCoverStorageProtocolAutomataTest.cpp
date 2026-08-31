// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolAutomata.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageProtocolAutomataTest failure: " << message
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
                "add protocol-automaton witness");
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  passed &= check(graph.addDemand(std::move(demand)),
                  "add protocol-automaton demand");
}

struct ProtocolInputs {
  SyncCoverGraph graph;
  SyncCoverStorageLifecycleIndex lifecycle;
  SyncCoverStorageProtocolSeedIndex seeds;
  SyncCoverStorageProtocolGroupIndex groups;
  bool valid = false;
};

enum class ProjectionScopeRelation {
  SameLoop,
  RecurrenceNestedUnderPhase,
  RecurrenceEnclosesPhase,
};

void addUnguardedRoundTrip(SyncCoverGraph &graph, SyncCoverScopeId loop,
                           SyncCoverStorageDomainId domain,
                           SyncCoverStorageAccessFamilyId family,
                           SyncCoverStorageInterval extent,
                           std::size_t orderBase, bool &passed) {
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, orderBase, {}, {2}), passed,
                "add unguarded protocol producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, orderBase + 1, {}, {1}), passed,
                "add unguarded protocol consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(producer, domain, family, extent,
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add unguarded protocol write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(consumer, domain, family, extent,
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add unguarded protocol read");
  addDemand(graph, producer, consumer, loop, 0, SyncCoverDemandKind::MemoryRAW,
            write, read, passed);
  addDemand(graph, consumer, producer, loop, 1, SyncCoverDemandKind::MemoryWAR,
            read, write, passed);
}

void addGuardedRoundTrip(SyncCoverGraph &graph, SyncCoverScopeId loop,
                         SyncCoverControlId control,
                         SyncCoverStorageDomainId domain,
                         SyncCoverStorageAccessFamilyId family,
                         SyncCoverStorageInterval extent, std::size_t orderBase,
                         bool &passed) {
  SyncCoverNodeId producers[2];
  SyncCoverNodeId consumers[2];
  SyncCoverStorageAccessId writes[2];
  SyncCoverStorageAccessId reads[2];
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    const SyncCoverGuard guard{{{control, alternative}}};
    producers[alternative] = takeIndex(
        graph.addNode(1, 1, loop, orderBase + 2 * alternative, guard, {2}),
        passed, "add guarded protocol producer");
    consumers[alternative] = takeIndex(
        graph.addNode(2, 1, loop, orderBase + 2 * alternative + 1, guard, {1}),
        passed, "add guarded protocol consumer");
    writes[alternative] =
        takeIndex(graph.addStorageAccess(
                      producers[alternative], domain, family, extent,
                      SyncCoverStorageAccessMode::Write, std::nullopt, true),
                  passed, "add guarded protocol write");
    reads[alternative] =
        takeIndex(graph.addStorageAccess(
                      consumers[alternative], domain, family, extent,
                      SyncCoverStorageAccessMode::Read, std::nullopt, true),
                  passed, "add guarded protocol read");
    addDemand(graph, producers[alternative], consumers[alternative], loop, 0,
              SyncCoverDemandKind::MemoryRAW, writes[alternative],
              reads[alternative], passed);
  }
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    const unsigned successor = 1 - alternative;
    addDemand(graph, consumers[alternative], producers[successor], loop, 1,
              SyncCoverDemandKind::MemoryWAR, reads[alternative],
              writes[successor], passed);
  }
}

ProtocolInputs buildTwoPhaseRoundTrip(unsigned releaseDistance,
                                      bool releaseToSuccessorPhase) {
  ProtocolInputs result;
  bool passed = true;
  const SyncCoverScopeId loop = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add protocol-automaton loop");
  const SyncCoverControlId control =
      takeIndex(result.graph.addControl(2, loop), passed,
                "add protocol-automaton control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {loop, 0, {1, 0}, {0, 1}}),
      "set protocol-automaton phase relation");
  const SyncCoverStorageDomainId domain = takeIndex(
      result.graph.addStorageDomain(), passed, "add protocol-automaton domain");
  SyncCoverNodeId producers[2];
  SyncCoverNodeId consumers[2];
  SyncCoverStorageAccessId writes[2];
  SyncCoverStorageAccessId reads[2];
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    const SyncCoverGuard guard{{{control, alternative}}};
    producers[alternative] =
        takeIndex(result.graph.addNode(1, 1, loop, 2 * alternative, guard, {2}),
                  passed, "add protocol-automaton producer");
    consumers[alternative] = takeIndex(
        result.graph.addNode(2, 1, loop, 2 * alternative + 1, guard, {1}),
        passed, "add protocol-automaton consumer");
    writes[alternative] =
        takeIndex(result.graph.addStorageAccess(
                      producers[alternative], domain, 10, {0, 64},
                      SyncCoverStorageAccessMode::Write, std::nullopt, true),
                  passed, "add protocol-automaton write");
    reads[alternative] =
        takeIndex(result.graph.addStorageAccess(
                      consumers[alternative], domain, 10, {0, 64},
                      SyncCoverStorageAccessMode::Read, std::nullopt, true),
                  passed, "add protocol-automaton read");
    addDemand(result.graph, producers[alternative], consumers[alternative],
              loop, 0, SyncCoverDemandKind::MemoryRAW, writes[alternative],
              reads[alternative], passed);
  }
  for (unsigned alternative = 0; alternative < 2; ++alternative) {
    const unsigned targetAlternative =
        releaseToSuccessorPhase ? 1 - alternative : alternative;
    addDemand(result.graph, consumers[alternative],
              producers[targetAlternative], loop, releaseDistance,
              SyncCoverDemandKind::MemoryWAR, reads[alternative],
              writes[targetAlternative], passed);
  }
  passed &=
      check(result.graph.freezeStructure(), "freeze protocol-automaton graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  result.seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  result.groups = buildSyncCoverStorageProtocolGroupIndex(
      result.graph, result.lifecycle, result.seeds);
  result.valid = result.lifecycle.isComplete() && result.seeds.isComplete() &&
                 result.groups.isComplete();
  return result;
}

ProtocolInputs buildOuterRecurrenceUnderInnerPhase(unsigned nestingDepth = 1) {
  ProtocolInputs result;
  bool passed = true;
  const SyncCoverScopeId outer = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add outer protocol-automaton loop");
  SyncCoverScopeId inner = outer;
  for (unsigned depth = 0; depth < nestingDepth; ++depth) {
    inner = takeIndex(result.graph.addScope(
                          inner, true, SyncCoverTimelineInterval{0, 64}, true),
                      passed, "add inner protocol-automaton loop");
  }
  const SyncCoverControlId control =
      takeIndex(result.graph.addControl(2, inner), passed,
                "add inner protocol-automaton control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {inner, 0, {1, 0}, {0, 1}}),
      "set inner protocol-automaton phase relation");
  const SyncCoverStorageDomainId domain =
      takeIndex(result.graph.addStorageDomain(), passed,
                "add nested protocol-automaton domain");
  const SyncCoverGuard producerGuard{{{control, 0}}};
  const SyncCoverGuard consumerGuard{{{control, 1}}};
  const SyncCoverNodeId producer =
      takeIndex(result.graph.addNode(1, 1, inner, 2, producerGuard, {2}),
                passed, "add nested protocol producer");
  const SyncCoverNodeId consumer =
      takeIndex(result.graph.addNode(2, 1, inner, 3, consumerGuard, {1}),
                passed, "add nested protocol consumer");
  const SyncCoverStorageAccessId write =
      takeIndex(result.graph.addStorageAccess(producer, domain, 11, {0, 64},
                                              SyncCoverStorageAccessMode::Write,
                                              std::nullopt, true),
                passed, "add nested protocol write");
  const SyncCoverStorageAccessId read =
      takeIndex(result.graph.addStorageAccess(consumer, domain, 11, {0, 64},
                                              SyncCoverStorageAccessMode::Read,
                                              std::nullopt, true),
                passed, "add nested protocol read");
  addDemand(result.graph, producer, consumer, outer, 1,
            SyncCoverDemandKind::MemoryRAW, write, read, passed);
  addDemand(result.graph, consumer, producer, outer, 1,
            SyncCoverDemandKind::MemoryWAR, read, write, passed);
  passed &= check(result.graph.freezeStructure(),
                  "freeze nested protocol-automaton graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  result.seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  result.groups = buildSyncCoverStorageProtocolGroupIndex(
      result.graph, result.lifecycle, result.seeds);
  result.valid = result.lifecycle.isComplete() && result.seeds.isComplete() &&
                 result.groups.isComplete();
  return result;
}

ProtocolInputs buildSiblingScopeProtocolGroup() {
  ProtocolInputs result;
  bool passed = true;
  const SyncCoverScopeId outer = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add sibling protocol outer loop");
  const SyncCoverScopeId left =
      takeIndex(result.graph.addScope(outer, true,
                                      SyncCoverTimelineInterval{0, 64}, true),
                passed, "add sibling protocol left loop");
  const SyncCoverScopeId right =
      takeIndex(result.graph.addScope(outer, true,
                                      SyncCoverTimelineInterval{0, 64}, true),
                passed, "add sibling protocol right loop");
  const SyncCoverControlId control = takeIndex(
      result.graph.addControl(2, left), passed, "add sibling protocol control");
  passed &= check(
      result.graph.setControlPhaseRelation(control, {left, 0, {1, 0}, {0, 1}}),
      "set sibling protocol phase relation");
  const SyncCoverStorageDomainId domain = takeIndex(
      result.graph.addStorageDomain(), passed, "add sibling protocol domain");
  constexpr SyncCoverStorageAccessFamilyId family = 12;
  addUnguardedRoundTrip(result.graph, outer, domain, family, {0, 64}, 0,
                        passed);
  addGuardedRoundTrip(result.graph, left, control, domain, family, {64, 128}, 2,
                      passed);
  addUnguardedRoundTrip(result.graph, right, domain, family, {128, 192}, 6,
                        passed);
  passed &= check(result.graph.freezeStructure(),
                  "freeze sibling protocol-automaton graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  result.seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  result.groups = buildSyncCoverStorageProtocolGroupIndex(
      result.graph, result.lifecycle, result.seeds);
  result.valid = result.lifecycle.isComplete() && result.seeds.isComplete() &&
                 result.groups.isComplete();
  return result;
}

ProtocolInputs
buildProjectionScopeProtocolGroup(ProjectionScopeRelation relation) {
  ProtocolInputs result;
  bool passed = true;
  const SyncCoverScopeId outer = takeIndex(
      result.graph.addScope(0, true, SyncCoverTimelineInterval{0, 64}, true),
      passed, "add projection outer loop");
  SyncCoverScopeId phaseLoop = outer;
  SyncCoverScopeId recurrenceLoop = outer;
  if (relation != ProjectionScopeRelation::SameLoop) {
    const SyncCoverScopeId inner =
        takeIndex(result.graph.addScope(outer, true,
                                        SyncCoverTimelineInterval{0, 64}, true),
                  passed, "add projection inner loop");
    if (relation == ProjectionScopeRelation::RecurrenceNestedUnderPhase) {
      recurrenceLoop = inner;
    } else {
      phaseLoop = inner;
    }
  }
  const SyncCoverControlId control =
      takeIndex(result.graph.addControl(2, phaseLoop), passed,
                "add projection phase control");
  passed &= check(result.graph.setControlPhaseRelation(
                      control, {phaseLoop, 0, {1, 0}, {0, 1}}),
                  "set projection phase relation");
  const SyncCoverStorageDomainId domain = takeIndex(
      result.graph.addStorageDomain(), passed, "add projection storage domain");
  constexpr SyncCoverStorageAccessFamilyId family = 13;
  addGuardedRoundTrip(result.graph, phaseLoop, control, domain, family, {0, 64},
                      0, passed);
  addUnguardedRoundTrip(result.graph, recurrenceLoop, domain, family, {64, 128},
                        4, passed);
  passed &= check(result.graph.freezeStructure(),
                  "freeze projection-scope protocol graph");
  if (!passed) {
    return result;
  }
  result.lifecycle = buildSyncCoverStorageLifecycleIndex(result.graph);
  result.seeds =
      buildSyncCoverStorageProtocolSeedIndex(result.graph, result.lifecycle);
  result.groups = buildSyncCoverStorageProtocolGroupIndex(
      result.graph, result.lifecycle, result.seeds);
  result.valid = result.lifecycle.isComplete() && result.seeds.isComplete() &&
                 result.groups.isComplete();
  return result;
}

bool checkTransactionalLimit(
    const SyncCoverStorageProtocolAutomatonIndex &index,
    std::string_view message) {
  return check(index.getError() ==
                       SyncCoverStorageProtocolAutomatonError::LimitExceeded &&
                   index.getAutomata().empty() &&
                   index.getStatistics().truncated,
               message);
}

bool testDistanceShiftedProjectionAndBounds() {
  ProtocolInputs inputs = buildTwoPhaseRoundTrip(1, true);
  if (!check(inputs.valid, "build protocol-automaton inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonIndex index =
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups);
  const bool oneAutomaton =
      index.isComplete() && index.getAutomata().size() == 1;
  if (!check(oneAutomaton, "build one reachable protocol automaton")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomaton &automaton =
      index.getAutomata().front();
  const SyncCoverStorageProtocolAutomatonStatistics &statistics =
      index.getStatistics();
  bool passed = check(
      automaton.stateCount == 2 && automaton.transfers.size() == 4 &&
          automaton.statePairIncidences == 4 &&
          automaton.maximumDistance == 1 && statistics.eligibleGroups == 1 &&
          statistics.ineligibleGroups == 0 && statistics.states == 2 &&
          statistics.transfers == 4 && statistics.statePairIncidences == 4,
      "project ready and distance-one release edges onto exact phases");
  for (const SyncCoverStorageProtocolTransfer &transfer : automaton.transfers) {
    passed &=
        check(transfer.activeStatePairs.size() == 1,
              "retain one reachable endpoint-state pair per guarded transfer");
  }

  SyncCoverStorageProtocolAutomatonLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumAutomata = statistics.automata;
  exact.maximumStates = statistics.states;
  exact.maximumTransfers = statistics.transfers;
  exact.maximumStatePairIncidences = statistics.statePairIncidences;
  exact.maximumLanes = 1;
  passed &= check(
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups, exact)
          .isComplete(),
      "accept protocol automata at exact bounds");
  SyncCoverStorageProtocolAutomatonLimits oneLessWork = exact;
  --oneLessWork.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups,
          oneLessWork),
      "reject protocol automata one work unit below the exact bound");
  SyncCoverStorageProtocolAutomatonLimits oneLessStates = exact;
  --oneLessStates.maximumStates;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups,
          oneLessStates),
      "enforce the aggregate state bound transactionally");
  SyncCoverStorageProtocolAutomatonLimits oneLessTransfers = exact;
  --oneLessTransfers.maximumTransfers;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups,
          oneLessTransfers),
      "enforce the aggregate transfer bound transactionally");
  SyncCoverStorageProtocolAutomatonLimits oneLessActive = exact;
  --oneLessActive.maximumStatePairIncidences;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups,
          oneLessActive),
      "enforce the state-pair incidence bound transactionally");
  return passed;
}

bool testUnreachableAndLaneLimitedGroups() {
  ProtocolInputs unreachable = buildTwoPhaseRoundTrip(1, false);
  if (!check(unreachable.valid, "build unreachable protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonIndex unreachableIndex =
      buildSyncCoverStorageProtocolAutomatonIndex(
          unreachable.graph, unreachable.lifecycle, unreachable.seeds,
          unreachable.groups);
  bool passed = check(
      unreachableIndex.isComplete() && unreachableIndex.getAutomata().empty() &&
          unreachableIndex.getStatistics().eligibleGroups == 0 &&
          unreachableIndex.getStatistics().ineligibleGroups == 1,
      "omit a group containing a periodically unreachable recurrence edge");

  ProtocolInputs distanceTwo = buildTwoPhaseRoundTrip(2, false);
  if (!check(distanceTwo.valid, "build distance-two protocol inputs")) {
    return false;
  }
  SyncCoverStorageProtocolAutomatonLimits oneLane;
  oneLane.maximumLanes = 1;
  const SyncCoverStorageProtocolAutomatonIndex laneLimited =
      buildSyncCoverStorageProtocolAutomatonIndex(
          distanceTwo.graph, distanceTwo.lifecycle, distanceTwo.seeds,
          distanceTwo.groups, oneLane);
  passed &=
      check(laneLimited.isComplete() && laneLimited.getAutomata().empty() &&
                laneLimited.getStatistics().ineligibleGroups == 1 &&
                laneLimited.getStatistics().laneLimitedGroups == 1,
            "omit a reachable protocol exceeding the configured lane bound");

  SyncCoverStorageProtocolGroupLimits groupLimit;
  groupLimit.maximumWorkUnits = 1;
  const SyncCoverStorageProtocolGroupIndex incompleteGroups =
      buildSyncCoverStorageProtocolGroupIndex(distanceTwo.graph,
                                              distanceTwo.lifecycle,
                                              distanceTwo.seeds, groupLimit);
  passed &=
      check(buildSyncCoverStorageProtocolAutomatonIndex(
                distanceTwo.graph, distanceTwo.lifecycle, distanceTwo.seeds,
                incompleteGroups)
                    .getError() ==
                SyncCoverStorageProtocolAutomatonError::IncompleteGroupIndex,
            "reject an incomplete protocol-group input");
  return passed;
}

bool testOuterRecurrenceDoesNotAdvanceInnerPhase() {
  ProtocolInputs inputs = buildOuterRecurrenceUnderInnerPhase();
  if (!check(inputs.valid, "build nested-scope protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonIndex index =
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups);
  const bool hasOneAutomaton =
      index.isComplete() && index.getAutomata().size() == 1;
  if (!check(hasOneAutomaton,
             "retain outer recurrence under an inner periodic phase")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomaton &automaton =
      index.getAutomata().front();
  bool passed = check(automaton.transfers.size() == 2,
                      "project both nested-scope lifecycle transfers");
  passed &=
      check(automaton.transfers[0].distance == 1 &&
                automaton.transfers[0].activeStatePairs ==
                    std::vector<SyncCoverStorageProtocolStatePair>{{0, 1}} &&
                automaton.transfers[1].distance == 1 &&
                automaton.transfers[1].activeStatePairs ==
                    std::vector<SyncCoverStorageProtocolStatePair>{{1, 0}},
            "match independent child-loop phases across an outer recurrence");

  ProtocolInputs deep = buildOuterRecurrenceUnderInnerPhase(32);
  if (!check(deep.valid, "build deep nested-scope protocol inputs")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonIndex deepIndex =
      buildSyncCoverStorageProtocolAutomatonIndex(deep.graph, deep.lifecycle,
                                                  deep.seeds, deep.groups);
  const bool hasOneDeepAutomaton =
      deepIndex.isComplete() && deepIndex.getAutomata().size() == 1;
  if (!check(hasOneDeepAutomaton,
             "build deep nested-scope protocol automaton")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonStatistics &deepStatistics =
      deepIndex.getStatistics();
  SyncCoverStorageProtocolAutomatonLimits exact;
  exact.maximumWorkUnits = deepStatistics.workUnits;
  exact.maximumAutomata = deepStatistics.automata;
  exact.maximumStates = deepStatistics.states;
  exact.maximumTransfers = deepStatistics.transfers;
  exact.maximumStatePairIncidences = deepStatistics.statePairIncidences;
  exact.maximumLanes = 1;
  passed &=
      check(buildSyncCoverStorageProtocolAutomatonIndex(
                deep.graph, deep.lifecycle, deep.seeds, deep.groups, exact)
                .isComplete(),
            "accept deep recurrence projection at the exact work bound");
  --exact.maximumWorkUnits;
  passed &= checkTransactionalLimit(
      buildSyncCoverStorageProtocolAutomatonIndex(
          deep.graph, deep.lifecycle, deep.seeds, deep.groups, exact),
      "meter both deep ancestry queries before recurrence projection");
  return passed;
}

bool testSiblingRecurrenceScopeRejectsOnlyTheProposal() {
  ProtocolInputs inputs = buildSiblingScopeProtocolGroup();
  const bool hasOneGroup =
      inputs.valid && inputs.groups.getGroups().size() == 1;
  if (!check(hasOneGroup, "build one sibling-scope protocol group")) {
    return false;
  }
  const SyncCoverStorageProtocolAutomatonIndex index =
      buildSyncCoverStorageProtocolAutomatonIndex(
          inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups);
  const SyncCoverStorageProtocolAutomatonStatistics &statistics =
      index.getStatistics();
  return check(index.isComplete() && index.getAutomata().empty() &&
                   statistics.eligibleGroups == 0 &&
                   statistics.ineligibleGroups == 1 &&
                   statistics.scopeRejectedGroups == 1,
               "reject only the incomparable sibling-scope proposal");
}

bool testMultiStateProjectionModes() {
  const auto checkRelation =
      [](ProjectionScopeRelation relation,
         std::vector<SyncCoverStorageProtocolStatePair> expectedStatePairs,
         std::string_view message) {
        ProtocolInputs inputs = buildProjectionScopeProtocolGroup(relation);
        const bool hasOneGroup =
            inputs.valid && inputs.groups.getGroups().size() == 1;
        if (!check(hasOneGroup, "build one multi-state projection group")) {
          return false;
        }
        const SyncCoverStorageProtocolAutomatonIndex index =
            buildSyncCoverStorageProtocolAutomatonIndex(
                inputs.graph, inputs.lifecycle, inputs.seeds, inputs.groups);
        const bool hasOneAutomaton =
            index.isComplete() && index.getAutomata().size() == 1;
        if (!check(hasOneAutomaton,
                   "build one multi-state projection automaton")) {
          return false;
        }
        for (const SyncCoverStorageProtocolTransfer &transfer :
             index.getAutomata().front().transfers) {
          const SyncCoverDemand &demand =
              inputs.graph.getDemands()[transfer.demand];
          const bool isUnguardedRelease = transfer.distance == 1 &&
                                          demand.sourceGuard.literals.empty() &&
                                          demand.targetGuard.literals.empty();
          if (isUnguardedRelease) {
            return check(transfer.activeStatePairs == expectedStatePairs,
                         message);
          }
        }
        return check(false, "find the unguarded release projection");
      };

  return checkRelation(ProjectionScopeRelation::SameLoop, {{0, 1}, {1, 0}},
                       "advance a same-loop recurrence by its distance") &&
         checkRelation(
             ProjectionScopeRelation::RecurrenceNestedUnderPhase,
             {{0, 0}, {1, 1}},
             "retain the enclosing phase across a nested recurrence") &&
         checkRelation(
             ProjectionScopeRelation::RecurrenceEnclosesPhase,
             {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
             "match all independent child phases across an outer recurrence");
}

} // namespace

int main() {
  return testDistanceShiftedProjectionAndBounds() &&
                 testUnreachableAndLaneLimitedGroups() &&
                 testOuterRecurrenceDoesNotAdvanceInnerPhase() &&
                 testSiblingRecurrenceScopeRejectsOnlyTheProposal() &&
                 testMultiStateProjectionModes()
             ? 0
             : 1;
}
