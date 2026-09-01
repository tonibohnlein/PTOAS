// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageLifecycle.h"

#include <algorithm>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageLifecycleTest failure: " << message << '\n';
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

SyncCoverDemand makeMemoryDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                                 SyncCoverDemandKind kind,
                                 SyncCoverStorageWitnessId witness,
                                 SyncCoverScopeId scope = 0,
                                 unsigned distance = 0) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {kind};
  demand.storageWitnesses = {witness};
  return demand;
}

SyncCoverDemand makeSsaDemand(SyncCoverNodeId source,
                              SyncCoverNodeId target) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.provenanceKinds = {SyncCoverDemandKind::SSA};
  return demand;
}

bool testExactLifecycleIndex() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
                passed, "add lifecycle loop");
  SyncCoverScopeId occurrenceScope = loop;
  for (unsigned depth = 0; depth < 8; ++depth) {
    occurrenceScope =
        takeIndex(graph.addScope(occurrenceScope, true), passed,
                  "add deep lifecycle occurrence scope");
  }
  SyncCoverGuard deepGuard;
  for (unsigned index = 0; index < 12; ++index) {
    const SyncCoverControlId control =
        takeIndex(graph.addControl(2, loop), passed, "add lifecycle guard");
    deepGuard.literals.push_back({control, 0});
  }
  const SyncCoverNodeId write0 =
      takeIndex(graph.addNode(1, 1, occurrenceScope, 0, deepGuard), passed,
                "add first writer");
  const SyncCoverNodeId read0 =
      takeIndex(graph.addNode(2, 1, occurrenceScope, 1, deepGuard), passed,
                "add first reader");
  const SyncCoverNodeId write1 =
      takeIndex(graph.addNode(1, 1, occurrenceScope, 2, deepGuard), passed,
                "add reuse writer");
  const SyncCoverNodeId write2 =
      takeIndex(graph.addNode(1, 1, occurrenceScope, 3, deepGuard), passed,
                "add second-slot writer");
  const SyncCoverNodeId read2 =
      takeIndex(graph.addNode(2, 1, occurrenceScope, 4, deepGuard), passed,
                "add second-slot reader");
  const SyncCoverNodeId partialWrite =
      takeIndex(graph.addNode(1, 1, occurrenceScope, 5, deepGuard), passed,
                "add partial writer");
  const SyncCoverNodeId partialRead =
      takeIndex(graph.addNode(2, 1, occurrenceScope, 6, deepGuard), passed,
                "add partial reader");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add storage domain");
  constexpr SyncCoverStorageAccessFamilyId family = 17;
  const SyncCoverStorageAccessId write0Access =
      takeIndex(graph.addStorageAccess(write0, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add first exact write");
  const SyncCoverStorageAccessId read0Access =
      takeIndex(graph.addStorageAccess(read0, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add first exact read");
  const SyncCoverStorageAccessId write1Access =
      takeIndex(graph.addStorageAccess(write1, domain, family, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add exact reuse write");
  const SyncCoverStorageAccessId write2Access =
      takeIndex(graph.addStorageAccess(write2, domain, family, {64, 128},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add second-slot exact write");
  const SyncCoverStorageAccessId read2Access =
      takeIndex(graph.addStorageAccess(read2, domain, family, {64, 128},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add second-slot exact read");
  const SyncCoverStorageAccessId partialWriteAccess =
      takeIndex(graph.addStorageAccess(partialWrite, domain, family, {128, 192},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add partial exact write");
  const SyncCoverStorageAccessId partialReadAccess =
      takeIndex(graph.addStorageAccess(partialRead, domain, family, {160, 224},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add partial exact read");

  const SyncCoverStorageWitnessId ready =
      takeIndex(graph.addStorageWitness(write0Access, read0Access), passed,
                "add ready witness");
  const SyncCoverStorageWitnessId release =
      takeIndex(graph.addStorageWitness(read0Access, write0Access), passed,
                "add release witness");
  const SyncCoverStorageWitnessId exclusion =
      takeIndex(graph.addStorageWitness(write0Access, write1Access), passed,
                "add exclusion witness");
  const SyncCoverStorageWitnessId secondReady =
      takeIndex(graph.addStorageWitness(write2Access, read2Access), passed,
                "add second-slot ready witness");
  const SyncCoverStorageWitnessId partial =
      takeIndex(graph.addStorageWitness(partialWriteAccess, partialReadAccess),
                passed, "add partial overlap witness");

  passed &= check(graph.addDemand(makeMemoryDemand(
                      write0, read0, SyncCoverDemandKind::MemoryRAW, ready)),
                  "add ready demand");
  passed &= check(
      graph.addDemand(makeMemoryDemand(
          read0, write0, SyncCoverDemandKind::MemoryWAR, release, loop, 1)),
      "add release demand");
  passed &= check(
      graph.addDemand(makeMemoryDemand(
          write0, write1, SyncCoverDemandKind::MemoryWAW, exclusion, loop, 1)),
      "add exclusion demand");
  passed &=
      check(graph.addDemand(makeMemoryDemand(
                write2, read2, SyncCoverDemandKind::MemoryRAW, secondReady)),
            "add second-slot ready demand");
  passed &= check(
      graph.addDemand(makeMemoryDemand(
          partialWrite, partialRead, SyncCoverDemandKind::MemoryRAW, partial)),
      "add partial overlap demand");
  passed &= check(graph.freezeStructure(), "freeze lifecycle graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex index =
      buildSyncCoverStorageLifecycleIndex(graph);
  if (!check(index.isComplete(), "build complete lifecycle index")) {
    return false;
  }
  const SyncCoverStorageLifecycleStatistics &statistics = index.getStatistics();
  if (!check(statistics.workUnits > 5 && statistics.eligibleWitnesses == 4 &&
                 statistics.ineligibleWitnesses == 1 &&
                 statistics.components == 1 && statistics.slots == 2 &&
                 statistics.epochs == 5 && statistics.edges == 4 &&
                 statistics.demandIncidences == 4 && statistics.sccs == 4 &&
                 statistics.cyclicSccs == 1 &&
                 statistics.readyReleaseSccs == 1 &&
                 statistics.sccTransfers == 2 &&
                 statistics.maximumSccEpochs == 2 &&
                 statistics.transitionClasses == 3 &&
                 statistics.transitionGuardLiterals == 72 &&
                 statistics.maximumTransitionClassEdges == 2,
             "report exact retained and rejected lifecycle structure")) {
    return false;
  }
  const SyncCoverStorageLifecycleComponent &component =
      index.getComponents().front();
  const auto readyBit = syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Ready);
  const auto releaseBit = syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Release);
  const auto exclusionBit = syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Exclusion);
  passed &=
      check(component.family == family && component.owningScope == loop &&
                component.slots.size() == 2 && component.epochs.size() == 5 &&
                component.edges.size() == 4 &&
                component.transitionClasses.size() == 3,
            "group exact slots by storage family and owning loop");
  passed &= check(component.edges[0].kinds == readyBit &&
                      component.edges[0].distance == 0 &&
                      component.edges[1].kinds == releaseBit &&
                      component.edges[1].distance == 1 &&
                      component.edges[2].kinds == exclusionBit &&
                      component.edges[2].distance == 1 &&
                      component.edges[3].kinds == readyBit,
                  "retain ready, release, exclusion, and recurrence meaning");
  passed &= check(
      component.transitionClasses[0].kinds == readyBit &&
          component.transitionClasses[0].sourceResource == 1 &&
          component.transitionClasses[0].targetResource == 2 &&
          component.transitionClasses[0].distance == 0 &&
          component.transitionClasses[0].edges ==
              std::vector<SyncCoverStorageLifecycleEdgeId>({0, 3}) &&
          component.transitionClasses[1].kinds == releaseBit &&
          component.transitionClasses[1].sourceResource == 2 &&
          component.transitionClasses[1].targetResource == 1 &&
          component.transitionClasses[1].scope == loop &&
          component.transitionClasses[1].distance == 1 &&
          component.transitionClasses[2].kinds == exclusionBit,
      "classify compatible lifecycle edges without storage identity");
  passed &= check(component.slots[0].accesses ==
                          std::vector<SyncCoverStorageAccessId>(
                              {write0Access, read0Access, write1Access}) &&
                      component.slots[1].accesses ==
                          std::vector<SyncCoverStorageAccessId>(
                              {write2Access, read2Access}),
                  "deduplicate deterministic access epochs per exact slot");
  const auto readyReleaseKinds = readyBit | releaseBit;
  const bool hasReadyReleaseCycle = std::any_of(
      component.sccs.begin(), component.sccs.end(),
      [&](const SyncCoverStorageLifecycleScc &scc) {
        return scc.cyclic && scc.epochs.size() == 2 &&
               (scc.kinds & readyReleaseKinds) == readyReleaseKinds;
      });
  passed &= check(hasReadyReleaseCycle,
                  "detect the exact ready/release lifecycle SCC");
  passed &= check(component.epochSccs.size() == component.epochs.size() &&
                      component.sccTransfers.size() == 2,
                  "publish the deterministic SCC condensation mapping");

  SyncCoverStorageLifecycleLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumComponents = statistics.components;
  exact.maximumSlots = statistics.slots;
  exact.maximumEpochs = statistics.epochs;
  exact.maximumEdges = statistics.edges;
  exact.maximumDemandIncidences = statistics.demandIncidences;
  exact.maximumSccs = statistics.sccs;
  exact.maximumTransitionClasses = statistics.transitionClasses;
  exact.maximumTransitionGuardLiterals =
      statistics.transitionGuardLiterals;
  passed &=
      check(buildSyncCoverStorageLifecycleIndex(graph, exact).isComplete(),
            "accept every lifecycle bound exactly");
  --exact.maximumEdges;
  const SyncCoverStorageLifecycleIndex truncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      truncated.getError() == SyncCoverStorageLifecycleError::LimitExceeded &&
          truncated.getStatistics().truncated &&
          truncated.getComponents().empty(),
      "discard the complete staged index one edge below its exact bound");
  exact.maximumEdges = statistics.edges;
  --exact.maximumSccs;
  const SyncCoverStorageLifecycleIndex sccTruncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      sccTruncated.getError() ==
              SyncCoverStorageLifecycleError::LimitExceeded &&
          sccTruncated.getStatistics().truncated &&
          sccTruncated.getComponents().empty(),
      "discard every SCC when its retained bound is exhausted");
  exact.maximumSccs = statistics.sccs;
  --exact.maximumTransitionClasses;
  const SyncCoverStorageLifecycleIndex transitionTruncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      transitionTruncated.getError() ==
              SyncCoverStorageLifecycleError::LimitExceeded &&
          transitionTruncated.getStatistics().truncated &&
          transitionTruncated.getComponents().empty(),
      "discard every transition class below its exact retained bound");
  exact.maximumTransitionClasses = statistics.transitionClasses;
  --exact.maximumTransitionGuardLiterals;
  const SyncCoverStorageLifecycleIndex guardTruncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      guardTruncated.getError() ==
              SyncCoverStorageLifecycleError::LimitExceeded &&
          guardTruncated.getStatistics().truncated &&
          guardTruncated.getComponents().empty(),
      "bound retained transition guard literals transactionally");
  exact.maximumTransitionGuardLiterals = 0;
  passed &= check(
      buildSyncCoverStorageLifecycleIndex(graph, exact).getError() ==
          SyncCoverStorageLifecycleError::InvalidLimit,
      "reject a zero transition guard-literal bound");
  exact.maximumTransitionGuardLiterals =
      statistics.transitionGuardLiterals;
  --exact.maximumWorkUnits;
  const SyncCoverStorageLifecycleIndex workTruncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      workTruncated.getError() ==
              SyncCoverStorageLifecycleError::LimitExceeded &&
          workTruncated.getStatistics().truncated &&
          workTruncated.getComponents().empty(),
      "bound deep-scope and ordered-container work transactionally");
  exact.maximumWorkUnits = 0;
  passed &=
      check(buildSyncCoverStorageLifecycleIndex(graph, exact).getError() ==
                SyncCoverStorageLifecycleError::InvalidLimit,
            "reject zero lifecycle bounds");
  return passed;
}

bool testWitnesslessDemandWorkIsBounded() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add first SSA node");
  const SyncCoverNodeId second =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add second SSA node");
  const SyncCoverNodeId third =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add third SSA node");
  passed &= check(graph.addDemand(makeSsaDemand(first, second)),
                  "add first witnessless demand");
  passed &= check(graph.addDemand(makeSsaDemand(first, third)),
                  "add second witnessless demand");
  passed &= check(graph.freezeStructure(), "freeze witnessless demand graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex index =
      buildSyncCoverStorageLifecycleIndex(graph);
  const bool completeWithoutComponents =
      index.isComplete() && index.getComponents().empty();
  if (!check(completeWithoutComponents,
             "accept witnessless demands without lifecycle components")) {
    return false;
  }
  SyncCoverStorageLifecycleLimits exact;
  exact.maximumWorkUnits = index.getStatistics().workUnits;
  passed &= check(buildSyncCoverStorageLifecycleIndex(graph, exact).isComplete(),
                  "accept exact witnessless-demand work bound");
  --exact.maximumWorkUnits;
  const SyncCoverStorageLifecycleIndex truncated =
      buildSyncCoverStorageLifecycleIndex(graph, exact);
  passed &= check(
      truncated.getError() == SyncCoverStorageLifecycleError::LimitExceeded &&
          truncated.getStatistics().truncated &&
          truncated.getComponents().empty(),
      "charge every witnessless demand visit");
  return passed;
}

bool testAccumulatorReadReadIsAnExclusion() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId first =
      takeIndex(graph.addNode(1, 1, 0, 0), passed, "add first ACC reader");
  const SyncCoverNodeId second =
      takeIndex(graph.addNode(2, 1, 0, 1), passed, "add second ACC reader");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add ACC storage domain");
  constexpr SyncCoverStorageAccessFamilyId family = 29;
  const SyncCoverStorageAccessId firstAccess = takeIndex(
      graph.addStorageAccess(first, domain, family, {0, 64},
                             SyncCoverStorageAccessMode::Read, std::nullopt,
                             true),
      passed, "add first ACC read");
  const SyncCoverStorageAccessId secondAccess = takeIndex(
      graph.addStorageAccess(second, domain, family, {0, 64},
                             SyncCoverStorageAccessMode::Read, std::nullopt,
                             true),
      passed, "add second ACC read");
  const SyncCoverStorageWitnessId witness =
      takeIndex(graph.addStorageWitness(firstAccess, secondAccess), passed,
                "add ACC read/read witness");
  SyncCoverDemand demand = makeMemoryDemand(
      first, second, SyncCoverDemandKind::HardwareAccRAR, witness);
  demand.orderingRequirements =
      syncCoverOrderingRequirementBit(
          SyncCoverOrderingRequirement::PipelineCompletionBeforeAccess) |
      syncCoverOrderingRequirementBit(
          SyncCoverOrderingRequirement::HardwareSpecialOrder);
  passed &= check(graph.addDemand(std::move(demand)),
                  "add ACC read/read hardware demand");
  passed &= check(graph.freezeStructure(), "freeze ACC read/read graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex index =
      buildSyncCoverStorageLifecycleIndex(graph);
  const auto exclusionBit = syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Exclusion);
  return check(index.isComplete() && index.getComponents().size() == 1 &&
                   index.getComponents().front().edges.size() == 1 &&
                   index.getComponents().front().edges.front().kinds ==
                       exclusionBit,
               "classify the hardware read/read hazard as exclusion");
}

bool testRequiresFrozenGraph() {
  SyncCoverGraph graph;
  return check(buildSyncCoverStorageLifecycleIndex(graph).getError() ==
                   SyncCoverStorageLifecycleError::InvalidGraph,
               "require an immutable semantic graph");
}

} // namespace

int main() {
  bool passed = true;
  passed &= testExactLifecycleIndex();
  passed &= testWitnesslessDemandWorkIsBounded();
  passed &= testAccumulatorReadReadIsAnExclusion();
  passed &= testRequiresFrozenGraph();
  return passed ? 0 : 1;
}
