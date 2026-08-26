// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using namespace mlir::pto;

static_assert(!std::is_copy_constructible<SyncCoverMechanismUniverse>::value,
              "a mechanism universe must retain unique epoch ownership");
static_assert(!std::is_move_constructible<SyncCoverMechanismUniverse>::value,
              "moving a mechanism universe must not bypass epoch checks");

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverMechanismTest failure: " << message << '\n';
  }
  return condition;
}

bool check(const SyncCoverMechanismResult &result, std::string_view message) {
  return check(static_cast<bool>(result), message);
}

std::size_t takeGraphIndex(const SyncCoverGraphResult &result, bool &passed,
                           std::string_view message) {
  passed &=
      check(static_cast<bool>(result) && result.index.has_value(), message);
  return result.index.value_or(0);
}

std::size_t takeMechanismIndex(const SyncCoverMechanismResult &result,
                               bool &passed, std::string_view message) {
  passed &=
      check(static_cast<bool>(result) && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverEdge completionEdge(SyncCoverNodeId source, SyncCoverNodeId target,
                             SyncCoverScopeId scope = 0,
                             unsigned distance = 0) {
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.scope = scope;
  edge.distance = distance;
  return edge;
}

SyncCoverDemand makeDemand(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  return demand;
}

SyncCoverResourceAction action(SyncCoverResourceActionKind kind,
                               std::uint32_t resource,
                               SyncCoverAnchorKind anchorKind,
                               SyncCoverNodeId node) {
  SyncCoverResourceAction result;
  result.kind = kind;
  result.resource = resource;
  result.anchor = {anchorKind, node, 0};
  return result;
}

std::size_t appendCanonicalUse(SyncCoverMechanismDescriptor &descriptor,
                               SyncCoverResourceDomainId domain,
                               std::uint32_t sourceResource,
                               std::uint32_t targetResource,
                               SyncCoverNodeId source, SyncCoverNodeId target,
                               SyncCoverScopeId scope = 0,
                               unsigned distance = 0, std::size_t width = 1) {
  const std::size_t edge = descriptor.supplyEdges.size();
  const std::size_t produce = descriptor.actions.size();
  descriptor.actions.push_back(action(SyncCoverResourceActionKind::Produce,
                                      sourceResource,
                                      SyncCoverAnchorKind::AfterNode, source));
  const std::size_t consume = descriptor.actions.size();
  descriptor.actions.push_back(action(SyncCoverResourceActionKind::Consume,
                                      targetResource,
                                      SyncCoverAnchorKind::BeforeNode, target));
  const std::size_t use = descriptor.resourceUses.size();
  SyncCoverResourceUse resourceUse;
  resourceUse.domain = domain;
  resourceUse.scope = scope;
  resourceUse.distance = distance;
  resourceUse.width = width;
  resourceUse.actions = {produce, consume};
  resourceUse.supplyEdges = {edge};
  descriptor.resourceUses.push_back(std::move(resourceUse));
  descriptor.supplyEdges.push_back(
      completionEdge(source, target, scope, distance));
  descriptor.supplyBindings.push_back({edge, use, produce, consume});
  return use;
}

bool testDescriptorBuilderAndGraphEpoch() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add builder source");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, 0, 1), passed, "add builder target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId eventDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add builder event domain");
  const SyncCoverResourceDomainId tokenDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::BufferToken, 1, 2, 2,
                                 901),
      passed, "add builder token domain");
  const auto &domains = universe.getResourceDomains();

  std::optional<SyncCoverMechanismDescriptor> event =
      makeSyncCoverCanonicalEvent(domains[eventDomain], source, target, 0, 1,
                                  501);
  passed &= check(event && event->actions.size() == 2 &&
                      event->resourceUses.size() == 1 &&
                      event->supplyEdges.size() == 1 &&
                      event->supplyBindings.size() == 1,
                  "canonical-event builder owns all local index bookkeeping");
  const SyncCoverMechanismId eventId = takeMechanismIndex(
      universe.addMechanism(*event), passed, "add builder event");

  SyncCoverMechanismDescriptorBuilder protocol(
      SyncCoverMechanismKind::VerifiedProtocol, 502);
  const SyncCoverDescriptorActionRef produce = protocol.addAction(
      SyncCoverResourceActionKind::Produce, 1,
      {SyncCoverAnchorKind::AfterNode, source, 0});
  const SyncCoverDescriptorActionRef consume = protocol.addAction(
      SyncCoverResourceActionKind::Consume, 2,
      {SyncCoverAnchorKind::BeforeNode, target, 0});
  const bool addedEventLane = protocol.addProtocolLane(
      domains[eventDomain], 0, 0, 1, {produce, consume},
      {{completionEdge(source, target), produce, consume}});
  const bool addedTokenLane = protocol.addProtocolLane(
      domains[tokenDomain], 0, 0, 1, {produce, consume},
      {{completionEdge(source, target), produce, consume}});
  passed &= check(addedEventLane && addedTokenLane,
                  "protocol lanes can share physical actions across pools");
  SyncCoverMechanismDescriptor ownership =
      std::move(protocol).takeDescriptor();
  passed &= check(ownership.actions.size() == 2 &&
                      ownership.resourceUses.size() == 2 &&
                      ownership.supplyBindings.size() == 2,
                  "protocol builder does not duplicate shared actions");
  const SyncCoverMechanismId protocolId = takeMechanismIndex(
      universe.addVerifiedProtocol(ownership, [](const auto &) { return true; }),
      passed, "add builder protocol");
  for (std::size_t index = 0; index < 32; ++index) {
    std::optional<SyncCoverMechanismDescriptor> candidate =
        makeSyncCoverCanonicalEvent(domains[eventDomain], source, target, 0, 1,
                                    600 + index);
    passed &= check(candidate && universe.addMechanism(*candidate),
                    "bulk candidate construction remains valid");
  }
  passed &= check(universe.getStatistics().fullValidations == 1,
                  "candidate construction avoids per-addition validation");

  SyncCoverMechanismDescriptorBuilder rejected(
      SyncCoverMechanismKind::VerifiedProtocol, 503);
  const SyncCoverDescriptorActionRef rejectedProduce = rejected.addAction(
      SyncCoverResourceActionKind::Produce, 1,
      {SyncCoverAnchorKind::AfterNode, source, 0});
  const SyncCoverDescriptorActionRef rejectedConsume = rejected.addAction(
      SyncCoverResourceActionKind::Consume, 2,
      {SyncCoverAnchorKind::BeforeNode, target, 0});
  const SyncCoverMechanismDescriptor before = rejected.getDescriptor();
  passed &= check(!rejected.addProtocolLane(
                      domains[eventDomain], 0, 0, 1,
                      {rejectedProduce, rejectedConsume},
                      {{completionEdge(source, target), rejectedProduce,
                        SyncCoverDescriptorActionRef{99}}}),
                  "builder rejects unknown typed action references");
  passed &= check(rejected.getDescriptor().actions.size() ==
                          before.actions.size() &&
                      rejected.getDescriptor().resourceUses.empty() &&
                      rejected.getDescriptor().supplyEdges.empty(),
                  "failed builder additions leave the descriptor unchanged");

  const SyncCoverSelectionEvaluator epoch(universe);
  const SyncCoverCandidateIndex candidateIndex(graph);
  passed &= check(static_cast<bool>(universe.getInitializationResult()),
                  "universe exposes a successful structural freeze");
  passed &= check(epoch && epoch.evaluate({eventId, protocolId}),
                  "selection epoch accepts the completed builder universe");
  passed &= check(universe.getStatistics().fullValidations == 2,
                  "selection freezes one phase-boundary validation");
  passed &= check(graph.addNode(3, 1, 0, 2).error ==
                      SyncCoverGraphError::StructureFrozen,
                  "structural mutation is rejected after universe creation");
  passed &= check(epoch && epoch.evaluate({eventId, protocolId}),
                  "rejected structure changes preserve the selection epoch");
  passed &= check(static_cast<bool>(candidateIndex),
                  "mechanism supply edges preserve the candidate index epoch");
  passed &= check(universe.validate(),
                  "phase-boundary validation accepts the frozen graph");
  passed &= check(universe.validate() &&
                      universe.getStatistics().fullValidations == 2,
                  "unchanged graph and universe reuse cached validation");
  return passed;
}

bool testExpansionOverlayEpoch() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed,
      "add expansion source");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, 0, 1), passed, "add expansion target");
  passed &= check(static_cast<bool>(
                      graph.addDemand(makeDemand(source, target))),
                  "add expansion demand");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add expansion event domain");
  SyncCoverExpandedProgram expansion(graph);
  passed &= check(expansion && expansion.isCurrent(graph),
                  "initial expansion matches its empty overlay epoch");

  const std::optional<SyncCoverMechanismDescriptor> event =
      makeSyncCoverCanonicalEvent(universe.getResourceDomains()[domain],
                                  source, target, 0, 1, 801);
  passed &= check(event.has_value(), "build expansion event");
  if (!event) {
    return false;
  }
  passed &= check(universe.addMechanism(*event), "add expansion event");
  passed &= check(expansion.isStructuralCurrent(graph) &&
                      !expansion.isCurrent(graph),
                  "mechanism insertion invalidates only the overlay epoch");
  passed &= check(expansion.refreshMechanismOverlay(graph) ==
                          SyncCoverExpansionError::None &&
                      expansion.isCurrent(graph),
                  "mechanism overlay refresh restores the current epoch");
  const auto &overlay =
      expansion.getBaseArena().getMechanismEdges().getEdges();
  passed &= check(overlay.size() == 1 &&
                      overlay.front().graphEdge ==
                          graph.getStructuralEdgeCount(),
                  "refreshed overlay contains only mechanism-supplied edges");

  const SyncCoverMechanismId mechanism =
      universe.getMechanisms().front().id;
  SyncCoverCoverageOracle shared(
      graph, SyncCoverCoverageBackend::SharedExpansion);
  SyncCoverCoverageOracle legacy(
      graph, SyncCoverCoverageBackend::LegacyPerContext);
  const SyncCoverCoverageResult sharedCovered =
      shared.checkDemand(0, {mechanism});
  const SyncCoverCoverageResult legacyCovered =
      legacy.checkDemand(0, {mechanism});
  const SyncCoverCoverageResult sharedUncovered = shared.checkDemand(0, {});
  const SyncCoverCoverageResult legacyUncovered = legacy.checkDemand(0, {});
  passed &= check(sharedCovered.error == legacyCovered.error &&
                      sharedCovered.covered == legacyCovered.covered &&
                      sharedCovered.witnessMechanisms ==
                          legacyCovered.witnessMechanisms &&
                      sharedUncovered.error == legacyUncovered.error &&
                      sharedUncovered.covered == legacyUncovered.covered &&
                      sharedUncovered.cutMechanisms ==
                          legacyUncovered.cutMechanisms,
                  "shared and legacy coverage agree on the same frozen graph");
  const SyncCoverSingletonWitnessResult sharedSingleton =
      shared.getSingletonMechanismWitnesses(0);
  const SyncCoverSingletonWitnessResult legacySingleton =
      legacy.getSingletonMechanismWitnesses(0);
  const SyncCoverFactoryWitnessResult sharedFactory =
      shared.getFactoryMechanismWitnesses(0, 1);
  const SyncCoverFactoryWitnessResult legacyFactory =
      legacy.getFactoryMechanismWitnesses(0, 1);
  passed &= check(sharedSingleton.error == legacySingleton.error &&
                      sharedSingleton.mechanisms ==
                          legacySingleton.mechanisms &&
                      sharedFactory.error == legacyFactory.error &&
                      sharedFactory.singletons == legacyFactory.singletons &&
                      sharedFactory.pairs == legacyFactory.pairs,
                  "shared and legacy grounding agree on the same frozen graph");
  return passed;
}

bool testSharedRecurrenceCoverageParity() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeGraphIndex(
      graph.addScope(0, false, SyncCoverTimelineInterval{0, 12}, true), passed,
      "add parity recurrence loop");
  const SyncCoverScopeId body = takeGraphIndex(
      graph.addScope(loop, true, SyncCoverTimelineInterval{1, 11}), passed,
      "add parity recurrence body");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, body, 1), passed, "add parity target");
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, body, 2, {}, {2}), passed,
      "add parity source");
  const SyncCoverNodeId secondTarget = takeGraphIndex(
      graph.addNode(2, 1, body, 3), passed, "add second parity target");
  const SyncCoverNodeId secondSource = takeGraphIndex(
      graph.addNode(1, 1, body, 4, {}, {2}), passed,
      "add second parity source");
  SyncCoverDemand recurrence = makeDemand(source, target);
  recurrence.scope = loop;
  recurrence.distance = 1;
  passed &= check(static_cast<bool>(graph.addDemand(recurrence)),
                  "add parity recurrence demand");
  SyncCoverDemand secondRecurrence =
      makeDemand(secondSource, secondTarget);
  secondRecurrence.scope = loop;
  secondRecurrence.distance = 1;
  passed &= check(static_cast<bool>(graph.addDemand(secondRecurrence)),
                  "add second parity recurrence demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add parity recurrence domain");
  const auto descriptor = makeSyncCoverUnitRecurrenceEvent(
      universe.getResourceDomains()[domain], source, target, loop, 811);
  passed &= check(descriptor.has_value(), "build parity recurrence protocol");
  if (!descriptor) {
    return false;
  }
  const SyncCoverMechanismId mechanism = takeMechanismIndex(
      universe.addVerifiedProtocol(
          *descriptor, [&](const auto &candidate) {
            return verifySyncCoverUnitRecurrenceEvent(universe, candidate);
          }),
      passed, "add parity recurrence protocol");
  const auto secondDescriptor = makeSyncCoverUnitRecurrenceEvent(
      universe.getResourceDomains()[domain], secondSource, secondTarget, loop,
      812);
  passed &= check(secondDescriptor.has_value(),
                  "build second parity recurrence protocol");
  if (!secondDescriptor) {
    return false;
  }
  const SyncCoverMechanismId secondMechanism = takeMechanismIndex(
      universe.addVerifiedProtocol(
          *secondDescriptor, [&](const auto &candidate) {
            return verifySyncCoverUnitRecurrenceEvent(universe, candidate);
          }),
      passed, "add second parity recurrence protocol");

  SyncCoverCoverageOracle shared(
      graph, SyncCoverCoverageBackend::SharedExpansion);
  SyncCoverCoverageOracle legacy(
      graph, SyncCoverCoverageBackend::LegacyPerContext);
  const std::vector<SyncCoverMechanismId> selected{mechanism,
                                                   secondMechanism};
  for (SyncCoverDemandId demand : {1U, 0U, 1U}) {
    const SyncCoverCoverageResult sharedCovered =
        shared.checkDemand(demand, selected);
    const SyncCoverCoverageResult legacyCovered =
        legacy.checkDemand(demand, selected);
    const SyncCoverCoverageResult sharedUncovered =
        shared.checkDemand(demand, {});
    const SyncCoverCoverageResult legacyUncovered =
        legacy.checkDemand(demand, {});
    passed &= check(sharedCovered.error == legacyCovered.error &&
                        sharedCovered.covered == legacyCovered.covered &&
                        sharedCovered.witnessMechanisms ==
                            legacyCovered.witnessMechanisms &&
                        sharedUncovered.error == legacyUncovered.error &&
                        sharedUncovered.covered == legacyUncovered.covered &&
                        sharedUncovered.cutMechanisms ==
                            legacyUncovered.cutMechanisms,
                    "shared recurrence cache matches legacy for each demand");
  }
  return passed;
}

bool testSharedGuardedContextParity() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverControlId control = takeGraphIndex(
      graph.addControl(2), passed, "add guarded parity control");
  const SyncCoverGuard guard{{{control, 0}}};
  const SyncCoverNodeId firstSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, guard, {2}), passed,
      "add first guarded source");
  const SyncCoverNodeId firstTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 1, guard), passed,
      "add first guarded target");
  const SyncCoverNodeId secondSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 2, guard, {2}), passed,
      "add second guarded source");
  const SyncCoverNodeId secondTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 3, guard), passed,
      "add second guarded target");
  passed &= check(static_cast<bool>(
                      graph.addDemand(makeDemand(firstSource, firstTarget))),
                  "add first guarded demand");
  passed &= check(static_cast<bool>(
                      graph.addDemand(makeDemand(secondSource, secondTarget))),
                  "add second guarded demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add guarded parity domain");
  const auto firstEvent = makeSyncCoverCanonicalEvent(
      universe.getResourceDomains()[domain], firstSource, firstTarget, 0, 1,
      821);
  const auto secondEvent = makeSyncCoverCanonicalEvent(
      universe.getResourceDomains()[domain], secondSource, secondTarget, 0, 1,
      822);
  passed &= check(firstEvent && secondEvent, "build guarded parity events");
  if (!firstEvent || !secondEvent) {
    return false;
  }
  const SyncCoverMechanismId firstMechanism = takeMechanismIndex(
      universe.addMechanism(*firstEvent), passed, "add first guarded event");
  const SyncCoverMechanismId secondMechanism = takeMechanismIndex(
      universe.addMechanism(*secondEvent), passed, "add second guarded event");

  SyncCoverCoverageOracle shared(
      graph, SyncCoverCoverageBackend::SharedExpansion);
  SyncCoverCoverageOracle legacy(
      graph, SyncCoverCoverageBackend::LegacyPerContext);
  const std::vector<SyncCoverMechanismId> selected{firstMechanism,
                                                   secondMechanism};
  for (SyncCoverDemandId demand : {1U, 0U, 1U}) {
    const SyncCoverCoverageResult sharedResult =
        shared.checkDemand(demand, selected);
    const SyncCoverCoverageResult legacyResult =
        legacy.checkDemand(demand, selected);
    passed &= check(sharedResult.error == legacyResult.error &&
                        sharedResult.covered == legacyResult.covered &&
                        sharedResult.witnessMechanisms ==
                            legacyResult.witnessMechanisms,
                    "shared guarded cache matches legacy for each demand");
  }
  return passed;
}

bool testSharedAncestorBoundaryParity() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeGraphIndex(
      graph.addScope(0, false, SyncCoverTimelineInterval{0, 12}, true), passed,
      "add ancestor-boundary loop");
  const SyncCoverScopeId body = takeGraphIndex(
      graph.addScope(loop, true, SyncCoverTimelineInterval{1, 11}), passed,
      "add ancestor-boundary body");
  const SyncCoverNodeId recurrenceTarget = takeGraphIndex(
      graph.addNode(2, 1, body, 1), passed,
      "add recurrence protocol target");
  const SyncCoverNodeId ancestor = takeGraphIndex(
      graph.addNode(2, 1, 0, 2), passed, "add ancestor-scope boundary");
  const SyncCoverNodeId demandTarget = takeGraphIndex(
      graph.addNode(2, 1, body, 3), passed, "add boundary demand target");
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, body, 4, {}, {2}), passed,
      "add boundary recurrence source");
  SyncCoverEdge toAncestor;
  toAncestor.source = recurrenceTarget;
  toAncestor.target = ancestor;
  toAncestor.kind = SyncCoverEdgeKind::CompletionPreservingIssueOrder;
  passed &= check(static_cast<bool>(graph.addEdge(toAncestor)),
                  "add issue edge to ancestor scope");
  SyncCoverEdge fromAncestor;
  fromAncestor.source = ancestor;
  fromAncestor.target = demandTarget;
  fromAncestor.kind = SyncCoverEdgeKind::CompletionPreservingIssueOrder;
  passed &= check(static_cast<bool>(graph.addEdge(fromAncestor)),
                  "add issue edge from ancestor scope");
  SyncCoverDemand demand = makeDemand(source, demandTarget);
  demand.scope = loop;
  demand.distance = 1;
  passed &= check(static_cast<bool>(graph.addDemand(demand)),
                  "add ancestor-boundary recurrence demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add ancestor-boundary domain");
  const auto descriptor = makeSyncCoverUnitRecurrenceEvent(
      universe.getResourceDomains()[domain], source, recurrenceTarget, loop,
      831);
  passed &= check(descriptor.has_value(),
                  "build ancestor-boundary recurrence protocol");
  if (!descriptor) {
    return false;
  }
  const SyncCoverMechanismId mechanism = takeMechanismIndex(
      universe.addVerifiedProtocol(
          *descriptor, [&](const auto &candidate) {
            return verifySyncCoverUnitRecurrenceEvent(universe, candidate);
          }),
      passed, "add ancestor-boundary recurrence protocol");

  SyncCoverCoverageOracle shared(
      graph, SyncCoverCoverageBackend::SharedExpansion);
  SyncCoverCoverageOracle legacy(
      graph, SyncCoverCoverageBackend::LegacyPerContext);
  const SyncCoverCoverageResult sharedResult =
      shared.checkDemand(0, {mechanism});
  const SyncCoverCoverageResult legacyResult =
      legacy.checkDemand(0, {mechanism});
  passed &= check(sharedResult.error == legacyResult.error &&
                      sharedResult.covered && legacyResult.covered &&
                      sharedResult.witnessMechanisms ==
                          legacyResult.witnessMechanisms,
                  "shared expansion preserves ancestor-boundary recurrence "
                  "coverage");
  return passed;
}

bool testUniverseReportsFreezeFailure() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0), passed, "add contaminated source");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, 0, 1), passed, "add contaminated target");
  SyncCoverEdge edge = completionEdge(source, target);
  edge.mechanism = 7;
  passed &= check(static_cast<bool>(graph.addEdge(edge)),
                  "add premature mechanism edge");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverGraphResult initialization =
      universe.getInitializationResult();
  passed &= check(initialization.error ==
                          SyncCoverGraphError::InvalidEdgeOwnership &&
                      !universe.validate(),
                  "universe preserves the structural freeze failure cause");
  return passed;
}

bool testStockUnitRecurrenceProtocol() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeGraphIndex(
      graph.addScope(0, false, SyncCoverTimelineInterval{0, 9}, true), passed,
      "add stock recurrence loop");
  const SyncCoverScopeId body = takeGraphIndex(
      graph.addScope(loop, true, SyncCoverTimelineInterval{1, 8}), passed,
      "add mandatory recurrence body");
  const SyncCoverScopeId optionalRegion = takeGraphIndex(
      graph.addScope(loop, false, SyncCoverTimelineInterval{8, 9}), passed,
      "add optional recurrence region");
  const SyncCoverScopeId nestedLoop = takeGraphIndex(
      graph.addScope(body, true, SyncCoverTimelineInterval{6, 7}, true), passed,
      "add positive-trip nested recurrence loop");
  const SyncCoverScopeId nestedBody = takeGraphIndex(
      graph.addScope(nestedLoop, true, SyncCoverTimelineInterval{6, 7}), passed,
      "add nested recurrence body");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, body, 1), passed, "add stock recurrence target");
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, body, 2, {}, {2}), passed,
                     "add stock recurrence source");
  const SyncCoverNodeId optionalSource =
      takeGraphIndex(graph.addNode(1, 1, optionalRegion, 4, {}, {2}), passed,
                     "add optional recurrence source");
  const SyncCoverNodeId nestedSource =
      takeGraphIndex(graph.addNode(1, 1, nestedBody, 3, {}, {2}), passed,
                     "add nested-loop recurrence source");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add stock recurrence domain");
  const SyncCoverResourceDomainId tokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 1, 302),
                         passed, "add stock recurrence token domain");
  const SyncCoverResourceDomain &eventDomain =
      universe.getResourceDomains()[domain];
  const auto descriptor =
      makeSyncCoverUnitRecurrenceEvent(eventDomain, source, target, loop, 301);
  passed &= check(
      descriptor && verifySyncCoverUnitRecurrenceEvent(universe, *descriptor),
      "stock recurrence accepts a translator-shaped mandatory loop body");
  if (descriptor) {
    passed &= check(universe.addVerifiedProtocol(
                        *descriptor,
                        [&](const auto &candidate) {
                          return verifySyncCoverUnitRecurrenceEvent(universe,
                                                                    candidate);
                        }),
                    "stock unit recurrence protocol enters the universe");

    SyncCoverMechanismDescriptor longer = *descriptor;
    longer.resourceUses.front().distance = 3;
    longer.supplyEdges.front().distance = 3;
    passed &= check(
        universe.addVerifiedProtocol(
                    longer,
                    [&](const auto &candidate) {
                      return verifySyncCoverUnitRecurrenceEvent(universe,
                                                                candidate);
                    })
                .error == SyncCoverMechanismError::UnverifiedProtocol,
        "multi-distance recurrence requires a protocol-specific verifier");

    SyncCoverMechanismDescriptor wrongDomain = *descriptor;
    wrongDomain.resourceUses.front().domain = tokenDomain;
    passed &= check(
        universe.addVerifiedProtocol(
                    wrongDomain,
                    [&](const auto &candidate) {
                      return verifySyncCoverUnitRecurrenceEvent(universe,
                                                                candidate);
                    })
                .error == SyncCoverMechanismError::UnverifiedProtocol,
        "stock recurrence cannot evade event coloring through a token pool");

    SyncCoverMechanismDescriptor missingDrain = *descriptor;
    missingDrain.actions.pop_back();
    missingDrain.resourceUses.front().actions.pop_back();
    passed &=
        check(universe.addVerifiedProtocol(
                          missingDrain,
                          [&](const auto &candidate) {
                            return verifySyncCoverUnitRecurrenceEvent(
                                universe, candidate);
                          })
                      .error == SyncCoverMechanismError::UnverifiedProtocol,
              "stock recurrence rejects an incomplete token lifecycle");

    const auto optional = makeSyncCoverUnitRecurrenceEvent(
        eventDomain, optionalSource, target, loop, 302);
    passed &= check(
        optional && !verifySyncCoverUnitRecurrenceEvent(universe, *optional),
        "stock recurrence rejects an optional loop region");

    const auto nested = makeSyncCoverUnitRecurrenceEvent(
        eventDomain, nestedSource, target, loop, 303);
    passed &= check(
        nested && !verifySyncCoverUnitRecurrenceEvent(universe, *nested),
        "stock recurrence rejects nested-loop execution multiplicity");
  }
  return passed;
}

bool testAtomicSupplyAndCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add target");
  passed &=
      check(static_cast<bool>(graph.addDemand(makeDemand(source, target))),
            "add demand");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverCandidateIndex candidateIndex(graph);
  passed &= check(static_cast<bool>(candidateIndex),
                  "build candidate index before mechanism insertion");
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add event domain");

  SyncCoverMechanismDescriptor event;
  event.providerIdentity = 41;
  appendCanonicalUse(event, domain, 1, 2, source, target);
  const SyncCoverMechanismId eventId = takeMechanismIndex(
      universe.addMechanism(event), passed, "add event mechanism");
  passed &= check(eventId == 0 && graph.getEdges().size() == 1,
                  "universe assigns dense identity and attaches supply");
  passed &= check(static_cast<bool>(candidateIndex),
                  "mechanism-owned supply does not stale structural lookup");
  passed &=
      check(SyncCoverCoverageOracle(graph).checkDemand(0, {eventId}).covered,
            "selected atomic mechanism supplies graph completion");
  passed &= check(!SyncCoverCoverageOracle(graph).checkDemand(0, {}).covered,
                  "unselected mechanism supplies no partial edge");
  passed &= check(universe.validate(), "constructed universe validates");
  return passed;
}

bool testAtomicFailureAndProtocolGate() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add failure source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add failure target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId eventDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add failure event domain");
  const SyncCoverResourceDomainId tokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 2, 2, 2, 7001),
                         passed, "add failure token pool");

  SyncCoverMechanismDescriptor invalid;
  invalid.kind = SyncCoverMechanismKind::VerifiedProtocol;
  appendCanonicalUse(invalid, eventDomain, 1, 2, source, target);
  appendCanonicalUse(invalid, tokenDomain, 2, 2, target, target);
  const std::size_t edgeCount = graph.getEdges().size();
  const std::size_t graphGeneration = graph.getGeneration();
  bool invalidVerifierCalled = false;
  const SyncCoverMechanismResult invalidResult =
      universe.addVerifiedProtocol(invalid, [&](const auto &) {
        invalidVerifierCalled = true;
        return true;
      });
  passed &= check(invalidResult.error == SyncCoverMechanismError::InvalidSupply,
                  "invalid second edge rejects the whole mechanism");
  passed &= check(!invalidResult.index,
                  "failed insertion does not report a pseudo-object index");
  passed &= check(!invalidVerifierCalled,
                  "generic validation precedes semantic verification");
  passed &= check(graph.getEdges().size() == edgeCount &&
                      graph.getGeneration() == graphGeneration &&
                      universe.getMechanisms().empty(),
                  "failed mechanism addition restores content and generation");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::VerifiedProtocol;
  ownership.providerIdentity = 73;
  appendCanonicalUse(ownership, eventDomain, 1, 2, source, target);
  passed &= check(universe.addMechanism(ownership).error ==
                      SyncCoverMechanismError::UnverifiedProtocol,
                  "ownership supply requires protocol verification");
  const std::size_t preVerificationEdges = graph.getEdges().size();
  const std::size_t preVerificationGeneration = graph.getGeneration();
  bool verifierThrew = false;
  try {
    static_cast<void>(universe.addVerifiedProtocol(
        ownership, [](const auto &) -> bool {
          throw std::runtime_error("expected verifier failure");
        }));
  } catch (const std::runtime_error &) {
    verifierThrew = true;
  }
  passed &= check(verifierThrew &&
                      graph.getEdges().size() == preVerificationEdges &&
                      graph.getGeneration() == preVerificationGeneration &&
                      universe.getMechanisms().empty(),
                  "throwing verifier cannot enter the graph transaction");
  passed &= check(universe.addVerifiedProtocol(
                      ownership, [](const auto &) { return false; })
                          .error ==
                      SyncCoverMechanismError::UnverifiedProtocol,
                  "semantic protocol rejection fails closed");
  passed &= check(graph.getEdges().size() == preVerificationEdges &&
                      graph.getGeneration() == preVerificationGeneration &&
                      universe.getMechanisms().empty(),
                  "semantic rejection restores the graph checkpoint");
  const std::size_t preMutationNodes = graph.getNodes().size();
  const SyncCoverMechanismResult mutatingVerifier = universe.addVerifiedProtocol(
      ownership, [&](const auto &) {
        return static_cast<bool>(graph.addNode(3, 1, 0, 2));
      });
  passed &= check(mutatingVerifier.error ==
                          SyncCoverMechanismError::UnverifiedProtocol &&
                      graph.getEdges().size() == preVerificationEdges &&
                      graph.getNodes().size() == preMutationNodes &&
                      graph.getGeneration() == preVerificationGeneration &&
                      universe.getMechanisms().empty(),
                  "verified-protocol callbacks cannot mutate frozen structure");
  bool verifierCalled = false;
  passed &= check(universe.addVerifiedProtocol(
                      ownership,
                      [&](const auto &submitted) {
                        verifierCalled = true;
                        ownership.providerIdentity = 999;
                        return submitted.providerIdentity == 73 &&
                               submitted.supplyBindings.size() == 1;
                      }),
                  "verified ownership protocol is admitted atomically");
  passed &= check(verifierCalled,
                  "verification runs against the submitted descriptor");
  passed &= check(universe.getMechanisms().back().providerIdentity == 73,
                  "verification and commit use one descriptor snapshot");
  return passed;
}

bool testMultiEdgeAtomicSupply() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first bundle source");
  const SyncCoverNodeId secondSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second bundle source");
  const SyncCoverNodeId firstTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 2), passed, "add first bundle target");
  const SyncCoverNodeId secondTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 3), passed, "add second bundle target");
  SyncCoverEdge issueOrder;
  issueOrder.source = firstSource;
  issueOrder.target = secondSource;
  passed &= check(static_cast<bool>(graph.addEdge(issueOrder)),
                  "add pre-existing graph edge");
  passed &= check(
      static_cast<bool>(graph.addDemand(makeDemand(firstSource, firstTarget))),
      "add first bundle demand");
  passed &= check(static_cast<bool>(
                      graph.addDemand(makeDemand(secondSource, secondTarget))),
                  "add second bundle demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add multi-edge domain");
  SyncCoverMechanismDescriptor bundle;
  appendCanonicalUse(bundle, domain, 1, 2, firstSource, firstTarget);
  appendCanonicalUse(bundle, domain, 1, 2, secondSource, secondTarget);
  const SyncCoverMechanismId bundleId = takeMechanismIndex(
      universe.addMechanism(bundle), passed, "add multi-edge bundle");

  const SyncCoverMechanism &stored = universe.getMechanisms()[bundleId];
  passed &= check(
      stored.supplyEdges == std::vector<std::size_t>{1, 2} &&
          stored.resourceUses[0].supplyEdges == std::vector<std::size_t>{1} &&
          stored.resourceUses[1].supplyEdges == std::vector<std::size_t>{2} &&
          stored.supplyBindings[0].supplyEdge == 1 &&
          stored.supplyBindings[1].supplyEdge == 2,
      "bundle rewrites local supply indices to graph indices");
  const SyncCoverCoverageOracle oracle(graph);
  passed &= check(oracle.checkDemand(0, {bundleId}).covered &&
                      oracle.checkDemand(1, {bundleId}).covered,
                  "selecting a bundle enables every supplied edge");
  passed &= check(!oracle.checkDemand(0, {}).covered &&
                      !oracle.checkDemand(1, {}).covered,
                  "omitting a bundle disables every supplied edge");
  return passed;
}

bool testDomainsBarriersAndConflicts() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add conflict source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add conflict target");
  const SyncCoverNodeId barrierTarget =
      takeGraphIndex(graph.addNode(1, 1, 0, 2), passed, "add barrier target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8, 0,
                                 {12, 3, 3}),
      passed, "add conflict domain");
  passed &= check(universe.getResourceDomains()[domain].reservedIds ==
                      std::vector<unsigned>{3, 12},
                  "reserved IDs remain normalized and diagnostic");
  const SyncCoverMechanismResult duplicate = universe.addResourceDomain(
      SyncCoverResourceKind::EventId, 1, 2, 8, 0, {3, 12});
  passed &= check(duplicate && duplicate.index == domain,
                  "identical domain registration is idempotent");
  passed &= check(universe.addResourceDomain(SyncCoverResourceKind::EventId, 1,
                                             2, 8, 0, {3})
                          .error == SyncCoverMechanismError::InvalidDomain,
                  "one directed domain cannot change reservations");
  passed &=
      check(universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 1, 8)
                    .error == SyncCoverMechanismError::InvalidDomain,
            "event IDs require a cross-resource domain");
  passed &= check(universe.addResourceDomain(SyncCoverResourceKind::BufferToken,
                                             1, 1, 2, 1000, {1})
                          .error == SyncCoverMechanismError::InvalidDomain,
                  "buffer-token pools do not accept event reservations");
  passed &= check(universe.addResourceDomain(SyncCoverResourceKind::BufferToken,
                                             1, 1, 2)
                          .error == SyncCoverMechanismError::InvalidDomain,
                  "buffer-token pools require an explicit identity");
  const SyncCoverResourceDomainId firstToken =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 1, 2, 1001),
                         passed, "add first token pool");
  const SyncCoverResourceDomainId secondToken =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 1, 2, 1002),
                         passed, "add second token pool");
  passed &= check(firstToken != secondToken,
                  "independent token pools retain distinct domains");
  passed &= check(universe.addResourceDomain(SyncCoverResourceKind::BufferToken,
                                             2, 2, 2, 1001)
                          .error == SyncCoverMechanismError::InvalidDomain,
                  "one token-pool identity cannot span resource domains");

  SyncCoverMechanismDescriptor event;
  appendCanonicalUse(event, domain, 1, 2, source, target);
  const SyncCoverMechanismId eventId = takeMechanismIndex(
      universe.addMechanism(event), passed, "add conflict event");

  SyncCoverMechanismDescriptor barrier;
  barrier.kind = SyncCoverMechanismKind::Barrier;
  barrier.barrier = SyncCoverBarrierPlacement{1, barrierTarget, 0};
  barrier.supplyEdges.push_back(completionEdge(source, barrierTarget));
  const SyncCoverMechanismId barrierId = takeMechanismIndex(
      universe.addMechanism(barrier), passed, "add barrier mechanism");
  passed &= check(universe.addConflict(eventId, barrierId),
                  "record alternative implementations as a conflict");
  passed &= check(universe.getMechanisms()[eventId].conflicts ==
                          std::vector<SyncCoverMechanismId>{barrierId} &&
                      universe.getMechanisms()[barrierId].conflicts ==
                          std::vector<SyncCoverMechanismId>{eventId},
                  "conflicts are symmetric and deterministic");
  passed &= check(universe.validate(), "mixed mechanism universe validates");
  return passed;
}

bool testActionAndBindingRejections() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0), passed, "add unqualified source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add binding target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId eventDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add binding event domain");
  const SyncCoverResourceDomainId tokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 2, 9001),
                         passed, "add binding token domain");

  SyncCoverMechanismDescriptor unknownDomain;
  appendCanonicalUse(unknownDomain, 7, 1, 2, source, target);
  passed &= check(universe.addMechanism(unknownDomain).error ==
                      SyncCoverMechanismError::InvalidDomain,
                  "unknown resource domain is rejected");

  SyncCoverMechanismDescriptor unqualified;
  appendCanonicalUse(unqualified, eventDomain, 1, 2, source, target);
  passed &= check(universe.addMechanism(unqualified).error ==
                      SyncCoverMechanismError::InvalidBinding,
                  "event source needs destination-specific completion");

  SyncCoverMechanismDescriptor tokenOnly;
  appendCanonicalUse(tokenOnly, tokenDomain, 1, 2, source, target);
  passed &= check(universe.addMechanism(tokenOnly).error ==
                      SyncCoverMechanismError::InvalidResourceUse,
                  "ordinary event bundle must consume EventId resources");

  SyncCoverGraph qualifiedGraph;
  const SyncCoverNodeId qualifiedSource =
      takeGraphIndex(qualifiedGraph.addNode(1, 1, 0, 0, {}, {2}), passed,
                     "add qualified source");
  const SyncCoverNodeId qualifiedTarget = takeGraphIndex(
      qualifiedGraph.addNode(2, 1, 0, 1), passed, "add qualified target");
  const SyncCoverScopeId loop = takeGraphIndex(
      qualifiedGraph.addScope(0, false, SyncCoverTimelineInterval{0, 5}, true),
      passed, "add recurrence timeline");
  const SyncCoverNodeId loopSource =
      takeGraphIndex(qualifiedGraph.addNode(1, 1, loop, 1, {}, {2}), passed,
                     "add recurrence source");
  const SyncCoverNodeId loopTarget = takeGraphIndex(
      qualifiedGraph.addNode(2, 1, loop, 2), passed, "add recurrence target");
  SyncCoverMechanismUniverse qualified(qualifiedGraph);
  const SyncCoverResourceDomainId qualifiedDomain = takeMechanismIndex(
      qualified.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add qualified domain");

  SyncCoverMechanismDescriptor unboundAction;
  appendCanonicalUse(unboundAction, qualifiedDomain, 1, 2, qualifiedSource,
                     qualifiedTarget);
  unboundAction.actions.push_back(action(SyncCoverResourceActionKind::Produce,
                                         1, SyncCoverAnchorKind::AfterNode,
                                         qualifiedSource));
  passed &= check(qualified.addMechanism(unboundAction).error ==
                      SyncCoverMechanismError::InvalidAction,
                  "every physical action must be resource-accounted");

  SyncCoverMechanismDescriptor wrongBinding;
  appendCanonicalUse(wrongBinding, qualifiedDomain, 1, 2, qualifiedSource,
                     qualifiedTarget);
  wrongBinding.supplyBindings.front().consumeAction = 0;
  passed &= check(qualified.addMechanism(wrongBinding).error ==
                      SyncCoverMechanismError::InvalidBinding,
                  "supply binding requires distinct Produce and Consume");

  SyncCoverMechanismDescriptor recurrence;
  appendCanonicalUse(recurrence, qualifiedDomain, 1, 2, loopSource, loopTarget,
                     loop, 1);
  passed &= check(qualified.addMechanism(recurrence).error ==
                      SyncCoverMechanismError::InvalidBinding,
                  "ordinary EventBundle cannot claim recurrence protocols");
  recurrence.kind = SyncCoverMechanismKind::VerifiedProtocol;
  passed &=
      check(qualified.addVerifiedProtocol(recurrence,
                                          [](const auto &) { return true; }),
            "verified protocol may claim a positive-distance event lifetime");
  return passed;
}

bool testSharedPhysicalActionsAcrossResourceKinds() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add shared source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add shared target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId eventDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add shared event domain");
  const SyncCoverResourceDomainId tokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 2, 4401),
                         passed, "add shared token domain");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::VerifiedProtocol;
  ownership.supplyEdges.push_back(completionEdge(source, target));
  ownership.actions.push_back(action(SyncCoverResourceActionKind::Produce, 1,
                                     SyncCoverAnchorKind::AfterNode, source));
  ownership.actions.push_back(action(SyncCoverResourceActionKind::Consume, 2,
                                     SyncCoverAnchorKind::BeforeNode, target));
  SyncCoverResourceUse eventUse;
  eventUse.domain = eventDomain;
  eventUse.actions = {0, 1};
  eventUse.supplyEdges = {0};
  ownership.resourceUses.push_back(eventUse);
  SyncCoverResourceUse tokenUse = eventUse;
  tokenUse.domain = tokenDomain;
  ownership.resourceUses.push_back(tokenUse);
  ownership.supplyBindings.push_back({0, 0, 0, 1});
  passed &=
      check(universe.addVerifiedProtocol(ownership,
                                         [](const auto &) { return true; }),
            "one physical action may carry EventId and BufferToken proofs");

  SyncCoverMechanismDescriptor duplicateKind = ownership;
  duplicateKind.resourceUses.push_back(eventUse);
  passed &=
      check(universe.addVerifiedProtocol(duplicateKind,
                                         [](const auto &) { return true; })
                    .error == SyncCoverMechanismError::InvalidAction,
            "one action cannot be charged twice to the same resource kind");
  return passed;
}

bool testMixedStartupAndRecurrenceProtocolLifetime() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeGraphIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 15}, true), passed,
      "add mixed protocol loop");
  const SyncCoverScopeId startup = takeGraphIndex(
      graph.addScope(loop, true, SyncCoverTimelineInterval{2, 13}), passed,
      "add mixed protocol startup scope");
  const SyncCoverScopeId innerLoop = takeGraphIndex(
      graph.addScope(startup, false, SyncCoverTimelineInterval{6, 13}, true),
      passed, "add mixed protocol inner loop");
  const SyncCoverNodeId outerSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 0), passed, "add outer startup source");
  const SyncCoverNodeId startupSource = takeGraphIndex(
      graph.addNode(1, 1, startup, 1), passed, "add startup source");
  const SyncCoverNodeId recurrenceTarget = takeGraphIndex(
      graph.addNode(2, 1, innerLoop, 3), passed, "add recurrence target");
  const SyncCoverNodeId startupTarget = takeGraphIndex(
      graph.addNode(2, 1, startup, 4), passed, "add startup target");
  const SyncCoverNodeId recurrenceSource = takeGraphIndex(
      graph.addNode(1, 1, innerLoop, 6), passed, "add recurrence source");
  const SyncCoverNodeId outerTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 7), passed, "add outer drain target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId eventDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add mixed protocol domain");
  const SyncCoverResourceDomain &domain =
      universe.getResourceDomains()[eventDomain];

  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, 8801);
  const SyncCoverDescriptorActionRef startupSet = builder.addAction(
      SyncCoverResourceActionKind::Produce, 1,
      {SyncCoverAnchorKind::AfterNode, startupSource, 0});
  const SyncCoverDescriptorActionRef startupWait = builder.addAction(
      SyncCoverResourceActionKind::Consume, 2,
      {SyncCoverAnchorKind::BeforeNode, startupTarget, 0});
  const SyncCoverDescriptorActionRef bodySet = builder.addAction(
      SyncCoverResourceActionKind::Produce, 1,
      {SyncCoverAnchorKind::AfterNode, recurrenceSource, 0});
  const SyncCoverDescriptorActionRef bodyWait = builder.addAction(
      SyncCoverResourceActionKind::Consume, 2,
      {SyncCoverAnchorKind::BeforeNode, recurrenceTarget, 0});
  const SyncCoverDescriptorActionRef primeSet = builder.addAction(
      SyncCoverResourceActionKind::Produce, 1,
      {SyncCoverAnchorKind::ScopeEntry, 0, loop});
  const SyncCoverDescriptorActionRef drainWait = builder.addAction(
      SyncCoverResourceActionKind::Consume, 2,
      {SyncCoverAnchorKind::ScopeExit, 0, loop});
  SyncCoverEdge startupEdge = completionEdge(startupSource, startupTarget);
  startupEdge.scope = startup;
  SyncCoverEdge recurrenceEdge =
      completionEdge(recurrenceSource, recurrenceTarget, innerLoop, 1);
  SyncCoverEdge outerEdge = completionEdge(outerSource, outerTarget);
  passed &= check(
      builder.addProtocolLane(
          domain, loop, 1, 1,
          {startupSet, startupWait, bodySet, bodyWait, primeSet, drainWait},
          {{startupEdge, startupSet, startupWait},
           {recurrenceEdge, bodySet, bodyWait},
           {outerEdge, primeSet, drainWait}}),
      "one outer cyclic lifetime accepts descendant recurrence supplies");
  SyncCoverMechanismDescriptor descriptor =
      std::move(builder).takeDescriptor();
  passed &= check(
      universe.addVerifiedProtocol(descriptor,
                                   [](const auto &) { return true; }),
      "mixed startup and recurrence protocol validates atomically");

  SyncCoverMechanismDescriptor invalidScope = descriptor;
  invalidScope.supplyEdges[1].scope = 0;
  passed &= check(
      universe.addVerifiedProtocol(invalidScope,
                                   [](const auto &) { return true; })
              .error == SyncCoverMechanismError::InvalidResourceUse,
      "outer lifetime rejects recurrence supplies outside its scope tree");
  return passed;
}

bool testBarrierExecutionGuarantees() {
  bool passed = true;

  {
    SyncCoverGraph graph;
    const SyncCoverNodeId source = takeGraphIndex(
        graph.addNode(1, 1, 0, 0), passed, "add exact barrier source");
    const SyncCoverNodeId target = takeGraphIndex(
        graph.addNode(1, 1, 0, 2), passed, "add exact barrier target");
    const SyncCoverNodeId crossPipeTarget = takeGraphIndex(
        graph.addNode(2, 1, 0, 3), passed,
        "add exact cross-pipe barrier target");
    SyncCoverMechanismUniverse universe(graph);
    SyncCoverMechanismDescriptor barrier;
    barrier.kind = SyncCoverMechanismKind::Barrier;
    barrier.barrier = SyncCoverBarrierPlacement{
        1, {SyncCoverAnchorKind::TimelinePoint, 0, 0, 2}, 0};
    barrier.supplyEdges.push_back(completionEdge(source, target));
    passed &= check(universe.addMechanism(barrier),
                    "barrier may occupy an exact structured timeline point");

    SyncCoverMechanismDescriptor crossPipeBarrier = barrier;
    crossPipeBarrier.providerIdentity = 1;
    crossPipeBarrier.supplyEdges.clear();
    crossPipeBarrier.supplyEdges.push_back(
        completionEdge(source, crossPipeTarget));
    passed &= check(
        universe.addMechanism(crossPipeBarrier),
        "barrier completion may supply a later cross-pipe consumer");
  }

  {
    SyncCoverGraph graph;
    const SyncCoverControlId branch =
        takeGraphIndex(graph.addControl(2), passed, "add barrier branch");
    const SyncCoverNodeId source = takeGraphIndex(
        graph.addNode(1, 1, 0, 0), passed, "add conditional source");
    SyncCoverGuard branchGuard;
    branchGuard.literals.push_back({branch, 0});
    const SyncCoverNodeId anchor =
        takeGraphIndex(graph.addNode(1, 1, 0, 1, branchGuard), passed,
                       "add conditional anchor");
    const SyncCoverNodeId target = takeGraphIndex(
        graph.addNode(1, 1, 0, 2), passed, "add unconditional target");
    const SyncCoverNodeId guardedTarget =
        takeGraphIndex(graph.addNode(1, 1, 0, 3, branchGuard), passed,
                       "add guarded barrier target");
    SyncCoverMechanismUniverse universe(graph);
    SyncCoverMechanismDescriptor barrier;
    barrier.kind = SyncCoverMechanismKind::Barrier;
    barrier.barrier = SyncCoverBarrierPlacement{1, anchor, 0};
    barrier.supplyEdges.push_back(completionEdge(source, target));
    passed &= check(universe.addMechanism(barrier).error ==
                        SyncCoverMechanismError::InvalidSupply,
                    "conditional barrier cannot cover unconditional target");

    barrier.supplyEdges.clear();
    barrier.supplyEdges.push_back(completionEdge(source, guardedTarget));
    passed &= check(universe.addMechanism(barrier),
                    "matching target guard guarantees barrier execution");
  }

  {
    SyncCoverGraph graph;
    const SyncCoverScopeId optional = takeGraphIndex(
        graph.addScope(0, false), passed, "add optional barrier scope");
    const SyncCoverNodeId source = takeGraphIndex(
        graph.addNode(1, 1, 0, 0), passed, "add optional source");
    const SyncCoverNodeId anchor = takeGraphIndex(
        graph.addNode(1, 1, optional, 1), passed, "add optional anchor");
    const SyncCoverNodeId target =
        takeGraphIndex(graph.addNode(1, 1, 0, 2), passed, "add outer target");
    SyncCoverMechanismUniverse universe(graph);
    SyncCoverMechanismDescriptor barrier;
    barrier.kind = SyncCoverMechanismKind::Barrier;
    barrier.barrier = SyncCoverBarrierPlacement{1, anchor, optional};
    barrier.supplyEdges.push_back(completionEdge(source, target));
    passed &= check(universe.addMechanism(barrier).error ==
                        SyncCoverMechanismError::InvalidSupply,
                    "zero-trip nested barrier cannot cover outer target");
  }

  {
    SyncCoverGraph graph;
    const SyncCoverScopeId required = takeGraphIndex(
        graph.addScope(0, true), passed, "add required barrier scope");
    const SyncCoverNodeId source = takeGraphIndex(
        graph.addNode(1, 1, 0, 0), passed, "add required source");
    const SyncCoverNodeId anchor = takeGraphIndex(
        graph.addNode(1, 1, required, 1), passed, "add required anchor");
    const SyncCoverNodeId target = takeGraphIndex(
        graph.addNode(1, 1, 0, 2), passed, "add required outer target");
    SyncCoverMechanismUniverse universe(graph);
    SyncCoverMechanismDescriptor barrier;
    barrier.kind = SyncCoverMechanismKind::Barrier;
    barrier.barrier = SyncCoverBarrierPlacement{1, anchor, required};
    barrier.supplyEdges.push_back(completionEdge(source, target));
    passed &= check(universe.addMechanism(barrier),
                    "must-execute nested barrier may cover outer target");
  }

  return passed;
}

bool testResourceSelectionFeasibility() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add first pressure source");
  const SyncCoverNodeId secondSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 1, {}, {2}), passed, "add second pressure source");
  const SyncCoverNodeId secondTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 2), passed, "add second pressure target");
  const SyncCoverNodeId firstTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 3), passed, "add first pressure target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 4, 0,
                                 {1, 9}),
      passed, "add pressure domain");

  SyncCoverMechanismDescriptor first;
  appendCanonicalUse(first, domain, 1, 2, firstSource, firstTarget, 0, 0, 2);
  const SyncCoverMechanismId firstId = takeMechanismIndex(
      universe.addMechanism(first), passed, "add first pressure mechanism");
  SyncCoverMechanismDescriptor second;
  appendCanonicalUse(second, domain, 1, 2, secondSource, secondTarget, 0, 0, 2);
  const SyncCoverMechanismId secondId = takeMechanismIndex(
      universe.addMechanism(second), passed, "add second pressure mechanism");

  const SyncCoverResourceSelection pressure =
      universe.evaluateResourceSelection({secondId, firstId});
  const std::vector<SyncCoverResourceWitnessUse> expectedWitness = {
      {firstId, 0, 2}, {secondId, 0, 2}};
  passed &= check(
      pressure.isValid() && !pressure && !pressure.resourceFeasible &&
          pressure.domains.size() == 1 &&
          pressure.domains[0].required == 4 &&
          pressure.domains[0].available == 3 &&
          pressure.domains[0].overflow == 1 &&
          pressure.domains[0].maximumPoint == 3 &&
          pressure.domains[0].maximumClique == expectedWitness,
      "selection reports exact weighted pressure and stable owners");
  passed &= check(universe.evaluateStructuralCost({firstId, secondId}).error ==
                      SyncCoverStructuralCostError::ResourceInfeasible,
                  "structural cost rejects an over-budget complete selection");
  const SyncCoverResourceSelection single =
      universe.evaluateResourceSelection({firstId});
  passed &=
      check(single && single.resourceFeasible &&
                single.domains[0].allocations ==
                    std::vector<SyncCoverResourceAllocation>(
                        {{{firstId, 0, 2}, {0, 2}}}),
            "feasible uses receive deterministic nonreserved physical IDs");
  const SyncCoverResourceSelection empty =
      universe.evaluateResourceSelection({});
  passed &= check(
      empty && empty.resourceFeasible && empty.domains.size() == 1 &&
          empty.domains[0].required == 0 && empty.domains[0].available == 3,
      "empty selection retains domain availability diagnostics");
  const SyncCoverStructuralCost emptyCost = universe.evaluateStructuralCost({});
  passed &= check(emptyCost && emptyCost.eventDomainCount == 0 &&
                      emptyCost.peakEventPressure == 0 &&
                      emptyCost.minimumEventHeadroom == 0,
                  "unused event domains do not affect selected-plan cost");
  passed &=
      check(universe.evaluateResourceSelection({firstId, firstId}).error ==
                SyncCoverResourceSelectionError::InvalidSelection,
            "duplicate selection IDs fail closed");
  passed &= check(universe.evaluateResourceSelection({17}).error ==
                      SyncCoverResourceSelectionError::InvalidSelection,
                  "unknown selection IDs fail closed");
  const SyncCoverSelectionEvaluator epoch(universe);
  passed &= check(epoch && epoch.evaluate({firstId}),
                  "selection epoch validates the completed universe once");
  passed &= check(universe.addConflict(firstId, secondId),
                  "add pressure mechanism conflict");
  passed &= check(!epoch.evaluate({firstId}),
                  "selection epoch rejects later universe mutations");
  const SyncCoverResourceSelection conflict =
      universe.evaluateResourceSelection({firstId, secondId});
  passed &= check(conflict.error == SyncCoverResourceSelectionError::Conflict &&
                      conflict.firstConflict == firstId &&
                      conflict.secondConflict == secondId,
                  "selected conflicts are diagnosed deterministically");
  return passed;
}

bool testIndependentDomainsReuseAndOverflow() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId firstSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add forward source");
  const SyncCoverNodeId firstTarget =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add forward target");
  const SyncCoverNodeId reverseSource = takeGraphIndex(
      graph.addNode(2, 1, 0, 2, {}, {1}), passed, "add reverse source");
  const SyncCoverNodeId reverseTarget =
      takeGraphIndex(graph.addNode(1, 1, 0, 3), passed, "add reverse target");
  const SyncCoverNodeId laterSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 4, {}, {2}), passed, "add later forward source");
  const SyncCoverNodeId laterTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 5), passed, "add later forward target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId forwardDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 1),
      passed, "add forward event domain");
  const SyncCoverResourceDomainId reverseDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 1, 1),
      passed, "add reverse event domain");
  const SyncCoverResourceDomainId firstTokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 1, 101),
                         passed, "add first token pool");
  const SyncCoverResourceDomainId secondTokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 1, 202),
                         passed, "add second token pool");

  SyncCoverMechanismDescriptor first;
  appendCanonicalUse(first, forwardDomain, 1, 2, firstSource, firstTarget);
  const SyncCoverMechanismId firstId = takeMechanismIndex(
      universe.addMechanism(first), passed, "add first forward event");
  SyncCoverMechanismDescriptor reverse;
  appendCanonicalUse(reverse, reverseDomain, 2, 1, reverseSource,
                     reverseTarget);
  const SyncCoverMechanismId reverseId = takeMechanismIndex(
      universe.addMechanism(reverse), passed, "add reverse event");
  SyncCoverMechanismDescriptor later;
  appendCanonicalUse(later, forwardDomain, 1, 2, laterSource, laterTarget);
  const SyncCoverMechanismId laterId = takeMechanismIndex(
      universe.addMechanism(later), passed, "add later forward event");

  SyncCoverMechanismDescriptor firstToken;
  firstToken.kind = SyncCoverMechanismKind::VerifiedProtocol;
  appendCanonicalUse(firstToken, firstTokenDomain, 1, 2, firstSource,
                     firstTarget);
  const SyncCoverMechanismId firstTokenId =
      takeMechanismIndex(universe.addVerifiedProtocol(
                             firstToken, [](const auto &) { return true; }),
                         passed, "add first verified token protocol");
  SyncCoverMechanismDescriptor secondToken;
  secondToken.kind = SyncCoverMechanismKind::VerifiedProtocol;
  appendCanonicalUse(secondToken, secondTokenDomain, 1, 2, firstSource,
                     firstTarget);
  const SyncCoverMechanismId secondTokenId =
      takeMechanismIndex(universe.addVerifiedProtocol(
                             secondToken, [](const auto &) { return true; }),
                         passed, "add second verified token protocol");

  const SyncCoverResourceSelection independent =
      universe.evaluateResourceSelection(
          {firstId, reverseId, laterId, firstTokenId, secondTokenId});
  passed &= check(
      independent && independent.resourceFeasible &&
          independent.domains.size() == 4 &&
          independent.domains[forwardDomain].allocations ==
              std::vector<SyncCoverResourceAllocation>(
                  {{{firstId, 0, 1}, {0}}, {{laterId, 0, 1}, {0}}}) &&
          independent.domains[reverseDomain].allocations ==
              std::vector<SyncCoverResourceAllocation>(
                  {{{reverseId, 0, 1}, {0}}}) &&
          independent.domains[firstTokenDomain].allocations ==
              std::vector<SyncCoverResourceAllocation>(
                  {{{firstTokenId, 0, 1}, {0}}}) &&
          independent.domains[secondTokenDomain].allocations ==
              std::vector<SyncCoverResourceAllocation>(
                  {{{secondTokenId, 0, 1}, {0}}}),
      "directed event domains and token pools allocate independently while "
      "disjoint lifetimes reuse physical IDs");
  const bool hasForwardAllocations =
      independent && independent.domains.size() > forwardDomain &&
      independent.domains[forwardDomain].allocations.size() == 2 &&
      independent.domains[forwardDomain].allocations[0].ids.size() == 1 &&
      independent.domains[forwardDomain].allocations[1].ids.size() == 1;
  if (hasForwardAllocations) {
    const std::vector<SyncAllocatedInterval> forwardAllocation = {
        {{1, 2}, independent.domains[forwardDomain].allocations[0].ids[0]},
        {{9, 10}, independent.domains[forwardDomain].allocations[1].ids[0]}};
    passed &= check(verifySyncIntervalAllocation(1, {}, forwardAllocation),
                    "resource API allocations pass the independent verifier");
  } else {
    passed &= check(false, "resource API returned usable forward allocations");
  }

  SyncCoverMechanismDescriptor overflow;
  appendCanonicalUse(overflow, forwardDomain, 1, 2, firstSource, firstTarget, 0,
                     0, std::numeric_limits<std::size_t>::max());
  appendCanonicalUse(overflow, forwardDomain, 1, 2, firstSource, firstTarget);
  const SyncCoverMechanismId overflowId = takeMechanismIndex(
      universe.addMechanism(overflow), passed, "add overflow event bundle");
  passed &= check(
      universe.evaluateResourceSelection({overflowId}).error ==
          SyncCoverResourceSelectionError::ArithmeticOverflow,
      "resource selection propagates weighted-pressure arithmetic overflow");
  return passed;
}

bool testRecurrenceUsesWholeScopePressure() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeGraphIndex(
      graph.addScope(0, false, SyncCoverTimelineInterval{0, 9}, true), passed,
      "add pressure recurrence scope");
  const SyncCoverNodeId recurrenceSource = takeGraphIndex(
      graph.addNode(1, 1, loop, 1), passed, "add recurrence pressure source");
  const SyncCoverNodeId recurrenceTarget = takeGraphIndex(
      graph.addNode(2, 1, loop, 2), passed, "add recurrence pressure target");
  const SyncCoverNodeId straightSource =
      takeGraphIndex(graph.addNode(1, 1, loop, 3, {}, {2}), passed,
                     "add straight pressure source");
  const SyncCoverNodeId straightTarget = takeGraphIndex(
      graph.addNode(2, 1, loop, 4), passed, "add straight pressure target");
  const SyncCoverNodeId outerEventSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 5, {}, {2}), passed,
      "add outer cost event source");
  const SyncCoverNodeId outerEventTarget = takeGraphIndex(
      graph.addNode(2, 1, 0, 6), passed, "add outer cost event target");
  const SyncCoverNodeId outerBarrierSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 7), passed, "add outer cost barrier source");
  const SyncCoverNodeId outerBarrierTarget = takeGraphIndex(
      graph.addNode(1, 1, 0, 8), passed, "add outer cost barrier target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 3),
      passed, "add recurrence pressure domain");

  SyncCoverMechanismDescriptor recurrence;
  recurrence.kind = SyncCoverMechanismKind::VerifiedProtocol;
  appendCanonicalUse(recurrence, domain, 1, 2, recurrenceSource,
                     recurrenceTarget, loop, 1, 2);
  SyncCoverResourceAction prime;
  prime.kind = SyncCoverResourceActionKind::Produce;
  prime.resource = 1;
  prime.anchor = {SyncCoverAnchorKind::ScopeEntry, 0, loop};
  recurrence.actions.push_back(prime);
  recurrence.resourceUses.front().actions.push_back(2);
  SyncCoverResourceAction drain;
  drain.kind = SyncCoverResourceActionKind::Consume;
  drain.resource = 2;
  drain.anchor = {SyncCoverAnchorKind::ScopeExit, 0, loop};
  recurrence.actions.push_back(drain);
  recurrence.resourceUses.front().actions.push_back(3);
  const SyncCoverMechanismId recurrenceId =
      takeMechanismIndex(universe.addVerifiedProtocol(
                             recurrence, [](const auto &) { return true; }),
                         passed, "add verified recurrence pressure protocol");
  SyncCoverMechanismDescriptor straight;
  appendCanonicalUse(straight, domain, 1, 2, straightSource, straightTarget);
  const SyncCoverMechanismId straightId =
      takeMechanismIndex(universe.addMechanism(straight), passed,
                         "add straight pressure mechanism");

  const SyncCoverResourceSelection result =
      universe.evaluateResourceSelection({recurrenceId, straightId});
  passed &= check(
      result && result.resourceFeasible && result.domains[0].required == 3 &&
          result.domains[0].overflow == 0 &&
          result.domains[0].maximumClique ==
              std::vector<SyncCoverResourceWitnessUse>(
                  {{recurrenceId, 0, 2}, {straightId, 0, 1}}),
      "positive-distance resources occupy their complete scope timeline");
  const SyncCoverResourceSelection recurrenceOnly =
      universe.evaluateResourceSelection({recurrenceId});
  passed &= check(
      recurrenceOnly && recurrenceOnly.resourceFeasible &&
          recurrenceOnly.domains[0].allocations ==
              std::vector<SyncCoverResourceAllocation>(
                  {{{recurrenceId, 0, 2}, {0, 1}}}),
      "whole-scope recurrence allocations use the authoritative interval");

  SyncCoverMechanismDescriptor barrier;
  barrier.kind = SyncCoverMechanismKind::Barrier;
  barrier.barrier = SyncCoverBarrierPlacement{1, straightSource, loop};
  barrier.supplyEdges.push_back(
      completionEdge(recurrenceSource, straightSource, loop));
  const SyncCoverMechanismId barrierId =
      takeMechanismIndex(universe.addMechanism(barrier), passed,
                         "add loop-local structural-cost barrier");
  SyncCoverMechanismDescriptor outerEvent;
  appendCanonicalUse(outerEvent, domain, 1, 2, outerEventSource,
                     outerEventTarget);
  const SyncCoverMechanismId outerEventId = takeMechanismIndex(
      universe.addMechanism(outerEvent), passed,
      "add outer structural-cost event");
  SyncCoverMechanismDescriptor outerBarrier;
  outerBarrier.kind = SyncCoverMechanismKind::Barrier;
  outerBarrier.barrier =
      SyncCoverBarrierPlacement{1, outerBarrierTarget, 0};
  outerBarrier.supplyEdges.push_back(
      completionEdge(outerBarrierSource, outerBarrierTarget));
  const SyncCoverMechanismId outerBarrierId = takeMechanismIndex(
      universe.addMechanism(outerBarrier), passed,
      "add outer structural-cost barrier");
  const SyncCoverStructuralCost cost =
      universe.evaluateStructuralCost({barrierId, straightId, recurrenceId});
  passed &= check(
      cost && cost.actionProfile == std::vector<std::size_t>({4, 2}) &&
          cost.barrierActionProfile == std::vector<std::size_t>({1, 0}) &&
          cost.peakEventPressure == 3 && cost.totalEventPressure == 3 &&
          cost.minimumEventHeadroom == 0 && cost.eventDomainCount == 1 &&
          cost.mechanismCount == 3 &&
          cost.signature == std::vector<SyncCoverMechanismId>(
                                {recurrenceId, straightId, barrierId}),
      "structural cost separates body actions from prime and drain actions");
  SyncCoverStructuralCost fewerBodyActions = cost;
  --fewerBodyActions.actionProfile.front();
  fewerBodyActions.minimumEventHeadroom = 0;
  passed &= check(syncCoverStructuralCostLess(fewerBodyActions, cost),
                  "deeper-loop physical actions dominate cost ordering");
  SyncCoverStructuralCost bodyBarrier = cost;
  bodyBarrier.actionProfile = {0, 0};
  bodyBarrier.barrierActionProfile = {1, 0};
  SyncCoverStructuralCost bodyEvents = bodyBarrier;
  bodyEvents.actionProfile = {2, 0};
  bodyEvents.barrierActionProfile = {0, 0};
  passed &= check(
      syncCoverStructuralCostLess(bodyEvents, bodyBarrier),
      "same-depth event actions are preferable to a whole-pipe barrier");
  SyncCoverStructuralCost deepEvent = bodyEvents;
  deepEvent.actionProfile = {1, 0};
  SyncCoverStructuralCost shallowBarrier = bodyBarrier;
  shallowBarrier.barrierActionProfile = {0, 1};
  passed &= check(
      syncCoverStructuralCostLess(shallowBarrier, deepEvent),
      "loop depth is compared before shallower barrier strength");
  SyncCoverStructuralCost bodyBarrierWithAction = bodyBarrier;
  bodyBarrierWithAction.actionProfile.front() = 1;
  passed &= check(
      !syncCoverStructuralCostLess(bodyBarrier, bodyBarrier) &&
          syncCoverStructuralCostLess(bodyEvents, bodyBarrier) &&
          !syncCoverStructuralCostLess(bodyBarrier, bodyEvents) &&
          syncCoverStructuralCostLess(bodyBarrier, bodyBarrierWithAction) &&
          syncCoverStructuralCostLess(bodyEvents, bodyBarrierWithAction),
      "loop-profile ordering is irreflexive, asymmetric, and transitive");
  SyncCoverStructuralCost shortProfiles = bodyEvents;
  shortProfiles.actionProfile = {2};
  shortProfiles.barrierActionProfile = {0};
  passed &= check(
      !syncCoverStructuralCostLess(shortProfiles, bodyEvents) &&
          !syncCoverStructuralCostLess(bodyEvents, shortProfiles),
      "missing trailing profile entries are equivalent to zero");

  const auto additionIsMonotone = [&](SyncCoverMechanismId base,
                                      SyncCoverMechanismId added) {
    const SyncCoverStructuralCost baseCost =
        universe.evaluateStructuralCost({base});
    std::vector<SyncCoverMechanismId> combined{base, added};
    std::sort(combined.begin(), combined.end());
    const SyncCoverStructuralCost combinedCost =
        universe.evaluateStructuralCost(combined);
    return baseCost && combinedCost &&
           !syncCoverStructuralCostLess(combinedCost, baseCost);
  };
  passed &= check(
      additionIsMonotone(straightId, barrierId) &&
          additionIsMonotone(barrierId, straightId) &&
          additionIsMonotone(straightId, outerEventId) &&
          additionIsMonotone(straightId, outerBarrierId),
      "adding deep or shallow event and barrier mechanisms is monotone");
  SyncCoverStructuralCost moreHeadroom = cost;
  moreHeadroom.minimumEventHeadroom = 1;
  moreHeadroom.peakEventPressure = 2;
  passed &= check(!syncCoverStructuralCostLess(moreHeadroom, cost) &&
                      !syncCoverStructuralCostLess(cost, moreHeadroom),
                  "nonseparable pressure diagnostics do not order plans");
  SyncCoverStructuralCost lowerTotalPressure = cost;
  --lowerTotalPressure.totalEventPressure;
  passed &=
      check(!syncCoverStructuralCostLess(lowerTotalPressure, cost) &&
                !syncCoverStructuralCostLess(cost, lowerTotalPressure),
            "domain pressure remains a diagnostic rather than an objective");
  SyncCoverStructuralCost fewerMechanisms = cost;
  --fewerMechanisms.mechanismCount;
  passed &= check(syncCoverStructuralCostLess(fewerMechanisms, cost),
                  "mechanism count is the final separable cost metric");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testDescriptorBuilderAndGraphEpoch();
  passed &= testExpansionOverlayEpoch();
  passed &= testSharedRecurrenceCoverageParity();
  passed &= testSharedGuardedContextParity();
  passed &= testSharedAncestorBoundaryParity();
  passed &= testUniverseReportsFreezeFailure();
  passed &= testStockUnitRecurrenceProtocol();
  passed &= testAtomicSupplyAndCoverage();
  passed &= testAtomicFailureAndProtocolGate();
  passed &= testMultiEdgeAtomicSupply();
  passed &= testDomainsBarriersAndConflicts();
  passed &= testActionAndBindingRejections();
  passed &= testSharedPhysicalActionsAcrossResourceKinds();
  passed &= testMixedStartupAndRecurrenceProtocolLifetime();
  passed &= testBarrierExecutionGuarantees();
  passed &= testResourceSelectionFeasibility();
  passed &= testIndependentDomainsReuseAndOverflow();
  passed &= testRecurrenceUsesWholeScopePressure();
  return passed ? 0 : 1;
}
