// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncAlgorithms.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

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
  bool invalidVerifierCalled = false;
  passed &= check(universe.addVerifiedProtocol(invalid,
                                               [&](const auto &) {
                                                 invalidVerifierCalled = true;
                                                 return true;
                                               })
                          .error == SyncCoverMechanismError::InvalidSupply,
                  "invalid second edge rejects the whole mechanism");
  passed &= check(!invalidVerifierCalled,
                  "generic validation precedes semantic verification");
  passed &= check(graph.getEdges().size() == edgeCount &&
                      universe.getMechanisms().empty(),
                  "failed mechanism addition is atomic");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::VerifiedProtocol;
  ownership.providerIdentity = 73;
  appendCanonicalUse(ownership, eventDomain, 1, 2, source, target);
  passed &= check(universe.addMechanism(ownership).error ==
                      SyncCoverMechanismError::UnverifiedProtocol,
                  "ownership supply requires protocol verification");
  bool verifierCalled = false;
  passed &= check(universe.addVerifiedProtocol(
                      ownership,
                      [&](const auto &submitted) {
                        verifierCalled = true;
                        return submitted.providerIdentity == 73 &&
                               submitted.supplyBindings.size() == 1;
                      }),
                  "verified ownership protocol is admitted atomically");
  passed &= check(verifierCalled,
                  "verification runs against the submitted descriptor");
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

  const SyncCoverScopeId loop = takeGraphIndex(
      qualifiedGraph.addScope(0, false, SyncCoverTimelineInterval{0, 5}, true),
      passed, "add recurrence timeline");
  const SyncCoverNodeId loopSource =
      takeGraphIndex(qualifiedGraph.addNode(1, 1, loop, 1, {}, {2}), passed,
                     "add recurrence source");
  const SyncCoverNodeId loopTarget = takeGraphIndex(
      qualifiedGraph.addNode(2, 1, loop, 2), passed, "add recurrence target");
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

bool testBarrierExecutionGuarantees() {
  bool passed = true;

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
    SyncCoverMechanismUniverse universe(graph);
    SyncCoverMechanismDescriptor barrier;
    barrier.kind = SyncCoverMechanismKind::Barrier;
    barrier.barrier = SyncCoverBarrierPlacement{1, anchor, 0};
    barrier.supplyEdges.push_back(completionEdge(source, target));
    passed &= check(universe.addMechanism(barrier).error ==
                        SyncCoverMechanismError::InvalidSupply,
                    "conditional barrier cannot cover unconditional target");

    const SyncCoverNodeId guardedTarget =
        takeGraphIndex(graph.addNode(1, 1, 0, 3, branchGuard), passed,
                       "add guarded barrier target");
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
      pressure && !pressure.resourceFeasible && pressure.domains.size() == 1 &&
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
  passed &= check(universe.addConflict(firstId, secondId),
                  "add pressure mechanism conflict");
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
  SyncCoverStructuralCost moreHeadroom = cost;
  moreHeadroom.minimumEventHeadroom = 1;
  passed &= check(syncCoverStructuralCostLess(moreHeadroom, cost),
                  "event headroom breaks otherwise equal structural costs");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testAtomicSupplyAndCoverage();
  passed &= testAtomicFailureAndProtocolGate();
  passed &= testMultiEdgeAtomicSupply();
  passed &= testDomainsBarriersAndConflicts();
  passed &= testActionAndBindingRejections();
  passed &= testSharedPhysicalActionsAcrossResourceKinds();
  passed &= testBarrierExecutionGuarantees();
  passed &= testResourceSelectionFeasibility();
  passed &= testIndependentDomainsReuseAndOverflow();
  passed &= testRecurrenceUsesWholeScopePressure();
  return passed ? 0 : 1;
}
