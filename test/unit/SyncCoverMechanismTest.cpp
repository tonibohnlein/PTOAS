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
#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include <iostream>
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

SyncCoverEdge completionEdge(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverEdge edge;
  edge.source = source;
  edge.target = target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  return edge;
}

SyncCoverDemand makeDemand(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  return demand;
}

SyncCoverResourceUse makeUse(SyncCoverResourceDomainId domain,
                             SyncCoverNodeId begin, SyncCoverNodeId end,
                             std::size_t supplyEdge = 0) {
  SyncCoverResourceUse use;
  use.domain = domain;
  use.begin = begin;
  use.end = end;
  use.supplyEdges.push_back(supplyEdge);
  return use;
}

bool testAtomicSupplyAndCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add source");
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
  event.kind = SyncCoverMechanismKind::EventBundle;
  event.providerIdentity = 41;
  event.supplyEdges.push_back(completionEdge(source, target));
  event.resourceUses.push_back(makeUse(domain, source, target));
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
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add failure domain");

  const SyncCoverResourceDomainId tokenDomain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::BufferToken, 2, 2, 32,
                                 7001),
      passed, "add failure token pool");
  SyncCoverMechanismDescriptor invalid;
  invalid.kind = SyncCoverMechanismKind::OwnershipProtocol;
  invalid.resourceUses.push_back(makeUse(domain, source, target));
  invalid.resourceUses.push_back(makeUse(tokenDomain, target, target, 1));
  invalid.supplyEdges.push_back(completionEdge(source, target));
  invalid.supplyEdges.push_back(completionEdge(target, target));
  const std::size_t edgeCount = graph.getEdges().size();
  passed &= check(
      universe.addVerifiedProtocol(invalid, [](const auto &) { return true; })
              .error == SyncCoverMechanismError::InvalidSupply,
      "invalid second edge rejects the whole mechanism");
  passed &= check(graph.getEdges().size() == edgeCount &&
                      universe.getMechanisms().empty(),
                  "failed mechanism addition is atomic");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::OwnershipProtocol;
  ownership.providerIdentity = 73;
  ownership.resourceUses.push_back(makeUse(domain, source, target));
  ownership.supplyEdges.push_back(completionEdge(source, target));
  passed &= check(universe.addMechanism(ownership).error ==
                      SyncCoverMechanismError::UnverifiedProtocol,
                  "ownership supply requires protocol verification");
  bool verifierCalled = false;
  passed &= check(
      universe.addVerifiedProtocol(ownership,
                                   [&](const auto &submitted) {
                                     verifierCalled = true;
                                     return submitted.providerIdentity == 73 &&
                                            submitted.supplyEdges.size() == 1;
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
      graph.addNode(1, 1, 0, 0), passed, "add first bundle source");
  const SyncCoverNodeId secondSource = takeGraphIndex(
      graph.addNode(1, 1, 0, 1), passed, "add second bundle source");
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
  bundle.supplyEdges.push_back(completionEdge(firstSource, firstTarget));
  bundle.supplyEdges.push_back(completionEdge(secondSource, secondTarget));
  bundle.resourceUses.push_back(makeUse(domain, firstSource, firstTarget, 0));
  bundle.resourceUses.push_back(makeUse(domain, secondSource, secondTarget, 1));
  const SyncCoverMechanismId bundleId = takeMechanismIndex(
      universe.addMechanism(bundle), passed, "add multi-edge bundle");

  const SyncCoverMechanism &stored = universe.getMechanisms()[bundleId];
  passed &= check(
      stored.supplyEdges == std::vector<std::size_t>{1, 2} &&
          stored.resourceUses[0].supplyEdges == std::vector<std::size_t>{1} &&
          stored.resourceUses[1].supplyEdges == std::vector<std::size_t>{2},
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
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add conflict source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add conflict target");
  const SyncCoverNodeId barrierTarget =
      takeGraphIndex(graph.addNode(1, 1, 0, 2), passed, "add barrier target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add conflict domain");
  const SyncCoverMechanismResult duplicate =
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8);
  passed &= check(duplicate && duplicate.index == domain,
                  "identical domain registration is idempotent");
  passed &=
      check(universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 7)
                    .error == SyncCoverMechanismError::InvalidDomain,
            "one directed domain cannot have conflicting budgets");
  passed &=
      check(universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 1, 8)
                    .error == SyncCoverMechanismError::InvalidDomain,
            "event IDs require a cross-resource domain");
  const SyncCoverResourceDomainId firstToken = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::BufferToken, 1, 1, 32,
                                 1001),
      passed, "add first token pool");
  const SyncCoverResourceDomainId secondToken = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::BufferToken, 1, 1, 32,
                                 1002),
      passed, "add second token pool");
  passed &= check(firstToken != secondToken,
                  "independent token pools retain distinct domains");

  SyncCoverMechanismDescriptor event;
  event.resourceUses.push_back(makeUse(domain, source, target));
  event.supplyEdges.push_back(completionEdge(source, target));
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

bool testInvalidResourceUses() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add invalid source");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 1), passed, "add invalid target");
  SyncCoverMechanismUniverse universe(graph);
  SyncCoverMechanismDescriptor event;
  event.supplyEdges.push_back(completionEdge(source, target));
  event.resourceUses.push_back(makeUse(3, source, target));
  passed &= check(universe.addMechanism(event).error ==
                      SyncCoverMechanismError::InvalidDomain,
                  "unknown resource domain is rejected");
  event.resourceUses.clear();
  passed &= check(universe.addMechanism(event).error ==
                      SyncCoverMechanismError::InvalidResourceUse,
                  "event bundle must declare its resource lifetime");
  return passed;
}

bool testPlacementBindingAndVerifierRejections() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source =
      takeGraphIndex(graph.addNode(1, 1, 0, 0), passed, "add bound source");
  const SyncCoverNodeId anchor =
      takeGraphIndex(graph.addNode(1, 1, 0, 1), passed, "add bound anchor");
  const SyncCoverNodeId target =
      takeGraphIndex(graph.addNode(2, 1, 0, 2), passed, "add bound target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId domain = takeMechanismIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add binding domain");

  SyncCoverMechanismDescriptor mismatched;
  mismatched.supplyEdges.push_back(completionEdge(source, target));
  mismatched.resourceUses.push_back(makeUse(domain, anchor, target));
  passed &= check(universe.addMechanism(mismatched).error ==
                      SyncCoverMechanismError::InvalidResourceUse,
                  "resource lifetime endpoints must match supplied edge");
  mismatched.resourceUses.clear();
  mismatched.resourceUses.push_back(makeUse(domain, source, target, 1));
  passed &= check(universe.addMechanism(mismatched).error ==
                      SyncCoverMechanismError::InvalidResourceUse,
                  "resource lifetime must name an existing local supply");

  SyncCoverMechanismDescriptor barrier;
  barrier.kind = SyncCoverMechanismKind::Barrier;
  barrier.supplyEdges.push_back(completionEdge(source, anchor));
  passed &= check(universe.addMechanism(barrier).error ==
                      SyncCoverMechanismError::InvalidMechanism,
                  "barrier supply requires a concrete placement");
  barrier.barrier = SyncCoverBarrierPlacement{1, anchor, 0};
  barrier.supplyEdges.clear();
  barrier.supplyEdges.push_back(completionEdge(source, target));
  passed &= check(universe.addMechanism(barrier).error ==
                      SyncCoverMechanismError::InvalidSupply,
                  "barrier cannot supply a cross-resource edge");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::OwnershipProtocol;
  ownership.supplyEdges.push_back(completionEdge(source, target));
  ownership.resourceUses.push_back(makeUse(domain, source, target));
  const std::size_t edgeCount = graph.getEdges().size();
  passed &= check(universe.addVerifiedProtocol(
                              ownership, [](const auto &) { return false; })
                          .error == SyncCoverMechanismError::UnverifiedProtocol,
                  "failed exact-descriptor verification rejects protocol");
  passed &= check(graph.getEdges().size() == edgeCount,
                  "failed protocol verification leaves graph unchanged");
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

bool testMechanismResourceComposition() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeGraphIndex(
      graph.addNode(1, 1, 0, 0), passed, "add composition source");
  const SyncCoverNodeId target = takeGraphIndex(
      graph.addNode(2, 1, 0, 1), passed, "add composition target");
  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId tokenDomain =
      takeMechanismIndex(universe.addResourceDomain(
                             SyncCoverResourceKind::BufferToken, 1, 2, 2, 9001),
                         passed, "add composition token pool");

  SyncCoverMechanismDescriptor event;
  event.kind = SyncCoverMechanismKind::EventBundle;
  event.supplyEdges.push_back(completionEdge(source, target));
  event.resourceUses.push_back(makeUse(tokenDomain, source, target));
  passed &= check(universe.addMechanism(event).error ==
                      SyncCoverMechanismError::InvalidResourceUse,
                  "event bundle must consume an event-ID resource");

  SyncCoverMechanismDescriptor ownership;
  ownership.kind = SyncCoverMechanismKind::OwnershipProtocol;
  ownership.supplyEdges.push_back(completionEdge(source, target));
  ownership.resourceUses.push_back(makeUse(tokenDomain, source, target));
  passed &=
      check(universe.addVerifiedProtocol(ownership,
                                         [](const auto &) { return true; }),
            "verified ownership protocol may consume a buffer-token resource");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testAtomicSupplyAndCoverage();
  passed &= testAtomicFailureAndProtocolGate();
  passed &= testMultiEdgeAtomicSupply();
  passed &= testDomainsBarriersAndConflicts();
  passed &= testInvalidResourceUses();
  passed &= testPlacementBindingAndVerifierRejections();
  passed &= testBarrierExecutionGuarantees();
  passed &= testMechanismResourceComposition();
  return passed ? 0 : 1;
}
