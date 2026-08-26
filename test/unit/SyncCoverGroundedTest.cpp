// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGrounded.h"

#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverGroundedTest failure: " << message << '\n';
  }
  return condition;
}

template <typename Result>
std::size_t takeIndex(const Result &result, bool &passed,
                      std::string_view message) {
  passed &= check(result && result.index.has_value(), message);
  return result.index.value_or(0);
}

SyncCoverEdge supply(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverEdge result;
  result.source = source;
  result.target = target;
  result.kind = SyncCoverEdgeKind::CompletionSupply;
  return result;
}

SyncCoverDemand demand(SyncCoverNodeId source, SyncCoverNodeId target) {
  SyncCoverDemand result;
  result.source = source;
  result.target = target;
  return result;
}

SyncCoverMechanismId addEvent(SyncCoverMechanismUniverse &universe,
                              SyncCoverResourceDomainId domain,
                              std::uint32_t sourceResource,
                              std::uint32_t targetResource,
                              SyncCoverNodeId source,
                              SyncCoverNodeId target, bool &passed) {
  SyncCoverMechanismDescriptor descriptor;
  descriptor.supplyEdges.push_back(supply(source, target));
  descriptor.actions.push_back(
      {SyncCoverResourceActionKind::Produce, sourceResource,
       {SyncCoverAnchorKind::AfterNode, source, 0, 0}});
  descriptor.actions.push_back(
      {SyncCoverResourceActionKind::Consume, targetResource,
       {SyncCoverAnchorKind::BeforeNode, target, 0, 0}});
  descriptor.resourceUses.push_back({domain, 0, 0, 1, {0, 1}, {0}});
  descriptor.supplyBindings.push_back({0, 0, 0, 1});
  return takeIndex(universe.addMechanism(descriptor), passed,
                   "add event mechanism");
}

SyncCoverMechanismId addBarrier(SyncCoverMechanismUniverse &universe,
                                std::uint32_t resource,
                                SyncCoverNodeId source,
                                SyncCoverNodeId target, bool &passed) {
  SyncCoverMechanismDescriptor descriptor;
  descriptor.kind = SyncCoverMechanismKind::Barrier;
  descriptor.barrier = SyncCoverBarrierPlacement{resource, target, 0};
  descriptor.supplyEdges.push_back(supply(source, target));
  return takeIndex(universe.addMechanism(descriptor), passed,
                   "add barrier mechanism");
}

bool testDemandSet() {
  bool passed = true;
  SyncCoverDemandSet first(130);
  SyncCoverDemandSet second(130);
  passed &= check(first.add(0) && first.add(64) && first.add(129),
                  "bitset accepts boundary indices");
  passed &= check(!first.add(130), "bitset rejects an out-of-range index");
  second.add(64);
  passed &= check(first.containsAll(second) && first.intersects(second) &&
                      first.count() == 3,
                  "bitset subset and intersection operations are exact");
  second.add(7);
  first.unite(second);
  passed &= check(first.contains(7) && first.count() == 4,
                  "bitset union preserves all demands");
  return passed;
}

bool testGroundedColumnsAndMetadata() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2, 3}), passed, "add source");
  const SyncCoverNodeId middle = takeIndex(
      graph.addNode(2, 1, 0, 1, {}, {3}), passed, "add middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(3, 1, 0, 2), passed, "add target");
  const SyncCoverDemandId longDemand = takeIndex(
      graph.addDemand(demand(source, target)), passed, "add long demand");
  const SyncCoverDemandId shortDemand = takeIndex(
      graph.addDemand(demand(source, middle)), passed, "add short demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId firstDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add first domain");
  const SyncCoverResourceDomainId secondDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 3, 8),
      passed, "add second domain");
  const SyncCoverResourceDomainId directDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 3, 8),
      passed, "add direct domain");
  const SyncCoverMechanismId first =
      addEvent(universe, firstDomain, 1, 2, source, middle, passed);
  const SyncCoverMechanismId second =
      addEvent(universe, secondDomain, 2, 3, middle, target, passed);
  const SyncCoverMechanismId direct =
      addEvent(universe, directDomain, 1, 3, source, target, passed);

  const SyncCoverGroundingResult grounded =
      groundSyncCoverInstance(universe, {longDemand, shortDemand});
  passed &= check(grounded && grounded.instance.columns.size() == 2 &&
                      grounded.instance.demandsNeedingPricing.empty(),
                  "grounding materializes factory-declared singleton columns");
  if (!grounded) {
    return false;
  }
  const SyncCoverGroundedInstance &instance = grounded.instance;
  passed &= check(instance.columns[0].members ==
                          std::vector<SyncCoverMechanismId>{first} &&
                      instance.columns[0].coverage.contains(1) &&
                      instance.columns[1].members ==
                          std::vector<SyncCoverMechanismId>{direct} &&
                      instance.columns[1].coverage.contains(0),
                  "columns are deterministic and retain exact demand sets");
  passed &= check(!instance.coversAll({first, second}) &&
                      instance.coveredBy({direct}).count() == 1,
                  "grounding does not infer undeclared transitive coverage");
  passed &= check(instance.mechanisms.size() == 3 &&
                      instance.mechanisms[first].resourceUses.size() == 1 &&
                      instance.resourceDomains.size() == 3,
                  "resource lifetimes and domains are attached once");
  passed &= check(grounded.statistics.groundingQueries == 2 &&
                      grounded.statistics.coverageQueries == 0,
                  "singleton incidence is built in one batched grounding "
                  "traversal without selected-plan queries");
  passed &= check(instance.isCurrent(universe),
                  "fresh instance matches its universe epoch");

  SyncCoverGroundingOptions boundedOptions;
  boundedOptions.maximumColumns = 1;
  const SyncCoverGroundingResult bounded = groundSyncCoverInstance(
      universe, {longDemand, shortDemand}, boundedOptions);
  passed &= check(bounded && bounded.instance.columnsTruncated &&
                      !bounded.instance.demandsNeedingPricing.empty(),
                  "column caps produce an explicit pricing requirement");

  const SyncCoverGroundingResult seeded = groundSyncCoverInstance(
      universe, {longDemand, shortDemand},
      {{{first, second}, {longDemand}}});
  const bool hasSeedColumn = seeded && std::any_of(
      seeded.instance.columns.begin(), seeded.instance.columns.end(),
      [&](const SyncCoverGroundedColumn &column) {
        return column.members ==
               std::vector<SyncCoverMechanismId>{first, second};
      });
  passed &= check(seeded && seeded.instance.coversAll({first, second}) &&
                      hasSeedColumn,
                  "verified columns attach declared transitive coverage");
  passed &= check(seeded.statistics.groundingQueries == 3 &&
                      seeded.statistics.coverageQueries == 0,
                  "shared-cost columns use batched grounding verification");

  const SyncCoverGroundingResult overclaimed = groundSyncCoverInstance(
      universe, {longDemand, shortDemand}, {{{first}, {longDemand}}});
  passed &= check(overclaimed && !overclaimed.instance.coversAll({first}) &&
                      overclaimed.instance.coveredBy({first}).count() == 1,
                  "factory declarations cannot overclaim oracle coverage");

  addEvent(universe, directDomain, 1, 3, source, target, passed);
  passed &= check(!instance.isCurrent(universe),
                  "universe growth invalidates grounded columns");
  const SyncCoverGroundingResult duplicateDemand =
      groundSyncCoverInstance(universe, {longDemand, longDemand});
  passed &= check(duplicateDemand.error ==
                      SyncCoverGroundingError::InvalidDemand,
                  "noncanonical active demands fail closed");
  return passed;
}

bool testMissingFactoryCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2, 3, 4}), passed, "add source");
  const SyncCoverNodeId firstMiddle = takeIndex(
      graph.addNode(2, 1, 0, 1, {}, {3, 4}), passed, "add first middle");
  const SyncCoverNodeId secondMiddle = takeIndex(
      graph.addNode(3, 1, 0, 2, {}, {4}), passed, "add second middle");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(4, 1, 0, 3), passed, "add target");
  const SyncCoverDemandId pricedDemand = takeIndex(
      graph.addDemand(demand(source, target)), passed, "add priced demand");
  const SyncCoverNodeId unreachableTarget =
      takeIndex(graph.addNode(5, 1, 0, 4), passed, "add unreachable target");
  const SyncCoverDemandId unreachableDemand =
      takeIndex(graph.addDemand(demand(source, unreachableTarget)), passed,
                "add unreachable demand");

  SyncCoverMechanismUniverse universe(graph);
  const auto firstDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add first domain");
  const auto secondDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 3, 8),
      passed, "add second domain");
  const auto thirdDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 3, 4, 8),
      passed, "add third domain");
  const SyncCoverMechanismId first = addEvent(
      universe, firstDomain, 1, 2, source, firstMiddle, passed);
  const SyncCoverMechanismId second = addEvent(
      universe, secondDomain, 2, 3, firstMiddle, secondMiddle, passed);
  const SyncCoverMechanismId third = addEvent(
      universe, thirdDomain, 3, 4, secondMiddle, target, passed);

  const SyncCoverGroundingResult grounded = groundSyncCoverInstance(
      universe, {pricedDemand, unreachableDemand});
  passed &= check(
      grounded &&
          grounded.instance.demandsNeedingPricing ==
              std::vector<SyncCoverDemandId>{pricedDemand,
                                             unreachableDemand},
      "missing factories are reported without claiming infeasibility");
  if (grounded) {
    passed &= check(
        !grounded.instance.coveredBy({first, second, third}).contains(
            pricedDemand) &&
            grounded.instance.demandColumns[pricedDemand].empty(),
        "selection never invents undeclared transitive coverage");
    passed &= check(grounded.statistics.coverageQueries == 0 &&
                        grounded.statistics.groundingQueries == 2,
                    "grounding queries each demand once and prices nothing");
    passed &= check(grounded.instance.pricingRestricted,
                    "the witness universe reports itself as restricted");
  }
  return passed;
}

bool testPricesBarrierOnlyCoverage() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverNodeId source = takeIndex(
      graph.addNode(1, 1, 0, 0, {}, {2}), passed, "add cycle source");
  const SyncCoverNodeId middle = takeIndex(
      graph.addNode(2, 1, 0, 1, {}, {1}), passed, "add cycle middle");
  const SyncCoverNodeId target = takeIndex(
      graph.addNode(1, 1, 0, 2), passed, "add cycle target");
  const SyncCoverDemandId demandId = takeIndex(
      graph.addDemand(demand(source, target)), passed, "add cycle demand");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverResourceDomainId forwardDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 1, 2, 8),
      passed, "add forward domain");
  const SyncCoverResourceDomainId reverseDomain = takeIndex(
      universe.addResourceDomain(SyncCoverResourceKind::EventId, 2, 1, 8),
      passed, "add reverse domain");
  const SyncCoverMechanismId barrier =
      addBarrier(universe, 1, source, target, passed);
  const SyncCoverMechanismId forward =
      addEvent(universe, forwardDomain, 1, 2, source, middle, passed);
  const SyncCoverMechanismId reverse =
      addEvent(universe, reverseDomain, 2, 1, middle, target, passed);

  const SyncCoverGroundingResult grounded =
      groundSyncCoverInstance(universe, {demandId});
  const std::vector<SyncCoverMechanismId> eventPair{forward, reverse};
  const bool hasPair = grounded && std::any_of(
      grounded.instance.columns.begin(), grounded.instance.columns.end(),
      [&](const SyncCoverGroundedColumn &column) {
        return column.members == eventPair && column.coverage.contains(0);
      });
  passed &= check(grounded && !hasPair &&
                      grounded.instance.coversAll({barrier}) &&
                      grounded.instance.pricingRestricted,
                  "grounding declares no composite columns; the solver "
                  "prices barrier-covered demands lazily");
  return passed;
}

} // namespace

int main() {
  return testDemandSet() && testGroundedColumnsAndMetadata() &&
                 testMissingFactoryCoverage() &&
                 testPricesBarrierOnlyCoverage()
             ? 0
             : 1;
}
