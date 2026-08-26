// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverColumnGeneration.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"

#include <iostream>
#include <string_view>

using namespace mlir::pto;

namespace {

constexpr std::uint32_t kPipeS = 0;
constexpr std::uint32_t kPipeV = 1;
constexpr std::uint32_t kPipeM = 2;
constexpr std::uint32_t kPipeMTE2 = 4;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverColumnGenerationTest failure: " << message << '\n';
  }
  return condition;
}

SyncCoverTargetCapabilities makeA2A3Capabilities(bool trustPrefix = false) {
  SyncCoverTargetCapabilities target;
  target.name = "a2a3";
  target.hardwareCompletionResources.insert(kPipeS);
  if (trustPrefix) {
    target.prefixSetResources = {kPipeV, kPipeM, kPipeMTE2};
    target.prefixEvidence = SyncCoverEvidenceLevel::Measured;
  }
  return target;
}

const SyncCoverColumnGeneratorReport *
findReport(const SyncCoverColumnGenerationResult &result,
           std::string_view name) {
  for (const SyncCoverColumnGeneratorReport &report : result.reports) {
    if (report.generator == name) {
      return &report;
    }
  }
  return nullptr;
}

struct Scenario {
  SyncCoverGraph graph;
  std::vector<SyncCoverDemandId> demands;
  bool ok = false;
};

Scenario buildForwardScenario() {
  Scenario scenario;
  std::vector<SyncCoverNodeId> loads;
  std::vector<SyncCoverNodeId> consumers;
  for (std::size_t index = 0; index < 4; ++index) {
    auto node = scenario.graph.addNode(kPipeMTE2, 1, 0, index, {},
                                       {kPipeM});
    if (!node || !node.index) {
      return scenario;
    }
    loads.push_back(*node.index);
  }
  for (std::size_t index = 0; index < 2; ++index) {
    auto node = scenario.graph.addNode(kPipeM, 1, 0, 4 + index, {},
                                       {kPipeMTE2});
    if (!node || !node.index) {
      return scenario;
    }
    consumers.push_back(*node.index);
  }
  for (std::size_t index = 0; index + 1 < loads.size(); ++index) {
    if (!scenario.graph.addEdge(
            {loads[index], loads[index + 1],
             SyncCoverEdgeKind::CompletionPreservingIssueOrder, 0, 0})) {
      return scenario;
    }
  }
  if (!scenario.graph.addEdge(
          {consumers[0], consumers[1],
           SyncCoverEdgeKind::CompletionPreservingIssueOrder, 0, 0})) {
    return scenario;
  }
  auto storage = scenario.graph.addStorageDomain();
  if (!storage || !storage.index) {
    return scenario;
  }
  std::vector<SyncCoverStorageAccessId> writes;
  for (std::size_t index = 0; index < loads.size(); ++index) {
    auto access = scenario.graph.addStorageAccess(
        loads[index], *storage.index, index,
        {index * 128, (index + 1) * 128},
        SyncCoverStorageAccessMode::Write);
    if (!access || !access.index) {
      return scenario;
    }
    writes.push_back(*access.index);
  }
  std::size_t family = writes.size();
  auto addDemand = [&](std::size_t load, SyncCoverNodeId consumer) {
    auto read = scenario.graph.addStorageAccess(
        consumer, *storage.index, family++, {load * 128, (load + 1) * 128},
        SyncCoverStorageAccessMode::Read);
    if (!read || !read.index) {
      return false;
    }
    auto witness =
        scenario.graph.addStorageWitness(writes[load], *read.index);
    if (!witness || !witness.index) {
      return false;
    }
    SyncCoverDemand demand;
    demand.source = loads[load];
    demand.target = consumer;
    demand.kind = SyncCoverDemandKind::MemoryRAW;
    demand.storageProvenance = SyncCoverStorageProvenance::Complete;
    demand.storageWitnesses.push_back(*witness.index);
    auto added = scenario.graph.addDemand(demand);
    if (!added || !added.index) {
      return false;
    }
    scenario.demands.push_back(*added.index);
    return true;
  };
  for (std::size_t index = 0; index < loads.size(); ++index) {
    if (!addDemand(index, consumers[0])) {
      return scenario;
    }
  }
  if (!addDemand(2, consumers[1]) || !addDemand(3, consumers[1])) {
    return scenario;
  }
  scenario.ok = static_cast<bool>(scenario.graph.freezeStructure());
  return scenario;
}

Scenario buildSamePipeScenario(std::uint32_t pipe) {
  Scenario scenario;
  std::vector<SyncCoverNodeId> nodes;
  for (std::size_t index = 0; index < 4; ++index) {
    auto node = scenario.graph.addNode(pipe, 1, 0, index);
    if (!node || !node.index) {
      return scenario;
    }
    nodes.push_back(*node.index);
  }
  auto storage = scenario.graph.addStorageDomain();
  if (!storage || !storage.index) {
    return scenario;
  }
  std::vector<SyncCoverStorageAccessId> accesses;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    auto access = scenario.graph.addStorageAccess(
        nodes[index], *storage.index, index, {0, 128},
        SyncCoverStorageAccessMode::Write);
    if (!access || !access.index) {
      return scenario;
    }
    accesses.push_back(*access.index);
  }
  for (const auto endpoints :
       {std::pair<std::size_t, std::size_t>{0, 2}, {1, 3}}) {
    auto witness = scenario.graph.addStorageWitness(
        accesses[endpoints.first], accesses[endpoints.second]);
    if (!witness || !witness.index) {
      return scenario;
    }
    SyncCoverDemand demand;
    demand.source = nodes[endpoints.first];
    demand.target = nodes[endpoints.second];
    demand.kind = SyncCoverDemandKind::MemoryWAW;
    demand.storageProvenance = SyncCoverStorageProvenance::Complete;
    demand.storageWitnesses.push_back(*witness.index);
    auto added = scenario.graph.addDemand(demand);
    if (!added || !added.index) {
      return scenario;
    }
    scenario.demands.push_back(*added.index);
  }
  scenario.ok = static_cast<bool>(scenario.graph.freezeStructure());
  return scenario;
}

bool testCapabilitiesAndMergedEvents() {
  bool passed = true;
  const SyncCoverTargetCapabilities conservative = makeA2A3Capabilities();
  const SyncCoverTargetCapabilities measured = makeA2A3Capabilities(true);
  passed &= check(!conservative.hasPrefixSetSemantics(kPipeMTE2),
                  "unproven A2/A3 prefix semantics stay disabled");
  passed &= check(measured.hasPrefixSetSemantics(kPipeMTE2),
                  "measured prefix semantics may be enabled");

  for (bool enableMerged : {false, true}) {
    Scenario scenario = buildForwardScenario();
    passed &= check(scenario.ok, "forward scenario builds");
    SyncCoverMechanismUniverse universe(scenario.graph);
    passed &= check(static_cast<bool>(universe.addResourceDomain(
                        SyncCoverResourceKind::EventId, kPipeMTE2, kPipeM, 1)),
                    "forward event domain registers");
    SyncCoverColumnGenerationContext context{measured, scenario.demands};
    std::vector<std::unique_ptr<SyncCoverColumnGenerator>> generators;
    generators.push_back(makeSyncCoverCanonicalEventGenerator());
    if (enableMerged) {
      generators.push_back(makeSyncCoverMergedPrefixEventGenerator());
    }
    const auto generated =
        runSyncCoverColumnGenerators(context, universe, generators);
    const SyncCoverSelectionResult selection =
        solveSyncCoverSelection(universe, scenario.demands);
    if (enableMerged) {
      const auto *report = findReport(generated, "merged-prefix");
      passed &= check(report && report->admitted != 0,
                      "merged candidates are admitted when proven");
      passed &= check(selection && selection.mechanisms.size() == 1,
                      "one merged event solves the budget-one fan-in");
    } else {
      passed &= check(!selection,
                      "canonical events alone exceed the budget-one domain");
    }
  }

  Scenario conservativeScenario = buildForwardScenario();
  SyncCoverMechanismUniverse conservativeUniverse(conservativeScenario.graph);
  conservativeUniverse.addResourceDomain(SyncCoverResourceKind::EventId,
                                         kPipeMTE2, kPipeM, 8);
  SyncCoverColumnGenerationContext conservativeContext{
      conservative, conservativeScenario.demands};
  std::vector<std::unique_ptr<SyncCoverColumnGenerator>> generators;
  generators.push_back(makeSyncCoverMergedPrefixEventGenerator());
  const auto generated = runSyncCoverColumnGenerators(
      conservativeContext, conservativeUniverse, generators);
  const auto *report = findReport(generated, "merged-prefix");
  passed &= check(report && report->admitted == 0 &&
                      report->skippedByCapability != 0,
                  "merged events fail closed without prefix evidence");

  const auto verifyMergedShape = [&](std::size_t setOrder,
                                     std::size_t waitOrder,
                                     const SyncCoverTargetCapabilities &target) {
    SyncCoverGraph graph;
    const SyncCoverGraphResult earlySource =
        graph.addNode(kPipeMTE2, 1, 0, 0, {}, {kPipeM});
    const SyncCoverGraphResult set =
        graph.addNode(kPipeMTE2, 1, 0, setOrder, {}, {kPipeM});
    const SyncCoverGraphResult wait =
        graph.addNode(kPipeM, 1, 0, waitOrder);
    const SyncCoverGraphResult lateTarget =
        graph.addNode(kPipeM, 1, 0, 3);
    if (!earlySource || !earlySource.index || !set || !set.index || !wait ||
        !wait.index || !lateTarget || !lateTarget.index ||
        !graph.freezeStructure()) {
      return false;
    }
    SyncCoverResourceDomain domain;
    domain.kind = SyncCoverResourceKind::EventId;
    domain.sourceResource = kPipeMTE2;
    domain.targetResource = kPipeM;
    domain.budget = 8;
    SyncCoverMechanismDescriptor descriptor;
    descriptor.kind = SyncCoverMechanismKind::VerifiedProtocol;
    descriptor.actions = {
        {SyncCoverResourceActionKind::Produce, kPipeMTE2,
         {SyncCoverAnchorKind::AfterNode, *set.index, 0}},
        {SyncCoverResourceActionKind::Consume, kPipeM,
         {SyncCoverAnchorKind::BeforeNode, *wait.index, 0}}};
    descriptor.supplyEdges = {
        {*earlySource.index, *wait.index,
         SyncCoverEdgeKind::CompletionSupply, 0, 0},
        {*set.index, *lateTarget.index,
         SyncCoverEdgeKind::CompletionSupply, 0, 0}};
    descriptor.resourceUses = {{0, 0, 0, 1, {0, 1}, {0, 1}}};
    descriptor.supplyBindings = {{0, 0, 0, 1}, {1, 0, 0, 1}};
    return verifySyncCoverMergedPrefixEvent(graph, domain, descriptor, target);
  };
  passed &= check(verifyMergedShape(1, 2, measured),
                  "merged verifier accepts a forward trusted protocol");
  passed &= check(!verifyMergedShape(2, 1, measured),
                  "merged verifier rejects a wait ordered before its set");
  passed &= check(!verifyMergedShape(1, 2, conservative),
                  "merged verifier rejects an untrusted prefix protocol");

  const auto verifyMergedSupplies =
      [&](bool dominatedMembers, bool anchorIsMember) {
        SyncCoverGraph graph;
        const SyncCoverGraphResult earlySource =
            graph.addNode(kPipeMTE2, 1, 0, 0, {}, {kPipeM});
        const SyncCoverGraphResult set =
            graph.addNode(kPipeMTE2, 1, 0, 1, {}, {kPipeM});
        const SyncCoverGraphResult lateSource =
            graph.addNode(kPipeMTE2, 1, 0, 2, {}, {kPipeM});
        const SyncCoverGraphResult wait = graph.addNode(kPipeM, 1, 0, 3);
        const SyncCoverGraphResult lateTarget = graph.addNode(kPipeM, 1, 0, 4);
        if (!earlySource.index || !set.index || !lateSource.index ||
            !wait.index || !lateTarget.index || !graph.freezeStructure()) {
          return false;
        }
        SyncCoverResourceDomain domain;
        domain.kind = SyncCoverResourceKind::EventId;
        domain.sourceResource = kPipeMTE2;
        domain.targetResource = kPipeM;
        domain.budget = 8;
        SyncCoverMechanismDescriptor descriptor;
        descriptor.kind = SyncCoverMechanismKind::VerifiedProtocol;
        descriptor.actions = {
            {SyncCoverResourceActionKind::Produce, kPipeMTE2,
             {SyncCoverAnchorKind::AfterNode, *set.index, 0}},
            {SyncCoverResourceActionKind::Consume, kPipeM,
             {SyncCoverAnchorKind::BeforeNode, *wait.index, 0}}};
        const std::size_t firstSource =
            anchorIsMember ? *set.index : *earlySource.index;
        const std::size_t secondSource =
            dominatedMembers ? *earlySource.index : *lateSource.index;
        descriptor.supplyEdges = {
            {firstSource, *wait.index, SyncCoverEdgeKind::CompletionSupply, 0,
             0},
            {secondSource, *lateTarget.index,
             SyncCoverEdgeKind::CompletionSupply, 0, 0}};
        descriptor.resourceUses = {{0, 0, 0, 1, {0, 1}, {0, 1}}};
        descriptor.supplyBindings = {{0, 0, 0, 1}, {1, 0, 0, 1}};
        return verifySyncCoverMergedPrefixEvent(graph, domain, descriptor,
                                                measured);
      };
  passed &= check(verifyMergedSupplies(true, true),
                  "merged verifier accepts dominated anchored members");
  passed &= check(!verifyMergedSupplies(false, true),
                  "merged verifier rejects a member set after the anchor");
  passed &= check(!verifyMergedSupplies(true, false),
                  "merged verifier rejects a set anchor outside the members");
  return passed;
}

bool testPiercingAndDeterminism() {
  bool passed = true;
  Scenario samePipe = buildSamePipeScenario(kPipeMTE2);
  passed &= check(samePipe.ok, "same-pipe scenario builds");
  SyncCoverMechanismUniverse universe(samePipe.graph);
  const SyncCoverTargetCapabilities target = makeA2A3Capabilities();
  SyncCoverColumnGenerationContext context{target, samePipe.demands};
  std::vector<std::unique_ptr<SyncCoverColumnGenerator>> generators;
  generators.push_back(makeSyncCoverPiercedBarrierGenerator());
  const auto generated =
      runSyncCoverColumnGenerators(context, universe, generators);
  const auto *report = findReport(generated, "pierce-barrier");
  passed &= check(report && report->admitted == 1,
                  "one piercing barrier covers overlapping intervals");
  const SyncCoverSelectionResult selection =
      solveSyncCoverSelection(universe, samePipe.demands);
  passed &= check(selection && selection.mechanisms.size() == 1,
                  "the pierced barrier is selected");

  const auto runShape = [] {
    Scenario scenario = buildForwardScenario();
    SyncCoverMechanismUniverse localUniverse(scenario.graph);
    localUniverse.addResourceDomain(SyncCoverResourceKind::EventId,
                                    kPipeMTE2, kPipeM, 8);
    const SyncCoverTargetCapabilities localTarget = makeA2A3Capabilities(true);
    SyncCoverColumnGenerationContext localContext{localTarget,
                                                  scenario.demands};
    std::vector<std::unique_ptr<SyncCoverColumnGenerator>> localGenerators;
    localGenerators.push_back(makeSyncCoverCanonicalEventGenerator());
    localGenerators.push_back(makeSyncCoverMergedPrefixEventGenerator());
    localGenerators.push_back(makeSyncCoverPiercedBarrierGenerator());
    const auto result = runSyncCoverColumnGenerators(
        localContext, localUniverse, localGenerators);
    std::vector<std::size_t> shape;
    for (const SyncCoverColumnGeneratorReport &entry : result.reports) {
      shape.push_back(entry.candidates);
      shape.push_back(entry.admitted);
      shape.push_back(entry.skippedByCapability);
    }
    return shape;
  };
  passed &= check(runShape() == runShape(),
                  "the generator pipeline is deterministic");
  return passed;
}

} // namespace

int main() {
  bool passed = true;
  passed &= testCapabilitiesAndMergedEvents();
  passed &= testPiercingAndDeterminism();
  std::cout << (passed ? "SyncCoverColumnGenerationTest PASS"
                       : "SyncCoverColumnGenerationTest FAIL")
            << std::endl;
  return passed ? 0 : 1;
}
