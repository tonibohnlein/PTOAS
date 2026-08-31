// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageRectangleGrounding.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageRectangleGroundingTest failure: " << message
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

SyncCoverDemand makeRawDemand(SyncCoverNodeId source, SyncCoverNodeId target,
                              SyncCoverStorageWitnessId witness,
                              SyncCoverScopeId scope,
                              std::size_t distance = 0) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.distance = distance;
  demand.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  demand.storageWitnesses = {witness};
  return demand;
}

bool testGroundFactoredRectangles() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true), passed,
      "add grounding loop");
  constexpr unsigned guardLiteralCount = 32;
  SyncCoverGuard guard;
  for (unsigned index = 0; index < guardLiteralCount; ++index) {
    const SyncCoverControlId control = takeIndex(
        graph.addControl(2, loop), passed, "add grounding guard control");
    guard.literals.push_back({control, 0});
  }
  passed &= check(graph.setResourceRecurrenceCarryKind(
                      2, SyncCoverEdgeKind::CompletionPreservingIssueOrder),
                  "set grounding target recurrence carry");
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, loop, 0, guard, {2}), passed,
                "add first grounding source");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(1, 1, loop, 1, guard, {2}, std::nullopt, true),
                passed,
                "add second grounding source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, loop, 2, guard), passed,
                "add first grounding target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, loop, 3, guard), passed,
                "add second grounding target");
  const SyncCoverNodeId baselineSource =
      takeIndex(graph.addNode(3, 1, loop, 4, {}, {4}), passed,
                "add fixed baseline source");
  const SyncCoverNodeId baselineTarget =
      takeIndex(graph.addNode(4, 1, loop, 5), passed,
                "add fixed baseline target");
  passed &= check(
      graph.addEdge({firstSource, secondSource,
                     SyncCoverEdgeKind::CertifiedCompletionFrontier, loop, 0,
                     guard, guard}),
      "add source completion frontier");
  passed &= check(graph.addCompletionDominance(firstSource, secondSource),
                  "certify the later source completion frontier");
  passed &= check(
      graph.addEdge({firstTarget, secondTarget,
                     SyncCoverEdgeKind::NonCompletionPreservingIssueOrder,
                     loop, 0, guard, guard}),
      "add target issue order");
  passed &= check(
      graph.addEdge({baselineSource, baselineTarget,
                     SyncCoverEdgeKind::CompletionSupply, loop, 0, {}, {}}),
      "add fixed baseline completion");

  const auto addExactRaw = [&](SyncCoverNodeId source,
                               SyncCoverNodeId target,
                               SyncCoverStorageAccessFamilyId family,
                               std::string_view label) {
    const SyncCoverStorageDomainId domain =
        takeIndex(graph.addStorageDomain(), passed, label);
    const SyncCoverStorageAccessId write =
        takeIndex(graph.addStorageAccess(source, domain, family, {0, 64},
                                         SyncCoverStorageAccessMode::Write,
                                         std::nullopt, true),
                  passed, label);
    const SyncCoverStorageAccessId read =
        takeIndex(graph.addStorageAccess(target, domain, family, {0, 64},
                                         SyncCoverStorageAccessMode::Read,
                                         std::nullopt, true),
                  passed, label);
    const SyncCoverStorageWitnessId witness =
        takeIndex(graph.addStorageWitness(write, read), passed, label);
    passed &= check(
        graph.addDemand(makeRawDemand(source, target, witness, loop)), label);
    return witness;
  };
  const SyncCoverStorageWitnessId firstWitness =
      addExactRaw(firstSource, firstTarget, 11, "add first grounding RAW");
  passed &= check(graph.addDemand(makeRawDemand(
                      firstSource, firstTarget, firstWitness, loop, 1)),
                  "add grounding recurrence RAW");
  addExactRaw(secondSource, secondTarget, 12, "add second grounding RAW");
  addExactRaw(baselineSource, baselineTarget, 13,
              "add fixed baseline grounding RAW");
  passed &= check(graph.freezeStructure(), "freeze grounding graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageCutIndex cuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle);
  const SyncCoverStorageFactoredRectangleIndex rectangles =
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts);
  const std::vector<SyncCoverDemandId> demands{0, 1, 2, 3};
  const SyncCoverExpandedProgram expansion(graph, demands);
  const SyncCoverSyntheticRectangleGrounding grounding =
      groundSyncCoverSyntheticStorageRectangles(graph, expansion, cuts,
                                                rectangles, demands);
  const bool completeInput =
      lifecycle.isComplete() && cuts.isComplete() && rectangles.isComplete() &&
      grounding.isComplete();
  if (!check(completeInput, "build and ground factored rectangles")) {
    return false;
  }
  const SyncCoverSyntheticRectangleGroundingStatistics &statistics =
      grounding.getStatistics();
  passed &= check(
      statistics.evaluatedSyntheticRectangles == 2 &&
          statistics.syntheticRectanglesWithCoverage == 2 &&
          statistics.syntheticRectanglesCoveringMultipleRows == 1 &&
          statistics.maximumCoverageRows == 3 &&
          statistics.totalCoverageRows == 4 && !statistics.truncated,
      "ground recurrence coverage after subtracting fixed baseline coverage");
  passed &= check(
      grounding.getDetails().size() == 2 &&
          grounding.getDetails()[0].coverageRows == 3,
      "rank the useful synthetic rectangle first");

  SyncCoverSyntheticRectangleGroundingLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumDetails = grounding.getDetails().size();
  passed &= check(groundSyncCoverSyntheticStorageRectangles(
                       graph, expansion, cuts, rectangles, demands, exact)
                       .isComplete(),
                  "accept the exact grounding work bound");
  --exact.maximumWorkUnits;
  const SyncCoverSyntheticRectangleGrounding truncated =
      groundSyncCoverSyntheticStorageRectangles(
          graph, expansion, cuts, rectangles, demands, exact);
  passed &= check(
      truncated.getError() ==
              SyncCoverSyntheticRectangleGroundingError::WorkLimitExceeded &&
          truncated.getStatistics().truncated,
      "truncate streamed grounding at one-less work");
  return passed;
}

} // namespace

int main() { return testGroundFactoredRectangles() ? 0 : 1; }
