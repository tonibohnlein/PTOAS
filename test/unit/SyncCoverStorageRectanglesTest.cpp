// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageRectangles.h"

#include <iostream>
#include <optional>
#include <string_view>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageRectanglesTest failure: " << message << '\n';
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
                              SyncCoverScopeId scope) {
  SyncCoverDemand demand;
  demand.source = source;
  demand.target = target;
  demand.scope = scope;
  demand.provenanceKinds = {SyncCoverDemandKind::MemoryRAW};
  demand.storageWitnesses = {witness};
  return demand;
}

bool testFactoredRectangles() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true), passed,
      "add rectangle loop");
  constexpr unsigned guardLiteralCount = 32;
  SyncCoverGuard guard;
  for (unsigned index = 0; index < guardLiteralCount; ++index) {
    const SyncCoverControlId control = takeIndex(
        graph.addControl(2, loop), passed, "add rectangle guard control");
    guard.literals.push_back({control, 0});
  }
  const SyncCoverNodeId firstSource =
      takeIndex(graph.addNode(1, 1, loop, 0, guard, {2}), passed,
                "add first rectangle source");
  const SyncCoverNodeId secondSource =
      takeIndex(graph.addNode(1, 1, loop, 1, guard, {2}), passed,
                "add second rectangle source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, loop, 2, guard), passed,
                "add first rectangle target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, loop, 3, guard), passed,
                "add second rectangle target");

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
  };
  addExactRaw(firstSource, firstTarget, 11, "add first diagonal RAW");
  addExactRaw(secondSource, secondTarget, 12, "add second diagonal RAW");
  passed &= check(graph.freezeStructure(), "freeze rectangle graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageCutIndex cuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle);
  const SyncCoverStorageFactoredRectangleIndex rectangles =
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts);
  const bool completeInput =
      lifecycle.isComplete() && cuts.isComplete() && rectangles.isComplete();
  if (!check(completeInput, "build factored rectangle input and index")) {
    return false;
  }
  const SyncCoverStorageFactoredRectangleStatistics &statistics =
      rectangles.getStatistics();
  passed &= check(statistics.inspections == 8 && statistics.rectangles == 4 &&
                      statistics.directRectangles == 2 &&
                      statistics.syntheticRectangles == 2 &&
                      statistics.guardLiterals == 4 * guardLiteralCount,
                  "enumerate direct and synthetic schedule rectangles");
  const auto &descriptions = rectangles.getRectangles();
  passed &= check(descriptions.size() == 4 &&
                      descriptions[0].directRectangle.has_value() &&
                      !descriptions[1].directRectangle.has_value() &&
                      !descriptions[2].directRectangle.has_value() &&
                      descriptions[3].directRectangle.has_value(),
                  "retain exact-rectangle derivation provenance");

  SyncCoverStorageFactoredRectangleLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumInspections = statistics.inspections;
  exact.maximumRectangles = statistics.rectangles;
  exact.maximumGuardLiterals = statistics.guardLiterals;
  passed &= check(buildSyncCoverStorageFactoredRectangleIndex(graph, cuts,
                                                               exact)
                      .isComplete(),
                  "accept exact factored-rectangle bounds");
  --exact.maximumWorkUnits;
  passed &= check(
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts, exact)
              .getError() ==
          SyncCoverStorageFactoredRectangleError::LimitExceeded,
      "bound factored-rectangle work transactionally");
  exact.maximumWorkUnits = statistics.workUnits;
  --exact.maximumInspections;
  passed &= check(
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts, exact)
              .getError() ==
          SyncCoverStorageFactoredRectangleError::LimitExceeded,
      "bound factored-rectangle inspections transactionally");
  exact.maximumInspections = statistics.inspections;
  --exact.maximumRectangles;
  passed &= check(
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts, exact)
              .getError() ==
          SyncCoverStorageFactoredRectangleError::LimitExceeded,
      "bound factored rectangles transactionally");
  exact.maximumRectangles = statistics.rectangles;
  --exact.maximumGuardLiterals;
  const SyncCoverStorageFactoredRectangleIndex guardTruncated =
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts, exact);
  passed &= check(
      guardTruncated.getError() ==
              SyncCoverStorageFactoredRectangleError::LimitExceeded &&
          guardTruncated.getStatistics().truncated &&
          guardTruncated.getRectangles().empty(),
      "bound retained rectangle guards transactionally");

  SyncCoverStorageCutLimits incompleteLimits;
  incompleteLimits.maximumRectangles = 1;
  const SyncCoverStorageCutIndex incompleteCuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle, incompleteLimits);
  passed &= check(
      buildSyncCoverStorageFactoredRectangleIndex(graph, incompleteCuts)
              .getError() ==
          SyncCoverStorageFactoredRectangleError::IncompleteCutIndex,
      "reject an incomplete direct-cut index");
  return passed;
}

} // namespace

int main() { return testFactoredRectangles() ? 0 : 1; }
