// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageCuts.h"

#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "SyncCoverStorageCutsTest failure: " << message << '\n';
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

bool testDirectStorageCuts() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
                passed, "add cut loop");
  constexpr unsigned guardLiteralCount = 128;
  SyncCoverGuard guard;
  for (unsigned index = 0; index < guardLiteralCount; ++index) {
    const SyncCoverControlId control =
        takeIndex(graph.addControl(2, loop), passed, "add cut guard");
    guard.literals.push_back({control, 0});
  }
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, guard, {2}), passed,
                "add completion source");
  const SyncCoverNodeId firstTarget =
      takeIndex(graph.addNode(2, 1, loop, 1, guard), passed,
                "add first acquisition target");
  const SyncCoverNodeId secondTarget =
      takeIndex(graph.addNode(2, 1, loop, 2, guard), passed,
                "add second acquisition target");

  const auto addExactRaw = [&](SyncCoverStorageAccessFamilyId family,
                               SyncCoverNodeId target,
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
    passed &= check(graph.addDemand(makeRawDemand(source, target, witness,
                                                  loop)),
                    label);
  };
  addExactRaw(11, firstTarget, "add first exact RAW");
  addExactRaw(12, firstTarget, "add storage-equivalent exact RAW");
  addExactRaw(13, secondTarget, "add source-factored exact RAW");
  passed &= check(graph.freezeStructure(), "freeze cut graph");
  if (!passed) {
    return false;
  }

  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  if (!check(lifecycle.isComplete(), "build cut lifecycle input")) {
    return false;
  }
  const SyncCoverStorageCutIndex cuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle);
  if (!check(cuts.isComplete(), "build direct storage cuts")) {
    return false;
  }
  const SyncCoverStorageCutStatistics &statistics = cuts.getStatistics();
  passed &= check(
      statistics.eligibleEdges == 3 && statistics.ineligibleEdges == 0 &&
          statistics.completionCuts == 1 &&
          statistics.acquisitionCuts == 2 && statistics.rectangles == 2 &&
          statistics.incidences == 9 &&
          statistics.guardLiterals == 3 * guardLiteralCount &&
          statistics.maximumRectangleEdges == 2,
      "report compact source, target, and rectangle structure");
  passed &= check(
      cuts.getCuts().size() == 3 && cuts.getCuts()[0].edges.size() == 3 &&
          cuts.getCuts()[1].edges.size() == 2 &&
          cuts.getCuts()[2].edges.size() == 1,
      "factor storage identities at identical direct cuts");
  passed &= check(cuts.getRectangles().size() == 2 &&
                      cuts.getRectangles()[0].edges.size() == 2 &&
                      cuts.getRectangles()[1].edges.size() == 1,
                  "retain compact rectangle edge incidences");

  SyncCoverStorageCutLimits exact;
  exact.maximumWorkUnits = statistics.workUnits;
  exact.maximumCuts = statistics.completionCuts + statistics.acquisitionCuts;
  exact.maximumRectangles = statistics.rectangles;
  exact.maximumIncidences = statistics.incidences;
  exact.maximumGuardLiterals = statistics.guardLiterals;
  passed &= check(buildSyncCoverStorageCutIndex(graph, lifecycle, exact)
                      .isComplete(),
                  "accept exact cut bounds");
  --exact.maximumRectangles;
  passed &= check(
      buildSyncCoverStorageCutIndex(graph, lifecycle, exact).getError() ==
          SyncCoverStorageCutError::LimitExceeded,
      "bound retained rectangles transactionally");
  exact.maximumRectangles = statistics.rectangles;
  --exact.maximumGuardLiterals;
  passed &= check(
      buildSyncCoverStorageCutIndex(graph, lifecycle, exact).getError() ==
          SyncCoverStorageCutError::LimitExceeded,
      "bound retained cut guards transactionally");
  exact.maximumGuardLiterals = statistics.guardLiterals;
  --exact.maximumWorkUnits;
  const SyncCoverStorageCutIndex workTruncated =
      buildSyncCoverStorageCutIndex(graph, lifecycle, exact);
  passed &= check(
      workTruncated.getError() == SyncCoverStorageCutError::LimitExceeded &&
          workTruncated.getStatistics().truncated &&
          workTruncated.getCuts().empty() &&
          workTruncated.getRectangles().empty(),
      "bound cut construction work transactionally");

  SyncCoverStorageLifecycleLimits incompleteLimits;
  incompleteLimits.maximumEdges = 2;
  const SyncCoverStorageLifecycleIndex incomplete =
      buildSyncCoverStorageLifecycleIndex(graph, incompleteLimits);
  passed &= check(
      buildSyncCoverStorageCutIndex(graph, incomplete).getError() ==
          SyncCoverStorageCutError::IncompleteLifecycleIndex,
      "reject a partial lifecycle input");
  return passed;
}

std::optional<std::size_t>
measureIneligibleGuardWork(unsigned guardLiteralCount) {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 32}, true),
                passed, "add ineligible loop");
  SyncCoverGuard guard;
  for (unsigned index = 0; index < guardLiteralCount; ++index) {
    const SyncCoverControlId control =
        takeIndex(graph.addControl(2, loop), passed,
                  "add ineligible guard");
    guard.literals.push_back({control, 0});
  }
  const SyncCoverNodeId source =
      takeIndex(graph.addNode(1, 1, loop, 0, guard), passed,
                "add ineligible source");
  const SyncCoverNodeId target =
      takeIndex(graph.addNode(1, 1, loop, 1, guard), passed,
                "add ineligible same-resource target");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add ineligible domain");
  const SyncCoverStorageAccessId write =
      takeIndex(graph.addStorageAccess(source, domain, 1, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add ineligible write");
  const SyncCoverStorageAccessId read =
      takeIndex(graph.addStorageAccess(target, domain, 1, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add ineligible read");
  const SyncCoverStorageWitnessId witness =
      takeIndex(graph.addStorageWitness(write, read), passed,
                "add ineligible witness");
  passed &= check(graph.addDemand(makeRawDemand(source, target, witness, loop)),
                  "add ineligible demand");
  passed &= check(graph.freezeStructure(), "freeze ineligible graph");
  if (!passed) {
    return std::nullopt;
  }
  const SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  const SyncCoverStorageCutIndex cuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle);
  const SyncCoverStorageCutStatistics &statistics = cuts.getStatistics();
  const bool validResult =
      lifecycle.isComplete() && cuts.isComplete() &&
      statistics.eligibleEdges == 0 && statistics.ineligibleEdges == 1 &&
      cuts.getCuts().empty() && cuts.getRectangles().empty();
  if (!check(validResult,
             "reject a guarded same-resource edge before guard work")) {
    return std::nullopt;
  }
  return statistics.workUnits;
}

bool testStructurallyIneligibleGuardWork() {
  const std::optional<std::size_t> shortGuardWork =
      measureIneligibleGuardWork(1);
  const std::optional<std::size_t> longGuardWork =
      measureIneligibleGuardWork(512);
  return check(shortGuardWork && longGuardWork &&
                   *shortGuardWork == *longGuardWork,
               "do not copy or scan guards for structurally ineligible edges");
}

} // namespace

int main() {
  const bool directCutsPassed = testDirectStorageCuts();
  const bool ineligibleGuardWorkPassed = testStructurallyIneligibleGuardWork();
  return directCutsPassed && ineligibleGuardWorkPassed ? 0 : 1;
}
