// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotLifecycle.h"

#include <iostream>
#include <optional>
#include <string>

using namespace mlir::pto;

namespace {

bool check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "SyncCoverSlotLifecycleTest failure: " << message << '\n';
  }
  return condition;
}

bool check(const SyncCoverGraphResult &result, const std::string &message) {
  return check(static_cast<bool>(result), message);
}

template <typename T>
std::size_t takeIndex(const T &result, bool &passed, const char *message) {
  passed &= check(result && result.index.has_value(), message);
  if (!result) {
    std::cerr << "  graph error=" << static_cast<unsigned>(result.error)
              << '\n';
  }
  return result.index.value_or(0);
}

SyncCoverDemand demand(SyncCoverNodeId source, SyncCoverNodeId target,
                       SyncCoverDemandKind kind, SyncCoverScopeId scope,
                       unsigned distance,
                       SyncCoverStorageWitnessId witness) {
  SyncCoverDemand result;
  result.source = source;
  result.target = target;
  result.kind = kind;
  result.scope = scope;
  result.distance = distance;
  result.storageProvenance = SyncCoverStorageProvenance::Complete;
  result.storageWitnesses = {witness};
  return result;
}

bool testExactRoundTripDiscovery() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 80}, true), passed,
      "add recurrence loop");
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, 10), passed, "add producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, 20), passed, "add consumer");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add storage domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(producer, domain, 10, {0, 64},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "add producer write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(consumer, domain, 11, {0, 64},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add consumer read");
  const SyncCoverStorageWitnessId readyWitness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add ready witness");
  const SyncCoverStorageWitnessId releaseWitness = takeIndex(
      graph.addStorageWitness(read, write), passed, "add release witness");
  const SyncCoverNodeId external =
      takeIndex(graph.addNode(1, 1, loop, 30), passed, "add external writer");
  const SyncCoverStorageAccessId externalWrite = takeIndex(
      graph.addStorageAccess(external, domain, 12, {0, 64},
                             SyncCoverStorageAccessMode::Write, 0),
      passed, "add unmodeled slot writer");
  passed &= check(graph.addDemand(demand(producer, consumer,
                                          SyncCoverDemandKind::MemoryRAW, loop,
                                          0, readyWitness)),
                  "add ready demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          2, releaseWitness)),
                  "add release demand");
  passed &= check(graph.freezeStructure(), "freeze exact lifecycle graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult result =
      discoverSyncCoverSlotLifecycles(graph, index);
  passed &= check(result && result.lifecycles.size() == 1,
                  "discover one exact physical-slot round trip");
  if (!result.lifecycles.empty()) {
    const SyncCoverSlotLifecycle &lifecycle = result.lifecycles.front();
    passed &= check(lifecycle.slot.domain == domain &&
                        lifecycle.slot.extent ==
                            SyncCoverStorageInterval{0, 64} &&
                        lifecycle.producerResource == 1 &&
                        lifecycle.consumerResource == 2 &&
                        lifecycle.recurrenceScope == loop &&
                        lifecycle.distance == 2 &&
                        lifecycle.ready ==
                            std::vector<SyncCoverCandidateOpportunityId>{0} &&
                        lifecycle.release ==
                            std::vector<SyncCoverCandidateOpportunityId>{1} &&
                        lifecycle.managedAccesses ==
                            std::vector<SyncCoverStorageAccessId>{write, read,
                                                                  externalWrite} &&
                        lifecycle.hasUnrepresentedAccesses &&
                        !lifecycle.requiresPathSensitiveProof,
                    "retain slot, pipe, recurrence, and opportunity identity");
  }
  SyncCoverSlotLifecycleOptions noCandidates;
  noCandidates.maximumLifecycles = 0;
  const SyncCoverSlotLifecycleResult capped =
      discoverSyncCoverSlotLifecycles(graph, index, noCandidates);
  passed &= check(capped && capped.lifecycles.empty() && capped.truncated,
                  "an explicit generation cap reports unknown via truncation");
  SyncCoverGraph other;
  passed &= check(other.freezeStructure(), "freeze unrelated graph");
  passed &= check(discoverSyncCoverSlotLifecycles(other, index).error ==
                      SyncCoverSlotLifecycleError::InvalidCandidateIndex,
                  "reject a candidate index from a different graph");
  return passed;
}

bool testFailClosedEvidence() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 80}, true), passed,
      "add fail-closed loop");
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(1, 1, loop, 10), passed, "add partial producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(2, 1, loop, 20), passed, "add partial consumer");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add partial domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(producer, domain, 20, {0, 64},
                             SyncCoverStorageAccessMode::Write),
      passed, "add partial write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(consumer, domain, 21, {32, 64},
                             SyncCoverStorageAccessMode::Read),
      passed, "add partial read");
  const SyncCoverStorageWitnessId readyWitness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add partial ready");
  const SyncCoverStorageWitnessId releaseWitness = takeIndex(
      graph.addStorageWitness(read, write), passed, "add partial release");
  passed &= check(graph.addDemand(demand(producer, consumer,
                                          SyncCoverDemandKind::MemoryRAW, loop,
                                          0, readyWitness)),
                  "add partial ready demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          1, releaseWitness)),
                  "add partial release demand");
  passed &= check(graph.freezeStructure(), "freeze partial lifecycle graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult result =
      discoverSyncCoverSlotLifecycles(graph, index);
  passed &= check(result && result.lifecycles.empty() &&
                      result.partialSlotOpportunities == 2,
                  "partial overlaps cannot establish slot ownership");
  return passed;
}

bool testGuardedRoundTripDiscovery() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 80}, true), passed,
      "add guarded loop");
  const SyncCoverControlId control =
      takeIndex(graph.addControl(2, loop), passed, "add guarded control");
  const SyncCoverGuard guard{{SyncCoverGuardLiteral{control, 0}}};
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(3, 1, loop, 10), passed, "add plain producer");
  const SyncCoverNodeId consumer =
      takeIndex(graph.addNode(4, 1, loop, 20), passed, "add plain consumer");
  const SyncCoverNodeId guardedProducer = takeIndex(
      graph.addNode(3, 1, loop, 30, guard), passed, "add guarded producer");
  const SyncCoverNodeId guardedConsumer = takeIndex(
      graph.addNode(4, 1, loop, 35, guard), passed, "add guarded consumer");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add guarded domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(producer, domain, 30, {128, 192},
                             SyncCoverStorageAccessMode::ReadWrite),
      passed, "add guarded write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(consumer, domain, 31, {128, 192},
                             SyncCoverStorageAccessMode::Read),
      passed, "add guarded read");
  const SyncCoverStorageAccessId guardedWrite = takeIndex(
      graph.addStorageAccess(guardedProducer, domain, 32, {128, 192},
                             SyncCoverStorageAccessMode::Write),
      passed, "add second guarded write");
  const SyncCoverStorageAccessId guardedRead = takeIndex(
      graph.addStorageAccess(guardedConsumer, domain, 33, {128, 192},
                             SyncCoverStorageAccessMode::Read),
      passed, "add second guarded read");
  const SyncCoverStorageWitnessId readyWitness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add guarded ready");
  const SyncCoverStorageWitnessId releaseWitness = takeIndex(
      graph.addStorageWitness(read, write), passed, "add guarded release");
  const SyncCoverStorageWitnessId secondReadyWitness = takeIndex(
      graph.addStorageWitness(guardedWrite, guardedRead), passed,
      "add second guarded ready");
  passed &= check(graph.addDemand(demand(producer, consumer,
                                          SyncCoverDemandKind::MemoryRAW, loop,
                                          0, readyWitness)),
                  "add guarded ready demand");
  passed &= check(
      graph.addDemand(demand(guardedProducer, guardedConsumer,
                             SyncCoverDemandKind::MemoryRAW, loop, 0,
                             secondReadyWitness)),
      "add second guarded ready demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          1, releaseWitness)),
                  "add guarded release demand");
  passed &= check(graph.freezeStructure(), "freeze guarded lifecycle graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult result =
      discoverSyncCoverSlotLifecycles(graph, index);
  passed &= check(result && result.lifecycles.size() == 1 &&
                      result.lifecycles.front().ready ==
                          std::vector<SyncCoverCandidateOpportunityId>{0, 1} &&
                      !result.lifecycles.front().hasUnrepresentedAccesses &&
                      result.lifecycles.front().requiresPathSensitiveProof,
                  "retain guarded lifecycle for independent protocol proof");
  return passed;
}

bool testNestedLoopRequiresPathProof() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId recurrence = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 100}, true), passed,
      "add outer recurrence loop");
  const SyncCoverScopeId zeroTrip = takeIndex(
      graph.addScope(recurrence, false, SyncCoverTimelineInterval{10, 80}, true),
      passed, "add nested zero-trip loop");
  const SyncCoverNodeId producer = takeIndex(
      graph.addNode(5, 1, zeroTrip, 20), passed, "add nested producer");
  const SyncCoverNodeId consumer = takeIndex(
      graph.addNode(6, 1, zeroTrip, 30), passed, "add nested consumer");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add nested domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(producer, domain, 40, {256, 320},
                             SyncCoverStorageAccessMode::Write),
      passed, "add nested write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(consumer, domain, 41, {256, 320},
                             SyncCoverStorageAccessMode::Read),
      passed, "add nested read");
  const SyncCoverStorageWitnessId readyWitness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add nested ready");
  const SyncCoverStorageWitnessId releaseWitness = takeIndex(
      graph.addStorageWitness(read, write), passed, "add nested release");
  passed &= check(graph.addDemand(demand(producer, consumer,
                                          SyncCoverDemandKind::MemoryRAW,
                                          zeroTrip, 0, readyWitness)),
                  "add nested ready demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR,
                                          recurrence, 1, releaseWitness)),
                  "add nested release demand");
  passed &= check(graph.freezeStructure(), "freeze nested lifecycle graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult result =
      discoverSyncCoverSlotLifecycles(graph, index);
  passed &= check(
      result && result.lifecycles.size() == 1 &&
          result.lifecycles.front().requiresPathSensitiveProof,
      "a nested zero-trip lifecycle cannot use an unconditional protocol");
  return passed;
}

} // namespace

int main() {
  return testExactRoundTripDiscovery() && testFailClosedEvidence() &&
                 testGuardedRoundTripDiscovery() &&
                 testNestedLoopRequiresPathProof()
             ? 0
             : 1;
}
