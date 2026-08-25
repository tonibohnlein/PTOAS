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
#include "PTO/Transforms/CanonicalSync/SyncCoverSlotProtocol.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

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

bool check(const SyncCoverMechanismResult &result,
           const std::string &message) {
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
                                          1, releaseWitness)),
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
                        lifecycle.distance == 1 &&
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
  const SyncCoverSlotProtocolResult openProtocols =
      buildSyncCoverSlotProtocolCandidates(graph, index, result);
  passed &= check(openProtocols && openProtocols.candidates.empty() &&
                      openProtocols.accessOpenLifecycles == 1,
                  "protocol factory rejects an unrepresented writer");
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
  const SyncCoverSlotProtocolResult protocols =
      buildSyncCoverSlotProtocolCandidates(graph, index, result);
  passed &= check(protocols && protocols.candidates.empty() &&
                      protocols.partialSlotOpportunities == 2,
                  "protocol factory preserves partial-view diagnostics");
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
  const SyncCoverSlotProtocolResult protocols =
      buildSyncCoverSlotProtocolCandidates(graph, index, result);
  passed &= check(protocols && protocols.candidates.empty() &&
                      protocols.pathSensitiveLifecycles == 1,
                  "protocol factory defers path-sensitive lifecycles");
  SyncCoverSlotLifecycleResult effectOnly = result;
  effectOnly.lifecycles.front().requiresPathSensitiveProof = false;
  const SyncCoverSlotProtocolResult unsupportedEffect =
      buildSyncCoverSlotProtocolCandidates(graph, index, effectOnly);
  passed &= check(unsupportedEffect &&
                      unsupportedEffect.candidates.empty() &&
                      unsupportedEffect.unsupportedEffectLifecycles == 1,
                  "protocol factory rejects non-exact access effects");
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

bool testUnitReleaseProtocolFactory() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId loop = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 80}, true), passed,
      "add protocol loop");
  const SyncCoverNodeId producer =
      takeIndex(graph.addNode(7, 1, loop, 10), passed, "add protocol producer");
  const SyncCoverNodeId consumer = takeIndex(
      graph.addNode(8, 1, loop, 20, {}, {7}), passed,
      "add protocol consumer");
  const SyncCoverNodeId lateConsumer = takeIndex(
      graph.addNode(8, 1, loop, 25, {}, {7}), passed,
      "add late protocol consumer");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add protocol domain");
  const SyncCoverStorageAccessId write = takeIndex(
      graph.addStorageAccess(producer, domain, 50, {384, 448},
                             SyncCoverStorageAccessMode::Write),
      passed, "add protocol write");
  const SyncCoverStorageAccessId read = takeIndex(
      graph.addStorageAccess(consumer, domain, 51, {384, 448},
                             SyncCoverStorageAccessMode::Read),
      passed, "add protocol read");
  const SyncCoverStorageAccessId lateRead = takeIndex(
      graph.addStorageAccess(lateConsumer, domain, 52, {384, 448},
                             SyncCoverStorageAccessMode::Read),
      passed, "add late protocol read");
  const SyncCoverStorageWitnessId readyWitness = takeIndex(
      graph.addStorageWitness(write, read), passed, "add protocol ready");
  const SyncCoverStorageWitnessId releaseWitness = takeIndex(
      graph.addStorageWitness(read, write), passed, "add protocol release");
  const SyncCoverStorageWitnessId lateReadyWitness = takeIndex(
      graph.addStorageWitness(write, lateRead), passed,
      "add late protocol ready");
  const SyncCoverStorageWitnessId lateReleaseWitness = takeIndex(
      graph.addStorageWitness(lateRead, write), passed,
      "add late protocol release");
  passed &= check(graph.addDemand(demand(producer, consumer,
                                          SyncCoverDemandKind::MemoryRAW, loop,
                                          0, readyWitness)),
                  "add protocol ready demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          1, releaseWitness)),
                  "add protocol release demand");
  passed &= check(graph.addDemand(demand(consumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          2, releaseWitness)),
                  "add non-unit protocol release demand");
  passed &= check(graph.addDemand(demand(producer, lateConsumer,
                                          SyncCoverDemandKind::MemoryRAW, loop,
                                          0, lateReadyWitness)),
                  "add late protocol ready demand");
  passed &= check(graph.addDemand(demand(lateConsumer, producer,
                                          SyncCoverDemandKind::MemoryWAR, loop,
                                          1, lateReleaseWitness)),
                  "add late protocol release demand");
  passed &= check(graph.freezeStructure(), "freeze protocol graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult lifecycles =
      discoverSyncCoverSlotLifecycles(graph, index);
  const SyncCoverSlotProtocolResult protocols =
      buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles);
  passed &= check(protocols && protocols.candidates.size() == 1 &&
                      protocols.pathSensitiveLifecycles == 0 &&
                      protocols.accessOpenLifecycles == 0 &&
                      protocols.unsupportedEffectLifecycles == 0 &&
                      protocols.unsupportedDistanceReleases == 1 &&
                      protocols.partialSlotOpportunities == 0 &&
                      protocols.nonBoundaryReleases == 1 &&
                      !protocols.truncated,
                  "build one access-closed unit release protocol");
  if (!protocols.candidates.empty()) {
    const SyncCoverSlotProtocolCandidate &candidate =
        protocols.candidates.front();
    passed &= check(candidate.lifecycle == 0 &&
                        candidate.releases ==
                            std::vector<SyncCoverCandidateOpportunityId>{4} &&
                        candidate.source == lateConsumer &&
                        candidate.targets ==
                            std::vector<SyncCoverNodeId>{producer} &&
                        candidate.sourceResource == 8 &&
                        candidate.targetResource == 7 &&
                        candidate.recurrenceScope == loop &&
                        candidate.distance == 1 &&
                        verifySyncCoverSlotProtocolCandidate(
                            graph, index, lifecycles.lifecycles.front(),
                            candidate),
                    "unit protocol candidate preserves the verified release");
    SyncCoverMechanismUniverse universe(graph);
    const SyncCoverMechanismResult eventDomain = universe.addResourceDomain(
        SyncCoverResourceKind::EventId, 8, 7, 8);
    passed &= check(eventDomain && eventDomain.index,
                    "add unit release event domain");
    if (eventDomain && eventDomain.index) {
      const auto descriptor = makeSyncCoverSlotProtocolDescriptor(
          universe.getResourceDomains()[*eventDomain.index], candidate, 91);
      passed &= check(descriptor.has_value(),
                      "build stock unit recurrence descriptor");
      if (descriptor) {
        passed &= check(
            universe.addVerifiedProtocol(
                *descriptor, [&](const auto &actual) {
                  return verifySyncCoverSlotProtocol(
                      index, lifecycles.lifecycles.front(), universe, candidate,
                      actual);
                }),
            "factory candidate passes the delegated stock token verifier");
      }
    }
  }
  SyncCoverSlotProtocolOptions capped;
  capped.maximumCandidates = 0;
  const SyncCoverSlotProtocolResult truncated =
      buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles, capped);
  passed &= check(truncated && truncated.candidates.empty() &&
                      truncated.truncated,
                  "protocol factory reports an explicit candidate cap");
  SyncCoverSlotProtocolOptions noEvaluations;
  noEvaluations.maximumEvaluations = 0;
  const SyncCoverSlotProtocolResult evaluationLimited =
      buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles,
                                           noEvaluations);
  passed &= check(evaluationLimited &&
                      evaluationLimited.candidates.empty() &&
                      evaluationLimited.evaluations == 0 &&
                      evaluationLimited.truncated,
                  "protocol factory reports an explicit evaluation cap");
  return passed;
}

bool testHierarchicalReleaseProtocolFactory() {
  bool passed = true;
  SyncCoverGraph graph;
  const SyncCoverScopeId outer = takeIndex(
      graph.addScope(0, true, SyncCoverTimelineInterval{2, 240}, true), passed,
      "add hierarchical outer loop");
  const SyncCoverScopeId inner = takeIndex(
      graph.addScope(outer, false, SyncCoverTimelineInterval{10, 70}, true),
      passed, "add zero-trip-capable inner loop");
  const SyncCoverControlId branch =
      takeIndex(graph.addControl(2, inner), passed, "add MMAD branch");
  const SyncCoverGuard firstGuard{{SyncCoverGuardLiteral{branch, 0}}};
  const SyncCoverGuard secondGuard{{SyncCoverGuardLiteral{branch, 1}}};
  const SyncCoverNodeId firstMmad = takeIndex(
      graph.addNode(7, 1, inner, 20, firstGuard), passed,
      "add first guarded MMAD");
  const SyncCoverNodeId secondMmad = takeIndex(
      graph.addNode(7, 1, inner, 30, secondGuard), passed,
      "add second guarded MMAD");
  const SyncCoverNodeId fix = takeIndex(
      graph.addNode(9, 1, outer, 90, {}, {7}), passed,
      "add outer FIX release");
  const SyncCoverStorageDomainId domain =
      takeIndex(graph.addStorageDomain(), passed, "add accumulator domain");
  const SyncCoverStorageAccessId firstAccumulator = takeIndex(
      graph.addStorageAccess(firstMmad, domain, 60, {0, 128},
                             SyncCoverStorageAccessMode::ReadWrite, 0),
      passed, "add first accumulator update");
  const SyncCoverStorageAccessId secondAccumulator = takeIndex(
      graph.addStorageAccess(secondMmad, domain, 61, {0, 128},
                             SyncCoverStorageAccessMode::ReadWrite, 0),
      passed, "add second accumulator update");
  const SyncCoverStorageAccessId fixRead = takeIndex(
      graph.addStorageAccess(fix, domain, 62, {0, 128},
                             SyncCoverStorageAccessMode::Read, 0),
      passed, "add FIX accumulator read");
  const SyncCoverStorageWitnessId firstReady = takeIndex(
      graph.addStorageWitness(firstAccumulator, fixRead), passed,
      "add first ready witness");
  const SyncCoverStorageWitnessId secondReady = takeIndex(
      graph.addStorageWitness(secondAccumulator, fixRead), passed,
      "add second ready witness");
  const SyncCoverStorageWitnessId firstRelease = takeIndex(
      graph.addStorageWitness(fixRead, firstAccumulator), passed,
      "add first release witness");
  const SyncCoverStorageWitnessId secondRelease = takeIndex(
      graph.addStorageWitness(fixRead, secondAccumulator), passed,
      "add second release witness");
  passed &= check(graph.addDemand(demand(firstMmad, fix,
                                          SyncCoverDemandKind::MemoryRAW,
                                          outer, 0, firstReady)),
                  "add first ready demand");
  passed &= check(graph.addDemand(demand(secondMmad, fix,
                                          SyncCoverDemandKind::MemoryRAW,
                                          outer, 0, secondReady)),
                  "add second ready demand");
  passed &= check(graph.addDemand(demand(fix, firstMmad,
                                          SyncCoverDemandKind::MemoryWAR,
                                          outer, 1, firstRelease)),
                  "add first guarded release demand");
  passed &= check(graph.addDemand(demand(fix, secondMmad,
                                          SyncCoverDemandKind::MemoryWAR,
                                          outer, 1, secondRelease)),
                  "add second guarded release demand");
  passed &= check(graph.freezeStructure(), "freeze hierarchical graph");

  const SyncCoverCandidateIndex index(graph);
  const SyncCoverSlotLifecycleResult lifecycles =
      discoverSyncCoverSlotLifecycles(graph, index);
  const SyncCoverSlotProtocolResult protocols =
      buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles);
  passed &= check(lifecycles && lifecycles.lifecycles.size() == 1 &&
                      lifecycles.lifecycles.front().requiresPathSensitiveProof,
                  "discover guarded nested accumulator lifecycle");
  passed &= check(protocols && protocols.candidates.size() == 1 &&
                      protocols.pathSensitiveLifecycles == 0,
                  "build one hierarchical release candidate");
  SyncCoverSlotProtocolOptions limited;
  limited.maximumEvaluations = 2;
  const SyncCoverSlotProtocolResult truncated =
      buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles, limited);
  passed &= check(truncated && truncated.candidates.empty() &&
                      truncated.evaluations == 0 && truncated.truncated,
                  "bound hierarchical release-edge evaluation explicitly");
  if (protocols.candidates.empty()) {
    return false;
  }
  const SyncCoverSlotProtocolCandidate &candidate =
      protocols.candidates.front();
  passed &= check(
      candidate.kind == SyncCoverSlotProtocolKind::HierarchicalRelease &&
          candidate.source == fix && candidate.targets ==
              std::vector<SyncCoverNodeId>{firstMmad, secondMmad} &&
          candidate.recurrenceScope == outer &&
          candidate.targetLoop == inner && candidate.distance == 1 &&
          verifySyncCoverSlotProtocolCandidate(
              graph, index, lifecycles.lifecycles.front(), candidate),
      "hierarchical candidate covers every guarded MMAD from loop entry");

  SyncCoverMechanismUniverse universe(graph);
  const SyncCoverMechanismResult eventDomain = universe.addResourceDomain(
      SyncCoverResourceKind::EventId, 9, 7, 8);
  passed &= check(eventDomain && eventDomain.index,
                  "add hierarchical event domain");
  if (eventDomain && eventDomain.index) {
    const auto descriptor = makeSyncCoverSlotProtocolDescriptor(
        universe.getResourceDomains()[*eventDomain.index], candidate, 101);
    passed &= check(descriptor.has_value(),
                    "build hierarchical prime/body/drain descriptor");
    if (descriptor) {
      passed &= check(
          universe.addVerifiedProtocol(
              *descriptor, [&](const auto &actual) {
                return verifySyncCoverSlotProtocol(
                    index, lifecycles.lifecycles.front(), universe, candidate,
                    actual);
              }),
          "admit independently verified hierarchical protocol");
      SyncCoverMechanismDescriptor malformed = *descriptor;
      malformed.actions[1].anchor.scope = outer;
      passed &= check(
          !verifySyncCoverSlotProtocol(
              index, lifecycles.lifecycles.front(), universe, candidate,
              malformed),
          "reject a wait that moved outside the nested-loop boundary");
      malformed = *descriptor;
      malformed.actions[0].anchor.kind = SyncCoverAnchorKind::ScopeExit;
      passed &= check(
          !verifySyncCoverSlotProtocol(
              index, lifecycles.lifecycles.front(), universe, candidate,
              malformed),
          "reject a malformed hierarchical prime");
      malformed = *descriptor;
      malformed.supplyBindings.front().consumeAction = 3;
      passed &= check(
          !verifySyncCoverSlotProtocol(
              index, lifecycles.lifecycles.front(), universe, candidate,
              malformed),
          "reject a malformed hierarchical supply binding");
      SyncCoverGraph unrelated;
      passed &= check(unrelated.freezeStructure(),
                      "freeze unrelated protocol graph");
      SyncCoverMechanismUniverse unrelatedUniverse(unrelated);
      passed &= check(
          !verifySyncCoverSlotProtocol(
              index, lifecycles.lifecycles.front(), unrelatedUniverse,
              candidate, *descriptor),
          "reject protocol evidence from a different graph");
      SyncCoverSlotProtocolCandidate outOfRange = candidate;
      outOfRange.source = graph.getNodes().size();
      passed &= check(
          !verifySyncCoverSlotProtocolCandidate(
              graph, index, lifecycles.lifecycles.front(), outOfRange),
          "reject an out-of-range hierarchical release source");
    }
  }
  return passed;
}

bool testHierarchicalReleaseRejectsUnsafeNesting() {
  bool passed = true;
  enum class Scenario {
    InterveningLoop,
    OptionalParent,
    EarlyFix,
    ExtraAccess,
  };
  const Scenario scenarios[] = {Scenario::InterveningLoop,
                                Scenario::OptionalParent,
                                Scenario::EarlyFix, Scenario::ExtraAccess};
  for (Scenario scenario : scenarios) {
    SyncCoverGraph graph;
    const SyncCoverScopeId outer = takeIndex(
        graph.addScope(0, true, SyncCoverTimelineInterval{2, 300}, true),
        passed, "add rejected outer loop");
    SyncCoverScopeId parent = outer;
    if (scenario == Scenario::InterveningLoop) {
      parent = takeIndex(
          graph.addScope(outer, true, SyncCoverTimelineInterval{20, 180},
                         true),
          passed, "add intervening loop");
    } else if (scenario == Scenario::OptionalParent) {
      parent = takeIndex(
          graph.addScope(outer, false, SyncCoverTimelineInterval{20, 180},
                         false),
          passed, "add optional target-loop parent");
    }
    const SyncCoverTimelineInterval innerTimeline =
        scenario == Scenario::EarlyFix
            ? SyncCoverTimelineInterval{40, 200}
            : SyncCoverTimelineInterval{40, 140};
    const SyncCoverScopeId inner = takeIndex(
        graph.addScope(parent, false, innerTimeline, true), passed,
        "add rejected target loop");
    const SyncCoverNodeId mmad = takeIndex(
        graph.addNode(7, 1, inner, 60), passed, "add rejected MMAD");
    if (scenario == Scenario::EarlyFix) {
      static_cast<void>(takeIndex(graph.addNode(11, 1, inner, 90), passed,
                                  "add late inner operation"));
    }
    const std::size_t fixOrder =
        scenario == Scenario::EarlyFix ? 80 : 120;
    const SyncCoverNodeId fix = takeIndex(
        graph.addNode(9, 1, outer, fixOrder, {}, {7}), passed,
        "add rejected FIX");
    const SyncCoverStorageDomainId domain = takeIndex(
        graph.addStorageDomain(), passed, "add rejected accumulator domain");
    const SyncCoverStorageAccessId accumulator = takeIndex(
        graph.addStorageAccess(mmad, domain, 70, {0, 128},
                               SyncCoverStorageAccessMode::ReadWrite, 0),
        passed, "add rejected accumulator update");
    const SyncCoverStorageAccessId fixRead = takeIndex(
        graph.addStorageAccess(fix, domain, 71, {0, 128},
                               SyncCoverStorageAccessMode::Read, 0),
        passed, "add rejected FIX read");
    if (scenario == Scenario::ExtraAccess) {
      const SyncCoverNodeId extra = takeIndex(
          graph.addNode(12, 1, outer, 100), passed,
          "add unrepresented slot accessor");
      static_cast<void>(takeIndex(
          graph.addStorageAccess(extra, domain, 72, {0, 128},
                                 SyncCoverStorageAccessMode::Read, 0),
          passed, "add unrepresented overlapping access"));
    }
    const SyncCoverStorageWitnessId ready = takeIndex(
        graph.addStorageWitness(accumulator, fixRead), passed,
        "add rejected ready witness");
    const SyncCoverStorageWitnessId release = takeIndex(
        graph.addStorageWitness(fixRead, accumulator), passed,
        "add rejected release witness");
    passed &= check(graph.addDemand(demand(mmad, fix,
                                            SyncCoverDemandKind::MemoryRAW,
                                            outer, 0, ready)),
                    "add rejected ready demand");
    passed &= check(graph.addDemand(demand(fix, mmad,
                                            SyncCoverDemandKind::MemoryWAR,
                                            outer, 1, release)),
                    "add rejected release demand");
    passed &= check(graph.freezeStructure(), "freeze rejected graph");
    const SyncCoverCandidateIndex index(graph);
    const SyncCoverSlotLifecycleResult lifecycles =
        discoverSyncCoverSlotLifecycles(graph, index);
    const SyncCoverSlotProtocolResult protocols =
        buildSyncCoverSlotProtocolCandidates(graph, index, lifecycles);
    const bool classifiedSafely =
        scenario == Scenario::ExtraAccess
            ? protocols.accessOpenLifecycles == 1
            : protocols.pathSensitiveLifecycles == 1;
    passed &= check(lifecycles && lifecycles.lifecycles.size() == 1 &&
                        protocols && protocols.candidates.empty() &&
                        classifiedSafely,
                    "reject unsafe hierarchical execution cardinality");
  }
  return passed;
}

} // namespace

int main() {
  return testExactRoundTripDiscovery() && testFailClosedEvidence() &&
                 testGuardedRoundTripDiscovery() &&
                 testNestedLoopRequiresPathProof() &&
                 testUnitReleaseProtocolFactory() &&
                 testHierarchicalReleaseProtocolFactory() &&
                 testHierarchicalReleaseRejectsUnsafeNesting()
             ? 0
             : 1;
}
