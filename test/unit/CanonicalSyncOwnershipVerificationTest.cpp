// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <iostream>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
  }
  return condition;
}

CanonicalOwnershipCycle makeCycle() {
  CanonicalOwnershipCycle cycle;
  cycle.id = 7;
  cycle.producerPipe = PipelineType::PIPE_MTE1;
  cycle.consumerPipe = PipelineType::PIPE_M;
  cycle.lanes.resize(2);
  cycle.lanes[0].id = 0;
  cycle.lanes[1].id = 1;
  CanonicalOwnershipPath path;
  for (unsigned lane = 0; lane < 2; ++lane) {
    CanonicalOwnershipUse use;
    use.lane = lane;
    use.producers.push_back(lane * 2);
    use.consumers.push_back(lane * 2 + 1);
    use.writeAcquireAnchor = {nullptr, true};
    use.readyAnchor = {nullptr, false};
    use.readAcquireAnchor = {nullptr, true};
    use.releaseAnchor = {nullptr, false};
    path.uses.push_back(std::move(use));
  }
  cycle.paths.push_back(std::move(path));
  return cycle;
}

CanonicalEvent makeReadyEvent(const CanonicalOwnershipCycle &cycle) {
  return buildCanonicalOwnershipProtocols(cycle).first;
}

CanonicalEvent makeReleaseEvent(const CanonicalOwnershipCycle &cycle) {
  return buildCanonicalOwnershipProtocols(cycle).second;
}

bool verifyPair(const CanonicalOwnershipCycle &cycle,
                const CanonicalEvent &first, const CanonicalEvent &second) {
  const CanonicalEvent *events[] = {&first, &second};
  return verifyCanonicalOwnershipEventPair(cycle, events);
}

bool testOwnershipPairVerification() {
  const CanonicalOwnershipCycle cycle = makeCycle();
  const CanonicalEvent ready = makeReadyEvent(cycle);
  const CanonicalEvent release = makeReleaseEvent(cycle);
  bool passed = check(verifyPair(cycle, ready, release),
                      "a matching ready/release pair is valid");
  passed &= check(verifyPair(cycle, release, ready),
                  "pair verification is independent of event order");

  const CanonicalEvent *single[] = {&ready};
  passed &= check(!verifyCanonicalOwnershipEventPair(cycle, single),
                  "an ownership protocol requires both lifecycle roles");

  CanonicalEvent malformed = ready;
  malformed.ownershipRole = CanonicalOwnershipEventRole::Release;
  passed &= check(!verifyPair(cycle, malformed, release),
                  "duplicate release roles are rejected");

  malformed = ready;
  malformed.targetPipe = PipelineType::PIPE_MTE2;
  passed &= check(!verifyPair(cycle, malformed, release),
                  "a non-complementary pipe direction is rejected");

  malformed = release;
  malformed.width = 1;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "a width that differs from the cycle is rejected");

  malformed = release;
  malformed.ownershipCycle = cycle.id + 1;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "an event naming another cycle is rejected");

  malformed = release;
  malformed.actions[1].phase = CanonicalEventActionPhase::Straight;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "a malformed release lifecycle is rejected");

  malformed = ready;
  malformed.completions.front().target =
      cycle.paths.front().uses.front().producers.front();
  passed &= check(!verifyPair(cycle, malformed, release),
                  "completion endpoints must preserve ownership roles");

  malformed = ready;
  malformed.completions.front().target =
      cycle.paths.front().uses.back().consumers.front();
  passed &= check(!verifyPair(cycle, malformed, release),
                  "ready completions cannot cross ownership lanes");

  malformed = ready;
  malformed.completions.pop_back();
  passed &= check(!verifyPair(cycle, malformed, release),
                  "ready completion coverage must be complete");

  malformed = release;
  malformed.completions.pop_back();
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "release completion coverage must be complete");

  malformed = ready;
  malformed.completions.push_back(malformed.completions.front());
  passed &= check(!verifyPair(cycle, malformed, release),
                  "duplicate ready completions are rejected");

  malformed = release;
  malformed.completions.front().waitAction =
      malformed.completions.front().setAction;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "completion action indices are part of the protocol");

  malformed = release;
  malformed.traces.front().actions.push_back(1);
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "trace membership is part of the protocol");

  CanonicalOwnershipCycle accumulator = makeCycle();
  accumulator.kind = CanonicalOwnershipKind::L0Accumulator;
  accumulator.producerPipe = PipelineType::PIPE_M;
  accumulator.consumerPipe = PipelineType::PIPE_FIX;
  accumulator.lanes.resize(1);
  accumulator.paths.front().uses.resize(1);
  const CanonicalEvent accumulatorReady = makeReadyEvent(accumulator);
  const CanonicalEvent accumulatorRelease = makeReleaseEvent(accumulator);
  passed &= check(verifyPair(accumulator, accumulatorReady,
                             accumulatorRelease),
                  "a one-lane L0C ownership lifecycle is valid");
  return passed;
}

bool testOwnershipCandidateTransaction() {
  CanonicalBarrier baselineBarrier;
  baselineBarrier.pipe = PipelineType::PIPE_M;
  CanonicalEvent baselineEvent;
  baselineEvent.source = 100;
  std::vector<CanonicalBarrier> barriers{baselineBarrier};
  std::vector<CanonicalEvent> events{baselineEvent};
  std::vector<CanonicalEvent> accepted;

  CanonicalEvent ready;
  ready.source = 1;
  ready.ownershipRole = CanonicalOwnershipEventRole::Ready;
  CanonicalEvent release;
  release.source = 2;
  release.ownershipRole = CanonicalOwnershipEventRole::Release;
  bool evaluatedCompletePair = false;
  const bool rejected = tryCommitCanonicalOwnershipCandidate(
      accepted, barriers, events, ready, release, [&]() {
        evaluatedCompletePair = accepted.size() == 2;
        barriers.clear();
        events.clear();
        return false;
      });
  bool passed = check(!rejected && evaluatedCompletePair,
                      "a rejected transaction evaluates the complete pair");
  passed &= check(accepted.empty() && barriers.size() == 1 &&
                      barriers.front().pipe == PipelineType::PIPE_M &&
                      events.size() == 1 && events.front().source == 100,
                  "a rejected transaction restores every state vector");

  const bool committed = tryCommitCanonicalOwnershipCandidate(
      accepted, barriers, events, ready, release, [&]() {
        barriers.front().pipe = PipelineType::PIPE_FIX;
        events.front().source = 200;
        return true;
      });
  passed &= check(committed && accepted.size() == 2 &&
                      barriers.front().pipe == PipelineType::PIPE_FIX &&
                      events.front().source == 200,
                  "an accepted transaction retains its pair and plan");

  CanonicalEvent secondReady = ready;
  secondReady.source = 3;
  CanonicalEvent secondRelease = release;
  secondRelease.source = 4;
  const bool secondCommitted = tryCommitCanonicalOwnershipCandidate(
      accepted, barriers, events, secondReady, secondRelease, [&]() {
        barriers.clear();
        events.clear();
        return false;
      });
  passed &= check(!secondCommitted && accepted.size() == 2 &&
                      accepted[0].source == 1 && accepted[1].source == 2 &&
                      barriers.size() == 1 &&
                      barriers.front().pipe == PipelineType::PIPE_FIX &&
                      events.size() == 1 && events.front().source == 200,
                  "rejecting a later pair preserves the committed plan");
  return passed;
}

bool testEventBundleConstruction() {
  CanonicalEvent standalone;

  CanonicalEvent syntheticReady;
  syntheticReady.protocolBundle = 3;
  CanonicalEvent syntheticRelease;
  syntheticRelease.protocolBundle = 3;

  CanonicalEvent ownershipReady;
  ownershipReady.ownershipProtocol = true;
  ownershipReady.ownershipCycle = 7;
  ownershipReady.ownershipRole = CanonicalOwnershipEventRole::Ready;
  CanonicalEvent ownershipRelease = ownershipReady;
  ownershipRelease.ownershipRole = CanonicalOwnershipEventRole::Release;

  const std::vector<CanonicalEvent> events = {
      standalone, syntheticReady, ownershipReady, syntheticRelease,
      ownershipRelease};
  const std::vector<CanonicalEventBundleCandidate> bundles =
      buildCanonicalEventBundles(events);
  bool passed = check(bundles.size() == 3,
                      "standalone and paired protocols form three bundles");
  passed &= check(bundles[0].kind == CanonicalEventBundleKind::Standalone &&
                      bundles[0].events.size() == 1,
                  "a standalone event forms a one-event bundle");
  passed &= check(
      bundles[1].kind == CanonicalEventBundleKind::SyntheticRoundTrip &&
          bundles[1].id == 1 && bundles[1].protocolIdentity == 3 &&
          bundles[1].events.size() == 2,
      "synthetic events are grouped by protocol identity");
  passed &= check(bundles[2].kind == CanonicalEventBundleKind::Ownership &&
                      bundles[2].id == 2 &&
                      bundles[2].protocolIdentity == 7 &&
                      bundles[2].events.size() == 2,
                  "ownership ready and release form one atomic bundle");

  const std::vector<CanonicalEvent> flattened =
      flattenCanonicalEventBundles(bundles);
  passed &= check(flattened.size() == events.size(),
                  "flattening preserves every event");
  passed &= check(flattened[1].protocolBundle == 3 &&
                      flattened[2].protocolBundle == 3 &&
                      flattened[3].ownershipCycle == 7 &&
                      flattened[4].ownershipCycle == 7,
                  "flattening keeps stable bundle order");
  return passed;
}

bool testSyntheticRoundTripVerification() {
  CanonicalEvent sourceToBridge;
  sourceToBridge.source = 1;
  sourceToBridge.target = 2;
  sourceToBridge.sourcePipe = PipelineType::PIPE_MTE3;
  sourceToBridge.targetPipe = PipelineType::PIPE_V;
  sourceToBridge.protocolBundle = 5;

  CanonicalEvent bridgeToTarget;
  bridgeToTarget.source = 2;
  bridgeToTarget.target = 4;
  bridgeToTarget.sourcePipe = PipelineType::PIPE_V;
  bridgeToTarget.targetPipe = PipelineType::PIPE_MTE3;
  bridgeToTarget.protocolBundle = 5;

  const CanonicalEvent *valid[] = {&bridgeToTarget, &sourceToBridge};
  bool passed = check(verifyCanonicalSyntheticRoundTripBundle(valid),
                      "a complementary two-event chain is a round trip");
  CanonicalDependency witness;
  witness.source = 1;
  witness.target = 4;
  passed &= check(verifyCanonicalSyntheticRoundTripWitness(valid, witness),
                  "a round trip verifies its same-pipe completion witness");

  witness.target = 3;
  passed &= check(!verifyCanonicalSyntheticRoundTripWitness(valid, witness),
                  "a round trip cannot claim another completion witness");
  witness.target = 4;
  witness.iterationDistance = 1;
  passed &= check(!verifyCanonicalSyntheticRoundTripWitness(valid, witness),
                  "a forward round trip cannot claim a recurrence witness");

  const CanonicalEvent *missingHalf[] = {&sourceToBridge};
  passed &= check(!verifyCanonicalSyntheticRoundTripBundle(missingHalf),
                  "a synthetic round trip requires both event halves");

  CanonicalEvent wrongDirection = bridgeToTarget;
  wrongDirection.targetPipe = PipelineType::PIPE_MTE2;
  const CanonicalEvent *malformed[] = {&sourceToBridge, &wrongDirection};
  passed &= check(!verifyCanonicalSyntheticRoundTripBundle(malformed),
                  "a synthetic round trip requires complementary pipes");
  return passed;
}

bool testEventBundleIdentityRestoration() {
  CanonicalEvent sourceToBridge;
  sourceToBridge.source = 1;
  sourceToBridge.target = 2;
  sourceToBridge.sourcePipe = PipelineType::PIPE_MTE3;
  sourceToBridge.targetPipe = PipelineType::PIPE_V;
  sourceToBridge.protocolBundle = 5;

  CanonicalEvent bridgeToTarget;
  bridgeToTarget.source = 2;
  bridgeToTarget.target = 4;
  bridgeToTarget.sourcePipe = PipelineType::PIPE_V;
  bridgeToTarget.targetPipe = PipelineType::PIPE_MTE3;
  bridgeToTarget.protocolBundle = 5;

  const CanonicalOwnershipCycle cycle = makeCycle();
  CanonicalEvent ownershipReady = makeReadyEvent(cycle);
  CanonicalEvent ownershipRelease = makeReleaseEvent(cycle);
  std::vector<CanonicalEventBundleCandidate> known =
      buildCanonicalEventBundles({sourceToBridge, ownershipReady,
                                  bridgeToTarget, ownershipRelease});
  auto knownSynthetic = llvm::find_if(known, [](const auto &bundle) {
    return bundle.kind == CanonicalEventBundleKind::SyntheticRoundTrip;
  });
  auto knownOwnership = llvm::find_if(known, [](const auto &bundle) {
    return bundle.kind == CanonicalEventBundleKind::Ownership;
  });
  knownSynthetic->id = 41;
  knownSynthetic->conflicts.push_back(42);
  knownOwnership->id = 42;
  knownOwnership->conflicts.push_back(41);
  CanonicalDependency witness;
  witness.source = 1;
  witness.target = 4;
  knownSynthetic->completionWitness = witness;

  CanonicalEvent newStandalone;
  newStandalone.source = 8;
  newStandalone.target = 9;
  std::vector<CanonicalEventBundleCandidate> rebuilt =
      buildCanonicalEventBundles({bridgeToTarget, ownershipRelease,
                                  newStandalone, sourceToBridge,
                                  ownershipReady});
  std::size_t nextId = 100;
  bool passed = check(succeeded(restoreCanonicalEventBundleIdentities(
                          rebuilt, known, nextId)),
                      "scarcity reconciliation restores selected identities");
  auto rebuiltSynthetic = llvm::find_if(rebuilt, [](const auto &bundle) {
    return bundle.kind == CanonicalEventBundleKind::SyntheticRoundTrip;
  });
  auto rebuiltOwnership = llvm::find_if(rebuilt, [](const auto &bundle) {
    return bundle.kind == CanonicalEventBundleKind::Ownership;
  });
  auto rebuiltStandalone = llvm::find_if(rebuilt, [](const auto &bundle) {
    return bundle.kind == CanonicalEventBundleKind::Standalone;
  });
  passed &= check(rebuilt.size() == 3 && rebuiltSynthetic != rebuilt.end() &&
                      rebuiltOwnership != rebuilt.end() &&
                      rebuiltStandalone != rebuilt.end(),
                  "reconciliation retains atomic pairs and a new standalone");
  passed &= check(rebuiltSynthetic->id == 41 &&
                      rebuiltSynthetic->conflicts.size() == 1 &&
                      rebuiltSynthetic->conflicts.front() == 42 &&
                      rebuiltSynthetic->completionWitness &&
                      rebuiltSynthetic->completionWitness->source == 1 &&
                      rebuiltSynthetic->completionWitness->target == 4,
                  "a synthetic pair keeps its conflict and completion witness");
  passed &= check(rebuiltOwnership->id == 42 &&
                      rebuiltOwnership->conflicts.size() == 1 &&
                      rebuiltOwnership->conflicts.front() == 41,
                  "an ownership pair keeps its selected conflict identity");
  passed &= check(rebuiltStandalone->id == 100 && nextId == 101,
                  "a new standalone receives a fresh non-conflicting identity");

  std::vector<CanonicalEvent> projection =
      flattenCanonicalEventBundles(rebuilt);
  passed &= check(canonicalEventBundleProjectionMatches(rebuilt, projection),
                  "the selected bundle projection matches exactly");
  projection.front().target += 1;
  passed &= check(!canonicalEventBundleProjectionMatches(rebuilt, projection),
                  "a stale non-empty event projection is rejected");
  passed &= check(canonicalEventBundleProjectionMatches({}, {}),
                  "an empty selected plan has an empty valid projection");

  std::vector<CanonicalEventBundleCandidate> overflow = rebuilt;
  std::vector<CanonicalEventBundleCandidate> exhaustedKnown = known;
  exhaustedKnown.front().id = std::numeric_limits<std::size_t>::max();
  nextId = 0;
  passed &= check(failed(restoreCanonicalEventBundleIdentities(
                      overflow, exhaustedKnown, nextId)),
                  "the maximum stable identity is reserved as exhaustion");
  overflow = buildCanonicalEventBundles({newStandalone});
  nextId = std::numeric_limits<std::size_t>::max();
  passed &= check(failed(restoreCanonicalEventBundleIdentities(
                      overflow, {}, nextId)),
                  "fresh identity exhaustion fails instead of wrapping");
  return passed;
}

bool testEventBundleConflicts() {
  CanonicalEventBundleCandidate first;
  first.id = 1;
  first.conflicts.push_back(2);
  CanonicalEventBundleCandidate second;
  second.id = 2;
  bool passed = check(!canonicalEventBundlesHaveNoConflicts({first, second}),
                      "selected conflicting bundles are rejected");
  passed &= check(canonicalEventBundlesHaveNoConflicts({first}),
                  "a conflict with an unselected bundle is harmless");
  second.id = 1;
  passed &= check(!canonicalEventBundlesHaveNoConflicts({first, second}),
                  "duplicate stable bundle identities are rejected");
  return passed;
}

bool testDiagnosticEventBundleEquivalence() {
  CanonicalEvent event;
  event.source = 1;
  event.target = 2;
  event.sourcePipe = PipelineType::PIPE_MTE2;
  event.targetPipe = PipelineType::PIPE_V;
  event.intervalBegin = 3;
  event.intervalEnd = 8;
  CanonicalEventAction action;
  action.kind = CanonicalEventActionKind::Set;
  action.phase = CanonicalEventActionPhase::Straight;
  event.actions.push_back(action);

  CanonicalEventBundleCandidate first;
  first.id = 1;
  first.events.push_back(event);
  CanonicalEventBundleCandidate second = first;
  second.id = 2;
  std::vector<CanonicalEventBundleCandidate> universe{first, second};
  bool passed = check(canonicalDiagnosticEventBundlesEquivalent(
                          universe[0], universe[1], universe),
                      "stable IDs do not distinguish equivalent protocols");

  universe[1].events.front().actions.front().phase =
      CanonicalEventActionPhase::Body;
  passed &= check(!canonicalDiagnosticEventBundlesEquivalent(
                      universe[0], universe[1], universe),
                  "different emitted actions are distinct protocols");
  passed &= check(!canonicalDiagnosticEventBundleMatchesSelected(
                      universe[1], ArrayRef(universe).take_front(), universe),
                  "a distinct protocol variant remains non-incumbent");
  universe[1].events.front().actions.front().phase =
      CanonicalEventActionPhase::Straight;
  passed &= check(canonicalDiagnosticEventBundleMatchesSelected(
                      universe[1], ArrayRef(universe).take_front(), universe),
                  "an equivalent selected protocol is incumbent");
  universe[1] = second;
  CanonicalDependency witness;
  witness.source = 1;
  witness.target = 4;
  universe[0].completionWitness = witness;
  passed &= check(!canonicalDiagnosticEventBundlesEquivalent(
                      universe[0], universe[1], universe),
                  "different completion witnesses are distinct protocols");

  universe[0].completionWitness.reset();
  CanonicalEventBundleCandidate firstConflict = first;
  firstConflict.id = 3;
  firstConflict.events.front().target = 5;
  CanonicalEventBundleCandidate secondConflict = firstConflict;
  secondConflict.id = 4;
  universe.push_back(firstConflict);
  universe.push_back(secondConflict);
  universe[0].conflicts.push_back(firstConflict.id);
  universe[1].conflicts.push_back(secondConflict.id);
  passed &= check(canonicalDiagnosticEventBundlesEquivalent(
                      universe[0], universe[1], universe),
                  "equivalent conflict protocols preserve equivalence");
  universe[3].events.front().target = 6;
  passed &= check(!canonicalDiagnosticEventBundlesEquivalent(
                      universe[0], universe[1], universe),
                  "different conflict behavior is not deduplicated");
  return passed;
}

bool testEventBundleAtomicExchange() {
  CanonicalEventBundleCandidate conflicting;
  conflicting.id = 1;
  conflicting.conflicts.push_back(3);
  CanonicalEventBundleCandidate unrelated;
  unrelated.id = 2;
  std::vector<CanonicalEventBundleCandidate> selected{conflicting, unrelated};

  const CanonicalOwnershipCycle cycle = makeCycle();
  CanonicalEventBundleCandidate ownership;
  ownership.id = 3;
  ownership.kind = CanonicalEventBundleKind::Ownership;
  ownership.protocolIdentity = cycle.id;
  ownership.events.push_back(makeReadyEvent(cycle));
  ownership.events.push_back(makeReleaseEvent(cycle));
  ownership.conflicts.push_back(conflicting.id);

  bool passed = check(!appendCanonicalEventBundleCandidate(selected,
                                                             ownership),
                      "diagnostic append rejects a retained conflict");
  passed &= check(selected.size() == 2 &&
                      selected[0].id == conflicting.id &&
                      selected[1].id == unrelated.id,
                  "rejected diagnostic append preserves the selected plan");
  passed &= check(exchangeCanonicalEventBundleCandidate(selected,
                                                          ownership),
                      "an ownership candidate can replace a conflict");
  passed &= check(selected.size() == 2 && selected[0].id == unrelated.id &&
                      selected[1].id == ownership.id &&
                      selected[1].events.size() == 2 &&
                      selected[1].events[0].ownershipRole ==
                          CanonicalOwnershipEventRole::Ready &&
                      selected[1].events[1].ownershipRole ==
                          CanonicalOwnershipEventRole::Release,
                  "bundle exchange keeps the ownership pair atomic");
  passed &= check(canonicalEventBundlesHaveNoConflicts(selected),
                  "bundle exchange removes conflicts in both directions");
  passed &= check(!exchangeCanonicalEventBundleCandidate(selected,
                                                          ownership) &&
                      selected.size() == 2,
                  "reselecting an equivalent bundle is a no-op");
  return passed;
}

bool testReservedEventColorOverflow() {
  CanonicalEvent event;
  event.sourcePipe = PipelineType::PIPE_S;
  event.targetPipe = PipelineType::PIPE_V;
  event.width = 2;
  event.intervalBegin = 4;
  event.intervalEnd = 9;
  const CanonicalEventDomainKey key{event.sourcePipe, event.targetPipe};
  std::map<CanonicalEventDomainKey, std::set<unsigned>> reserved;

  bool passed = check(calculateCanonicalEventColorOverflow({event}, 2,
                                                            reserved) == 0,
                      "two overlapping lanes fit two unreserved IDs");
  reserved[key].insert(0);
  passed &= check(calculateCanonicalEventColorOverflow({event}, 2,
                                                        reserved) == 1,
                  "a reserved in-range ID contributes exact overflow");
  reserved[key].insert(7);
  passed &= check(calculateCanonicalEventColorOverflow({event}, 2,
                                                        reserved) == 1,
                  "an out-of-range reservation does not reduce the budget");
  reserved[key].insert(1);
  passed &= check(calculateCanonicalEventColorOverflow({event}, 2,
                                                        reserved) == 2,
                  "reserving the full domain budget counts both lanes");
  return passed;
}

bool testMechanismPlanScoreOrdering() {
  CanonicalMechanismPlanScore baseline;
  baseline.dynamicActionProfile = {8, 2};
  baseline.barrierCount = 1;
  baseline.candidateSignature = {3};

  CanonicalMechanismPlanScore ownership = baseline;
  ownership.usefulOwnershipBundles = 1;
  ownership.ownershipSignature = {7};
  bool passed = check(canonicalMechanismPlanScoreLess(ownership, baseline),
                      "useful ownership is the primary plan preference");

  CanonicalMechanismPlanScore fewerInnerActions = baseline;
  fewerInnerActions.dynamicActionProfile = {7, 100};
  passed &= check(
      canonicalMechanismPlanScoreLess(fewerInnerActions, baseline),
      "the deepest repeated action count is minimized before outer actions");

  CanonicalMechanismPlanScore fewerBarriers = baseline;
  fewerBarriers.barrierCount = 0;
  passed &= check(canonicalMechanismPlanScoreLess(fewerBarriers, baseline),
                  "barrier count breaks otherwise equal structural scores");

  CanonicalMechanismPlanScore innerBarrier = baseline;
  innerBarrier.barrierActionProfile = {1, 0};
  CanonicalMechanismPlanScore outerBarrier = baseline;
  outerBarrier.barrierActionProfile = {0, 1};
  passed &= check(canonicalMechanismPlanScoreLess(outerBarrier, innerBarrier),
                  "an outer barrier is preferred to an inner-loop barrier");

  CanonicalMechanismPlanScore stableTie = baseline;
  stableTie.candidateSignature = {2};
  passed &= check(canonicalMechanismPlanScoreLess(stableTie, baseline),
                  "stable candidate identity resolves the final tie");
  passed &= check(!canonicalMechanismPlanScoreLess(baseline, baseline),
                  "the total plan order is irreflexive");
  return passed;
}

bool testBarrierActionProfileConstruction() {
  MLIRContext context;
  context.getOrLoadDialect<scf::SCFDialect>();
  const Location location = UnknownLoc::get(&context);

  OperationState loopState(location, scf::ForOp::getOperationName());
  loopState.addRegion();
  Operation *loop = Operation::create(loopState);
  Block *body = new Block();
  loop->getRegion(0).push_back(body);
  OperationState bodyAnchorState(location, scf::YieldOp::getOperationName());
  Operation *bodyAnchor = Operation::create(bodyAnchorState);
  body->push_back(bodyAnchor);

  CanonicalBarrier outside;
  outside.anchor = {loop, true};
  CanonicalBarrier inside;
  inside.anchor = {bodyAnchor, true};
  const std::vector<std::size_t> outsideProfile =
      buildCanonicalBarrierActionProfile({outside}, 1);
  const std::vector<std::size_t> insideProfile =
      buildCanonicalBarrierActionProfile({inside}, 1);

  bool passed = check(outsideProfile == std::vector<std::size_t>({0, 1}),
                      "a barrier before scf.for is in the outer bucket");
  passed &= check(insideProfile == std::vector<std::size_t>({1, 0}),
                  "a barrier in the loop body is in the repeated bucket");

  CanonicalMechanismPlanScore outsideScore;
  outsideScore.barrierActionProfile = outsideProfile;
  CanonicalMechanismPlanScore insideScore;
  insideScore.barrierActionProfile = insideProfile;
  passed &= check(canonicalMechanismPlanScoreLess(outsideScore, insideScore),
                  "the constructed outer profile wins the score tie");
  loop->destroy();
  return passed;
}

bool testCandidateFrontierTruncation() {
  CanonicalEventBundleCandidate standalone;
  standalone.id = 1;
  standalone.kind = CanonicalEventBundleKind::Standalone;
  CanonicalEventBundleCandidate synthetic;
  synthetic.id = 9;
  synthetic.kind = CanonicalEventBundleKind::SyntheticRoundTrip;
  CanonicalEventBundleCandidate ownershipHigh;
  ownershipHigh.id = 5;
  ownershipHigh.kind = CanonicalEventBundleKind::Ownership;
  CanonicalEventBundleCandidate ownershipLow = ownershipHigh;
  ownershipLow.id = 2;
  const CanonicalEventBundleCandidate *candidates[] = {
      &standalone, &synthetic, &ownershipHigh, &ownershipLow};

  const auto frontier =
      selectCanonicalEventCandidateFrontier(candidates, 3);
  bool passed = check(frontier.size() == 3,
                      "the candidate frontier obeys its explicit bound");
  passed &= check(frontier[0]->id == 2 && frontier[1]->id == 5 &&
                      frontier[2]->id == 9,
                  "frontier truncation uses protocol priority and stable ids");
  passed &= check(selectCanonicalEventCandidateFrontier(candidates, 0).empty(),
                  "a zero-width frontier deterministically selects nothing");
  return passed;
}

bool testAlternatingOwnershipPairVerification() {
  MLIRContext context;
  context.getOrLoadDialect<scf::SCFDialect>();
  OperationState loopState(UnknownLoc::get(&context),
                           scf::ForOp::getOperationName());
  loopState.addRegion();
  Operation *loop = Operation::create(loopState);
  Region firstPathRegion;
  Region secondPathRegion;
  Region *pathRegions[] = {&firstPathRegion, &secondPathRegion};

  CanonicalOwnershipCycle cycle;
  cycle.id = 11;
  cycle.kind = CanonicalOwnershipKind::L1Tile;
  cycle.protocol = CanonicalOwnershipProtocolKind::AlternatingPrefetch;
  cycle.producerPipe = PipelineType::PIPE_MTE2;
  cycle.consumerPipe = PipelineType::PIPE_MTE1;
  cycle.loop = loop;
  cycle.lanes.resize(2);
  cycle.lanes[0].id = 0;
  cycle.lanes[1].id = 1;
  cycle.initialProducers.push_back(0);
  cycle.initialReadyAnchor = {loop, true};
  cycle.initialReadyLane = 0;
  cycle.initiallyFreeLanes.push_back(1);
  for (auto [lane, region] : llvm::enumerate(pathRegions)) {
    CanonicalOwnershipUse use;
    use.lane = lane;
    use.producerLane = 1 - lane;
    use.producers.push_back(lane * 2 + 1);
    use.consumers.push_back(lane * 2 + 2);
    use.writeAcquireAnchor = {loop, true};
    use.readyAnchor = {loop, false};
    use.readAcquireAnchor = {loop, true};
    use.releaseAnchor = {loop, false};
    CanonicalOwnershipPath path;
    path.region = region;
    path.uses.push_back(std::move(use));
    cycle.paths.push_back(std::move(path));
  }

  auto [ready, release] = buildCanonicalOwnershipProtocols(cycle);
  bool passed = check(verifyPair(cycle, ready, release),
                      "a complete alternating ownership protocol is valid");
  passed &= check(!verifyCanonicalAlternatingPathMapping(cycle),
                  "unrelated regions cannot prove an alternating path mapping");

  CanonicalEvent malformed = ready;
  malformed.actions.front().nonEmptyLoopGuard = nullptr;
  passed &= check(!verifyPair(cycle, malformed, release),
                  "an alternating ready prime requires a non-empty guard");

  malformed = release;
  malformed.actions.front().nonEmptyLoopGuard = nullptr;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "an alternating release prime requires a non-empty guard");

  malformed = ready;
  malformed.actions[2].lane.index = 0;
  passed &= check(!verifyPair(cycle, malformed, release),
                  "alternating ready transitions must change lanes");

  malformed = release;
  malformed.actions.pop_back();
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "an alternating release protocol requires its drain");

  malformed = release;
  malformed.actions.back().nonEmptyLoopGuard = nullptr;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "an alternating release drain requires a non-empty guard");

  CanonicalOwnershipCycle malformedCycle = cycle;
  malformedCycle.paths[1].uses.front().producerLane = 1;
  auto [malformedReady, malformedRelease] =
      buildCanonicalOwnershipProtocols(malformedCycle);
  passed &= check(!verifyPair(malformedCycle, malformedReady, malformedRelease),
                  "an alternating cycle requires complementary path transitions");

  const CanonicalEvent *single[] = {&ready};
  passed &= check(!verifyCanonicalOwnershipEventPair(cycle, single),
                  "an alternating protocol requires a complete event pair");
  loop->destroy();
  return passed;
}

} // namespace

int main() {
  const bool passed = testOwnershipPairVerification() &&
                      testOwnershipCandidateTransaction() &&
                      testEventBundleConstruction() &&
                      testSyntheticRoundTripVerification() &&
                      testEventBundleIdentityRestoration() &&
                      testEventBundleConflicts() &&
                      testDiagnosticEventBundleEquivalence() &&
                      testEventBundleAtomicExchange() &&
                      testReservedEventColorOverflow() &&
                      testMechanismPlanScoreOrdering() &&
                      testBarrierActionProfileConstruction() &&
                      testCandidateFrontierTruncation() &&
                      testAlternatingOwnershipPairVerification();
  return passed ? 0 : 1;
}
