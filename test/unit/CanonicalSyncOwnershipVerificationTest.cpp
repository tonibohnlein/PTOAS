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
                      testAlternatingOwnershipPairVerification();
  return passed ? 0 : 1;
}
