// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#include <iostream>

using namespace mlir;
using namespace mlir::pto;

namespace {

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "CanonicalSyncProtocolVerificationTest failure: " << message
              << '\n';
  }
  return condition;
}

CanonicalOwnershipUse makeUse(unsigned lane, unsigned producerLane,
                              std::size_t producer, std::size_t consumer,
                              Operation *anchor = nullptr) {
  CanonicalOwnershipUse use;
  use.lane = lane;
  use.producerLane = producerLane;
  use.producers.push_back(producer);
  use.consumers.push_back(consumer);
  use.writeAcquireAnchor = {anchor, true};
  use.readyAnchor = {anchor, false};
  use.readAcquireAnchor = {anchor, true};
  use.releaseAnchor = {anchor, false};
  return use;
}

bool verifyPair(const CanonicalOwnershipCycle &cycle,
                const CanonicalEvent &ready, const CanonicalEvent &release) {
  const CanonicalEvent *events[] = {&ready, &release};
  return verifyCanonicalOwnershipEventPair(cycle, events);
}

CanonicalOwnershipCycle makeRoundTripCycle() {
  CanonicalOwnershipCycle cycle;
  cycle.id = 7;
  cycle.kind = CanonicalOwnershipKind::L0Operand;
  cycle.protocol = CanonicalOwnershipProtocolKind::RoundTrip;
  cycle.producerPipe = PipelineType::PIPE_MTE1;
  cycle.consumerPipe = PipelineType::PIPE_M;
  cycle.lanes = {{0, {{AddressSpace::LEFT, 0, 64}}},
                 {1, {{AddressSpace::LEFT, 64, 64}}}};
  CanonicalOwnershipPath path;
  path.uses.push_back(makeUse(0, 0, 0, 1));
  path.uses.push_back(makeUse(1, 1, 2, 3));
  cycle.paths.push_back(std::move(path));
  return cycle;
}

bool testRoundTripVerification() {
  const CanonicalOwnershipCycle cycle = makeRoundTripCycle();
  auto [ready, release] = buildCanonicalOwnershipProtocols(cycle);
  bool passed = check(verifyPair(cycle, ready, release),
                      "accept an exact two-lane ownership lifecycle");
  const CanonicalEvent *reversed[] = {&release, &ready};
  passed &= check(verifyCanonicalOwnershipEventPair(cycle, reversed),
                  "verification is independent of event order");
  const CanonicalEvent *single[] = {&ready};
  passed &= check(!verifyCanonicalOwnershipEventPair(cycle, single),
                  "reject a partial ownership protocol");

  CanonicalEvent malformed = release;
  malformed.width = 1;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "reject a wrong protocol width");
  malformed = ready;
  malformed.completions.pop_back();
  passed &= check(!verifyPair(cycle, malformed, release),
                  "reject incomplete ready coverage");
  malformed = release;
  malformed.actions[1].phase = CanonicalEventActionPhase::Straight;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "reject a malformed release lifecycle");
  return passed;
}

bool testAlternatingVerification() {
  MLIRContext context;
  context.getOrLoadDialect<scf::SCFDialect>();
  OperationState loopState(UnknownLoc::get(&context),
                           scf::ForOp::getOperationName());
  loopState.addRegion();
  Operation *loop = Operation::create(loopState);
  Region firstRegion;
  Region secondRegion;

  CanonicalOwnershipCycle cycle;
  cycle.id = 11;
  cycle.kind = CanonicalOwnershipKind::L1Tile;
  cycle.protocol = CanonicalOwnershipProtocolKind::AlternatingPrefetch;
  cycle.producerPipe = PipelineType::PIPE_MTE2;
  cycle.consumerPipe = PipelineType::PIPE_MTE1;
  cycle.loop = loop;
  cycle.lanes = {{0, {{AddressSpace::MAT, 0, 64}}},
                 {1, {{AddressSpace::MAT, 64, 64}}}};
  cycle.initialProducers.push_back(0);
  cycle.initialReadyAnchor = {loop, true};
  cycle.initialReadyLane = 0;
  cycle.initiallyFreeLanes.push_back(1);
  CanonicalOwnershipPath first;
  first.region = &firstRegion;
  first.uses.push_back(makeUse(0, 1, 1, 2, loop));
  CanonicalOwnershipPath second;
  second.region = &secondRegion;
  second.uses.push_back(makeUse(1, 0, 3, 4, loop));
  cycle.paths = {std::move(first), std::move(second)};

  auto [ready, release] = buildCanonicalOwnershipProtocols(cycle);
  bool passed = check(verifyPair(cycle, ready, release),
                      "accept a complete alternating protocol");
  CanonicalEvent malformed = ready;
  malformed.actions.front().guard = {};
  passed &= check(!verifyPair(cycle, malformed, release),
                  "require the alternating prime guard");
  malformed = release;
  malformed.actions.pop_back();
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "require the alternating drain");
  CanonicalOwnershipCycle wrongTransition = cycle;
  wrongTransition.paths[1].uses.front().producerLane = 1;
  auto [wrongReady, wrongRelease] =
      buildCanonicalOwnershipProtocols(wrongTransition);
  passed &= check(!verifyPair(wrongTransition, wrongReady, wrongRelease),
                  "reject a non-complementary lane transition");
  loop->destroy();
  return passed;
}

bool testBoundaryGuardedVerification() {
  MLIRContext context;
  context.getOrLoadDialect<scf::SCFDialect>();
  OperationState loopState(UnknownLoc::get(&context),
                           scf::ForOp::getOperationName());
  loopState.addRegion();
  Operation *loop = Operation::create(loopState);
  Block *body = new Block();
  loop->getRegion(0).push_back(body);
  OperationState anchorState(UnknownLoc::get(&context),
                             scf::YieldOp::getOperationName());
  Operation *anchor = Operation::create(anchorState);
  body->push_back(anchor);

  CanonicalOwnershipCycle cycle;
  cycle.id = 12;
  cycle.kind = CanonicalOwnershipKind::L0Accumulator;
  cycle.protocol = CanonicalOwnershipProtocolKind::BoundaryGuardedRoundTrip;
  cycle.producerPipe = PipelineType::PIPE_M;
  cycle.consumerPipe = PipelineType::PIPE_FIX;
  cycle.loop = loop;
  cycle.lanes = {{0, {{AddressSpace::ACC, 0, 128}}}};
  CanonicalOwnershipPath path;
  path.region = &loop->getRegion(0);
  path.uses.push_back(makeUse(0, 0, 3, 4, anchor));
  cycle.paths.push_back(std::move(path));

  auto [ready, release] = buildCanonicalOwnershipProtocols(cycle);
  bool passed = check(verifyPair(cycle, ready, release),
                      "accept a boundary-guarded accumulator protocol");
  passed &= check(
      release.actions.size() == 2 &&
          release.actions[0].guard.kind ==
              CanonicalEventExecutionGuardKind::NotFirstIteration &&
          release.actions[1].guard.kind ==
              CanonicalEventExecutionGuardKind::HasSuccessor,
      "encode complementary recurrence-boundary guards");
  CanonicalEvent malformed = release;
  malformed.actions[0].guard.kind =
      CanonicalEventExecutionGuardKind::HasSuccessor;
  passed &= check(!verifyPair(cycle, ready, malformed),
                  "reject a swapped recurrence guard");
  CanonicalOwnershipCycle wrongKind = cycle;
  wrongKind.kind = CanonicalOwnershipKind::L1Tile;
  auto [wrongReady, wrongRelease] = buildCanonicalOwnershipProtocols(wrongKind);
  passed &= check(wrongReady.actions.empty() && wrongRelease.actions.empty(),
                  "restrict boundary guards to accumulators");
  loop->destroy();
  return passed;
}

} // namespace

int main() {
  return testRoundTripVerification() && testAlternatingVerification() &&
                 testBoundaryGuardedVerification()
             ? 0
             : 1;
}
