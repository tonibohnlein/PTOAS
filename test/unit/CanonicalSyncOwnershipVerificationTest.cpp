// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "CanonicalSyncInternal.h"

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

CanonicalEventLane staticLane(unsigned lane) {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::Static;
  result.index = lane;
  return result;
}

CanonicalEventLane allLanes() {
  CanonicalEventLane result;
  result.kind = CanonicalEventLaneKind::All;
  return result;
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
  CanonicalEvent event;
  event.sourcePipe = cycle.producerPipe;
  event.targetPipe = cycle.consumerPipe;
  event.scopeLoop = cycle.loop;
  event.width = cycle.lanes.size();
  event.ownershipCycle = cycle.id;
  event.ownershipRole = CanonicalOwnershipEventRole::Ready;
  event.ownershipProtocol = true;
  for (const CanonicalOwnershipUse &use : cycle.paths.front().uses) {
    const unsigned set = event.actions.size();
    event.actions.push_back({CanonicalEventActionKind::Set,
                             CanonicalEventActionPhase::Straight,
                             use.readyAnchor, staticLane(use.lane)});
    const unsigned wait = event.actions.size();
    event.actions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Straight,
                             use.readAcquireAnchor, staticLane(use.lane)});
    event.completions.push_back(
        {use.producers.front(), use.consumers.front(), 0, nullptr, set, wait});
  }
  return event;
}

CanonicalEvent makeReleaseEvent(const CanonicalOwnershipCycle &cycle) {
  CanonicalEvent event;
  event.sourcePipe = cycle.consumerPipe;
  event.targetPipe = cycle.producerPipe;
  event.recurrenceLoop = cycle.loop;
  event.scopeLoop = cycle.loop;
  event.iterationDistance = 1;
  event.width = cycle.lanes.size();
  event.ownershipCycle = cycle.id;
  event.ownershipRole = CanonicalOwnershipEventRole::Release;
  event.ownershipProtocol = true;
  event.actions.push_back({CanonicalEventActionKind::Set,
                           CanonicalEventActionPhase::Prime,
                           {cycle.loop, true}, allLanes()});
  for (const CanonicalOwnershipUse &use : cycle.paths.front().uses) {
    const unsigned wait = event.actions.size();
    event.actions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Body,
                             use.writeAcquireAnchor, staticLane(use.lane)});
    const unsigned set = event.actions.size();
    event.actions.push_back({CanonicalEventActionKind::Set,
                             CanonicalEventActionPhase::Body,
                             use.releaseAnchor, staticLane(use.lane)});
    event.completions.push_back({use.consumers.front(), use.producers.front(),
                                 1, cycle.loop, set, wait});
  }
  event.actions.push_back({CanonicalEventActionKind::Wait,
                           CanonicalEventActionPhase::Drain,
                           {cycle.loop, false}, allLanes()});
  return event;
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
  return passed;
}

} // namespace

int main() { return testOwnershipPairVerification() ? 0 : 1; }
