// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// CanonicalSync protocols are verified before event allocation and emitted
// verbatim afterwards. The verifier deliberately accepts only the finite
// straight-line and single-loop token schemas constructed by this pass.

#include "CanonicalSyncInternal.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::pto;

namespace {

bool isInWhileBeforeRegion(Operation *operation, scf::WhileOp whileOp) {
  Operation *current = operation;
  while (current && current->getParentOp() != whileOp) {
    current = current->getParentOp();
  }
  return current && current->getParentRegion() == &whileOp.getBefore();
}

SmallVector<unsigned, 8> allLanes(unsigned width) {
  SmallVector<unsigned, 8> lanes;
  lanes.reserve(width);
  for (unsigned lane = 0; lane < width; ++lane) {
    lanes.push_back(lane);
  }
  return lanes;
}

bool appendConcreteAction(const CanonicalEventAction &action, unsigned width,
                          SmallVectorImpl<SyncTokenAction> &trace) {
  const SyncTokenActionKind kind = action.kind == CanonicalEventActionKind::Set
                                       ? SyncTokenActionKind::Set
                                       : SyncTokenActionKind::Wait;
  if (action.lane.kind == CanonicalEventLaneKind::All) {
    for (unsigned lane = 0; lane < width; ++lane) {
      trace.push_back({kind, lane});
    }
    return true;
  }
  if (action.lane.kind == CanonicalEventLaneKind::Static) {
    if (action.lane.index >= width) {
      return false;
    }
    trace.push_back({kind, action.lane.index});
    return true;
  }
  if (!action.lane.selector || width <= 1) {
    return false;
  }
  // A dynamic action represents one symmetric lane. Protocol construction
  // separately proves that paired dynamic selectors choose the same lane.
  trace.push_back({kind, 0});
  return true;
}

bool sameLaneAtDistance(const CanonicalEventAction &set,
                        const CanonicalEventAction &wait, unsigned width,
                        Operation *loop, unsigned distance) {
  if (set.lane.kind != wait.lane.kind) {
    return false;
  }
  if (set.lane.kind == CanonicalEventLaneKind::Static) {
    return set.lane.index == wait.lane.index && set.lane.index < width;
  }
  if (set.lane.kind == CanonicalEventLaneKind::All) {
    return false;
  }
  if (!set.lane.selector || !wait.lane.selector || width <= 1) {
    return false;
  }
  if (distance == 0) {
    return compareSlotSSA(set.lane.selector, wait.lane.selector, width) ==
           SlotRelation::kEqual;
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(loop);
  APInt step;
  if (!forOp || !matchPattern(forOp.getStep(), m_ConstantInt(&step)) ||
      !step.isStrictlyPositive() || step.getActiveBits() > 63) {
    return false;
  }
  const std::uint64_t unsignedStep = step.getZExtValue();
  const std::uint64_t maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (distance != 0 && unsignedStep > maximum / distance) {
    return false;
  }
  const auto offset = static_cast<std::int64_t>(unsignedStep * distance);
  return compareSlotSSAWithOffset(set.lane.selector, wait.lane.selector, width,
                                  forOp.getInductionVar(),
                                  offset) == SlotRelation::kEqual;
}

bool verifyActionTrace(ArrayRef<CanonicalEventAction> actions, unsigned width,
                       ArrayRef<unsigned> initial,
                       ArrayRef<unsigned> expected) {
  SmallVector<SyncTokenAction, 16> trace;
  for (const CanonicalEventAction &action : actions) {
    if (!appendConcreteAction(action, width, trace)) {
      return false;
    }
  }
  return verifySyncTokenTrace(
      width, std::vector<unsigned>(initial.begin(), initial.end()),
      std::vector<SyncTokenAction>(trace.begin(), trace.end()),
      std::vector<unsigned>(expected.begin(), expected.end()));
}

void addTrace(CanonicalEvent &event, CanonicalEventTraceKind kind,
              std::initializer_list<unsigned> actions) {
  CanonicalEventTrace trace;
  trace.kind = kind;
  trace.actions.append(actions.begin(), actions.end());
  event.traces.push_back(std::move(trace));
}

bool isActionInRegion(const CanonicalEventAction &action, Region *region) {
  if (!region || !action.anchor.operation) {
    return false;
  }
  Operation *operation = action.anchor.operation;
  return operation->getParentRegion() == region ||
         llvm::any_of(region->getOps(), [&](Operation &candidate) {
           return candidate.isAncestor(operation);
         });
}

bool isAnchoredIn(Operation *operation, Operation *container) {
  return operation && container &&
         (operation == container || container->isAncestor(operation));
}

std::optional<unsigned> getWhileRegionOrder(Operation *operation,
                                            scf::WhileOp whileOp) {
  Operation *child = operation;
  while (child && child->getParentOp() != whileOp) {
    child = child->getParentOp();
  }
  if (!child) {
    return std::nullopt;
  }
  if (child->getParentRegion() == &whileOp.getBefore()) {
    return 0;
  }
  if (child->getParentRegion() == &whileOp.getAfter()) {
    return 1;
  }
  return std::nullopt;
}

bool hasValidPhaseAnchor(const CanonicalEvent &event,
                         const CanonicalEventAction &action) {
  switch (action.phase) {
  case CanonicalEventActionPhase::Straight:
    return !event.recurrenceLoop && !event.forwardDrainLoop;
  case CanonicalEventActionPhase::Prime:
    return event.recurrenceLoop &&
           action.anchor.operation == event.recurrenceLoop &&
           action.anchor.before;
  case CanonicalEventActionPhase::Body:
    return isAnchoredIn(action.anchor.operation, event.recurrenceLoop
                                                     ? event.recurrenceLoop
                                                     : event.forwardDrainLoop);
  case CanonicalEventActionPhase::Condition:
    return isa_and_nonnull<scf::WhileOp>(event.recurrenceLoop
                                             ? event.recurrenceLoop
                                             : event.forwardDrainLoop) &&
           isAnchoredIn(action.anchor.operation, event.recurrenceLoop
                                                     ? event.recurrenceLoop
                                                     : event.forwardDrainLoop);
  case CanonicalEventActionPhase::Drain:
    return (event.recurrenceLoop || event.forwardDrainLoop) &&
           action.anchor.operation == (event.recurrenceLoop
                                           ? event.recurrenceLoop
                                           : event.forwardDrainLoop) &&
           !action.anchor.before;
  }
  return false;
}

bool isPhaseAllowedInTrace(const CanonicalEvent &event,
                           const CanonicalEventTrace &trace,
                           CanonicalEventActionPhase phase) {
  switch (trace.kind) {
  case CanonicalEventTraceKind::Straight:
    return phase == CanonicalEventActionPhase::Straight ||
           (event.forwardDrainLoop &&
            (phase == CanonicalEventActionPhase::Condition ||
             phase == CanonicalEventActionPhase::Body ||
             phase == CanonicalEventActionPhase::Drain));
  case CanonicalEventTraceKind::Prime:
    return phase == CanonicalEventActionPhase::Prime;
  case CanonicalEventTraceKind::Cycle:
    return phase == CanonicalEventActionPhase::Body ||
           phase == CanonicalEventActionPhase::Condition;
  case CanonicalEventTraceKind::Final:
    return phase == CanonicalEventActionPhase::Drain ||
           phase == CanonicalEventActionPhase::Condition ||
           (event.ownershipProtocol && trace.hasExplicitTokenState &&
            phase == CanonicalEventActionPhase::Body);
  }
  return false;
}

bool hasBalancedDynamicCycle(ArrayRef<CanonicalEventAction> actions,
                             unsigned width) {
  SmallVector<const CanonicalEventAction *, 2> dynamic;
  for (const CanonicalEventAction &action : actions) {
    if (action.lane.kind == CanonicalEventLaneKind::Dynamic) {
      dynamic.push_back(&action);
    }
  }
  if (dynamic.empty()) {
    return true;
  }
  return actions.size() == 2 && dynamic.size() == 2 &&
         dynamic[0]->kind == CanonicalEventActionKind::Wait &&
         dynamic[1]->kind == CanonicalEventActionKind::Set &&
         compareSlotSSA(dynamic[0]->lane.selector, dynamic[1]->lane.selector,
                        width) == SlotRelation::kEqual;
}

} // namespace

void CanonicalSyncPlanBuilder::initializeForwardProtocol(
    CanonicalEvent &event) const {
  event.actions.clear();
  event.completions.clear();
  event.traces.clear();
  const CanonicalEventActionPhase setPhase =
      event.forwardDrainLoop ? CanonicalEventActionPhase::Condition
                             : CanonicalEventActionPhase::Straight;
  const CanonicalEventActionPhase waitPhase =
      event.forwardDrainLoop ? CanonicalEventActionPhase::Body
                             : CanonicalEventActionPhase::Straight;
  event.actions.push_back(
      {CanonicalEventActionKind::Set, setPhase, event.setAnchor, {}});
  event.actions.push_back(
      {CanonicalEventActionKind::Wait, waitPhase, event.waitAnchor, {}});
  if (event.forwardDrainLoop) {
    event.actions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Drain,
                             {event.forwardDrainLoop, false},
                             {}});
  }
  event.completions.push_back({event.source, event.target, 0, nullptr, 0, 1});
  if (event.forwardDrainLoop) {
    addTrace(event, CanonicalEventTraceKind::Straight, {0, 1});
    addTrace(event, CanonicalEventTraceKind::Straight, {0, 2});
  } else {
    addTrace(event, CanonicalEventTraceKind::Straight, {0, 1});
  }
  deriveEventInterval(event);
}

void CanonicalSyncPlanBuilder::initializeRecurrenceProtocol(
    CanonicalEvent &event) const {
  event.actions.clear();
  event.completions.clear();
  event.traces.clear();
  CanonicalEventLane all;
  all.kind = CanonicalEventLaneKind::All;
  CanonicalEventLane waitLane;
  CanonicalEventLane setLane;
  if (event.width > 1) {
    waitLane.kind = CanonicalEventLaneKind::Dynamic;
    waitLane.selector = event.waitSlot;
    setLane.kind = CanonicalEventLaneKind::Dynamic;
    setLane.selector = event.setSlot;
  }

  event.actions.push_back({CanonicalEventActionKind::Set,
                           CanonicalEventActionPhase::Prime,
                           {event.recurrenceLoop, true},
                           all});
  CanonicalEventActionPhase waitPhase = CanonicalEventActionPhase::Body;
  bool needsDrain = true;
  if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(event.recurrenceLoop)) {
    const bool setInBefore =
        isInWhileBeforeRegion(event.setAnchor.operation, whileOp);
    const bool waitInBefore =
        isInWhileBeforeRegion(event.waitAnchor.operation, whileOp);
    if (waitInBefore && !setInBefore) {
      waitPhase = CanonicalEventActionPhase::Condition;
      needsDrain = false;
    }
  }
  event.actions.push_back(
      {CanonicalEventActionKind::Wait, waitPhase, event.waitAnchor, waitLane});
  event.actions.push_back({CanonicalEventActionKind::Set,
                           CanonicalEventActionPhase::Body, event.setAnchor,
                           setLane});
  if (needsDrain) {
    event.actions.push_back({CanonicalEventActionKind::Wait,
                             CanonicalEventActionPhase::Drain,
                             {event.recurrenceLoop, false},
                             all});
  }
  event.completions.push_back({event.source, event.target,
                               event.iterationDistance, event.recurrenceLoop, 2,
                               1});
  addTrace(event, CanonicalEventTraceKind::Prime, {0});
  addTrace(event, CanonicalEventTraceKind::Cycle, {1, 2});
  addTrace(event, CanonicalEventTraceKind::Final,
           needsDrain ? std::initializer_list<unsigned>{3}
                      : std::initializer_list<unsigned>{1});
  deriveEventInterval(event);
}

void CanonicalSyncPlanBuilder::deriveEventInterval(
    CanonicalEvent &event) const {
  event.intervalBegin = std::numeric_limits<std::size_t>::max();
  event.intervalEnd = 0;
  for (const CanonicalEventAction &action : event.actions) {
    const std::size_t position = getAnchorPosition(action.anchor);
    event.intervalBegin = std::min(event.intervalBegin, position);
    event.intervalEnd = std::max(event.intervalEnd, position);
  }
  Operation *scope = event.scopeLoop ? event.scopeLoop : event.forwardDrainLoop;
  if (scope) {
    event.intervalBegin =
        std::min(event.intervalBegin, getAnchorPosition({scope, true}));
    event.intervalEnd =
        std::max(event.intervalEnd, getAnchorPosition({scope, false}));
  }
}

bool CanonicalSyncPlanBuilder::verifyEventProtocol(const CanonicalEvent &event,
                                                   bool requireAllocation,
                                                   bool diagnose) const {
  const auto reject = [&](StringRef reason) {
    if (diagnose) {
      llvm::errs() << "invalid canonical event protocol: " << reason << '\n';
    }
    return false;
  };
  if (event.sourcePipe == event.targetPipe || event.width == 0 ||
      event.width > kMaxMultiBufferCount || event.actions.empty() ||
      event.completions.empty()) {
    return reject("invalid domain, width, or empty action/completion list");
  }
  const bool hasOwnershipIdentity = event.ownershipCycle != 0;
  const bool hasOwnershipRole =
      event.ownershipRole != CanonicalOwnershipEventRole::None;
  if (event.ownershipProtocol != hasOwnershipIdentity ||
      event.ownershipProtocol != hasOwnershipRole) {
    return reject("ownership protocol has no unique cycle identity");
  }
  if (requireAllocation) {
    if (event.eventIds.size() != event.width ||
        llvm::any_of(event.eventIds,
                     [&](unsigned id) { return id >= eventIdMax_; })) {
      return reject("physical event ids do not match the logical lanes");
    }
    SmallVector<unsigned, 8> ids(event.eventIds.begin(), event.eventIds.end());
    llvm::sort(ids);
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
      return reject("one protocol assigns one physical id to multiple lanes");
    }
  }

  for (const CanonicalEventAction &action : event.actions) {
    const bool validNonEmptyGuard =
        !action.nonEmptyLoopGuard ||
        (event.ownershipProtocol &&
         action.nonEmptyLoopGuard == event.recurrenceLoop &&
         isa<scf::ForOp>(action.nonEmptyLoopGuard) &&
         (action.phase == CanonicalEventActionPhase::Prime ||
          action.phase == CanonicalEventActionPhase::Drain));
    if (!action.anchor.operation ||
        (action.lane.kind == CanonicalEventLaneKind::Static &&
         action.lane.index >= event.width) ||
        (action.lane.kind == CanonicalEventLaneKind::Dynamic &&
         (!action.lane.selector || event.width <= 1)) ||
        !validNonEmptyGuard) {
      return reject("invalid action anchor or lane selector");
    }
    if (!hasValidPhaseAnchor(event, action) ||
        (action.lane.kind == CanonicalEventLaneKind::All &&
         action.phase != CanonicalEventActionPhase::Prime &&
         action.phase != CanonicalEventActionPhase::Drain)) {
      return reject("event action has an invalid phase anchor or all-lane use");
    }
    const std::size_t position = getAnchorPosition(action.anchor);
    if (position < event.intervalBegin || position > event.intervalEnd) {
      return reject("event lifetime does not enclose an emitted action");
    }
  }
  const bool hasRepresentative = llvm::any_of(
      event.completions, [&](const CanonicalEventCompletion &completion) {
        if (completion.source != event.source ||
            completion.target != event.target ||
            completion.iterationDistance != event.iterationDistance ||
            completion.recurrenceLoop != event.recurrenceLoop ||
            completion.setAction >= event.actions.size() ||
            completion.waitAction >= event.actions.size()) {
          return false;
        }
        const CanonicalAnchor &set = event.actions[completion.setAction].anchor;
        const CanonicalAnchor &wait =
            event.actions[completion.waitAction].anchor;
        return set.operation == event.setAnchor.operation &&
               set.before == event.setAnchor.before &&
               wait.operation == event.waitAnchor.operation &&
               wait.before == event.waitAnchor.before;
      });
  if (!hasRepresentative) {
    return reject("representative event fields do not name a completion");
  }
  for (const CanonicalEventCompletion &completion : event.completions) {
    if (completion.source >= plan_.nodes_.size() ||
        completion.target >= plan_.nodes_.size() ||
        completion.setAction >= event.actions.size() ||
        completion.waitAction >= event.actions.size()) {
      return reject("completion edge has an invalid endpoint or action");
    }
    const CanonicalEventAction &set = event.actions[completion.setAction];
    const CanonicalEventAction &wait = event.actions[completion.waitAction];
    if (set.kind != CanonicalEventActionKind::Set ||
        wait.kind != CanonicalEventActionKind::Wait ||
        plan_.nodes_[completion.source].pipe != event.sourcePipe ||
        plan_.nodes_[completion.target].pipe != event.targetPipe ||
        !sameLaneAtDistance(set, wait, event.width, completion.recurrenceLoop,
                            completion.iterationDistance)) {
      return reject("completion edge is not implemented by its set/wait pair");
    }
    if (completion.iterationDistance == 0) {
      const std::size_t setPosition = getAnchorPosition(set.anchor);
      const std::size_t waitPosition = getAnchorPosition(wait.anchor);
      if (setPosition < plan_.nodes_[completion.source].order * 2 + 1 ||
          waitPosition > plan_.nodes_[completion.target].order * 2) {
        return reject("completion action is not anchored around its endpoint");
      }
      if (!isAnchorGuaranteedForRequirement(set.anchor, completion.source,
                                            completion.target) ||
          !isAnchorGuaranteedForRequirement(wait.anchor, completion.source,
                                            completion.target)) {
        return reject(
            "completion actions are conditional for an unconditional edge");
      }
    } else if (!completion.recurrenceLoop ||
               completion.recurrenceLoop != event.recurrenceLoop ||
               !completion.recurrenceLoop->isAncestor(
                   plan_.nodes_[completion.source].operation) ||
               !completion.recurrenceLoop->isAncestor(
                   plan_.nodes_[completion.target].operation) ||
               set.phase != CanonicalEventActionPhase::Body ||
               (wait.phase != CanonicalEventActionPhase::Body &&
                wait.phase != CanonicalEventActionPhase::Condition)) {
      return reject("recurrence completion actions are outside their loop");
    } else {
      const std::size_t setPosition = getAnchorPosition(set.anchor);
      const std::size_t waitPosition = getAnchorPosition(wait.anchor);
      if (setPosition < plan_.nodes_[completion.source].order * 2 + 1) {
        return reject("recurrence set is not after its source");
      }
      if (waitPosition > plan_.nodes_[completion.target].order * 2) {
        return reject("recurrence wait is not before its target");
      }
      if (!isRecurrenceAnchorGuaranteedForEndpoint(
              set.anchor, 0, completion.source, 0, completion.recurrenceLoop)) {
        return reject("recurrence set is conditional relative to its source");
      }
      if (!isRecurrenceAnchorGuaranteedForEndpoint(
              wait.anchor, completion.iterationDistance, completion.target,
              completion.iterationDistance, completion.recurrenceLoop)) {
        return reject("recurrence wait is conditional relative to its target");
      }
    }
  }

  const SmallVector<unsigned, 8> lanes = allLanes(event.width);
  SmallVector<bool, 16> actionUsed(event.actions.size(), false);
  unsigned primeCount = 0;
  unsigned cycleCount = 0;
  unsigned finalCount = 0;
  bool hasExplicitStateTrace = false;
  for (const CanonicalEventTrace &trace : event.traces) {
    SmallVector<CanonicalEventAction, 16> actions;
    SmallVector<bool, 16> seen(event.actions.size(), false);
    const CanonicalEventAction *previousAction = nullptr;
    for (unsigned actionIndex : trace.actions) {
      if (actionIndex >= event.actions.size() || seen[actionIndex]) {
        return reject("trace contains an invalid or duplicate action");
      }
      seen[actionIndex] = true;
      actionUsed[actionIndex] = true;
      const CanonicalEventAction &action = event.actions[actionIndex];
      const std::size_t position = getAnchorPosition(action.anchor);
      bool ordered = true;
      if (previousAction && trace.kind != CanonicalEventTraceKind::Prime &&
          trace.kind != CanonicalEventTraceKind::Final) {
        if (auto whileOp =
                dyn_cast_or_null<scf::WhileOp>(event.recurrenceLoop)) {
          const std::optional<unsigned> previousRegion =
              getWhileRegionOrder(previousAction->anchor.operation, whileOp);
          const std::optional<unsigned> currentRegion =
              getWhileRegionOrder(action.anchor.operation, whileOp);
          ordered = previousRegion && currentRegion &&
                    (*previousRegion < *currentRegion ||
                     (*previousRegion == *currentRegion &&
                      getAnchorPosition(previousAction->anchor) <= position));
        } else {
          ordered = getAnchorPosition(previousAction->anchor) <= position;
        }
      }
      const bool validTraceAction =
          isPhaseAllowedInTrace(event, trace, action.phase) && ordered &&
          (!trace.controlRegion ||
           isActionInRegion(action, trace.controlRegion));
      if (!validTraceAction) {
        return reject("trace action has an invalid phase, order, or path");
      }
      previousAction = &action;
      actions.push_back(action);
    }
    ArrayRef<unsigned> initial = trace.initialTokens;
    ArrayRef<unsigned> expected = trace.expectedTokens;
    hasExplicitStateTrace |= trace.hasExplicitTokenState;
    if (!trace.hasExplicitTokenState) {
      switch (trace.kind) {
      case CanonicalEventTraceKind::Straight:
        break;
      case CanonicalEventTraceKind::Prime:
        expected = lanes;
        break;
      case CanonicalEventTraceKind::Cycle:
        initial = lanes;
        expected = lanes;
        break;
      case CanonicalEventTraceKind::Final:
        initial = lanes;
        break;
      }
    }
    switch (trace.kind) {
    case CanonicalEventTraceKind::Straight:
      break;
    case CanonicalEventTraceKind::Prime:
      ++primeCount;
      break;
    case CanonicalEventTraceKind::Cycle:
      ++cycleCount;
      break;
    case CanonicalEventTraceKind::Final:
      ++finalCount;
      break;
    }
    if ((trace.kind == CanonicalEventTraceKind::Cycle &&
         !hasBalancedDynamicCycle(actions, event.width)) ||
        !verifyActionTrace(actions, event.width, initial, expected)) {
      return reject("event token trace is not balanced");
    }
  }
  if (llvm::any_of(actionUsed, [](bool used) { return !used; })) {
    return reject("event action is not covered by a verified trace");
  }
  if (event.recurrenceLoop) {
    const bool validFinalCount =
        finalCount == 1 ||
        (event.ownershipProtocol && hasExplicitStateTrace && finalCount != 0);
    return (primeCount == 1 && cycleCount != 0 && validFinalCount) ||
           reject(
               "cyclic protocol lacks one prime, cycles, or one final trace");
  }
  return (primeCount == 0 && cycleCount == 0 && finalCount == 0 &&
          !event.traces.empty()) ||
         reject("straight protocol contains cyclic lifecycle traces");
}

LogicalResult
CanonicalSyncPlanBuilder::verifyEventProtocols(ArrayRef<CanonicalEvent> events,
                                               bool requireAllocation,
                                               bool diagnose) const {
  std::map<std::size_t, unsigned> protocolBundles;
  std::map<std::size_t, SmallVector<const CanonicalEvent *, 2>>
      ownershipEvents;
  for (const CanonicalEvent &event : events) {
    if (!verifyEventProtocol(event, requireAllocation, diagnose)) {
      return failure();
    }
    if (event.protocolBundle != 0) {
      ++protocolBundles[event.protocolBundle];
    }
    if (event.ownershipCycle != 0) {
      ownershipEvents[event.ownershipCycle].push_back(&event);
    }
  }
  const auto incompletePair =
      [](const auto &entry) { return entry.second != 2; };
  const bool hasIncompleteProtocol =
      llvm::any_of(protocolBundles, incompletePair);
  if (hasIncompleteProtocol) {
    if (diagnose) {
      llvm::errs() << "invalid canonical event protocol bundle\n";
    }
    return failure();
  }
  for (const auto &entry : ownershipEvents) {
    const CanonicalOwnershipCycle *cycle = nullptr;
    bool duplicateCycleId = false;
    for (const CanonicalOwnershipCycle &candidate : plan_.ownershipCycles_) {
      if (candidate.id != entry.first) {
        continue;
      }
      duplicateCycleId = cycle != nullptr;
      cycle = &candidate;
    }
    const bool hasKnownCycle = cycle && !duplicateCycleId;
    const bool hasPair = entry.second.size() == 2;
    if (!hasKnownCycle || !hasPair) {
      if (diagnose) {
        llvm::errs() << "invalid canonical ownership event pair\n";
      }
      return failure();
    }
    const bool validPair =
        verifyCanonicalOwnershipEventPair(*cycle, entry.second);
    if (!validPair) {
      if (diagnose) {
        llvm::errs() << "invalid canonical ownership event lifecycle\n";
      }
      return failure();
    }
  }
  if (!requireAllocation) {
    return success();
  }
  std::map<CanonicalEventDomainKey, std::vector<SyncAllocatedInterval>>
      allocations;
  for (const CanonicalEvent &event : events) {
    auto &domain = allocations[{event.sourcePipe, event.targetPipe}];
    for (unsigned eventId : event.eventIds) {
      domain.push_back({{event.intervalBegin, event.intervalEnd}, eventId});
    }
  }
  for (const auto &entry : allocations) {
    std::vector<unsigned> reserved;
    auto reservedIt = reservedIds_.find(entry.first);
    if (reservedIt != reservedIds_.end()) {
      reserved.assign(reservedIt->second.begin(), reservedIt->second.end());
    }
    if (!verifySyncIntervalAllocation(eventIdMax_, reserved, entry.second)) {
      if (diagnose) {
        llvm::errs() << "invalid canonical event allocation in domain\n";
      }
      return failure();
    }
  }
  return success();
}
