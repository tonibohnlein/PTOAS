// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverProtocolInternal.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::sync_cover_protocol_detail;

namespace {

enum class ActionKind : std::uint8_t { Set, Wait };
enum class ActionSegment : std::uint8_t { Entry, Body, Exit };

struct DynamicAction {
  ActionKind kind = ActionKind::Set;
  ActionSegment segment = ActionSegment::Body;
  SyncCoverProtocolChannelId channel = 0;
  std::size_t lane = 0;
  /// Static explicit-recipe action that instantiated this dynamic action.
  /// Legacy channel transfers have no static action identity.
  std::optional<std::size_t> staticAction;
  std::size_t iteration = 0;
  std::uint32_t resource = 0;
  SyncCoverTimelinePosition position = 0;
  std::size_t ordinal = 0;
  std::optional<std::size_t> mate;
  std::size_t sequence = 0;
};

struct DynamicProtocol {
  std::vector<DynamicAction> actions;
  std::vector<std::vector<std::size_t>> successors;
  std::vector<std::vector<std::vector<std::size_t>>> sets;
  std::vector<std::vector<std::vector<std::size_t>>> waits;
};

using SupplyWitnesses = std::vector<std::vector<bool>>;

bool phaseIsActive(const SyncCoverEventTransfer &transfer, std::size_t phase) {
  return transfer.activePhases.empty() ||
         std::binary_search(transfer.activePhases.begin(),
                            transfer.activePhases.end(), phase);
}

bool phaseIsActive(const SyncCoverProtocolAction &action, std::size_t phase) {
  return action.activePhases.empty() ||
         std::binary_search(action.activePhases.begin(),
                            action.activePhases.end(), phase);
}

bool temporalGuardIsActive(SyncCoverProtocolActionGuard guard,
                           std::size_t iteration, std::size_t tripCount) {
  switch (guard) {
  case SyncCoverProtocolActionGuard::Always:
    return true;
  case SyncCoverProtocolActionGuard::LoopNonEmpty:
    return tripCount != 0;
  case SyncCoverProtocolActionGuard::LoopEmpty:
    return tripCount == 0;
  case SyncCoverProtocolActionGuard::FirstIteration:
    return tripCount != 0 && iteration == 0;
  case SyncCoverProtocolActionGuard::NotFirstIteration:
    return iteration != 0;
  case SyncCoverProtocolActionGuard::HasSuccessor:
    return iteration < tripCount && iteration + 1 < tripCount;
  }
  return false;
}

ActionKind dynamicActionKind(SyncCoverProtocolActionKind kind) {
  return kind == SyncCoverProtocolActionKind::Set ? ActionKind::Set
                                                  : ActionKind::Wait;
}

ActionSegment dynamicActionSegment(SyncCoverProtocolActionSegment segment) {
  switch (segment) {
  case SyncCoverProtocolActionSegment::Entry:
    return ActionSegment::Entry;
  case SyncCoverProtocolActionSegment::Body:
    return ActionSegment::Body;
  case SyncCoverProtocolActionSegment::Exit:
    return ActionSegment::Exit;
  }
  return ActionSegment::Body;
}

std::size_t logarithmicLookupWork(std::size_t size) {
  std::size_t work = 1;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++work;
  }
  return work;
}

bool checkedProduct(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow =
      left != 0 && right > std::numeric_limits<std::size_t>::max() / left;
  if (overflow) {
    return false;
  }
  result = left * right;
  return true;
}

bool reserveSortWork(std::size_t size, SyncCoverCoverageWorkBudget *budget) {
  std::size_t logarithm = 0;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++logarithm;
  }
  const bool additionOverflow =
      logarithm == std::numeric_limits<std::size_t>::max();
  std::size_t work = 0;
  return !additionOverflow && checkedProduct(size, logarithm + 1, work) &&
         checkedProduct(work, 3, work) && consumeWork(budget, work);
}

bool chargeRearmLookup(SyncCoverProtocolLimits limits,
                       SyncCoverProtocolStatistics &statistics,
                       SyncCoverCoverageWorkBudget *workBudget,
                       std::size_t amount = 1) {
  const bool limitExceeded =
      amount >
      limits.maximumRearmLookupWork -
          std::min(statistics.rearmLookupWork, limits.maximumRearmLookupWork);
  if (limitExceeded || !consumeWork(workBudget, amount)) {
    return false;
  }
  statistics.rearmLookupWork += amount;
  return true;
}

std::size_t addAction(DynamicProtocol &dynamic, DynamicAction action,
                      SyncCoverProtocolLimits limits,
                      SyncCoverProtocolStatistics &statistics,
                      SyncCoverCoverageWorkBudget *workBudget) {
  const bool actionLimitReached =
      dynamic.actions.size() == limits.maximumDynamicActions;
  const bool aggregateLimitReached =
      statistics.totalDynamicActions == limits.maximumTotalDynamicActions;
  if (actionLimitReached || aggregateLimitReached || !consumeWork(workBudget)) {
    return std::numeric_limits<std::size_t>::max();
  }
  const std::size_t id = dynamic.actions.size();
  const std::size_t channel = action.channel;
  const std::size_t lane = action.lane;
  const bool isSet = action.kind == ActionKind::Set;
  dynamic.actions.push_back(std::move(action));
  (isSet ? dynamic.sets : dynamic.waits)[channel][lane].push_back(id);
  ++statistics.totalDynamicActions;
  return id;
}

bool addEdge(DynamicProtocol &dynamic, std::size_t source, std::size_t target,
             SyncCoverProtocolLimits limits,
             SyncCoverProtocolStatistics &statistics,
             SyncCoverCoverageWorkBudget *workBudget) {
  const bool invalidEdge = source == target ||
                           source >= dynamic.actions.size() ||
                           target >= dynamic.actions.size();
  const bool edgeLimitReached =
      statistics.automatonEdges == limits.maximumAutomatonEdges;
  const bool workUnavailable = !consumeWork(workBudget);
  if (invalidEdge || edgeLimitReached || workUnavailable) {
    return false;
  }
  dynamic.successors[source].push_back(target);
  ++statistics.automatonEdges;
  return true;
}

std::tuple<std::size_t, unsigned, std::size_t, SyncCoverTimelinePosition,
           std::size_t, unsigned, SyncCoverProtocolChannelId>
orderKey(const DynamicAction &action) {
  const unsigned segment = static_cast<unsigned>(action.segment);
  const unsigned kind = action.kind == ActionKind::Wait ? 0U : 1U;
  return {action.sequence, segment, action.iteration, action.position,
          action.ordinal,  kind,    action.channel};
}

bool isLifetimeEntry(const SyncCoverEventProtocol &protocol,
                     const SyncCoverProtocolAction &action) {
  return protocol.lifetimeScope &&
         action.point.anchor.kind == SyncCoverAnchorKind::ScopeEntry &&
         action.point.anchor.scope == *protocol.lifetimeScope;
}

bool isLifetimeExit(const SyncCoverEventProtocol &protocol,
                    const SyncCoverProtocolAction &action) {
  return protocol.lifetimeScope &&
         action.point.anchor.kind == SyncCoverAnchorKind::ScopeExit &&
         action.point.anchor.scope == *protocol.lifetimeScope;
}

SyncCoverProtocolError
buildActions(const SyncCoverEventProtocol &protocol,
             const ResolvedProtocol &resolved, std::size_t tripCount,
             std::size_t invocationSequence, std::size_t lifetimeExitSequence,
             bool includeInvocation, bool includeLifetimeEntry,
             bool includeLifetimeExit, SyncCoverProtocolLimits limits,
             DynamicProtocol &dynamic, SyncCoverProtocolStatistics &statistics,
             SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t laneWork = protocol.channels.size();
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    if (channel.width >
        (std::numeric_limits<std::size_t>::max() - laneWork) / 2) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    laneWork += channel.width * 2;
  }
  const bool laneWorkLimitExceeded =
      laneWork > limits.maximumLaneInitializationWork -
                     std::min(statistics.laneInitializationWork,
                              limits.maximumLaneInitializationWork);
  if (laneWorkLimitExceeded || !consumeWork(workBudget, laneWork)) {
    return workBudget && workBudget->exhausted
               ? SyncCoverProtocolError::WorkLimitExceeded
               : SyncCoverProtocolError::LimitExceeded;
  }
  statistics.laneInitializationWork += laneWork;
  dynamic.sets.resize(protocol.channels.size());
  dynamic.waits.resize(protocol.channels.size());
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    dynamic.sets[channel.id].resize(channel.width);
    dynamic.waits[channel.id].resize(channel.width);
  }

  if (protocol.loop) {
    for (const ResolvedChannel &resolvedChannel : resolved.channels) {
      const SyncCoverEventChannel &channel = *resolvedChannel.description;
      if (!resolvedChannel.actions.empty()) {
        for (const ResolvedAction &resolvedAction : resolvedChannel.actions) {
          const SyncCoverProtocolAction &action = resolvedAction.description;
          const bool lifetimeAction = isLifetimeEntry(protocol, action);
          if (action.segment != SyncCoverProtocolActionSegment::Entry ||
              (lifetimeAction ? !includeLifetimeEntry : !includeInvocation) ||
              !temporalGuardIsActive(action.guard, 0, tripCount)) {
            continue;
          }
          if (addAction(dynamic,
                        {dynamicActionKind(action.kind),
                         dynamicActionSegment(action.segment), channel.id,
                         action.lane, action.id, 0, action.point.resource,
                         resolvedAction.position, action.point.ordinal,
                         std::nullopt, lifetimeAction ? 0 : invocationSequence},
                        limits, statistics, workBudget) ==
              std::numeric_limits<std::size_t>::max()) {
            return workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
          }
        }
        continue;
      }
      if (!includeInvocation ||
          channel.flow != SyncCoverEventChannelFlow::LoopCarry) {
        continue;
      }
      for (std::size_t lane = 0; lane < channel.width; ++lane) {
        if (addAction(dynamic,
                      {ActionKind::Set, ActionSegment::Entry, channel.id, lane,
                       std::nullopt, 0, channel.set.resource, 0, lane,
                       std::nullopt, invocationSequence},
                      limits, statistics,
                      workBudget) == std::numeric_limits<std::size_t>::max()) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
      }
    }
  }

  std::size_t phase = resolved.initialPhase;
  for (std::size_t iteration = 0; includeInvocation && iteration < tripCount;
       ++iteration) {
    for (const ResolvedChannel &resolvedChannel : resolved.channels) {
      const SyncCoverEventChannel &channel = *resolvedChannel.description;
      if (!resolvedChannel.actions.empty()) {
        for (const ResolvedAction &resolvedAction : resolvedChannel.actions) {
          const SyncCoverProtocolAction &action = resolvedAction.description;
          if (action.segment != SyncCoverProtocolActionSegment::Body ||
              !phaseIsActive(action, phase) ||
              !temporalGuardIsActive(action.guard, iteration, tripCount)) {
            continue;
          }
          if (addAction(dynamic,
                        {dynamicActionKind(action.kind),
                         dynamicActionSegment(action.segment), channel.id,
                         action.lane, action.id, iteration,
                         action.point.resource, resolvedAction.position,
                         action.point.ordinal, std::nullopt,
                         invocationSequence},
                        limits, statistics, workBudget) ==
              std::numeric_limits<std::size_t>::max()) {
            return workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
          }
        }
        continue;
      }
      for (const ResolvedTransfer &resolvedTransfer :
           resolvedChannel.transfers) {
        const SyncCoverEventTransfer &transfer = resolvedTransfer.description;
        if (!consumeWork(workBudget,
                         logarithmicLookupWork(transfer.activePhases.size()))) {
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        if (!phaseIsActive(transfer, phase)) {
          continue;
        }
        const bool explicitTransfer = !channel.transfers.empty();
        const std::size_t setLane =
            explicitTransfer
                ? transfer.setLane
                : (protocol.loop ? protocol.loop->laneByPhase[phase] : 0);
        const std::size_t waitLane =
            explicitTransfer
                ? transfer.waitLane
                : (protocol.loop ? protocol.loop->laneByPhase[phase] : 0);
        const auto append =
            [&](ActionKind kind, std::size_t lane, std::uint32_t resource,
                SyncCoverTimelinePosition position, std::size_t ordinal) {
              return addAction(dynamic,
                               {kind, ActionSegment::Body, channel.id, lane,
                                std::nullopt, iteration, resource, position,
                                ordinal, std::nullopt, invocationSequence},
                               limits, statistics, workBudget);
            };
        if (channel.flow == SyncCoverEventChannelFlow::LoopCarry &&
            append(ActionKind::Wait, waitLane, transfer.wait.resource,
                   resolvedTransfer.waitPosition, transfer.wait.ordinal) ==
                std::numeric_limits<std::size_t>::max()) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
        if (append(ActionKind::Set, setLane, transfer.set.resource,
                   resolvedTransfer.setPosition, transfer.set.ordinal) ==
            std::numeric_limits<std::size_t>::max()) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
        if (channel.flow != SyncCoverEventChannelFlow::LoopCarry &&
            append(ActionKind::Wait, waitLane, transfer.wait.resource,
                   resolvedTransfer.waitPosition, transfer.wait.ordinal) ==
                std::numeric_limits<std::size_t>::max()) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
      }
    }
    if (protocol.loop) {
      phase = resolved.nextPhase[phase];
    }
  }

  if (protocol.loop) {
    for (const ResolvedChannel &resolvedChannel : resolved.channels) {
      const SyncCoverEventChannel &channel = *resolvedChannel.description;
      if (!resolvedChannel.actions.empty()) {
        for (const ResolvedAction &resolvedAction : resolvedChannel.actions) {
          const SyncCoverProtocolAction &action = resolvedAction.description;
          const bool lifetimeAction = isLifetimeExit(protocol, action);
          if (action.segment != SyncCoverProtocolActionSegment::Exit ||
              (lifetimeAction ? !includeLifetimeExit : !includeInvocation) ||
              !temporalGuardIsActive(action.guard, tripCount, tripCount)) {
            continue;
          }
          if (addAction(
                  dynamic,
                  {dynamicActionKind(action.kind),
                   dynamicActionSegment(action.segment), channel.id,
                   action.lane, action.id, tripCount, action.point.resource,
                   resolvedAction.position, action.point.ordinal, std::nullopt,
                   lifetimeAction ? lifetimeExitSequence : invocationSequence},
                  limits, statistics,
                  workBudget) == std::numeric_limits<std::size_t>::max()) {
            return workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
          }
        }
        continue;
      }
      if (!includeInvocation ||
          channel.flow != SyncCoverEventChannelFlow::LoopCarry) {
        continue;
      }
      for (std::size_t lane = 0; lane < channel.width; ++lane) {
        if (addAction(dynamic,
                      {ActionKind::Wait, ActionSegment::Exit, channel.id, lane,
                       std::nullopt, tripCount, channel.wait.resource,
                       std::numeric_limits<SyncCoverTimelinePosition>::max(),
                       lane, std::nullopt, invocationSequence},
                      limits, statistics,
                      workBudget) == std::numeric_limits<std::size_t>::max()) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
      }
    }
  }
  dynamic.successors.resize(dynamic.actions.size());
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError addIssueOrder(DynamicProtocol &dynamic,
                                     SyncCoverProtocolLimits limits,
                                     SyncCoverProtocolStatistics &statistics,
                                     SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::size_t> order(dynamic.actions.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  if (!reserveSortWork(order.size(), workBudget)) {
    return workBudget && workBudget->exhausted
               ? SyncCoverProtocolError::WorkLimitExceeded
               : SyncCoverProtocolError::LimitExceeded;
  }
  std::sort(order.begin(), order.end(),
            [&](std::size_t left, std::size_t right) {
              const DynamicAction &leftAction = dynamic.actions[left];
              const DynamicAction &rightAction = dynamic.actions[right];
              if (leftAction.resource != rightAction.resource) {
                return leftAction.resource < rightAction.resource;
              }
              const auto leftKey = orderKey(leftAction);
              const auto rightKey = orderKey(rightAction);
              return leftKey != rightKey ? leftKey < rightKey : left < right;
            });
  for (std::size_t index = 1; index < order.size(); ++index) {
    const std::size_t previous = order[index - 1];
    const std::size_t current = order[index];
    if (dynamic.actions[previous].resource ==
            dynamic.actions[current].resource &&
        !addEdge(dynamic, previous, current, limits, statistics, workBudget)) {
      return workBudget && workBudget->exhausted
                 ? SyncCoverProtocolError::WorkLimitExceeded
                 : SyncCoverProtocolError::LimitExceeded;
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError pairTokens(const SyncCoverEventProtocol &protocol,
                                  DynamicProtocol &dynamic,
                                  SyncCoverProtocolLimits limits,
                                  SyncCoverProtocolStatistics &statistics,
                                  SyncCoverCoverageWorkBudget *workBudget) {
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    for (std::size_t lane = 0; lane < channel.width; ++lane) {
      std::vector<std::size_t> &sets = dynamic.sets[channel.id][lane];
      std::vector<std::size_t> &waits = dynamic.waits[channel.id][lane];
      const bool tokenImbalance = sets.size() != waits.size();
      if (tokenImbalance) {
        return SyncCoverProtocolError::InvalidTokenLifecycle;
      }
      for (std::size_t index = 0; index < sets.size(); ++index) {
        const std::size_t set = sets[index];
        const std::size_t wait = waits[index];
        dynamic.actions[set].mate = wait;
        dynamic.actions[wait].mate = set;
        if (!addEdge(dynamic, set, wait, limits, statistics, workBudget)) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
      }
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError recordSupplyWitnesses(
    const SyncCoverEventProtocol &protocol, const DynamicProtocol &dynamic,
    SupplyWitnesses &witnesses, SyncCoverCoverageWorkBudget *workBudget) {
  if (witnesses.size() != protocol.channels.size()) {
    return SyncCoverProtocolError::InvalidProtocol;
  }
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    if (channel.actions.empty()) {
      continue;
    }
    if (channel.id >= witnesses.size() ||
        witnesses[channel.id].size() != channel.supplies.size()) {
      return SyncCoverProtocolError::InvalidProtocol;
    }
    for (std::size_t supplyIndex = 0; supplyIndex < channel.supplies.size();
         ++supplyIndex) {
      const SyncCoverProtocolSupply &supply = channel.supplies[supplyIndex];
      if (supply.kind == SyncCoverProtocolSupplyKind::CompletionExport) {
        continue;
      }
      const SyncCoverProtocolAction &setDescription =
          channel.actions[supply.setAction];
      const SyncCoverProtocolAction &waitDescription =
          channel.actions[supply.waitAction];
      const std::vector<std::size_t> &sets =
          dynamic.sets[channel.id][setDescription.lane];
      for (std::size_t setId : sets) {
        if (!consumeWork(workBudget)) {
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        const DynamicAction &set = dynamic.actions[setId];
        if (set.staticAction != supply.setAction || !set.mate) {
          continue;
        }
        const DynamicAction &wait = dynamic.actions[*set.mate];
        if (wait.staticAction != supply.waitAction ||
            wait.lane != waitDescription.lane) {
          continue;
        }
        bool displacementMatches = false;
        if (supply.distanceScope) {
          // Hierarchical recipes are expanded as a sequence of child-loop
          // invocations owned by the lifetime loop. A parent-distance supply
          // is real only when the named Set token is consumed by the named
          // Wait exactly d invocations later.
          displacementMatches =
              protocol.lifetimeScope &&
              *supply.distanceScope == *protocol.lifetimeScope &&
              wait.sequence >= set.sequence &&
              wait.sequence - set.sequence == supply.distance;
        } else {
          displacementMatches =
              wait.sequence == set.sequence &&
              wait.iteration >= set.iteration &&
              wait.iteration - set.iteration == supply.distance;
          // A lifetime-entry prime uses sequence zero and is consumed by the
          // first child invocation. Its logical local distance is zero.
          displacementMatches |= protocol.lifetimeScope &&
                                 supply.distance == 0 &&
                                 set.segment == ActionSegment::Entry &&
                                 set.sequence == 0 && wait.sequence == 1;
        }
        if (displacementMatches) {
          witnesses[channel.id][supplyIndex] = true;
          break;
        }
      }
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError addRearmProofs(const SyncCoverEventProtocol &protocol,
                                      DynamicProtocol &dynamic,
                                      SyncCoverProtocolLimits limits,
                                      SyncCoverProtocolStatistics &statistics,
                                      SyncCoverCoverageWorkBudget *workBudget) {
  for (const SyncCoverProtocolRearmProof &proof : protocol.rearmProofs) {
    const SyncCoverEventChannel &from =
        protocol.channels[proof.fromWaitChannel];
    const SyncCoverEventChannel &to = protocol.channels[proof.toSetChannel];
    const std::size_t commonWidth = std::min(from.width, to.width);
    const bool incidenceLimitExceeded =
        commonWidth > limits.maximumRearmProofLaneIncidences -
                          std::min(statistics.rearmProofLaneIncidences,
                                   limits.maximumRearmProofLaneIncidences);
    if (incidenceLimitExceeded) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    statistics.rearmProofLaneIncidences += commonWidth;
    for (std::size_t lane = 0; lane < commonWidth; ++lane) {
      if (!chargeRearmLookup(limits, statistics, workBudget)) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      const std::vector<std::size_t> &waits = dynamic.waits[from.id][lane];
      const std::vector<std::size_t> &sets = dynamic.sets[to.id][lane];
      for (std::size_t wait : waits) {
        std::size_t lookupWork = 1;
        for (std::size_t value = sets.size(); value > 1;
             value = (value + 1) / 2) {
          ++lookupWork;
        }
        if (!chargeRearmLookup(limits, statistics, workBudget, lookupWork)) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
        const DynamicAction &waitAction = dynamic.actions[wait];
        if (waitAction.segment != ActionSegment::Body) {
          continue;
        }
        if (proof.iterationDistance >
            std::numeric_limits<std::size_t>::max() - waitAction.iteration) {
          return SyncCoverProtocolError::LimitExceeded;
        }
        const std::size_t targetIteration =
            waitAction.iteration + proof.iterationDistance;
        const auto found = std::lower_bound(
            sets.begin(), sets.end(), targetIteration,
            [&](std::size_t set, std::size_t iteration) {
              return dynamic.actions[set].iteration < iteration;
            });
        const bool matchingSet =
            found != sets.end() &&
            dynamic.actions[*found].segment == ActionSegment::Body &&
            dynamic.actions[*found].iteration == targetIteration;
        const bool edgeFailed =
            matchingSet &&
            !addEdge(dynamic, wait, *found, limits, statistics, workBudget);
        if (edgeFailed) {
          return workBudget && workBudget->exhausted
                     ? SyncCoverProtocolError::WorkLimitExceeded
                     : SyncCoverProtocolError::LimitExceeded;
        }
      }
    }
  }
  return SyncCoverProtocolError::None;
}

bool topologicalOrder(const DynamicProtocol &dynamic,
                      std::vector<std::size_t> &order,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::size_t> indegree(dynamic.actions.size());
  for (const std::vector<std::size_t> &successors : dynamic.successors) {
    for (std::size_t target : successors) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      ++indegree[target];
    }
  }
  std::vector<std::size_t> ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (indegree[index] == 0) {
      ready.push_back(index);
    }
  }
  while (!ready.empty()) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    const std::size_t node = ready.back();
    ready.pop_back();
    order.push_back(node);
    for (std::size_t target : dynamic.successors[node]) {
      if (!consumeWork(workBudget)) {
        return false;
      }
      if (--indegree[target] == 0) {
        ready.push_back(target);
      }
    }
  }
  return order.size() == dynamic.actions.size();
}

SyncCoverProtocolError verifyRearm(const SyncCoverEventProtocol &protocol,
                                   const DynamicProtocol &dynamic,
                                   const std::vector<std::size_t> &order,
                                   SyncCoverProtocolLimits limits,
                                   SyncCoverProtocolStatistics &statistics,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  std::vector<std::pair<std::size_t, std::size_t>> queries;
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    for (std::size_t lane = 0; lane < channel.width; ++lane) {
      const std::vector<std::size_t> &sets = dynamic.sets[channel.id][lane];
      for (std::size_t index = 1; index < sets.size(); ++index) {
        const std::optional<std::size_t> previousWait =
            dynamic.actions[sets[index - 1]].mate;
        if (!previousWait) {
          return SyncCoverProtocolError::InvalidTokenLifecycle;
        }
        const bool queryLimitReached =
            queries.size() == limits.maximumRearmQueries;
        if (queryLimitReached) {
          return SyncCoverProtocolError::LimitExceeded;
        }
        queries.emplace_back(*previousWait, sets[index]);
      }
    }
  }
  const bool aggregateQueryLimitExceeded =
      queries.size() >
      limits.maximumRearmQueries -
          std::min(statistics.rearmQueries, limits.maximumRearmQueries);
  if (aggregateQueryLimitExceeded) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  statistics.rearmQueries += queries.size();
  if (queries.empty()) {
    return SyncCoverProtocolError::None;
  }

  const std::size_t wordsPerNode = (queries.size() + 63) / 64;
  const bool wordOverflow =
      !dynamic.actions.empty() &&
      wordsPerNode >
          std::numeric_limits<std::size_t>::max() / dynamic.actions.size();
  const std::size_t totalWords = wordOverflow
                                     ? std::numeric_limits<std::size_t>::max()
                                     : wordsPerNode * dynamic.actions.size();
  if (wordOverflow || totalWords > limits.maximumReachabilityWords) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  std::vector<std::uint64_t> reachesTargets(totalWords);
  for (std::size_t query = 0; query < queries.size(); ++query) {
    reachesTargets[queries[query].second * wordsPerNode + query / 64] |=
        std::uint64_t{1} << (query % 64);
  }
  for (auto node = order.rbegin(); node != order.rend(); ++node) {
    std::uint64_t *destination = reachesTargets.data() + *node * wordsPerNode;
    for (std::size_t successor : dynamic.successors[*node]) {
      const std::uint64_t *source =
          reachesTargets.data() + successor * wordsPerNode;
      if (wordsPerNode > limits.maximumReachabilityWork -
                             std::min(statistics.reachabilityWork,
                                      limits.maximumReachabilityWork) ||
          !consumeWork(workBudget, wordsPerNode)) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      statistics.reachabilityWork += wordsPerNode;
      for (std::size_t word = 0; word < wordsPerNode; ++word) {
        destination[word] |= source[word];
      }
    }
  }
  for (std::size_t query = 0; query < queries.size(); ++query) {
    const std::size_t source = queries[query].first;
    const bool reached = (reachesTargets[source * wordsPerNode + query / 64] &
                          (std::uint64_t{1} << (query % 64))) != 0;
    if (!reached) {
      return SyncCoverProtocolError::InvalidTokenLifecycle;
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError recordCompletionExportWitnesses(
    const SyncCoverEventProtocol &protocol, const DynamicProtocol &dynamic,
    const std::vector<std::size_t> &order, SyncCoverProtocolLimits limits,
    SyncCoverProtocolStatistics &statistics, SupplyWitnesses &witnesses,
    SyncCoverCoverageWorkBudget *workBudget) {
  struct Query {
    std::size_t source = 0;
    std::size_t target = 0;
    std::size_t channel = 0;
    std::size_t supply = 0;
  };
  std::vector<Query> queries;
  std::vector<std::vector<bool>> invalid;
  invalid.reserve(witnesses.size());
  for (const std::vector<bool> &channel : witnesses) {
    invalid.emplace_back(channel.size(), false);
  }
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    for (std::size_t supplyIndex = 0; supplyIndex < channel.supplies.size();
         ++supplyIndex) {
      const SyncCoverProtocolSupply &supply = channel.supplies[supplyIndex];
      if (supply.kind != SyncCoverProtocolSupplyKind::CompletionExport) {
        continue;
      }
      const SyncCoverProtocolAction &setDescription =
          channel.actions[supply.setAction];
      const SyncCoverProtocolAction &waitDescription =
          channel.actions[supply.waitAction];
      const std::vector<std::size_t> &sets =
          dynamic.sets[channel.id][setDescription.lane];
      const std::vector<std::size_t> &waits =
          dynamic.waits[channel.id][waitDescription.lane];
      for (std::size_t set : sets) {
        if (dynamic.actions[set].staticAction != supply.setAction) {
          continue;
        }
        for (std::size_t wait : waits) {
          if (!consumeWork(workBudget)) {
            return SyncCoverProtocolError::WorkLimitExceeded;
          }
          const DynamicAction &setAction = dynamic.actions[set];
          const DynamicAction &waitAction = dynamic.actions[wait];
          if (waitAction.staticAction != supply.waitAction) {
            continue;
          }
          const bool matchingDisplacement =
              supply.distanceScope
                  ? waitAction.sequence >= setAction.sequence &&
                        waitAction.sequence - setAction.sequence ==
                            supply.distance
                  : waitAction.sequence == setAction.sequence &&
                        waitAction.iteration >= setAction.iteration &&
                        waitAction.iteration - setAction.iteration ==
                            supply.distance;
          if (!matchingDisplacement) {
            continue;
          }
          if (queries.size() == limits.maximumCompletionExportQueries) {
            return SyncCoverProtocolError::LimitExceeded;
          }
          queries.push_back({set, wait, channel.id, supplyIndex});
          witnesses[channel.id][supplyIndex] = true;
        }
      }
    }
  }
  if (queries.size() > limits.maximumCompletionExportQueries -
                           std::min(statistics.completionExportQueries,
                                    limits.maximumCompletionExportQueries)) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  statistics.completionExportQueries += queries.size();
  if (queries.empty()) {
    return SyncCoverProtocolError::None;
  }
  const std::size_t wordsPerNode = (queries.size() + 63) / 64;
  const bool wordOverflow =
      !dynamic.actions.empty() &&
      wordsPerNode >
          std::numeric_limits<std::size_t>::max() / dynamic.actions.size();
  const std::size_t totalWords = wordOverflow
                                     ? std::numeric_limits<std::size_t>::max()
                                     : wordsPerNode * dynamic.actions.size();
  if (wordOverflow || totalWords > limits.maximumReachabilityWords) {
    return SyncCoverProtocolError::LimitExceeded;
  }
  std::vector<std::uint64_t> reachesTargets(totalWords);
  for (std::size_t query = 0; query < queries.size(); ++query) {
    reachesTargets[queries[query].target * wordsPerNode + query / 64] |=
        std::uint64_t{1} << (query % 64);
  }
  for (auto node = order.rbegin(); node != order.rend(); ++node) {
    std::uint64_t *destination = reachesTargets.data() + *node * wordsPerNode;
    for (std::size_t successor : dynamic.successors[*node]) {
      const std::uint64_t *source =
          reachesTargets.data() + successor * wordsPerNode;
      if (wordsPerNode > limits.maximumReachabilityWork -
                             std::min(statistics.reachabilityWork,
                                      limits.maximumReachabilityWork) ||
          !consumeWork(workBudget, wordsPerNode)) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      statistics.reachabilityWork += wordsPerNode;
      for (std::size_t word = 0; word < wordsPerNode; ++word) {
        destination[word] |= source[word];
      }
    }
  }
  for (std::size_t query = 0; query < queries.size(); ++query) {
    const Query &description = queries[query];
    const bool reached =
        (reachesTargets[description.source * wordsPerNode + query / 64] &
         (std::uint64_t{1} << (query % 64))) != 0;
    if (!reached) {
      invalid[description.channel][description.supply] = true;
    }
  }
  for (std::size_t channel = 0; channel < witnesses.size(); ++channel) {
    for (std::size_t supply = 0; supply < witnesses[channel].size(); ++supply) {
      if (invalid[channel][supply]) {
        witnesses[channel][supply] = false;
      }
    }
  }
  return SyncCoverProtocolError::None;
}

SyncCoverProtocolError verifyDynamicProtocol(
    const SyncCoverEventProtocol &protocol, DynamicProtocol &dynamic,
    SyncCoverProtocolLimits limits, SyncCoverProtocolStatistics &statistics,
    SupplyWitnesses &supplyWitnesses, SyncCoverCoverageWorkBudget *workBudget) {
  statistics.maximumDynamicActions =
      std::max(statistics.maximumDynamicActions, dynamic.actions.size());
  SyncCoverProtocolError error =
      addIssueOrder(dynamic, limits, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  error = pairTokens(protocol, dynamic, limits, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  error = recordSupplyWitnesses(protocol, dynamic, supplyWitnesses, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  error = addRearmProofs(protocol, dynamic, limits, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  std::vector<std::size_t> order;
  order.reserve(dynamic.actions.size());
  if (!topologicalOrder(dynamic, order, workBudget)) {
    return workBudget && workBudget->exhausted
               ? SyncCoverProtocolError::WorkLimitExceeded
               : SyncCoverProtocolError::InvalidTokenLifecycle;
  }
  error = verifyRearm(protocol, dynamic, order, limits, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  return recordCompletionExportWitnesses(protocol, dynamic, order, limits,
                                         statistics, supplyWitnesses,
                                         workBudget);
}

SyncCoverProtocolError verifyTripCount(
    const SyncCoverEventProtocol &protocol, const ResolvedProtocol &resolved,
    std::size_t tripCount, SyncCoverProtocolLimits limits,
    SyncCoverProtocolStatistics &statistics, SupplyWitnesses &supplyWitnesses,
    SyncCoverCoverageWorkBudget *workBudget) {
  DynamicProtocol dynamic;
  SyncCoverProtocolError error =
      buildActions(protocol, resolved, tripCount, 0, 1, true, true, true,
                   limits, dynamic, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  return verifyDynamicProtocol(protocol, dynamic, limits, statistics,
                               supplyWitnesses, workBudget);
}

SyncCoverProtocolError verifyInvocationSequence(
    const SyncCoverEventProtocol &protocol, const ResolvedProtocol &resolved,
    const std::vector<std::size_t> &tripCounts, SyncCoverProtocolLimits limits,
    SyncCoverProtocolStatistics &statistics, SupplyWitnesses &supplyWitnesses,
    SyncCoverCoverageWorkBudget *workBudget) {
  DynamicProtocol dynamic;
  const std::size_t exitSequence = tripCounts.size() + 1;
  if (tripCounts.empty()) {
    const SyncCoverProtocolError error =
        buildActions(protocol, resolved, 0, 1, exitSequence, false, true, true,
                     limits, dynamic, statistics, workBudget);
    if (error != SyncCoverProtocolError::None) {
      return error;
    }
  } else {
    for (std::size_t invocation = 0; invocation < tripCounts.size();
         ++invocation) {
      const SyncCoverProtocolError error =
          buildActions(protocol, resolved, tripCounts[invocation],
                       invocation + 1, exitSequence, true, invocation == 0,
                       invocation + 1 == tripCounts.size(), limits, dynamic,
                       statistics, workBudget);
      if (error != SyncCoverProtocolError::None) {
        return error;
      }
    }
  }
  return verifyDynamicProtocol(protocol, dynamic, limits, statistics,
                               supplyWitnesses, workBudget);
}

} // namespace

SyncCoverProtocolError
mlir::pto::sync_cover_protocol_detail::verifyResolvedProtocolAutomaton(
    const SyncCoverEventProtocol &protocol, const ResolvedProtocol &resolved,
    SyncCoverProtocolLimits limits, SyncCoverProtocolStatistics &statistics,
    SyncCoverCoverageWorkBudget *workBudget,
    std::vector<std::vector<bool>> *witnessedSupplies) {
  SupplyWitnesses supplyWitnesses;
  supplyWitnesses.reserve(protocol.channels.size());
  for (const SyncCoverEventChannel &channel : protocol.channels) {
    supplyWitnesses.emplace_back(channel.supplies.size(), false);
  }
  const auto finish = [&]() {
    if (witnessedSupplies) {
      *witnessedSupplies = supplyWitnesses;
    }
    for (std::size_t channel = 0; channel < supplyWitnesses.size(); ++channel) {
      const std::vector<bool> &channelWitnesses = supplyWitnesses[channel];
      for (std::size_t supply = 0; supply < channelWitnesses.size(); ++supply) {
        const bool witnessed = channelWitnesses[supply];
        if (!consumeWork(workBudget)) {
          return SyncCoverProtocolError::WorkLimitExceeded;
        }
        if (!witnessed) {
          return SyncCoverProtocolError::InvalidTokenLifecycle;
        }
      }
    }
    return SyncCoverProtocolError::None;
  };
  if (!protocol.loop) {
    statistics.tripCountsChecked = 1;
    const SyncCoverProtocolError error = verifyTripCount(
        protocol, resolved, 1, limits, statistics, supplyWitnesses, workBudget);
    return error == SyncCoverProtocolError::None ? finish() : error;
  }

  const bool hierarchicalLifetime =
      protocol.lifetimeScope && *protocol.lifetimeScope != protocol.loop->scope;
  if (hierarchicalLifetime) {
    std::vector<std::size_t> childTripCounts;
    for (std::size_t tripCount = protocol.loop->mayExecuteZeroTimes ? 0 : 1;
         tripCount <= resolved.verificationHorizon; ++tripCount) {
      childTripCounts.push_back(tripCount);
    }
    std::size_t sequenceCount = childTripCounts.size();
    const bool lifetimeMayBeEmpty = protocol.lifetimeMayExecuteZeroTimes;
    if (lifetimeMayBeEmpty) {
      ++sequenceCount;
    }
    std::size_t pairCount = 0;
    if (!checkedProduct(childTripCounts.size(), childTripCounts.size(),
                        pairCount) ||
        pairCount >
            limits.maximumInvocationSequences -
                std::min(sequenceCount, limits.maximumInvocationSequences)) {
      return SyncCoverProtocolError::LimitExceeded;
    }
    sequenceCount += pairCount;
    if (lifetimeMayBeEmpty) {
      const SyncCoverProtocolError error =
          verifyInvocationSequence(protocol, resolved, {}, limits, statistics,
                                   supplyWitnesses, workBudget);
      if (error != SyncCoverProtocolError::None) {
        return error;
      }
      ++statistics.tripCountsChecked;
    }
    for (std::size_t first : childTripCounts) {
      SyncCoverProtocolError error =
          verifyInvocationSequence(protocol, resolved, {first}, limits,
                                   statistics, supplyWitnesses, workBudget);
      if (error != SyncCoverProtocolError::None) {
        return error;
      }
      ++statistics.tripCountsChecked;
      for (std::size_t second : childTripCounts) {
        error = verifyInvocationSequence(protocol, resolved, {first, second},
                                         limits, statistics, supplyWitnesses,
                                         workBudget);
        if (error != SyncCoverProtocolError::None) {
          return error;
        }
        ++statistics.tripCountsChecked;
      }
    }
    return finish();
  }

  if (protocol.loop->mayExecuteZeroTimes) {
    SyncCoverProtocolError error = verifyTripCount(
        protocol, resolved, 0, limits, statistics, supplyWitnesses, workBudget);
    if (error != SyncCoverProtocolError::None) {
      return error;
    }
    ++statistics.tripCountsChecked;
  }
  for (std::size_t tripCount = 1; tripCount <= resolved.verificationHorizon;
       ++tripCount) {
    SyncCoverProtocolError error =
        verifyTripCount(protocol, resolved, tripCount, limits, statistics,
                        supplyWitnesses, workBudget);
    if (error != SyncCoverProtocolError::None) {
      return error;
    }
    ++statistics.tripCountsChecked;
  }
  return finish();
}
