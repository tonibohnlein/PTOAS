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
  std::size_t iteration = 0;
  std::uint32_t resource = 0;
  SyncCoverTimelinePosition position = 0;
  std::size_t ordinal = 0;
  std::optional<std::size_t> mate;
};

struct DynamicProtocol {
  std::vector<DynamicAction> actions;
  std::vector<std::vector<std::size_t>> successors;
  std::vector<std::vector<std::vector<std::size_t>>> sets;
  std::vector<std::vector<std::vector<std::size_t>>> waits;
};

bool phaseIsActive(const SyncCoverEventChannel &channel, std::size_t phase) {
  return channel.activePhases.empty() ||
         std::binary_search(channel.activePhases.begin(),
                            channel.activePhases.end(), phase);
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

std::tuple<unsigned, std::size_t, SyncCoverTimelinePosition, std::size_t,
           unsigned, SyncCoverProtocolChannelId>
orderKey(const DynamicAction &action) {
  const unsigned segment = static_cast<unsigned>(action.segment);
  const unsigned kind = action.kind == ActionKind::Wait ? 0U : 1U;
  return {segment, action.iteration, action.position, action.ordinal,
          kind,    action.channel};
}

SyncCoverProtocolError buildActions(const SyncCoverEventProtocol &protocol,
                                    const ResolvedProtocol &resolved,
                                    std::size_t tripCount,
                                    SyncCoverProtocolLimits limits,
                                    DynamicProtocol &dynamic,
                                    SyncCoverProtocolStatistics &statistics,
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
      if (channel.flow != SyncCoverEventChannelFlow::LoopCarry) {
        continue;
      }
      for (std::size_t lane = 0; lane < channel.width; ++lane) {
        if (addAction(dynamic,
                      {ActionKind::Set, ActionSegment::Entry, channel.id, lane,
                       0, channel.set.resource, 0, lane, std::nullopt},
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
  for (std::size_t iteration = 0; iteration < tripCount; ++iteration) {
    for (const ResolvedChannel &resolvedChannel : resolved.channels) {
      const SyncCoverEventChannel &channel = *resolvedChannel.description;
      if (!consumeWork(workBudget,
                       logarithmicLookupWork(channel.activePhases.size()))) {
        return SyncCoverProtocolError::WorkLimitExceeded;
      }
      if (!phaseIsActive(channel, phase)) {
        continue;
      }
      const std::size_t lane =
          protocol.loop ? protocol.loop->laneByPhase[phase] : 0;
      const auto append = [&](ActionKind kind, std::uint32_t resource,
                              SyncCoverTimelinePosition position,
                              std::size_t ordinal) {
        return addAction(dynamic,
                         {kind, ActionSegment::Body, channel.id, lane,
                          iteration, resource, position, ordinal, std::nullopt},
                         limits, statistics, workBudget);
      };
      if (channel.flow == SyncCoverEventChannelFlow::LoopCarry &&
          append(ActionKind::Wait, channel.wait.resource,
                 resolvedChannel.waitPosition, channel.wait.ordinal) ==
              std::numeric_limits<std::size_t>::max()) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      if (append(ActionKind::Set, channel.set.resource,
                 resolvedChannel.setPosition, channel.set.ordinal) ==
          std::numeric_limits<std::size_t>::max()) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
      if (channel.flow != SyncCoverEventChannelFlow::LoopCarry &&
          append(ActionKind::Wait, channel.wait.resource,
                 resolvedChannel.waitPosition, channel.wait.ordinal) ==
              std::numeric_limits<std::size_t>::max()) {
        return workBudget && workBudget->exhausted
                   ? SyncCoverProtocolError::WorkLimitExceeded
                   : SyncCoverProtocolError::LimitExceeded;
      }
    }
    if (protocol.loop) {
      phase = resolved.nextPhase[phase];
    }
  }

  if (protocol.loop) {
    for (const ResolvedChannel &resolvedChannel : resolved.channels) {
      const SyncCoverEventChannel &channel = *resolvedChannel.description;
      if (channel.flow != SyncCoverEventChannelFlow::LoopCarry) {
        continue;
      }
      for (std::size_t lane = 0; lane < channel.width; ++lane) {
        if (addAction(dynamic,
                      {ActionKind::Wait, ActionSegment::Exit, channel.id, lane,
                       tripCount, channel.wait.resource,
                       std::numeric_limits<SyncCoverTimelinePosition>::max(),
                       lane, std::nullopt},
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

SyncCoverProtocolError
verifyTripCount(const SyncCoverEventProtocol &protocol,
                const ResolvedProtocol &resolved, std::size_t tripCount,
                SyncCoverProtocolLimits limits,
                SyncCoverProtocolStatistics &statistics,
                SyncCoverCoverageWorkBudget *workBudget) {
  DynamicProtocol dynamic;
  SyncCoverProtocolError error = buildActions(
      protocol, resolved, tripCount, limits, dynamic, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  statistics.maximumDynamicActions =
      std::max(statistics.maximumDynamicActions, dynamic.actions.size());
  error = addIssueOrder(dynamic, limits, statistics, workBudget);
  if (error != SyncCoverProtocolError::None) {
    return error;
  }
  error = pairTokens(protocol, dynamic, limits, statistics, workBudget);
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
  return verifyRearm(protocol, dynamic, order, limits, statistics, workBudget);
}

} // namespace

SyncCoverProtocolError
mlir::pto::sync_cover_protocol_detail::verifyResolvedProtocolAutomaton(
    const SyncCoverEventProtocol &protocol, const ResolvedProtocol &resolved,
    SyncCoverProtocolLimits limits, SyncCoverProtocolStatistics &statistics,
    SyncCoverCoverageWorkBudget *workBudget) {
  if (!protocol.loop) {
    statistics.tripCountsChecked = 1;
    return verifyTripCount(protocol, resolved, 1, limits, statistics,
                           workBudget);
  }

  if (protocol.loop->mayExecuteZeroTimes) {
    SyncCoverProtocolError error =
        verifyTripCount(protocol, resolved, 0, limits, statistics, workBudget);
    if (error != SyncCoverProtocolError::None) {
      return error;
    }
    ++statistics.tripCountsChecked;
  }
  for (std::size_t tripCount = 1; tripCount <= resolved.verificationHorizon;
       ++tripCount) {
    SyncCoverProtocolError error = verifyTripCount(
        protocol, resolved, tripCount, limits, statistics, workBudget);
    if (error != SyncCoverProtocolError::None) {
      return error;
    }
    ++statistics.tripCountsChecked;
  }
  return SyncCoverProtocolError::None;
}
