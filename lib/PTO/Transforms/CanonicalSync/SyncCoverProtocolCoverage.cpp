// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "SyncCoverCoverageInternal.h"
#include "SyncCoverProtocolInternal.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::sync_cover_protocol_detail;

namespace {

bool checkedAdd(std::size_t left, std::size_t right, std::size_t &result) {
  const bool overflow = right > std::numeric_limits<std::size_t>::max() - left;
  if (overflow) {
    return false;
  }
  result = left + right;
  return true;
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

bool checkedAccumulate(std::size_t &target, std::size_t amount) {
  std::size_t result = 0;
  if (!checkedAdd(target, amount, result)) {
    return false;
  }
  target = result;
  return true;
}

std::size_t logarithmicLookupWork(std::size_t size) {
  std::size_t work = 1;
  for (std::size_t value = size; value > 1; value = (value + 1) / 2) {
    ++work;
  }
  return work;
}

bool supportsEventDomain(const SyncCoverProtocolTargetContract &target,
                         std::uint32_t source, std::uint32_t destination) {
  using Capability = SyncCoverProtocolTargetContract::EventCapability;
  const auto found = std::lower_bound(target.eventCapabilities.begin(),
                                      target.eventCapabilities.end(),
                                      Capability{source, destination, 0});
  return found != target.eventCapabilities.end() &&
         found->sourceResource == source &&
         found->targetResource == destination;
}

bool scopeContains(const SyncCoverGraph &graph, SyncCoverScopeId ancestor,
                   SyncCoverScopeId descendant,
                   SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  const bool invalidScope =
      ancestor >= scopes.size() || descendant >= scopes.size();
  if (invalidScope) {
    return false;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (descendant == ancestor) {
      return true;
    }
    if (descendant == 0) {
      return false;
    }
    descendant = scopes[descendant].parent;
  }
}

bool regionContains(const SyncCoverGraph &graph, SyncCoverRegionId ancestor,
                    SyncCoverRegionId descendant,
                    SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverRegion> &regions = graph.getRegions();
  const bool invalidRegion =
      ancestor >= regions.size() || descendant >= regions.size();
  if (invalidRegion) {
    return false;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return false;
    }
    if (descendant == ancestor) {
      return true;
    }
    if (descendant == 0) {
      return false;
    }
    descendant = regions[descendant].parent;
  }
}

std::optional<SyncCoverScopeId>
nearestEnclosingLoop(const SyncCoverGraph &graph, SyncCoverScopeId scope,
                     SyncCoverCoverageWorkBudget *workBudget) {
  const std::vector<SyncCoverScope> &scopes = graph.getScopes();
  if (scope >= scopes.size()) {
    return std::nullopt;
  }
  while (true) {
    if (!consumeWork(workBudget)) {
      return std::nullopt;
    }
    if (scopes[scope].isLoop) {
      return scope;
    }
    if (scope == 0) {
      return std::nullopt;
    }
    scope = scopes[scope].parent;
  }
}

bool nodeOccursInProtocolIteration(const SyncCoverGraph &graph,
                                   const SyncCoverEventProtocol &protocol,
                                   SyncCoverNodeId node,
                                   SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverScopeId> nearestLoop =
      nearestEnclosingLoop(graph, graph.getNodes()[node].scope, workBudget);
  return protocol.loop ? nearestLoop && *nearestLoop == protocol.loop->scope
                       : !nearestLoop;
}

bool demandForcesRegionAtCopy(const SyncCoverGraph &graph,
                              const SyncCoverDemand &demand,
                              SyncCoverRegionId region, unsigned copy,
                              SyncCoverCoverageWorkBudget *workBudget) {
  const bool sourceForces =
      copy == 0 &&
      regionContains(graph, region, graph.getNodes()[demand.source].region,
                     workBudget);
  const bool targetForces =
      copy == demand.distance &&
      regionContains(graph, region, graph.getNodes()[demand.target].region,
                     workBudget);
  return sourceForces || targetForces;
}

bool nodeInstanceMustExecute(const SyncCoverGraph &graph,
                             const SyncCoverDemand &demand,
                             const sync_cover_detail::DemandContext &context,
                             SyncCoverNodeId node, unsigned copy,
                             SyncCoverCoverageWorkBudget *workBudget) {
  bool available = sync_cover_detail::nodeInstanceAvailable(graph, demand, node,
                                                            copy, workBudget);
  if (workBudget && workBudget->exhausted) {
    return false;
  }
  available |= scopeContains(graph, demand.scope, graph.getNodes()[node].scope,
                             workBudget);
  const bool guardIsProven = sync_cover_detail::guardIsImplied(
      graph, demand, context, graph.getNodes()[node].guard, copy, workBudget);
  if (!available || !guardIsProven) {
    return false;
  }
  const bool isDemandEndpoint =
      (node == demand.source && copy == 0) ||
      (node == demand.target && copy == demand.distance);
  if (isDemandEndpoint) {
    return true;
  }
  SyncCoverRegionId region = graph.getNodes()[node].region;
  const SyncCoverRegionId owner = demand.ownerRegion;
  while (region != owner) {
    const bool invalidRegion = !consumeWork(workBudget) || region == 0 ||
                               region >= graph.getRegions().size();
    if (invalidRegion) {
      return false;
    }
    const SyncCoverRegion &description = graph.getRegions()[region];
    const bool forced =
        demandForcesRegionAtCopy(graph, demand, region, copy, workBudget);
    if (workBudget && workBudget->exhausted) {
      return false;
    }
    const bool repeated =
        description.kind == SyncCoverRegionKind::Loop ||
        description.cardinality == SyncCoverRegionCardinality::ZeroOrMore ||
        description.cardinality == SyncCoverRegionCardinality::OneOrMore;
    const bool guardIsPresent = !description.guard.literals.empty();
    const bool guardIsProven =
        guardIsPresent &&
        sync_cover_detail::guardIsImplied(graph, demand, context,
                                          description.guard, copy, workBudget);
    const bool unprovenOptional =
        !forced &&
        description.cardinality == SyncCoverRegionCardinality::ZeroOrOne &&
        (!guardIsPresent || !guardIsProven);
    const bool unprovenGuard = !forced && guardIsPresent && !guardIsProven;
    if (repeated || unprovenOptional || unprovenGuard) {
      return false;
    }
    region = description.parent;
  }
  return true;
}

bool pointExecutesAtCopy(const SyncCoverGraph &graph,
                         const SyncCoverDemand &demand,
                         const sync_cover_detail::DemandContext &context,
                         const SyncCoverCutPoint &point, unsigned copy,
                         SyncCoverCoverageWorkBudget *workBudget) {
  const std::optional<SyncCoverGuard> guard =
      effectivePointGuard(graph, point, workBudget);
  return guard &&
         nodeInstanceMustExecute(graph, demand, context, point.anchor.node,
                                 copy, workBudget) &&
         sync_cover_detail::guardIsImplied(graph, demand, context, *guard, copy,
                                           workBudget);
}

bool nodeInSourcePrefix(const SyncCoverGraph &graph,
                        const SyncCoverEventProtocol &protocol,
                        const SyncCoverEventChannel &channel,
                        const SyncCoverDemand &demand,
                        const sync_cover_detail::DemandContext &context,
                        SyncCoverNodeId node, unsigned copy,
                        SyncCoverCoverageWorkBudget *workBudget) {
  if (graph.getNodes()[node].resource != channel.set.resource ||
      !nodeOccursInProtocolIteration(graph, protocol, node, workBudget) ||
      !nodeInstanceMustExecute(graph, demand, context, node, copy,
                               workBudget) ||
      !pointExecutesAtCopy(graph, demand, context, channel.set, copy,
                           workBudget)) {
    return false;
  }
  const std::optional<SyncCoverTimelinePosition> set =
      resolveSyncCoverAnchor(graph, channel.set.anchor);
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::AfterNode, node,
                                     graph.getNodes()[node].scope, 0});
  return set && position && *position <= *set;
}

bool nodeInTargetSuffix(const SyncCoverGraph &graph,
                        const SyncCoverEventProtocol &protocol,
                        const SyncCoverEventChannel &channel,
                        const SyncCoverDemand &demand,
                        const sync_cover_detail::DemandContext &context,
                        SyncCoverNodeId node, unsigned copy, bool exitExport,
                        bool sourceProvesNonzero,
                        SyncCoverCoverageWorkBudget *workBudget) {
  if (graph.getNodes()[node].resource != channel.wait.resource ||
      !nodeInstanceMustExecute(graph, demand, context, node, copy,
                               workBudget)) {
    return false;
  }
  const std::optional<SyncCoverTimelinePosition> wait =
      resolveSyncCoverAnchor(graph, channel.wait.anchor);
  const std::optional<SyncCoverTimelinePosition> position =
      resolveSyncCoverAnchor(graph, {SyncCoverAnchorKind::BeforeNode, node,
                                     graph.getNodes()[node].scope, 0});
  if (!wait || !position) {
    return false;
  }
  if (nodeOccursInProtocolIteration(graph, protocol, node, workBudget)) {
    return pointExecutesAtCopy(graph, demand, context, channel.wait, copy,
                               workBudget) &&
           *wait <= *position;
  }
  if (!protocol.loop || !exitExport ||
      (protocol.loop->mayExecuteZeroTimes && !sourceProvesNonzero)) {
    return false;
  }
  const std::optional<SyncCoverTimelineInterval> &timeline =
      graph.getScopes()[protocol.loop->scope].timeline;
  return timeline && *position > timeline->end &&
         scopeContains(graph, graph.getNodes()[node].scope,
                       protocol.loop->scope, workBudget);
}

} // namespace

SyncCoverProtocolCoverageResult mlir::pto::computeSyncCoverProtocolExactWorlds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverProtocolCoverageResult result;
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits) ||
                                  !targetFitsProtocolLimits(target, limits);
  if (graphLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  const SyncCoverProtocolError targetError =
      validateProtocolTargetContract(target, limits, workBudget);
  if (targetError != SyncCoverProtocolError::None) {
    result.error = targetError;
    return result;
  }
  const bool rowLimitExceeded = protocols.size() > limits.maximumProtocols ||
                                worlds.size() > limits.maximumWorlds ||
                                worlds.size() > limits.maximumResultRows;
  const std::size_t wordsPerWorld = (graph.getDemands().size() + 63) / 64;
  std::size_t resultWords = 0;
  const bool wordLimitExceeded =
      !checkedProduct(wordsPerWorld, worlds.size(), resultWords) ||
      resultWords > limits.maximumResultWords;
  if (rowLimitExceeded || wordLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  if (!consumeWork(workBudget, worlds.size())) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  std::size_t worldIncidences = 0;
  for (const SyncCoverExactWorld &world : worlds) {
    const bool worldIncidenceLimitExceeded =
        !checkedAccumulate(worldIncidences, world.enabledMechanisms.size()) ||
        worldIncidences > limits.maximumWorldMechanismIncidences;
    if (worldIncidenceLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      return result;
    }
  }
  std::size_t protocolIndexWork = 0;
  std::size_t worldIncidenceWork = 0;
  std::size_t worldWorkPerIncidence = 0;
  const bool indexWorkOverflow =
      !checkedProduct(protocols.size(),
                      logarithmicLookupWork(protocols.size()) + 1,
                      protocolIndexWork) ||
      !checkedAdd(logarithmicLookupWork(protocols.size()), 3,
                  worldWorkPerIncidence) ||
      !checkedProduct(worldIncidences, worldWorkPerIncidence,
                      worldIncidenceWork);
  std::size_t preparationWork = 0;
  const bool preparationFailed =
      indexWorkOverflow ||
      !checkedAdd(resultWords, protocolIndexWork, preparationWork) ||
      !checkedAdd(preparationWork, worldIncidenceWork, preparationWork) ||
      !consumeWork(workBudget, preparationWork);
  if (preparationFailed) {
    result.error = SyncCoverProtocolError::WorkLimitExceeded;
    return result;
  }
  result.coveredByWorld.reserve(worlds.size());
  std::map<SyncCoverMechanismId, std::size_t> protocolByMechanism;
  std::vector<std::vector<bool>> exitExportsByProtocol;
  exitExportsByProtocol.reserve(protocols.size());
  std::size_t rearmProofs = 0;
  std::size_t exitExports = 0;
  std::size_t exitExportGuardLiterals = 0;
  for (std::size_t index = 0; index < protocols.size(); ++index) {
    const bool proofLimitExceeded =
        !checkedAccumulate(rearmProofs, protocols[index].rearmProofs.size()) ||
        rearmProofs > limits.maximumRearmProofs;
    if (proofLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    const bool duplicateMechanism =
        !protocolByMechanism.emplace(protocols[index].mechanism, index).second;
    if (duplicateMechanism) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = index;
      return result;
    }
    SyncCoverProtocolLimits remaining = limits;
    remaining.maximumTotalDynamicActions -=
        result.statistics.totalDynamicActions;
    remaining.maximumAutomatonEdges -= result.statistics.automatonEdges;
    remaining.maximumRearmQueries -= result.statistics.rearmQueries;
    remaining.maximumRearmLookupWork -= result.statistics.rearmLookupWork;
    remaining.maximumReachabilityWork -= result.statistics.reachabilityWork;
    remaining.maximumRearmProofLaneIncidences -=
        result.statistics.rearmProofLaneIncidences;
    remaining.maximumLaneInitializationWork -=
        result.statistics.laneInitializationWork;
    remaining.maximumExitExports -= exitExports;
    remaining.maximumExitExportGuardLiterals -= exitExportGuardLiterals;
    SyncCoverProtocolVerificationResult verification =
        verifyProtocolAssumingValidGraph(graph, target, protocols[index],
                                         remaining, workBudget, false);
    if (!verification) {
      result.error = verification.error;
      result.invalidIndex = index;
      return result;
    }
    if (!consumeWork(workBudget, verification.exitExports.size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    std::vector<bool> exported(protocols[index].channels.size());
    for (const SyncCoverProtocolExitExport &candidate :
         verification.exitExports) {
      if (candidate.channel >= exported.size()) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = index;
        return result;
      }
      exported[candidate.channel] = candidate.availableOnNonzeroTrip;
      const bool exportOverflow =
          !checkedAccumulate(exitExports, 1) ||
          !checkedAccumulate(exitExportGuardLiterals,
                             candidate.guard.literals.size());
      if (exportOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = index;
        return result;
      }
    }
    exitExportsByProtocol.push_back(std::move(exported));
    const bool statisticsOverflow =
        !checkedAccumulate(result.statistics.reachablePhases,
                           verification.statistics.reachablePhases) ||
        !checkedAccumulate(result.statistics.tripCountsChecked,
                           verification.statistics.tripCountsChecked) ||
        !checkedAccumulate(result.statistics.automatonEdges,
                           verification.statistics.automatonEdges) ||
        !checkedAccumulate(result.statistics.rearmQueries,
                           verification.statistics.rearmQueries) ||
        !checkedAccumulate(result.statistics.rearmProofLaneIncidences,
                           verification.statistics.rearmProofLaneIncidences) ||
        !checkedAccumulate(result.statistics.rearmLookupWork,
                           verification.statistics.rearmLookupWork) ||
        !checkedAccumulate(result.statistics.totalDynamicActions,
                           verification.statistics.totalDynamicActions) ||
        !checkedAccumulate(result.statistics.laneInitializationWork,
                           verification.statistics.laneInitializationWork) ||
        !checkedAccumulate(result.statistics.reachabilityWork,
                           verification.statistics.reachabilityWork);
    if (statisticsOverflow ||
        result.statistics.totalDynamicActions >
            limits.maximumTotalDynamicActions ||
        result.statistics.automatonEdges > limits.maximumAutomatonEdges ||
        result.statistics.rearmQueries > limits.maximumRearmQueries ||
        result.statistics.rearmLookupWork > limits.maximumRearmLookupWork ||
        result.statistics.rearmProofLaneIncidences >
            limits.maximumRearmProofLaneIncidences ||
        result.statistics.laneInitializationWork >
            limits.maximumLaneInitializationWork ||
        result.statistics.reachabilityWork > limits.maximumReachabilityWork ||
        exitExports > limits.maximumExitExports ||
        exitExportGuardLiterals > limits.maximumExitExportGuardLiterals) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    result.statistics.maximumDynamicActions =
        std::max(result.statistics.maximumDynamicActions,
                 verification.statistics.maximumDynamicActions);
  }

  std::size_t coverageTransitions = 0;
  for (std::size_t worldIndex = 0; worldIndex < worlds.size(); ++worldIndex) {
    const SyncCoverExactWorld &world = worlds[worldIndex];
    if (!std::is_sorted(world.enabledMechanisms.begin(),
                        world.enabledMechanisms.end()) ||
        std::adjacent_find(world.enabledMechanisms.begin(),
                           world.enabledMechanisms.end()) !=
            world.enabledMechanisms.end()) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = worldIndex;
      return result;
    }
    SyncCoverDemandSet covered(graph.getDemands().size());
    std::vector<std::size_t> enabledProtocols;
    enabledProtocols.reserve(world.enabledMechanisms.size());
    for (SyncCoverMechanismId mechanism : world.enabledMechanisms) {
      const auto found = protocolByMechanism.find(mechanism);
      if (found == protocolByMechanism.end()) {
        result.error = SyncCoverProtocolError::InvalidProtocol;
        result.invalidIndex = worldIndex;
        return result;
      }
      enabledProtocols.push_back(found->second);
    }
    for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
         ++demandId) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      const sync_cover_detail::DemandContext context =
          sync_cover_detail::makeDemandContext(graph, demand, workBudget);
      if (!context.valid) {
        result.error = workBudget && workBudget->exhausted
                           ? SyncCoverProtocolError::WorkLimitExceeded
                           : SyncCoverProtocolError::InvalidGraph;
        result.invalidIndex = demandId;
        return result;
      }
      std::size_t distanceCount = 0;
      std::size_t reachedEntries = 0;
      const bool stateOverflow =
          !checkedAdd(static_cast<std::size_t>(demand.distance), 1,
                      distanceCount) ||
          !checkedProduct(distanceCount, graph.getNodes().size(),
                          reachedEntries) ||
          reachedEntries > limits.maximumCoverageStates / 16;
      if (stateOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = demandId;
        return result;
      }
      std::vector<std::uint16_t> reached(reachedEntries);
      struct State {
        SyncCoverNodeId node = 0;
        unsigned distance = 0;
        SyncCoverOrderingRequirementMask capabilities = 0;
      };
      std::deque<State> pending;
      const auto addState = [&](SyncCoverNodeId node, unsigned distance,
                                SyncCoverOrderingRequirementMask capabilities) {
        if (distance > demand.distance ||
            capabilities > kAllSyncCoverOrderingRequirements) {
          return;
        }
        const std::size_t entry =
            static_cast<std::size_t>(distance) * graph.getNodes().size() + node;
        const std::uint16_t bit =
            static_cast<std::uint16_t>(std::uint16_t{1} << capabilities);
        const bool newState = (reached[entry] & bit) == 0;
        if (newState) {
          reached[entry] |= bit;
          pending.push_back({node, distance, capabilities});
        }
      };
      const auto chargeTransition = [&]() {
        if (coverageTransitions == limits.maximumCoverageTransitions ||
            !consumeWork(workBudget)) {
          return false;
        }
        ++coverageTransitions;
        return true;
      };
      const auto activateChannel =
          [&](std::size_t protocolIndex, const SyncCoverEventChannel &channel,
              SyncCoverNodeId source, unsigned incomingDistance,
              SyncCoverOrderingRequirementMask incomingCapabilities) {
            const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
            const std::optional<SyncCoverScopeId> demandSourceLoop =
                protocol.loop
                    ? std::optional<SyncCoverScopeId>{}
                    : nearestEnclosingLoop(
                          graph, graph.getNodes()[demand.source].scope,
                          workBudget);
            const std::optional<SyncCoverScopeId> demandTargetLoop =
                protocol.loop
                    ? std::optional<SyncCoverScopeId>{}
                    : nearestEnclosingLoop(
                          graph, graph.getNodes()[demand.target].scope,
                          workBudget);
            const bool localProtocol =
                protocol.loop ? protocol.loop->scope == demand.scope
                              : demand.distance == 0 && !demandSourceLoop &&
                                    !demandTargetLoop;
            const bool nestedExport =
                protocol.loop && protocol.loop->scope != demand.scope &&
                scopeContains(graph, demand.scope, protocol.loop->scope,
                              workBudget);
            if (workBudget && workBudget->exhausted) {
              return false;
            }
            const bool sourceOutsideProtocolWorld =
                !localProtocol && !nestedExport;
            const bool sourceOutsidePrefix =
                !sourceOutsideProtocolWorld &&
                !nodeInSourcePrefix(graph, protocol, channel, demand, context,
                                    source, incomingDistance, workBudget);
            if (workBudget && workBudget->exhausted) {
              return false;
            }
            if (sourceOutsideProtocolWorld || sourceOutsidePrefix) {
              return true;
            }
            const unsigned channelDistance =
                localProtocol &&
                        channel.flow == SyncCoverEventChannelFlow::LoopCarry
                    ? channel.distance
                    : 0;
            const SyncCoverOrderingRequirementMask transferred =
                incomingCapabilities == 0
                    ? channel.suppliedRequirements
                    : static_cast<SyncCoverOrderingRequirementMask>(
                          incomingCapabilities & channel.suppliedRequirements);
            if (transferred == 0) {
              return true;
            }
            const bool exitExport =
                channel.id < exitExportsByProtocol[protocolIndex].size() &&
                exitExportsByProtocol[protocolIndex][channel.id];
            const bool sourceProvesNonzero =
                protocol.loop &&
                nearestEnclosingLoop(graph, graph.getNodes()[source].scope,
                                     workBudget) ==
                    std::optional<SyncCoverScopeId>(protocol.loop->scope);
            if (workBudget && workBudget->exhausted) {
              return false;
            }
            for (const SyncCoverNode &targetNode : graph.getNodes()) {
              if (!chargeTransition()) {
                return false;
              }
              const bool targetIsLocal = nodeOccursInProtocolIteration(
                  graph, protocol, targetNode.id, workBudget);
              if (workBudget && workBudget->exhausted) {
                return false;
              }
              const bool targetOutsideProtocolWorld =
                  (targetIsLocal && !localProtocol) ||
                  (!targetIsLocal && !nestedExport);
              if (targetOutsideProtocolWorld) {
                continue;
              }
              const unsigned appliedDistance =
                  targetIsLocal ? channelDistance : 0;
              if (appliedDistance > demand.distance - incomingDistance) {
                continue;
              }
              const bool inTargetSuffix = nodeInTargetSuffix(
                  graph, protocol, channel, demand, context, targetNode.id,
                  incomingDistance + appliedDistance, exitExport,
                  sourceProvesNonzero, workBudget);
              if (workBudget && workBudget->exhausted) {
                return false;
              }
              if (!inTargetSuffix) {
                continue;
              }
              for (SyncCoverOrderingRequirementMask subset = transferred;
                   subset != 0;
                   subset = static_cast<SyncCoverOrderingRequirementMask>(
                       (subset - 1) & transferred)) {
                addState(targetNode.id, incomingDistance + appliedDistance,
                         subset);
              }
            }
            return !workBudget || !workBudget->exhausted;
          };

      addState(demand.source, 0, 0);
      while (!pending.empty()) {
        const State state = pending.front();
        pending.pop_front();
        for (const SyncCoverEdge &edge : graph.getEdges()) {
          if (!chargeTransition()) {
            result.error = workBudget && workBudget->exhausted
                               ? SyncCoverProtocolError::WorkLimitExceeded
                               : SyncCoverProtocolError::LimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          const bool typedDistance =
              edge.distance == 0 || edge.scope == demand.scope;
          if (edge.source != state.node || !typedDistance ||
              edge.distance > demand.distance - state.distance) {
            continue;
          }
          const unsigned targetDistance = state.distance + edge.distance;
          const bool edgeActive =
              nodeInstanceMustExecute(graph, demand, context, edge.source,
                                      state.distance, workBudget) &&
              nodeInstanceMustExecute(graph, demand, context, edge.target,
                                      targetDistance, workBudget) &&
              sync_cover_detail::edgeGuardsActive(graph, demand, context, edge,
                                                  state.distance,
                                                  targetDistance, workBudget);
          if (workBudget && workBudget->exhausted) {
            result.error = SyncCoverProtocolError::WorkLimitExceeded;
            result.invalidIndex = demandId;
            return result;
          }
          if (!edgeActive) {
            continue;
          }
          if (edge.kind == SyncCoverEdgeKind::CompletionSupply) {
            const SyncCoverOrderingRequirementMask supplied =
                syncCoverOrderingRequirementBit(
                    SyncCoverOrderingRequirement::
                        PipelineCompletionBeforeAccess);
            addState(edge.target, targetDistance,
                     static_cast<SyncCoverOrderingRequirementMask>(
                         state.capabilities | supplied));
          } else if (edge.kind ==
                         SyncCoverEdgeKind::CertifiedCompletionFrontier ||
                     state.capabilities != 0) {
            addState(edge.target, targetDistance, state.capabilities);
          }
        }
        for (std::size_t protocolIndex : enabledProtocols) {
          for (const SyncCoverEventChannel &channel :
               protocols[protocolIndex].channels) {
            if (!chargeTransition()) {
              result.error = workBudget && workBudget->exhausted
                                 ? SyncCoverProtocolError::WorkLimitExceeded
                                 : SyncCoverProtocolError::LimitExceeded;
              result.invalidIndex = demandId;
              return result;
            }
            if (!activateChannel(protocolIndex, channel, state.node,
                                 state.distance, state.capabilities)) {
              result.error = workBudget && workBudget->exhausted
                                 ? SyncCoverProtocolError::WorkLimitExceeded
                                 : SyncCoverProtocolError::LimitExceeded;
              result.invalidIndex = demandId;
              return result;
            }
          }
        }
      }
      const std::size_t targetEntry =
          static_cast<std::size_t>(demand.distance) * graph.getNodes().size() +
          demand.target;
      bool demandCovered = false;
      for (SyncCoverOrderingRequirementMask mask = 1;
           mask <= kAllSyncCoverOrderingRequirements; ++mask) {
        const std::uint16_t bit =
            static_cast<std::uint16_t>(std::uint16_t{1} << mask);
        demandCovered |=
            (reached[targetEntry] & bit) != 0 &&
            (mask & demand.orderingRequirements) == demand.orderingRequirements;
      }
      if (demandCovered) {
        covered.insert(demandId);
      }
    }
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = worldIndex;
      return result;
    }
    result.coveredByWorld.push_back(std::move(covered));
  }
  result.statistics.coverageTransitions = coverageTransitions;
  return result;
}

SyncCoverProtocolAllocationResult mlir::pto::allocateSyncCoverProtocolEventIds(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const std::vector<SyncCoverEventProtocol> &protocols,
    const std::vector<SyncCoverProtocolEventReservation> &reservations,
    SyncCoverProtocolLimits limits, SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverProtocolAllocationResult result;
  const bool graphLimitExceeded = !graphFitsProtocolLimits(graph, limits) ||
                                  !targetFitsProtocolLimits(target, limits);
  if (graphLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  const bool invalidGraph = !graph.isStructureFrozen() || !graph.validate();
  if (invalidGraph) {
    result.error = SyncCoverProtocolError::InvalidGraph;
    return result;
  }
  const SyncCoverProtocolError targetError =
      validateProtocolTargetContract(target, limits, workBudget);
  if (targetError != SyncCoverProtocolError::None) {
    result.error = targetError;
    return result;
  }
  const bool catalogLimitExceeded =
      protocols.size() > limits.maximumProtocols ||
      reservations.size() > limits.maximumReservations;
  if (catalogLimitExceeded) {
    result.error = SyncCoverProtocolError::LimitExceeded;
    return result;
  }

  using Domain = std::pair<std::uint32_t, std::uint32_t>;
  std::map<Domain, std::set<unsigned>> unavailable;
  std::size_t reservationIdIncidences = 0;
  for (std::size_t index = 0; index < reservations.size(); ++index) {
    const SyncCoverProtocolEventReservation &reservation = reservations[index];
    const bool incidenceLimitExceeded =
        !checkedAccumulate(reservationIdIncidences,
                           reservation.eventIds.size()) ||
        reservationIdIncidences > limits.maximumReservationIdIncidences;
    if (incidenceLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    std::size_t workPerId = 0;
    std::size_t idWork = 0;
    std::size_t reservationWork = 0;
    const std::size_t domainLookupWork =
        logarithmicLookupWork(unavailable.size() + 1);
    const bool reservationWorkOverflow =
        !checkedAdd(logarithmicLookupWork(target.compilerUsableEventIds.size()),
                    logarithmicLookupWork(reservation.eventIds.size() + 1),
                    workPerId) ||
        !checkedAdd(workPerId, 3, workPerId) ||
        !checkedProduct(reservation.eventIds.size(), workPerId, idWork) ||
        !checkedAdd(domainLookupWork, domainLookupWork, reservationWork) ||
        !checkedAdd(reservationWork,
                    logarithmicLookupWork(target.eventCapabilities.size()),
                    reservationWork) ||
        !checkedAdd(reservationWork, idWork, reservationWork) ||
        !consumeWork(workBudget, reservationWork);
    if (reservationWorkOverflow) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    const Domain domain{reservation.sourceResource, reservation.targetResource};
    const bool duplicateDomain = unavailable.count(domain) != 0;
    const bool supportedDomain =
        supportsEventDomain(target, domain.first, domain.second);
    if (duplicateDomain || !supportedDomain ||
        !std::is_sorted(reservation.eventIds.begin(),
                        reservation.eventIds.end()) ||
        std::adjacent_find(reservation.eventIds.begin(),
                           reservation.eventIds.end()) !=
            reservation.eventIds.end() ||
        std::any_of(reservation.eventIds.begin(), reservation.eventIds.end(),
                    [&](unsigned id) {
                      return !std::binary_search(
                          target.compilerUsableEventIds.begin(),
                          target.compilerUsableEventIds.end(), id);
                    })) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = index;
      return result;
    }
    const bool domainLimitReached =
        unavailable.size() == limits.maximumReservationDomains;
    if (domainLimitReached) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = index;
      return result;
    }
    unavailable[domain].insert(reservation.eventIds.begin(),
                               reservation.eventIds.end());
  }

  struct Request {
    Domain domain;
    SyncCoverMechanismId mechanism = 0;
    SyncCoverProtocolChannelId channel = 0;
    std::size_t width = 0;
    std::size_t protocolIndex = 0;
  };
  std::vector<Request> requests;
  std::set<SyncCoverMechanismId> mechanisms;
  std::size_t requestedIds = 0;
  std::size_t verifiedActions = 0;
  std::size_t verifiedAutomatonEdges = 0;
  std::size_t verifiedQueries = 0;
  std::size_t verifiedRearmLookupWork = 0;
  std::size_t verifiedReachabilityWork = 0;
  std::size_t verifiedProofLaneIncidences = 0;
  std::size_t verifiedLaneInitializationWork = 0;
  std::size_t verifiedExitExports = 0;
  std::size_t verifiedExitExportGuardLiterals = 0;
  std::size_t rearmProofs = 0;
  std::size_t channelIncidences = 0;
  for (const SyncCoverEventProtocol &protocol : protocols) {
    const bool channelLimitExceeded =
        !checkedAccumulate(channelIncidences, protocol.channels.size()) ||
        channelIncidences > limits.maximumChannelRequests;
    if (channelLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      return result;
    }
  }
  std::size_t catalogWork = 0;
  const bool catalogWorkOverflow =
      !checkedProduct(protocols.size(),
                      logarithmicLookupWork(protocols.size()) + 1,
                      catalogWork) ||
      !checkedAdd(catalogWork, channelIncidences, catalogWork) ||
      !consumeWork(workBudget, catalogWork);
  if (catalogWorkOverflow) {
    result.error = workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  requests.reserve(channelIncidences);
  for (std::size_t protocolIndex = 0; protocolIndex < protocols.size();
       ++protocolIndex) {
    const SyncCoverEventProtocol &protocol = protocols[protocolIndex];
    const bool proofLimitExceeded =
        !checkedAccumulate(rearmProofs, protocol.rearmProofs.size()) ||
        rearmProofs > limits.maximumRearmProofs;
    if (proofLimitExceeded) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!mechanisms.insert(protocol.mechanism).second) {
      result.error = SyncCoverProtocolError::InvalidProtocol;
      result.invalidIndex = protocolIndex;
      return result;
    }
    SyncCoverProtocolLimits remaining = limits;
    remaining.maximumTotalDynamicActions -= verifiedActions;
    remaining.maximumAutomatonEdges -= verifiedAutomatonEdges;
    remaining.maximumRearmQueries -= verifiedQueries;
    remaining.maximumRearmLookupWork -= verifiedRearmLookupWork;
    remaining.maximumReachabilityWork -= verifiedReachabilityWork;
    remaining.maximumRearmProofLaneIncidences -= verifiedProofLaneIncidences;
    remaining.maximumLaneInitializationWork -= verifiedLaneInitializationWork;
    remaining.maximumExitExports -= verifiedExitExports;
    remaining.maximumExitExportGuardLiterals -= verifiedExitExportGuardLiterals;
    const SyncCoverProtocolVerificationResult verification =
        verifyProtocolAssumingValidGraph(graph, target, protocol, remaining,
                                         workBudget, false);
    if (!verification) {
      result.error = verification.error;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!consumeWork(workBudget, verification.exitExports.size())) {
      result.error = SyncCoverProtocolError::WorkLimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    if (!checkedAccumulate(verifiedActions,
                           verification.statistics.totalDynamicActions) ||
        !checkedAccumulate(verifiedAutomatonEdges,
                           verification.statistics.automatonEdges) ||
        !checkedAccumulate(verifiedQueries,
                           verification.statistics.rearmQueries) ||
        !checkedAccumulate(verifiedRearmLookupWork,
                           verification.statistics.rearmLookupWork) ||
        !checkedAccumulate(verifiedProofLaneIncidences,
                           verification.statistics.rearmProofLaneIncidences) ||
        !checkedAccumulate(verifiedLaneInitializationWork,
                           verification.statistics.laneInitializationWork) ||
        !checkedAccumulate(verifiedReachabilityWork,
                           verification.statistics.reachabilityWork)) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = protocolIndex;
      return result;
    }
    for (const SyncCoverProtocolExitExport &candidate :
         verification.exitExports) {
      const bool exportOverflow =
          !checkedAccumulate(verifiedExitExports, 1) ||
          !checkedAccumulate(verifiedExitExportGuardLiterals,
                             candidate.guard.literals.size());
      if (exportOverflow) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = protocolIndex;
        return result;
      }
    }
    for (const SyncCoverEventChannel &channel : protocol.channels) {
      const Domain domain{channel.set.resource, channel.wait.resource};
      const bool requestLimitExceeded =
          requests.size() == limits.maximumChannelRequests ||
          !checkedAccumulate(requestedIds, channel.width) ||
          requestedIds > limits.maximumAllocatedEventIds;
      if (requestLimitExceeded) {
        result.error = SyncCoverProtocolError::LimitExceeded;
        result.invalidIndex = protocolIndex;
        return result;
      }
      requests.push_back({domain, protocol.mechanism, channel.id, channel.width,
                          protocolIndex});
    }
  }
  std::size_t logarithm = 0;
  for (std::size_t value = requests.size(); value > 1;
       value = (value + 1) / 2) {
    ++logarithm;
  }
  std::size_t sortWork = 0;
  std::size_t sortWorkPerRequest = 0;
  const bool sortFailed =
      !checkedAdd(logarithm, 1, sortWorkPerRequest) ||
      !checkedProduct(requests.size(), sortWorkPerRequest, sortWork) ||
      !checkedProduct(sortWork, 3, sortWork) ||
      !consumeWork(workBudget, sortWork);
  if (sortFailed) {
    result.error = workBudget && workBudget->exhausted
                       ? SyncCoverProtocolError::WorkLimitExceeded
                       : SyncCoverProtocolError::LimitExceeded;
    return result;
  }
  std::sort(requests.begin(), requests.end(),
            [](const Request &left, const Request &right) {
              return std::tie(left.domain, left.mechanism, left.channel) <
                     std::tie(right.domain, right.mechanism, right.channel);
            });
  const bool duplicateRequest =
      std::adjacent_find(requests.begin(), requests.end(),
                         [](const Request &left, const Request &right) {
                           return left.mechanism == right.mechanism &&
                                  left.channel == right.channel;
                         }) != requests.end();
  if (duplicateRequest) {
    result.error = SyncCoverProtocolError::InvalidProtocol;
    return result;
  }

  result.channels.reserve(requests.size());
  for (const Request &request : requests) {
    const std::size_t domainLookupWork =
        logarithmicLookupWork(unavailable.size() + 1);
    std::size_t unavailableEntries = 0;
    const bool unavailableOverflow =
        !checkedAdd(limits.maximumReservationIdIncidences,
                    limits.maximumAllocatedEventIds, unavailableEntries) ||
        !checkedAdd(unavailableEntries, 1, unavailableEntries);
    const std::size_t eventLookupWork =
        unavailableOverflow ? 0 : logarithmicLookupWork(unavailableEntries);
    std::size_t workPerEventId = 0;
    std::size_t eventIdWork = 0;
    std::size_t allocationWork = 0;
    const bool allocationWorkOverflow =
        unavailableOverflow ||
        !checkedAdd(domainLookupWork, domainLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, eventLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, eventLookupWork, workPerEventId) ||
        !checkedAdd(workPerEventId, 1, workPerEventId) ||
        !checkedProduct(target.compilerUsableEventIds.size(), workPerEventId,
                        eventIdWork) ||
        !checkedAdd(domainLookupWork, eventIdWork, allocationWork) ||
        !consumeWork(workBudget, allocationWork);
    if (allocationWorkOverflow) {
      result.error = workBudget && workBudget->exhausted
                         ? SyncCoverProtocolError::WorkLimitExceeded
                         : SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    const bool domainLimitReached =
        unavailable.count(request.domain) == 0 &&
        unavailable.size() == limits.maximumReservationDomains;
    if (domainLimitReached) {
      result.error = SyncCoverProtocolError::LimitExceeded;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    SyncCoverProtocolChannelAllocation allocation;
    allocation.mechanism = request.mechanism;
    allocation.channel = request.channel;
    for (unsigned id : target.compilerUsableEventIds) {
      const bool available = unavailable[request.domain].count(id) == 0;
      if (available) {
        allocation.eventIds.push_back(id);
        unavailable[request.domain].insert(id);
        const bool complete = allocation.eventIds.size() == request.width;
        if (complete) {
          break;
        }
      }
    }
    const bool scarce = allocation.eventIds.size() != request.width;
    if (scarce) {
      result.error = SyncCoverProtocolError::ResourceInfeasible;
      result.invalidIndex = request.protocolIndex;
      return result;
    }
    result.channels.push_back(std::move(allocation));
  }
  return result;
}
