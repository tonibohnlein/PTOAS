// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncOwnership.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::pto;

namespace {

struct UseActions {
  std::size_t wait = 0;
  std::vector<std::pair<SyncCoverNodeId, std::size_t>> sets;
};

SyncCoverEdge makeEdge(const SyncCoverGraph &graph, SyncCoverNodeId source,
                       SyncCoverNodeId target, SyncCoverScopeId scope,
                       unsigned distance) {
  return {source,
          target,
          SyncCoverEdgeKind::CompletionSupply,
          scope,
          distance,
          graph.getNodes()[source].guard,
          graph.getNodes()[target].guard};
}

CanonicalSyncAction makeAction(CanonicalSyncActionKind kind,
                               std::uint32_t resource, SyncCoverAnchor anchor,
                               std::size_t eventUse, unsigned lane) {
  CanonicalSyncAction result;
  result.kind = kind;
  result.resource = resource;
  result.anchor = anchor;
  result.eventUse = eventUse;
  result.eventLane = lane;
  return result;
}

void addBinding(CanonicalSyncMechanismDescriptor &descriptor,
                SyncCoverEdge edge, std::size_t eventUse, std::size_t set,
                std::size_t wait,
                ArrayRef<SyncCoverDemandId> allowedDemands = {}) {
  CanonicalSyncSupplyBinding binding;
  binding.edge = std::move(edge);
  binding.eventUse = eventUse;
  binding.produceAction = set;
  binding.consumeAction = wait;
  binding.proof = CanonicalSyncSupplyProof::VerifiedProtocol;
  binding.allowedDemands.assign(allowedDemands.begin(), allowedDemands.end());
  descriptor.supplies.push_back(std::move(binding));
}

bool intervalContains(SyncCoverStorageInterval extent,
                      SyncCoverStorageInterval overlap) {
  return extent.begin <= overlap.begin && overlap.end <= extent.end;
}

std::vector<SyncCoverDemandId>
getLaneOwnershipDemands(const SyncCoverGraph &graph,
                        const CanonicalSyncOwnershipCycle &cycle,
                        unsigned lane) {
  std::vector<SyncCoverNodeId> consumers;
  std::vector<SyncCoverNodeId> producers;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (use.lane == lane) {
        consumers.insert(consumers.end(), use.consumers.begin(),
                         use.consumers.end());
      }
      if (use.producerLane == lane) {
        producers.insert(producers.end(), use.producers.begin(),
                         use.producers.end());
      }
    }
  }
  llvm::sort(consumers);
  consumers.erase(std::unique(consumers.begin(), consumers.end()),
                  consumers.end());
  llvm::sort(producers);
  producers.erase(std::unique(producers.begin(), producers.end()),
                  producers.end());

  std::vector<SyncCoverDemandId> result;
  for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
    const bool onlyOwnershipHazards =
        !demand.provenanceKinds.empty() &&
        llvm::all_of(demand.provenanceKinds, [](auto kind) {
          return kind == SyncCoverDemandKind::MemoryWAR ||
                 kind == SyncCoverDemandKind::MemoryWAW;
        });
    if (!onlyOwnershipHazards || demand.storageWitnesses.empty()) {
      continue;
    }
    const bool allWitnessesMatch = llvm::all_of(
        demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
          if (witnessId >= graph.getStorageWitnesses().size()) {
            return false;
          }
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[witnessId];
          if (witness.sourceAccess >= graph.getStorageAccesses().size() ||
              witness.targetAccess >= graph.getStorageAccesses().size()) {
            return false;
          }
          const SyncCoverStorageAccess &source =
              graph.getStorageAccesses()[witness.sourceAccess];
          const SyncCoverStorageAccess &target =
              graph.getStorageAccesses()[witness.targetAccess];
          const bool sourceOwnsSlot =
              (source.mode == SyncCoverStorageAccessMode::Read &&
               std::binary_search(consumers.begin(), consumers.end(),
                                  source.node)) ||
              (source.mode == SyncCoverStorageAccessMode::Write &&
               std::binary_search(producers.begin(), producers.end(),
                                  source.node));
          const bool roles = source.exactPhysical && target.exactPhysical &&
                             sourceOwnsSlot &&
                             target.mode == SyncCoverStorageAccessMode::Write &&
                             std::binary_search(producers.begin(),
                                                producers.end(), target.node);
          const bool managedSlot = llvm::any_of(
              cycle.lanes[lane].slots,
              [&](const CanonicalSyncOwnershipSlot &slot) {
                return source.domain == slot.domain &&
                       target.domain == slot.domain &&
                       intervalContains(slot.extent, witness.overlap);
              });
          return roles && managedSlot;
        });
    if (allWitnessesMatch) {
      result.push_back(demandId);
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor
buildReady(const SyncCoverGraph &graph,
           const CanonicalSyncOwnershipCycle &cycle,
           CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  const std::size_t producerCount =
      cycle.paths.front().uses.front().producers.size();
  for (std::size_t producer = 0; producer < producerCount; ++producer) {
    result.eventUses.push_back(
        {domain, cycle.lanes.size(), cycle.recurrenceScope});
  }
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      for (auto [producerIndex, producer] : llvm::enumerate(use.producers)) {
        const std::size_t set = result.actions.size();
        result.actions.push_back(makeAction(
            CanonicalSyncActionKind::EventSet, cycle.producerResource,
            {SyncCoverAnchorKind::AfterNode, producer, 0, 0}, producerIndex,
            use.lane));
        for (SyncCoverNodeId consumer : use.consumers) {
          const std::size_t wait = result.actions.size();
          result.actions.push_back(makeAction(
              CanonicalSyncActionKind::EventWait, cycle.consumerResource,
              {SyncCoverAnchorKind::BeforeNode, consumer, 0, 0}, producerIndex,
              use.lane));
          const SyncCoverScopeId scope =
              graph
                  .getLowestCommonScope(graph.getNodes()[producer].scope,
                                        graph.getNodes()[consumer].scope)
                  .value_or(path.scope);
          addBinding(result, makeEdge(graph, producer, consumer, scope, 0),
                     producerIndex, set, wait);
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor
buildRoundTripRelease(const SyncCoverGraph &graph,
                      const CanonicalSyncOwnershipCycle &cycle,
                      CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back(
      {domain, cycle.lanes.size(), cycle.recurrenceScope});
  std::vector<std::vector<SyncCoverDemandId>> releaseDemands(
      cycle.lanes.size());
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    releaseDemands[lane] = getLaneOwnershipDemands(graph, cycle, lane);
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    result.actions.push_back(makeAction(
        CanonicalSyncActionKind::EventSet, cycle.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, cycle.recurrenceScope}, 0, lane));
  }
  std::vector<std::vector<UseActions>> actions(cycle.paths.size());
  for (std::size_t pathIndex = 0; pathIndex < cycle.paths.size(); ++pathIndex) {
    const CanonicalSyncOwnershipPath &path = cycle.paths[pathIndex];
    std::map<unsigned, std::size_t> previous;
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      const std::size_t wait = result.actions.size();
      result.actions.push_back(
          makeAction(CanonicalSyncActionKind::EventWait, cycle.producerResource,
                     use.writeAcquire, 0, use.producerLane));
      UseActions useActions;
      useActions.wait = wait;
      for (SyncCoverNodeId consumer : use.consumers) {
        const std::size_t set = result.actions.size();
        result.actions.push_back(makeAction(
            CanonicalSyncActionKind::EventSet, cycle.consumerResource,
            {SyncCoverAnchorKind::AfterNode, consumer, 0, 0}, 0, use.lane));
        useActions.sets.push_back({consumer, set});
      }
      actions[pathIndex].push_back(std::move(useActions));
      const auto prior = previous.find(use.lane);
      if (prior != previous.end()) {
        const UseActions &sourceActions = actions[pathIndex][prior->second];
        for (auto [consumer, sourceSet] : sourceActions.sets) {
          for (SyncCoverNodeId producer : use.producers) {
            addBinding(result,
                       makeEdge(graph, consumer, producer, path.scope, 0), 0,
                       sourceSet, wait, releaseDemands[use.lane]);
          }
        }
      }
      previous[use.lane] = actions[pathIndex].size() - 1;
    }
  }
  for (std::size_t sourcePath = 0; sourcePath < cycle.paths.size();
       ++sourcePath) {
    for (std::size_t targetPath = 0; targetPath < cycle.paths.size();
         ++targetPath) {
      for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
        std::optional<std::size_t> sourceUse;
        std::optional<std::size_t> targetUse;
        for (std::size_t index = 0; index < cycle.paths[sourcePath].uses.size();
             ++index) {
          if (cycle.paths[sourcePath].uses[index].lane == lane) {
            sourceUse = index;
          }
        }
        for (std::size_t index = 0; index < cycle.paths[targetPath].uses.size();
             ++index) {
          if (cycle.paths[targetPath].uses[index].producerLane == lane) {
            targetUse = index;
            break;
          }
        }
        if (!sourceUse || !targetUse) {
          continue;
        }
        const CanonicalSyncOwnershipUse &source =
            cycle.paths[sourcePath].uses[*sourceUse];
        const CanonicalSyncOwnershipUse &target =
            cycle.paths[targetPath].uses[*targetUse];
        for (SyncCoverNodeId consumer : source.consumers) {
          const auto &sourceSets = actions[sourcePath][*sourceUse].sets;
          const auto sourceSet = std::find_if(
              sourceSets.begin(), sourceSets.end(),
              [&](const auto &entry) { return entry.first == consumer; });
          if (sourceSet == sourceSets.end()) {
            continue;
          }
          for (SyncCoverNodeId producer : target.producers) {
            addBinding(
                result,
                makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1),
                0, sourceSet->second, actions[targetPath][*targetUse].wait,
                releaseDemands[lane]);
          }
        }
      }
    }
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    result.actions.push_back(makeAction(
        CanonicalSyncActionKind::EventWait, cycle.producerResource,
        {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope}, 0, lane));
  }
  return result;
}

bool edgeEqual(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return std::tie(left.source, left.target, left.kind, left.scope,
                  left.distance, left.sourceGuard.literals,
                  left.targetGuard.literals) ==
         std::tie(right.source, right.target, right.kind, right.scope,
                  right.distance, right.sourceGuard.literals,
                  right.targetGuard.literals);
}

bool bindingsEqual(ArrayRef<CanonicalSyncSupplyBinding> actual,
                   ArrayRef<CanonicalSyncSupplyBinding> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  std::vector<bool> matched(actual.size(), false);
  for (const CanonicalSyncSupplyBinding &wanted : expected) {
    bool found = false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const CanonicalSyncSupplyBinding &candidate = actual[index];
      const bool same =
          !matched[index] && edgeEqual(candidate.edge, wanted.edge) &&
          candidate.allowedDemands == wanted.allowedDemands &&
          candidate.eventUse == wanted.eventUse &&
          candidate.produceAction == wanted.produceAction &&
          candidate.consumeAction == wanted.consumeAction &&
          candidate.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          !candidate.barrierAction;
      if (same) {
        matched[index] = true;
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

bool anchorEqual(const SyncCoverAnchor &left, const SyncCoverAnchor &right) {
  return std::tie(left.kind, left.node, left.scope, left.position) ==
         std::tie(right.kind, right.node, right.scope, right.position);
}

bool isValidProtocolEdge(const SyncCoverGraph &graph,
                         const SyncCoverEdge &edge) {
  SyncCoverEdge checked = edge;
  return graph.canonicalizeCompletionEdge(checked) == SyncCoverGraphError::None;
}

std::optional<std::size_t>
claimAction(const CanonicalSyncMechanismDescriptor &descriptor,
            std::vector<bool> &claimed, CanonicalSyncActionKind kind,
            std::uint32_t resource, SyncCoverAnchor anchor,
            std::size_t eventUse, unsigned lane) {
  for (std::size_t index = 0; index < descriptor.actions.size(); ++index) {
    const CanonicalSyncAction &action = descriptor.actions[index];
    const bool matches =
        !claimed[index] && action.kind == kind && action.resource == resource &&
        anchorEqual(action.anchor, anchor) && action.eventUse == eventUse &&
        action.eventLane == lane && action.drainedResources.empty() &&
        action.barrierKind == CanonicalSyncBarrierKind::Targeted;
    if (matches) {
      claimed[index] = true;
      return index;
    }
  }
  return std::nullopt;
}

bool hasProtocolHeader(const CanonicalSyncMechanismDescriptor &descriptor,
                       CanonicalSyncEventDomainId domain, std::size_t uses,
                       std::size_t width, SyncCoverScopeId recurrenceScope) {
  return descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
         descriptor.eventUses.size() == uses &&
         llvm::all_of(descriptor.eventUses,
                      [&](const CanonicalSyncEventUse &use) {
                        return use.domain == domain && use.width == width &&
                               use.recurrenceScope == recurrenceScope;
                      });
}

bool allActionsClaimed(const std::vector<bool> &claimed) {
  return std::all_of(claimed.begin(), claimed.end(),
                     [](bool value) { return value; });
}

bool verifyReadyDescriptor(const SyncCoverGraph &graph,
                           const CanonicalSyncOwnershipCycle &cycle,
                           CanonicalSyncEventDomainId domain,
                           const CanonicalSyncMechanismDescriptor &descriptor) {
  const std::size_t producerCount =
      cycle.paths.front().uses.front().producers.size();
  if (!hasProtocolHeader(descriptor, domain, producerCount, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  CanonicalSyncMechanismDescriptor expected;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (use.producers.size() != producerCount) {
        return false;
      }
      for (auto [producerIndex, producer] : llvm::enumerate(use.producers)) {
        if (!syncCoverNodeCanProduceCompletion(graph, producer,
                                               cycle.consumerResource)) {
          return false;
        }
        const std::optional<std::size_t> set =
            claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                        cycle.producerResource,
                        {SyncCoverAnchorKind::AfterNode, producer, 0, 0},
                        producerIndex, use.lane);
        if (!set) {
          return false;
        }
        for (SyncCoverNodeId consumer : use.consumers) {
          const std::optional<std::size_t> wait = claimAction(
              descriptor, claimed, CanonicalSyncActionKind::EventWait,
              cycle.consumerResource,
              {SyncCoverAnchorKind::BeforeNode, consumer, 0, 0}, producerIndex,
              use.lane);
          if (!wait) {
            return false;
          }
          const SyncCoverScopeId scope =
              graph
                  .getLowestCommonScope(graph.getNodes()[producer].scope,
                                        graph.getNodes()[consumer].scope)
                  .value_or(path.scope);
          SyncCoverEdge edge = makeEdge(graph, producer, consumer, scope, 0);
          if (!isValidProtocolEdge(graph, edge)) {
            return false;
          }
          addBinding(expected, std::move(edge), producerIndex, *set, *wait);
        }
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyRoundTripDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  std::vector<std::vector<SyncCoverDemandId>> releaseDemands(
      cycle.lanes.size());
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    releaseDemands[lane] = getLaneOwnershipDemands(graph, cycle, lane);
    if (releaseDemands[lane].empty()) {
      return false;
    }
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    if (!claimAction(
            descriptor, claimed, CanonicalSyncActionKind::EventSet,
            cycle.consumerResource,
            {SyncCoverAnchorKind::ScopeEntry, 0, cycle.recurrenceScope}, 0,
            lane)) {
      return false;
    }
  }

  std::vector<std::vector<UseActions>> actions(cycle.paths.size());
  for (std::size_t pathIndex = 0; pathIndex < cycle.paths.size(); ++pathIndex) {
    for (const CanonicalSyncOwnershipUse &use : cycle.paths[pathIndex].uses) {
      const std::optional<std::size_t> wait = claimAction(
          descriptor, claimed, CanonicalSyncActionKind::EventWait,
          cycle.producerResource, use.writeAcquire, 0, use.producerLane);
      if (!wait) {
        return false;
      }
      UseActions useActions;
      useActions.wait = *wait;
      for (SyncCoverNodeId consumer : use.consumers) {
        const std::optional<std::size_t> set = claimAction(
            descriptor, claimed, CanonicalSyncActionKind::EventSet,
            cycle.consumerResource,
            {SyncCoverAnchorKind::AfterNode, consumer, 0, 0}, 0, use.lane);
        if (!set) {
          return false;
        }
        useActions.sets.push_back({consumer, *set});
      }
      actions[pathIndex].push_back(std::move(useActions));
    }
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    if (!claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                     cycle.producerResource,
                     {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope},
                     0, lane)) {
      return false;
    }
  }

  CanonicalSyncMechanismDescriptor expected;
  for (std::size_t pathIndex = 0; pathIndex < cycle.paths.size(); ++pathIndex) {
    std::map<unsigned, std::size_t> previous;
    const CanonicalSyncOwnershipPath &path = cycle.paths[pathIndex];
    for (std::size_t useIndex = 0; useIndex < path.uses.size(); ++useIndex) {
      const CanonicalSyncOwnershipUse &use = path.uses[useIndex];
      const auto prior = previous.find(use.lane);
      if (prior != previous.end()) {
        for (auto [consumer, set] : actions[pathIndex][prior->second].sets) {
          for (SyncCoverNodeId producer : use.producers) {
            SyncCoverEdge edge =
                makeEdge(graph, consumer, producer, path.scope, 0);
            if (!isValidProtocolEdge(graph, edge)) {
              return false;
            }
            addBinding(expected, std::move(edge), 0, set,
                       actions[pathIndex][useIndex].wait,
                       releaseDemands[use.lane]);
          }
        }
      }
      previous[use.lane] = useIndex;
    }
  }
  for (std::size_t sourcePath = 0; sourcePath < cycle.paths.size();
       ++sourcePath) {
    for (std::size_t targetPath = 0; targetPath < cycle.paths.size();
         ++targetPath) {
      for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
        std::optional<std::size_t> sourceUse;
        std::optional<std::size_t> targetUse;
        for (std::size_t index = 0; index < cycle.paths[sourcePath].uses.size();
             ++index) {
          if (cycle.paths[sourcePath].uses[index].lane == lane) {
            sourceUse = index;
          }
        }
        for (std::size_t index = 0; index < cycle.paths[targetPath].uses.size();
             ++index) {
          if (cycle.paths[targetPath].uses[index].producerLane == lane) {
            targetUse = index;
            break;
          }
        }
        if (!sourceUse || !targetUse) {
          continue;
        }
        for (auto [consumer, set] : actions[sourcePath][*sourceUse].sets) {
          for (SyncCoverNodeId producer :
               cycle.paths[targetPath].uses[*targetUse].producers) {
            // PIPE_M cannot certify MMAD completion. This verified ownership
            // protocol instead proves the narrower fact needed here: after
            // issue, the MMAD no longer owns its exact L0 operand slots.
            SyncCoverEdge edge =
                makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1);
            if (!isValidProtocolEdge(graph, edge)) {
              return false;
            }
            addBinding(expected, std::move(edge), 0, set,
                       actions[targetPath][*targetUse].wait,
                       releaseDemands[lane]);
          }
        }
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

} // namespace

std::optional<CanonicalSyncOwnershipProtocol>
mlir::pto::makeCanonicalSyncOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain) {
  if (!verifyCanonicalSyncOwnershipCycle(program, cycle)) {
    return std::nullopt;
  }
  CanonicalSyncOwnershipProtocol result;
  result.ready = buildReady(program.getGraph(), cycle, readyDomain);
  result.release =
      buildRoundTripRelease(program.getGraph(), cycle, releaseDomain);
  return result;
}

bool mlir::pto::verifyCanonicalSyncOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncOwnershipProtocol &protocol) {
  if (!verifyCanonicalSyncOwnershipCycle(program, cycle)) {
    return false;
  }
  const bool ready = verifyReadyDescriptor(program.getGraph(), cycle,
                                           readyDomain, protocol.ready);
  const bool release = verifyRoundTripDescriptor(
      program.getGraph(), cycle, releaseDomain, protocol.release);
  return ready && release;
}
