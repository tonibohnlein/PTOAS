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
#include <set>
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

CanonicalSyncAction makeAction(
    CanonicalSyncActionKind kind, std::uint32_t resource,
    SyncCoverAnchor anchor, std::size_t eventUse, unsigned lane,
    CanonicalSyncActionGuardKind guard = CanonicalSyncActionGuardKind::None,
    std::optional<SyncCoverScopeId> guardScope = std::nullopt) {
  CanonicalSyncAction result;
  result.kind = kind;
  result.resource = resource;
  result.anchor = anchor;
  result.eventUse = eventUse;
  result.eventLane = lane;
  result.guard = guard;
  result.guardScope = guardScope;
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

void addCompositeBinding(CanonicalSyncMechanismDescriptor &descriptor,
                         SyncCoverEdge edge,
                         ArrayRef<SyncCoverDemandId> allowedDemands) {
  CanonicalSyncSupplyBinding binding;
  binding.edge = std::move(edge);
  binding.proof = CanonicalSyncSupplyProof::VerifiedCompositeProtocol;
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

std::vector<SyncCoverDemandId>
getLaneWawDemands(const SyncCoverGraph &graph,
                  const CanonicalSyncOwnershipCycle &cycle, unsigned lane) {
  std::vector<SyncCoverDemandId> result =
      getLaneOwnershipDemands(graph, cycle, lane);
  llvm::erase_if(result, [&](SyncCoverDemandId demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    return demand.provenanceKinds.empty() ||
           !llvm::all_of(demand.provenanceKinds, [](SyncCoverDemandKind kind) {
             return kind == SyncCoverDemandKind::MemoryWAW;
           });
  });
  return result;
}

bool demandMatchesDerivedWaw(const SyncCoverGraph &graph,
                             const CanonicalSyncOwnershipCycle &cycle,
                             const SyncCoverDemand &demand,
                             const SyncCoverEdge &derived) {
  const bool sameEdge =
      demand.source == derived.source && demand.target == derived.target &&
      demand.scope == derived.scope && demand.distance == derived.distance &&
      demand.sourceGuard.literals == derived.sourceGuard.literals &&
      demand.targetGuard.literals == derived.targetGuard.literals;
  const bool waw = !demand.provenanceKinds.empty() &&
                   llvm::all_of(demand.provenanceKinds, [](auto kind) {
                     return kind == SyncCoverDemandKind::MemoryWAW;
                   });
  if (!sameEdge || !waw || demand.storageWitnesses.empty()) {
    return false;
  }
  return llvm::all_of(
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
        const bool exactWrites =
            source.exactPhysical && target.exactPhysical &&
            source.mode == SyncCoverStorageAccessMode::Write &&
            target.mode == SyncCoverStorageAccessMode::Write;
        const bool managed = llvm::any_of(
            cycle.lanes, [&](const CanonicalSyncOwnershipLane &lane) {
              return llvm::any_of(
                  lane.slots, [&](const CanonicalSyncOwnershipSlot &slot) {
                    return source.domain == slot.domain &&
                           target.domain == slot.domain &&
                           intervalContains(slot.extent, witness.overlap);
                  });
            });
        return exactWrites && managed;
      });
}

void addAlternatingCompositeBindings(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const CanonicalSyncOwnershipCycle &cycle,
    const CanonicalSyncOwnershipProtocol &protocol) {
  std::set<SyncCoverDemandId> matched;
  std::set<SyncCoverNodeId> initialProducers(cycle.initialProducers.begin(),
                                             cycle.initialProducers.end());
  std::set<SyncCoverNodeId> bodyProducers;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      bodyProducers.insert(use.producers.begin(), use.producers.end());
    }
  }
  for (const CanonicalSyncSupplyBinding &ready : protocol.ready.supplies) {
    for (const CanonicalSyncSupplyBinding &release :
         protocol.release.supplies) {
      const bool composable =
          ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          ready.edge.target == release.edge.source &&
          ready.edge.targetGuard.literals == release.edge.sourceGuard.literals;
      if (!composable) {
        continue;
      }
      const bool initialTransition =
          initialProducers.count(ready.edge.source) != 0 &&
          bodyProducers.count(release.edge.target) != 0;
      const bool bodyTransition = ready.edge.source == release.edge.target &&
                                  bodyProducers.count(ready.edge.source) != 0;
      if (!initialTransition && !bodyTransition) {
        continue;
      }
      const unsigned distance = bodyTransition ? 1 : 0;
      const SyncCoverScopeId scope =
          bodyTransition ? cycle.recurrenceScope
                         : graph
                               .getLowestCommonScope(
                                   graph.getNodes()[ready.edge.source].scope,
                                   graph.getNodes()[release.edge.target].scope)
                               .value_or(cycle.recurrenceScope);
      // The verified phase relation maps these two physical legs onto the next
      // execution of one lane producer. Guarded body executions therefore have
      // occurrence distance one; the unique preloop producer has distance zero.
      SyncCoverEdge derived{ready.edge.source,
                            release.edge.target,
                            SyncCoverEdgeKind::CompletionSupply,
                            scope,
                            distance,
                            ready.edge.sourceGuard,
                            release.edge.targetGuard};
      for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
        if (matched.count(demandId) == 0 &&
            demandMatchesDerivedWaw(graph, cycle, demand, derived)) {
          matched.insert(demandId);
          addCompositeBinding(descriptor, derived,
                              ArrayRef<SyncCoverDemandId>(&demandId, 1));
        }
      }
    }
  }
}

CanonicalSyncMechanismDescriptor
buildL0Ready(const SyncCoverGraph &graph,
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
buildStableL1Ready(const SyncCoverGraph &graph,
                   const CanonicalSyncOwnershipCycle &cycle,
                   CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back(
      {domain, cycle.lanes.size(), cycle.recurrenceScope});
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      const std::size_t set = result.actions.size();
      result.actions.push_back(makeAction(CanonicalSyncActionKind::EventSet,
                                          cycle.producerResource, use.ready, 0,
                                          use.lane));
      const std::size_t wait = result.actions.size();
      result.actions.push_back(makeAction(CanonicalSyncActionKind::EventWait,
                                          cycle.consumerResource,
                                          use.readAcquire, 0, use.lane));
      for (SyncCoverNodeId producer : use.producers) {
        for (SyncCoverNodeId consumer : use.consumers) {
          const SyncCoverScopeId scope =
              graph
                  .getLowestCommonScope(graph.getNodes()[producer].scope,
                                        graph.getNodes()[consumer].scope)
                  .value_or(path.scope);
          addBinding(result, makeEdge(graph, producer, consumer, scope, 0), 0,
                     set, wait);
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor
buildL0RoundTripRelease(const SyncCoverGraph &graph,
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

CanonicalSyncMechanismDescriptor
buildStableL1Release(const SyncCoverGraph &graph,
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
      UseActions useActions;
      useActions.wait = result.actions.size();
      result.actions.push_back(
          makeAction(CanonicalSyncActionKind::EventWait, cycle.producerResource,
                     use.writeAcquire, 0, use.producerLane));
      const std::size_t set = result.actions.size();
      result.actions.push_back(makeAction(CanonicalSyncActionKind::EventSet,
                                          cycle.consumerResource, use.release,
                                          0, use.lane));
      for (SyncCoverNodeId consumer : use.consumers) {
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
                       sourceSet, actions[pathIndex].back().wait,
                       releaseDemands[use.lane]);
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
        const std::size_t set =
            actions[sourcePath][*sourceUse].sets.front().second;
        for (SyncCoverNodeId consumer : source.consumers) {
          for (SyncCoverNodeId producer : target.producers) {
            addBinding(
                result,
                makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1),
                0, set, actions[targetPath][*targetUse].wait,
                releaseDemands[lane]);
          }
        }
        const std::vector<SyncCoverDemandId> wawDemands =
            getLaneWawDemands(graph, cycle, lane);
        if (!wawDemands.empty()) {
          for (SyncCoverNodeId sourceProducer : source.producers) {
            for (SyncCoverNodeId targetProducer : target.producers) {
              addCompositeBinding(result,
                                  makeEdge(graph, sourceProducer,
                                           targetProducer,
                                           cycle.recurrenceScope, 1),
                                  wawDemands);
            }
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

CanonicalSyncMechanismDescriptor
buildAlternatingReady(const SyncCoverGraph &graph,
                      const CanonicalSyncOwnershipCycle &cycle,
                      CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back(
      {domain, cycle.lanes.size(), cycle.recurrenceScope});
  const std::size_t initialSet = result.actions.size();
  result.actions.push_back(makeAction(
      CanonicalSyncActionKind::EventSet, cycle.producerResource,
      cycle.initialReady, 0, cycle.initialReadyLane,
      CanonicalSyncActionGuardKind::LoopNonEmpty, cycle.recurrenceScope));
  std::vector<std::size_t> waits(cycle.paths.size());
  std::vector<std::size_t> sets(cycle.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    waits[pathIndex] = result.actions.size();
    result.actions.push_back(makeAction(CanonicalSyncActionKind::EventWait,
                                        cycle.consumerResource, use.readAcquire,
                                        0, use.lane));
    sets[pathIndex] = result.actions.size();
    result.actions.push_back(makeAction(CanonicalSyncActionKind::EventSet,
                                        cycle.producerResource, use.ready, 0,
                                        use.producerLane));
  }
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    if (use.lane == cycle.initialReadyLane) {
      for (SyncCoverNodeId producer : cycle.initialProducers) {
        for (SyncCoverNodeId consumer : use.consumers) {
          const SyncCoverScopeId scope =
              graph
                  .getLowestCommonScope(graph.getNodes()[producer].scope,
                                        graph.getNodes()[consumer].scope)
                  .value_or(cycle.recurrenceScope);
          addBinding(result, makeEdge(graph, producer, consumer, scope, 0), 0,
                     initialSet, waits[pathIndex]);
        }
      }
    }
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalSyncOwnershipUse &target = targetPath.uses.front();
      if (target.lane != use.producerLane) {
        continue;
      }
      for (SyncCoverNodeId producer : use.producers) {
        for (SyncCoverNodeId consumer : target.consumers) {
          addBinding(
              result,
              makeEdge(graph, producer, consumer, cycle.recurrenceScope, 1), 0,
              sets[pathIndex], waits[targetIndex]);
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor
buildAlternatingRelease(const SyncCoverGraph &graph,
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
  for (unsigned lane : cycle.initiallyFreeLanes) {
    result.actions.push_back(makeAction(
        CanonicalSyncActionKind::EventSet, cycle.consumerResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, cycle.recurrenceScope}, 0, lane,
        CanonicalSyncActionGuardKind::LoopNonEmpty, cycle.recurrenceScope));
  }
  std::vector<std::size_t> sets(cycle.paths.size());
  std::vector<std::size_t> waits(cycle.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    sets[pathIndex] = result.actions.size();
    result.actions.push_back(makeAction(CanonicalSyncActionKind::EventSet,
                                        cycle.consumerResource, use.release, 0,
                                        use.lane));
    waits[pathIndex] = result.actions.size();
    result.actions.push_back(makeAction(CanonicalSyncActionKind::EventWait,
                                        cycle.producerResource,
                                        use.writeAcquire, 0, use.producerLane));
  }
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalSyncOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        for (SyncCoverNodeId producer : target.producers) {
          addBinding(
              result,
              makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1), 0,
              sets[pathIndex], waits[targetIndex], releaseDemands[use.lane]);
        }
      }
    }
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    result.actions.push_back(makeAction(
        CanonicalSyncActionKind::EventWait, cycle.producerResource,
        {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope}, 0, lane,
        CanonicalSyncActionGuardKind::LoopNonEmpty, cycle.recurrenceScope));
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
          candidate.proof == wanted.proof && !candidate.barrierAction;
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

std::optional<std::size_t> claimAction(
    const CanonicalSyncMechanismDescriptor &descriptor,
    std::vector<bool> &claimed, CanonicalSyncActionKind kind,
    std::uint32_t resource, SyncCoverAnchor anchor, std::size_t eventUse,
    unsigned lane,
    CanonicalSyncActionGuardKind guard = CanonicalSyncActionGuardKind::None,
    std::optional<SyncCoverScopeId> guardScope = std::nullopt) {
  for (std::size_t index = 0; index < descriptor.actions.size(); ++index) {
    const CanonicalSyncAction &action = descriptor.actions[index];
    const bool matches =
        !claimed[index] && action.kind == kind && action.resource == resource &&
        anchorEqual(action.anchor, anchor) && action.eventUse == eventUse &&
        action.eventLane == lane && action.drainedResources.empty() &&
        action.barrierKind == CanonicalSyncBarrierKind::Targeted &&
        action.guard == guard && action.guardScope == guardScope;
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

bool verifyAlternatingReadyDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope) ||
      cycle.lanes.size() != 2 || cycle.paths.size() != 2 ||
      cycle.initialProducers.size() != 1) {
    return false;
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  const std::optional<std::size_t> initialSet = claimAction(
      descriptor, claimed, CanonicalSyncActionKind::EventSet,
      cycle.producerResource, cycle.initialReady, 0, cycle.initialReadyLane,
      CanonicalSyncActionGuardKind::LoopNonEmpty, cycle.recurrenceScope);
  if (!initialSet) {
    return false;
  }
  for (SyncCoverNodeId producer : cycle.initialProducers) {
    if (!syncCoverNodeCanProduceCompletion(graph, producer,
                                           cycle.consumerResource)) {
      return false;
    }
  }
  std::vector<std::size_t> waits(cycle.paths.size());
  std::vector<std::size_t> sets(cycle.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    if (path.uses.size() != 1) {
      return false;
    }
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    if (!llvm::all_of(use.producers, [&](SyncCoverNodeId producer) {
          return syncCoverNodeCanProduceCompletion(graph, producer,
                                                   cycle.consumerResource);
        })) {
      return false;
    }
    const auto wait =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                    cycle.consumerResource, use.readAcquire, 0, use.lane);
    const auto set =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                    cycle.producerResource, use.ready, 0, use.producerLane);
    if (!wait || !set) {
      return false;
    }
    waits[pathIndex] = *wait;
    sets[pathIndex] = *set;
  }
  CanonicalSyncMechanismDescriptor expected;
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    if (use.lane == cycle.initialReadyLane) {
      for (SyncCoverNodeId producer : cycle.initialProducers) {
        for (SyncCoverNodeId consumer : use.consumers) {
          const SyncCoverScopeId scope =
              graph
                  .getLowestCommonScope(graph.getNodes()[producer].scope,
                                        graph.getNodes()[consumer].scope)
                  .value_or(cycle.recurrenceScope);
          SyncCoverEdge edge = makeEdge(graph, producer, consumer, scope, 0);
          if (!isValidProtocolEdge(graph, edge)) {
            return false;
          }
          addBinding(expected, std::move(edge), 0, *initialSet,
                     waits[pathIndex]);
        }
      }
    }
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalSyncOwnershipUse &target = targetPath.uses.front();
      if (target.lane != use.producerLane) {
        continue;
      }
      for (SyncCoverNodeId producer : use.producers) {
        for (SyncCoverNodeId consumer : target.consumers) {
          SyncCoverEdge edge =
              makeEdge(graph, producer, consumer, cycle.recurrenceScope, 1);
          if (!isValidProtocolEdge(graph, edge)) {
            return false;
          }
          addBinding(expected, std::move(edge), 0, sets[pathIndex],
                     waits[targetIndex]);
        }
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyAlternatingReleaseDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope) ||
      cycle.lanes.size() != 2 || cycle.paths.size() != 2 ||
      cycle.initiallyFreeLanes.size() != 1) {
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
  for (unsigned lane : cycle.initiallyFreeLanes) {
    if (!claimAction(
            descriptor, claimed, CanonicalSyncActionKind::EventSet,
            cycle.consumerResource,
            {SyncCoverAnchorKind::ScopeEntry, 0, cycle.recurrenceScope}, 0,
            lane, CanonicalSyncActionGuardKind::LoopNonEmpty,
            cycle.recurrenceScope)) {
      return false;
    }
  }
  std::vector<std::size_t> sets(cycle.paths.size());
  std::vector<std::size_t> waits(cycle.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    if (path.uses.size() != 1) {
      return false;
    }
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    const auto set =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                    cycle.consumerResource, use.release, 0, use.lane);
    const auto wait = claimAction(
        descriptor, claimed, CanonicalSyncActionKind::EventWait,
        cycle.producerResource, use.writeAcquire, 0, use.producerLane);
    if (!set || !wait) {
      return false;
    }
    sets[pathIndex] = *set;
    waits[pathIndex] = *wait;
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    if (!claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                     cycle.producerResource,
                     {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope},
                     0, lane, CanonicalSyncActionGuardKind::LoopNonEmpty,
                     cycle.recurrenceScope)) {
      return false;
    }
  }
  CanonicalSyncMechanismDescriptor expected;
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    if (!llvm::all_of(use.consumers, [&](SyncCoverNodeId consumer) {
          return syncCoverNodeCanProduceCompletion(graph, consumer,
                                                   cycle.producerResource);
        })) {
      return false;
    }
    for (auto [targetIndex, targetPath] : llvm::enumerate(cycle.paths)) {
      const CanonicalSyncOwnershipUse &target = targetPath.uses.front();
      if (target.producerLane != use.lane) {
        continue;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        for (SyncCoverNodeId producer : target.producers) {
          SyncCoverEdge edge =
              makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1);
          if (!isValidProtocolEdge(graph, edge)) {
            return false;
          }
          addBinding(expected, std::move(edge), 0, sets[pathIndex],
                     waits[targetIndex], releaseDemands[use.lane]);
        }
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyStableL1ReadyDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  CanonicalSyncMechanismDescriptor expected;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (use.producers.size() != 1) {
        return false;
      }
      const SyncCoverNodeId producer = use.producers.front();
      if (!syncCoverNodeCanProduceCompletion(graph, producer,
                                             cycle.consumerResource)) {
        return false;
      }
      const std::optional<std::size_t> set =
          claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                      cycle.producerResource, use.ready, 0, use.lane);
      const std::optional<std::size_t> wait =
          claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                      cycle.consumerResource, use.readAcquire, 0, use.lane);
      if (!set || !wait) {
        return false;
      }
      for (SyncCoverNodeId consumer : use.consumers) {
        const SyncCoverScopeId scope =
            graph
                .getLowestCommonScope(graph.getNodes()[producer].scope,
                                      graph.getNodes()[consumer].scope)
                .value_or(path.scope);
        SyncCoverEdge edge = makeEdge(graph, producer, consumer, scope, 0);
        if (!isValidProtocolEdge(graph, edge)) {
          return false;
        }
        addBinding(expected, std::move(edge), 0, *set, *wait);
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyStableL1ReleaseDescriptor(
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
      const std::optional<std::size_t> set =
          claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                      cycle.consumerResource, use.release, 0, use.lane);
      if (!wait || !set) {
        return false;
      }
      UseActions useActions;
      useActions.wait = *wait;
      for (SyncCoverNodeId consumer : use.consumers) {
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
        const std::vector<SyncCoverDemandId> wawDemands =
            getLaneWawDemands(graph, cycle, lane);
        if (!wawDemands.empty()) {
          for (SyncCoverNodeId sourceProducer :
               cycle.paths[sourcePath].uses[*sourceUse].producers) {
            for (SyncCoverNodeId targetProducer :
                 cycle.paths[targetPath].uses[*targetUse].producers) {
              SyncCoverEdge edge =
                  makeEdge(graph, sourceProducer, targetProducer,
                           cycle.recurrenceScope, 1);
              if (!isValidProtocolEdge(graph, edge)) {
                return false;
              }
              addCompositeBinding(expected, std::move(edge), wawDemands);
            }
          }
        }
      }
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyL0ReadyDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
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

bool verifyL0RoundTripDescriptor(
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

CanonicalSyncMechanismDescriptor
mergeOwnershipProtocol(const CanonicalSyncOwnershipProtocol &protocol) {
  CanonicalSyncMechanismDescriptor result = protocol.ready;
  const std::size_t useOffset = result.eventUses.size();
  const std::size_t actionOffset = result.actions.size();
  result.eventUses.insert(result.eventUses.end(),
                          protocol.release.eventUses.begin(),
                          protocol.release.eventUses.end());
  for (CanonicalSyncAction action : protocol.release.actions) {
    if (action.eventUse) {
      *action.eventUse += useOffset;
    }
    result.actions.push_back(std::move(action));
  }
  for (CanonicalSyncSupplyBinding binding : protocol.release.supplies) {
    if (binding.eventUse) {
      *binding.eventUse += useOffset;
    }
    if (binding.produceAction) {
      *binding.produceAction += actionOffset;
    }
    if (binding.consumeAction) {
      *binding.consumeAction += actionOffset;
    }
    result.supplies.push_back(std::move(binding));
  }
  return result;
}

std::optional<CanonicalSyncOwnershipProtocol>
splitOwnershipProtocol(const CanonicalSyncMechanismDescriptor &descriptor,
                       std::size_t readyUses) {
  if (descriptor.kind != CanonicalSyncMechanismKind::Protocol ||
      readyUses == 0 || descriptor.eventUses.size() <= readyUses) {
    return std::nullopt;
  }
  CanonicalSyncOwnershipProtocol result;
  result.ready.kind = CanonicalSyncMechanismKind::Protocol;
  result.release.kind = CanonicalSyncMechanismKind::Protocol;
  result.ready.eventUses.assign(descriptor.eventUses.begin(),
                                descriptor.eventUses.begin() + readyUses);
  result.release.eventUses.assign(descriptor.eventUses.begin() + readyUses,
                                  descriptor.eventUses.end());

  std::vector<std::optional<std::size_t>> readyActions(
      descriptor.actions.size());
  std::vector<std::optional<std::size_t>> releaseActions(
      descriptor.actions.size());
  for (auto [index, action] : llvm::enumerate(descriptor.actions)) {
    if (!action.eventUse || *action.eventUse >= descriptor.eventUses.size() ||
        action.barrierKind != CanonicalSyncBarrierKind::Targeted ||
        !action.drainedResources.empty()) {
      return std::nullopt;
    }
    CanonicalSyncAction copy = action;
    if (*action.eventUse < readyUses) {
      readyActions[index] = result.ready.actions.size();
      result.ready.actions.push_back(std::move(copy));
    } else {
      *copy.eventUse -= readyUses;
      releaseActions[index] = result.release.actions.size();
      result.release.actions.push_back(std::move(copy));
    }
  }

  for (CanonicalSyncSupplyBinding binding : descriptor.supplies) {
    if (binding.proof == CanonicalSyncSupplyProof::VerifiedCompositeProtocol) {
      if (binding.eventUse || binding.barrierAction || binding.produceAction ||
          binding.consumeAction) {
        return std::nullopt;
      }
      result.release.supplies.push_back(std::move(binding));
      continue;
    }
    if (!binding.eventUse || *binding.eventUse >= descriptor.eventUses.size() ||
        !binding.produceAction || !binding.consumeAction ||
        *binding.produceAction >= descriptor.actions.size() ||
        *binding.consumeAction >= descriptor.actions.size() ||
        binding.barrierAction) {
      return std::nullopt;
    }
    const bool ready = *binding.eventUse < readyUses;
    const auto &actionMap = ready ? readyActions : releaseActions;
    if (!actionMap[*binding.produceAction] ||
        !actionMap[*binding.consumeAction]) {
      return std::nullopt;
    }
    binding.produceAction = actionMap[*binding.produceAction];
    binding.consumeAction = actionMap[*binding.consumeAction];
    if (!ready) {
      *binding.eventUse -= readyUses;
    }
    (ready ? result.ready.supplies : result.release.supplies)
        .push_back(std::move(binding));
  }
  return result;
}

} // namespace

namespace {

bool requiresMte1ScopePrefix(const CanonicalSyncOwnershipCycle &cycle) {
  if (cycle.kind != CanonicalSyncOwnershipKind::L1Tile ||
      cycle.protocol != CanonicalSyncOwnershipProtocolKind::RoundTrip) {
    return false;
  }
  return llvm::any_of(cycle.paths, [](const auto &path) {
    return llvm::any_of(path.uses, [](const auto &use) {
      return use.release.kind == SyncCoverAnchorKind::ScopeExit;
    });
  });
}

} // namespace

std::optional<CanonicalSyncOwnershipProtocol>
mlir::pto::makeCanonicalSyncOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain) {
  if (requiresMte1ScopePrefix(cycle) &&
      !program.getTargetCapabilities().mte1ScopeExitSetCompletesPrefix) {
    return std::nullopt;
  }
  if (!verifyCanonicalSyncOwnershipCycle(program, cycle)) {
    return std::nullopt;
  }
  CanonicalSyncOwnershipProtocol result;
  if (cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch) {
    result.ready =
        buildAlternatingReady(program.getGraph(), cycle, readyDomain);
    result.release =
        buildAlternatingRelease(program.getGraph(), cycle, releaseDomain);
  } else if (cycle.kind == CanonicalSyncOwnershipKind::L1Tile) {
    result.ready = buildStableL1Ready(program.getGraph(), cycle, readyDomain);
    result.release =
        buildStableL1Release(program.getGraph(), cycle, releaseDomain);
  } else {
    result.ready = buildL0Ready(program.getGraph(), cycle, readyDomain);
    result.release =
        buildL0RoundTripRelease(program.getGraph(), cycle, releaseDomain);
  }
  return result;
}

bool mlir::pto::verifyCanonicalSyncOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncOwnershipProtocol &protocol) {
  if (requiresMte1ScopePrefix(cycle) &&
      !program.getTargetCapabilities().mte1ScopeExitSetCompletesPrefix) {
    return false;
  }
  if (!verifyCanonicalSyncOwnershipCycle(program, cycle)) {
    return false;
  }
  const bool alternating =
      cycle.protocol == CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
  const bool l1 = cycle.kind == CanonicalSyncOwnershipKind::L1Tile;
  const bool ready =
      alternating ? verifyAlternatingReadyDescriptor(
                        program.getGraph(), cycle, readyDomain, protocol.ready)
      : l1 ? verifyStableL1ReadyDescriptor(program.getGraph(), cycle,
                                           readyDomain, protocol.ready)
           : verifyL0ReadyDescriptor(program.getGraph(), cycle, readyDomain,
                                     protocol.ready);
  const bool release =
      alternating
          ? verifyAlternatingReleaseDescriptor(program.getGraph(), cycle,
                                               releaseDomain, protocol.release)
      : l1 ? verifyStableL1ReleaseDescriptor(program.getGraph(), cycle,
                                             releaseDomain, protocol.release)
           : verifyL0RoundTripDescriptor(program.getGraph(), cycle,
                                         releaseDomain, protocol.release);
  return ready && release;
}

std::optional<CanonicalSyncMechanismDescriptor>
mlir::pto::makeCanonicalSyncAtomicOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain) {
  std::optional<CanonicalSyncOwnershipProtocol> protocol =
      makeCanonicalSyncOwnershipProtocol(program, cycle, readyDomain,
                                         releaseDomain);
  if (!protocol || !verifyCanonicalSyncOwnershipProtocol(
                       program, cycle, readyDomain, releaseDomain, *protocol)) {
    return std::nullopt;
  }
  CanonicalSyncMechanismDescriptor result = mergeOwnershipProtocol(*protocol);
  if (cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch) {
    addAlternatingCompositeBindings(result, program.getGraph(), cycle,
                                    *protocol);
  }
  return result;
}

bool mlir::pto::verifyCanonicalSyncAtomicOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!verifyCanonicalSyncOwnershipCycle(program, cycle) ||
      cycle.paths.empty() || cycle.paths.front().uses.empty()) {
    return false;
  }
  CanonicalSyncMechanismDescriptor eventDescriptor = descriptor;
  std::vector<CanonicalSyncSupplyBinding> composite;
  if (cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch) {
    llvm::erase_if(eventDescriptor.supplies,
                   [&](const CanonicalSyncSupplyBinding &binding) {
                     if (binding.proof !=
                         CanonicalSyncSupplyProof::VerifiedCompositeProtocol) {
                       return false;
                     }
                     composite.push_back(binding);
                     return true;
                   });
  }
  const std::size_t readyUses =
      cycle.kind == CanonicalSyncOwnershipKind::L1Tile
          ? 1
          : cycle.paths.front().uses.front().producers.size();
  const std::optional<CanonicalSyncOwnershipProtocol> protocol =
      splitOwnershipProtocol(eventDescriptor, readyUses);
  if (!protocol || !verifyCanonicalSyncOwnershipProtocol(
                       program, cycle, readyDomain, releaseDomain, *protocol)) {
    return false;
  }
  if (cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch) {
    CanonicalSyncMechanismDescriptor expected;
    addAlternatingCompositeBindings(expected, program.getGraph(), cycle,
                                    *protocol);
    if (!bindingsEqual(composite, expected.supplies)) {
      return false;
    }
  }
  return true;
}
