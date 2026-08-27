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
#include <iterator>
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

void addOwnershipClosureBinding(CanonicalSyncMechanismDescriptor &descriptor,
                                SyncCoverEdge edge,
                                SyncCoverDemandId allowedDemand) {
  CanonicalSyncSupplyBinding binding;
  binding.edge = std::move(edge);
  binding.proof = CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
  binding.allowedDemands.push_back(allowedDemand);
  descriptor.supplies.push_back(std::move(binding));
}

void addNestedRecurrenceSummaryBinding(
    CanonicalSyncMechanismDescriptor &descriptor, SyncCoverEdge edge,
    SyncCoverDemandId allowedDemand) {
  CanonicalSyncSupplyBinding binding;
  binding.edge = std::move(edge);
  binding.proof = CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary;
  binding.allowedDemands.push_back(allowedDemand);
  descriptor.supplies.push_back(std::move(binding));
}

std::optional<std::size_t>
findAction(const CanonicalSyncMechanismDescriptor &descriptor,
           CanonicalSyncActionKind kind, std::uint32_t resource,
           const SyncCoverAnchor &anchor, unsigned lane) {
  std::optional<std::size_t> result;
  for (auto [index, action] : llvm::enumerate(descriptor.actions)) {
    if (action.kind != kind || action.resource != resource ||
        std::tie(action.anchor.kind, action.anchor.node, action.anchor.scope,
                 action.anchor.position) !=
            std::tie(anchor.kind, anchor.node, anchor.scope, anchor.position) ||
        action.eventUse != 0 || action.eventLane != lane) {
      continue;
    }
    if (result) {
      return std::nullopt;
    }
    result = index;
  }
  return result;
}

bool intervalContains(SyncCoverStorageInterval extent,
                      SyncCoverStorageInterval overlap) {
  return extent.begin <= overlap.begin && overlap.end <= extent.end;
}

std::vector<SyncCoverDemandId>
getLaneOwnershipDemands(const SyncCoverGraph &graph,
                        const CanonicalSyncOwnershipCycle &cycle, unsigned lane,
                        ArrayRef<SyncCoverNodeId> additionalProducers = {}) {
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
  producers.insert(producers.end(), additionalProducers.begin(),
                   additionalProducers.end());
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
              (syncCoverStorageModeReads(source.mode) &&
               std::binary_search(consumers.begin(), consumers.end(),
                                  source.node)) ||
              (syncCoverStorageModeWrites(source.mode) &&
               std::binary_search(producers.begin(), producers.end(),
                                  source.node));
          const bool roles = source.exactPhysical && target.exactPhysical &&
                             sourceOwnsSlot &&
                             syncCoverStorageModeWrites(target.mode) &&
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

std::vector<SyncCoverDemandId> getAccumulatorReadyDemands(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    unsigned lane, SyncCoverNodeId producer, SyncCoverNodeId consumer) {
  std::vector<SyncCoverDemandId> result;
  for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
    const bool exactEdge =
        demand.source == producer && demand.target == consumer;
    const bool raw = !demand.provenanceKinds.empty() &&
                     llvm::all_of(demand.provenanceKinds, [](auto kind) {
                       return kind == SyncCoverDemandKind::MemoryRAW;
                     });
    if (!exactEdge || !raw || demand.storageWitnesses.empty()) {
      continue;
    }
    const bool exactManaged = llvm::all_of(
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
          const bool roles = source.node == producer &&
                             target.node == consumer && source.exactPhysical &&
                             target.exactPhysical &&
                             syncCoverStorageModeWrites(source.mode) &&
                             target.mode == SyncCoverStorageAccessMode::Read;
          const bool managed = llvm::any_of(
              cycle.lanes[lane].slots,
              [&](const CanonicalSyncOwnershipSlot &slot) {
                return source.domain == slot.domain &&
                       target.domain == slot.domain &&
                       intervalContains(slot.extent, witness.overlap);
              });
          return roles && managed;
        });
    if (exactManaged) {
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

bool demandMatchesDerivedWriteOwnership(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    const SyncCoverDemand &demand, const SyncCoverEdge &derived) {
  const bool sameEdge =
      demand.source == derived.source && demand.target == derived.target &&
      demand.scope == derived.scope && demand.distance == derived.distance &&
      demand.sourceGuard.literals == derived.sourceGuard.literals &&
      demand.targetGuard.literals == derived.targetGuard.literals;
  const bool writeOwnership =
      llvm::is_contained(demand.provenanceKinds,
                         SyncCoverDemandKind::MemoryWAW) &&
      llvm::all_of(demand.provenanceKinds, [](auto kind) {
        return kind == SyncCoverDemandKind::MemoryWAR ||
               kind == SyncCoverDemandKind::MemoryWAW;
      });
  if (!sameEdge || !writeOwnership || demand.storageWitnesses.empty()) {
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
        const bool exactWrites = source.exactPhysical && target.exactPhysical &&
                                 syncCoverStorageModeWrites(source.mode) &&
                                 syncCoverStorageModeWrites(target.mode);
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
            demandMatchesDerivedWriteOwnership(graph, cycle, demand, derived)) {
          matched.insert(demandId);
          addCompositeBinding(descriptor, derived,
                              ArrayRef<SyncCoverDemandId>(&demandId, 1));
        }
      }
    }
  }
}

void addAccumulatorCompositeBindings(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const CanonicalSyncOwnershipCycle &cycle,
    const CanonicalSyncOwnershipProtocol &protocol) {
  std::set<SyncCoverDemandId> matched;
  for (const CanonicalSyncSupplyBinding &ready : protocol.ready.supplies) {
    for (const CanonicalSyncSupplyBinding &release :
         protocol.release.supplies) {
      const bool composable =
          ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          ready.edge.target == release.edge.source;
      if (!composable) {
        continue;
      }
      SyncCoverEdge derived{ready.edge.source,
                            release.edge.target,
                            SyncCoverEdgeKind::CompletionSupply,
                            cycle.recurrenceScope,
                            1,
                            ready.edge.sourceGuard,
                            release.edge.targetGuard};
      for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
        if (matched.count(demandId) == 0 &&
            demandMatchesDerivedWriteOwnership(graph, cycle, demand, derived)) {
          matched.insert(demandId);
          addCompositeBinding(descriptor, derived,
                              ArrayRef<SyncCoverDemandId>(&demandId, 1));
        }
      }
    }
  }
}

void addL0CompositeBindings(CanonicalSyncMechanismDescriptor &descriptor,
                            const SyncCoverGraph &graph,
                            const CanonicalSyncOwnershipCycle &cycle) {
  std::set<SyncCoverDemandId> matched;
  const auto addDerived = [&](SyncCoverNodeId source, SyncCoverNodeId target,
                              SyncCoverScopeId scope, unsigned distance) {
    SyncCoverEdge derived = makeEdge(graph, source, target, scope, distance);
    for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
      const bool unmatched = matched.count(demandId) == 0;
      const bool matches =
          demandMatchesDerivedWriteOwnership(graph, cycle, demand, derived);
      if (unmatched && matches) {
        matched.insert(demandId);
        // The round trip proves the source MTE1 operation completed before
        // the target producer issues. The intermediate MMAD contributes only
        // operand-ownership release, but the resulting producer-to-producer
        // completion edge is safe for ordinary graph composition.
        addCompositeBinding(descriptor, derived, {});
      }
    }
  };

  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    std::map<unsigned, const CanonicalSyncOwnershipUse *> previous;
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      const auto prior = previous.find(use.lane);
      if (prior != previous.end()) {
        for (SyncCoverNodeId source : prior->second->producers) {
          for (SyncCoverNodeId target : use.producers) {
            addDerived(source, target, path.scope, 0);
          }
        }
      }
      previous[use.lane] = &use;
    }
  }
  for (const CanonicalSyncOwnershipPath &sourcePath : cycle.paths) {
    for (const CanonicalSyncOwnershipPath &targetPath : cycle.paths) {
      for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
        const CanonicalSyncOwnershipUse *sourceUse = nullptr;
        const CanonicalSyncOwnershipUse *targetUse = nullptr;
        for (const CanonicalSyncOwnershipUse &use : sourcePath.uses) {
          if (use.lane == lane) {
            sourceUse = &use;
          }
        }
        for (const CanonicalSyncOwnershipUse &use : targetPath.uses) {
          if (!targetUse && use.producerLane == lane) {
            targetUse = &use;
          }
        }
        if (!sourceUse || !targetUse) {
          continue;
        }
        for (SyncCoverNodeId source : sourceUse->producers) {
          for (SyncCoverNodeId target : targetUse->producers) {
            addDerived(source, target, cycle.recurrenceScope, 1);
          }
        }
      }
    }
  }
}

void addHierarchicalOuterCompositeBindings(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const CanonicalSyncOwnershipCycle &cycle, SyncCoverScopeId outerScope,
    const CanonicalSyncOwnershipProtocol &protocol) {
  std::set<SyncCoverDemandId> matched;
  for (const CanonicalSyncSupplyBinding &ready : protocol.ready.supplies) {
    for (const CanonicalSyncSupplyBinding &release :
         protocol.release.supplies) {
      const bool composable =
          ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.edge.scope == outerScope && release.edge.distance == 1 &&
          ready.edge.target == release.edge.source;
      if (!composable) {
        continue;
      }
      SyncCoverEdge derived = makeEdge(graph, ready.edge.source,
                                       release.edge.target, outerScope, 1);
      std::vector<SyncCoverDemandId> allowedDemands;
      for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
        const bool unmatched = matched.count(demandId) == 0;
        const bool matches =
            demandMatchesDerivedWriteOwnership(graph, cycle, demand, derived);
        if (unmatched && matches) {
          matched.insert(demandId);
          allowedDemands.push_back(demandId);
        }
      }
      if (!allowedDemands.empty()) {
        // The exact managed WAW proves that this is the lifecycle's outer
        // ownership handoff. Both constituent event edges certify operation
        // completion, so their verified composition is generally usable by
        // the completion graph rather than only by that WAW demand.
        addCompositeBinding(descriptor, std::move(derived), {});
      }
    }
  }
}

bool isAlternatingReadyTransition(const CanonicalSyncOwnershipCycle &cycle,
                                  SyncCoverNodeId source,
                                  SyncCoverNodeId target) {
  return llvm::any_of(cycle.paths, [&](const auto &sourcePath) {
    const CanonicalSyncOwnershipUse &sourceUse = sourcePath.uses.front();
    if (!llvm::is_contained(sourceUse.producers, source)) {
      return false;
    }
    return llvm::any_of(cycle.paths, [&](const auto &targetPath) {
      const CanonicalSyncOwnershipUse &targetUse = targetPath.uses.front();
      return targetUse.lane == sourceUse.producerLane &&
             llvm::is_contained(targetUse.consumers, target);
    });
  });
}

void addHierarchicalAlternatingReadySummaries(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const CanonicalSyncOwnershipCycle &cycle, SyncCoverScopeId outerScope,
    const CanonicalSyncOwnershipProtocol &protocol) {
  std::set<SyncCoverDemandId> claimed;
  for (const CanonicalSyncSupplyBinding &ready : protocol.ready.supplies) {
    const bool innerReady =
        ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
        ready.edge.scope == cycle.recurrenceScope && ready.edge.distance == 1 &&
        ready.allowedDemands.empty() &&
        isAlternatingReadyTransition(cycle, ready.edge.source,
                                     ready.edge.target);
    if (!innerReady) {
      continue;
    }
    for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
      const bool exactDemand = demand.source == ready.edge.source &&
                               demand.target == ready.edge.target &&
                               demand.scope == outerScope &&
                               demand.distance == 1;
      if (!exactDemand || !claimed.insert(demandId).second) {
        continue;
      }
      addNestedRecurrenceSummaryBinding(
          descriptor,
          makeEdge(graph, demand.source, demand.target, outerScope, 1),
          demandId);
    }
  }
}

bool verifyNestedRecurrenceSummaryBindings(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, const CanonicalSyncOwnershipProtocol &protocol,
    ArrayRef<CanonicalSyncSupplyBinding> bindings) {
  std::set<SyncCoverDemandId> claimed;
  for (const CanonicalSyncSupplyBinding &binding : bindings) {
    const bool validShape =
        binding.proof ==
            CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary &&
        !binding.eventUse && !binding.barrierAction && !binding.produceAction &&
        !binding.consumeAction && binding.allowedDemands.size() == 1;
    if (!validShape) {
      return false;
    }
    const SyncCoverDemandId demandId = binding.allowedDemands.front();
    const bool invalidDemand = demandId >= graph.getDemands().size() ||
                               !claimed.insert(demandId).second;
    if (invalidDemand) {
      return false;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverEdge expected = makeEdge(graph, demand.source, demand.target,
                                            demand.scope, demand.distance);
    const bool exactEdge =
        std::tie(binding.edge.source, binding.edge.target, binding.edge.kind,
                 binding.edge.scope, binding.edge.distance,
                 binding.edge.sourceGuard.literals,
                 binding.edge.targetGuard.literals) ==
        std::tie(expected.source, expected.target, expected.kind,
                 expected.scope, expected.distance,
                 expected.sourceGuard.literals, expected.targetGuard.literals);
    const bool exactOuterDemand =
        exactEdge && demand.scope == outerScope && demand.distance == 1 &&
        isAlternatingReadyTransition(cycle, demand.source, demand.target);
    const bool hasVerifiedInnerReady =
        llvm::any_of(protocol.ready.supplies, [&](const auto &ready) {
          return ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
                 ready.edge.source == demand.source &&
                 ready.edge.target == demand.target &&
                 ready.edge.scope == cycle.recurrenceScope &&
                 ready.edge.distance == 1 && ready.allowedDemands.empty() &&
                 ready.eventUse && ready.produceAction && ready.consumeAction;
        });
    if (!exactOuterDemand || !hasVerifiedInnerReady) {
      return false;
    }
  }
  return true;
}

void addHierarchicalOwnershipClosureBindings(
    CanonicalSyncMechanismDescriptor &descriptor, const SyncCoverGraph &graph,
    const CanonicalSyncOwnershipCycle &cycle, SyncCoverScopeId outerScope) {
  std::set<SyncCoverDemandId> claimed;
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    const std::vector<SyncCoverDemandId> laneDemands =
        getLaneOwnershipDemands(graph, cycle, lane);
    for (SyncCoverDemandId demandId : laneDemands) {
      const SyncCoverDemand &demand = graph.getDemands()[demandId];
      if (demand.scope != outerScope || demand.distance != 1 ||
          !claimed.insert(demandId).second) {
        continue;
      }
      SyncCoverEdge edge = makeEdge(graph, demand.source, demand.target,
                                    demand.scope, demand.distance);
      const bool alreadySupplied =
          llvm::any_of(descriptor.supplies, [&](const auto &binding) {
            return std::tie(binding.edge.source, binding.edge.target,
                            binding.edge.kind, binding.edge.scope,
                            binding.edge.distance,
                            binding.edge.sourceGuard.literals,
                            binding.edge.targetGuard.literals) ==
                   std::tie(edge.source, edge.target, edge.kind, edge.scope,
                            edge.distance, edge.sourceGuard.literals,
                            edge.targetGuard.literals);
          });
      if (!alreadySupplied) {
        addOwnershipClosureBinding(descriptor, std::move(edge), demandId);
      }
    }
  }
}

bool nodeProducesLane(const CanonicalSyncOwnershipCycle &cycle, unsigned lane,
                      SyncCoverNodeId node) {
  if (lane == cycle.initialReadyLane &&
      llvm::is_contained(cycle.initialProducers, node)) {
    return true;
  }
  return llvm::any_of(cycle.paths, [&](const auto &path) {
    return llvm::any_of(path.uses, [&](const auto &use) {
      return use.producerLane == lane &&
             llvm::is_contained(use.producers, node);
    });
  });
}

bool nodeConsumesLane(const CanonicalSyncOwnershipCycle &cycle, unsigned lane,
                      SyncCoverNodeId node) {
  return llvm::any_of(cycle.paths, [&](const auto &path) {
    return llvm::any_of(path.uses, [&](const auto &use) {
      return use.lane == lane && llvm::is_contained(use.consumers, node);
    });
  });
}

bool witnessMatchesOwnershipLane(const SyncCoverGraph &graph,
                                 const CanonicalSyncOwnershipCycle &cycle,
                                 unsigned lane, const SyncCoverDemand &demand,
                                 SyncCoverStorageWitnessId witnessId) {
  const bool invalidLaneOrWitness =
      lane >= cycle.lanes.size() ||
      witnessId >= graph.getStorageWitnesses().size();
  if (invalidLaneOrWitness) {
    return false;
  }
  const SyncCoverStorageWitness &witness =
      graph.getStorageWitnesses()[witnessId];
  const bool invalidAccess =
      witness.sourceAccess >= graph.getStorageAccesses().size() ||
      witness.targetAccess >= graph.getStorageAccesses().size();
  if (invalidAccess) {
    return false;
  }
  const SyncCoverStorageAccess &source =
      graph.getStorageAccesses()[witness.sourceAccess];
  const SyncCoverStorageAccess &target =
      graph.getStorageAccesses()[witness.targetAccess];
  const bool sourceRole = (syncCoverStorageModeReads(source.mode) &&
                           nodeConsumesLane(cycle, lane, source.node)) ||
                          (syncCoverStorageModeWrites(source.mode) &&
                           nodeProducesLane(cycle, lane, source.node));
  const bool targetRole = syncCoverStorageModeWrites(target.mode) &&
                          nodeProducesLane(cycle, lane, target.node);
  const bool exactEndpoints = source.exactPhysical && target.exactPhysical &&
                              source.node == demand.source &&
                              target.node == demand.target;
  const bool managedSlot = llvm::any_of(
      cycle.lanes[lane].slots, [&](const CanonicalSyncOwnershipSlot &slot) {
        return source.domain == slot.domain && target.domain == slot.domain &&
               intervalContains(slot.extent, witness.overlap);
      });
  return sourceRole && targetRole && exactEndpoints && managedSlot;
}

bool demandMatchesOwnershipLane(const SyncCoverGraph &graph,
                                const CanonicalSyncOwnershipCycle &cycle,
                                unsigned lane, const SyncCoverDemand &demand) {
  const bool ownershipHazard =
      !demand.provenanceKinds.empty() &&
      llvm::all_of(demand.provenanceKinds, [](SyncCoverDemandKind kind) {
        return kind == SyncCoverDemandKind::MemoryWAR ||
               kind == SyncCoverDemandKind::MemoryWAW;
      });
  return ownershipHazard && !demand.storageWitnesses.empty() &&
         llvm::all_of(demand.storageWitnesses,
                      [&](SyncCoverStorageWitnessId witnessId) {
                        return witnessMatchesOwnershipLane(graph, cycle, lane,
                                                           demand, witnessId);
                      });
}

bool actionMatchesAnchor(const CanonicalSyncAction &action,
                         CanonicalSyncActionKind kind, std::uint32_t resource,
                         const SyncCoverAnchor &anchor, unsigned lane) {
  return action.kind == kind && action.resource == resource &&
         action.eventUse.has_value() && action.eventLane == lane &&
         std::tie(action.anchor.kind, action.anchor.node, action.anchor.scope,
                  action.anchor.position) ==
             std::tie(anchor.kind, anchor.node, anchor.scope, anchor.position);
}

bool releasePathStartsAtConsumer(const CanonicalSyncOwnershipProtocol &protocol,
                                 const CanonicalSyncOwnershipCycle &cycle,
                                 unsigned lane, SyncCoverNodeId consumer) {
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (use.lane != lane || !llvm::is_contained(use.consumers, consumer)) {
        continue;
      }
      const SyncCoverAnchor direct{SyncCoverAnchorKind::AfterNode, consumer, 0,
                                   0};
      return llvm::any_of(
          protocol.release.actions, [&](const CanonicalSyncAction &action) {
            return actionMatchesAnchor(action,
                                       CanonicalSyncActionKind::EventSet,
                                       cycle.consumerResource, direct, lane) ||
                   actionMatchesAnchor(
                       action, CanonicalSyncActionKind::EventSet,
                       cycle.consumerResource, use.release, lane);
          });
    }
  }
  return false;
}

bool releasePathEndsAtProducer(const CanonicalSyncOwnershipProtocol &protocol,
                               const CanonicalSyncOwnershipCycle &cycle,
                               unsigned lane, SyncCoverNodeId producer) {
  return llvm::any_of(cycle.paths, [&](const auto &path) {
    return llvm::any_of(path.uses, [&](const auto &use) {
      if (use.producerLane != lane ||
          !llvm::is_contained(use.producers, producer)) {
        return false;
      }
      return llvm::any_of(protocol.release.actions,
                          [&](const CanonicalSyncAction &action) {
                            return actionMatchesAnchor(
                                action, CanonicalSyncActionKind::EventWait,
                                cycle.producerResource, use.writeAcquire, lane);
                          });
    });
  });
}

bool producerCompletionReachesRelease(
    const CanonicalSyncOwnershipProtocol &protocol,
    const CanonicalSyncOwnershipCycle &cycle, unsigned lane,
    SyncCoverNodeId producer) {
  return llvm::any_of(
      protocol.ready.supplies, [&](const CanonicalSyncSupplyBinding &ready) {
        return ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
               ready.edge.source == producer && ready.allowedDemands.empty() &&
               nodeConsumesLane(cycle, lane, ready.edge.target) &&
               releasePathStartsAtConsumer(protocol, cycle, lane,
                                           ready.edge.target);
      });
}

bool demandHasOwnershipTokenPath(const SyncCoverGraph &graph,
                                 const CanonicalSyncOwnershipCycle &cycle,
                                 const CanonicalSyncOwnershipProtocol &protocol,
                                 unsigned lane, const SyncCoverDemand &demand) {
  if (!releasePathEndsAtProducer(protocol, cycle, lane, demand.target)) {
    return false;
  }
  return llvm::all_of(
      demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
        const SyncCoverStorageWitness &witness =
            graph.getStorageWitnesses()[witnessId];
        const SyncCoverStorageAccess &source =
            graph.getStorageAccesses()[witness.sourceAccess];
        const bool readPath =
            !syncCoverStorageModeReads(source.mode) ||
            releasePathStartsAtConsumer(protocol, cycle, lane, source.node);
        const bool writePath = !syncCoverStorageModeWrites(source.mode) ||
                               producerCompletionReachesRelease(
                                   protocol, cycle, lane, source.node);
        return readPath && writePath;
      });
}

bool verifyOwnershipClosureBindings(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    const CanonicalSyncOwnershipProtocol &protocol, SyncCoverScopeId outerScope,
    ArrayRef<CanonicalSyncSupplyBinding> bindings) {
  std::set<SyncCoverDemandId> claimed;
  for (const CanonicalSyncSupplyBinding &binding : bindings) {
    if (binding.proof != CanonicalSyncSupplyProof::VerifiedOwnershipClosure ||
        binding.eventUse || binding.barrierAction || binding.produceAction ||
        binding.consumeAction || binding.allowedDemands.size() != 1) {
      return false;
    }
    const SyncCoverDemandId demandId = binding.allowedDemands.front();
    const bool invalidDemand = demandId >= graph.getDemands().size() ||
                               !claimed.insert(demandId).second;
    if (invalidDemand) {
      return false;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverEdge expected = makeEdge(graph, demand.source, demand.target,
                                            demand.scope, demand.distance);
    const bool exactEdge =
        std::tie(binding.edge.source, binding.edge.target, binding.edge.kind,
                 binding.edge.scope, binding.edge.distance,
                 binding.edge.sourceGuard.literals,
                 binding.edge.targetGuard.literals) ==
        std::tie(expected.source, expected.target, expected.kind,
                 expected.scope, expected.distance,
                 expected.sourceGuard.literals, expected.targetGuard.literals);
    if (!exactEdge || demand.scope != outerScope || demand.distance != 1) {
      return false;
    }
    unsigned matchingLanes = 0;
    for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
      matchingLanes +=
          demandMatchesOwnershipLane(graph, cycle, lane, demand) &&
          demandHasOwnershipTokenPath(graph, cycle, protocol, lane, demand);
    }
    if (matchingLanes != 1) {
      return false;
    }
  }
  return true;
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
buildPrefixL0Ready(const SyncCoverGraph &graph,
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
buildAccumulatorReady(const SyncCoverGraph &graph,
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
          const std::vector<SyncCoverDemandId> allowed =
              getAccumulatorReadyDemands(graph, cycle, use.lane, producer,
                                         consumer);
          if (!allowed.empty()) {
            addBinding(result,
                       makeEdge(graph, producer, consumer, path.scope, 0), 0,
                       set, wait, allowed);
          }
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor
buildBoundaryGuardedRelease(const SyncCoverGraph &graph,
                            const CanonicalSyncOwnershipCycle &cycle,
                            CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back(
      {domain, cycle.lanes.size(), cycle.recurrenceScope});
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      const std::vector<SyncCoverDemandId> allowed =
          getLaneOwnershipDemands(graph, cycle, use.lane);
      const std::size_t wait = result.actions.size();
      result.actions.push_back(
          makeAction(CanonicalSyncActionKind::EventWait, cycle.producerResource,
                     use.writeAcquire, 0, use.producerLane,
                     CanonicalSyncActionGuardKind::NotFirstIteration,
                     cycle.recurrenceScope));
      const std::size_t set = result.actions.size();
      result.actions.push_back(makeAction(
          CanonicalSyncActionKind::EventSet, cycle.consumerResource,
          use.release, 0, use.lane, CanonicalSyncActionGuardKind::HasSuccessor,
          cycle.recurrenceScope));
      for (SyncCoverNodeId consumer : use.consumers) {
        for (SyncCoverNodeId producer : use.producers) {
          addBinding(
              result,
              makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1), 0,
              set, wait, allowed);
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor buildL0RoundTripRelease(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain, bool mergeAlternativeRelease) {
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
      const bool hasAlternativeConsumers = use.consumers.size() > 1;
      if (mergeAlternativeRelease && hasAlternativeConsumers) {
        const std::size_t set = result.actions.size();
        result.actions.push_back(makeAction(CanonicalSyncActionKind::EventSet,
                                            cycle.consumerResource, use.release,
                                            0, use.lane));
        for (SyncCoverNodeId consumer : use.consumers) {
          useActions.sets.push_back({consumer, set});
        }
      } else {
        for (SyncCoverNodeId consumer : use.consumers) {
          const std::size_t set = result.actions.size();
          result.actions.push_back(makeAction(
              CanonicalSyncActionKind::EventSet, cycle.consumerResource,
              {SyncCoverAnchorKind::AfterNode, consumer, 0, 0}, 0, use.lane));
          useActions.sets.push_back({consumer, set});
        }
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

CanonicalSyncMechanismDescriptor buildHierarchicalStableL1Release(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result =
      buildStableL1Release(graph, cycle, domain);
  result.eventUses.front().lifetimeScope = outerScope;
  for (CanonicalSyncAction &action : result.actions) {
    if (action.anchor.kind == SyncCoverAnchorKind::ScopeEntry &&
        action.anchor.scope == cycle.recurrenceScope) {
      action.anchor.scope = outerScope;
    } else if (action.anchor.kind == SyncCoverAnchorKind::ScopeExit &&
               action.anchor.scope == cycle.recurrenceScope) {
      action.anchor.scope = outerScope;
    }
  }
  const CanonicalSyncOwnershipPath &firstPath = cycle.paths.front();
  for (const CanonicalSyncOwnershipPath &sourcePath : cycle.paths) {
    for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
      const auto source = llvm::find_if(
          sourcePath.uses, [&](const auto &use) { return use.lane == lane; });
      const auto target = llvm::find_if(firstPath.uses, [&](const auto &use) {
        return use.producerLane == lane;
      });
      const bool missingEndpoint =
          source == sourcePath.uses.end() || target == firstPath.uses.end();
      if (missingEndpoint) {
        return {};
      }
      const std::optional<std::size_t> set =
          findAction(result, CanonicalSyncActionKind::EventSet,
                     cycle.consumerResource, source->release, source->lane);
      const std::optional<std::size_t> wait = findAction(
          result, CanonicalSyncActionKind::EventWait, cycle.producerResource,
          target->writeAcquire, target->producerLane);
      const std::vector<SyncCoverDemandId> allowed =
          getLaneOwnershipDemands(graph, cycle, lane);
      if (!set || !wait || allowed.empty()) {
        return {};
      }
      for (SyncCoverNodeId consumer : source->consumers) {
        for (SyncCoverNodeId producer : target->producers) {
          addBinding(result, makeEdge(graph, consumer, producer, outerScope, 1),
                     0, *set, *wait, allowed);
        }
      }
    }
  }
  return result;
}

CanonicalSyncMechanismDescriptor buildHierarchicalAlternatingReady(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result =
      buildAlternatingReady(graph, cycle, domain);
  if (result.actions.empty()) {
    return {};
  }
  result.eventUses.front().lifetimeScope = outerScope;
  result.actions.front().guard = CanonicalSyncActionGuardKind::None;
  result.actions.front().guardScope.reset();
  result.actions.push_back(makeAction(
      CanonicalSyncActionKind::EventWait, cycle.consumerResource,
      {SyncCoverAnchorKind::ScopeEntry, 0, cycle.recurrenceScope}, 0,
      cycle.initialReadyLane, CanonicalSyncActionGuardKind::LoopEmpty,
      cycle.recurrenceScope));
  return result;
}

CanonicalSyncMechanismDescriptor buildHierarchicalAlternatingRelease(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain) {
  CanonicalSyncMechanismDescriptor result;
  result.kind = CanonicalSyncMechanismKind::Protocol;
  result.eventUses.push_back(
      {domain, cycle.lanes.size(), cycle.recurrenceScope, outerScope});
  std::vector<std::vector<SyncCoverDemandId>> releaseDemands(
      cycle.lanes.size());
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    releaseDemands[lane] = getLaneOwnershipDemands(graph, cycle, lane);
    result.actions.push_back(
        makeAction(CanonicalSyncActionKind::EventSet, cycle.consumerResource,
                   {SyncCoverAnchorKind::ScopeEntry, 0, outerScope}, 0, lane));
  }
  const std::size_t initialWait = result.actions.size();
  result.actions.push_back(
      makeAction(CanonicalSyncActionKind::EventWait, cycle.producerResource,
                 cycle.initialWriteAcquire, 0, cycle.initialReadyLane));

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
  result.actions.push_back(makeAction(
      CanonicalSyncActionKind::EventSet, cycle.consumerResource,
      {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope}, 0,
      cycle.initialReadyLane, CanonicalSyncActionGuardKind::LoopEmpty,
      cycle.recurrenceScope));
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    result.actions.push_back(
        makeAction(CanonicalSyncActionKind::EventWait, cycle.producerResource,
                   {SyncCoverAnchorKind::ScopeExit, 0, outerScope}, 0, lane));
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
  const auto initialPath = llvm::find_if(cycle.paths, [&](const auto &path) {
    return path.uses.front().lane == cycle.initialReadyLane;
  });
  if (initialPath == cycle.paths.end()) {
    return {};
  }
  const std::size_t sourcePath =
      static_cast<std::size_t>(initialPath - cycle.paths.begin());
  const std::vector<SyncCoverDemandId> outerDemands = getLaneOwnershipDemands(
      graph, cycle, cycle.initialReadyLane, cycle.initialProducers);
  for (SyncCoverNodeId consumer : initialPath->uses.front().consumers) {
    for (SyncCoverNodeId producer : cycle.initialProducers) {
      addBinding(result, makeEdge(graph, consumer, producer, outerScope, 1), 0,
                 sets[sourcePath], initialWait, outerDemands);
    }
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
  const bool sizeMismatch = actual.size() != expected.size();
  if (sizeMismatch) {
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

bool isActionlessCompositeBinding(const CanonicalSyncSupplyBinding &binding) {
  return binding.proof == CanonicalSyncSupplyProof::VerifiedCompositeProtocol &&
         !binding.eventUse && !binding.barrierAction &&
         !binding.produceAction && !binding.consumeAction;
}

bool exactManagedWriteDemand(const SyncCoverGraph &graph,
                             const CanonicalSyncOwnershipCycle &cycle,
                             const CanonicalSyncSupplyBinding &binding,
                             bool demandQualified) {
  const bool invalidQualifiedDemand =
      demandQualified && binding.allowedDemands.size() != 1;
  if (invalidQualifiedDemand) {
    return false;
  }
  if (!demandQualified && !binding.allowedDemands.empty()) {
    return false;
  }
  bool found = false;
  for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
    if (!edgeEqual(binding.edge, makeEdge(graph, demand.source, demand.target,
                                          demand.scope, demand.distance)) ||
        !demandMatchesDerivedWriteOwnership(graph, cycle, demand,
                                            binding.edge)) {
      continue;
    }
    const bool wrongQualifiedDemand =
        demandQualified && binding.allowedDemands.front() != demandId;
    if (wrongQualifiedDemand) {
      return false;
    }
    found = true;
  }
  return found;
}

bool hasAlternatingCompositePath(const SyncCoverGraph &graph,
                                 const CanonicalSyncOwnershipCycle &cycle,
                                 const CanonicalSyncOwnershipProtocol &protocol,
                                 const CanonicalSyncSupplyBinding &binding) {
  std::set<SyncCoverNodeId> initial(cycle.initialProducers.begin(),
                                    cycle.initialProducers.end());
  std::set<SyncCoverNodeId> body;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      body.insert(use.producers.begin(), use.producers.end());
    }
  }
  return llvm::any_of(protocol.ready.supplies, [&](const auto &ready) {
    return llvm::any_of(protocol.release.supplies, [&](const auto &release) {
      const bool physicalPath =
          ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          ready.edge.target == release.edge.source &&
          ready.edge.targetGuard.literals == release.edge.sourceGuard.literals;
      if (!physicalPath) {
        return false;
      }
      const bool initialTransition = initial.count(ready.edge.source) != 0 &&
                                     body.count(release.edge.target) != 0;
      const bool bodyTransition = ready.edge.source == release.edge.target &&
                                  body.count(ready.edge.source) != 0;
      if (!initialTransition && !bodyTransition) {
        return false;
      }
      const unsigned distance = bodyTransition ? 1 : 0;
      const SyncCoverScopeId scope =
          bodyTransition ? cycle.recurrenceScope
                         : graph
                               .getLowestCommonScope(
                                   graph.getNodes()[ready.edge.source].scope,
                                   graph.getNodes()[release.edge.target].scope)
                               .value_or(cycle.recurrenceScope);
      SyncCoverEdge derived{ready.edge.source,
                            release.edge.target,
                            SyncCoverEdgeKind::CompletionSupply,
                            scope,
                            distance,
                            ready.edge.sourceGuard,
                            release.edge.targetGuard};
      return edgeEqual(binding.edge, derived);
    });
  });
}

bool hasAccumulatorCompositePath(const CanonicalSyncOwnershipCycle &cycle,
                                 const CanonicalSyncOwnershipProtocol &protocol,
                                 const CanonicalSyncSupplyBinding &binding) {
  return llvm::any_of(protocol.ready.supplies, [&](const auto &ready) {
    return llvm::any_of(protocol.release.supplies, [&](const auto &release) {
      if (ready.proof != CanonicalSyncSupplyProof::VerifiedProtocol ||
          release.proof != CanonicalSyncSupplyProof::VerifiedProtocol ||
          ready.edge.target != release.edge.source) {
        return false;
      }
      SyncCoverEdge derived{ready.edge.source,
                            release.edge.target,
                            SyncCoverEdgeKind::CompletionSupply,
                            cycle.recurrenceScope,
                            1,
                            ready.edge.sourceGuard,
                            release.edge.targetGuard};
      return edgeEqual(binding.edge, derived);
    });
  });
}

bool hasL0CompositePath(const SyncCoverGraph &graph,
                        const CanonicalSyncOwnershipCycle &cycle,
                        const CanonicalSyncOwnershipProtocol &protocol,
                        const CanonicalSyncSupplyBinding &binding) {
  return llvm::any_of(protocol.ready.supplies, [&](const auto &ready) {
    return llvm::any_of(protocol.release.supplies, [&](const auto &release) {
      if (ready.proof != CanonicalSyncSupplyProof::VerifiedProtocol ||
          release.proof != CanonicalSyncSupplyProof::VerifiedProtocol ||
          ready.edge.target != release.edge.source ||
          ready.edge.distance != 0 || release.edge.distance > 1) {
        return false;
      }
      const unsigned distance = release.edge.distance;
      const SyncCoverScopeId scope =
          distance == 0 ? graph
                              .getLowestCommonScope(
                                  graph.getNodes()[ready.edge.source].scope,
                                  graph.getNodes()[release.edge.target].scope)
                              .value_or(release.edge.scope)
                        : cycle.recurrenceScope;
      SyncCoverEdge derived = makeEdge(graph, ready.edge.source,
                                       release.edge.target, scope, distance);
      return edgeEqual(binding.edge, derived);
    });
  });
}

bool hasHierarchicalOuterCompositePath(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipProtocol &protocol,
    SyncCoverScopeId outerScope, const CanonicalSyncSupplyBinding &binding) {
  return llvm::any_of(protocol.ready.supplies, [&](const auto &ready) {
    return llvm::any_of(protocol.release.supplies, [&](const auto &release) {
      const bool physicalPath =
          ready.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.proof == CanonicalSyncSupplyProof::VerifiedProtocol &&
          release.edge.scope == outerScope && release.edge.distance == 1 &&
          ready.edge.target == release.edge.source;
      if (!physicalPath) {
        return false;
      }
      return edgeEqual(binding.edge,
                       makeEdge(graph, ready.edge.source, release.edge.target,
                                outerScope, 1));
    });
  });
}

bool verifyAtomicCompositeBindings(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    const CanonicalSyncOwnershipProtocol &protocol,
    ArrayRef<CanonicalSyncSupplyBinding> bindings) {
  return llvm::all_of(bindings, [&](const auto &binding) {
    if (!isActionlessCompositeBinding(binding)) {
      return false;
    }
    switch (cycle.kind) {
    case CanonicalSyncOwnershipKind::L0Operand:
      return exactManagedWriteDemand(graph, cycle, binding, false) &&
             hasL0CompositePath(graph, cycle, protocol, binding);
    case CanonicalSyncOwnershipKind::L1Tile:
      return cycle.protocol ==
                 CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch &&
             exactManagedWriteDemand(graph, cycle, binding, true) &&
             hasAlternatingCompositePath(graph, cycle, protocol, binding);
    case CanonicalSyncOwnershipKind::L0Accumulator:
      return exactManagedWriteDemand(graph, cycle, binding, true) &&
             hasAccumulatorCompositePath(cycle, protocol, binding);
    }
    return false;
  });
}

bool verifyHierarchicalCompositeBindings(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    const CanonicalSyncOwnershipProtocol &protocol, SyncCoverScopeId outerScope,
    ArrayRef<CanonicalSyncSupplyBinding> bindings) {
  return llvm::all_of(bindings, [&](const auto &binding) {
    if (!isActionlessCompositeBinding(binding)) {
      return false;
    }
    const bool alternating =
        cycle.protocol ==
            CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch &&
        hasAlternatingCompositePath(graph, cycle, protocol, binding) &&
        exactManagedWriteDemand(graph, cycle, binding, true);
    const bool outer = hasHierarchicalOuterCompositePath(graph, protocol,
                                                         outerScope, binding) &&
                       exactManagedWriteDemand(graph, cycle, binding, false);
    return alternating || outer;
  });
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

bool hasProtocolHeader(
    const CanonicalSyncMechanismDescriptor &descriptor,
    CanonicalSyncEventDomainId domain, std::size_t uses, std::size_t width,
    SyncCoverScopeId recurrenceScope,
    std::optional<SyncCoverScopeId> lifetimeScope = std::nullopt) {
  return descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
         descriptor.eventUses.size() == uses &&
         llvm::all_of(descriptor.eventUses,
                      [&](const CanonicalSyncEventUse &use) {
                        return use.domain == domain && use.width == width &&
                               use.recurrenceScope == recurrenceScope &&
                               use.lifetimeScope == lifetimeScope;
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

bool hasBoundaryGuardedCycleShape(const CanonicalSyncOwnershipCycle &cycle) {
  if (cycle.kind != CanonicalSyncOwnershipKind::L0Accumulator ||
      cycle.protocol !=
          CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip ||
      cycle.lanes.empty() || cycle.paths.size() != 1 ||
      cycle.paths.front().scope != cycle.recurrenceScope ||
      cycle.paths.front().uses.size() != cycle.lanes.size()) {
    return false;
  }
  std::vector<bool> lanes(cycle.lanes.size(), false);
  for (const CanonicalSyncOwnershipUse &use : cycle.paths.front().uses) {
    if (use.lane >= lanes.size() || use.producerLane != use.lane ||
        lanes[use.lane]) {
      return false;
    }
    lanes[use.lane] = true;
  }
  return llvm::all_of(lanes, [](bool seen) { return seen; });
}

bool verifyAccumulatorReadyDescriptor(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const SyncCoverGraph &graph = program.getGraph();
  if (!program.getTargetCapabilities().mToFixAccumulatorBoundaryCompletes ||
      !hasBoundaryGuardedCycleShape(cycle) ||
      !hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  CanonicalSyncMechanismDescriptor expected;
  for (const CanonicalSyncOwnershipUse &use : cycle.paths.front().uses) {
    if (use.consumers.size() != 1) {
      return false;
    }
    const auto set =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                    cycle.producerResource, use.ready, 0, use.lane);
    const auto wait =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                    cycle.consumerResource, use.readAcquire, 0, use.lane);
    if (!set || !wait) {
      return false;
    }
    bool supplied = false;
    for (SyncCoverNodeId producer : use.producers) {
      for (SyncCoverNodeId consumer : use.consumers) {
        const std::vector<SyncCoverDemandId> allowed =
            getAccumulatorReadyDemands(graph, cycle, use.lane, producer,
                                       consumer);
        if (allowed.empty()) {
          continue;
        }
        SyncCoverEdge edge =
            makeEdge(graph, producer, consumer, cycle.paths.front().scope, 0);
        if (!isValidProtocolEdge(graph, edge)) {
          return false;
        }
        addBinding(expected, std::move(edge), 0, *set, *wait, allowed);
        supplied = true;
      }
    }
    if (!supplied) {
      return false;
    }
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyBoundaryGuardedReleaseDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasBoundaryGuardedCycleShape(cycle) ||
      !hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  std::vector<bool> claimed(descriptor.actions.size(), false);
  CanonicalSyncMechanismDescriptor expected;
  for (const CanonicalSyncOwnershipUse &use : cycle.paths.front().uses) {
    if (use.consumers.size() != 1) {
      return false;
    }
    const std::vector<SyncCoverDemandId> allowed =
        getLaneOwnershipDemands(graph, cycle, use.lane);
    if (allowed.empty() ||
        !llvm::all_of(use.consumers, [&](SyncCoverNodeId consumer) {
          return syncCoverNodeCanProduceCompletion(graph, consumer,
                                                   cycle.producerResource);
        })) {
      return false;
    }
    const auto wait = claimAction(
        descriptor, claimed, CanonicalSyncActionKind::EventWait,
        cycle.producerResource, use.writeAcquire, 0, use.producerLane,
        CanonicalSyncActionGuardKind::NotFirstIteration, cycle.recurrenceScope);
    const auto set = claimAction(
        descriptor, claimed, CanonicalSyncActionKind::EventSet,
        cycle.consumerResource, use.release, 0, use.lane,
        CanonicalSyncActionGuardKind::HasSuccessor, cycle.recurrenceScope);
    if (!wait || !set) {
      return false;
    }
    for (SyncCoverNodeId consumer : use.consumers) {
      for (SyncCoverNodeId producer : use.producers) {
        SyncCoverEdge edge =
            makeEdge(graph, consumer, producer, cycle.recurrenceScope, 1);
        if (!isValidProtocolEdge(graph, edge)) {
          return false;
        }
        addBinding(expected, std::move(edge), 0, *set, *wait, allowed);
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
    const CanonicalSyncOwnershipPath &path = cycle.paths[pathIndex];
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
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

bool isNearestEnclosingLoop(const SyncCoverGraph &graph, SyncCoverScopeId inner,
                            SyncCoverScopeId outer) {
  const bool invalidScope = inner >= graph.getScopes().size() ||
                            outer >= graph.getScopes().size() ||
                            !graph.getScopes()[outer].isLoop;
  if (invalidScope) {
    return false;
  }
  return graph.getScopes()[inner].parent == outer;
}

bool verifyHierarchicalStableL1ReleaseDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope, outerScope)) {
    return false;
  }
  CanonicalSyncMechanismDescriptor normalized = descriptor;
  normalized.eventUses.front().lifetimeScope.reset();
  for (CanonicalSyncAction &action : normalized.actions) {
    if (action.anchor.kind == SyncCoverAnchorKind::ScopeEntry &&
        action.anchor.scope == outerScope) {
      action.anchor.scope = cycle.recurrenceScope;
    } else if (action.anchor.kind == SyncCoverAnchorKind::ScopeExit &&
               action.anchor.scope == outerScope) {
      action.anchor.scope = cycle.recurrenceScope;
    }
  }

  CanonicalSyncMechanismDescriptor expectedOuter;
  const CanonicalSyncOwnershipPath &firstPath = cycle.paths.front();
  for (const CanonicalSyncOwnershipPath &sourcePath : cycle.paths) {
    for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
      const auto source = llvm::find_if(
          sourcePath.uses, [&](const auto &use) { return use.lane == lane; });
      const auto target = llvm::find_if(firstPath.uses, [&](const auto &use) {
        return use.producerLane == lane;
      });
      const bool missingEndpoint =
          source == sourcePath.uses.end() || target == firstPath.uses.end();
      if (missingEndpoint) {
        return false;
      }
      const std::optional<std::size_t> set =
          findAction(descriptor, CanonicalSyncActionKind::EventSet,
                     cycle.consumerResource, source->release, source->lane);
      const std::optional<std::size_t> wait = findAction(
          descriptor, CanonicalSyncActionKind::EventWait,
          cycle.producerResource, target->writeAcquire, target->producerLane);
      const std::vector<SyncCoverDemandId> allowed =
          getLaneOwnershipDemands(graph, cycle, lane);
      if (!set || !wait || allowed.empty()) {
        return false;
      }
      for (SyncCoverNodeId consumer : source->consumers) {
        for (SyncCoverNodeId producer : target->producers) {
          SyncCoverEdge edge =
              makeEdge(graph, consumer, producer, outerScope, 1);
          if (!isValidProtocolEdge(graph, edge)) {
            return false;
          }
          addBinding(expectedOuter, std::move(edge), 0, *set, *wait, allowed);
        }
      }
    }
  }

  CanonicalSyncMechanismDescriptor actualOuter;
  llvm::erase_if(
      normalized.supplies, [&](const CanonicalSyncSupplyBinding &binding) {
        if (binding.edge.scope != outerScope || binding.edge.distance != 1) {
          return false;
        }
        actualOuter.supplies.push_back(binding);
        return true;
      });
  const bool outerMatches =
      bindingsEqual(actualOuter.supplies, expectedOuter.supplies);
  const bool innerMatches =
      verifyStableL1ReleaseDescriptor(graph, cycle, domain, normalized);
  return outerMatches && innerMatches;
}

bool verifyHierarchicalAlternatingReadyDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope, outerScope)) {
    return false;
  }
  std::optional<std::size_t> emptyWait;
  for (auto [index, action] : llvm::enumerate(descriptor.actions)) {
    const bool matches =
        action.kind == CanonicalSyncActionKind::EventWait &&
        action.resource == cycle.consumerResource &&
        anchorEqual(action.anchor, {SyncCoverAnchorKind::ScopeEntry, 0,
                                    cycle.recurrenceScope}) &&
        action.eventUse == 0 && action.eventLane == cycle.initialReadyLane &&
        action.guard == CanonicalSyncActionGuardKind::LoopEmpty &&
        action.guardScope == cycle.recurrenceScope;
    if (!matches) {
      continue;
    }
    if (emptyWait) {
      return false;
    }
    emptyWait = index;
  }
  if (!emptyWait) {
    return false;
  }

  CanonicalSyncMechanismDescriptor normalized = descriptor;
  for (const CanonicalSyncSupplyBinding &binding : normalized.supplies) {
    if (binding.produceAction == *emptyWait ||
        binding.consumeAction == *emptyWait) {
      return false;
    }
  }
  normalized.actions.erase(normalized.actions.begin() + *emptyWait);
  for (CanonicalSyncSupplyBinding &binding : normalized.supplies) {
    if (binding.produceAction && *binding.produceAction > *emptyWait) {
      --*binding.produceAction;
    }
    if (binding.consumeAction && *binding.consumeAction > *emptyWait) {
      --*binding.consumeAction;
    }
  }
  normalized.eventUses.front().lifetimeScope.reset();
  const std::optional<std::size_t> initialSet = findAction(
      normalized, CanonicalSyncActionKind::EventSet, cycle.producerResource,
      cycle.initialReady, cycle.initialReadyLane);
  if (!initialSet ||
      normalized.actions[*initialSet].guard !=
          CanonicalSyncActionGuardKind::None ||
      normalized.actions[*initialSet].guardScope) {
    return false;
  }
  normalized.actions[*initialSet].guard =
      CanonicalSyncActionGuardKind::LoopNonEmpty;
  normalized.actions[*initialSet].guardScope = cycle.recurrenceScope;
  return verifyAlternatingReadyDescriptor(graph, cycle, domain, normalized);
}

bool verifyHierarchicalAlternatingReleaseDescriptor(
    const SyncCoverGraph &graph, const CanonicalSyncOwnershipCycle &cycle,
    SyncCoverScopeId outerScope, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope, outerScope) ||
      cycle.paths.size() != 2 || cycle.initialProducers.size() != 1) {
    return false;
  }
  std::vector<std::vector<SyncCoverDemandId>> releaseDemands(
      cycle.lanes.size());
  std::vector<bool> claimed(descriptor.actions.size(), false);
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    releaseDemands[lane] = getLaneOwnershipDemands(graph, cycle, lane);
    const bool claimedPrime =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                    cycle.consumerResource,
                    {SyncCoverAnchorKind::ScopeEntry, 0, outerScope}, 0, lane)
            .has_value();
    const bool missingReleaseLane =
        releaseDemands[lane].empty() || !claimedPrime;
    if (missingReleaseLane) {
      return false;
    }
  }
  const auto initialWait =
      claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                  cycle.producerResource, cycle.initialWriteAcquire, 0,
                  cycle.initialReadyLane);
  if (!initialWait) {
    return false;
  }
  std::vector<std::size_t> sets(cycle.paths.size());
  std::vector<std::size_t> waits(cycle.paths.size());
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const bool hasSingleUse = path.uses.size() == 1;
    if (!hasSingleUse) {
      return false;
    }
    const CanonicalSyncOwnershipUse &use = path.uses.front();
    const auto set =
        claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                    cycle.consumerResource, use.release, 0, use.lane);
    const auto wait = claimAction(
        descriptor, claimed, CanonicalSyncActionKind::EventWait,
        cycle.producerResource, use.writeAcquire, 0, use.producerLane);
    if (!set || !wait ||
        !llvm::all_of(use.consumers, [&](SyncCoverNodeId consumer) {
          return syncCoverNodeCanProduceCompletion(graph, consumer,
                                                   cycle.producerResource);
        })) {
      return false;
    }
    sets[pathIndex] = *set;
    waits[pathIndex] = *wait;
  }
  if (!claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                   cycle.consumerResource,
                   {SyncCoverAnchorKind::ScopeExit, 0, cycle.recurrenceScope},
                   0, cycle.initialReadyLane,
                   CanonicalSyncActionGuardKind::LoopEmpty,
                   cycle.recurrenceScope)) {
    return false;
  }
  for (unsigned lane = 0; lane < cycle.lanes.size(); ++lane) {
    if (!claimAction(descriptor, claimed, CanonicalSyncActionKind::EventWait,
                     cycle.producerResource,
                     {SyncCoverAnchorKind::ScopeExit, 0, outerScope}, 0,
                     lane)) {
      return false;
    }
  }

  CanonicalSyncMechanismDescriptor expected;
  for (auto [pathIndex, path] : llvm::enumerate(cycle.paths)) {
    const CanonicalSyncOwnershipUse &use = path.uses.front();
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
  const auto initialPath = llvm::find_if(cycle.paths, [&](const auto &path) {
    return path.uses.front().lane == cycle.initialReadyLane;
  });
  if (initialPath == cycle.paths.end()) {
    return false;
  }
  const std::size_t sourcePath =
      static_cast<std::size_t>(initialPath - cycle.paths.begin());
  const std::vector<SyncCoverDemandId> outerDemands = getLaneOwnershipDemands(
      graph, cycle, cycle.initialReadyLane, cycle.initialProducers);
  if (outerDemands.empty()) {
    return false;
  }
  for (SyncCoverNodeId consumer : initialPath->uses.front().consumers) {
    for (SyncCoverNodeId producer : cycle.initialProducers) {
      SyncCoverEdge edge = makeEdge(graph, consumer, producer, outerScope, 1);
      if (!isValidProtocolEdge(graph, edge)) {
        return false;
      }
      addBinding(expected, std::move(edge), 0, sets[sourcePath], *initialWait,
                 outerDemands);
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

bool verifyPrefixL0ReadyDescriptor(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!program.getTargetCapabilities().mte1L0ReadySetCompletesPrefix ||
      !hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  const SyncCoverGraph &graph = program.getGraph();
  std::vector<bool> claimed(descriptor.actions.size(), false);
  CanonicalSyncMechanismDescriptor expected;
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      if (use.producers.empty()) {
        return false;
      }
      const SyncCoverNodeId lastProducer = use.producers.back();
      const SyncCoverAnchor expectedReady{SyncCoverAnchorKind::AfterNode,
                                          lastProducer, 0, 0};
      const bool correctReadyAnchor =
          std::tie(use.ready.kind, use.ready.node, use.ready.scope,
                   use.ready.position) ==
          std::tie(expectedReady.kind, expectedReady.node, expectedReady.scope,
                   expectedReady.position);
      const bool orderedPrefix =
          llvm::all_of(use.producers, [&](SyncCoverNodeId producer) {
            return graph.getNodes()[producer].resource ==
                       cycle.producerResource &&
                   graph.getNodes()[producer].scope ==
                       graph.getNodes()[lastProducer].scope &&
                   graph.getNodes()[producer].order <=
                       graph.getNodes()[lastProducer].order;
          });
      if (!correctReadyAnchor || !orderedPrefix ||
          !syncCoverNodeCanProduceCompletion(graph, lastProducer,
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
      for (SyncCoverNodeId producer : use.producers) {
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
  }
  return allActionsClaimed(claimed) &&
         bindingsEqual(descriptor.supplies, expected.supplies);
}

bool verifyL0RoundTripDescriptor(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle, CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  if (!hasProtocolHeader(descriptor, domain, 1, cycle.lanes.size(),
                         cycle.recurrenceScope)) {
    return false;
  }
  const SyncCoverGraph &graph = program.getGraph();
  const bool mergeAlternativeRelease =
      program.getTargetCapabilities().mL0AlternativeJoinSetCompletes;
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
    const CanonicalSyncOwnershipPath &path = cycle.paths[pathIndex];
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      const std::optional<std::size_t> wait = claimAction(
          descriptor, claimed, CanonicalSyncActionKind::EventWait,
          cycle.producerResource, use.writeAcquire, 0, use.producerLane);
      if (!wait) {
        return false;
      }
      UseActions useActions;
      useActions.wait = *wait;
      const bool hasAlternativeConsumers = use.consumers.size() > 1;
      if (mergeAlternativeRelease && hasAlternativeConsumers) {
        if (use.readAcquire.kind != SyncCoverAnchorKind::ControlEntry ||
            use.release.kind != SyncCoverAnchorKind::ControlExit ||
            use.readAcquire.node != use.release.node ||
            use.readAcquire.scope != use.release.scope ||
            use.release.node >= graph.getControls().size() ||
            use.release.scope != path.scope) {
          return false;
        }
        const SyncCoverControl &control = graph.getControls()[use.release.node];
        std::set<unsigned> alternatives;
        const SyncCoverGuard &pathGuard = graph.getScopes()[path.scope].guard;
        for (SyncCoverNodeId consumer : use.consumers) {
          const bool invalidConsumer =
              consumer >= graph.getNodes().size() ||
              graph.getNodes()[consumer].resource != cycle.consumerResource;
          if (invalidConsumer) {
            return false;
          }
          const SyncCoverGuard &guard = graph.getNodes()[consumer].guard;
          std::vector<SyncCoverGuardLiteral> residual;
          std::set_difference(guard.literals.begin(), guard.literals.end(),
                              pathGuard.literals.begin(),
                              pathGuard.literals.end(),
                              std::back_inserter(residual));
          const bool invalidAlternative =
              residual.size() != 1 || residual.front().control != control.id ||
              residual.front().alternative >= control.alternatives ||
              !alternatives.insert(residual.front().alternative).second;
          if (invalidAlternative) {
            return false;
          }
          const SyncCoverScope &scope =
              graph.getScopes()[graph.getNodes()[consumer].scope];
          if (scope.parent != path.scope || !scope.mustExecuteWithinParent ||
              scope.isLoop || scope.guard.literals != guard.literals) {
            return false;
          }
        }
        const bool exhaustive =
            control.scope == path.scope &&
            alternatives.size() == control.alternatives &&
            *alternatives.begin() == 0 &&
            *alternatives.rbegin() + 1 == control.alternatives;
        if (!exhaustive) {
          return false;
        }
        const std::optional<std::size_t> set =
            claimAction(descriptor, claimed, CanonicalSyncActionKind::EventSet,
                        cycle.consumerResource, use.release, 0, use.lane);
        if (!set) {
          return false;
        }
        for (SyncCoverNodeId consumer : use.consumers) {
          useActions.sets.push_back({consumer, *set});
        }
      } else {
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
    const bool derivedProof =
        binding.proof == CanonicalSyncSupplyProof::VerifiedCompositeProtocol ||
        binding.proof == CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
    if (derivedProof) {
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
  if (cycle.kind != CanonicalSyncOwnershipKind::L1Tile) {
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
  } else if (cycle.protocol ==
             CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip) {
    result.ready =
        buildAccumulatorReady(program.getGraph(), cycle, readyDomain);
    result.release =
        buildBoundaryGuardedRelease(program.getGraph(), cycle, releaseDomain);
  } else if (cycle.kind == CanonicalSyncOwnershipKind::L1Tile) {
    result.ready = buildStableL1Ready(program.getGraph(), cycle, readyDomain);
    result.release =
        buildStableL1Release(program.getGraph(), cycle, releaseDomain);
  } else {
    result.ready =
        program.getTargetCapabilities().mte1L0ReadySetCompletesPrefix
            ? buildPrefixL0Ready(program.getGraph(), cycle, readyDomain)
            : buildL0Ready(program.getGraph(), cycle, readyDomain);
    result.release = buildL0RoundTripRelease(
        program.getGraph(), cycle, releaseDomain,
        program.getTargetCapabilities().mL0AlternativeJoinSetCompletes);
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
  const bool boundaryGuarded =
      cycle.protocol ==
      CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip;
  const bool l1 = cycle.kind == CanonicalSyncOwnershipKind::L1Tile;
  const bool ready =
      boundaryGuarded ? verifyAccumulatorReadyDescriptor(
                            program, cycle, readyDomain, protocol.ready)
      : alternating
          ? verifyAlternatingReadyDescriptor(program.getGraph(), cycle,
                                             readyDomain, protocol.ready)
      : l1 ? verifyStableL1ReadyDescriptor(program.getGraph(), cycle,
                                           readyDomain, protocol.ready)
      : program.getTargetCapabilities().mte1L0ReadySetCompletesPrefix
          ? verifyPrefixL0ReadyDescriptor(program, cycle, readyDomain,
                                          protocol.ready)
          : verifyL0ReadyDescriptor(program.getGraph(), cycle, readyDomain,
                                    protocol.ready);
  const bool release =
      boundaryGuarded
          ? verifyBoundaryGuardedReleaseDescriptor(
                program.getGraph(), cycle, releaseDomain, protocol.release)
      : alternating
          ? verifyAlternatingReleaseDescriptor(program.getGraph(), cycle,
                                               releaseDomain, protocol.release)
      : l1 ? verifyStableL1ReleaseDescriptor(program.getGraph(), cycle,
                                             releaseDomain, protocol.release)
           : verifyL0RoundTripDescriptor(program, cycle, releaseDomain,
                                         protocol.release);
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
  } else if (cycle.protocol ==
             CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip) {
    addAccumulatorCompositeBindings(result, program.getGraph(), cycle,
                                    *protocol);
  } else if (cycle.kind == CanonicalSyncOwnershipKind::L0Operand) {
    addL0CompositeBindings(result, program.getGraph(), cycle);
    const SyncCoverScopeId parent =
        program.getGraph().getScopes()[cycle.recurrenceScope].parent;
    const bool parentIsLoop = parent < program.getGraph().getScopes().size() &&
                              program.getGraph().getScopes()[parent].isLoop;
    if (parentIsLoop) {
      addHierarchicalOwnershipClosureBindings(result, program.getGraph(), cycle,
                                              parent);
    }
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
  std::vector<CanonicalSyncSupplyBinding> ownershipClosure;
  const bool compositeProtocol =
      cycle.protocol ==
          CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch ||
      cycle.protocol ==
          CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip ||
      cycle.kind == CanonicalSyncOwnershipKind::L0Operand;
  if (compositeProtocol) {
    llvm::erase_if(eventDescriptor.supplies,
                   [&](const CanonicalSyncSupplyBinding &binding) {
                     if (binding.proof ==
                         CanonicalSyncSupplyProof::VerifiedOwnershipClosure) {
                       ownershipClosure.push_back(binding);
                       return true;
                     }
                     if (binding.proof !=
                         CanonicalSyncSupplyProof::VerifiedCompositeProtocol) {
                       return false;
                     }
                     composite.push_back(binding);
                     return true;
                   });
  }
  const bool hasOneReadyUse =
      cycle.kind == CanonicalSyncOwnershipKind::L1Tile ||
      cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator ||
      (cycle.kind == CanonicalSyncOwnershipKind::L0Operand &&
       program.getTargetCapabilities().mte1L0ReadySetCompletesPrefix);
  const std::size_t readyUses =
      hasOneReadyUse ? 1 : cycle.paths.front().uses.front().producers.size();
  const std::optional<CanonicalSyncOwnershipProtocol> protocol =
      splitOwnershipProtocol(eventDescriptor, readyUses);
  if (!protocol || !verifyCanonicalSyncOwnershipProtocol(
                       program, cycle, readyDomain, releaseDomain, *protocol)) {
    return false;
  }
  if (compositeProtocol &&
      !verifyAtomicCompositeBindings(program.getGraph(), cycle, *protocol,
                                     composite)) {
    return false;
  }
  if (cycle.kind != CanonicalSyncOwnershipKind::L0Operand) {
    return ownershipClosure.empty();
  }
  const SyncCoverScopeId parent =
      program.getGraph().getScopes()[cycle.recurrenceScope].parent;
  const bool parentIsLoop = parent < program.getGraph().getScopes().size() &&
                            program.getGraph().getScopes()[parent].isLoop;
  return parentIsLoop ? verifyOwnershipClosureBindings(program.getGraph(),
                                                       cycle, *protocol, parent,
                                                       ownershipClosure)
                      : ownershipClosure.empty();
}

std::optional<CanonicalSyncMechanismDescriptor>
mlir::pto::makeCanonicalSyncHierarchicalL1Protocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle, SyncCoverScopeId outerScope,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain) {
  const SyncCoverGraph &graph = program.getGraph();
  const bool alternating =
      cycle.protocol == CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
  const bool valid =
      cycle.kind == CanonicalSyncOwnershipKind::L1Tile &&
      (cycle.protocol == CanonicalSyncOwnershipProtocolKind::RoundTrip ||
       alternating) &&
      program.getTargetCapabilities().mte1ScopeExitSetCompletesPrefix &&
      isNearestEnclosingLoop(graph, cycle.recurrenceScope, outerScope) &&
      verifyCanonicalSyncOwnershipCycle(program, cycle);
  if (!valid) {
    return std::nullopt;
  }

  CanonicalSyncOwnershipProtocol protocol;
  if (alternating) {
    protocol.ready = buildHierarchicalAlternatingReady(graph, cycle, outerScope,
                                                       readyDomain);
    protocol.release = buildHierarchicalAlternatingRelease(
        graph, cycle, outerScope, releaseDomain);
  } else {
    protocol.ready = buildStableL1Ready(graph, cycle, readyDomain);
    protocol.release = buildHierarchicalStableL1Release(
        graph, cycle, outerScope, releaseDomain);
  }
  const bool completeProtocol =
      !protocol.ready.actions.empty() && !protocol.release.actions.empty();
  if (!completeProtocol) {
    return std::nullopt;
  }
  CanonicalSyncMechanismDescriptor result = mergeOwnershipProtocol(protocol);
  if (alternating) {
    addAlternatingCompositeBindings(result, graph, cycle, protocol);
  }
  addHierarchicalOuterCompositeBindings(result, graph, cycle, outerScope,
                                        protocol);
  if (alternating) {
    addHierarchicalAlternatingReadySummaries(result, graph, cycle, outerScope,
                                             protocol);
  }
  addHierarchicalOwnershipClosureBindings(result, graph, cycle, outerScope);
  return result;
}

bool mlir::pto::verifyCanonicalSyncHierarchicalL1Protocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle, SyncCoverScopeId outerScope,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncMechanismDescriptor &descriptor) {
  const SyncCoverGraph &graph = program.getGraph();
  const bool alternating =
      cycle.protocol == CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
  const bool valid =
      cycle.kind == CanonicalSyncOwnershipKind::L1Tile &&
      (cycle.protocol == CanonicalSyncOwnershipProtocolKind::RoundTrip ||
       alternating) &&
      program.getTargetCapabilities().mte1ScopeExitSetCompletesPrefix &&
      isNearestEnclosingLoop(graph, cycle.recurrenceScope, outerScope) &&
      verifyCanonicalSyncOwnershipCycle(program, cycle);
  if (!valid) {
    return false;
  }

  CanonicalSyncMechanismDescriptor eventDescriptor = descriptor;
  std::vector<CanonicalSyncSupplyBinding> composite;
  std::vector<CanonicalSyncSupplyBinding> nestedSummaries;
  std::vector<CanonicalSyncSupplyBinding> ownershipClosure;
  llvm::erase_if(
      eventDescriptor.supplies, [&](const CanonicalSyncSupplyBinding &binding) {
        if (binding.proof ==
            CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary) {
          nestedSummaries.push_back(binding);
          return true;
        }
        if (binding.proof ==
            CanonicalSyncSupplyProof::VerifiedOwnershipClosure) {
          ownershipClosure.push_back(binding);
          return true;
        }
        if (binding.proof !=
                CanonicalSyncSupplyProof::VerifiedCompositeProtocol ||
            (!alternating && binding.edge.scope != outerScope)) {
          return false;
        }
        composite.push_back(binding);
        return true;
      });
  const std::optional<CanonicalSyncOwnershipProtocol> protocol =
      splitOwnershipProtocol(eventDescriptor, 1);
  if (!protocol) {
    return false;
  }
  const bool ready =
      alternating ? verifyHierarchicalAlternatingReadyDescriptor(
                        graph, cycle, outerScope, readyDomain, protocol->ready)
                  : verifyStableL1ReadyDescriptor(graph, cycle, readyDomain,
                                                  protocol->ready);
  const bool release =
      alternating
          ? verifyHierarchicalAlternatingReleaseDescriptor(
                graph, cycle, outerScope, releaseDomain, protocol->release)
          : verifyHierarchicalStableL1ReleaseDescriptor(
                graph, cycle, outerScope, releaseDomain, protocol->release);
  if (!ready || !release) {
    return false;
  }
  return verifyHierarchicalCompositeBindings(graph, cycle, *protocol,
                                             outerScope, composite) &&
         (alternating
              ? verifyNestedRecurrenceSummaryBindings(
                    graph, cycle, outerScope, *protocol, nestedSummaries)
              : nestedSummaries.empty()) &&
         verifyOwnershipClosureBindings(graph, cycle, *protocol, outerScope,
                                        ownershipClosure);
}
