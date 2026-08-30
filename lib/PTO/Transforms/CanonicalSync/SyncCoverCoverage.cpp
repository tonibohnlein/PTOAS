// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"

#include "SyncCoverCoverageInternal.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

namespace {

constexpr std::size_t kBitsPerWord = 64;

bool consumeProduct(SyncCoverCoverageWorkBudget *budget, std::size_t left,
                    std::size_t right) {
  if (!budget || left == 0 || right == 0) {
    return true;
  }
  const bool productOverflows =
      right > std::numeric_limits<std::size_t>::max() / left;
  if (productOverflows) {
    budget->exhausted = true;
    return false;
  }
  return budget->consume(left * right);
}

bool checkedSum(std::size_t first, std::size_t second, std::size_t &result) {
  const bool sumOverflows =
      second > std::numeric_limits<std::size_t>::max() - first;
  if (sumOverflows) {
    return false;
  }
  result = first + second;
  return true;
}

bool checkedProduct(std::size_t first, std::size_t second,
                    std::size_t &result) {
  const bool productOverflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (productOverflows) {
    return false;
  }
  result = first * second;
  return true;
}

bool denseWordsFit(std::size_t resultWords, std::size_t workspaceWords,
                   const SyncCoverCoverageLimits &limits) {
  return resultWords <= limits.maximumResultWords &&
         workspaceWords <= limits.maximumWorkspaceWords &&
         resultWords <= limits.maximumTotalWords &&
         workspaceWords <= limits.maximumTotalWords - resultWords;
}

bool checkedSum(std::size_t first, std::size_t second, std::size_t third,
                std::size_t &result) {
  return checkedSum(first, second, result) && checkedSum(result, third, result);
}

bool consumeVectorCopy(SyncCoverCoverageWorkBudget *budget,
                       std::size_t elements) {
  return consumeWork(budget, elements == 0 ? 1 : elements);
}

template <typename T, typename Compare>
bool meteredStableSort(std::vector<T> &values, Compare compare,
                       SyncCoverCoverageWorkBudget *budget) {
  if (!budget) {
    std::stable_sort(values.begin(), values.end(), compare);
    return true;
  }
  for (std::size_t index = 1; index < values.size(); ++index) {
    T value = std::move(values[index]);
    std::size_t position = index;
    while (position != 0) {
      if (!budget->consume()) {
        return false;
      }
      const bool precedes = compare(value, values[position - 1]);
      if (budget->exhausted) {
        return false;
      }
      if (!precedes) {
        break;
      }
      values[position] = std::move(values[position - 1]);
      --position;
    }
    values[position] = std::move(value);
  }
  return true;
}

bool chargeProjection(const SyncCoverGraph &graph,
                      const SyncCoverExpandedArena &arena,
                      bool completionProjection,
                      SyncCoverCoverageWorkBudget *budget) {
  if (!budget) {
    return true;
  }
  // An immediate-child lookup may repeatedly search enclosing loops. Charging
  // one complete scope walk for every possible enclosing scope dominates that
  // nested ancestry search. Endpoint projection reserves two such searches;
  // completion projection reserves two endpoints plus its export checks.
  const std::size_t hierarchyFactors = completionProjection ? 6 : 2;
  const std::size_t lookupScans = completionProjection ? 4 : 2;
  for (std::size_t factor = 0; factor < hierarchyFactors; ++factor) {
    if (!consumeProduct(budget, graph.getScopes().size(),
                        graph.getScopes().size())) {
      return false;
    }
  }
  return consumeProduct(budget, arena.getVirtualNodeCount(), lookupScans);
}

bool chargeSupplyCopy(const std::vector<SyncCoverCompletionSupply> &supplies,
                      SyncCoverCoverageWorkBudget *budget) {
  for (const SyncCoverCompletionSupply &supply : supplies) {
    const bool supplyUnavailable =
        !consumeWork(budget) ||
        !consumeVectorCopy(budget, supply.edge.sourceGuard.literals.size()) ||
        !consumeVectorCopy(budget, supply.edge.targetGuard.literals.size()) ||
        !consumeVectorCopy(budget, supply.allowedDemands.size());
    if (supplyUnavailable) {
      return false;
    }
  }
  return true;
}

bool chargeCompletionCanonicalization(const SyncCoverGraph &graph,
                                      const SyncCoverCompletionSupply &supply,
                                      SyncCoverCoverageWorkBudget *budget) {
  if (!budget) {
    return true;
  }
  const SyncCoverEdge &edge = supply.edge;
  const bool invalidEndpoint = edge.source >= graph.getNodes().size() ||
                               edge.target >= graph.getNodes().size();
  if (invalidEndpoint) {
    return consumeWork(budget);
  }
  std::size_t sourceLiterals = 0;
  std::size_t targetLiterals = 0;
  const std::size_t scopes = graph.getScopes().size();
  std::size_t totalLiterals = 0;
  std::size_t scopeCharge = 0;
  if (!checkedSum(edge.sourceGuard.literals.size(),
                  graph.getNodes()[edge.source].guard.literals.size(),
                  sourceLiterals) ||
      !checkedSum(edge.targetGuard.literals.size(),
                  graph.getNodes()[edge.target].guard.literals.size(),
                  targetLiterals) ||
      !checkedSum(sourceLiterals, targetLiterals, totalLiterals) ||
      !checkedSum(scopes, 3, scopeCharge)) {
    budget->exhausted = true;
    return false;
  }
  // Reserve the complete variable work performed by endpoint-guard insertion,
  // sorting/deduplication, validation scope walks, and endpoint compatibility.
  return consumeProduct(budget, sourceLiterals, sourceLiterals) &&
         consumeProduct(budget, targetLiterals, targetLiterals) &&
         consumeProduct(budget, totalLiterals, scopeCharge) &&
         consumeProduct(budget, scopes, 3);
}

struct IndexedSupply {
  std::size_t source = 0;
  std::size_t target = 0;
  unsigned sourceCopy = 0;
  unsigned targetCopy = 0;
  SyncCoverMechanismId mechanism = 0;
  std::size_t supply = 0;
};

struct SupplyIndex {
  std::vector<IndexedSupply> edges;
  std::vector<std::size_t> outgoingOffsets;
};

class CoverageWorkspace {
  static std::size_t getStateCount(std::size_t virtualNodeCount) {
    return virtualNodeCount <= std::numeric_limits<std::size_t>::max() / 2
               ? virtualNodeCount * 2
               : 0;
  }

public:
  explicit CoverageWorkspace(std::size_t virtualNodeCount)
      : seenEpoch_(getStateCount(virtualNodeCount), 0),
        activeEpoch_(virtualNodeCount, 0), activeNodes_(virtualNodeCount, 0) {}

  void beginDemand() {
    if (epoch_ == std::numeric_limits<std::uint32_t>::max()) {
      std::fill(seenEpoch_.begin(), seenEpoch_.end(), 0);
      std::fill(activeEpoch_.begin(), activeEpoch_.end(), 0);
      epoch_ = 1;
    } else {
      ++epoch_;
    }
    ready_.clear();
  }

  bool wasSeen(std::size_t state) const { return seenEpoch_[state] == epoch_; }

  void markSeen(std::size_t state) {
    seenEpoch_[state] = epoch_;
    ready_.push_back(state);
  }

  bool isNodeActive(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                    const DemandContext &context,
                    const SyncCoverExpandedArena &arena,
                    std::size_t virtualNode,
                    SyncCoverCoverageWorkBudget *budget = nullptr) {
    if (activeEpoch_[virtualNode] == epoch_) {
      return activeNodes_[virtualNode] != 0;
    }
    activeEpoch_[virtualNode] = epoch_;
    const std::optional<SyncCoverNodeId> operation =
        arena.getOperationForVirtualNode(virtualNode);
    if (!operation) {
      activeNodes_[virtualNode] = 1;
      return true;
    }
    const std::optional<unsigned> copy =
        arena.getCopyForVirtualNode(virtualNode);
    const bool active =
        copy &&
        nodeInstanceAvailable(graph, demand, *operation, *copy, budget) &&
        guardIsImplied(graph, demand, context,
                       graph.getNodes()[*operation].guard, *copy, budget);
    activeNodes_[virtualNode] = active ? 1 : 0;
    return active;
  }

  const std::vector<std::size_t> &getReady() const { return ready_; }

private:
  std::uint32_t epoch_ = 0;
  std::vector<std::uint32_t> seenEpoch_;
  std::vector<std::uint32_t> activeEpoch_;
  std::vector<std::uint8_t> activeNodes_;
  std::vector<std::size_t> ready_;
};

std::size_t getStateIndex(std::size_t virtualNode, bool completion) {
  return virtualNode * 2 + static_cast<std::size_t>(completion);
}

std::optional<std::size_t> transition(SyncCoverEdgeKind kind,
                                      std::size_t target, std::size_t state) {
  const bool hasCompletion = (state % 2) != 0;
  switch (kind) {
  case SyncCoverEdgeKind::CompletionSupply:
    return getStateIndex(target, true);
  case SyncCoverEdgeKind::CertifiedCompletionFrontier:
    return getStateIndex(target, hasCompletion);
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
    return hasCompletion
               ? std::optional<std::size_t>(getStateIndex(target, true))
               : std::nullopt;
  }
  return std::nullopt;
}

SupplyIndex
buildSupplyIndex(const SyncCoverGraph &graph,
                 const SyncCoverExpandedProgram &expansion,
                 const SyncCoverExpandedArena &arena,
                 const std::vector<SyncCoverCompletionSupply> &supplies,
                 SyncCoverCoverageWorkBudget *budget = nullptr) {
  SupplyIndex result;
  for (std::size_t supplyId = 0; supplyId < supplies.size(); ++supplyId) {
    const SyncCoverCompletionSupply &supply = supplies[supplyId];
    for (unsigned copy = 0; copy <= arena.getHorizon(); ++copy) {
      const bool projectionWorkUnavailable =
          !consumeWork(budget) || !chargeProjection(graph, arena, true, budget);
      if (projectionWorkUnavailable) {
        return result;
      }
      const std::optional<SyncCoverProjectedCompletion> projected =
          expansion.projectCompletion(graph, arena, supply.edge, copy,
                                      supply.exportsCompletionAtScopeExit);
      if (projected) {
        result.edges.push_back({projected->source, projected->target,
                                projected->sourceCopy, projected->targetCopy,
                                supply.mechanism, supplyId});
      }
    }
  }
  const bool sorted = meteredStableSort(
      result.edges,
      [](const IndexedSupply &left, const IndexedSupply &right) {
        return std::tie(left.source, left.target, left.sourceCopy,
                        left.targetCopy, left.mechanism, left.supply) <
               std::tie(right.source, right.target, right.sourceCopy,
                        right.targetCopy, right.mechanism, right.supply);
      },
      budget);
  if (!sorted) {
    return result;
  }
  const bool offsetsUnavailable =
      arena.getVirtualNodeCount() == std::numeric_limits<std::size_t>::max() ||
      !consumeWork(budget, arena.getVirtualNodeCount() + 1);
  if (offsetsUnavailable) {
    return result;
  }
  result.outgoingOffsets.assign(arena.getVirtualNodeCount() + 1, 0);
  for (const IndexedSupply &edge : result.edges) {
    if (!consumeWork(budget)) {
      return result;
    }
    ++result.outgoingOffsets[edge.source + 1];
  }
  for (std::size_t index = 1; index < result.outgoingOffsets.size(); ++index) {
    if (!consumeWork(budget)) {
      return result;
    }
    result.outgoingOffsets[index] += result.outgoingOffsets[index - 1];
  }
  return result;
}

template <typename Visitor>
bool visitSupplyOutgoing(const SupplyIndex &index, std::size_t source,
                         Visitor &&visitor) {
  const bool invalidNode = index.outgoingOffsets.empty() ||
                           source >= index.outgoingOffsets.size() - 1;
  if (invalidNode) {
    return true;
  }
  for (std::size_t edge = index.outgoingOffsets[source];
       edge < index.outgoingOffsets[source + 1]; ++edge) {
    if (!visitor(index.edges[edge])) {
      return false;
    }
  }
  return true;
}

bool coversDemand(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                  SyncCoverDemandId demandId,
                  const SyncCoverExpandedProgram &expansion,
                  const SyncCoverExpandedArena &arena,
                  const SupplyIndex &supplyIndex,
                  const std::vector<SyncCoverCompletionSupply> &supplies,
                  CoverageWorkspace &workspace,
                  SyncCoverCoverageWorkBudget *budget = nullptr) {
  const DemandContext context = makeDemandContext(graph, demand, budget);
  if (budget && budget->exhausted) {
    return false;
  }
  const bool projectionWorkUnavailable =
      !chargeProjection(graph, arena, false, budget) ||
      !chargeProjection(graph, arena, false, budget);
  if (projectionWorkUnavailable) {
    return false;
  }
  const std::optional<std::size_t> source =
      expansion.projectEndpoint(graph, arena, demand.source, 0);
  const std::optional<std::size_t> target =
      expansion.projectEndpoint(graph, arena, demand.target, demand.distance);
  const bool invalidDemand =
      !context.valid || !source || !target ||
      arena.getVirtualNodeCount() > std::numeric_limits<std::size_t>::max() / 2;
  if (invalidDemand) {
    return false;
  }

  const std::size_t start = getStateIndex(*source, false);
  const std::size_t goal = getStateIndex(*target, true);
  workspace.beginDemand();
  workspace.markSeen(start);
  auto enqueue = [&](std::size_t next) {
    if (!workspace.wasSeen(next)) {
      workspace.markSeen(next);
    }
  };

  for (std::size_t readyIndex = 0; readyIndex < workspace.getReady().size();
       ++readyIndex) {
    if (!consumeWork(budget)) {
      return false;
    }
    const std::size_t state = workspace.getReady()[readyIndex];
    const std::size_t virtualNode = state / 2;
    for (const SyncCoverExpandedEdge &edge :
         arena.getOutgoingEdges(virtualNode)) {
      if (!consumeWork(budget)) {
        return false;
      }
      const bool activeEndpoints =
          edge.targetCopy <= demand.distance &&
          workspace.isNodeActive(graph, demand, context, arena, edge.source,
                                 budget) &&
          workspace.isNodeActive(graph, demand, context, arena, edge.target,
                                 budget);
      const bool active =
          activeEndpoints &&
          (!edge.graphEdge ||
           edgeGuardsActive(graph, demand, context,
                            graph.getEdges()[*edge.graphEdge], edge.sourceCopy,
                            edge.targetCopy, budget));
      if (active) {
        if (const std::optional<std::size_t> next =
                transition(edge.kind, edge.target, state)) {
          enqueue(*next);
        }
      }
    }

    const bool suppliesVisited = visitSupplyOutgoing(
        supplyIndex, virtualNode, [&](const IndexedSupply &indexed) -> bool {
          if (!consumeWork(budget)) {
            return false;
          }
          const SyncCoverCompletionSupply &description =
              supplies[indexed.supply];
          if (description.applicability ==
                  SyncCoverSupplyApplicability::DistanceZeroOnly &&
              demand.distance != 0) {
            return true;
          }
          const bool restricted = !description.allowedDemands.empty();
          bool demandAllowed = false;
          for (SyncCoverDemandId allowed : description.allowedDemands) {
            if (!consumeWork(budget)) {
              return false;
            }
            if (allowed == demandId) {
              demandAllowed = true;
              break;
            }
            if (allowed > demandId) {
              break;
            }
          }
          if (restricted && !demandAllowed) {
            return true;
          }
          const SyncCoverEdge &supply = description.edge;
          const bool active =
              indexed.targetCopy <= demand.distance &&
              nodeInstanceAvailable(graph, demand, supply.source,
                                    indexed.sourceCopy, budget) &&
              nodeInstanceAvailable(graph, demand, supply.target,
                                    indexed.targetCopy, budget) &&
              workspace.isNodeActive(graph, demand, context, arena, virtualNode,
                                     budget) &&
              workspace.isNodeActive(graph, demand, context, arena,
                                     indexed.target, budget) &&
              edgeGuardsActive(graph, demand, context, supply,
                               indexed.sourceCopy, indexed.targetCopy, budget);
          if (active) {
            enqueue(getStateIndex(indexed.target, true));
          }
          return !budget || !budget->exhausted;
        });
    if (!suppliesVisited) {
      return false;
    }
  }
  return (!budget || !budget->exhausted) && workspace.wasSeen(goal);
}

bool coverageInputsValid(const SyncCoverGraph &graph,
                         const SyncCoverExpandedProgram &expansion) {
  return graph.isStructureFrozen() && expansion.isForGraph(graph) &&
         expansion.getError() != SyncCoverExpansionError::InvalidGraph &&
         expansion.getError() != SyncCoverExpansionError::InvalidLimits;
}

std::vector<SyncCoverDemandId> allDemandIds(const SyncCoverGraph &graph) {
  std::vector<SyncCoverDemandId> result(graph.getDemands().size());
  for (SyncCoverDemandId demand = 0; demand < result.size(); ++demand) {
    result[demand] = demand;
  }
  return result;
}

std::vector<SyncCoverDemandId>
hierarchicalDemandOrder(const SyncCoverGraph &graph,
                        const std::vector<SyncCoverDemandId> &demands,
                        SyncCoverCoverageWorkBudget *budget = nullptr) {
  const bool workspaceUnavailable =
      !consumeVectorCopy(budget, graph.getScopes().size()) ||
      !consumeVectorCopy(budget, demands.size());
  if (workspaceUnavailable) {
    return {};
  }
  std::vector<std::size_t> depths(graph.getScopes().size(), 0);
  for (SyncCoverScopeId scope = 1; scope < graph.getScopes().size(); ++scope) {
    SyncCoverScopeId current = scope;
    while (current != 0) {
      if (!consumeWork(budget)) {
        return {};
      }
      ++depths[scope];
      current = graph.getScopes()[current].parent;
    }
  }
  std::vector<SyncCoverDemandId> result = demands;
  const bool sorted = meteredStableSort(
      result,
      [&](auto first, auto second) {
        const SyncCoverScopeId firstScope = graph.getDemands()[first].scope;
        const SyncCoverScopeId secondScope = graph.getDemands()[second].scope;
        return std::make_pair(depths[firstScope], first) >
               std::make_pair(depths[secondScope], second);
      },
      budget);
  if (!sorted) {
    return {};
  }
  return result;
}

bool demandIdsValid(const SyncCoverGraph &graph,
                    const std::vector<SyncCoverDemandId> &demands,
                    SyncCoverCoverageWorkBudget *budget = nullptr) {
  for (std::size_t index = 0; index < demands.size(); ++index) {
    const bool workUnavailable = !consumeWork(budget);
    const bool invalidDemand = demands[index] >= graph.getDemands().size();
    const bool nonIncreasing =
        index != 0 && demands[index - 1] >= demands[index];
    if (workUnavailable || invalidDemand || nonIncreasing) {
      return false;
    }
  }
  return true;
}

bool canonicalizeSupplies(const SyncCoverGraph &graph,
                          std::vector<SyncCoverCompletionSupply> &supplies,
                          std::size_t mechanismCount,
                          SyncCoverCoverageWorkBudget *budget = nullptr) {
  for (SyncCoverCompletionSupply &supply : supplies) {
    const bool invalidApplicability =
        supply.applicability != SyncCoverSupplyApplicability::AllDemands &&
        supply.applicability != SyncCoverSupplyApplicability::DistanceZeroOnly;
    const bool invalidDemandFilter =
        !demandIdsValid(graph, supply.allowedDemands, budget);
    const bool canonicalizationWorkUnavailable =
        !consumeWork(budget) ||
        !chargeCompletionCanonicalization(graph, supply, budget);
    if (canonicalizationWorkUnavailable) {
      return false;
    }
    const SyncCoverGraphError edgeError =
        graph.canonicalizeCompletionEdge(supply.edge);
    const bool invalidExport =
        supply.exportsCompletionAtScopeExit &&
        (edgeError != SyncCoverGraphError::None || supply.edge.distance == 0 ||
         !graph.getScopes()[supply.edge.scope].isLoop);
    if (supply.mechanism >= mechanismCount || invalidDemandFilter ||
        invalidApplicability || edgeError != SyncCoverGraphError::None ||
        invalidExport) {
      return false;
    }
  }
  return meteredStableSort(
      supplies,
      [&](const SyncCoverCompletionSupply &left,
          const SyncCoverCompletionSupply &right) {
        std::size_t leftMetadata = 0;
        std::size_t rightMetadata = 0;
        const bool validSizes =
            checkedSum(left.edge.sourceGuard.literals.size(),
                       left.edge.targetGuard.literals.size(),
                       left.allowedDemands.size(), leftMetadata) &&
            checkedSum(right.edge.sourceGuard.literals.size(),
                       right.edge.targetGuard.literals.size(),
                       right.allowedDemands.size(), rightMetadata);
        if (!validSizes) {
          if (budget) {
            budget->exhausted = true;
          }
          return false;
        }
        const bool comparisonWorkUnavailable =
            !consumeWork(budget, leftMetadata) ||
            !consumeWork(budget, rightMetadata);
        if (comparisonWorkUnavailable) {
          return false;
        }
        return std::tie(left.mechanism, left.edge.source, left.edge.target,
                        left.edge.scope, left.edge.distance,
                        left.edge.sourceGuard.literals,
                        left.edge.targetGuard.literals, left.allowedDemands,
                        left.exportsCompletionAtScopeExit, left.applicability) <
               std::tie(right.mechanism, right.edge.source, right.edge.target,
                        right.edge.scope, right.edge.distance,
                        right.edge.sourceGuard.literals,
                        right.edge.targetGuard.literals, right.allowedDemands,
                        right.exportsCompletionAtScopeExit,
                        right.applicability);
      },
      budget);
}

class SingletonWorkspace {
public:
  SingletonWorkspace(std::size_t virtualNodes, std::size_t mechanisms)
      : wordsPerNode_(mechanisms / kBitsPerWord +
                      (mechanisms % kBitsPerWord != 0 ? 1 : 0)),
        reachable_(virtualNodes * wordsPerNode_, 0), queued_(virtualNodes, 0) {}

  bool add(std::size_t node, std::size_t mechanism) {
    const std::size_t word = mechanism / kBitsPerWord;
    const std::uint64_t bit = std::uint64_t{1} << (mechanism % kBitsPerWord);
    std::uint64_t &value = reachable_[node * wordsPerNode_ + word];
    const bool alreadyPresent = (value & bit) != 0;
    if (alreadyPresent) {
      return false;
    }
    value |= bit;
    enqueue(node);
    return true;
  }

  bool contains(std::size_t node, std::size_t mechanism) const {
    const std::size_t word = mechanism / kBitsPerWord;
    const std::uint64_t bit = std::uint64_t{1} << (mechanism % kBitsPerWord);
    return (reachable_[node * wordsPerNode_ + word] & bit) != 0;
  }

  void unite(std::size_t source, std::size_t target) {
    bool changed = false;
    for (std::size_t word = 0; word < wordsPerNode_; ++word) {
      std::uint64_t &targetWord = reachable_[target * wordsPerNode_ + word];
      const std::uint64_t combined =
          targetWord | reachable_[source * wordsPerNode_ + word];
      changed |= combined != targetWord;
      targetWord = combined;
    }
    if (changed) {
      enqueue(target);
    }
  }

  std::optional<std::size_t> pop() {
    if (ready_.empty()) {
      return std::nullopt;
    }
    const std::size_t node = ready_.front();
    ready_.pop_front();
    queued_[node] = 0;
    return node;
  }

  const std::uint64_t *words(std::size_t node) const {
    return reachable_.data() + node * wordsPerNode_;
  }
  std::size_t wordCount() const { return wordsPerNode_; }

private:
  void enqueue(std::size_t node) {
    if (queued_[node] == 0) {
      queued_[node] = 1;
      ready_.push_back(node);
    }
  }

  std::size_t wordsPerNode_ = 0;
  std::vector<std::uint64_t> reachable_;
  std::vector<std::uint8_t> queued_;
  std::deque<std::size_t> ready_;
};

class PairWorkspace {
public:
  PairWorkspace(std::size_t virtualNodes, std::size_t pairs)
      : wordsPerNode_(pairs / kBitsPerWord +
                      (pairs % kBitsPerWord != 0 ? 1 : 0)),
        reachable_(virtualNodes * wordsPerNode_, 0), queued_(virtualNodes, 0) {}

  bool add(std::size_t node, std::size_t pair) {
    const std::size_t word = pair / kBitsPerWord;
    const std::uint64_t bit = std::uint64_t{1} << (pair % kBitsPerWord);
    std::uint64_t &value = reachable_[node * wordsPerNode_ + word];
    const bool alreadyReachable = (value & bit) != 0;
    if (alreadyReachable) {
      return false;
    }
    value |= bit;
    enqueue(node);
    return true;
  }

  bool contains(std::size_t node, std::size_t pair) const {
    const std::size_t word = pair / kBitsPerWord;
    const std::uint64_t bit = std::uint64_t{1} << (pair % kBitsPerWord);
    return (reachable_[node * wordsPerNode_ + word] & bit) != 0;
  }

  void unite(std::size_t source, std::size_t target) {
    bool changed = false;
    for (std::size_t word = 0; word < wordsPerNode_; ++word) {
      std::uint64_t &targetWord = reachable_[target * wordsPerNode_ + word];
      const std::uint64_t combined =
          targetWord | reachable_[source * wordsPerNode_ + word];
      changed |= combined != targetWord;
      targetWord = combined;
    }
    if (changed) {
      enqueue(target);
    }
  }

  std::optional<std::size_t> pop() {
    if (ready_.empty()) {
      return std::nullopt;
    }
    const std::size_t node = ready_.front();
    ready_.pop_front();
    queued_[node] = 0;
    return node;
  }

  const std::uint64_t *words(std::size_t node) const {
    return reachable_.data() + node * wordsPerNode_;
  }

private:
  void enqueue(std::size_t node) {
    if (queued_[node] == 0) {
      queued_[node] = 1;
      ready_.push_back(node);
    }
  }

  std::size_t wordsPerNode_ = 0;
  std::vector<std::uint64_t> reachable_;
  std::vector<std::uint8_t> queued_;
  std::deque<std::size_t> ready_;
};

bool supplyIsActive(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                    const DemandContext &context,
                    const SyncCoverCompletionSupply &description,
                    const IndexedSupply &indexed, std::size_t &targetNode) {
  const SyncCoverEdge &supply = description.edge;
  const bool active =
      indexed.targetCopy <= demand.distance &&
      nodeInstanceAvailable(graph, demand, supply.source, indexed.sourceCopy) &&
      nodeInstanceAvailable(graph, demand, supply.target, indexed.targetCopy) &&
      edgeGuardsActive(graph, demand, context, supply, indexed.sourceCopy,
                       indexed.targetCopy);
  if (!active) {
    return false;
  }
  targetNode = indexed.target;
  return true;
}

bool virtualNodeAvailable(const SyncCoverGraph &graph,
                          const SyncCoverDemand &demand,
                          const DemandContext &context,
                          const SyncCoverExpandedArena &arena,
                          std::size_t virtualNode, unsigned copy) {
  const std::optional<SyncCoverNodeId> operation =
      arena.getOperationForVirtualNode(virtualNode);
  return !operation ||
         (nodeInstanceAvailable(graph, demand, *operation, copy) &&
          guardIsImplied(graph, demand, context,
                         graph.getNodes()[*operation].guard, copy));
}

struct SingletonSeed {
  SyncCoverMechanismId mechanism = 0;
  std::size_t target = 0;
};

struct SingletonSeedCollection {
  std::vector<SingletonSeed> seeds;
  bool baselineCovers = false;
};

SingletonSeedCollection collectSingletonSeeds(
    const SyncCoverGraph &graph, const SyncCoverDemand &demand,
    SyncCoverDemandId demandId, const DemandContext &context,
    const SyncCoverExpandedProgram &expansion,
    const SyncCoverExpandedArena &arena, const SupplyIndex &supplyIndex,
    const std::vector<SyncCoverCompletionSupply> &supplies) {
  SingletonSeedCollection result;
  const std::optional<std::size_t> source =
      expansion.projectEndpoint(graph, arena, demand.source, 0);
  const bool invalidSource =
      !source ||
      arena.getVirtualNodeCount() > std::numeric_limits<std::size_t>::max() / 2;
  if (invalidSource) {
    return result;
  }
  std::vector<std::uint8_t> seen(arena.getVirtualNodeCount() * 2, 0);
  std::deque<std::size_t> ready;
  const auto enqueue = [&](std::size_t state) {
    if (seen[state] == 0) {
      seen[state] = 1;
      ready.push_back(state);
    }
  };
  enqueue(getStateIndex(*source, false));
  while (!ready.empty()) {
    const std::size_t state = ready.front();
    ready.pop_front();
    const std::size_t virtualNode = state / 2;
    for (const SyncCoverExpandedEdge &edge :
         arena.getOutgoingEdges(virtualNode)) {
      const bool active = edge.targetCopy <= demand.distance &&
                          virtualNodeAvailable(graph, demand, context, arena,
                                               edge.source, edge.sourceCopy) &&
                          virtualNodeAvailable(graph, demand, context, arena,
                                               edge.target, edge.targetCopy) &&
                          (!edge.graphEdge ||
                           edgeGuardsActive(graph, demand, context,
                                            graph.getEdges()[*edge.graphEdge],
                                            edge.sourceCopy, edge.targetCopy));
      if (active) {
        if (const std::optional<std::size_t> next =
                transition(edge.kind, edge.target, state)) {
          enqueue(*next);
        }
      }
    }

    visitSupplyOutgoing(
        supplyIndex, virtualNode, [&](const IndexedSupply &indexed) -> bool {
          const SyncCoverCompletionSupply &description =
              supplies[indexed.supply];
          if (description.applicability ==
                  SyncCoverSupplyApplicability::DistanceZeroOnly &&
              demand.distance != 0) {
            return true;
          }
          const bool restricted = !description.allowedDemands.empty();
          const bool demandAllowed =
              std::binary_search(description.allowedDemands.begin(),
                                 description.allowedDemands.end(), demandId);
          if (restricted && !demandAllowed) {
            return true;
          }
          std::size_t target = 0;
          if (supplyIsActive(graph, demand, context, description, indexed,
                             target)) {
            result.seeds.push_back({indexed.mechanism, target});
          }
          return true;
        });
  }
  const std::optional<std::size_t> goal =
      expansion.projectEndpoint(graph, arena, demand.target, demand.distance);
  result.baselineCovers = goal && seen[getStateIndex(*goal, true)] != 0;
  std::sort(result.seeds.begin(), result.seeds.end(),
            [](const SingletonSeed &left, const SingletonSeed &right) {
              return std::tie(left.mechanism, left.target) <
                     std::tie(right.mechanism, right.target);
            });
  result.seeds.erase(
      std::unique(result.seeds.begin(), result.seeds.end(),
                  [](const SingletonSeed &left, const SingletonSeed &right) {
                    return left.mechanism == right.mechanism &&
                           left.target == right.target;
                  }),
      result.seeds.end());
  return result;
}

} // namespace

SyncCoverDemandSet::SyncCoverDemandSet(std::size_t size)
    : size_(size),
      words_(size / kBitsPerWord + (size % kBitsPerWord != 0 ? 1 : 0), 0) {}

std::size_t SyncCoverDemandSet::count() const {
  std::size_t result = 0;
  for (std::uint64_t word : words_) {
    result += static_cast<std::size_t>(__builtin_popcountll(word));
  }
  return result;
}

bool SyncCoverDemandSet::contains(SyncCoverDemandId demand) const {
  return demand < size_ && (words_[demand / kBitsPerWord] &
                            (std::uint64_t{1} << (demand % kBitsPerWord))) != 0;
}

bool SyncCoverDemandSet::insert(SyncCoverDemandId demand) {
  if (demand >= size_) {
    return false;
  }
  words_[demand / kBitsPerWord] |= std::uint64_t{1} << (demand % kBitsPerWord);
  return true;
}

void SyncCoverDemandSet::unite(const SyncCoverDemandSet &other) {
  if (size_ != other.size_) {
    return;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    words_[index] |= other.words_[index];
  }
}

void SyncCoverDemandSet::subtract(const SyncCoverDemandSet &other) {
  if (size_ != other.size_) {
    return;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    words_[index] &= ~other.words_[index];
  }
}

bool SyncCoverDemandSet::containsAll(const SyncCoverDemandSet &other) const {
  if (size_ != other.size_) {
    return false;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    const bool missing = (other.words_[index] & ~words_[index]) != 0;
    if (missing) {
      return false;
    }
  }
  return true;
}

SyncCoverCoverageResult mlir::pto::computeSyncCoverCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const std::vector<SyncCoverCompletionSupply> &inputSupplies) {
  return computeSyncCoverCoverage(graph, expansion, inputSupplies,
                                  allDemandIds(graph), nullptr);
}

SyncCoverCoverageResult mlir::pto::computeSyncCoverCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const std::vector<SyncCoverCompletionSupply> &inputSupplies,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverCoverageResult result;
  const std::size_t demandWords =
      graph.getDemands().size() / kBitsPerWord +
      (graph.getDemands().size() % kBitsPerWord != 0 ? 1 : 0);
  if (!consumeVectorCopy(workBudget, demandWords)) {
    result.error = SyncCoverCoverageError::WorkLimitExceeded;
    return result;
  }
  result.covered = SyncCoverDemandSet(graph.getDemands().size());
  const bool invalidInputs = !coverageInputsValid(graph, expansion) ||
                             !demandIdsValid(graph, activeDemands, workBudget);
  if (workBudget && workBudget->exhausted) {
    result.error = SyncCoverCoverageError::WorkLimitExceeded;
    return result;
  }
  if (invalidInputs) {
    result.error = SyncCoverCoverageError::InvalidGraph;
    return result;
  }
  const bool baseUnavailable =
      expansion.getError() == SyncCoverExpansionError::BaseLimitExceeded;
  if (baseUnavailable) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
    result.unavailableDemands.reserve(activeDemands.size());
    for (SyncCoverDemandId demand : activeDemands) {
      result.unavailableDemands.push_back(demand);
    }
    return result;
  }

  if (!chargeSupplyCopy(inputSupplies, workBudget)) {
    result.error = SyncCoverCoverageError::WorkLimitExceeded;
    return result;
  }
  std::vector<SyncCoverCompletionSupply> supplies = inputSupplies;
  if (!consumeVectorCopy(workBudget, supplies.size())) {
    result.error = SyncCoverCoverageError::WorkLimitExceeded;
    return result;
  }
  const std::size_t mechanismCount =
      supplies.empty() ? 0
                       : std::max_element(
                             supplies.begin(), supplies.end(),
                             [](const auto &left, const auto &right) {
                               return left.mechanism < right.mechanism;
                             })->mechanism +
                             1;
  if (!canonicalizeSupplies(graph, supplies, mechanismCount, workBudget)) {
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverCoverageError::WorkLimitExceeded;
      return result;
    }
    result.error = SyncCoverCoverageError::InvalidSupply;
    return result;
  }

  std::map<const SyncCoverExpandedArena *, CoverageWorkspace> workspaces;
  std::map<const SyncCoverExpandedArena *, SupplyIndex> supplyIndices;
  const std::vector<SyncCoverDemandId> orderedDemands =
      hierarchicalDemandOrder(graph, activeDemands, workBudget);
  if (workBudget && workBudget->exhausted) {
    result.error = SyncCoverCoverageError::WorkLimitExceeded;
    return result;
  }
  for (SyncCoverDemandId demandId : orderedDemands) {
    if (!consumeWork(workBudget)) {
      result.error = SyncCoverCoverageError::WorkLimitExceeded;
      return result;
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverExpandedArena *arena = expansion.getArena(demand);
    if (!arena) {
      result.unavailableDemands.push_back(demandId);
      continue;
    }
    const bool arenaLookupWorkUnavailable =
        !consumeWork(workBudget, workspaces.size() + 1) ||
        !consumeWork(workBudget, supplyIndices.size() + 1);
    if (arenaLookupWorkUnavailable) {
      result.error = SyncCoverCoverageError::WorkLimitExceeded;
      return result;
    }
    auto existingWorkspace = workspaces.find(arena);
    const bool workspaceUnavailable =
        existingWorkspace == workspaces.end() &&
        !consumeProduct(workBudget, arena->getVirtualNodeCount(), 4);
    if (workspaceUnavailable) {
      result.error = SyncCoverCoverageError::WorkLimitExceeded;
      return result;
    }
    auto [workspace, inserted] =
        workspaces.try_emplace(arena, arena->getVirtualNodeCount());
    (void)inserted;
    auto [supplyIndex, indexInserted] = supplyIndices.try_emplace(arena);
    if (indexInserted) {
      supplyIndex->second =
          buildSupplyIndex(graph, expansion, *arena, supplies, workBudget);
      if (workBudget && workBudget->exhausted) {
        result.error = SyncCoverCoverageError::WorkLimitExceeded;
        return result;
      }
    }
    if (coversDemand(graph, demand, demandId, expansion, *arena,
                     supplyIndex->second, supplies, workspace->second,
                     workBudget)) {
      result.covered.insert(demandId);
    }
    if (workBudget && workBudget->exhausted) {
      result.error = SyncCoverCoverageError::WorkLimitExceeded;
      return result;
    }
  }
  if (!result.unavailableDemands.empty()) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
  }
  return result;
}

SyncCoverSingletonCoverageResult mlir::pto::computeSyncCoverSingletonCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &inputSupplies,
    SyncCoverCoverageLimits limits) {
  return computeSyncCoverSingletonCoverage(graph, expansion, mechanismCount,
                                           inputSupplies, allDemandIds(graph),
                                           limits);
}

SyncCoverSingletonCoverageResult mlir::pto::computeSyncCoverSingletonCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &inputSupplies,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageLimits limits) {
  SyncCoverSingletonCoverageResult result;
  const bool invalidInputs = !coverageInputsValid(graph, expansion) ||
                             !demandIdsValid(graph, activeDemands);
  if (invalidInputs) {
    result.error = SyncCoverCoverageError::InvalidGraph;
    return result;
  }
  const std::size_t demandWords =
      graph.getDemands().size() / kBitsPerWord +
      (graph.getDemands().size() % kBitsPerWord != 0 ? 1 : 0);
  std::size_t resultRows = 0;
  std::size_t resultWords = 0;
  const bool resultLimitExceeded =
      mechanismCount > limits.maximumResultRows ||
      !checkedSum(mechanismCount, 1, resultRows) ||
      !checkedProduct(resultRows, demandWords, resultWords) ||
      !denseWordsFit(resultWords, 0, limits);
  if (resultLimitExceeded) {
    result.error = SyncCoverCoverageError::LimitExceeded;
    return result;
  }
  result.baseline = SyncCoverDemandSet(graph.getDemands().size());
  result.mechanisms.reserve(mechanismCount);
  for (std::size_t mechanism = 0; mechanism < mechanismCount; ++mechanism) {
    result.mechanisms.emplace_back(graph.getDemands().size());
  }
  const bool baseUnavailable =
      expansion.getError() == SyncCoverExpansionError::BaseLimitExceeded;
  if (baseUnavailable) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
    for (SyncCoverDemandId demand : activeDemands) {
      result.unavailableDemands.push_back(demand);
    }
    return result;
  }

  std::vector<SyncCoverCompletionSupply> supplies = inputSupplies;
  if (!canonicalizeSupplies(graph, supplies, mechanismCount)) {
    result.error = SyncCoverCoverageError::InvalidSupply;
    return result;
  }
  std::map<const SyncCoverExpandedArena *, SupplyIndex> supplyIndices;

  for (SyncCoverDemandId demandId :
       hierarchicalDemandOrder(graph, activeDemands)) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverExpandedArena *arena = expansion.getArena(demand);
    if (!arena) {
      result.unavailableDemands.push_back(demandId);
      continue;
    }
    const DemandContext context = makeDemandContext(graph, demand);
    const std::optional<std::size_t> source =
        expansion.projectEndpoint(graph, *arena, demand.source, 0);
    const std::optional<std::size_t> goal = expansion.projectEndpoint(
        graph, *arena, demand.target, demand.distance);
    if (!context.valid || !source || !goal) {
      result.error = SyncCoverCoverageError::InvalidGraph;
      return result;
    }

    auto [supplyIndex, inserted] = supplyIndices.try_emplace(arena);
    if (inserted) {
      supplyIndex->second =
          buildSupplyIndex(graph, expansion, *arena, supplies);
    }
    const SingletonSeedCollection seedCollection =
        collectSingletonSeeds(graph, demand, demandId, context, expansion,
                              *arena, supplyIndex->second, supplies);
    if (seedCollection.baselineCovers) {
      result.baseline.insert(demandId);
    }
    const std::vector<SingletonSeed> &seeds = seedCollection.seeds;
    std::vector<SyncCoverMechanismId> candidates;
    candidates.reserve(seeds.size());
    for (const SingletonSeed &seed : seeds) {
      candidates.push_back(seed.mechanism);
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    if (candidates.empty()) {
      continue;
    }

    const std::size_t wordsPerNode =
        candidates.size() / kBitsPerWord +
        (candidates.size() % kBitsPerWord != 0 ? 1 : 0);
    std::size_t workspaceWords = 0;
    const bool workspaceLimitExceeded =
        !checkedProduct(arena->getVirtualNodeCount(), wordsPerNode,
                        workspaceWords) ||
        !denseWordsFit(resultWords, workspaceWords, limits);
    if (workspaceLimitExceeded) {
      result.error = SyncCoverCoverageError::LimitExceeded;
      result.baseline = SyncCoverDemandSet();
      result.mechanisms.clear();
      return result;
    }

    SingletonWorkspace workspace(arena->getVirtualNodeCount(),
                                 candidates.size());
    for (const SingletonSeed &seed : seeds) {
      const auto mechanism = std::lower_bound(candidates.begin(),
                                              candidates.end(), seed.mechanism);
      workspace.add(seed.target,
                    static_cast<std::size_t>(mechanism - candidates.begin()));
    }

    while (const std::optional<std::size_t> node = workspace.pop()) {
      for (const SyncCoverExpandedEdge &edge : arena->getOutgoingEdges(*node)) {
        const bool active =
            edge.targetCopy <= demand.distance &&
            virtualNodeAvailable(graph, demand, context, *arena, edge.source,
                                 edge.sourceCopy) &&
            virtualNodeAvailable(graph, demand, context, *arena, edge.target,
                                 edge.targetCopy) &&
            (!edge.graphEdge ||
             edgeGuardsActive(graph, demand, context,
                              graph.getEdges()[*edge.graphEdge],
                              edge.sourceCopy, edge.targetCopy));
        if (active) {
          workspace.unite(*node, edge.target);
        }
      }

      visitSupplyOutgoing(
          supplyIndex->second, *node,
          [&](const IndexedSupply &indexed) -> bool {
            const SyncCoverCompletionSupply &description =
                supplies[indexed.supply];
            if (description.applicability ==
                    SyncCoverSupplyApplicability::DistanceZeroOnly &&
                demand.distance != 0) {
              return true;
            }
            const bool restricted = !description.allowedDemands.empty();
            const bool demandAllowed =
                std::binary_search(description.allowedDemands.begin(),
                                   description.allowedDemands.end(), demandId);
            if (restricted && !demandAllowed) {
              return true;
            }
            const auto mechanism = std::lower_bound(
                candidates.begin(), candidates.end(), indexed.mechanism);
            const bool absent = mechanism == candidates.end() ||
                                *mechanism != indexed.mechanism;
            if (absent) {
              return true;
            }
            const std::size_t local =
                static_cast<std::size_t>(mechanism - candidates.begin());
            if (!workspace.contains(*node, local)) {
              return true;
            }
            std::size_t target = 0;
            if (supplyIsActive(graph, demand, context, description, indexed,
                               target)) {
              workspace.add(target, local);
            }
            return true;
          });
    }

    const std::uint64_t *goalWords = workspace.words(*goal);
    for (std::size_t local = 0; local < candidates.size(); ++local) {
      if ((goalWords[local / kBitsPerWord] &
           (std::uint64_t{1} << (local % kBitsPerWord))) != 0) {
        result.mechanisms[candidates[local]].insert(demandId);
      }
    }
  }
  if (!result.unavailableDemands.empty()) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
  }
  return result;
}

SyncCoverPairCoverageResult mlir::pto::computeSyncCoverPairCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &inputSupplies,
    const std::vector<SyncCoverMechanismPair> &pairs,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageLimits limits) {
  SyncCoverPairCoverageResult result;
  const bool invalidInputs =
      !coverageInputsValid(graph, expansion) ||
      !demandIdsValid(graph, activeDemands) ||
      std::any_of(pairs.begin(), pairs.end(), [&](const auto &pair) {
        return pair.first >= pair.second || pair.second >= mechanismCount;
      });
  if (invalidInputs) {
    result.error = SyncCoverCoverageError::InvalidGraph;
    return result;
  }
  if (mechanismCount > limits.maximumMechanismRows) {
    result.error = SyncCoverCoverageError::LimitExceeded;
    return result;
  }
  const std::size_t demandWords =
      graph.getDemands().size() / kBitsPerWord +
      (graph.getDemands().size() % kBitsPerWord != 0 ? 1 : 0);
  std::size_t resultWords = 0;
  const bool resultLimitExceeded =
      pairs.size() > limits.maximumResultRows ||
      !checkedProduct(pairs.size(), demandWords, resultWords) ||
      !denseWordsFit(resultWords, 0, limits);
  if (resultLimitExceeded) {
    result.error = SyncCoverCoverageError::LimitExceeded;
    return result;
  }
  result.pairs.reserve(pairs.size());
  for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
    result.pairs.emplace_back(graph.getDemands().size());
  }
  const bool baseExpansionUnavailable =
      expansion.getError() == SyncCoverExpansionError::BaseLimitExceeded;
  if (baseExpansionUnavailable) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
    result.unavailableDemands = activeDemands;
    return result;
  }

  std::vector<SyncCoverCompletionSupply> supplies = inputSupplies;
  if (!canonicalizeSupplies(graph, supplies, mechanismCount)) {
    result.error = SyncCoverCoverageError::InvalidSupply;
    return result;
  }
  std::map<const SyncCoverExpandedArena *, SupplyIndex> supplyIndices;
  std::vector<std::vector<std::size_t>> pairsByMechanism(mechanismCount);
  for (std::size_t pair = 0; pair < pairs.size(); ++pair) {
    pairsByMechanism[pairs[pair].first].push_back(pair);
    pairsByMechanism[pairs[pair].second].push_back(pair);
  }

  constexpr std::size_t kAbsent = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> globalToLocal(pairs.size(), kAbsent);
  for (SyncCoverDemandId demandId :
       hierarchicalDemandOrder(graph, activeDemands)) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverExpandedArena *arena = expansion.getArena(demand);
    if (!arena) {
      result.unavailableDemands.push_back(demandId);
      continue;
    }
    const DemandContext context = makeDemandContext(graph, demand);
    const std::optional<std::size_t> goal = expansion.projectEndpoint(
        graph, *arena, demand.target, demand.distance);
    if (!context.valid || !goal) {
      result.error = SyncCoverCoverageError::InvalidGraph;
      return result;
    }
    auto [supplyIndex, inserted] = supplyIndices.try_emplace(arena);
    if (inserted) {
      supplyIndex->second =
          buildSupplyIndex(graph, expansion, *arena, supplies);
    }
    const SingletonSeedCollection seeds =
        collectSingletonSeeds(graph, demand, demandId, context, expansion,
                              *arena, supplyIndex->second, supplies);
    if (seeds.baselineCovers) {
      continue;
    }

    std::vector<std::size_t> candidates;
    for (const SingletonSeed &seed : seeds.seeds) {
      const auto &memberPairs = pairsByMechanism[seed.mechanism];
      candidates.insert(candidates.end(), memberPairs.begin(),
                        memberPairs.end());
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    if (candidates.empty()) {
      continue;
    }
    const std::size_t wordsPerNode =
        candidates.size() / kBitsPerWord +
        (candidates.size() % kBitsPerWord != 0 ? 1 : 0);
    std::size_t workspaceWords = 0;
    const bool workspaceLimitExceeded =
        !checkedProduct(arena->getVirtualNodeCount(), wordsPerNode,
                        workspaceWords) ||
        !denseWordsFit(resultWords, workspaceWords, limits);
    if (workspaceLimitExceeded) {
      result.error = SyncCoverCoverageError::LimitExceeded;
      result.pairs.clear();
      return result;
    }
    for (std::size_t local = 0; local < candidates.size(); ++local) {
      globalToLocal[candidates[local]] = local;
    }

    PairWorkspace workspace(arena->getVirtualNodeCount(), candidates.size());
    for (const SingletonSeed &seed : seeds.seeds) {
      for (std::size_t pair : pairsByMechanism[seed.mechanism]) {
        const std::size_t local = globalToLocal[pair];
        if (local != kAbsent) {
          workspace.add(seed.target, local);
        }
      }
    }
    while (const std::optional<std::size_t> node = workspace.pop()) {
      for (const SyncCoverExpandedEdge &edge : arena->getOutgoingEdges(*node)) {
        const bool active =
            edge.targetCopy <= demand.distance &&
            virtualNodeAvailable(graph, demand, context, *arena, edge.source,
                                 edge.sourceCopy) &&
            virtualNodeAvailable(graph, demand, context, *arena, edge.target,
                                 edge.targetCopy) &&
            (!edge.graphEdge ||
             edgeGuardsActive(graph, demand, context,
                              graph.getEdges()[*edge.graphEdge],
                              edge.sourceCopy, edge.targetCopy));
        if (active) {
          workspace.unite(*node, edge.target);
        }
      }

      visitSupplyOutgoing(
          supplyIndex->second, *node,
          [&](const IndexedSupply &indexed) -> bool {
            const SyncCoverCompletionSupply &description =
                supplies[indexed.supply];
            if (description.applicability ==
                    SyncCoverSupplyApplicability::DistanceZeroOnly &&
                demand.distance != 0) {
              return true;
            }
            const bool constrained = !description.allowedDemands.empty();
            const bool demandAllowed =
                !constrained ||
                std::binary_search(description.allowedDemands.begin(),
                                   description.allowedDemands.end(), demandId);
            if (!demandAllowed) {
              return true;
            }
            std::size_t target = 0;
            if (!supplyIsActive(graph, demand, context, description, indexed,
                                target)) {
              return true;
            }
            for (std::size_t pair : pairsByMechanism[indexed.mechanism]) {
              const std::size_t local = globalToLocal[pair];
              if (local != kAbsent && workspace.contains(*node, local)) {
                workspace.add(target, local);
              }
            }
            return true;
          });
    }

    const std::uint64_t *goalWords = workspace.words(*goal);
    for (std::size_t local = 0; local < candidates.size(); ++local) {
      if ((goalWords[local / kBitsPerWord] &
           (std::uint64_t{1} << (local % kBitsPerWord))) != 0) {
        result.pairs[candidates[local]].insert(demandId);
      }
      globalToLocal[candidates[local]] = kAbsent;
    }
  }
  if (!result.unavailableDemands.empty()) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
  }
  return result;
}
