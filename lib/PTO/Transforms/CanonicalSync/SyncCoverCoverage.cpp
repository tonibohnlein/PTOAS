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
#include <cassert>
#include <deque>
#include <functional>
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
  const bool alreadySorted = values.size() < 2;
  if (alreadySorted) {
    return true;
  }
  if (!budget->consume(values.size())) {
    return false;
  }
  std::vector<T> source = std::move(values);
  std::vector<T> target;
  target.reserve(source.size());
  for (std::size_t width = 1; width < source.size();) {
    target.clear();
    for (std::size_t begin = 0; begin < source.size();) {
      const std::size_t middle = std::min(source.size(), begin + width);
      const std::size_t end = std::min(source.size(), middle + width);
      std::size_t left = begin;
      std::size_t right = middle;
      while (left < middle && right < end) {
        if (!budget->consume(2)) {
          return false;
        }
        const bool takeRight = compare(source[right], source[left]);
        if (budget->exhausted) {
          return false;
        }
        if (takeRight) {
          target.push_back(std::move(source[right++]));
        } else {
          target.push_back(std::move(source[left++]));
        }
      }
      while (left < middle) {
        if (!budget->consume()) {
          return false;
        }
        target.push_back(std::move(source[left++]));
      }
      while (right < end) {
        if (!budget->consume()) {
          return false;
        }
        target.push_back(std::move(source[right++]));
      }
      begin = end;
    }
    source.swap(target);
    const bool finalPass = width > source.size() / 2;
    if (finalPass) {
      break;
    }
    width *= 2;
  }
  values = std::move(source);
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

namespace {

struct FlatNodePositions {
  SyncCoverTimelinePosition before = 0;
  SyncCoverTimelinePosition after = 0;
};

bool sameAnchor(const SyncCoverAnchor &left, const SyncCoverAnchor &right) {
  return std::tie(left.kind, left.node, left.scope, left.position) ==
         std::tie(right.kind, right.node, right.scope, right.position);
}

bool normalizedGuardImplies(const SyncCoverGuard &condition,
                            const SyncCoverGuard &required) {
  return std::includes(condition.literals.begin(), condition.literals.end(),
                       required.literals.begin(), required.literals.end());
}

bool normalizedGuardsCompatible(const SyncCoverGuard &first,
                                const SyncCoverGuard &second) {
  std::size_t firstIndex = 0;
  std::size_t secondIndex = 0;
  while (firstIndex < first.literals.size() &&
         secondIndex < second.literals.size()) {
    const SyncCoverGuardLiteral &firstLiteral = first.literals[firstIndex];
    const SyncCoverGuardLiteral &secondLiteral = second.literals[secondIndex];
    if (firstLiteral.control < secondLiteral.control) {
      ++firstIndex;
      continue;
    }
    if (secondLiteral.control < firstLiteral.control) {
      ++secondIndex;
      continue;
    }
    if (firstLiteral.alternative != secondLiteral.alternative) {
      return false;
    }
    ++firstIndex;
    ++secondIndex;
  }
  return true;
}

std::optional<SyncCoverGuard>
mergeNormalizedGuards(const SyncCoverGuard &first, const SyncCoverGuard &second,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t mergeWork = 0;
  if (!checkedSum(first.literals.size(), second.literals.size(), mergeWork) ||
      !checkedSum(mergeWork, 1, mergeWork) ||
      !consumeWork(workBudget, mergeWork)) {
    return std::nullopt;
  }
  SyncCoverGuard result;
  result.literals.reserve(first.literals.size() + second.literals.size());
  std::size_t firstIndex = 0;
  std::size_t secondIndex = 0;
  while (firstIndex < first.literals.size() ||
         secondIndex < second.literals.size()) {
    if (secondIndex == second.literals.size() ||
        (firstIndex < first.literals.size() &&
         first.literals[firstIndex].control <
             second.literals[secondIndex].control)) {
      result.literals.push_back(first.literals[firstIndex++]);
      continue;
    }
    if (firstIndex == first.literals.size() ||
        second.literals[secondIndex].control <
            first.literals[firstIndex].control) {
      result.literals.push_back(second.literals[secondIndex++]);
      continue;
    }
    const SyncCoverGuardLiteral &firstLiteral = first.literals[firstIndex++];
    const SyncCoverGuardLiteral &secondLiteral = second.literals[secondIndex++];
    if (firstLiteral.alternative != secondLiteral.alternative) {
      return std::nullopt;
    }
    result.literals.push_back(firstLiteral);
  }
  return result;
}

bool flatPointGuardValid(const SyncCoverGraph &graph,
                         const SyncCoverGuard &guard,
                         SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t validationWork = 0;
  if (!checkedSum(guard.literals.size(), 1, validationWork) ||
      !consumeWork(workBudget, validationWork)) {
    return false;
  }
  for (std::size_t index = 0; index < guard.literals.size(); ++index) {
    const SyncCoverGuardLiteral &literal = guard.literals[index];
    if (literal.control >= graph.getControls().size() ||
        literal.alternative >=
            graph.getControls()[literal.control].alternatives) {
      return false;
    }
    if (index != 0) {
      const SyncCoverGuardLiteral &previous = guard.literals[index - 1];
      if (!(previous < literal) || previous.control == literal.control) {
        return false;
      }
    }
  }
  return true;
}

std::optional<SyncCoverGuard>
effectivePointGuard(const SyncCoverGraph &graph, const SyncCoverCutPoint &point,
                    SyncCoverCoverageWorkBudget *workBudget) {
  if (point.anchor.kind == SyncCoverAnchorKind::BeforeNode ||
      point.anchor.kind == SyncCoverAnchorKind::AfterNode) {
    if (point.anchor.node >= graph.getNodes().size()) {
      return std::nullopt;
    }
    const SyncCoverGuard &nodeGuard = graph.getNodes()[point.anchor.node].guard;
    return mergeNormalizedGuards(point.guard, nodeGuard, workBudget);
  }
  return point.guard;
}

bool flatCutPointValid(const SyncCoverGraph &graph,
                       const SyncCoverCutPoint &point,
                       SyncCoverCutPointKind expected,
                       SyncCoverCoverageWorkBudget *workBudget) {
  const bool invalid =
      point.kind != expected ||
      !flatPointGuardValid(graph, point.guard, workBudget) ||
      (point.anchor.kind != SyncCoverAnchorKind::BeforeNode &&
       point.anchor.kind != SyncCoverAnchorKind::AfterNode) ||
      point.anchor.node >= graph.getNodes().size() ||
      graph.getNodes()[point.anchor.node].resource != point.resource;
  if (invalid) {
    return false;
  }
  return resolveSyncCoverAnchor(graph, point.anchor).has_value();
}

std::optional<SyncCoverCompletionOrigin>
makeFlatCompletionOrigin(const SyncCoverGraph &graph,
                         const SyncCoverDirectCut &cut,
                         SyncCoverCoverageWorkBudget *workBudget) {
  const bool validRequirements =
      cut.suppliedRequirements != 0 &&
      (cut.suppliedRequirements & ~kAllSyncCoverOrderingRequirements) == 0;
  if (!validRequirements) {
    return std::nullopt;
  }

  const bool event = cut.kind == SyncCoverDirectCutKind::Event;
  const bool barrier = cut.kind == SyncCoverDirectCutKind::PipeBarrier;
  const bool invalidPoints =
      (!event && !barrier) ||
      !flatCutPointValid(graph, cut.source,
                         event ? SyncCoverCutPointKind::EventSet
                               : SyncCoverCutPointKind::PipeBarrier,
                         workBudget) ||
      !flatCutPointValid(graph, cut.target,
                         event ? SyncCoverCutPointKind::EventWait
                               : SyncCoverCutPointKind::PipeBarrier,
                         workBudget);
  if (invalidPoints) {
    return std::nullopt;
  }
  const bool invalidResources =
      (event && cut.source.resource == cut.target.resource) ||
      (barrier && (cut.source.resource != cut.target.resource ||
                   !sameAnchor(cut.source.anchor, cut.target.anchor)));
  if (invalidResources) {
    return std::nullopt;
  }

  const std::optional<SyncCoverTimelinePosition> sourceBoundary =
      resolveSyncCoverAnchor(graph, cut.source.anchor);
  const std::optional<SyncCoverTimelinePosition> targetBoundary =
      resolveSyncCoverAnchor(graph, cut.target.anchor);
  const std::optional<SyncCoverGuard> sourceGuard =
      effectivePointGuard(graph, cut.source, workBudget);
  const std::optional<SyncCoverGuard> targetGuard =
      effectivePointGuard(graph, cut.target, workBudget);
  if (!sourceBoundary || !targetBoundary || !sourceGuard || !targetGuard ||
      *sourceBoundary > *targetBoundary) {
    return std::nullopt;
  }

  SyncCoverCompletionOrigin result;
  result.mechanism = cut.mechanism;
  result.kind = cut.kind;
  result.sourceResource = cut.source.resource;
  result.targetResource = cut.target.resource;
  result.sourceBoundary = *sourceBoundary;
  result.targetBoundary = *targetBoundary;
  result.sourceGuard = *sourceGuard;
  result.targetGuard = *targetGuard;
  result.suppliedRequirements = cut.suppliedRequirements;
  result.sourcePrefixCompletion = cut.sourcePrefixCompletion;
  result.sourceNode = cut.source.anchor.node;
  result.sourceOrdinal = cut.source.ordinal;
  result.targetOrdinal = cut.target.ordinal;
  return result;
}

bool flatDemandCondition(const SyncCoverGraph &graph,
                         const SyncCoverDemand &demand,
                         SyncCoverGuard &condition,
                         SyncCoverCoverageWorkBudget *workBudget) {
  const SyncCoverGuard &sourceGuard = graph.getNodes()[demand.source].guard;
  const SyncCoverGuard &targetGuard = graph.getNodes()[demand.target].guard;
  std::optional<SyncCoverGuard> merged =
      mergeNormalizedGuards(demand.sourceGuard, demand.targetGuard, workBudget);
  if (merged) {
    merged = mergeNormalizedGuards(*merged, sourceGuard, workBudget);
  }
  if (merged) {
    merged = mergeNormalizedGuards(*merged, targetGuard, workBudget);
  }
  if (!merged) {
    return false;
  }
  condition = std::move(*merged);
  return true;
}

bool flatOriginActive(const SyncCoverGuard &condition,
                      const SyncCoverCompletionOrigin &origin,
                      SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t comparisonWork = 0;
  return checkedSum(condition.literals.size(), condition.literals.size(),
                    comparisonWork) &&
         checkedSum(comparisonWork, origin.sourceGuard.literals.size(),
                    comparisonWork) &&
         checkedSum(comparisonWork, origin.targetGuard.literals.size(),
                    comparisonWork) &&
         checkedSum(comparisonWork, 2, comparisonWork) &&
         consumeWork(workBudget, comparisonWork) &&
         normalizedGuardImplies(condition, origin.sourceGuard) &&
         normalizedGuardImplies(condition, origin.targetGuard);
}

bool flatNodeActive(const SyncCoverGraph &graph,
                    const SyncCoverGuard &condition, SyncCoverNodeId node,
                    SyncCoverCoverageWorkBudget *workBudget) {
  if (node >= graph.getNodes().size()) {
    return false;
  }
  std::size_t comparisonWork = 0;
  return checkedSum(condition.literals.size(),
                    graph.getNodes()[node].guard.literals.size(),
                    comparisonWork) &&
         checkedSum(comparisonWork, 1, comparisonWork) &&
         consumeWork(workBudget, comparisonWork) &&
         normalizedGuardImplies(condition, graph.getNodes()[node].guard);
}

bool flatNodeInSourcePrefix(const SyncCoverGraph &graph,
                            const std::vector<FlatNodePositions> &positions,
                            const SyncCoverCompletionOrigin &origin,
                            const SyncCoverGuard &condition,
                            SyncCoverNodeId node,
                            SyncCoverCoverageWorkBudget *workBudget) {
  return node < graph.getNodes().size() &&
         graph.getNodes()[node].resource == origin.sourceResource &&
         positions[node].after <= origin.sourceBoundary &&
         flatNodeActive(graph, condition, node, workBudget);
}

bool flatNodeInTargetSuffix(const SyncCoverGraph &graph,
                            const std::vector<FlatNodePositions> &positions,
                            const SyncCoverCompletionOrigin &origin,
                            const SyncCoverGuard &condition,
                            SyncCoverNodeId node,
                            SyncCoverCoverageWorkBudget *workBudget) {
  return node < graph.getNodes().size() &&
         graph.getNodes()[node].resource == origin.targetResource &&
         positions[node].before >= origin.targetBoundary &&
         flatNodeActive(graph, condition, node, workBudget);
}

bool flatEdgeActive(const SyncCoverGraph &graph, const SyncCoverEdge &edge,
                    const SyncCoverGuard &condition,
                    SyncCoverCoverageWorkBudget *workBudget) {
  std::size_t comparisonWork = 0;
  const bool workAvailable =
      checkedSum(condition.literals.size(), condition.literals.size(),
                 comparisonWork) &&
      checkedSum(comparisonWork, edge.sourceGuard.literals.size(),
                 comparisonWork) &&
      checkedSum(comparisonWork, edge.targetGuard.literals.size(),
                 comparisonWork) &&
      checkedSum(comparisonWork, 2, comparisonWork) &&
      consumeWork(workBudget, comparisonWork);
  const bool invalid =
      edge.distance != 0 || edge.source >= graph.getNodes().size() ||
      edge.target >= graph.getNodes().size() || !workAvailable ||
      !normalizedGuardImplies(condition, edge.sourceGuard) ||
      !normalizedGuardImplies(condition, edge.targetGuard) ||
      !flatNodeActive(graph, condition, edge.source, workBudget) ||
      !flatNodeActive(graph, condition, edge.target, workBudget);
  if (invalid) {
    return false;
  }
  return edge.kind == SyncCoverEdgeKind::CertifiedCompletionFrontier ||
         edge.kind == SyncCoverEdgeKind::CompletionPreservingIssueOrder ||
         edge.kind == SyncCoverEdgeKind::NonCompletionPreservingIssueOrder;
}

bool originLess(const SyncCoverCompletionOrigin &left,
                const SyncCoverCompletionOrigin &right) {
  return std::tie(left.mechanism, left.kind, left.sourceResource,
                  left.targetResource, left.sourceBoundary, left.targetBoundary,
                  left.sourceGuard.literals, left.targetGuard.literals,
                  left.suppliedRequirements, left.sourcePrefixCompletion,
                  left.sourceNode, left.sourceOrdinal, left.targetOrdinal) <
         std::tie(right.mechanism, right.kind, right.sourceResource,
                  right.targetResource, right.sourceBoundary,
                  right.targetBoundary, right.sourceGuard.literals,
                  right.targetGuard.literals, right.suppliedRequirements,
                  right.sourcePrefixCompletion, right.sourceNode,
                  right.sourceOrdinal, right.targetOrdinal);
}

} // namespace

SyncCoverFlatWorldResult mlir::pto::computeSyncCoverFlatExactWorld(
    const SyncCoverGraph &graph, const std::vector<SyncCoverDirectCut> &cuts,
    const SyncCoverExactWorld &world, SyncCoverFlatWorldLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverFlatWorldResult result;
  result.covered = SyncCoverDemandSet(graph.getDemands().size());
  const auto fail = [&](SyncCoverFlatWorldError error,
                        std::optional<std::size_t> index = std::nullopt) {
    result.error = error;
    result.invalidIndex = index;
    result.completionOrigins.clear();
    return result;
  };
  // freezeStructure() validates before making the graph immutable. Repeating
  // full validation here would duplicate guard normalization and sorting
  // outside this exact world's work budget.
  const bool invalidGraph = !graph.isStructureFrozen();
  if (invalidGraph) {
    return fail(SyncCoverFlatWorldError::InvalidGraph);
  }
  const bool limitsExceeded =
      cuts.size() > limits.maximumCuts ||
      world.enabledMechanisms.size() > limits.maximumEnabledMechanisms;
  if (limitsExceeded) {
    return fail(SyncCoverFlatWorldError::LimitExceeded);
  }
  if (!std::is_sorted(world.enabledMechanisms.begin(),
                      world.enabledMechanisms.end()) ||
      std::adjacent_find(world.enabledMechanisms.begin(),
                         world.enabledMechanisms.end()) !=
          world.enabledMechanisms.end()) {
    return fail(SyncCoverFlatWorldError::InvalidWorld);
  }
  const bool unsupportedStructure =
      graph.getScopes().size() != 1 ||
      std::any_of(graph.getDemands().begin(), graph.getDemands().end(),
                  [](const SyncCoverDemand &demand) {
                    return demand.scope != 0 || demand.distance != 0;
                  }) ||
      std::any_of(graph.getNodes().begin(), graph.getNodes().end(),
                  [](const SyncCoverNode &node) { return node.scope != 0; });
  if (unsupportedStructure) {
    return fail(SyncCoverFlatWorldError::UnsupportedStructure);
  }

  std::vector<FlatNodePositions> positions(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    if (!consumeWork(workBudget, 2)) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
    const auto before = resolveSyncCoverAnchor(
        graph, {SyncCoverAnchorKind::BeforeNode, node.id, 0, 0});
    const auto after = resolveSyncCoverAnchor(
        graph, {SyncCoverAnchorKind::AfterNode, node.id, 0, 0});
    if (!before || !after || *before > *after) {
      return fail(SyncCoverFlatWorldError::InvalidGraph, node.id);
    }
    positions[node.id] = {*before, *after};
  }

  std::vector<SyncCoverMechanismId> describedMechanisms;
  describedMechanisms.reserve(cuts.size());
  std::size_t totalGuardLiterals = 0;
  for (std::size_t index = 0; index < cuts.size(); ++index) {
    const std::size_t sourceLiterals = cuts[index].source.guard.literals.size();
    const std::size_t targetLiterals = cuts[index].target.guard.literals.size();
    std::size_t cutLiterals = 0;
    if (sourceLiterals > limits.maximumGuardLiteralsPerPoint ||
        targetLiterals > limits.maximumGuardLiteralsPerPoint ||
        !checkedSum(sourceLiterals, targetLiterals, cutLiterals) ||
        !checkedSum(totalGuardLiterals, cutLiterals, totalGuardLiterals) ||
        totalGuardLiterals > limits.maximumTotalGuardLiterals) {
      return fail(SyncCoverFlatWorldError::LimitExceeded, index);
    }
    std::size_t cutWork = 0;
    const bool cutWorkUnavailable = !checkedSum(cutLiterals, 1, cutWork) ||
                                    !consumeWork(workBudget, cutWork);
    if (cutWorkUnavailable) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
    const std::optional<SyncCoverCompletionOrigin> origin =
        makeFlatCompletionOrigin(graph, cuts[index], workBudget);
    if (!origin) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverFlatWorldError::WorkLimitExceeded
                      : SyncCoverFlatWorldError::InvalidCut,
                  index);
    }
    describedMechanisms.push_back(cuts[index].mechanism);
    if (std::binary_search(world.enabledMechanisms.begin(),
                           world.enabledMechanisms.end(),
                           cuts[index].mechanism)) {
      const bool originLimitReached =
          result.completionOrigins.size() == limits.maximumCompletionOrigins;
      if (originLimitReached) {
        return fail(SyncCoverFlatWorldError::LimitExceeded, index);
      }
      result.completionOrigins.push_back(*origin);
    }
  }
  if (!meteredStableSort(describedMechanisms, std::less<>(), workBudget)) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }
  describedMechanisms.erase(
      std::unique(describedMechanisms.begin(), describedMechanisms.end()),
      describedMechanisms.end());
  std::size_t inclusionWork = 0;
  if (!checkedSum(describedMechanisms.size(), world.enabledMechanisms.size(),
                  inclusionWork) ||
      !consumeWork(workBudget, inclusionWork)) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }
  if (!std::includes(describedMechanisms.begin(), describedMechanisms.end(),
                     world.enabledMechanisms.begin(),
                     world.enabledMechanisms.end())) {
    return fail(SyncCoverFlatWorldError::InvalidWorld);
  }
  if (!meteredStableSort(result.completionOrigins, originLess, workBudget)) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }

  constexpr std::size_t maskCount =
      static_cast<std::size_t>(kAllSyncCoverOrderingRequirements) + 1;
  static_assert(maskCount <= 16,
                "flat-world capability states must fit one uint16_t");
  const bool stateOverflow =
      !graph.getNodes().empty() &&
      maskCount >
          std::numeric_limits<std::size_t>::max() / graph.getNodes().size();
  const std::size_t maximumStates =
      stateOverflow ? std::numeric_limits<std::size_t>::max()
                    : graph.getNodes().size() * maskCount;
  if (stateOverflow || maximumStates > limits.maximumStates) {
    return fail(SyncCoverFlatWorldError::LimitExceeded);
  }

  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    if (!consumeWork(workBudget)) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    SyncCoverGuard condition;
    if (!flatDemandCondition(graph, demand, condition, workBudget)) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverFlatWorldError::WorkLimitExceeded
                      : SyncCoverFlatWorldError::InvalidGraph,
                  demandId);
    }

    std::vector<std::uint16_t> reached(graph.getNodes().size(), 0);
    std::deque<std::pair<SyncCoverNodeId, SyncCoverOrderingRequirementMask>>
        ready;
    const auto addState = [&](SyncCoverNodeId node,
                              SyncCoverOrderingRequirementMask mask) {
      const bool invalidState =
          node >= reached.size() || mask == 0 ||
          (mask & ~kAllSyncCoverOrderingRequirements) != 0;
      if (invalidState) {
        return false;
      }
      const std::uint16_t bit =
          static_cast<std::uint16_t>(std::uint16_t{1} << mask);
      const bool alreadyReached = (reached[node] & bit) != 0;
      if (alreadyReached) {
        return false;
      }
      reached[node] |= bit;
      ready.emplace_back(node, mask);
      return true;
    };
    const auto applyOrigin = [&](const SyncCoverCompletionOrigin &origin,
                                 SyncCoverOrderingRequirementMask incoming,
                                 bool direct) {
      if (!flatOriginActive(condition, origin, workBudget)) {
        if (workBudget && workBudget->exhausted) {
          return false;
        }
        return true;
      }
      const SyncCoverOrderingRequirementMask transferred =
          direct ? origin.suppliedRequirements
                 : static_cast<SyncCoverOrderingRequirementMask>(
                       incoming & origin.suppliedRequirements);
      if (transferred == 0) {
        return true;
      }
      for (const SyncCoverNode &node : graph.getNodes()) {
        if (!consumeWork(workBudget)) {
          return false;
        }
        if (flatNodeInTargetSuffix(graph, positions, origin, condition, node.id,
                                   workBudget)) {
          addState(node.id, transferred);
        } else if (workBudget && workBudget->exhausted) {
          return false;
        }
      }
      return true;
    };
    const auto originAcceptsSource =
        [&](const SyncCoverCompletionOrigin &origin, SyncCoverNodeId node) {
          return node == origin.sourceNode ||
                 (origin.sourcePrefixCompletion &&
                  flatNodeInSourcePrefix(graph, positions, origin, condition,
                                         node, workBudget));
        };

    for (const SyncCoverCompletionOrigin &origin : result.completionOrigins) {
      if (!consumeWork(workBudget)) {
        return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
      }
      const bool acceptsSource = originAcceptsSource(origin, demand.source);
      if ((workBudget && workBudget->exhausted) ||
          (acceptsSource && !applyOrigin(origin, 0, true))) {
        return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
      }
    }

    while (!ready.empty()) {
      if (!consumeWork(workBudget)) {
        return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
      }
      const auto [node, mask] = ready.front();
      ready.pop_front();
      for (const SyncCoverEdge &edge : graph.getEdges()) {
        if (!consumeWork(workBudget)) {
          return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
        }
        const bool active = edge.source == node &&
                            flatEdgeActive(graph, edge, condition, workBudget);
        if (workBudget && workBudget->exhausted) {
          return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
        }
        if (active) {
          addState(edge.target, mask);
        }
      }
      for (const SyncCoverCompletionOrigin &origin : result.completionOrigins) {
        if (!consumeWork(workBudget)) {
          return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
        }
        const bool acceptsSource = originAcceptsSource(origin, node);
        const bool activationFailed =
            acceptsSource && !applyOrigin(origin, mask, false);
        if (activationFailed || (workBudget && workBudget->exhausted)) {
          return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
        }
      }
    }

    const SyncCoverOrderingRequirementMask required =
        demand.orderingRequirements;
    const std::uint16_t targetMasks = reached[demand.target];
    bool covered = false;
    for (SyncCoverOrderingRequirementMask mask = 1;
         mask <= kAllSyncCoverOrderingRequirements; ++mask) {
      const std::uint16_t bit =
          static_cast<std::uint16_t>(std::uint16_t{1} << mask);
      covered |= (targetMasks & bit) != 0 && (mask & required) == required;
    }
    if (covered) {
      result.covered.insert(demandId);
    }
  }
  return result;
}

namespace {

enum class RegionCutActionKind : std::uint8_t {
  Set,
  Wait,
  Barrier,
};

struct RegionCutAction {
  RegionCutActionKind kind = RegionCutActionKind::Set;
  std::size_t cut = 0;
  std::size_t ordinal = 0;
};

struct RegionWorldState {
  std::vector<std::uint16_t> completed;
  std::vector<std::uint16_t> tokens;
  std::vector<std::uint8_t> fixedSourcesReached;
  bool sourceIssued = false;
  bool targetCovered = false;
};

std::uint16_t capabilityClosure(SyncCoverOrderingRequirementMask mask) {
  std::uint16_t result = 0;
  for (SyncCoverOrderingRequirementMask subset = mask; subset != 0;
       subset =
           static_cast<SyncCoverOrderingRequirementMask>((subset - 1) & mask)) {
    result |= static_cast<std::uint16_t>(std::uint16_t{1} << subset);
  }
  return result;
}

std::uint16_t
transferCapabilities(std::uint16_t incoming,
                     SyncCoverOrderingRequirementMask suppliedRequirements) {
  std::uint16_t result = 0;
  for (SyncCoverOrderingRequirementMask mask = 1;
       mask <= kAllSyncCoverOrderingRequirements; ++mask) {
    const std::uint16_t bit =
        static_cast<std::uint16_t>(std::uint16_t{1} << mask);
    const bool capabilityAbsent = (incoming & bit) == 0;
    if (capabilityAbsent) {
      continue;
    }
    const SyncCoverOrderingRequirementMask transferred =
        static_cast<SyncCoverOrderingRequirementMask>(mask &
                                                      suppliedRequirements);
    result |= capabilityClosure(transferred);
  }
  return result;
}

bool regionActionLess(const RegionCutAction &left,
                      const RegionCutAction &right) {
  return std::tie(left.ordinal, left.kind, left.cut) <
         std::tie(right.ordinal, right.kind, right.cut);
}

class RegionWorldEvaluator {
public:
  RegionWorldEvaluator(
      const SyncCoverGraph &graph,
      const std::vector<SyncCoverCompletionOrigin> &origins,
      const std::vector<SyncCoverExactWorld> &worlds,
      const std::vector<std::uint32_t> &resources,
      const std::vector<std::vector<RegionCutAction>> &beforeActions,
      const std::vector<std::vector<RegionCutAction>> &afterActions,
      const std::vector<const SyncCoverEdge *> &fixedSupplies,
      const std::vector<std::vector<std::size_t>> &fixedIncoming,
      const std::vector<std::vector<std::size_t>> &fixedOutgoing,
      const SyncCoverDemand &demand, SyncCoverGuard demandCondition,
      SyncCoverRegionWorldLimits limits, std::size_t baseLiveStateWords,
      SyncCoverRegionWorldStatistics &statistics,
      SyncCoverCoverageWorkBudget *workBudget)
      : graph_(graph), origins_(origins), worlds_(worlds),
        resources_(resources), beforeActions_(beforeActions),
        afterActions_(afterActions), fixedSupplies_(fixedSupplies),
        fixedIncoming_(fixedIncoming), fixedOutgoing_(fixedOutgoing),
        demand_(demand), demandCondition_(std::move(demandCondition)),
        limits_(limits), liveStateWords_(baseLiveStateWords),
        statistics_(statistics), workBudget_(workBudget) {}

  bool run(std::vector<RegionWorldState> &states) {
    return evaluateRegion(0, states, demandCondition_, true);
  }

  SyncCoverFlatWorldError getError() const { return error_; }
  std::optional<std::size_t> getInvalidIndex() const { return invalidIndex_; }

private:
  bool charge(std::size_t amount = 1) {
    if (statistics_.regionsEvaluated > limits_.maximumRegionEvaluations ||
        !consumeWork(workBudget_, amount)) {
      error_ = workBudget_ && workBudget_->exhausted
                   ? SyncCoverFlatWorldError::WorkLimitExceeded
                   : SyncCoverFlatWorldError::LimitExceeded;
      return false;
    }
    return true;
  }

  bool chargeStateTraversal(const std::vector<RegionWorldState> &states) {
    const std::optional<std::size_t> words = getStateWords(states);
    if (!words) {
      error_ = SyncCoverFlatWorldError::LimitExceeded;
      return false;
    }
    return charge(*words);
  }

  std::optional<std::size_t>
  getStateWords(const std::vector<RegionWorldState> &states) const {
    if (states.empty()) {
      return 0;
    }
    std::size_t wordsPerState = 0;
    std::size_t totalWords = 0;
    if (!checkedSum(states.front().completed.size(),
                    states.front().tokens.size(), wordsPerState) ||
        !checkedSum(wordsPerState, states.front().fixedSourcesReached.size(),
                    wordsPerState) ||
        !checkedSum(wordsPerState, 2, wordsPerState) ||
        !checkedProduct(wordsPerState, states.size(), totalWords)) {
      return std::nullopt;
    }
    return totalWords;
  }

  bool reserveStateClone(const std::vector<RegionWorldState> &states,
                         std::size_t &reservedWords) {
    const std::optional<std::size_t> words = getStateWords(states);
    std::size_t nextLiveWords = 0;
    if (!words || !checkedSum(liveStateWords_, *words, nextLiveWords) ||
        nextLiveWords > limits_.maximumStateWords || !charge(*words)) {
      if (error_ == SyncCoverFlatWorldError::None) {
        error_ = workBudget_ && workBudget_->exhausted
                     ? SyncCoverFlatWorldError::WorkLimitExceeded
                     : SyncCoverFlatWorldError::LimitExceeded;
      }
      return false;
    }
    reservedWords = *words;
    liveStateWords_ = nextLiveWords;
    statistics_.maximumLiveStateWords =
        std::max(statistics_.maximumLiveStateWords, liveStateWords_);
    return true;
  }

  void releaseStateClone(std::size_t reservedWords) {
    assert(liveStateWords_ >= reservedWords &&
           "state-clone accounting underflow");
    liveStateWords_ -= reservedWords;
  }

  std::optional<std::size_t> resourceIndex(std::uint32_t resource) const {
    const auto position =
        std::lower_bound(resources_.begin(), resources_.end(), resource);
    const bool resourceAbsent =
        position == resources_.end() || *position != resource;
    if (resourceAbsent) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(position - resources_.begin());
  }

  bool worldEnables(std::size_t world, SyncCoverMechanismId mechanism) {
    std::size_t searchWork = 1;
    for (std::size_t remaining = worlds_[world].enabledMechanisms.size();
         remaining > 1; remaining = (remaining + 1) / 2) {
      ++searchWork;
    }
    if (!charge(searchWork)) {
      return false;
    }
    return std::binary_search(worlds_[world].enabledMechanisms.begin(),
                              worlds_[world].enabledMechanisms.end(),
                              mechanism);
  }

  bool guardImplies(const SyncCoverGuard &condition,
                    const SyncCoverGuard &required, bool &result) {
    std::size_t comparisonWork = 0;
    if (!checkedSum(condition.literals.size(), required.literals.size(),
                    comparisonWork) ||
        !checkedSum(comparisonWork, 1, comparisonWork) ||
        !charge(comparisonWork)) {
      return false;
    }
    result = normalizedGuardImplies(condition, required);
    return true;
  }

  bool guardsCompatible(const SyncCoverGuard &first,
                        const SyncCoverGuard &second, bool &result) {
    std::size_t comparisonWork = 0;
    if (!checkedSum(first.literals.size(), second.literals.size(),
                    comparisonWork) ||
        !checkedSum(comparisonWork, 1, comparisonWork) ||
        !charge(comparisonWork)) {
      return false;
    }
    result = normalizedGuardsCompatible(first, second);
    return true;
  }

  bool actionGuardActive(const SyncCoverGuard &pathCondition,
                         const RegionCutAction &action, bool &active) {
    const SyncCoverCompletionOrigin &origin = origins_[action.cut];
    const SyncCoverGuard &guard = action.kind == RegionCutActionKind::Wait
                                      ? origin.targetGuard
                                      : origin.sourceGuard;
    return guardImplies(pathCondition, guard, active);
  }

  bool applyAction(const RegionCutAction &action,
                   std::vector<RegionWorldState> &states,
                   const SyncCoverGuard &pathCondition) {
    bool active = false;
    if (!actionGuardActive(pathCondition, action, active)) {
      return false;
    }
    if (!active) {
      return true;
    }
    const SyncCoverCompletionOrigin &origin = origins_[action.cut];
    const std::optional<std::size_t> sourceResource =
        resourceIndex(origin.sourceResource);
    const std::optional<std::size_t> targetResource =
        resourceIndex(origin.targetResource);
    const bool invalidResources = !sourceResource || !targetResource;
    if (invalidResources) {
      error_ = SyncCoverFlatWorldError::InvalidCut;
      invalidIndex_ = action.cut;
      return false;
    }

    for (std::size_t world = 0; world < states.size(); ++world) {
      if (!worldEnables(world, origin.mechanism)) {
        if (error_ != SyncCoverFlatWorldError::None) {
          return false;
        }
        continue;
      }
      RegionWorldState &state = states[world];
      if (action.kind == RegionCutActionKind::Set) {
        std::uint16_t produced = transferCapabilities(
            state.completed[*sourceResource], origin.suppliedRequirements);
        const bool directlyCompletesSource =
            state.sourceIssued &&
            graph_.getNodes()[demand_.source].resource ==
                origin.sourceResource &&
            (origin.sourcePrefixCompletion ||
             demand_.source == origins_[action.cut].sourceNode);
        if (directlyCompletesSource) {
          produced |= capabilityClosure(origin.suppliedRequirements);
        }
        state.tokens[action.cut] |= produced;
      } else if (action.kind == RegionCutActionKind::Wait) {
        state.completed[*targetResource] |= state.tokens[action.cut];
        state.tokens[action.cut] = 0;
      } else {
        const bool directlyCompletesSource =
            state.sourceIssued &&
            graph_.getNodes()[demand_.source].resource == origin.sourceResource;
        if (directlyCompletesSource) {
          state.completed[*sourceResource] |=
              capabilityClosure(origin.suppliedRequirements);
        }
      }
    }
    return true;
  }

  bool processActions(const std::vector<RegionCutAction> &actions,
                      std::vector<RegionWorldState> &states,
                      const SyncCoverGuard &pathCondition) {
    for (const RegionCutAction &action : actions) {
      const bool actionProcessed =
          charge() && applyAction(action, states, pathCondition);
      if (!actionProcessed) {
        return false;
      }
    }
    return true;
  }

  bool evaluateNode(SyncCoverNodeId node, std::vector<RegionWorldState> &states,
                    const SyncCoverGuard &pathCondition) {
    bool nodeActive = false;
    if (!guardImplies(pathCondition, graph_.getNodes()[node].guard,
                      nodeActive)) {
      return false;
    }
    if (!nodeActive) {
      return true;
    }
    if (!processActions(beforeActions_[node], states, pathCondition)) {
      return false;
    }
    for (std::size_t fixed : fixedIncoming_[node]) {
      if (!charge()) {
        return false;
      }
      const SyncCoverEdge &edge = *fixedSupplies_[fixed];
      bool sourceActive = false;
      bool targetActive = false;
      if (!guardImplies(pathCondition, edge.sourceGuard, sourceActive) ||
          !guardImplies(pathCondition, edge.targetGuard, targetActive)) {
        return false;
      }
      if (!sourceActive || !targetActive) {
        continue;
      }
      const std::optional<std::size_t> sourceResource =
          resourceIndex(graph_.getNodes()[edge.source].resource);
      const std::optional<std::size_t> targetResource =
          resourceIndex(graph_.getNodes()[edge.target].resource);
      if (!sourceResource || !targetResource) {
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = fixed;
        return false;
      }
      if (!charge(states.size())) {
        return false;
      }
      for (RegionWorldState &state : states) {
        if (!state.fixedSourcesReached[fixed]) {
          continue;
        }
        std::uint16_t supplied = transferCapabilities(
            state.completed[*sourceResource], edge.suppliedRequirements);
        if (edge.source == demand_.source && state.sourceIssued) {
          supplied |= capabilityClosure(edge.suppliedRequirements);
        }
        state.completed[*targetResource] |= supplied;
      }
    }
    if (node == demand_.target) {
      const std::optional<std::size_t> targetResource =
          resourceIndex(graph_.getNodes()[node].resource);
      if (!targetResource) {
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = node;
        return false;
      }
      const std::uint16_t required = static_cast<std::uint16_t>(
          std::uint16_t{1} << demand_.orderingRequirements);
      if (!charge(states.size())) {
        return false;
      }
      for (RegionWorldState &state : states) {
        state.targetCovered |=
            (state.completed[*targetResource] & required) != 0;
      }
    }
    if (node == demand_.source) {
      if (!charge(states.size())) {
        return false;
      }
      for (RegionWorldState &state : states) {
        state.sourceIssued = true;
      }
    }
    for (std::size_t fixed : fixedOutgoing_[node]) {
      if (!charge(states.size())) {
        return false;
      }
      for (RegionWorldState &state : states) {
        state.fixedSourcesReached[fixed] = 1;
      }
    }
    return processActions(afterActions_[node], states, pathCondition);
  }

  bool intersectStates(std::vector<RegionWorldState> &target,
                       const std::vector<RegionWorldState> &source) {
    if (!chargeStateTraversal(target)) {
      return false;
    }
    for (std::size_t world = 0; world < target.size(); ++world) {
      RegionWorldState &left = target[world];
      const RegionWorldState &right = source[world];
      for (std::size_t resource = 0; resource < left.completed.size();
           ++resource) {
        left.completed[resource] &= right.completed[resource];
      }
      for (std::size_t token = 0; token < left.tokens.size(); ++token) {
        left.tokens[token] &= right.tokens[token];
      }
      for (std::size_t fixed = 0; fixed < left.fixedSourcesReached.size();
           ++fixed) {
        left.fixedSourcesReached[fixed] &= right.fixedSourcesReached[fixed];
      }
      left.sourceIssued &= right.sourceIssued;
      left.targetCovered &= right.targetCovered;
    }
    return true;
  }

  std::optional<SyncCoverGuard>
  extendCondition(const SyncCoverGuard &condition,
                  const SyncCoverGuard &extension) {
    std::size_t mergeWork = 0;
    if (!checkedSum(condition.literals.size(), extension.literals.size(),
                    mergeWork) ||
        !checkedSum(mergeWork, 1, mergeWork) || !charge(mergeWork)) {
      return std::nullopt;
    }
    SyncCoverGuard result;
    result.literals.reserve(condition.literals.size() +
                            extension.literals.size());
    std::size_t conditionIndex = 0;
    std::size_t extensionIndex = 0;
    while (conditionIndex < condition.literals.size() ||
           extensionIndex < extension.literals.size()) {
      if (extensionIndex == extension.literals.size() ||
          (conditionIndex < condition.literals.size() &&
           condition.literals[conditionIndex].control <
               extension.literals[extensionIndex].control)) {
        result.literals.push_back(condition.literals[conditionIndex++]);
        continue;
      }
      if (conditionIndex == condition.literals.size() ||
          extension.literals[extensionIndex].control <
              condition.literals[conditionIndex].control) {
        result.literals.push_back(extension.literals[extensionIndex++]);
        continue;
      }
      const SyncCoverGuardLiteral &conditionLiteral =
          condition.literals[conditionIndex++];
      const SyncCoverGuardLiteral &extensionLiteral =
          extension.literals[extensionIndex++];
      if (conditionLiteral.alternative != extensionLiteral.alternative) {
        return std::nullopt;
      }
      result.literals.push_back(conditionLiteral);
    }
    return result;
  }

  bool evaluateChoice(const SyncCoverRegion &region,
                      std::vector<RegionWorldState> &states,
                      const SyncCoverGuard &pathCondition) {
    std::vector<SyncCoverRegionId> selected;
    std::vector<SyncCoverRegionId> feasible;
    for (const SyncCoverRegionElement &element : region.elements) {
      if (!charge()) {
        return false;
      }
      if (element.kind != SyncCoverRegionElementKind::ChildRegion) {
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = region.id;
        return false;
      }
      const SyncCoverRegion &alternative = graph_.getRegions()[element.value];
      if (alternative.kind != SyncCoverRegionKind::Alternative) {
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = alternative.id;
        return false;
      }
      bool implied = false;
      if (!guardImplies(pathCondition, alternative.guard, implied)) {
        return false;
      }
      if (implied) {
        selected.push_back(alternative.id);
      } else {
        bool compatible = false;
        if (!guardsCompatible(pathCondition, alternative.guard, compatible)) {
          return false;
        }
        if (compatible) {
          feasible.push_back(alternative.id);
        }
      }
    }
    const bool ambiguousSpecialization = selected.size() > 1;
    const bool noPath = selected.empty() && feasible.empty();
    if (ambiguousSpecialization || noPath) {
      error_ = SyncCoverFlatWorldError::InvalidGraph;
      invalidIndex_ = region.id;
      return false;
    }
    const std::vector<SyncCoverRegionId> &alternatives =
        selected.empty() ? feasible : selected;
    if (!selected.empty()) {
      ++statistics_.guardSpecializations;
    }

    std::optional<std::vector<RegionWorldState>> mustState;
    std::size_t mustStateWords = 0;
    for (SyncCoverRegionId alternativeId : alternatives) {
      const SyncCoverRegion &alternative = graph_.getRegions()[alternativeId];
      const std::optional<SyncCoverGuard> alternativeCondition =
          extendCondition(pathCondition, alternative.guard);
      if (!alternativeCondition) {
        if (error_ != SyncCoverFlatWorldError::None) {
          return false;
        }
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = alternativeId;
        return false;
      }
      std::size_t alternativeStateWords = 0;
      if (!reserveStateClone(states, alternativeStateWords)) {
        return false;
      }
      std::vector<RegionWorldState> alternativeState = states;
      if (!evaluateRegion(alternativeId, alternativeState,
                          *alternativeCondition, true)) {
        releaseStateClone(alternativeStateWords);
        return false;
      }
      if (!mustState) {
        mustState = std::move(alternativeState);
        mustStateWords = alternativeStateWords;
      } else {
        if (!intersectStates(*mustState, alternativeState)) {
          releaseStateClone(alternativeStateWords);
          return false;
        }
        releaseStateClone(alternativeStateWords);
        ++statistics_.choiceIntersections;
      }
    }
    states = std::move(*mustState);
    releaseStateClone(mustStateWords);
    return true;
  }

  bool evaluateLinear(const SyncCoverRegion &region,
                      std::vector<RegionWorldState> &states,
                      const SyncCoverGuard &pathCondition) {
    for (const SyncCoverRegionElement &element : region.elements) {
      if (!charge()) {
        return false;
      }
      if (element.kind == SyncCoverRegionElementKind::Node) {
        if (!evaluateNode(element.value, states, pathCondition)) {
          return false;
        }
      } else if (!evaluateRegion(element.value, states, pathCondition, false)) {
        return false;
      }
    }
    return true;
  }

  bool evaluateRegion(SyncCoverRegionId regionId,
                      std::vector<RegionWorldState> &states,
                      const SyncCoverGuard &pathCondition, bool forced) {
    ++statistics_.regionsEvaluated;
    const bool validRegion = charge() && regionId < graph_.getRegions().size();
    if (!validRegion) {
      if (error_ == SyncCoverFlatWorldError::None) {
        error_ = SyncCoverFlatWorldError::InvalidGraph;
        invalidIndex_ = regionId;
      }
      return false;
    }
    const SyncCoverRegion &region = graph_.getRegions()[regionId];
    if (region.kind == SyncCoverRegionKind::Loop ||
        region.cardinality == SyncCoverRegionCardinality::ZeroOrMore ||
        region.cardinality == SyncCoverRegionCardinality::OneOrMore) {
      error_ = SyncCoverFlatWorldError::UnsupportedStructure;
      invalidIndex_ = regionId;
      return false;
    }
    bool guardCompatible = false;
    if (!guardsCompatible(pathCondition, region.guard, guardCompatible)) {
      return false;
    }
    if (!guardCompatible) {
      return true;
    }
    const std::optional<SyncCoverGuard> presentCondition =
        extendCondition(pathCondition, region.guard);
    if (!presentCondition) {
      if (error_ != SyncCoverFlatWorldError::None) {
        return false;
      }
      error_ = SyncCoverFlatWorldError::InvalidGraph;
      invalidIndex_ = regionId;
      return false;
    }
    if (region.kind == SyncCoverRegionKind::Choice) {
      return evaluateChoice(region, states, *presentCondition);
    }
    bool guardImplied = false;
    if (!guardImplies(pathCondition, region.guard, guardImplied)) {
      return false;
    }
    const bool unguardedOptional =
        region.cardinality == SyncCoverRegionCardinality::ZeroOrOne &&
        region.guard.literals.empty();
    const bool maySkip = !forced && (unguardedOptional || !guardImplied);
    if (!maySkip) {
      return evaluateLinear(region, states, *presentCondition);
    }
    std::size_t presentWords = 0;
    if (!reserveStateClone(states, presentWords)) {
      return false;
    }
    std::vector<RegionWorldState> present = states;
    if (!evaluateLinear(region, present, *presentCondition)) {
      releaseStateClone(presentWords);
      return false;
    }
    if (!intersectStates(states, present)) {
      releaseStateClone(presentWords);
      return false;
    }
    releaseStateClone(presentWords);
    ++statistics_.choiceIntersections;
    return true;
  }

  const SyncCoverGraph &graph_;
  const std::vector<SyncCoverCompletionOrigin> &origins_;
  const std::vector<SyncCoverExactWorld> &worlds_;
  const std::vector<std::uint32_t> &resources_;
  const std::vector<std::vector<RegionCutAction>> &beforeActions_;
  const std::vector<std::vector<RegionCutAction>> &afterActions_;
  const std::vector<const SyncCoverEdge *> &fixedSupplies_;
  const std::vector<std::vector<std::size_t>> &fixedIncoming_;
  const std::vector<std::vector<std::size_t>> &fixedOutgoing_;
  const SyncCoverDemand &demand_;
  SyncCoverGuard demandCondition_;
  SyncCoverRegionWorldLimits limits_;
  std::size_t liveStateWords_ = 0;
  SyncCoverRegionWorldStatistics &statistics_;
  SyncCoverCoverageWorkBudget *workBudget_ = nullptr;
  SyncCoverFlatWorldError error_ = SyncCoverFlatWorldError::None;
  std::optional<std::size_t> invalidIndex_;
};

} // namespace

SyncCoverRegionWorldResult mlir::pto::computeSyncCoverRegionExactWorlds(
    const SyncCoverGraph &graph, const std::vector<SyncCoverDirectCut> &cuts,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverRegionWorldLimits limits,
    SyncCoverCoverageWorkBudget *workBudget) {
  SyncCoverRegionWorldResult result;
  const auto fail = [&](SyncCoverFlatWorldError error,
                        std::optional<std::size_t> index = std::nullopt) {
    result.error = error;
    result.invalidIndex = index;
    result.coveredByWorld.clear();
    return result;
  };
  const bool limitsExceeded = worlds.size() > limits.maximumWorldsPerBatch ||
                              cuts.size() > limits.maximumCuts;
  if (limitsExceeded) {
    return fail(SyncCoverFlatWorldError::LimitExceeded);
  }
  std::size_t worldMechanismIncidences = 0;
  for (const SyncCoverExactWorld &world : worlds) {
    if (!checkedSum(worldMechanismIncidences, world.enabledMechanisms.size(),
                    worldMechanismIncidences) ||
        worldMechanismIncidences > limits.maximumWorldMechanismIncidences) {
      return fail(SyncCoverFlatWorldError::LimitExceeded);
    }
  }
  const std::size_t demandWords =
      graph.getDemands().size() / 64 + (graph.getDemands().size() % 64 != 0);
  std::size_t resultWords = 0;
  if (!checkedProduct(worlds.size(), demandWords, resultWords) ||
      resultWords > limits.maximumResultWords) {
    return fail(SyncCoverFlatWorldError::LimitExceeded);
  }
  std::size_t structuralWork = 0;
  const bool structuralWorkAvailable =
      checkedSum(graph.getNodes().size(), graph.getEdges().size(),
                 structuralWork) &&
      checkedSum(structuralWork, graph.getDemands().size(), structuralWork) &&
      checkedSum(structuralWork, graph.getRegions().size(), structuralWork) &&
      checkedSum(structuralWork, cuts.size(), structuralWork) &&
      checkedSum(structuralWork, worlds.size(), structuralWork) &&
      checkedSum(structuralWork, worldMechanismIncidences, structuralWork) &&
      consumeWork(workBudget, structuralWork);
  if (!structuralWorkAvailable) {
    return fail(workBudget && workBudget->exhausted
                    ? SyncCoverFlatWorldError::WorkLimitExceeded
                    : SyncCoverFlatWorldError::LimitExceeded);
  }
  // A frozen graph has already passed structural validation and cannot be
  // mutated. Exact-world work accounting therefore starts from that immutable
  // certificate instead of revalidating guards outside the supplied budget.
  const bool invalidGraph = !graph.isStructureFrozen();
  if (invalidGraph) {
    return fail(SyncCoverFlatWorldError::InvalidGraph);
  }
  const bool unsupportedDemand = std::any_of(
      graph.getDemands().begin(), graph.getDemands().end(),
      [](const SyncCoverDemand &demand) { return demand.distance != 0; });
  if (unsupportedDemand) {
    return fail(SyncCoverFlatWorldError::UnsupportedStructure);
  }

  std::vector<SyncCoverCompletionOrigin> origins;
  origins.reserve(cuts.size());
  std::vector<SyncCoverMechanismId> describedMechanisms;
  describedMechanisms.reserve(cuts.size());
  std::vector<std::vector<RegionCutAction>> beforeActions(
      graph.getNodes().size());
  std::vector<std::vector<RegionCutAction>> afterActions(
      graph.getNodes().size());
  std::vector<const SyncCoverEdge *> fixedSupplies;
  std::vector<std::vector<std::size_t>> fixedIncoming(graph.getNodes().size());
  std::vector<std::vector<std::size_t>> fixedOutgoing(graph.getNodes().size());
  for (const SyncCoverEdge &edge : graph.getEdges()) {
    if (edge.kind != SyncCoverEdgeKind::CompletionSupply) {
      continue;
    }
    if (edge.distance != 0) {
      return fail(SyncCoverFlatWorldError::UnsupportedStructure);
    }
    const std::size_t fixed = fixedSupplies.size();
    fixedSupplies.push_back(&edge);
    fixedIncoming[edge.target].push_back(fixed);
    fixedOutgoing[edge.source].push_back(fixed);
  }
  std::size_t actionCount = 0;
  const auto addAction = [&](const SyncCoverCutPoint &point,
                             RegionCutActionKind kind, std::size_t cut) {
    const bool actionLimitReached = actionCount == limits.maximumCutActions;
    if (actionLimitReached) {
      return false;
    }
    RegionCutAction action{kind, cut, point.ordinal};
    std::vector<std::vector<RegionCutAction>> &actions =
        point.anchor.kind == SyncCoverAnchorKind::BeforeNode ? beforeActions
                                                             : afterActions;
    actions[point.anchor.node].push_back(action);
    ++actionCount;
    return true;
  };
  for (std::size_t cut = 0; cut < cuts.size(); ++cut) {
    const std::optional<SyncCoverCompletionOrigin> origin =
        makeFlatCompletionOrigin(graph, cuts[cut], workBudget);
    if (!origin) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverFlatWorldError::WorkLimitExceeded
                      : SyncCoverFlatWorldError::InvalidCut,
                  cut);
    }
    origins.push_back(*origin);
    describedMechanisms.push_back(origin->mechanism);
    const bool event = origin->kind == SyncCoverDirectCutKind::Event;
    const bool actionsAdded =
        event ? addAction(cuts[cut].source, RegionCutActionKind::Set, cut) &&
                    addAction(cuts[cut].target, RegionCutActionKind::Wait, cut)
              : addAction(cuts[cut].source, RegionCutActionKind::Barrier, cut);
    if (!actionsAdded) {
      return fail(SyncCoverFlatWorldError::LimitExceeded, cut);
    }
  }
  for (auto &actions : beforeActions) {
    if (!meteredStableSort(actions, regionActionLess, workBudget)) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
  }
  for (auto &actions : afterActions) {
    if (!meteredStableSort(actions, regionActionLess, workBudget)) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
  }
  if (!meteredStableSort(describedMechanisms, std::less<>(), workBudget) ||
      !consumeWork(workBudget, describedMechanisms.size())) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }
  describedMechanisms.erase(
      std::unique(describedMechanisms.begin(), describedMechanisms.end()),
      describedMechanisms.end());
  std::size_t worldValidationWork = 0;
  std::size_t includesWork = 0;
  const bool worldValidationAvailable =
      checkedProduct(worldMechanismIncidences, 2, worldValidationWork) &&
      checkedProduct(worlds.size(), describedMechanisms.size(), includesWork) &&
      checkedSum(worldValidationWork, includesWork, worldValidationWork) &&
      checkedSum(worldValidationWork, worlds.size(), worldValidationWork) &&
      consumeWork(workBudget, worldValidationWork);
  if (!worldValidationAvailable) {
    return fail(workBudget && workBudget->exhausted
                    ? SyncCoverFlatWorldError::WorkLimitExceeded
                    : SyncCoverFlatWorldError::LimitExceeded);
  }
  for (std::size_t world = 0; world < worlds.size(); ++world) {
    const bool invalidWorld =
        !std::is_sorted(worlds[world].enabledMechanisms.begin(),
                        worlds[world].enabledMechanisms.end()) ||
        std::adjacent_find(worlds[world].enabledMechanisms.begin(),
                           worlds[world].enabledMechanisms.end()) !=
            worlds[world].enabledMechanisms.end() ||
        !std::includes(describedMechanisms.begin(), describedMechanisms.end(),
                       worlds[world].enabledMechanisms.begin(),
                       worlds[world].enabledMechanisms.end());
    if (invalidWorld) {
      return fail(SyncCoverFlatWorldError::InvalidWorld, world);
    }
  }

  std::vector<std::uint32_t> resources;
  resources.reserve(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    resources.push_back(node.resource);
  }
  if (!meteredStableSort(resources, std::less<>(), workBudget) ||
      !consumeWork(workBudget, resources.size())) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }
  resources.erase(std::unique(resources.begin(), resources.end()),
                  resources.end());
  std::size_t wordsPerWorld = 0;
  std::size_t liveStateWords = 0;
  const bool stateLimitExceeded =
      !checkedSum(resources.size(), cuts.size(), 2, wordsPerWorld) ||
      !checkedSum(wordsPerWorld, fixedSupplies.size(), wordsPerWorld) ||
      !checkedProduct(wordsPerWorld, worlds.size(), liveStateWords) ||
      liveStateWords > limits.maximumStateWords;
  if (stateLimitExceeded) {
    return fail(SyncCoverFlatWorldError::LimitExceeded);
  }
  result.statistics.maximumLiveStateWords = liveStateWords;
  if (!consumeWork(workBudget, resultWords)) {
    return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
  }
  result.coveredByWorld.assign(worlds.size(),
                               SyncCoverDemandSet(graph.getDemands().size()));

  for (SyncCoverDemandId demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    SyncCoverGuard demandCondition;
    if (!flatDemandCondition(graph, demand, demandCondition, workBudget)) {
      return fail(workBudget && workBudget->exhausted
                      ? SyncCoverFlatWorldError::WorkLimitExceeded
                      : SyncCoverFlatWorldError::InvalidGraph,
                  demandId);
    }
    if (!consumeWork(workBudget, liveStateWords)) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
    std::vector<RegionWorldState> states(worlds.size());
    for (RegionWorldState &state : states) {
      state.completed.assign(resources.size(), 0);
      state.tokens.assign(cuts.size(), 0);
      state.fixedSourcesReached.assign(fixedSupplies.size(), 0);
    }
    RegionWorldEvaluator evaluator(
        graph, origins, worlds, resources, beforeActions, afterActions,
        fixedSupplies, fixedIncoming, fixedOutgoing, demand,
        std::move(demandCondition), limits, liveStateWords, result.statistics,
        workBudget);
    if (!evaluator.run(states)) {
      return fail(evaluator.getError(), evaluator.getInvalidIndex());
    }
    if (!consumeWork(workBudget, worlds.size())) {
      return fail(SyncCoverFlatWorldError::WorkLimitExceeded);
    }
    for (std::size_t world = 0; world < worlds.size(); ++world) {
      if (states[world].targetCovered) {
        result.coveredByWorld[world].insert(demandId);
      }
    }
  }
  return result;
}
