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
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;
using namespace mlir::pto::sync_cover_detail;

namespace {

constexpr std::size_t kBitsPerWord = 64;

struct IndexedSupply {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
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
                    std::size_t virtualNode) {
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
        copy && nodeInstanceAvailable(graph, demand, *operation, *copy) &&
        guardIsImplied(graph, demand, context,
                       graph.getNodes()[*operation].guard, *copy);
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
                                      std::size_t target,
                                      std::size_t state) {
  const bool hasCompletion = (state % 2) != 0;
  switch (kind) {
  case SyncCoverEdgeKind::CompletionSupply:
    return getStateIndex(target, true);
  case SyncCoverEdgeKind::CompletionPreservingIssueOrder:
  case SyncCoverEdgeKind::NonCompletionPreservingIssueOrder:
    return hasCompletion
               ? std::optional<std::size_t>(getStateIndex(target, true))
               : std::nullopt;
  }
  return std::nullopt;
}

SupplyIndex buildSupplyIndex(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverCompletionSupply> &supplies) {
  SupplyIndex result;
  result.edges.reserve(supplies.size());
  for (std::size_t supplyId = 0; supplyId < supplies.size(); ++supplyId) {
    const SyncCoverCompletionSupply &supply = supplies[supplyId];
    result.edges.push_back({supply.edge.source, supply.edge.target,
                            supply.mechanism, supplyId});
  }
  std::sort(result.edges.begin(), result.edges.end(),
            [](const IndexedSupply &left, const IndexedSupply &right) {
              return std::tie(left.source, left.target, left.mechanism,
                              left.supply) <
                     std::tie(right.source, right.target, right.mechanism,
                              right.supply);
            });
  result.outgoingOffsets.assign(graph.getNodes().size() + 1, 0);
  for (const IndexedSupply &edge : result.edges) {
    ++result.outgoingOffsets[edge.source + 1];
  }
  for (std::size_t index = 1; index < result.outgoingOffsets.size(); ++index) {
    result.outgoingOffsets[index] += result.outgoingOffsets[index - 1];
  }
  return result;
}

template <typename Visitor>
void visitSupplyOutgoing(const SupplyIndex &index, SyncCoverNodeId source,
                         Visitor &&visitor) {
  const bool invalidNode = index.outgoingOffsets.empty() ||
                           source >= index.outgoingOffsets.size() - 1;
  if (invalidNode) {
    return;
  }
  for (std::size_t edge = index.outgoingOffsets[source];
       edge < index.outgoingOffsets[source + 1]; ++edge) {
    visitor(index.edges[edge]);
  }
}

bool coversDemand(const SyncCoverGraph &graph, const SyncCoverDemand &demand,
                  const SyncCoverExpandedArena &arena,
                  const SupplyIndex &supplyIndex,
                  const std::vector<SyncCoverCompletionSupply> &supplies,
                  CoverageWorkspace &workspace) {
  const DemandContext context = makeDemandContext(graph, demand);
  const std::optional<std::size_t> source =
      arena.getVirtualOperation(demand.source, 0);
  const std::optional<std::size_t> target =
      arena.getVirtualOperation(demand.target, demand.distance);
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

  for (std::size_t readyIndex = 0;
       readyIndex < workspace.getReady().size(); ++readyIndex) {
    const std::size_t state = workspace.getReady()[readyIndex];
    const std::size_t virtualNode = state / 2;
    for (const SyncCoverExpandedEdge &edge :
         arena.getOutgoingEdges(virtualNode)) {
      const bool activeEndpoints =
          edge.targetCopy <= demand.distance &&
          workspace.isNodeActive(graph, demand, context, arena, edge.source) &&
          workspace.isNodeActive(graph, demand, context, arena, edge.target);
      const bool active =
          activeEndpoints &&
          (!edge.graphEdge ||
           edgeGuardsActive(graph, demand, context,
                            graph.getEdges()[*edge.graphEdge], edge.sourceCopy,
                            edge.targetCopy));
      if (active) {
        if (const std::optional<std::size_t> next =
                transition(edge.kind, edge.target, state)) {
          enqueue(*next);
        }
      }
    }

    const std::optional<SyncCoverNodeId> sourceOperation =
        arena.getOperationForVirtualNode(virtualNode);
    const std::optional<unsigned> sourceCopy =
        arena.getCopyForVirtualNode(virtualNode);
    if (!sourceOperation || !sourceCopy) {
      continue;
    }
    visitSupplyOutgoing(
        supplyIndex, *sourceOperation, [&](const IndexedSupply &indexed) {
          const SyncCoverEdge &supply = supplies[indexed.supply].edge;
          const bool distanceOutOfRange =
              supply.distance > arena.getHorizon() - *sourceCopy;
          if (distanceOutOfRange) {
            return;
          }
          const unsigned targetCopy = *sourceCopy + supply.distance;
          const std::optional<std::size_t> targetNode =
              arena.getVirtualOperation(indexed.target, targetCopy);
          const bool wrongArena =
              supply.distance != 0 && supply.scope != arena.getScope();
          const bool active =
              !wrongArena && targetNode && targetCopy <= demand.distance &&
              workspace.isNodeActive(graph, demand, context, arena,
                                     virtualNode) &&
              workspace.isNodeActive(graph, demand, context, arena,
                                     *targetNode) &&
              edgeGuardsActive(graph, demand, context, supply, *sourceCopy,
                               targetCopy);
          if (active) {
            enqueue(getStateIndex(*targetNode, true));
          }
        });
  }
  return workspace.wasSeen(goal);
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
  return demand < size_ &&
         (words_[demand / kBitsPerWord] &
          (std::uint64_t{1} << (demand % kBitsPerWord))) != 0;
}

bool SyncCoverDemandSet::insert(SyncCoverDemandId demand) {
  if (demand >= size_) {
    return false;
  }
  words_[demand / kBitsPerWord] |=
      std::uint64_t{1} << (demand % kBitsPerWord);
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
  SyncCoverCoverageResult result;
  result.covered = SyncCoverDemandSet(graph.getDemands().size());
  const bool invalidGraph = !graph.isStructureFrozen() ||
                            !expansion.isForGraph(graph) ||
                            expansion.getError() ==
                                SyncCoverExpansionError::InvalidGraph ||
                            expansion.getError() ==
                                SyncCoverExpansionError::InvalidLimits;
  if (invalidGraph) {
    result.error = SyncCoverCoverageError::InvalidGraph;
    return result;
  }
  const bool baseUnavailable =
      expansion.getError() == SyncCoverExpansionError::BaseLimitExceeded;
  if (baseUnavailable) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
    result.unavailableDemands.reserve(graph.getDemands().size());
    for (SyncCoverDemandId demand = 0; demand < graph.getDemands().size();
         ++demand) {
      result.unavailableDemands.push_back(demand);
    }
    return result;
  }

  std::vector<SyncCoverCompletionSupply> supplies = inputSupplies;
  for (SyncCoverCompletionSupply &supply : supplies) {
    const bool invalidSupply = graph.canonicalizeCompletionEdge(supply.edge) !=
                               SyncCoverGraphError::None;
    if (invalidSupply) {
      result.error = SyncCoverCoverageError::InvalidSupply;
      return result;
    }
  }
  std::stable_sort(
      supplies.begin(), supplies.end(),
      [](const SyncCoverCompletionSupply &left,
         const SyncCoverCompletionSupply &right) {
        return std::tie(left.mechanism, left.edge.source, left.edge.target,
                        left.edge.scope, left.edge.distance,
                        left.edge.sourceGuard.literals,
                        left.edge.targetGuard.literals) <
               std::tie(right.mechanism, right.edge.source, right.edge.target,
                        right.edge.scope, right.edge.distance,
                        right.edge.sourceGuard.literals,
                        right.edge.targetGuard.literals);
      });

  const SupplyIndex supplyIndex = buildSupplyIndex(graph, supplies);
  std::map<const SyncCoverExpandedArena *, CoverageWorkspace> workspaces;
  for (SyncCoverDemandId demandId = 0;
       demandId < graph.getDemands().size(); ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const SyncCoverExpandedArena *arena = expansion.getArena(demand);
    if (!arena) {
      result.unavailableDemands.push_back(demandId);
      continue;
    }
    auto [workspace, inserted] =
        workspaces.try_emplace(arena, arena->getVirtualNodeCount());
    (void)inserted;
    if (coversDemand(graph, demand, *arena, supplyIndex, supplies,
                     workspace->second)) {
      result.covered.insert(demandId);
    }
  }
  if (!result.unavailableDemands.empty()) {
    result.error = SyncCoverCoverageError::ExpansionUnavailable;
  }
  return result;
}
