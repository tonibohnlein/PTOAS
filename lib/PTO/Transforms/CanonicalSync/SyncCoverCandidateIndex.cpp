// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverCandidateIndex.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir::pto;

namespace {

struct ContextKey {
  std::uint32_t resource = 0;
  SyncCoverScopeId timelineScope = 0;

  bool operator<(const ContextKey &other) const {
    return std::tie(resource, timelineScope) <
           std::tie(other.resource, other.timelineScope);
  }
};

struct RegionKey {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval interval;

  bool operator<(const RegionKey &other) const {
    return std::tie(domain, interval.begin, interval.end) <
           std::tie(other.domain, other.interval.begin, other.interval.end);
  }
};

} // namespace

SyncCoverCandidateIndex::SyncCoverCandidateIndex(const SyncCoverGraph &graph)
    : graph_(graph), structuralGeneration_(graph.getStructuralGeneration()) {
  if (!graph.isStructureFrozen()) {
    buildError_ = SyncCoverCandidateIndexError::StructureNotFrozen;
    return;
  }
  std::map<ContextKey, SyncCoverCandidateTimelineId> timelineIds;
  nodePositions_.resize(graph.getNodes().size());
  for (const SyncCoverNode &node : graph.getNodes()) {
    const std::optional<SyncCoverScopeId> timelineScope =
        graph.getOwningTimelineScope(node.scope);
    if (!timelineScope) {
      buildError_ = SyncCoverCandidateIndexError::InvalidGraph;
      return;
    }
    ContextKey key{node.resource, *timelineScope};
    auto [position, inserted] =
        timelineIds.emplace(key, timelines_.size());
    if (inserted) {
      SyncCoverCandidateTimeline timeline;
      timeline.id = position->second;
      timeline.context = {node.resource, *timelineScope};
      timelines_.push_back(std::move(timeline));
    }
    timelines_[position->second].nodes.push_back(node.id);
  }
  for (SyncCoverCandidateTimeline &timeline : timelines_) {
    std::sort(timeline.nodes.begin(), timeline.nodes.end(),
              [&](SyncCoverNodeId first, SyncCoverNodeId second) {
                return graph.getNodes()[first].order <
                       graph.getNodes()[second].order;
              });
    for (std::size_t ordinal = 0; ordinal < timeline.nodes.size(); ++ordinal) {
      nodePositions_[timeline.nodes[ordinal]] =
          SyncCoverCandidateNodePosition{timeline.id, ordinal};
    }
  }

  domainAccesses_.resize(graph.getStorageDomains().size());
  for (const SyncCoverStorageAccess &access : graph.getStorageAccesses()) {
    domainAccesses_[access.domain].push_back(access.id);
  }
  for (std::vector<SyncCoverStorageAccessId> &accesses : domainAccesses_) {
    std::sort(accesses.begin(), accesses.end(),
              [&](SyncCoverStorageAccessId first,
                  SyncCoverStorageAccessId second) {
                const SyncCoverStorageAccess &lhs =
                    graph.getStorageAccesses()[first];
                const SyncCoverStorageAccess &rhs =
                    graph.getStorageAccesses()[second];
                return std::tie(lhs.extent.begin, lhs.extent.end, lhs.node,
                                lhs.family, lhs.addressOrdinal, lhs.id) <
                       std::tie(rhs.extent.begin, rhs.extent.end, rhs.node,
                                rhs.family, rhs.addressOrdinal, rhs.id);
              });
  }

  demandWitnesses_.reserve(graph.getDemands().size());
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    demandWitnesses_.push_back(demand.storageWitnesses);
  }

  std::map<RegionKey, std::vector<SyncCoverStorageWitnessId>> regionWitnesses;
  std::vector<std::vector<std::size_t>> witnessDemands(
      graph.getStorageWitnesses().size());
  for (std::size_t demand = 0; demand < graph.getDemands().size(); ++demand) {
    for (SyncCoverStorageWitnessId witness :
         graph.getDemands()[demand].storageWitnesses) {
      witnessDemands[witness].push_back(demand);
    }
  }
  for (const SyncCoverStorageWitness &witness : graph.getStorageWitnesses()) {
    const SyncCoverStorageAccess &source =
        graph.getStorageAccesses()[witness.sourceAccess];
    regionWitnesses[{source.domain, witness.overlap}].push_back(witness.id);
  }
  for (auto &entry : regionWitnesses) {
    std::vector<std::size_t> demands;
    for (SyncCoverStorageWitnessId witness : entry.second) {
      demands.insert(demands.end(), witnessDemands[witness].begin(),
                     witnessDemands[witness].end());
    }
    std::sort(demands.begin(), demands.end());
    demands.erase(std::unique(demands.begin(), demands.end()), demands.end());
    witnessOverlapGroups_.push_back({entry.first.domain, entry.first.interval,
                                     std::move(entry.second),
                                     std::move(demands)});
  }

  demandOpportunities_.resize(graph.getDemands().size());
  for (std::size_t demandId = 0; demandId < graph.getDemands().size();
       ++demandId) {
    const SyncCoverDemand &demand = graph.getDemands()[demandId];
    const auto appendOpportunity =
        [&](std::optional<SyncCoverStorageWitnessId> witnessId) {
          SyncCoverCandidateOpportunity opportunity;
          opportunity.id = opportunities_.size();
          opportunity.demand = demandId;
          opportunity.kind = demand.kind;
          opportunity.source = demand.source;
          opportunity.target = demand.target;
          opportunity.sourceResource = graph.getNodes()[demand.source].resource;
          opportunity.targetResource = graph.getNodes()[demand.target].resource;
          opportunity.sourcePosition = *nodePositions_[demand.source];
          opportunity.targetPosition = *nodePositions_[demand.target];
          opportunity.scope = demand.scope;
          opportunity.distance = demand.distance;
          opportunity.sourceGuard = demand.sourceGuard;
          opportunity.targetGuard = demand.targetGuard;
          opportunity.storageProvenance = demand.storageProvenance;
          if (witnessId) {
            const SyncCoverStorageWitness &witness =
                graph.getStorageWitnesses()[*witnessId];
            const SyncCoverStorageAccess &source =
                graph.getStorageAccesses()[witness.sourceAccess];
            const SyncCoverStorageAccess &target =
                graph.getStorageAccesses()[witness.targetAccess];
            opportunity.slot = SyncCoverCandidateSlot{
                source.domain, witness.overlap, source.extent, target.extent,
                source.id, target.id, source.family, target.family,
                source.mode, target.mode, source.addressOrdinal,
                target.addressOrdinal};
          }
          demandOpportunities_[opportunity.demand].push_back(opportunity.id);
          opportunities_.push_back(std::move(opportunity));
        };
    if (demand.storageProvenance != SyncCoverStorageProvenance::Complete) {
      appendOpportunity(std::nullopt);
      continue;
    }
    for (SyncCoverStorageWitnessId witness : demand.storageWitnesses) {
      appendOpportunity(witness);
    }
  }
}

SyncCoverCandidateIndex::operator bool() const {
  return currentError() == SyncCoverCandidateIndexError::None;
}

SyncCoverCandidateIndexError SyncCoverCandidateIndex::getError() const {
  return currentError();
}

bool SyncCoverCandidateIndex::isCurrentFor(const SyncCoverGraph &graph) const {
  return &graph == &graph_ &&
         currentError() == SyncCoverCandidateIndexError::None;
}

SyncCoverCandidateIndexError SyncCoverCandidateIndex::currentError() const {
  if (buildError_ != SyncCoverCandidateIndexError::None) {
    return buildError_;
  }
  const bool stale = !graph_.isStructureFrozen() ||
                     graph_.getStructuralGeneration() != structuralGeneration_;
  if (stale) {
    return SyncCoverCandidateIndexError::Stale;
  }
  return SyncCoverCandidateIndexError::None;
}

SyncCoverCandidateLookup<std::vector<SyncCoverCandidateTimeline>>
SyncCoverCandidateIndex::getTimelines() const {
  const SyncCoverCandidateIndexError error = currentError();
  return {error, error == SyncCoverCandidateIndexError::None ? &timelines_
                                                             : nullptr};
}

SyncCoverCandidateLookup<SyncCoverCandidateNodePosition>
SyncCoverCandidateIndex::getNodePosition(SyncCoverNodeId node) const {
  const SyncCoverCandidateIndexError error = currentError();
  if (error != SyncCoverCandidateIndexError::None) {
    return {error, nullptr};
  }
  const bool invalidNode =
      node >= nodePositions_.size() || !nodePositions_[node];
  if (invalidNode) {
    return {SyncCoverCandidateIndexError::InvalidIndex, nullptr};
  }
  return {SyncCoverCandidateIndexError::None, &*nodePositions_[node]};
}

SyncCoverCandidateLookup<std::vector<SyncCoverStorageAccessId>>
SyncCoverCandidateIndex::getDomainAccesses(
    SyncCoverStorageDomainId domain) const {
  const SyncCoverCandidateIndexError error = currentError();
  if (error != SyncCoverCandidateIndexError::None) {
    return {error, nullptr};
  }
  if (domain >= domainAccesses_.size()) {
    return {SyncCoverCandidateIndexError::InvalidIndex, nullptr};
  }
  return {SyncCoverCandidateIndexError::None, &domainAccesses_[domain]};
}

SyncCoverCandidateLookup<std::vector<SyncCoverStorageWitnessId>>
SyncCoverCandidateIndex::getDemandWitnesses(std::size_t demand) const {
  const SyncCoverCandidateIndexError error = currentError();
  if (error != SyncCoverCandidateIndexError::None) {
    return {error, nullptr};
  }
  if (demand >= demandWitnesses_.size()) {
    return {SyncCoverCandidateIndexError::InvalidIndex, nullptr};
  }
  return {SyncCoverCandidateIndexError::None, &demandWitnesses_[demand]};
}

SyncCoverCandidateLookup<
    std::vector<SyncCoverCandidateWitnessOverlapGroup>>
SyncCoverCandidateIndex::getWitnessOverlapGroups() const {
  const SyncCoverCandidateIndexError error = currentError();
  return {error, error == SyncCoverCandidateIndexError::None
                     ? &witnessOverlapGroups_
                     : nullptr};
}

SyncCoverCandidateLookup<std::vector<SyncCoverCandidateOpportunity>>
SyncCoverCandidateIndex::getOpportunities() const {
  const SyncCoverCandidateIndexError error = currentError();
  return {error, error == SyncCoverCandidateIndexError::None ? &opportunities_
                                                             : nullptr};
}

SyncCoverCandidateLookup<std::vector<SyncCoverCandidateOpportunityId>>
SyncCoverCandidateIndex::getDemandOpportunities(std::size_t demand) const {
  const SyncCoverCandidateIndexError error = currentError();
  if (error != SyncCoverCandidateIndexError::None) {
    return {error, nullptr};
  }
  if (demand >= demandOpportunities_.size()) {
    return {SyncCoverCandidateIndexError::InvalidIndex, nullptr};
  }
  return {SyncCoverCandidateIndexError::None,
          &demandOpportunities_[demand]};
}
