// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverGrounded.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir::pto;

namespace {

constexpr std::size_t kWordBits = 64;

bool canonicalIdentitySet(const std::vector<std::size_t> &values) {
  return std::is_sorted(values.begin(), values.end()) &&
         std::adjacent_find(values.begin(), values.end()) == values.end();
}

std::size_t countBits(std::uint64_t value) {
  std::size_t result = 0;
  while (value != 0) {
    value &= value - 1;
    ++result;
  }
  return result;
}

std::optional<std::size_t>
getActionLoopDepth(const SyncCoverGraph &graph,
                   const SyncCoverResourceAction &action) {
  switch (action.anchor.kind) {
  case SyncCoverAnchorKind::BeforeNode:
  case SyncCoverAnchorKind::AfterNode:
    if (action.anchor.node >= graph.getNodes().size()) {
      return std::nullopt;
    }
    return graph.getScopeLoopDepth(
        graph.getNodes()[action.anchor.node].scope);
  case SyncCoverAnchorKind::ScopeEntry:
  case SyncCoverAnchorKind::ScopeExit:
  case SyncCoverAnchorKind::TimelinePoint:
    if (action.anchor.scope >= graph.getScopes().size()) {
      return std::nullopt;
    }
    return graph.getScopeLoopDepth(action.anchor.scope, false);
  }
  return std::nullopt;
}

bool increment(std::vector<std::size_t> &profile, std::size_t maximumDepth,
               std::size_t depth) {
  if (depth > maximumDepth) {
    return false;
  }
  std::size_t &value = profile[maximumDepth - depth];
  if (value == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  ++value;
  return true;
}

SyncCoverGroundingError snapshotMechanisms(
    const SyncCoverMechanismUniverse &universe,
    SyncCoverGroundedInstance &instance) {
  const SyncCoverGraph &graph = universe.getGraph();
  std::size_t maximumDepth = 0;
  for (const SyncCoverScope &scope : graph.getScopes()) {
    const std::optional<std::size_t> depth =
        graph.getScopeLoopDepth(scope.id);
    if (!depth) {
      return SyncCoverGroundingError::InvalidUniverse;
    }
    maximumDepth = std::max(maximumDepth, *depth);
  }

  instance.mechanisms.reserve(universe.getMechanisms().size());
  for (const SyncCoverMechanism &mechanism : universe.getMechanisms()) {
    SyncCoverGroundedMechanism grounded;
    grounded.id = mechanism.id;
    grounded.kind = mechanism.kind;
    grounded.conflicts = mechanism.conflicts;
    grounded.actionProfile.assign(maximumDepth + 1, 0);
    grounded.barrierActionProfile.assign(maximumDepth + 1, 0);
    for (const SyncCoverResourceAction &action : mechanism.actions) {
      const std::optional<std::size_t> depth =
          getActionLoopDepth(graph, action);
      if (!depth ||
          !increment(grounded.actionProfile, maximumDepth, *depth)) {
        return SyncCoverGroundingError::ArithmeticOverflow;
      }
    }
    if (mechanism.barrier) {
      const std::optional<std::size_t> depth =
          graph.getScopeLoopDepth(mechanism.barrier->scope);
      if (!depth ||
          !increment(grounded.barrierActionProfile, maximumDepth, *depth)) {
        return SyncCoverGroundingError::ArithmeticOverflow;
      }
    }
    for (std::size_t useId = 0; useId < mechanism.resourceUses.size();
         ++useId) {
      const SyncCoverResourceUse &use = mechanism.resourceUses[useId];
      const std::optional<SyncCoverTimelineInterval> lifetime =
          getSyncCoverResourceLifetime(graph, mechanism, use);
      if (!lifetime) {
        return SyncCoverGroundingError::InvalidMechanism;
      }
      grounded.resourceUses.push_back(
          {useId, use.domain, use.width, *lifetime});
    }
    instance.mechanisms.push_back(std::move(grounded));
  }
  instance.resourceDomains = universe.getResourceDomains();
  return SyncCoverGroundingError::None;
}

} // namespace

SyncCoverDemandSet::SyncCoverDemandSet(std::size_t size) : size_(size) {
  const std::size_t fullWords = size / kWordBits;
  const std::size_t partialWord = size % kWordBits == 0 ? 0 : 1;
  words_.assign(fullWords + partialWord, 0);
}

bool SyncCoverDemandSet::add(std::size_t demand) {
  if (demand >= size_) {
    return false;
  }
  words_[demand / kWordBits] |=
      std::uint64_t{1} << static_cast<unsigned>(demand % kWordBits);
  return true;
}

bool SyncCoverDemandSet::contains(std::size_t demand) const {
  return demand < size_ &&
         (words_[demand / kWordBits] &
          (std::uint64_t{1} << static_cast<unsigned>(demand % kWordBits))) !=
             0;
}

bool SyncCoverDemandSet::containsAll(const SyncCoverDemandSet &other) const {
  if (size_ != other.size_) {
    return false;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    const bool missingWord =
        (words_[index] & other.words_[index]) != other.words_[index];
    if (missingWord) {
      return false;
    }
  }
  return true;
}

bool SyncCoverDemandSet::intersects(const SyncCoverDemandSet &other) const {
  if (size_ != other.size_) {
    return false;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    const bool sharesWord = (words_[index] & other.words_[index]) != 0;
    if (sharesWord) {
      return true;
    }
  }
  return false;
}

void SyncCoverDemandSet::unite(const SyncCoverDemandSet &other) {
  if (size_ != other.size_) {
    return;
  }
  for (std::size_t index = 0; index < words_.size(); ++index) {
    words_[index] |= other.words_[index];
  }
}

std::size_t SyncCoverDemandSet::count() const {
  std::size_t result = 0;
  for (std::uint64_t word : words_) {
    result += countBits(word);
  }
  return result;
}

bool SyncCoverGroundedInstance::isCurrent(
    const SyncCoverMechanismUniverse &universe) const {
  return universeVersion == universe.getVersion() &&
         graphGeneration == universe.getGraph().getGeneration();
}

SyncCoverDemandSet SyncCoverGroundedInstance::coveredBy(
    const std::vector<SyncCoverMechanismId> &selected) const {
  std::vector<SyncCoverMechanismId> normalized = selected;
  std::sort(normalized.begin(), normalized.end());
  normalized.erase(std::unique(normalized.begin(), normalized.end()),
                   normalized.end());
  SyncCoverDemandSet result(demands.size());
  for (const SyncCoverGroundedColumn &column : columns) {
    if (std::includes(normalized.begin(), normalized.end(),
                      column.members.begin(), column.members.end())) {
      result.unite(column.coverage);
    }
  }
  return result;
}

bool SyncCoverGroundedInstance::coversAll(
    const std::vector<SyncCoverMechanismId> &selected) const {
  return coveredBy(selected).count() == demands.size();
}

SyncCoverGroundingResult mlir::pto::groundSyncCoverInstance(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverGroundingOptions &options) {
  return groundSyncCoverInstance(universe, activeDemands, {}, options);
}

SyncCoverGroundingResult mlir::pto::groundSyncCoverInstance(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<std::vector<SyncCoverMechanismId>> &selectionColumns,
    const SyncCoverGroundingOptions &options) {
  SyncCoverGroundingResult result;
  if (options.maximumMembers == 0 || options.maximumMembers > 2 ||
      options.maximumPricingMembers < options.maximumMembers ||
      options.maximumPricingMembers > 4 ||
      options.maximumLabelsPerState == 0) {
    result.error = SyncCoverGroundingError::InvalidOptions;
    return result;
  }
  if (!universe.validate()) {
    result.error = SyncCoverGroundingError::InvalidUniverse;
    return result;
  }
  const bool invalidDemandOrder = !canonicalIdentitySet(activeDemands);
  const bool invalidDemandId =
      !activeDemands.empty() &&
      activeDemands.back() >= universe.getGraph().getDemands().size();
  if (invalidDemandOrder || invalidDemandId) {
    result.error = SyncCoverGroundingError::InvalidDemand;
    return result;
  }

  SyncCoverGroundedInstance &instance = result.instance;
  instance.universeVersion = universe.getVersion();
  instance.graphGeneration = universe.getGraph().getGeneration();
  instance.demands = activeDemands;
  instance.demandColumns.resize(activeDemands.size());
  result.error = snapshotMechanisms(universe, instance);
  if (result.error != SyncCoverGroundingError::None) {
    return result;
  }

  std::map<std::vector<SyncCoverMechanismId>, SyncCoverDemandSet> grounded;
  SyncCoverCoverageOracle topology(universe.getGraph());
  std::vector<SyncCoverMechanismId> allMechanisms(
      universe.getMechanisms().size());
  std::iota(allMechanisms.begin(), allMechanisms.end(), 0);
  for (std::size_t localDemand = 0; localDemand < activeDemands.size();
       ++localDemand) {
    const SyncCoverDemandId demand = activeDemands[localDemand];
    SyncCoverMinimalWitnessResult witnesses =
        topology.getMinimalMechanismWitnesses(demand,
                                              options.maximumMembers,
                                              options.maximumLabelsPerState);
    if (!witnesses) {
      result.error = SyncCoverGroundingError::CoverageFailure;
      result.failedDemand = demand;
      result.coverageError = witnesses.error;
      result.statistics = topology.getStatistics();
      return result;
    }
    const bool needsPricing =
        witnesses.witnesses.empty() || witnesses.truncated;
    instance.columnsTruncated |= witnesses.truncated;
    if (needsPricing &&
        options.maximumPricingMembers > options.maximumMembers) {
      SyncCoverMinimalWitnessResult priced =
          topology.getMinimalMechanismWitnesses(
              demand, options.maximumPricingMembers,
              options.maximumLabelsPerState);
      if (!priced) {
        result.error = SyncCoverGroundingError::CoverageFailure;
        result.failedDemand = demand;
        result.coverageError = priced.error;
        result.statistics = topology.getStatistics();
        return result;
      }
      instance.columnsTruncated |= priced.truncated;
      witnesses = std::move(priced);
    }
    const bool unresolved =
        witnesses.witnesses.empty() || witnesses.truncated;
    if (unresolved) {
      const SyncCoverCoverageResult fullUniverse =
          topology.checkDemandCanonicalSelection(demand, allMechanisms);
      if (!fullUniverse) {
        result.error = SyncCoverGroundingError::CoverageFailure;
        result.failedDemand = demand;
        result.coverageError = fullUniverse.error;
        result.statistics = topology.getStatistics();
        return result;
      }
      if (!fullUniverse.covered) {
        instance.provenUncoverableDemands.push_back(demand);
      } else if (witnesses.witnesses.empty() || witnesses.truncated) {
        instance.demandsNeedingPricing.push_back(demand);
      }
    }
    for (const std::vector<SyncCoverMechanismId> &members :
         witnesses.witnesses) {
      const bool invalidMembers =
          !canonicalIdentitySet(members) ||
          (!members.empty() &&
           members.back() >= universe.getMechanisms().size());
      if (invalidMembers) {
        result.error = SyncCoverGroundingError::InvalidMechanism;
        result.failedDemand = demand;
        result.statistics = topology.getStatistics();
        return result;
      }
      auto insertion = grounded.emplace(
          members, SyncCoverDemandSet(activeDemands.size()));
      insertion.first->second.add(localDemand);
    }
  }

  for (const std::vector<SyncCoverMechanismId> &selection : selectionColumns) {
    const bool invalidSelection =
        !canonicalIdentitySet(selection) ||
        (!selection.empty() &&
         selection.back() >= universe.getMechanisms().size());
    if (invalidSelection) {
      result.error = SyncCoverGroundingError::InvalidMechanism;
      result.statistics = topology.getStatistics();
      return result;
    }
    SyncCoverDemandSet selectionCoverage(activeDemands.size());
    for (const auto &[members, coverage] : grounded) {
      if (std::includes(selection.begin(), selection.end(), members.begin(),
                        members.end())) {
        selectionCoverage.unite(coverage);
      }
    }
    for (std::size_t localDemand = 0; localDemand < activeDemands.size();
         ++localDemand) {
      if (selectionCoverage.contains(localDemand)) {
        continue;
      }
      const SyncCoverCoverageResult coverage =
          topology.checkDemandCanonicalSelection(activeDemands[localDemand],
                                                 selection);
      if (!coverage) {
        result.error = SyncCoverGroundingError::CoverageFailure;
        result.failedDemand = activeDemands[localDemand];
        result.coverageError = coverage.error;
        result.statistics = topology.getStatistics();
        return result;
      }
      if (coverage.covered) {
        selectionCoverage.add(localDemand);
      }
    }
    auto insertion = grounded.emplace(
        selection, SyncCoverDemandSet(activeDemands.size()));
    insertion.first->second.unite(selectionCoverage);
  }

  instance.columns.reserve(grounded.size());
  for (auto &entry : grounded) {
    SyncCoverGroundedColumn column;
    column.id = instance.columns.size();
    column.members = entry.first;
    column.coverage = std::move(entry.second);
    for (std::size_t demand = 0; demand < activeDemands.size(); ++demand) {
      if (column.coverage.contains(demand)) {
        instance.demandColumns[demand].push_back(column.id);
      }
    }
    instance.columns.push_back(std::move(column));
  }
  result.statistics = topology.getStatistics();
  return result;
}
