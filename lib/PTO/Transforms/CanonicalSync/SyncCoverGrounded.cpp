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
    const std::vector<SyncCoverVerifiedFactoryColumn> &factoryColumns,
    const SyncCoverGroundingOptions &options) {
  SyncCoverGroundingResult result;
  if (options.maximumColumns == 0 || options.maximumPricingDemands == 0 ||
      options.maximumPairsPerDemand == 0) {
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
  std::map<SyncCoverDemandId, std::size_t> activeLocal;
  for (std::size_t localDemand = 0; localDemand < activeDemands.size();
       ++localDemand) {
    activeLocal.emplace(activeDemands[localDemand], localDemand);
  }

  // Construct every singleton incidence in one context-batched propagation.
  // This is opportunity grounding, not selected-plan search: each mechanism
  // is evaluated simultaneously and the result is cached as a set-cover
  // column before the solver starts.
  SyncCoverCoverageOracle incidence(universe.getGraph());
  const auto singletonWitnesses =
      incidence.getSingletonMechanismWitnessesForDemands(
          activeDemands, universe.getMechanisms().size());
  for (std::size_t localDemand = 0; localDemand < singletonWitnesses.size();
       ++localDemand) {
    if (!singletonWitnesses[localDemand]) {
      result.error = SyncCoverGroundingError::CoverageFailure;
      result.failedDemand = activeDemands[localDemand];
      result.coverageError = singletonWitnesses[localDemand].error;
      result.statistics = incidence.getStatistics();
      return result;
    }
    for (SyncCoverMechanismId mechanism :
         singletonWitnesses[localDemand].mechanisms) {
      auto insertion = grounded.emplace(
          std::vector<SyncCoverMechanismId>{mechanism},
          SyncCoverDemandSet(activeDemands.size()));
      insertion.first->second.add(localDemand);
    }
  }

  std::vector<SyncCoverVerifiedFactoryColumn> candidates;
  candidates.reserve(factoryColumns.size());
  for (const SyncCoverVerifiedFactoryColumn &column : factoryColumns) {
    if (column.members.size() > 1 || !column.demands.empty()) {
      candidates.push_back(column);
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto &first,
                                                     const auto &second) {
    return std::tie(first.members, first.demands) <
           std::tie(second.members, second.demands);
  });
  candidates.erase(
      std::unique(candidates.begin(), candidates.end(), [](const auto &first,
                                                           const auto &second) {
        return first.members == second.members && first.demands == second.demands;
      }),
      candidates.end());

  for (const SyncCoverVerifiedFactoryColumn &column : candidates) {
    if (column.members.empty() ||
        column.members.back() >= universe.getMechanisms().size() ||
        !canonicalIdentitySet(column.members) ||
        !canonicalIdentitySet(column.demands)) {
      result.error = SyncCoverGroundingError::InvalidMechanism;
      return result;
    }
    SyncCoverDemandSet coverage(activeDemands.size());
    for (SyncCoverMechanismId member : column.members) {
      auto singleton = grounded.find({member});
      if (singleton != grounded.end()) {
        coverage.unite(singleton->second);
      }
    }
    const std::vector<SyncCoverSelectionWitnessResult> verifiedCoverage =
        incidence.getSelectionWitnessesForDemands(
            column.demands, {column.members},
            universe.getMechanisms().size());
    if (verifiedCoverage.size() != column.demands.size()) {
      result.error = SyncCoverGroundingError::CoverageFailure;
      return result;
    }
    for (std::size_t index = 0; index < column.demands.size(); ++index) {
      const SyncCoverDemandId demand = column.demands[index];
      auto local = activeLocal.find(demand);
      if (local == activeLocal.end()) {
        result.error = SyncCoverGroundingError::InvalidDemand;
        result.failedDemand = demand;
        return result;
      }
      if (verifiedCoverage[index].error != SyncCoverCoverageError::None) {
        result.error = SyncCoverGroundingError::CoverageFailure;
        result.failedDemand = demand;
        result.coverageError = verifiedCoverage[index].error;
        result.statistics = incidence.getStatistics();
        return result;
      }
      if (!verifiedCoverage[index].selections.empty()) {
        coverage.add(local->second);
      }
    }
    if (coverage.count() != 0) {
      auto insertion = grounded.emplace(
          column.members, SyncCoverDemandSet(activeDemands.size()));
      insertion.first->second.unite(coverage);
    }
  }

  SyncCoverDemandSet fineGrainedCovered(activeDemands.size());
  for (const auto &[members, coverage] : grounded) {
    const bool usesBarrier = std::any_of(
        members.begin(), members.end(), [&](SyncCoverMechanismId mechanism) {
          return universe.getMechanisms()[mechanism].kind ==
                 SyncCoverMechanismKind::Barrier;
        });
    if (!usesBarrier) {
      fineGrainedCovered.unite(coverage);
    }
  }
  std::size_t pricedDemands = 0;
  for (std::size_t localDemand = 0; localDemand < activeDemands.size();
       ++localDemand) {
    if (fineGrainedCovered.contains(localDemand)) {
      instance.pricingRestricted = true;
      continue;
    }
    if (pricedDemands == options.maximumPricingDemands) {
      instance.columnsTruncated = true;
      break;
    }
    ++pricedDemands;
    const SyncCoverFactoryWitnessResult priced =
        incidence.getFactoryMechanismWitnesses(
            activeDemands[localDemand], universe.getMechanisms().size(),
            options.maximumPairsPerDemand);
    if (!priced) {
      result.error = SyncCoverGroundingError::CoverageFailure;
      result.failedDemand = activeDemands[localDemand];
      result.coverageError = priced.error;
      result.statistics = incidence.getStatistics();
      return result;
    }
    instance.columnsTruncated |= priced.truncated;
    const std::size_t pairCount =
        std::min(priced.pairs.size(), options.maximumPairsPerDemand);
    instance.columnsTruncated |= pairCount != priced.pairs.size();
    for (std::size_t pairIndex = 0; pairIndex < pairCount; ++pairIndex) {
      const std::vector<SyncCoverMechanismId> &pair =
          priced.pairs[pairIndex];
      const bool usesBarrier = std::any_of(
          pair.begin(), pair.end(), [&](SyncCoverMechanismId mechanism) {
            return universe.getMechanisms()[mechanism].kind ==
                   SyncCoverMechanismKind::Barrier;
          });
      if (usesBarrier) {
        continue;
      }
      auto insertion = grounded.emplace(
          pair, SyncCoverDemandSet(activeDemands.size()));
      insertion.first->second.add(localDemand);
    }
  }

  if (grounded.size() > options.maximumColumns) {
    instance.columnsTruncated = true;
  }
  instance.columns.reserve(std::min(grounded.size(), options.maximumColumns));
  for (auto &entry : grounded) {
    if (instance.columns.size() == options.maximumColumns) {
      break;
    }
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
  for (std::size_t localDemand = 0; localDemand < activeDemands.size();
       ++localDemand) {
    if (instance.demandColumns[localDemand].empty()) {
      instance.demandsNeedingPricing.push_back(activeDemands[localDemand]);
    }
  }
  result.statistics = incidence.getStatistics();
  return result;
}
