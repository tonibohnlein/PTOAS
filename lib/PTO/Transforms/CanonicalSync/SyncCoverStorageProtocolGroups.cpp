// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolGroups.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

using ResourcePair = std::pair<std::uint32_t, std::uint32_t>;
using GroupKey = std::tuple<SyncCoverScopeId, SyncCoverStorageProtocolBehavior,
                            std::uint32_t, std::uint32_t, std::size_t>;
using SlotKey =
    std::tuple<SyncCoverStorageDomainId, std::uint64_t, std::uint64_t>;
using EdgeRef = std::pair<SyncCoverStorageLifecycleComponentId,
                          SyncCoverStorageLifecycleEdgeId>;

struct ClassifiedSeed {
  SyncCoverStorageProtocolSeedId seed = 0;
  GroupKey key;
  std::vector<SyncCoverControlId> periodicControls;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SlotKey> slots;
  unsigned maximumDistance = 0;
};

struct PendingGroup {
  GroupKey key;
  std::vector<SyncCoverStorageProtocolSeedId> seeds;
  std::set<SyncCoverControlId> periodicControls;
  std::vector<SyncCoverDemandId> demands;
  std::vector<SlotKey> slots;
  unsigned maximumDistance = 0;
};

enum class JointOrbitError : std::uint8_t {
  None,
  InvalidGraph,
  UnsupportedScope,
  LimitExceeded,
};

struct JointOrbit {
  JointOrbitError error = JointOrbitError::None;
  std::vector<std::vector<std::size_t>> phaseStates;
};

enum class SlotOverlapResult : std::uint8_t {
  Disjoint,
  Overlap,
  LimitExceeded,
};

class WorkBudget {
public:
  WorkBudget(std::size_t maximum, std::size_t &used)
      : maximum_(maximum), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (used_ > maximum_ || amount > maximum_ - used_) {
      return false;
    }
    used_ += amount;
    return true;
  }

private:
  std::size_t maximum_ = 0;
  std::size_t &used_;
};

bool consumeOrderedOperation(WorkBudget &budget, std::size_t elementCount) {
  return budget.consume(elementCount) && budget.consume();
}

bool consumeSortWork(WorkBudget &budget, std::size_t elementCount) {
  if (elementCount < 2) {
    return true;
  }
  std::size_t levels = 0;
  for (std::size_t covered = 1; covered < elementCount;) {
    ++levels;
    const bool coveredWouldOverflow =
        covered > std::numeric_limits<std::size_t>::max() / 2;
    if (coveredWouldOverflow) {
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t operationsPerLevel = 4;
  const bool levelWorkWouldOverflow =
      levels > std::numeric_limits<std::size_t>::max() / operationsPerLevel;
  const bool totalWorkWouldOverflow =
      !levelWorkWouldOverflow &&
      elementCount > std::numeric_limits<std::size_t>::max() /
                         (levels * operationsPerLevel);
  if (levelWorkWouldOverflow || totalWorkWouldOverflow) {
    return false;
  }
  return budget.consume(elementCount * levels * operationsPerLevel);
}

bool intervalOverlaps(const SlotKey &first, const SlotKey &second) {
  const bool differentDomain = std::get<0>(first) != std::get<0>(second);
  if (differentDomain) {
    return false;
  }
  return std::get<1>(first) < std::get<2>(second) &&
         std::get<1>(second) < std::get<2>(first);
}

JointOrbit
buildJointOrbit(const SyncCoverGraph &graph, SyncCoverScopeId owningScope,
                const std::vector<SyncCoverControlId> &periodicControls,
                const std::vector<SyncCoverControl> &controls,
                const SyncCoverStorageProtocolGroupLimits &limits,
                WorkBudget &budget) {
  JointOrbit result;
  const bool controlIncidenceLimitReached =
      periodicControls.size() > limits.maximumJointStateIncidences;
  const bool controlScratchWorkAvailable =
      controlIncidenceLimitReached || budget.consume(periodicControls.size());
  if (controlIncidenceLimitReached || !controlScratchWorkAvailable) {
    result.error = JointOrbitError::LimitExceeded;
    return result;
  }
  std::vector<std::size_t> state;
  state.reserve(periodicControls.size());
  std::optional<SyncCoverScopeId> phaseScope;
  for (SyncCoverControlId controlId : periodicControls) {
    const bool invalidControl = controlId >= controls.size();
    const bool controlWorkAvailable = invalidControl || budget.consume();
    if (invalidControl || !controlWorkAvailable) {
      result.error = invalidControl ? JointOrbitError::InvalidGraph
                                    : JointOrbitError::LimitExceeded;
      return result;
    }
    const SyncCoverControl &control = controls[controlId];
    if (!control.phaseRelation) {
      result.error = JointOrbitError::InvalidGraph;
      return result;
    }
    const SyncCoverScopeId controlPhaseScope = control.phaseRelation->loopScope;
    if (phaseScope) {
      if (*phaseScope != controlPhaseScope) {
        result.error = JointOrbitError::UnsupportedScope;
        return result;
      }
    } else {
      const bool ancestryWorkAvailable =
          budget.consume(graph.getScopes().size());
      if (!ancestryWorkAvailable) {
        result.error = JointOrbitError::LimitExceeded;
        return result;
      }
      if (!graph.scopeContains(owningScope, controlPhaseScope)) {
        result.error = JointOrbitError::UnsupportedScope;
        return result;
      }
      phaseScope = controlPhaseScope;
    }
    const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
    const std::size_t phases = relation.nextPhase.size();
    const bool invalidRelation = phases == 0 ||
                                 relation.activeAlternative.size() != phases ||
                                 relation.initialPhase >= phases;
    if (invalidRelation || !budget.consume(phases)) {
      result.error = invalidRelation ? JointOrbitError::InvalidGraph
                                     : JointOrbitError::LimitExceeded;
      return result;
    }
    for (std::size_t phase = 0; phase < phases; ++phase) {
      const bool invalidPhase =
          relation.nextPhase[phase] >= phases ||
          relation.activeAlternative[phase] >= control.alternatives;
      if (invalidPhase) {
        result.error = JointOrbitError::InvalidGraph;
        return result;
      }
    }
    state.push_back(relation.initialPhase);
  }

  while (true) {
    for (std::size_t prior = 0; prior < result.phaseStates.size(); ++prior) {
      const bool comparisonWorkAvailable =
          budget.consume(periodicControls.size() + 1);
      if (!comparisonWorkAvailable) {
        result.error = JointOrbitError::LimitExceeded;
        return result;
      }
      if (result.phaseStates[prior] == state) {
        if (prior != 0) {
          result.error = JointOrbitError::InvalidGraph;
        }
        return result;
      }
    }
    const bool phaseLimitReached =
        result.phaseStates.size() >= limits.maximumReachablePhases;
    const bool incidenceLimitReached =
        periodicControls.size() > limits.maximumJointStateIncidences ||
        result.phaseStates.size() >
            (limits.maximumJointStateIncidences - periodicControls.size()) /
                std::max<std::size_t>(periodicControls.size(), 1);
    if (phaseLimitReached || incidenceLimitReached ||
        !budget.consume(periodicControls.size())) {
      result.error = JointOrbitError::LimitExceeded;
      return result;
    }
    result.phaseStates.push_back(state);
    for (std::size_t controlIndex = 0; controlIndex < periodicControls.size();
         ++controlIndex) {
      const SyncCoverControlPhaseRelation &relation =
          *controls[periodicControls[controlIndex]].phaseRelation;
      state[controlIndex] = relation.nextPhase[state[controlIndex]];
    }
    if (periodicControls.empty()) {
      return result;
    }
  }
}

bool guardMatchesJointState(
    const SyncCoverGuard &guard,
    const std::vector<SyncCoverControlId> &periodicControls,
    const std::vector<std::size_t> &phaseState,
    const std::vector<SyncCoverControl> &controls) {
  std::size_t periodicControlIndex = 0;
  for (const SyncCoverGuardLiteral &literal : guard.literals) {
    while (true) {
      const bool controlsExhausted =
          periodicControlIndex >= periodicControls.size();
      if (controlsExhausted) {
        break;
      }
      const bool reachedLiteral =
          periodicControls[periodicControlIndex] >= literal.control;
      if (reachedLiteral) {
        break;
      }
      ++periodicControlIndex;
    }
    const bool controlIsPeriodic =
        periodicControlIndex < periodicControls.size() &&
        periodicControls[periodicControlIndex] == literal.control;
    if (!controlIsPeriodic) {
      continue;
    }
    const SyncCoverControlPhaseRelation &relation =
        *controls[literal.control].phaseRelation;
    if (relation.activeAlternative[phaseState[periodicControlIndex]] !=
        literal.alternative) {
      return false;
    }
  }
  return true;
}

bool consumeGuardMatchWork(WorkBudget &budget, const SyncCoverGuard &guard,
                           std::size_t periodicControlCount) {
  // The normalized guard and periodic-control list are merged linearly.
  return budget.consume(guard.literals.size()) &&
         budget.consume(periodicControlCount);
}

std::uint64_t rotateMask(std::uint64_t mask, std::size_t period,
                         std::size_t shift) {
  const std::uint64_t periodMask =
      period == 64 ? std::numeric_limits<std::uint64_t>::max()
                   : (std::uint64_t{1} << period) - 1;
  if (shift == 0 || period <= 1) {
    return mask & periodMask;
  }
  return ((mask >> shift) | (mask << (period - shift))) & periodMask;
}

std::optional<std::vector<std::uint64_t>>
canonicalizeReadinessMasks(const std::vector<std::uint64_t> &readinessMasks,
                           std::size_t period, WorkBudget &budget) {
  std::vector<std::uint64_t> best;
  for (std::size_t shift = 0; shift < period; ++shift) {
    if (!budget.consume(readinessMasks.size())) {
      return std::nullopt;
    }
    std::vector<std::uint64_t> candidate;
    candidate.reserve(readinessMasks.size());
    for (std::uint64_t mask : readinessMasks) {
      candidate.push_back(rotateMask(mask, period, shift));
    }
    if (!consumeSortWork(budget, candidate.size())) {
      return std::nullopt;
    }
    std::sort(candidate.begin(), candidate.end());
    if (shift == 0) {
      best = std::move(candidate);
      continue;
    }
    if (!budget.consume(candidate.size())) {
      return std::nullopt;
    }
    if (candidate < best) {
      best = std::move(candidate);
    }
  }
  return best;
}

std::optional<std::size_t>
internBehaviorSignature(std::vector<std::vector<std::uint64_t>> &signatures,
                        const std::vector<std::uint64_t> &candidate,
                        WorkBudget &budget) {
  for (std::size_t signatureId = 0; signatureId < signatures.size();
       ++signatureId) {
    const std::vector<std::uint64_t> &existing = signatures[signatureId];
    if (!budget.consume()) {
      return std::nullopt;
    }
    const bool differentSize = existing.size() != candidate.size();
    if (differentSize) {
      continue;
    }
    bool equal = true;
    for (std::size_t word = 0; word < candidate.size(); ++word) {
      if (!budget.consume()) {
        return std::nullopt;
      }
      if (existing[word] != candidate[word]) {
        equal = false;
        break;
      }
    }
    if (equal) {
      return signatureId;
    }
  }
  if (!budget.consume(candidate.size())) {
    return std::nullopt;
  }
  signatures.push_back(candidate);
  return signatures.size() - 1;
}

SlotOverlapResult seedSlotsOverlap(const ClassifiedSeed &seed,
                                   const PendingGroup &group,
                                   WorkBudget &budget) {
  for (const SlotKey &candidate : seed.slots) {
    for (const SlotKey &existing : group.slots) {
      if (!budget.consume()) {
        return SlotOverlapResult::LimitExceeded;
      }
      if (intervalOverlaps(candidate, existing)) {
        return SlotOverlapResult::Overlap;
      }
    }
  }
  return SlotOverlapResult::Disjoint;
}

SyncCoverStorageLifecycleEdgeKindMask readyBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Ready);
}

SyncCoverStorageLifecycleEdgeKindMask releaseBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Release);
}

} // namespace

SyncCoverStorageProtocolGroupPrefix
mlir::pto::boundedSyncCoverStorageProtocolGroupPrefix(
    const std::vector<SyncCoverStorageProtocolGroup> &groups,
    std::size_t maximumGroups, std::size_t maximumSeedIncidences,
    std::size_t maximumBehaviorSignatureEntries) {
  SyncCoverStorageProtocolGroupPrefix result;
  for (const SyncCoverStorageProtocolGroup &group : groups) {
    const bool groupLimitReached = result.retainedGroups >= maximumGroups;
    const bool seedLimitReached =
        result.retainedSeedIncidences > maximumSeedIncidences ||
        group.seeds.size() >
            maximumSeedIncidences - result.retainedSeedIncidences;
    const bool signatureLimitReached =
        result.retainedBehaviorSignatureEntries >
            maximumBehaviorSignatureEntries ||
        group.behaviorSignature.size() >
            maximumBehaviorSignatureEntries -
                result.retainedBehaviorSignatureEntries;
    if (groupLimitReached || seedLimitReached || signatureLimitReached) {
      result.truncated = true;
      return result;
    }
    ++result.retainedGroups;
    result.retainedSeedIncidences += group.seeds.size();
    result.retainedBehaviorSignatureEntries += group.behaviorSignature.size();
  }
  return result;
}

SyncCoverStorageProtocolGroupIndex
mlir::pto::buildSyncCoverStorageProtocolGroupIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupLimits &limits) {
  SyncCoverStorageProtocolGroupIndex result;
  const auto fail = [&](SyncCoverStorageProtocolGroupStatistics statistics,
                        SyncCoverStorageProtocolGroupError error) {
    result.groups_.clear();
    statistics.groups = 0;
    statistics.seedIncidences = 0;
    statistics.controlIncidences = 0;
    statistics.demandIncidences = 0;
    statistics.slotIncidences = 0;
    statistics.jointStateIncidences = 0;
    statistics.maximumGroupSeeds = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolGroupError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumGroups == 0 ||
      limits.maximumSeedIncidences == 0 ||
      limits.maximumControlIncidences == 0 ||
      limits.maximumDemandIncidences == 0 ||
      limits.maximumSlotIncidences == 0 ||
      limits.maximumJointStateIncidences == 0 ||
      limits.maximumReachablePhases == 0 || limits.maximumReachablePhases > 64;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolGroupError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolGroupError::InvalidGraph);
  }
  if (!lifecycleIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolGroupError::IncompleteLifecycleIndex);
  }
  if (!seedIndex.isComplete()) {
    return fail({}, SyncCoverStorageProtocolGroupError::IncompleteSeedIndex);
  }

  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverStorageProtocolSeed> &seeds = seedIndex.getSeeds();
  const std::vector<SyncCoverControl> &controls = graph.getControls();
  const std::vector<SyncCoverDemand> &graphDemands = graph.getDemands();
  SyncCoverStorageProtocolGroupStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  const bool seedPreflightLimitReached =
      seeds.size() > limits.maximumSeedIncidences;
  if (seedPreflightLimitReached) {
    return fail(statistics, SyncCoverStorageProtocolGroupError::LimitExceeded);
  }
  if (!budget.consume(seeds.size())) {
    return fail(statistics, SyncCoverStorageProtocolGroupError::LimitExceeded);
  }

  std::vector<ClassifiedSeed> classified;
  std::vector<std::vector<std::uint64_t>> behaviorSignatures;
  classified.reserve(seeds.size());
  for (const SyncCoverStorageProtocolSeed &seed : seeds) {
    if (!budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    const std::size_t seedIndexPosition =
        static_cast<std::size_t>(&seed - seeds.data());
    const bool invalidSeedId =
        seed.id >= seeds.size() || seed.id != seedIndexPosition;
    if (invalidSeedId) {
      return fail(statistics, SyncCoverStorageProtocolGroupError::InvalidGraph);
    }
    if (seed.readyReleaseSccs.empty()) {
      ++statistics.ineligibleSeeds;
      continue;
    }

    std::vector<ResourcePair> readyPairs;
    std::vector<ResourcePair> releasePairs;
    std::vector<SyncCoverControlId> periodicControls;
    std::vector<SyncCoverDemandId> demands;
    std::vector<EdgeRef> relevantEdges;
    std::vector<std::vector<EdgeRef>> edgesByScc;
    if (!budget.consume(seed.readyReleaseSccs.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    edgesByScc.reserve(seed.readyReleaseSccs.size());
    bool invalidSeed = false;
    for (const SyncCoverStorageProtocolSccRef &reference :
         seed.readyReleaseSccs) {
      const bool invalidReference =
          reference.component >= components.size() ||
          reference.scc >= components[reference.component].sccs.size();
      if (invalidReference) {
        invalidSeed = true;
        break;
      }
      const SyncCoverStorageLifecycleComponent &component =
          components[reference.component];
      const SyncCoverStorageLifecycleScc &scc = component.sccs[reference.scc];
      std::vector<EdgeRef> sccEdges;
      if (!budget.consume(scc.internalEdges.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      sccEdges.reserve(scc.internalEdges.size());
      for (SyncCoverStorageLifecycleEdgeId edge : scc.internalEdges) {
        if (!budget.consume()) {
          return fail(statistics,
                      SyncCoverStorageProtocolGroupError::LimitExceeded);
        }
        if (edge >= component.edges.size()) {
          invalidSeed = true;
          break;
        }
        const EdgeRef edgeReference{reference.component, edge};
        relevantEdges.push_back(edgeReference);
        sccEdges.push_back(edgeReference);
      }
      if (invalidSeed) {
        break;
      }
      edgesByScc.push_back(std::move(sccEdges));
    }
    if (invalidSeed) {
      return fail(statistics, SyncCoverStorageProtocolGroupError::InvalidGraph);
    }
    if (!consumeSortWork(budget, relevantEdges.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    std::sort(relevantEdges.begin(), relevantEdges.end());
    relevantEdges.erase(std::unique(relevantEdges.begin(), relevantEdges.end()),
                        relevantEdges.end());

    for (const auto &[componentId, edgeId] : relevantEdges) {
      const SyncCoverStorageLifecycleComponent &component =
          components[componentId];
      const SyncCoverStorageLifecycleEdge &edge = component.edges[edgeId];
      const bool invalidEdge = edge.source >= component.epochs.size() ||
                               edge.target >= component.epochs.size() ||
                               edge.demand >= graphDemands.size();
      if (invalidEdge) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleEpoch &source =
          component.epochs[edge.source];
      const SyncCoverStorageLifecycleEpoch &target =
          component.epochs[edge.target];
      const bool crossResourceReady =
          (edge.kinds & readyBit()) != 0 && source.resource != target.resource;
      if (crossResourceReady) {
        readyPairs.push_back({source.resource, target.resource});
      }
      const bool crossResourceRelease = (edge.kinds & releaseBit()) != 0 &&
                                        source.resource != target.resource;
      if (crossResourceRelease) {
        releasePairs.push_back({source.resource, target.resource});
      }
      demands.push_back(edge.demand);
      const SyncCoverDemand &demand = graphDemands[edge.demand];
      const auto recordPeriodicControl = [&](const SyncCoverGuard &guard) {
        for (const SyncCoverGuardLiteral &literal : guard.literals) {
          if (!budget.consume()) {
            return false;
          }
          if (literal.control >= controls.size()) {
            invalidSeed = true;
            return false;
          }
          if (!controls[literal.control].phaseRelation) {
            continue;
          }
          periodicControls.push_back(literal.control);
        }
        return true;
      };
      const bool sourceControlsRecorded =
          recordPeriodicControl(demand.sourceGuard);
      const bool targetControlsRecorded =
          sourceControlsRecorded && recordPeriodicControl(demand.targetGuard);
      if (!sourceControlsRecorded || !targetControlsRecorded) {
        if (invalidSeed) {
          return fail(statistics,
                      SyncCoverStorageProtocolGroupError::InvalidGraph);
        }
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
    }

    const bool sortWorkAvailable =
        consumeSortWork(budget, readyPairs.size()) &&
        consumeSortWork(budget, releasePairs.size()) &&
        consumeSortWork(budget, periodicControls.size()) &&
        consumeSortWork(budget, demands.size());
    if (!sortWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    std::sort(readyPairs.begin(), readyPairs.end());
    readyPairs.erase(std::unique(readyPairs.begin(), readyPairs.end()),
                     readyPairs.end());
    std::sort(releasePairs.begin(), releasePairs.end());
    releasePairs.erase(std::unique(releasePairs.begin(), releasePairs.end()),
                       releasePairs.end());
    std::sort(periodicControls.begin(), periodicControls.end());
    periodicControls.erase(
        std::unique(periodicControls.begin(), periodicControls.end()),
        periodicControls.end());
    std::sort(demands.begin(), demands.end());
    demands.erase(std::unique(demands.begin(), demands.end()), demands.end());
    const bool hasResourceCycle =
        readyPairs.size() == 1 && releasePairs.size() == 1 &&
        releasePairs.front().first == readyPairs.front().second &&
        releasePairs.front().second == readyPairs.front().first;
    if (!hasResourceCycle) {
      ++statistics.ineligibleSeeds;
      continue;
    }
    const ResourcePair readyPair = readyPairs.front();
    const JointOrbit orbit = buildJointOrbit(
        graph, seed.owningScope, periodicControls, controls, limits, budget);
    if (orbit.error == JointOrbitError::UnsupportedScope) {
      ++statistics.ineligibleSeeds;
      continue;
    }
    if (orbit.error != JointOrbitError::None) {
      const SyncCoverStorageProtocolGroupError error =
          orbit.error == JointOrbitError::InvalidGraph
              ? SyncCoverStorageProtocolGroupError::InvalidGraph
              : SyncCoverStorageProtocolGroupError::LimitExceeded;
      return fail(statistics, error);
    }
    const std::size_t orbitIncidences =
        orbit.phaseStates.size() * periodicControls.size();
    const bool orbitIncidenceLimitReached =
        statistics.jointStateIncidences > limits.maximumJointStateIncidences ||
        orbitIncidences > limits.maximumJointStateIncidences -
                              statistics.jointStateIncidences;
    if (orbitIncidenceLimitReached) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    statistics.jointStateIncidences += orbitIncidences;

    bool stable = true;
    std::vector<std::uint64_t> readinessMasks;
    readinessMasks.reserve(edgesByScc.size());
    for (const std::vector<EdgeRef> &sccEdges : edgesByScc) {
      std::uint64_t readinessMask = 0;
      for (std::size_t phase = 0; phase < orbit.phaseStates.size(); ++phase) {
        bool hasDirectReady = false;
        for (const auto &[componentId, edgeId] : sccEdges) {
          if (!budget.consume()) {
            return fail(statistics,
                        SyncCoverStorageProtocolGroupError::LimitExceeded);
          }
          const SyncCoverStorageLifecycleComponent &component =
              components[componentId];
          const SyncCoverStorageLifecycleEdge &edge = component.edges[edgeId];
          const SyncCoverStorageLifecycleEpoch &source =
              component.epochs[edge.source];
          const SyncCoverStorageLifecycleEpoch &target =
              component.epochs[edge.target];
          const bool unsuitableReady =
              edge.distance != 0 || (edge.kinds & readyBit()) == 0 ||
              ResourcePair{source.resource, target.resource} != readyPair;
          if (unsuitableReady) {
            continue;
          }
          const SyncCoverDemand &demand = graphDemands[edge.demand];
          const bool guardWorkAvailable =
              consumeGuardMatchWork(budget, demand.sourceGuard,
                                    periodicControls.size()) &&
              consumeGuardMatchWork(budget, demand.targetGuard,
                                    periodicControls.size());
          if (!guardWorkAvailable) {
            return fail(statistics,
                        SyncCoverStorageProtocolGroupError::LimitExceeded);
          }
          hasDirectReady =
              guardMatchesJointState(demand.sourceGuard, periodicControls,
                                     orbit.phaseStates[phase], controls) &&
              guardMatchesJointState(demand.targetGuard, periodicControls,
                                     orbit.phaseStates[phase], controls);
          if (hasDirectReady) {
            break;
          }
        }
        if (hasDirectReady) {
          readinessMask |= std::uint64_t{1} << phase;
        } else {
          stable = false;
        }
      }
      readinessMasks.push_back(readinessMask);
    }
    const std::optional<std::vector<std::uint64_t>> canonicalMasks =
        canonicalizeReadinessMasks(readinessMasks, orbit.phaseStates.size(),
                                   budget);
    if (!canonicalMasks) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    std::vector<std::uint64_t> behaviorSignature;
    const std::size_t signatureSize =
        3 + periodicControls.size() + canonicalMasks->size();
    if (!budget.consume(signatureSize)) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    behaviorSignature.reserve(signatureSize);
    behaviorSignature.push_back(orbit.phaseStates.size());
    behaviorSignature.push_back(periodicControls.size());
    for (SyncCoverControlId control : periodicControls) {
      behaviorSignature.push_back(control);
    }
    behaviorSignature.push_back(readinessMasks.size());
    behaviorSignature.insert(behaviorSignature.end(), canonicalMasks->begin(),
                             canonicalMasks->end());

    const std::optional<std::size_t> behaviorSignatureId =
        internBehaviorSignature(behaviorSignatures, behaviorSignature, budget);
    if (!behaviorSignatureId) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }

    const bool slotScratchLimitReached =
        seed.slots.size() > limits.maximumSlotIncidences ||
        statistics.slotIncidences >
            limits.maximumSlotIncidences - seed.slots.size();
    if (slotScratchLimitReached || !budget.consume(seed.slots.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    ClassifiedSeed description;
    description.seed = seed.id;
    const SyncCoverStorageProtocolBehavior behavior =
        stable ? SyncCoverStorageProtocolBehavior::StableRoundTrip
               : SyncCoverStorageProtocolBehavior::PhaseRotatingRoundTrip;
    description.key = {seed.owningScope, behavior, readyPair.first,
                       readyPair.second, *behaviorSignatureId};
    description.periodicControls = std::move(periodicControls);
    description.demands = std::move(demands);
    description.maximumDistance = seed.maximumDistance;
    description.slots.reserve(seed.slots.size());
    for (const SyncCoverStorageProtocolSlotRef &reference : seed.slots) {
      const bool invalidReference =
          reference.component >= components.size() ||
          reference.slot >= components[reference.component].slots.size();
      if (invalidReference) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleSlot &slot =
          components[reference.component].slots[reference.slot];
      description.slots.push_back(
          {slot.domain, slot.extent.begin, slot.extent.end});
    }
    if (!consumeSortWork(budget, description.slots.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    std::sort(description.slots.begin(), description.slots.end());
    description.slots.erase(
        std::unique(description.slots.begin(), description.slots.end()),
        description.slots.end());
    bool hasInternalSlotOverlap = false;
    for (std::size_t slot = 1; slot < description.slots.size(); ++slot) {
      if (!budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      if (intervalOverlaps(description.slots[slot - 1],
                           description.slots[slot])) {
        hasInternalSlotOverlap = true;
        break;
      }
    }
    if (hasInternalSlotOverlap) {
      ++statistics.ineligibleSeeds;
      continue;
    }
    statistics.slotIncidences += seed.slots.size();
    classified.push_back(std::move(description));
    ++statistics.eligibleSeeds;
    if (stable) {
      ++statistics.stableSeeds;
    } else {
      ++statistics.phaseRotatingSeeds;
    }
  }

  std::map<GroupKey, std::vector<std::size_t>> groupsByKey;
  std::vector<PendingGroup> pendingGroups;
  std::size_t pendingControlEntries = 0;
  std::size_t pendingDemandEntries = 0;
  std::size_t pendingSlotEntries = 0;
  for (const ClassifiedSeed &seed : classified) {
    if (!consumeOrderedOperation(budget, groupsByKey.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    auto keyPosition = groupsByKey.find(seed.key);
    if (keyPosition == groupsByKey.end()) {
      const bool mapInsertionWorkAvailable =
          consumeOrderedOperation(budget, groupsByKey.size());
      if (!mapInsertionWorkAvailable) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      keyPosition = groupsByKey.try_emplace(seed.key).first;
    }
    std::optional<std::size_t> destination;
    for (std::size_t group : keyPosition->second) {
      const SlotOverlapResult overlap =
          seedSlotsOverlap(seed, pendingGroups[group], budget);
      if (overlap == SlotOverlapResult::LimitExceeded) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      if (overlap == SlotOverlapResult::Disjoint) {
        destination = group;
        break;
      }
    }
    if (!destination) {
      const bool groupLimitReached =
          pendingGroups.size() >= limits.maximumGroups;
      if (groupLimitReached) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      const bool groupCreationWorkAvailable = budget.consume();
      if (!groupCreationWorkAvailable) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      destination = pendingGroups.size();
      pendingGroups.push_back({seed.key, {}, {}, {}, {}, 0});
      keyPosition->second.push_back(*destination);
    }
    PendingGroup &group = pendingGroups[*destination];
    if (statistics.seedIncidences >= limits.maximumSeedIncidences ||
        !budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    group.seeds.push_back(seed.seed);
    ++statistics.seedIncidences;
    const bool groupSlotLimitReached =
        pendingSlotEntries > limits.maximumSlotIncidences ||
        seed.slots.size() > limits.maximumSlotIncidences - pendingSlotEntries;
    if (groupSlotLimitReached || !budget.consume(seed.slots.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    pendingSlotEntries += seed.slots.size();
    group.slots.insert(group.slots.end(), seed.slots.begin(), seed.slots.end());
    group.maximumDistance =
        std::max(group.maximumDistance, seed.maximumDistance);
    for (SyncCoverControlId control : seed.periodicControls) {
      if (!consumeOrderedOperation(budget, group.periodicControls.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      const bool isNewControl =
          group.periodicControls.find(control) == group.periodicControls.end();
      if (!isNewControl) {
        continue;
      }
      if (pendingControlEntries >= limits.maximumControlIncidences ||
          !consumeOrderedOperation(budget, group.periodicControls.size())) {
        return fail(statistics,
                    SyncCoverStorageProtocolGroupError::LimitExceeded);
      }
      group.periodicControls.insert(control);
      ++pendingControlEntries;
    }
    if (pendingDemandEntries > limits.maximumDemandIncidences ||
        seed.demands.size() >
            limits.maximumDemandIncidences - pendingDemandEntries ||
        !budget.consume(seed.demands.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    pendingDemandEntries += seed.demands.size();
    group.demands.insert(group.demands.end(), seed.demands.begin(),
                         seed.demands.end());
  }

  if (!budget.consume(pendingGroups.size())) {
    return fail(statistics, SyncCoverStorageProtocolGroupError::LimitExceeded);
  }
  result.groups_.reserve(pendingGroups.size());
  for (PendingGroup &pending : pendingGroups) {
    if (!consumeSortWork(budget, pending.demands.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    std::sort(pending.demands.begin(), pending.demands.end());
    pending.demands.erase(
        std::unique(pending.demands.begin(), pending.demands.end()),
        pending.demands.end());
    if (statistics.controlIncidences > limits.maximumControlIncidences ||
        pending.periodicControls.size() >
            limits.maximumControlIncidences - statistics.controlIncidences ||
        statistics.demandIncidences > limits.maximumDemandIncidences ||
        pending.demands.size() >
            limits.maximumDemandIncidences - statistics.demandIncidences) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    statistics.controlIncidences += pending.periodicControls.size();
    statistics.demandIncidences += pending.demands.size();
    const bool publicationWorkAvailable =
        budget.consume(pending.seeds.size()) &&
        budget.consume(pending.periodicControls.size()) &&
        budget.consume(pending.demands.size());
    if (!publicationWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    SyncCoverStorageProtocolGroup group;
    group.id = result.groups_.size();
    group.owningScope = std::get<0>(pending.key);
    group.behavior = std::get<1>(pending.key);
    group.readySourceResource = std::get<2>(pending.key);
    group.readyTargetResource = std::get<3>(pending.key);
    const std::size_t behaviorSignatureId = std::get<4>(pending.key);
    const bool invalidBehaviorSignature =
        behaviorSignatureId >= behaviorSignatures.size();
    if (invalidBehaviorSignature) {
      return fail(statistics, SyncCoverStorageProtocolGroupError::InvalidGraph);
    }
    const bool signaturePublicationWorkAvailable =
        budget.consume(behaviorSignatures[behaviorSignatureId].size());
    if (!signaturePublicationWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageProtocolGroupError::LimitExceeded);
    }
    group.behaviorSignature = behaviorSignatures[behaviorSignatureId];
    group.seeds = std::move(pending.seeds);
    group.periodicControls.assign(pending.periodicControls.begin(),
                                  pending.periodicControls.end());
    group.demands = std::move(pending.demands);
    group.maximumDistance = pending.maximumDistance;
    statistics.maximumGroupSeeds =
        std::max(statistics.maximumGroupSeeds, group.seeds.size());
    result.groups_.push_back(std::move(group));
  }
  statistics.groups = result.groups_.size();
  result.statistics_ = statistics;
  return result;
}
