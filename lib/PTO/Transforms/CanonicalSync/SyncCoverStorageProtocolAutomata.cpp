// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverStorageProtocolAutomata.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

using EdgeRef = SyncCoverStorageLifecycleEdgeRef;

enum class GroupRejectionReason : std::uint8_t {
  None,
  Membership,
  Direction,
  IncomparableScope,
  UnreachableTransfer,
};

enum class PhaseProjectionKind : std::uint8_t {
  SameState,
  SameLoopDistance,
  IndependentChildInvocation,
};

bool edgeRefLess(const EdgeRef &left, const EdgeRef &right) {
  return std::tie(left.component, left.edge) <
         std::tie(right.component, right.edge);
}

bool edgeRefEqual(const EdgeRef &left, const EdgeRef &right) {
  return left.component == right.component && left.edge == right.edge;
}

class WorkBudget {
public:
  WorkBudget(std::size_t maximum, std::size_t &used)
      : maximum_(maximum), used_(used) {}

  bool consume(std::size_t amount = 1) {
    if (used_ > maximum_ || amount > maximum_ - used_) {
      exhausted_ = true;
      return false;
    }
    used_ += amount;
    return true;
  }

  bool exhausted() const { return exhausted_; }

private:
  std::size_t maximum_ = 0;
  std::size_t &used_;
  bool exhausted_ = false;
};

bool consumeSortWork(WorkBudget &budget, std::size_t elementCount) {
  if (elementCount < 2) {
    return true;
  }
  std::size_t levels = 0;
  for (std::size_t covered = 1; covered < elementCount;) {
    ++levels;
    const bool wouldOverflow =
        covered > std::numeric_limits<std::size_t>::max() / 2;
    if (wouldOverflow) {
      return false;
    }
    covered *= 2;
  }
  constexpr std::size_t operationsPerLevel = 4;
  const bool levelWorkWouldOverflow =
      levels > std::numeric_limits<std::size_t>::max() / operationsPerLevel;
  if (levelWorkWouldOverflow) {
    return false;
  }
  const std::size_t workPerElement = levels * operationsPerLevel;
  const bool totalWorkWouldOverflow =
      elementCount > std::numeric_limits<std::size_t>::max() / workPerElement;
  return !totalWorkWouldOverflow &&
         budget.consume(elementCount * workPerElement);
}

std::optional<bool> meteredContains(const std::vector<SyncCoverDemandId> &ids,
                                    SyncCoverDemandId value,
                                    WorkBudget &budget) {
  std::size_t begin = 0;
  std::size_t end = ids.size();
  while (begin < end) {
    if (!budget.consume()) {
      return std::nullopt;
    }
    const std::size_t middle = begin + (end - begin) / 2;
    if (ids[middle] < value) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < ids.size() && ids[begin] == value;
}

bool consumeGuardMatchWork(WorkBudget &budget, const SyncCoverGuard &guard,
                           std::size_t periodicControlCount) {
  return budget.consume(guard.literals.size()) &&
         budget.consume(periodicControlCount);
}

bool guardMatchesState(const SyncCoverGuard &guard,
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
    const bool periodicLiteral =
        periodicControlIndex < periodicControls.size() &&
        periodicControls[periodicControlIndex] == literal.control;
    if (!periodicLiteral) {
      continue;
    }
    const SyncCoverControlPhaseRelation &relation =
        *controls[literal.control].phaseRelation;
    const std::size_t phase = phaseState[periodicControlIndex];
    if (relation.activeAlternative[phase] != literal.alternative) {
      return false;
    }
  }
  return true;
}

SyncCoverStorageLifecycleEdgeKindMask supportedKinds() {
  return syncCoverStorageLifecycleEdgeKindBit(
             SyncCoverStorageLifecycleEdgeKind::Ready) |
         syncCoverStorageLifecycleEdgeKindBit(
             SyncCoverStorageLifecycleEdgeKind::Release) |
         syncCoverStorageLifecycleEdgeKindBit(
             SyncCoverStorageLifecycleEdgeKind::Exclusion);
}

SyncCoverStorageLifecycleEdgeKindMask readyBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Ready);
}

SyncCoverStorageLifecycleEdgeKindMask releaseBit() {
  return syncCoverStorageLifecycleEdgeKindBit(
      SyncCoverStorageLifecycleEdgeKind::Release);
}

struct JointStateValidation {
  bool valid = false;
  std::optional<SyncCoverScopeId> phaseScope;
};

JointStateValidation
validateJointStates(const SyncCoverGraph &graph,
                    const SyncCoverStorageProtocolGroup &group,
                    WorkBudget &budget) {
  JointStateValidation result;
  const std::vector<SyncCoverControl> &controls = graph.getControls();
  const std::vector<std::vector<std::size_t>> &states =
      group.reachablePhaseStates;
  const bool invalidStateCount = states.empty() || states.size() > 64;
  if (invalidStateCount) {
    return result;
  }
  const bool invalidSignature =
      group.behaviorSignature.empty() ||
      group.behaviorSignature.front() != states.size();
  if (invalidSignature) {
    return result;
  }
  for (std::size_t state = 0; state < states.size(); ++state) {
    const bool invalidStateWidth =
        states[state].size() != group.periodicControls.size();
    if (invalidStateWidth) {
      return result;
    }
    const std::size_t successor = (state + 1) % states.size();
    for (std::size_t controlIndex = 0;
         controlIndex < group.periodicControls.size(); ++controlIndex) {
      if (!budget.consume()) {
        return result;
      }
      const SyncCoverControlId controlId = group.periodicControls[controlIndex];
      const bool invalidControl =
          controlId >= controls.size() || !controls[controlId].phaseRelation;
      if (invalidControl) {
        return result;
      }
      const SyncCoverControl &control = controls[controlId];
      const SyncCoverControlPhaseRelation &relation = *control.phaseRelation;
      if (result.phaseScope && *result.phaseScope != relation.loopScope) {
        return result;
      }
      result.phaseScope = relation.loopScope;
      const std::size_t phase = states[state][controlIndex];
      const std::size_t successorPhase = states[successor][controlIndex];
      const bool invalidPhase =
          phase >= relation.nextPhase.size() ||
          successorPhase >= relation.nextPhase.size() ||
          relation.nextPhase[phase] != successorPhase ||
          relation.activeAlternative[phase] >= control.alternatives;
      if (invalidPhase) {
        return result;
      }
      if (state == 0 && relation.initialPhase != phase) {
        return result;
      }
    }
  }
  result.valid = true;
  return result;
}

} // namespace

SyncCoverStorageProtocolAutomatonIndex
mlir::pto::buildSyncCoverStorageProtocolAutomatonIndex(
    const SyncCoverGraph &graph,
    const SyncCoverStorageLifecycleIndex &lifecycleIndex,
    const SyncCoverStorageProtocolSeedIndex &seedIndex,
    const SyncCoverStorageProtocolGroupIndex &groupIndex,
    const SyncCoverStorageProtocolAutomatonLimits &limits) {
  SyncCoverStorageProtocolAutomatonIndex result;
  const auto fail = [&](SyncCoverStorageProtocolAutomatonStatistics statistics,
                        SyncCoverStorageProtocolAutomatonError error) {
    result.automata_.clear();
    statistics.automata = 0;
    statistics.states = 0;
    statistics.transfers = 0;
    statistics.statePairIncidences = 0;
    statistics.maximumAutomatonTransfers = 0;
    statistics.maximumTransferStatePairs = 0;
    statistics.truncated =
        error == SyncCoverStorageProtocolAutomatonError::LimitExceeded;
    result.statistics_ = statistics;
    result.error_ = error;
    return std::move(result);
  };
  const bool invalidLimit =
      limits.maximumWorkUnits == 0 || limits.maximumAutomata == 0 ||
      limits.maximumStates == 0 || limits.maximumTransfers == 0 ||
      limits.maximumStatePairIncidences == 0 || limits.maximumLanes == 0 ||
      limits.maximumLanes > 8;
  if (invalidLimit) {
    return fail({}, SyncCoverStorageProtocolAutomatonError::InvalidLimit);
  }
  if (!graph.isStructureFrozen()) {
    return fail({}, SyncCoverStorageProtocolAutomatonError::InvalidGraph);
  }
  result.bindToGraph(graph);
  if (!lifecycleIndex.isComplete()) {
    return fail(
        {}, SyncCoverStorageProtocolAutomatonError::IncompleteLifecycleIndex);
  }
  if (!seedIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolAutomatonError::IncompleteSeedIndex);
  }
  if (!groupIndex.isComplete()) {
    return fail({},
                SyncCoverStorageProtocolAutomatonError::IncompleteGroupIndex);
  }

  const std::vector<SyncCoverStorageLifecycleComponent> &components =
      lifecycleIndex.getComponents();
  const std::vector<SyncCoverStorageProtocolSeed> &seeds = seedIndex.getSeeds();
  const std::vector<SyncCoverStorageProtocolGroup> &groups =
      groupIndex.getGroups();
  const std::vector<SyncCoverDemand> &demands = graph.getDemands();
  const std::vector<SyncCoverControl> &controls = graph.getControls();
  SyncCoverStorageProtocolAutomatonStatistics statistics;
  WorkBudget budget(limits.maximumWorkUnits, statistics.workUnits);
  if (!budget.consume(groups.size())) {
    return fail(statistics,
                SyncCoverStorageProtocolAutomatonError::LimitExceeded);
  }
  result.automata_.reserve(std::min(groups.size(), limits.maximumAutomata));

  for (std::size_t groupIndexPosition = 0; groupIndexPosition < groups.size();
       ++groupIndexPosition) {
    const SyncCoverStorageProtocolGroup &group = groups[groupIndexPosition];
    if (!budget.consume()) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    if (group.id != groupIndexPosition ||
        group.owningScope >= graph.getScopes().size()) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::InvalidGraph);
    }
    const std::size_t laneCount =
        std::max<std::size_t>(group.maximumDistance, 1);
    if (laneCount > limits.maximumLanes) {
      ++statistics.ineligibleGroups;
      ++statistics.laneLimitedGroups;
      continue;
    }
    const std::size_t stateCount = group.reachablePhaseStates.size();
    const bool stateLimitReached =
        statistics.states > limits.maximumStates ||
        stateCount > limits.maximumStates - statistics.states;
    if (stateLimitReached) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    const JointStateValidation jointStateValidation =
        validateJointStates(graph, group, budget);
    if (!jointStateValidation.valid) {
      const SyncCoverStorageProtocolAutomatonError error =
          budget.exhausted()
              ? SyncCoverStorageProtocolAutomatonError::LimitExceeded
              : SyncCoverStorageProtocolAutomatonError::InvalidGraph;
      return fail(statistics, error);
    }

    std::vector<EdgeRef> edgeRefs;
    GroupRejectionReason rejectionReason = GroupRejectionReason::None;
    for (SyncCoverStorageProtocolSeedId seedId : group.seeds) {
      if (!budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::LimitExceeded);
      }
      if (seedId >= seeds.size()) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::InvalidGraph);
      }
      const SyncCoverStorageProtocolSeed &seed = seeds[seedId];
      for (const SyncCoverStorageProtocolSccRef &sccRef :
           seed.readyReleaseSccs) {
        if (!budget.consume()) {
          return fail(statistics,
                      SyncCoverStorageProtocolAutomatonError::LimitExceeded);
        }
        const bool invalidScc =
            sccRef.component >= components.size() ||
            sccRef.scc >= components[sccRef.component].sccs.size();
        if (invalidScc) {
          return fail(statistics,
                      SyncCoverStorageProtocolAutomatonError::InvalidGraph);
        }
        const SyncCoverStorageLifecycleScc &scc =
            components[sccRef.component].sccs[sccRef.scc];
        for (SyncCoverStorageLifecycleEdgeId edge : scc.internalEdges) {
          const bool transferLimitReached =
              statistics.transfers > limits.maximumTransfers ||
              edgeRefs.size() >= limits.maximumTransfers - statistics.transfers;
          if (transferLimitReached || !budget.consume()) {
            return fail(statistics,
                        SyncCoverStorageProtocolAutomatonError::LimitExceeded);
          }
          edgeRefs.push_back({sccRef.component, edge});
        }
      }
    }
    if (!consumeSortWork(budget, edgeRefs.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    std::sort(edgeRefs.begin(), edgeRefs.end(), edgeRefLess);
    edgeRefs.erase(std::unique(edgeRefs.begin(), edgeRefs.end(), edgeRefEqual),
                   edgeRefs.end());

    SyncCoverStorageProtocolAutomaton automaton;
    automaton.id = result.automata_.size();
    automaton.group = group.id;
    automaton.owningScope = group.owningScope;
    automaton.stateCount = stateCount;
    automaton.transfers.reserve(edgeRefs.size());
    std::vector<SyncCoverDemandId> activeDemands;
    activeDemands.reserve(edgeRefs.size());
    std::size_t pendingStatePairIncidences = 0;
    SyncCoverStorageLifecycleEdgeKindMask unreachableKinds = 0;
    for (const EdgeRef &edgeRef : edgeRefs) {
      if (!budget.consume()) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::LimitExceeded);
      }
      const bool invalidEdgeRef =
          edgeRef.component >= components.size() ||
          edgeRef.edge >= components[edgeRef.component].edges.size();
      if (invalidEdgeRef) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::InvalidGraph);
      }
      const SyncCoverStorageLifecycleComponent &component =
          components[edgeRef.component];
      const SyncCoverStorageLifecycleEdge &edge = component.edges[edgeRef.edge];
      const SyncCoverStorageLifecycleEdgeKindMask validKinds = supportedKinds();
      const bool invalidEdge = edge.source >= component.epochs.size() ||
                               edge.target >= component.epochs.size() ||
                               edge.demand >= demands.size() ||
                               edge.kinds == 0 ||
                               (edge.kinds & ~validKinds) != 0;
      if (invalidEdge) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::InvalidGraph);
      }
      const std::optional<bool> demandInGroup =
          meteredContains(group.demands, edge.demand, budget);
      if (!demandInGroup) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::LimitExceeded);
      }
      if (!*demandInGroup) {
        rejectionReason = GroupRejectionReason::Membership;
        break;
      }
      const SyncCoverStorageLifecycleEpoch &source =
          component.epochs[edge.source];
      const SyncCoverStorageLifecycleEpoch &target =
          component.epochs[edge.target];
      const bool crossResource = source.resource != target.resource;
      const bool invalidReadyDirection =
          crossResource && (edge.kinds & readyBit()) != 0 &&
          (source.resource != group.readySourceResource ||
           target.resource != group.readyTargetResource);
      const bool invalidReleaseDirection =
          crossResource && (edge.kinds & releaseBit()) != 0 &&
          (source.resource != group.readyTargetResource ||
           target.resource != group.readySourceResource);
      if (invalidReadyDirection || invalidReleaseDirection) {
        rejectionReason = GroupRejectionReason::Direction;
        break;
      }

      SyncCoverStorageProtocolTransfer transfer;
      transfer.id = automaton.transfers.size();
      transfer.edge = edgeRef;
      transfer.demand = edge.demand;
      transfer.kinds = edge.kinds;
      transfer.scope = edge.scope;
      transfer.distance = edge.distance;
      PhaseProjectionKind phaseProjection = PhaseProjectionKind::SameState;
      if (edge.distance != 0 && jointStateValidation.phaseScope) {
        const SyncCoverScopeId phaseScope = *jointStateValidation.phaseScope;
        if (edge.scope == phaseScope) {
          phaseProjection = PhaseProjectionKind::SameLoopDistance;
        } else {
          const std::size_t scopeCount = graph.getScopes().size();
          const bool scopeWorkOverflows =
              scopeCount > std::numeric_limits<std::size_t>::max() / 2;
          if (scopeWorkOverflows || !budget.consume(2 * scopeCount)) {
            return fail(statistics,
                        SyncCoverStorageProtocolAutomatonError::LimitExceeded);
          }
          const bool edgeContainsPhase =
              graph.scopeContains(edge.scope, phaseScope);
          const bool phaseContainsEdge =
              graph.scopeContains(phaseScope, edge.scope);
          if (!edgeContainsPhase && !phaseContainsEdge) {
            rejectionReason = GroupRejectionReason::IncomparableScope;
            break;
          }
          if (edgeContainsPhase) {
            phaseProjection = PhaseProjectionKind::IndependentChildInvocation;
          }
        }
      }
      transfer.sourceResource = source.resource;
      transfer.targetResource = target.resource;
      const SyncCoverDemand &demand = demands[edge.demand];
      if (!budget.consume(2 * stateCount)) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::LimitExceeded);
      }
      std::vector<bool> sourceActive(stateCount, false);
      std::vector<bool> targetActive(stateCount, false);
      for (std::size_t state = 0; state < stateCount; ++state) {
        const bool guardWorkAvailable =
            consumeGuardMatchWork(budget, demand.sourceGuard,
                                  group.periodicControls.size()) &&
            consumeGuardMatchWork(budget, demand.targetGuard,
                                  group.periodicControls.size());
        if (!guardWorkAvailable) {
          return fail(statistics,
                      SyncCoverStorageProtocolAutomatonError::LimitExceeded);
        }
        sourceActive[state] =
            guardMatchesState(demand.sourceGuard, group.periodicControls,
                              group.reachablePhaseStates[state], controls);
        targetActive[state] =
            guardMatchesState(demand.targetGuard, group.periodicControls,
                              group.reachablePhaseStates[state], controls);
      }
      const auto appendStatePair = [&](std::size_t sourceState,
                                       std::size_t targetState) {
        if (!budget.consume()) {
          return false;
        }
        if (!sourceActive[sourceState] || !targetActive[targetState]) {
          return true;
        }
        const bool pairLimitReached =
            statistics.statePairIncidences >
                limits.maximumStatePairIncidences ||
            pendingStatePairIncidences >= limits.maximumStatePairIncidences -
                                              statistics.statePairIncidences;
        if (pairLimitReached || !budget.consume()) {
          return false;
        }
        transfer.activeStatePairs.push_back({sourceState, targetState});
        ++pendingStatePairIncidences;
        return true;
      };
      bool pairWorkAvailable = true;
      for (std::size_t sourceState = 0;
           pairWorkAvailable && sourceState < stateCount; ++sourceState) {
        if (phaseProjection ==
            PhaseProjectionKind::IndependentChildInvocation) {
          for (std::size_t targetState = 0;
               pairWorkAvailable && targetState < stateCount; ++targetState) {
            pairWorkAvailable = appendStatePair(sourceState, targetState);
          }
          continue;
        }
        std::size_t targetState = sourceState;
        if (phaseProjection == PhaseProjectionKind::SameLoopDistance) {
          targetState =
              (sourceState + (edge.distance % stateCount)) % stateCount;
        }
        pairWorkAvailable = appendStatePair(sourceState, targetState);
      }
      if (!pairWorkAvailable) {
        return fail(statistics,
                    SyncCoverStorageProtocolAutomatonError::LimitExceeded);
      }
      if (transfer.activeStatePairs.empty()) {
        rejectionReason = GroupRejectionReason::UnreachableTransfer;
        unreachableKinds = edge.kinds;
        break;
      }
      activeDemands.push_back(edge.demand);
      automaton.maximumDistance =
          std::max(automaton.maximumDistance, edge.distance);
      automaton.transfers.push_back(std::move(transfer));
    }
    if (rejectionReason != GroupRejectionReason::None) {
      ++statistics.ineligibleGroups;
      switch (rejectionReason) {
      case GroupRejectionReason::Membership:
        ++statistics.membershipRejectedGroups;
        break;
      case GroupRejectionReason::Direction:
        ++statistics.directionRejectedGroups;
        break;
      case GroupRejectionReason::IncomparableScope:
        ++statistics.scopeRejectedGroups;
        break;
      case GroupRejectionReason::UnreachableTransfer:
        ++statistics.unreachableTransferGroups;
        statistics.unreachableReadyTransferGroups +=
            (unreachableKinds & readyBit()) != 0;
        statistics.unreachableReleaseTransferGroups +=
            (unreachableKinds & releaseBit()) != 0;
        statistics.unreachableExclusionTransferGroups +=
            (unreachableKinds &
             syncCoverStorageLifecycleEdgeKindBit(
                 SyncCoverStorageLifecycleEdgeKind::Exclusion)) != 0;
        break;
      case GroupRejectionReason::None:
        break;
      }
      continue;
    }
    if (!consumeSortWork(budget, activeDemands.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    std::sort(activeDemands.begin(), activeDemands.end());
    activeDemands.erase(std::unique(activeDemands.begin(), activeDemands.end()),
                        activeDemands.end());
    if (!budget.consume(activeDemands.size())) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    const bool demandSetMismatch = activeDemands != group.demands;
    const bool distanceMismatch =
        automaton.maximumDistance != group.maximumDistance;
    if (demandSetMismatch || distanceMismatch) {
      ++statistics.ineligibleGroups;
      statistics.demandSetMismatchGroups += demandSetMismatch;
      statistics.distanceMismatchGroups += distanceMismatch;
      continue;
    }
    const bool automatonLimitReached =
        result.automata_.size() >= limits.maximumAutomata;
    if (automatonLimitReached) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    const bool publicationWorkAvailable =
        budget.consume(automaton.transfers.size()) &&
        budget.consume(pendingStatePairIncidences);
    if (!publicationWorkAvailable) {
      return fail(statistics,
                  SyncCoverStorageProtocolAutomatonError::LimitExceeded);
    }
    statistics.states += stateCount;
    statistics.transfers += automaton.transfers.size();
    statistics.statePairIncidences += pendingStatePairIncidences;
    automaton.statePairIncidences = pendingStatePairIncidences;
    statistics.maximumAutomatonTransfers = std::max(
        statistics.maximumAutomatonTransfers, automaton.transfers.size());
    for (const SyncCoverStorageProtocolTransfer &transfer :
         automaton.transfers) {
      statistics.maximumTransferStatePairs =
          std::max(statistics.maximumTransferStatePairs,
                   transfer.activeStatePairs.size());
    }
    result.automata_.push_back(std::move(automaton));
    ++statistics.eligibleGroups;
  }
  statistics.automata = result.automata_.size();
  result.statistics_ = statistics;
  return result;
}
