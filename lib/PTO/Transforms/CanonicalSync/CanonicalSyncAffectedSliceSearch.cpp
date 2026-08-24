// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// This file owns bounded joint exchanges around a small requirement slice.
// The verified incumbent remains outside every frontier, and only a complete
// plan that passes CanonicalSync's centralized feasibility check can replace it.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-canonical-sync-affected-slice"

namespace {

constexpr std::size_t kExhaustiveSeedMechanismThreshold = 7;
constexpr std::size_t kSeedMechanismFrontierLimit = 8;
constexpr std::size_t kEvictionSeedLimit = 32;
constexpr std::size_t kDirectCandidateLimit = 24;
constexpr std::size_t kConnectorCandidateLimit = 8;
constexpr std::size_t kCombinedCandidateLimit = 32;
constexpr std::size_t kStructuralCandidateLimitMax = 128;
constexpr std::size_t kSearchDepthLimit = 10;
constexpr std::size_t kBeamWidthMin = 4;
constexpr std::size_t kBeamWidthMax = 16;
constexpr std::size_t kStateEvaluationLimit = 4096;
constexpr std::size_t kCompleteCleanupCandidateLimit = 8;

struct ForwardConnectorCone {
  std::vector<bool> fromSource;
  std::vector<bool> toTarget;
};

bool sameRequirementIdentity(const CanonicalDependency &first,
                             const CanonicalDependency &second) {
  return first.source == second.source && first.target == second.target &&
         first.kind == second.kind &&
         first.iterationDistance == second.iterationDistance &&
         first.recurrenceLoop == second.recurrenceLoop;
}

template <typename T>
void addSaturated(T &total, T value) {
  const T max = std::numeric_limits<T>::max();
  total = value > max - total ? max : total + value;
}

void addProfiles(std::vector<std::size_t> &target,
                 ArrayRef<std::size_t> source) {
  const std::size_t sourceSize = source.size();
  const std::size_t targetSize = target.size();
  if (targetSize < sourceSize) {
    target.resize(sourceSize, 0);
  }
  for (std::size_t index = 0; index < source.size(); ++index) {
    addSaturated(target[index], source[index]);
  }
}

bool removalMechanismLess(
    const CanonicalAffectedSliceEvictionMechanism &first,
    const CanonicalAffectedSliceEvictionMechanism &second) {
  if (first.actionProfile != second.actionProfile) {
    return first.actionProfile > second.actionProfile;
  }
  if (first.barrierProfile != second.barrierProfile) {
    return first.barrierProfile > second.barrierProfile;
  }
  return first.mechanism < second.mechanism;
}

bool evictionSeedLess(const CanonicalAffectedSliceEvictionSeed &first,
                      const CanonicalAffectedSliceEvictionSeed &second) {
  if (first.actionProfile != second.actionProfile) {
    return first.actionProfile > second.actionProfile;
  }
  if (first.barrierProfile != second.barrierProfile) {
    return first.barrierProfile > second.barrierProfile;
  }
  return std::lexicographical_compare(
      first.mechanisms.begin(), first.mechanisms.end(),
      second.mechanisms.begin(), second.mechanisms.end());
}

std::size_t getEnclosingLoopDepth(Operation *operation) {
  std::size_t depth = 0;
  Operation *parent = operation ? operation->getParentOp() : nullptr;
  while (parent) {
    depth += isa<LoopLikeOpInterface>(parent) ? 1U : 0U;
    parent = parent->getParentOp();
  }
  return depth;
}

std::vector<std::size_t>
getUniqueIndices(SmallVectorImpl<std::size_t> &indices) {
  llvm::sort(indices);
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
  return {indices.begin(), indices.end()};
}

std::vector<CanonicalSelectionMechanismRef> buildStateSignature(
    ArrayRef<CanonicalBarrier> barriers,
    ArrayRef<CanonicalEventBundleCandidate> bundles) {
  std::vector<CanonicalSelectionMechanismRef> signature;
  signature.reserve(barriers.size() + bundles.size());
  llvm::transform(barriers, std::back_inserter(signature),
                  [](const CanonicalBarrier &barrier) {
                    return CanonicalSelectionMechanismRef{
                        CanonicalSelectionMechanismKind::Barrier, barrier.id};
                  });
  llvm::transform(bundles, std::back_inserter(signature),
                  [](const CanonicalEventBundleCandidate &bundle) {
                    return CanonicalSelectionMechanismRef{
                        CanonicalSelectionMechanismKind::EventBundle,
                        bundle.id};
                  });
  llvm::sort(signature);
  return signature;
}

bool containsMechanism(ArrayRef<CanonicalSelectionMechanismRef> mechanisms,
                       const CanonicalSelectionMechanismRef &mechanism) {
  return llvm::is_contained(mechanisms, mechanism);
}

struct AffectedSliceCandidate {
  CanonicalSelectionMechanismRef mechanism;
  const CanonicalBarrierCandidate *barrier = nullptr;
  const CanonicalEventBundleCandidate *bundle = nullptr;
  std::vector<SyncGraphEdge> completionEdges;
  std::size_t potentialCoverage = 0;
  std::size_t directCoverage = 0;
  std::vector<std::size_t> coveredRequirements;
  std::vector<std::size_t> exactRequirements;
  bool hasInactiveOrigins = false;
  bool connector = false;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierProfile;
  std::size_t intervalSpan = 0;
  CanonicalMechanismPlanScore partialScore;
};

struct AffectedSliceState {
  std::vector<CanonicalSelectionMechanismRef> mechanisms;
  std::vector<std::size_t> uncovered;
  std::vector<std::size_t> unresolvedSupportProfile;
  CanonicalMechanismPlanScore score;
};

struct AffectedSliceCompleteCandidate {
  std::vector<CanonicalBarrier> barriers;
  std::vector<CanonicalEventBundleCandidate> bundles;
  std::vector<CanonicalSelectionMechanismRef> addedMechanisms;
  std::vector<CanonicalDependency> affectedRequirements;
  CanonicalMechanismPlanScore score;
  std::vector<CanonicalSelectionMechanismRef> signature;
};

bool completeCandidateLess(const AffectedSliceCompleteCandidate &first,
                           const AffectedSliceCompleteCandidate &second) {
  if (canonicalMechanismPlanScoreLess(first.score, second.score)) {
    return true;
  }
  if (canonicalMechanismPlanScoreLess(second.score, first.score)) {
    return false;
  }
  return first.signature < second.signature;
}

bool affectedSliceStateLess(const AffectedSliceState &first,
                            const AffectedSliceState &second) {
  if (first.unresolvedSupportProfile != second.unresolvedSupportProfile) {
    return first.unresolvedSupportProfile < second.unresolvedSupportProfile;
  }
  const std::size_t firstUncovered = first.uncovered.size();
  const std::size_t secondUncovered = second.uncovered.size();
  if (firstUncovered != secondUncovered) {
    return firstUncovered < secondUncovered;
  }
  if (canonicalMechanismPlanScoreLess(first.score, second.score)) {
    return true;
  }
  if (canonicalMechanismPlanScoreLess(second.score, first.score)) {
    return false;
  }
  return first.mechanisms < second.mechanisms;
}

std::vector<std::size_t>
getCoveredIndices(std::size_t requirementCount,
                  ArrayRef<std::size_t> uncovered) {
  std::vector<bool> isUncovered(requirementCount, false);
  for (std::size_t index : uncovered) {
    if (index < requirementCount) {
      isUncovered[index] = true;
    }
  }
  std::vector<std::size_t> covered;
  for (std::size_t index = 0; index < requirementCount; ++index) {
    if (!isUncovered[index]) {
      covered.push_back(index);
    }
  }
  return covered;
}

bool eventDirectlyRepresentsRequirement(
    const CanonicalEvent &event, const CanonicalDependency &requirement) {
  const bool matchesEndpoint = event.source == requirement.source &&
                               event.target == requirement.target &&
                               event.iterationDistance ==
                                   requirement.iterationDistance &&
                               event.recurrenceLoop == requirement.recurrenceLoop;
  if (matchesEndpoint) {
    return true;
  }
  return llvm::any_of(event.completions, [&](const auto &completion) {
    return completion.source == requirement.source &&
           completion.target == requirement.target &&
           completion.iterationDistance == requirement.iterationDistance &&
           completion.recurrenceLoop == requirement.recurrenceLoop;
  });
}

std::vector<bool> collectReachable(
    std::size_t vertexCount, ArrayRef<SyncGraphEdge> edges,
    std::size_t start, bool reverse) {
  std::vector<std::vector<std::size_t>> adjacency(vertexCount);
  for (const SyncGraphEdge &edge : edges) {
    if (edge.source >= vertexCount || edge.target >= vertexCount ||
        edge.kind == SyncGraphEdgeKind::NonCompletionPreservingIssueOrder) {
      continue;
    }
    const std::size_t source = reverse ? edge.target : edge.source;
    const std::size_t target = reverse ? edge.source : edge.target;
    adjacency[source].push_back(target);
  }
  std::vector<bool> reachable(vertexCount, false);
  if (start >= vertexCount) {
    return reachable;
  }
  std::deque<std::size_t> ready{start};
  reachable[start] = true;
  while (!ready.empty()) {
    const std::size_t source = ready.front();
    ready.pop_front();
    for (std::size_t target : adjacency[source]) {
      if (!reachable[target]) {
        reachable[target] = true;
        ready.push_back(target);
      }
    }
  }
  return reachable;
}

std::vector<ForwardConnectorCone> buildForwardConnectorCones(
    ArrayRef<CanonicalDependency> requirements,
    ArrayRef<SyncGraphEdge> potentialEdges, std::size_t vertexCount) {
  std::vector<ForwardConnectorCone> cones;
  for (const CanonicalDependency &requirement : requirements) {
    if (requirement.iterationDistance != 0 ||
        requirement.source >= vertexCount ||
        requirement.target >= vertexCount) {
      continue;
    }
    ForwardConnectorCone cone;
    cone.fromSource = collectReachable(vertexCount, potentialEdges,
                                       requirement.source,
                                       /*reverse=*/false);
    cone.toTarget = collectReachable(vertexCount, potentialEdges,
                                     requirement.target,
                                     /*reverse=*/true);
    cones.push_back(std::move(cone));
  }
  return cones;
}

std::size_t countForwardConnectorCones(
    ArrayRef<SyncGraphEdge> candidateEdges,
    ArrayRef<ForwardConnectorCone> cones, std::size_t vertexCount) {
  std::size_t coveredCones = 0;
  for (const ForwardConnectorCone &cone : cones) {
    const bool coversCone = llvm::any_of(
        candidateEdges, [&](const SyncGraphEdge &edge) {
      if (edge.source < vertexCount && edge.target < vertexCount &&
          cone.fromSource[edge.source] && cone.toTarget[edge.target]) {
        return true;
      }
      return false;
    });
    coveredCones += coversCone ? 1U : 0U;
  }
  return coveredCones;
}

std::size_t countTouchedRecurrenceEndpoints(
    const AffectedSliceCandidate &candidate,
    ArrayRef<CanonicalDependency> requirements) {
  std::size_t touchedRequirements = 0;
  for (const CanonicalDependency &requirement : requirements) {
    if (requirement.iterationDistance == 0) {
      continue;
    }
    if (candidate.barrier &&
        candidate.barrier->barrier.pipe != PipelineType::PIPE_UNASSIGNED) {
      continue;
    }
    if (!candidate.bundle) {
      continue;
    }
    const bool touches = llvm::any_of(
        candidate.bundle->events, [&](const CanonicalEvent &event) {
          return event.source == requirement.source ||
                 event.source == requirement.target ||
                 event.target == requirement.source ||
                 event.target == requirement.target;
        });
    if (touches) {
      ++touchedRequirements;
    }
  }
  return touchedRequirements;
}

} // namespace

bool mlir::pto::canonicalMechanismOriginsAreInactive(
    bool hasCompleteOriginProvenance,
    ArrayRef<CanonicalDependency> originRequirements,
    ArrayRef<CanonicalDependency> activeRequirements) {
  if (!hasCompleteOriginProvenance || originRequirements.empty()) {
    return false;
  }
  return llvm::all_of(originRequirements, [&](const auto &origin) {
    return llvm::none_of(activeRequirements, [&](const auto &active) {
      return sameRequirementIdentity(origin, active);
    });
  });
}

std::vector<CanonicalAffectedSliceEvictionSeed>
mlir::pto::buildCanonicalAffectedSliceEvictionSeeds(
    ArrayRef<CanonicalAffectedSliceEvictionMechanism> mechanisms,
    std::size_t exhaustiveMechanismThreshold,
    std::size_t mechanismFrontierLimit, std::size_t seedLimit) {
  std::vector<CanonicalAffectedSliceEvictionMechanism> frontier(
      mechanisms.begin(), mechanisms.end());
  llvm::stable_sort(frontier, removalMechanismLess);
  const std::size_t frontierSize = frontier.size();
  if (frontierSize > exhaustiveMechanismThreshold &&
      frontierSize > mechanismFrontierLimit) {
    frontier.resize(mechanismFrontierLimit);
  }

  std::vector<CanonicalAffectedSliceEvictionSeed> seeds;
  for (std::size_t first = 0; first < frontier.size(); ++first) {
    CanonicalAffectedSliceEvictionSeed singleton;
    singleton.mechanisms.push_back(frontier[first].mechanism);
    singleton.actionProfile = frontier[first].actionProfile;
    singleton.barrierProfile = frontier[first].barrierProfile;
    seeds.push_back(std::move(singleton));
    for (std::size_t second = first + 1; second < frontier.size(); ++second) {
      CanonicalAffectedSliceEvictionSeed pair;
      pair.mechanisms.push_back(frontier[first].mechanism);
      pair.mechanisms.push_back(frontier[second].mechanism);
      llvm::sort(pair.mechanisms);
      pair.actionProfile = frontier[first].actionProfile;
      addProfiles(pair.actionProfile, frontier[second].actionProfile);
      pair.barrierProfile = frontier[first].barrierProfile;
      addProfiles(pair.barrierProfile, frontier[second].barrierProfile);
      seeds.push_back(std::move(pair));
    }
  }
  llvm::stable_sort(seeds, evictionSeedLess);
  const std::size_t seedCount = seeds.size();
  if (seedCount > seedLimit) {
    seeds.resize(seedLimit);
  }
  return seeds;
}

CanonicalAffectedSliceSearchResult
mlir::pto::searchCanonicalAffectedSliceCandidates(
    ArrayRef<CanonicalAffectedSliceSearchCandidate> candidates,
    std::size_t requirementCount,
    const CanonicalMechanismPlanScore &initialScore, std::size_t beamWidth,
    std::size_t depthLimit, std::size_t evaluationLimit,
    std::size_t completeCandidateLimit,
    llvm::function_ref<std::optional<CanonicalAffectedSliceSearchEvaluation>(
        ArrayRef<CanonicalSelectionMechanismRef>)>
        evaluate) {
  CanonicalAffectedSliceSearchResult result;
  const bool invalidInput = candidates.empty() || requirementCount == 0 ||
                            beamWidth == 0 || depthLimit == 0 ||
                            completeCandidateLimit == 0;
  if (invalidInput) {
    return result;
  }

  std::vector<std::size_t> support(requirementCount, 0);
  for (const CanonicalAffectedSliceSearchCandidate &candidate : candidates) {
    for (std::size_t requirement : candidate.coveredRequirements) {
      if (requirement < support.size()) {
        ++support[requirement];
      }
    }
  }
  const auto buildSupportProfile = [&](ArrayRef<std::size_t> uncovered) {
    std::vector<std::size_t> profile(candidates.size(), 0);
    const std::size_t supportSize = support.size();
    for (std::size_t requirement : uncovered) {
      if (requirement < supportSize && support[requirement] != 0) {
        ++profile[support[requirement] - 1];
      }
    }
    return profile;
  };
  const auto completeLess = [](const CanonicalAffectedSliceSearchComplete &first,
                               const CanonicalAffectedSliceSearchComplete &second) {
    if (canonicalMechanismPlanScoreLess(first.score, second.score)) {
      return true;
    }
    if (canonicalMechanismPlanScoreLess(second.score, first.score)) {
      return false;
    }
    return first.mechanisms < second.mechanisms;
  };

  AffectedSliceState initial;
  initial.uncovered.resize(requirementCount);
  std::iota(initial.uncovered.begin(), initial.uncovered.end(), 0);
  initial.unresolvedSupportProfile = buildSupportProfile(initial.uncovered);
  initial.score = initialScore;
  std::vector<AffectedSliceState> beam{std::move(initial)};
  std::set<std::vector<CanonicalSelectionMechanismRef>> seen{{}};
  const std::size_t boundedDepth = std::min(depthLimit, candidates.size());

  for (std::size_t depth = 0; depth < boundedDepth; ++depth) {
    std::vector<AffectedSliceState> next;
    for (const AffectedSliceState &state : beam) {
      for (const CanonicalAffectedSliceSearchCandidate &candidate : candidates) {
        if (containsMechanism(state.mechanisms, candidate.mechanism)) {
          continue;
        }
        AffectedSliceState successor;
        successor.mechanisms = state.mechanisms;
        successor.mechanisms.push_back(candidate.mechanism);
        llvm::sort(successor.mechanisms);
        if (!seen.insert(successor.mechanisms).second) {
          continue;
        }
        if (result.evaluations == evaluationLimit) {
          result.budgetExhausted = true;
          break;
        }
        ++result.evaluations;
        std::optional<CanonicalAffectedSliceSearchEvaluation> projection =
            evaluate(successor.mechanisms);
        if (!projection) {
          continue;
        }
        successor.uncovered =
            std::move(projection->uncoveredRequirements);
        successor.unresolvedSupportProfile =
            buildSupportProfile(successor.uncovered);
        successor.score = std::move(projection->score);
        if (successor.uncovered.empty()) {
          result.complete.push_back(
              {successor.mechanisms, successor.score});
          llvm::stable_sort(result.complete, completeLess);
          const std::size_t completeSize = result.complete.size();
          if (completeSize > completeCandidateLimit) {
            result.complete.resize(completeCandidateLimit);
          }
        }
        next.push_back(std::move(successor));
      }
      if (result.budgetExhausted) {
        break;
      }
    }
    llvm::stable_sort(next, affectedSliceStateLess);
    const std::size_t nextSize = next.size();
    if (nextSize > beamWidth) {
      next.resize(beamWidth);
    }
    beam = std::move(next);
    const bool searchFinished = beam.empty() || result.budgetExhausted;
    if (searchFinished) {
      break;
    }
  }
  return result;
}

void CanonicalSyncPlanBuilder::optimizeAffectedSliceExchanges(
    std::vector<CanonicalBarrier> &incumbentBarriers,
    std::vector<CanonicalEventBundleCandidate> &incumbentBundles) const {
  std::size_t maxLoopDepth = 0;
  funcOperation_->walk([&](Operation *operation) {
    maxLoopDepth = std::max(maxLoopDepth, getEnclosingLoopDepth(operation));
  });

  std::vector<CanonicalAffectedSliceEvictionMechanism> eligible;
  for (const CanonicalBarrier &barrier : incumbentBarriers) {
    auto candidate = llvm::find_if(
        mechanismUniverse_.barriers, [&](const auto &entry) {
          return entry.id == barrier.id;
        });
    const bool hasCandidate = candidate != mechanismUniverse_.barriers.end();
    const bool hasInactiveOrigins =
        hasCandidate && canonicalMechanismOriginsAreInactive(
                            candidate->hasCompleteOriginProvenance,
                            candidate->originRequirements,
                            plan_.completionRequirements_);
    if (!hasInactiveOrigins) {
      continue;
    }
    CanonicalAffectedSliceEvictionMechanism mechanism;
    mechanism.mechanism = {CanonicalSelectionMechanismKind::Barrier,
                           barrier.id};
    mechanism.actionProfile.assign(maxLoopDepth + 1, 0);
    mechanism.barrierProfile =
        buildCanonicalBarrierActionProfile({barrier}, maxLoopDepth);
    eligible.push_back(std::move(mechanism));
  }
  for (const CanonicalEventBundleCandidate &bundle : incumbentBundles) {
    if (bundle.kind == CanonicalEventBundleKind::Ownership) {
      continue;
    }
    auto candidate = llvm::find_if(
        mechanismUniverse_.eventBundles,
        [&](const auto &entry) { return entry.id == bundle.id; });
    const bool hasCandidate =
        candidate != mechanismUniverse_.eventBundles.end();
    const bool hasInactiveOrigins =
        hasCandidate && canonicalMechanismOriginsAreInactive(
                            candidate->hasCompleteOriginProvenance,
                            candidate->originRequirements,
                            plan_.completionRequirements_);
    if (!hasInactiveOrigins) {
      continue;
    }
    CanonicalAffectedSliceEvictionMechanism mechanism;
    mechanism.mechanism = {CanonicalSelectionMechanismKind::EventBundle,
                           bundle.id};
    mechanism.actionProfile.assign(maxLoopDepth + 1, 0);
    mechanism.barrierProfile.assign(maxLoopDepth + 1, 0);
    for (const CanonicalEvent &event : bundle.events) {
      for (const CanonicalEventAction &action : event.actions) {
        const std::size_t depth = getEnclosingLoopDepth(action.anchor.operation);
        ++mechanism.actionProfile[maxLoopDepth - depth];
      }
    }
    eligible.push_back(std::move(mechanism));
  }

  const std::vector<CanonicalAffectedSliceEvictionSeed> seeds =
      buildCanonicalAffectedSliceEvictionSeeds(
          eligible, kExhaustiveSeedMechanismThreshold,
          kSeedMechanismFrontierLimit, kEvictionSeedLimit);
  if (seeds.empty()) {
    return;
  }

  CanonicalMechanismPlanScore incumbentScore =
      scoreCandidatePlan(incumbentBarriers, incumbentBundles);
  std::optional<std::vector<CanonicalBarrier>> bestBarriers;
  std::optional<std::vector<CanonicalEventBundleCandidate>> bestBundles;
  std::optional<CanonicalMechanismPlanScore> bestScore;
  std::vector<AffectedSliceCompleteCandidate> completeCandidates;
  std::size_t stateEvaluations = 0;
  bool stateBudgetExhausted = false;

  const auto getUncovered = [&](ArrayRef<CanonicalBarrier> barriers,
                                ArrayRef<CanonicalEventBundleCandidate> bundles,
                                ArrayRef<CanonicalDependency> requirements) {
    SmallVector<std::size_t, 32> indices;
    const std::vector<CanonicalEvent> events =
        flattenCanonicalEventBundles(bundles);
    countUncoveredRequirements(barriers, events, requirements,
                               /*diagnose=*/false, &indices);
    return getUniqueIndices(indices);
  };

  for (const CanonicalAffectedSliceEvictionSeed &seed : seeds) {
    if (stateBudgetExhausted) {
      break;
    }
    std::vector<CanonicalBarrier> remainingBarriers = incumbentBarriers;
    std::vector<CanonicalEventBundleCandidate> remainingBundles =
        incumbentBundles;
    for (const CanonicalSelectionMechanismRef &mechanism : seed.mechanisms) {
      if (mechanism.kind == CanonicalSelectionMechanismKind::Barrier) {
        remainingBarriers.erase(
            std::remove_if(remainingBarriers.begin(), remainingBarriers.end(),
                           [&](const CanonicalBarrier &barrier) {
                             return barrier.id == mechanism.id;
                           }),
            remainingBarriers.end());
      } else {
        remainingBundles.erase(
            std::remove_if(remainingBundles.begin(), remainingBundles.end(),
                           [&](const CanonicalEventBundleCandidate &bundle) {
                             return bundle.id == mechanism.id;
                           }),
            remainingBundles.end());
      }
    }
    if (!isCandidatePlanWellFormed(remainingBarriers, remainingBundles,
                                   plan_.completionRequirements_)) {
      continue;
    }
    const std::vector<std::size_t> affectedIndices = getUncovered(
        remainingBarriers, remainingBundles, plan_.completionRequirements_);
    if (affectedIndices.empty()) {
      continue;
    }
    std::vector<CanonicalDependency> affectedRequirements;
    affectedRequirements.reserve(affectedIndices.size());
    for (std::size_t index : affectedIndices) {
      if (index < plan_.completionRequirements_.size()) {
        affectedRequirements.push_back(plan_.completionRequirements_[index]);
      }
    }
    if (affectedRequirements.empty()) {
      continue;
    }

    std::vector<AffectedSliceCandidate> candidates;
    for (const CanonicalBarrierCandidate &candidate :
         mechanismUniverse_.barriers) {
      const CanonicalSelectionMechanismRef mechanism{
          CanonicalSelectionMechanismKind::Barrier, candidate.id};
      const bool unavailable =
          containsMechanism(seed.mechanisms, mechanism) ||
          llvm::any_of(remainingBarriers, [&](const CanonicalBarrier &barrier) {
            return barrier.id == candidate.id;
          });
      if (unavailable) {
        continue;
      }
      AffectedSliceCandidate descriptor;
      descriptor.mechanism = mechanism;
      descriptor.barrier = &candidate;
      descriptor.completionEdges =
          buildBarrierCompletionEdges({candidate.barrier});
      for (auto [requirementIndex, requirement] :
           llvm::enumerate(affectedRequirements)) {
        const bool representsRequirement = llvm::any_of(
            candidate.originRequirements, [&](const auto &origin) {
              return sameRequirementIdentity(origin, requirement);
            });
        if (representsRequirement) {
          descriptor.exactRequirements.push_back(requirementIndex);
        }
      }
      descriptor.directCoverage = descriptor.exactRequirements.size();
      descriptor.coveredRequirements = descriptor.exactRequirements;
      descriptor.actionProfile.assign(maxLoopDepth + 1, 0);
      descriptor.barrierProfile =
          buildCanonicalBarrierActionProfile({candidate.barrier}, maxLoopDepth);
      candidates.push_back(std::move(descriptor));
    }
    for (const CanonicalEventBundleCandidate &candidate :
         mechanismUniverse_.eventBundles) {
      const CanonicalSelectionMechanismRef mechanism{
          CanonicalSelectionMechanismKind::EventBundle, candidate.id};
      const bool unavailable =
          containsMechanism(seed.mechanisms, mechanism) ||
          llvm::any_of(remainingBundles, [&](const auto &selected) {
            return selected.id == candidate.id;
          });
      if (unavailable) {
        continue;
      }
      AffectedSliceCandidate descriptor;
      descriptor.mechanism = mechanism;
      descriptor.bundle = &candidate;
      descriptor.hasInactiveOrigins = canonicalMechanismOriginsAreInactive(
          candidate.hasCompleteOriginProvenance, candidate.originRequirements,
          plan_.completionRequirements_);
      descriptor.completionEdges = buildEventCompletionEdges(candidate.events);
      for (auto [requirementIndex, requirement] :
           llvm::enumerate(affectedRequirements)) {
        if (llvm::any_of(candidate.events, [&](const CanonicalEvent &event) {
              return eventDirectlyRepresentsRequirement(event, requirement);
            })) {
          descriptor.exactRequirements.push_back(requirementIndex);
        }
      }
      descriptor.actionProfile.assign(maxLoopDepth + 1, 0);
      for (const CanonicalEvent &event : candidate.events) {
        for (const CanonicalEventAction &action : event.actions) {
          const std::size_t depth =
              getEnclosingLoopDepth(action.anchor.operation);
          ++descriptor.actionProfile[maxLoopDepth - depth];
        }
        const std::size_t span = event.intervalEnd >= event.intervalBegin
                                     ? event.intervalEnd - event.intervalBegin
                                     : event.intervalBegin - event.intervalEnd;
        addSaturated(descriptor.intervalSpan, span);
      }
      descriptor.barrierProfile.assign(maxLoopDepth + 1, 0);
      candidates.push_back(std::move(descriptor));
    }

    std::vector<SyncGraphEdge> potentialEdges = plan_.fixedEdges_;
    std::vector<SyncGraphEdge> retainedBarrierEdges =
        buildBarrierCompletionEdges(remainingBarriers);
    std::vector<SyncGraphEdge> retainedEventEdges = buildEventCompletionEdges(
        flattenCanonicalEventBundles(remainingBundles));
    potentialEdges.insert(potentialEdges.end(), retainedBarrierEdges.begin(),
                          retainedBarrierEdges.end());
    potentialEdges.insert(potentialEdges.end(), retainedEventEdges.begin(),
                          retainedEventEdges.end());
    for (const AffectedSliceCandidate &candidate : candidates) {
      potentialEdges.insert(potentialEdges.end(),
                            candidate.completionEdges.begin(),
                            candidate.completionEdges.end());
    }
    const std::vector<ForwardConnectorCone> forwardCones =
        buildForwardConnectorCones(affectedRequirements, potentialEdges,
                                   plan_.nodes_.size());
    for (AffectedSliceCandidate &candidate : candidates) {
      candidate.potentialCoverage = countForwardConnectorCones(
          candidate.completionEdges, forwardCones, plan_.nodes_.size());
      addSaturated(candidate.potentialCoverage,
                   countTouchedRecurrenceEndpoints(candidate,
                                                   affectedRequirements));
      candidate.connector = candidate.potentialCoverage != 0;
      if (candidate.barrier) {
        candidate.connector |= candidate.directCoverage != 0;
      }
    }
    const auto structuralLess = [](const AffectedSliceCandidate &first,
                                   const AffectedSliceCandidate &second) {
      if (first.potentialCoverage != second.potentialCoverage) {
        return first.potentialCoverage > second.potentialCoverage;
      }
      const bool firstOwnership =
          first.bundle &&
          first.bundle->kind == CanonicalEventBundleKind::Ownership;
      const bool secondOwnership =
          second.bundle &&
          second.bundle->kind == CanonicalEventBundleKind::Ownership;
      if (firstOwnership != secondOwnership) {
        return firstOwnership;
      }
      if (first.actionProfile != second.actionProfile) {
        return first.actionProfile < second.actionProfile;
      }
      if (first.barrierProfile != second.barrierProfile) {
        return first.barrierProfile < second.barrierProfile;
      }
      if (first.intervalSpan != second.intervalSpan) {
        return first.intervalSpan < second.intervalSpan;
      }
      return first.mechanism < second.mechanism;
    };
    const std::size_t doubledAffected =
        affectedRequirements.size() >
                std::numeric_limits<std::size_t>::max() / 2
            ? std::numeric_limits<std::size_t>::max()
            : affectedRequirements.size() * 2;
    const std::size_t structuralLimit =
        std::max(kCombinedCandidateLimit,
                 std::min(kStructuralCandidateLimitMax, doubledAffected));
    const std::size_t rawCandidateCount = candidates.size();
    std::vector<AffectedSliceCandidate> directBarriers;
    std::vector<AffectedSliceCandidate> protectedEvents;
    std::vector<AffectedSliceCandidate> structuralRemainder;
    for (AffectedSliceCandidate &candidate : candidates) {
      if (candidate.barrier && candidate.directCoverage != 0) {
        directBarriers.push_back(std::move(candidate));
      } else if (candidate.bundle &&
                 (candidate.hasInactiveOrigins ||
                  !candidate.exactRequirements.empty())) {
        protectedEvents.push_back(std::move(candidate));
      } else if (candidate.connector) {
        structuralRemainder.push_back(std::move(candidate));
      }
    }
    llvm::stable_sort(directBarriers,
                      [&](const AffectedSliceCandidate &first,
                          const AffectedSliceCandidate &second) {
                        if (first.directCoverage != second.directCoverage) {
                          return first.directCoverage > second.directCoverage;
                        }
                        return structuralLess(first, second);
                      });
    llvm::stable_sort(protectedEvents,
                      [&](const AffectedSliceCandidate &first,
                          const AffectedSliceCandidate &second) {
                        if (first.hasInactiveOrigins !=
                            second.hasInactiveOrigins) {
                          return first.hasInactiveOrigins;
                        }
                        const std::size_t firstExact =
                            first.exactRequirements.size();
                        const std::size_t secondExact =
                            second.exactRequirements.size();
                        if (firstExact != secondExact) {
                          return firstExact > secondExact;
                        }
                        return structuralLess(first, second);
                      });
    llvm::stable_sort(structuralRemainder, structuralLess);
    candidates.clear();
    const std::size_t retainedBarriers =
        std::min(structuralLimit, directBarriers.size());
    candidates.insert(candidates.end(), directBarriers.begin(),
                      directBarriers.begin() + retainedBarriers);
    std::size_t remainingCapacity = structuralLimit - candidates.size();
    const std::size_t retainedProtected =
        std::min(remainingCapacity, protectedEvents.size());
    candidates.insert(candidates.end(), protectedEvents.begin(),
                      protectedEvents.begin() + retainedProtected);
    remainingCapacity = structuralLimit - candidates.size();
    const std::size_t retainedRemainder =
        std::min(remainingCapacity, structuralRemainder.size());
    candidates.insert(candidates.end(), structuralRemainder.begin(),
                      structuralRemainder.begin() + retainedRemainder);
    llvm::stable_sort(candidates, structuralLess);

    for (AffectedSliceCandidate &candidate : candidates) {
      if (!candidate.connector) {
        continue;
      }
      std::vector<CanonicalBarrier> barriers = remainingBarriers;
      std::vector<CanonicalEventBundleCandidate> bundles = remainingBundles;
      if (candidate.barrier) {
        barriers.push_back(candidate.barrier->barrier);
      } else {
        if (!candidate.bundle ||
            !appendCanonicalEventBundleCandidate(bundles, *candidate.bundle)) {
          candidate.connector = false;
          continue;
        }
        const std::vector<CanonicalEvent> events =
            flattenCanonicalEventBundles(bundles);
        if (calculateCanonicalEventColorOverflow(events, eventIdMax_,
                                                 reservedIds_) != 0) {
          candidate.connector = false;
          continue;
        }
      }
      const std::vector<std::size_t> uncovered =
          getUncovered(barriers, bundles, affectedRequirements);
      candidate.directCoverage = affectedRequirements.size() - uncovered.size();
      candidate.coveredRequirements =
          getCoveredIndices(affectedRequirements.size(), uncovered);
    }

    const auto prefilterLess = [](const AffectedSliceCandidate &first,
                                  const AffectedSliceCandidate &second) {
      if (first.directCoverage != second.directCoverage) {
        return first.directCoverage > second.directCoverage;
      }
      const bool firstOwnership =
          first.bundle &&
          first.bundle->kind == CanonicalEventBundleKind::Ownership;
      const bool secondOwnership =
          second.bundle &&
          second.bundle->kind == CanonicalEventBundleKind::Ownership;
      if (firstOwnership != secondOwnership) {
        return firstOwnership;
      }
      if (first.actionProfile != second.actionProfile) {
        return first.actionProfile < second.actionProfile;
      }
      if (first.barrierProfile != second.barrierProfile) {
        return first.barrierProfile < second.barrierProfile;
      }
      if (first.intervalSpan != second.intervalSpan) {
        return first.intervalSpan < second.intervalSpan;
      }
      return first.mechanism < second.mechanism;
    };
    std::vector<AffectedSliceCandidate> directPool;
    std::vector<AffectedSliceCandidate> connectorPool;
    for (AffectedSliceCandidate &candidate : candidates) {
      if (candidate.directCoverage != 0) {
        directPool.push_back(std::move(candidate));
      } else if (candidate.connector) {
        connectorPool.push_back(std::move(candidate));
      }
    }
    llvm::stable_sort(directPool, prefilterLess);
    llvm::stable_sort(connectorPool, prefilterLess);

    std::vector<AffectedSliceCandidate> direct;
    std::vector<CanonicalSelectionMechanismRef> acceptedMechanisms;
    SmallVector<const CanonicalEventBundleCandidate *, 32> acceptedBundles;
    const auto appendExactCandidate = [&](AffectedSliceCandidate candidate) {
      if (containsMechanism(acceptedMechanisms, candidate.mechanism)) {
        return false;
      }
      std::vector<CanonicalBarrier> barriers = remainingBarriers;
      std::vector<CanonicalEventBundleCandidate> bundles = remainingBundles;
      if (candidate.barrier) {
        barriers.push_back(candidate.barrier->barrier);
      } else {
        if (!candidate.bundle ||
            canonicalDiagnosticEventBundleMatchesSelected(
                *candidate.bundle, remainingBundles,
                mechanismUniverse_.eventBundles) ||
            llvm::any_of(acceptedBundles, [&](const auto *accepted) {
              return canonicalDiagnosticEventBundlesEquivalent(
                  *accepted, *candidate.bundle,
                  mechanismUniverse_.eventBundles);
            }) ||
            !appendCanonicalEventBundleCandidate(bundles, *candidate.bundle)) {
          return false;
        }
      }
      if (!isCandidatePlanWellFormed(barriers, bundles,
                                     plan_.completionRequirements_)) {
        return false;
      }
      const std::vector<CanonicalEvent> events =
          flattenCanonicalEventBundles(bundles);
      if (calculateCanonicalEventColorOverflow(events, eventIdMax_,
                                               reservedIds_) != 0) {
        return false;
      }
      candidate.partialScore = scoreCandidatePlan(barriers, bundles);
      if (candidate.bundle) {
        acceptedBundles.push_back(candidate.bundle);
      }
      acceptedMechanisms.push_back(candidate.mechanism);
      direct.push_back(std::move(candidate));
      return true;
    };
    std::vector<std::size_t> directSupport(affectedRequirements.size(), 0);
    for (const AffectedSliceCandidate &candidate : directPool) {
      for (std::size_t requirement : candidate.coveredRequirements) {
        if (requirement < directSupport.size()) {
          ++directSupport[requirement];
        }
      }
    }
    for (const AffectedSliceCandidate &candidate : directPool) {
      const bool isUniqueWitness = llvm::any_of(
          candidate.coveredRequirements, [&](std::size_t requirement) {
            return requirement < directSupport.size() &&
                   directSupport[requirement] == 1;
          });
      if (isUniqueWitness) {
        appendExactCandidate(candidate);
      }
      const std::size_t directSize = direct.size();
      if (directSize == kDirectCandidateLimit) {
        break;
      }
    }
    std::vector<bool> coveredByFrontier(affectedRequirements.size(), false);
    for (const AffectedSliceCandidate &candidate : direct) {
      for (std::size_t requirement : candidate.coveredRequirements) {
        if (requirement < coveredByFrontier.size()) {
          coveredByFrontier[requirement] = true;
        }
      }
    }
    std::vector<bool> attempted(directPool.size(), false);
    std::size_t directSize = direct.size();
    while (directSize < kDirectCandidateLimit) {
      std::optional<std::size_t> best;
      std::size_t bestGain = 0;
      for (auto [index, candidate] : llvm::enumerate(directPool)) {
        if (attempted[index] ||
            containsMechanism(acceptedMechanisms, candidate.mechanism)) {
          continue;
        }
        const std::size_t gain = llvm::count_if(
            candidate.coveredRequirements, [&](std::size_t requirement) {
              return requirement < coveredByFrontier.size() &&
                     !coveredByFrontier[requirement];
            });
        if (!best || gain > bestGain ||
            (gain == bestGain && prefilterLess(candidate, directPool[*best]))) {
          best = index;
          bestGain = gain;
        }
      }
      if (!best) {
        break;
      }
      attempted[*best] = true;
      const AffectedSliceCandidate &candidate = directPool[*best];
      if (!appendExactCandidate(candidate)) {
        continue;
      }
      for (std::size_t requirement : candidate.coveredRequirements) {
        if (requirement < coveredByFrontier.size()) {
          coveredByFrontier[requirement] = true;
        }
      }
      directSize = direct.size();
    }
    const std::size_t directCount = direct.size();
    for (const AffectedSliceCandidate &candidate : connectorPool) {
      const std::size_t connectorCount = direct.size() - directCount;
      if (connectorCount == kConnectorCandidateLimit) {
        break;
      }
      appendExactCandidate(candidate);
    }
    const auto candidateLess = [](const AffectedSliceCandidate &first,
                                  const AffectedSliceCandidate &second) {
      if (first.directCoverage != second.directCoverage) {
        return first.directCoverage > second.directCoverage;
      }
      if (canonicalMechanismPlanScoreLess(first.partialScore,
                                          second.partialScore)) {
        return true;
      }
      if (canonicalMechanismPlanScoreLess(second.partialScore,
                                          first.partialScore)) {
        return false;
      }
      return first.mechanism < second.mechanism;
    };
    llvm::stable_sort(direct, candidateLess);
    const std::size_t selectedCandidateCount = direct.size();
    if (selectedCandidateCount > kCombinedCandidateLimit) {
      direct.resize(kCombinedCandidateLimit);
    }
    LLVM_DEBUG(llvm::dbgs()
               << "affected-slice seed=" << seed.mechanisms.size()
               << " requirements=" << affectedRequirements.size()
               << " raw=" << rawCandidateCount
               << " structural=" << candidates.size()
               << " direct-pool=" << directPool.size()
               << " connector-pool=" << connectorPool.size()
               << " selected=" << direct.size() << '\n');
    LLVM_DEBUG({
      for (const AffectedSliceCandidate &candidate : direct) {
        llvm::dbgs() << "  candidate kind="
                     << static_cast<unsigned>(candidate.mechanism.kind)
                     << " id=" << candidate.mechanism.id
                     << " potential=" << candidate.potentialCoverage
                     << " direct=" << candidate.directCoverage << '\n';
      }
    });
    if (direct.empty()) {
      continue;
    }

    std::vector<CanonicalAffectedSliceSearchCandidate> searchCandidates;
    searchCandidates.reserve(direct.size());
    for (const AffectedSliceCandidate &candidate : direct) {
      searchCandidates.push_back(
          {candidate.mechanism, candidate.coveredRequirements});
    }
    const auto materialize =
        [&](ArrayRef<CanonicalSelectionMechanismRef> mechanisms,
            std::vector<CanonicalBarrier> &barriers,
            std::vector<CanonicalEventBundleCandidate> &bundles) {
          barriers = remainingBarriers;
          bundles = remainingBundles;
          for (const CanonicalSelectionMechanismRef &mechanism : mechanisms) {
            auto descriptor = llvm::find_if(
                direct, [&](const AffectedSliceCandidate &candidate) {
                  return candidate.mechanism == mechanism;
                });
            if (descriptor == direct.end()) {
              return false;
            }
            if (descriptor->barrier) {
              barriers.push_back(descriptor->barrier->barrier);
              continue;
            }
            if (!descriptor->bundle ||
                !appendCanonicalEventBundleCandidate(bundles,
                                                     *descriptor->bundle)) {
              return false;
            }
          }
          return true;
        };
    const auto evaluate =
        [&](ArrayRef<CanonicalSelectionMechanismRef> mechanisms)
        -> std::optional<CanonicalAffectedSliceSearchEvaluation> {
      std::vector<CanonicalBarrier> barriers;
      std::vector<CanonicalEventBundleCandidate> bundles;
      const bool materialized = materialize(mechanisms, barriers, bundles);
      const bool wellFormed =
          materialized && isCandidatePlanWellFormed(
                              barriers, bundles,
                              plan_.completionRequirements_);
      if (!wellFormed) {
        return std::nullopt;
      }
      const std::vector<CanonicalEvent> events =
          flattenCanonicalEventBundles(bundles);
      if (calculateCanonicalEventColorOverflow(events, eventIdMax_,
                                               reservedIds_) != 0) {
        return std::nullopt;
      }
      CanonicalAffectedSliceSearchEvaluation projection;
      projection.uncoveredRequirements =
          getUncovered(barriers, bundles, affectedRequirements);
      projection.score = scoreCandidatePlan(barriers, bundles);
      return projection;
    };
    const std::size_t beamWidth = std::max(
        kBeamWidthMin,
        std::min(kBeamWidthMax, affectedRequirements.size()));
    const std::size_t depthLimit =
        std::min(kSearchDepthLimit, direct.size());
    const std::size_t remainingEvaluationBudget =
        kStateEvaluationLimit - stateEvaluations;
    CanonicalAffectedSliceSearchResult search =
        searchCanonicalAffectedSliceCandidates(
            searchCandidates, affectedRequirements.size(),
            scoreCandidatePlan(remainingBarriers, remainingBundles), beamWidth,
            depthLimit, remainingEvaluationBudget,
            kCompleteCleanupCandidateLimit, evaluate);
    stateEvaluations += search.evaluations;
    stateBudgetExhausted = search.budgetExhausted;
    for (const CanonicalAffectedSliceSearchComplete &complete :
         search.complete) {
      std::vector<CanonicalBarrier> barriers;
      std::vector<CanonicalEventBundleCandidate> bundles;
      if (!materialize(complete.mechanisms, barriers, bundles)) {
        continue;
      }
      std::vector<CanonicalSelectionMechanismRef> signature =
          buildStateSignature(barriers, bundles);
      completeCandidates.push_back(
          {std::move(barriers), std::move(bundles), complete.mechanisms,
           affectedRequirements, complete.score, std::move(signature)});
      llvm::stable_sort(completeCandidates, completeCandidateLess);
      const std::size_t completeCandidateCount = completeCandidates.size();
      if (completeCandidateCount > kCompleteCleanupCandidateLimit) {
        completeCandidates.resize(kCompleteCleanupCandidateLimit);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs()
             << "affected-slice state-evaluations=" << stateEvaluations
             << " budget-exhausted=" << stateBudgetExhausted
             << " cleanup-candidates=" << completeCandidates.size() << '\n');
  for (AffectedSliceCompleteCandidate &candidate : completeCandidates) {
    for (std::size_t index = 0; index < candidate.addedMechanisms.size();) {
      const CanonicalSelectionMechanismRef mechanism =
          candidate.addedMechanisms[index];
      std::vector<CanonicalBarrier> reducedBarriers = candidate.barriers;
      std::vector<CanonicalEventBundleCandidate> reducedBundles =
          candidate.bundles;
      if (mechanism.kind == CanonicalSelectionMechanismKind::Barrier) {
        reducedBarriers.erase(
            std::remove_if(reducedBarriers.begin(), reducedBarriers.end(),
                           [&](const CanonicalBarrier &barrier) {
                             return barrier.id == mechanism.id;
                           }),
            reducedBarriers.end());
      } else {
        reducedBundles.erase(
            std::remove_if(reducedBundles.begin(), reducedBundles.end(),
                           [&](const CanonicalEventBundleCandidate &bundle) {
                             return bundle.id == mechanism.id;
                           }),
            reducedBundles.end());
      }
      if (getUncovered(reducedBarriers, reducedBundles,
                       candidate.affectedRequirements)
              .empty()) {
        candidate.barriers = std::move(reducedBarriers);
        candidate.bundles = std::move(reducedBundles);
        candidate.addedMechanisms.erase(candidate.addedMechanisms.begin() +
                                        index);
        continue;
      }
      ++index;
    }
    CanonicalMechanismPlanScore score =
        scoreCandidatePlan(candidate.barriers, candidate.bundles);
    const bool improvesIncumbent =
        canonicalMechanismPlanScoreLess(score, incumbentScore);
    const bool improvesBest =
        !bestScore || canonicalMechanismPlanScoreLess(score, *bestScore);
    if (improvesIncumbent && improvesBest &&
        isCandidatePlanFeasible(candidate.barriers, candidate.bundles,
                                plan_.completionRequirements_)) {
      bestBarriers = std::move(candidate.barriers);
      bestBundles = std::move(candidate.bundles);
      bestScore = std::move(score);
    }
  }

  if (bestScore) {
    incumbentBarriers = std::move(*bestBarriers);
    incumbentBundles = std::move(*bestBundles);
  }
}
