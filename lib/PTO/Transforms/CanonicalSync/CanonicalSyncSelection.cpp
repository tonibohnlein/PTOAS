// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// This file owns the stable mechanism representation used by CanonicalSync
// selection. Candidate construction is intentionally separate from the final
// CanonicalSyncPlan, which contains only selected barriers and events.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr std::size_t kJointSelectionCandidateLimit = 8;
constexpr std::size_t kJointSelectionRoundLimit = 4;

using BundleKey = std::pair<CanonicalEventBundleKind, std::size_t>;

BundleKey getBundleKey(const CanonicalEvent &event, std::size_t index) {
  if (event.ownershipProtocol && event.ownershipCycle != 0) {
    return {CanonicalEventBundleKind::Ownership, event.ownershipCycle};
  }
  if (event.protocolBundle != 0) {
    return {CanonicalEventBundleKind::SyntheticRoundTrip,
            event.protocolBundle};
  }
  return {CanonicalEventBundleKind::Standalone, index};
}

std::optional<std::pair<const CanonicalEvent *, const CanonicalEvent *>>
getSyntheticRoundTripChain(ArrayRef<const CanonicalEvent *> events) {
  const bool malformedPair =
      events.size() != 2 || !events[0] || !events[1];
  if (malformedPair) {
    return std::nullopt;
  }
  const auto formsChain = [](const CanonicalEvent &sourceToBridge,
                             const CanonicalEvent &bridgeToTarget) {
    return sourceToBridge.target == bridgeToTarget.source &&
           sourceToBridge.targetPipe == bridgeToTarget.sourcePipe &&
           sourceToBridge.sourcePipe == bridgeToTarget.targetPipe &&
           sourceToBridge.source < sourceToBridge.target &&
           sourceToBridge.target < bridgeToTarget.target;
  };
  if (formsChain(*events[0], *events[1])) {
    return std::make_pair(events[0], events[1]);
  }
  if (formsChain(*events[1], *events[0])) {
    return std::make_pair(events[1], events[0]);
  }
  return std::nullopt;
}

bool sameCandidateEventProtocol(const CanonicalEvent &first,
                                const CanonicalEvent &second) {
  return first.source == second.source && first.target == second.target &&
         first.sourcePipe == second.sourcePipe &&
         first.targetPipe == second.targetPipe &&
         first.setAnchor.operation == second.setAnchor.operation &&
         first.setAnchor.before == second.setAnchor.before &&
         first.waitAnchor.operation == second.waitAnchor.operation &&
         first.waitAnchor.before == second.waitAnchor.before &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.forwardDrainLoop == second.forwardDrainLoop &&
         first.iterationDistance == second.iterationDistance &&
         first.setSlot == second.setSlot && first.waitSlot == second.waitSlot &&
         first.width == second.width &&
         first.ownershipCycle == second.ownershipCycle &&
         first.ownershipRole == second.ownershipRole;
}

bool sameAnchor(const CanonicalAnchor &first, const CanonicalAnchor &second) {
  return first.operation == second.operation && first.before == second.before;
}

bool sameEventAction(const CanonicalEventAction &first,
                     const CanonicalEventAction &second) {
  return first.kind == second.kind && first.phase == second.phase &&
         sameAnchor(first.anchor, second.anchor) &&
         first.lane.kind == second.lane.kind &&
         first.lane.index == second.lane.index &&
         first.lane.selector == second.lane.selector &&
         first.nonEmptyLoopGuard == second.nonEmptyLoopGuard;
}

bool sameEventCompletion(const CanonicalEventCompletion &first,
                         const CanonicalEventCompletion &second) {
  return first.source == second.source && first.target == second.target &&
         first.iterationDistance == second.iterationDistance &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.setAction == second.setAction &&
         first.waitAction == second.waitAction;
}

bool sameEventTrace(const CanonicalEventTrace &first,
                    const CanonicalEventTrace &second) {
  return first.kind == second.kind && first.actions == second.actions &&
         first.controlRegion == second.controlRegion &&
         first.hasExplicitTokenState == second.hasExplicitTokenState &&
         first.initialTokens == second.initialTokens &&
         first.expectedTokens == second.expectedTokens;
}

bool sameEventProjection(const CanonicalEvent &first,
                         const CanonicalEvent &second) {
  return first.source == second.source && first.target == second.target &&
         first.sourcePipe == second.sourcePipe &&
         first.targetPipe == second.targetPipe &&
         sameAnchor(first.setAnchor, second.setAnchor) &&
         sameAnchor(first.waitAnchor, second.waitAnchor) &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.forwardDrainLoop == second.forwardDrainLoop &&
         first.scopeLoop == second.scopeLoop &&
         first.iterationDistance == second.iterationDistance &&
         first.setSlot == second.setSlot && first.waitSlot == second.waitSlot &&
         first.width == second.width && first.eventIds == second.eventIds &&
         first.intervalBegin == second.intervalBegin &&
         first.intervalEnd == second.intervalEnd &&
         llvm::equal(first.actions, second.actions, sameEventAction) &&
         llvm::equal(first.completions, second.completions,
                     sameEventCompletion) &&
         llvm::equal(first.traces, second.traces, sameEventTrace) &&
         first.protocolBundle == second.protocolBundle &&
         first.ownershipCycle == second.ownershipCycle &&
         first.ownershipRole == second.ownershipRole &&
         first.ownershipProtocol == second.ownershipProtocol;
}

bool physicalSlotsOverlap(const CanonicalPhysicalSlot &first,
                          const CanonicalPhysicalSlot &second) {
  if (first.space != second.space) {
    return false;
  }
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (first.size == 0 || second.size == 0 ||
      first.address > max - first.size ||
      second.address > max - second.size) {
    return true;
  }
  return std::max(first.address, second.address) <
         std::min(first.address + first.size, second.address + second.size);
}

bool ownershipCyclesConflict(const CanonicalOwnershipCycle &first,
                             const CanonicalOwnershipCycle &second) {
  if (first.loop && second.loop && first.loop != second.loop &&
      !first.loop->isAncestor(second.loop) &&
      !second.loop->isAncestor(first.loop)) {
    return false;
  }
  for (const CanonicalOwnershipLane &firstLane : first.lanes) {
    for (const CanonicalPhysicalSlot &firstSlot : firstLane.slots) {
      for (const CanonicalOwnershipLane &secondLane : second.lanes) {
        for (const CanonicalPhysicalSlot &secondSlot : secondLane.slots) {
          if (physicalSlotsOverlap(firstSlot, secondSlot)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void addBundleConflict(CanonicalEventBundleCandidate &first,
                       CanonicalEventBundleCandidate &second) {
  if (!llvm::is_contained(first.conflicts, second.id)) {
    first.conflicts.push_back(second.id);
  }
  if (!llvm::is_contained(second.conflicts, first.id)) {
    second.conflicts.push_back(first.id);
  }
}

bool candidateBundlesEquivalent(const CanonicalEventBundleCandidate &first,
                                const CanonicalEventBundleCandidate &second) {
  if (first.kind != second.kind ||
      first.events.size() != second.events.size()) {
    return false;
  }
  SmallVector<bool, 2> matched(second.events.size(), false);
  for (const CanonicalEvent &firstEvent : first.events) {
    bool found = false;
    for (auto [index, secondEvent] : llvm::enumerate(second.events)) {
      if (!matched[index] &&
          sameCandidateEventProtocol(firstEvent, secondEvent)) {
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

bool candidateBarriersEquivalent(const CanonicalBarrier &first,
                                 const CanonicalBarrier &second) {
  return first.pipe == second.pipe &&
         first.anchor.operation == second.anchor.operation &&
         first.anchor.before == second.anchor.before &&
         first.recurrenceLoop == second.recurrenceLoop;
}

std::size_t getEnclosingLoopDepth(Operation *operation) {
  std::size_t depth = 0;
  Operation *parent = operation ? operation->getParentOp() : nullptr;
  for (; parent; parent = parent->getParentOp()) {
    depth += isa<LoopLikeOpInterface>(parent) ? 1U : 0U;
  }
  return depth;
}

template <typename T>
void addSaturated(T &total, T value) {
  const T max = std::numeric_limits<T>::max();
  total = value > max - total ? max : total + value;
}

} // namespace

std::vector<CanonicalEventBundleCandidate>
mlir::pto::buildCanonicalEventBundles(ArrayRef<CanonicalEvent> events) {
  std::vector<CanonicalEventBundleCandidate> bundles;
  std::map<BundleKey, std::size_t> grouped;
  for (auto [index, event] : llvm::enumerate(events)) {
    const BundleKey key = getBundleKey(event, index);
    auto [position, inserted] = grouped.emplace(key, bundles.size());
    if (inserted) {
      CanonicalEventBundleCandidate bundle;
      bundle.id = bundles.size();
      bundle.protocolIdentity = key.second;
      bundle.kind = key.first;
      bundles.push_back(std::move(bundle));
    }
    bundles[position->second].events.push_back(event);
  }
  return bundles;
}

std::vector<CanonicalEvent> mlir::pto::flattenCanonicalEventBundles(
    ArrayRef<CanonicalEventBundleCandidate> bundles) {
  std::vector<CanonicalEvent> events;
  for (const CanonicalEventBundleCandidate &bundle : bundles) {
    events.insert(events.end(), bundle.events.begin(), bundle.events.end());
  }
  return events;
}

bool mlir::pto::canonicalEventBundleProjectionMatches(
    ArrayRef<CanonicalEventBundleCandidate> bundles,
    ArrayRef<CanonicalEvent> events) {
  const std::vector<CanonicalEvent> projected =
      flattenCanonicalEventBundles(bundles);
  return llvm::equal(projected, events, sameEventProjection);
}

LogicalResult mlir::pto::restoreCanonicalEventBundleIdentities(
    std::vector<CanonicalEventBundleCandidate> &bundles,
    ArrayRef<CanonicalEventBundleCandidate> knownBundles,
    std::size_t &nextFreshId) {
  for (const CanonicalEventBundleCandidate &known : knownBundles) {
    if (known.id == std::numeric_limits<std::size_t>::max()) {
      return failure();
    }
    nextFreshId = std::max(nextFreshId, known.id + 1);
  }
  std::set<std::size_t> usedIds;
  for (CanonicalEventBundleCandidate &bundle : bundles) {
    auto known = llvm::find_if(knownBundles, [&](const auto &candidate) {
      return !llvm::is_contained(usedIds, candidate.id) &&
             candidateBundlesEquivalent(bundle, candidate);
    });
    if (known != knownBundles.end()) {
      bundle.id = known->id;
      bundle.protocolIdentity = known->protocolIdentity;
      bundle.conflicts = known->conflicts;
      bundle.completionWitness = known->completionWitness;
    } else {
      while (llvm::is_contained(usedIds, nextFreshId)) {
        if (nextFreshId == std::numeric_limits<std::size_t>::max()) {
          return failure();
        }
        ++nextFreshId;
      }
      if (nextFreshId == std::numeric_limits<std::size_t>::max()) {
        return failure();
      }
      bundle.id = nextFreshId++;
      bundle.conflicts.clear();
      bundle.completionWitness.reset();
    }
    usedIds.insert(bundle.id);
  }
  return success();
}

bool mlir::pto::verifyCanonicalSyntheticRoundTripBundle(
    ArrayRef<const CanonicalEvent *> events) {
  const bool malformedPair =
      events.size() != 2 || !events[0] || !events[1];
  if (malformedPair) {
    return false;
  }
  const CanonicalEvent &first = *events[0];
  const CanonicalEvent &second = *events[1];
  if (first.ownershipProtocol || second.ownershipProtocol ||
      first.protocolBundle == 0 ||
      first.protocolBundle != second.protocolBundle || first.width != 1 ||
      second.width != 1 || first.recurrenceLoop || second.recurrenceLoop) {
    return false;
  }

  return getSyntheticRoundTripChain(events).has_value();
}

bool mlir::pto::verifyCanonicalSyntheticRoundTripWitness(
    ArrayRef<const CanonicalEvent *> events,
    const CanonicalDependency &requirement) {
  const auto chain = getSyntheticRoundTripChain(events);
  return chain && requirement.iterationDistance == 0 &&
         !requirement.recurrenceLoop &&
         requirement.source == chain->first->source &&
         requirement.target == chain->second->target;
}

bool mlir::pto::canonicalEventBundlesHaveNoConflicts(
    ArrayRef<CanonicalEventBundleCandidate> bundles) {
  std::set<std::size_t> selectedBundleIds;
  for (const CanonicalEventBundleCandidate &bundle : bundles) {
    if (!selectedBundleIds.insert(bundle.id).second) {
      return false;
    }
  }
  return llvm::none_of(bundles, [&](const auto &bundle) {
    return llvm::any_of(bundle.conflicts, [&](std::size_t conflict) {
      return selectedBundleIds.find(conflict) != selectedBundleIds.end();
    });
  });
}

bool mlir::pto::exchangeCanonicalEventBundleCandidate(
    std::vector<CanonicalEventBundleCandidate> &selected,
    const CanonicalEventBundleCandidate &candidate) {
  const bool alreadySelected = llvm::any_of(
      selected, [&](const CanonicalEventBundleCandidate &bundle) {
        return bundle.id == candidate.id ||
               candidateBundlesEquivalent(bundle, candidate);
      });
  if (alreadySelected) {
    return false;
  }
  selected.erase(
      std::remove_if(
          selected.begin(), selected.end(),
          [&](const CanonicalEventBundleCandidate &bundle) {
            return llvm::is_contained(candidate.conflicts, bundle.id) ||
                   llvm::is_contained(bundle.conflicts, candidate.id);
          }),
      selected.end());
  selected.push_back(candidate);
  return true;
}

std::size_t mlir::pto::calculateCanonicalEventColorOverflow(
    ArrayRef<CanonicalEvent> events, unsigned eventIdMax,
    const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds) {
  std::map<CanonicalEventDomainKey, std::vector<SyncInterval>> intervals;
  for (const CanonicalEvent &event : events) {
    auto &domain = intervals[{event.sourcePipe, event.targetPipe}];
    for (unsigned lane = 0; lane < event.width; ++lane) {
      domain.push_back({event.intervalBegin, event.intervalEnd});
    }
  }

  std::size_t overflow = 0;
  for (const auto &entry : intervals) {
    unsigned reserved = 0;
    auto reservedIt = reservedIds.find(entry.first);
    if (reservedIt != reservedIds.end()) {
      for (unsigned eventId : reservedIt->second) {
        reserved += eventId < eventIdMax ? 1U : 0U;
      }
    }
    const unsigned colors = colorSyncIntervals(entry.second).colorCount;
    const unsigned available = eventIdMax - reserved;
    if (colors > available) {
      addSaturated(overflow,
                   static_cast<std::size_t>(colors - available));
    }
  }
  return overflow;
}

std::vector<std::size_t> mlir::pto::buildCanonicalBarrierActionProfile(
    ArrayRef<CanonicalBarrier> barriers, std::size_t maxLoopDepth) {
  std::vector<std::size_t> profile(maxLoopDepth + 1, 0);
  for (const CanonicalBarrier &barrier : barriers) {
    const std::size_t depth =
        getEnclosingLoopDepth(barrier.anchor.operation);
    const std::size_t profileIndex = maxLoopDepth - depth;
    ++profile[profileIndex];
  }
  return profile;
}

bool mlir::pto::canonicalMechanismPlanScoreLess(
    const CanonicalMechanismPlanScore &first,
    const CanonicalMechanismPlanScore &second) {
  if (first.usefulOwnershipBundles != second.usefulOwnershipBundles) {
    return first.usefulOwnershipBundles > second.usefulOwnershipBundles;
  }
  return std::tie(first.ownershipSignature, first.dynamicActionProfile,
                  first.barrierActionProfile,
                  first.waitDistance, first.intervalSpan,
                  first.peakColorPressure, first.directedDomains,
                  first.barrierCount, first.candidateSignature) <
         std::tie(second.ownershipSignature, second.dynamicActionProfile,
                  second.barrierActionProfile,
                  second.waitDistance, second.intervalSpan,
                  second.peakColorPressure, second.directedDomains,
                  second.barrierCount, second.candidateSignature);
}

SmallVector<const CanonicalEventBundleCandidate *, 16>
mlir::pto::selectCanonicalEventCandidateFrontier(
    ArrayRef<const CanonicalEventBundleCandidate *> candidates,
    std::size_t limit) {
  SmallVector<const CanonicalEventBundleCandidate *, 16> frontier(
      candidates.begin(), candidates.end());
  llvm::stable_sort(frontier, [](const auto *first, const auto *second) {
    const auto priority = [](CanonicalEventBundleKind kind) {
      switch (kind) {
      case CanonicalEventBundleKind::Ownership:
        return 0;
      case CanonicalEventBundleKind::SyntheticRoundTrip:
        return 1;
      case CanonicalEventBundleKind::Standalone:
        return 2;
      }
      return 3;
    };
    return std::make_tuple(priority(first->kind), first->id) <
           std::make_tuple(priority(second->kind), second->id);
  });
  const bool exceedsLimit = frontier.size() > limit;
  if (exceedsLimit) {
    frontier.resize(limit);
  }
  return frontier;
}

void CanonicalSyncPlanBuilder::buildMechanismUniverse() {
  mechanismUniverse_ = {};
  DenseMap<Operation *, std::size_t> recurrenceScopeIds;
  std::size_t nextRecurrenceScopeId = 1;
  funcOperation_->walk([&](Operation *operation) {
    if (isa<LoopLikeOpInterface>(operation)) {
      recurrenceScopeIds[operation] = nextRecurrenceScopeId++;
    }
  });
  for (auto [requirementId, requirement] :
       llvm::enumerate(plan_.conservativeCompletionRequirements_)) {
    const CanonicalSyncNode &source = plan_.nodes_[requirement.source];
    const CanonicalSyncNode &target = plan_.nodes_[requirement.target];
    const bool needsBarrierCandidate =
        source.pipe == target.pipe && !hasHardwareCompletion(source.pipe) &&
        !hasIntrinsicMmadAccumulatorOrdering(requirement);
    if (!needsBarrierCandidate) {
      continue;
    }

    CanonicalBarrier barrier;
    barrier.pipe = source.pipe;
    barrier.anchor = requirement.iterationDistance == 0
                         ? getWaitAnchor(source.operation, target.operation)
                         : CanonicalAnchor{target.operation, true};
    auto anchorNodes = operationNodes_.find(barrier.anchor.operation);
    if (anchorNodes != operationNodes_.end()) {
      barrier.anchorNodes.append(anchorNodes->second.begin(),
                                 anchorNodes->second.end());
    }
    barrier.recurrenceLoop = requirement.recurrenceLoop;
    barrier.recurrenceScope =
        recurrenceScopeIds.lookup(requirement.recurrenceLoop);
    barrier.requirements.push_back(requirementId);

    auto duplicate = llvm::find_if(
        mechanismUniverse_.barriers, [&](const auto &candidate) {
          const CanonicalBarrier &old = candidate.barrier;
          return old.pipe == barrier.pipe &&
                 old.anchor.operation == barrier.anchor.operation &&
                 old.anchor.before == barrier.anchor.before &&
                 old.recurrenceLoop == barrier.recurrenceLoop;
        });
    if (duplicate != mechanismUniverse_.barriers.end()) {
      duplicate->barrier.requirements.push_back(requirementId);
      continue;
    }
    CanonicalBarrierCandidate candidate;
    candidate.id = mechanismUniverse_.barriers.size();
    barrier.id = candidate.id;
    candidate.barrier = barrier;
    mechanismUniverse_.barriers.push_back(std::move(candidate));
  }

  std::vector<CanonicalEvent> conservativeEvents;
  materializeEventsFrom(plan_.conservativeCompletionRequirements_,
                        conservativeEvents);
  mechanismUniverse_.eventBundles =
      buildCanonicalEventBundles(conservativeEvents);
  const std::size_t standaloneBundleCount =
      mechanismUniverse_.eventBundles.size();
  std::size_t nextProtocolIdentity = 1;
  for (const CanonicalDependency &requirement :
       plan_.conservativeCompletionRequirements_) {
    const CanonicalSyncNode &source = plan_.nodes_[requirement.source];
    const CanonicalSyncNode &target = plan_.nodes_[requirement.target];
    if (requirement.iterationDistance != 0 || source.pipe != target.pipe ||
        hasHardwareCompletion(source.pipe) ||
        hasIntrinsicMmadAccumulatorOrdering(requirement)) {
      continue;
    }
    for (std::size_t index = 0; index < standaloneBundleCount; ++index) {
      CanonicalEventBundleCandidate &incomingBundle =
          mechanismUniverse_.eventBundles[index];
      const bool isStandaloneCandidate = incomingBundle.events.size() == 1;
      if (!isStandaloneCandidate) {
        continue;
      }
      const CanonicalEvent &incoming = incomingBundle.events.front();
      if (incoming.recurrenceLoop || incoming.targetPipe != source.pipe ||
          incoming.target != requirement.target ||
          requirement.source >= incoming.source ||
          incoming.source >= requirement.target) {
        continue;
      }

      CanonicalEvent roundTrip =
          makeForwardEvent(requirement.source, incoming.source);
      if (roundTrip.sourcePipe == roundTrip.targetPipe) {
        continue;
      }
      CanonicalEvent incomingCopy = incoming;
      roundTrip.protocolBundle = nextProtocolIdentity;
      incomingCopy.protocolBundle = nextProtocolIdentity;

      CanonicalEventBundleCandidate bundle;
      bundle.id = mechanismUniverse_.eventBundles.size();
      bundle.protocolIdentity = nextProtocolIdentity++;
      bundle.kind = CanonicalEventBundleKind::SyntheticRoundTrip;
      bundle.events.push_back(std::move(roundTrip));
      bundle.events.push_back(std::move(incomingCopy));
      bundle.completionWitness = requirement;
      addBundleConflict(incomingBundle, bundle);
      mechanismUniverse_.eventBundles.push_back(std::move(bundle));
    }
  }

  for (const CanonicalOwnershipCycle &cycle : plan_.ownershipCycles_) {
    std::optional<CanonicalEventBundleCandidate> bundle =
        buildOwnershipEventBundle(cycle);
    if (!bundle) {
      continue;
    }
    bundle->id = mechanismUniverse_.eventBundles.size();
    mechanismUniverse_.eventBundles.push_back(std::move(*bundle));
  }

  for (std::size_t first = 0;
       first < mechanismUniverse_.eventBundles.size(); ++first) {
    CanonicalEventBundleCandidate &firstBundle =
        mechanismUniverse_.eventBundles[first];
    for (std::size_t second = first + 1;
         second < mechanismUniverse_.eventBundles.size(); ++second) {
      CanonicalEventBundleCandidate &secondBundle =
          mechanismUniverse_.eventBundles[second];
      const bool sharesProtocol = llvm::any_of(
          firstBundle.events, [&](const CanonicalEvent &firstEvent) {
            return llvm::any_of(
                secondBundle.events, [&](const CanonicalEvent &secondEvent) {
                  return sameCandidateEventProtocol(firstEvent, secondEvent);
                });
          });
      bool sharesOwnership = false;
      if (firstBundle.kind == CanonicalEventBundleKind::Ownership &&
          secondBundle.kind == CanonicalEventBundleKind::Ownership) {
        auto firstCycle = llvm::find_if(
            plan_.ownershipCycles_, [&](const CanonicalOwnershipCycle &cycle) {
              return cycle.id == firstBundle.protocolIdentity;
            });
        auto secondCycle = llvm::find_if(
            plan_.ownershipCycles_, [&](const CanonicalOwnershipCycle &cycle) {
              return cycle.id == secondBundle.protocolIdentity;
            });
        sharesOwnership = firstCycle != plan_.ownershipCycles_.end() &&
                          secondCycle != plan_.ownershipCycles_.end() &&
                          ownershipCyclesConflict(*firstCycle, *secondCycle);
      }
      if (sharesProtocol || sharesOwnership) {
        addBundleConflict(firstBundle, secondBundle);
      }
    }
  }
}

LogicalResult CanonicalSyncPlanBuilder::refreshSelectedEventBundles() {
  std::vector<CanonicalEventBundleCandidate> knownBundles =
      selectedEventBundles_;
  knownBundles.insert(knownBundles.end(),
                      mechanismUniverse_.eventBundles.begin(),
                      mechanismUniverse_.eventBundles.end());
  std::vector<CanonicalEventBundleCandidate> refreshed =
      buildCanonicalEventBundles(plan_.events_);
  std::size_t nextFreshId = mechanismUniverse_.eventBundles.size();
  if (failed(restoreCanonicalEventBundleIdentities(
          refreshed, knownBundles, nextFreshId))) {
    return func_.emitError(
        "canonical event bundle identity space is exhausted");
  }
  selectedEventBundles_ = std::move(refreshed);
  plan_.events_ = flattenCanonicalEventBundles(selectedEventBundles_);
  return success();
}

bool CanonicalSyncPlanBuilder::tryBuildConservativeIncumbent(
    std::vector<CanonicalBarrier> &barriers,
    std::vector<CanonicalEvent> &events) {
  const std::vector<CanonicalDependency> activeDependencies =
      plan_.dependencies_;
  const std::vector<CanonicalDependency> activeRequirements =
      plan_.completionRequirements_;
  const std::vector<CanonicalRecurrenceScope> activeRecurrenceScopes =
      plan_.recurrenceScopes_;
  const std::vector<CanonicalBarrier> activeBarriers = plan_.barriers_;
  const std::vector<CanonicalEvent> activeEvents = plan_.events_;
  const std::vector<CanonicalEvent> activeEventCandidates = eventCandidates_;

  plan_.dependencies_ = plan_.conservativeCompletionRequirements_;
  plan_.completionRequirements_ = plan_.conservativeCompletionRequirements_;
  plan_.recurrenceScopes_.clear();
  plan_.barriers_.clear();
  plan_.events_.clear();
  eventCandidates_.clear();

  reduceForwardDependencies();
  materializeBarriers();
  materializeEvents();
  synthesizeOwnershipProtocols();

  const bool feasible = isCandidatePlanFeasible(
      plan_.barriers_, buildCanonicalEventBundles(plan_.events_),
      plan_.conservativeCompletionRequirements_);
  if (feasible) {
    barriers = plan_.barriers_;
    events = plan_.events_;
  }

  plan_.dependencies_ = activeDependencies;
  plan_.completionRequirements_ = activeRequirements;
  plan_.recurrenceScopes_ = activeRecurrenceScopes;
  plan_.barriers_ = activeBarriers;
  plan_.events_ = activeEvents;
  eventCandidates_ = activeEventCandidates;
  return feasible;
}

void CanonicalSyncPlanBuilder::removeRedundantMechanisms() {
  std::vector<CanonicalBarrier> barriers = plan_.barriers_;
  std::vector<CanonicalEventBundleCandidate> bundles =
      buildCanonicalEventBundles(plan_.events_);

  for (std::size_t index = 0; index < barriers.size();) {
    std::vector<CanonicalBarrier> reduced = barriers;
    reduced.erase(reduced.begin() + index);
    if (isCandidatePlanFeasible(reduced, bundles,
                                plan_.completionRequirements_)) {
      barriers = std::move(reduced);
      continue;
    }
    ++index;
  }

  for (std::size_t index = 0; index < bundles.size();) {
    std::vector<CanonicalEventBundleCandidate> reduced = bundles;
    reduced.erase(reduced.begin() + index);
    if (isCandidatePlanFeasible(barriers, reduced,
                                plan_.completionRequirements_)) {
      bundles = std::move(reduced);
      continue;
    }
    ++index;
  }

  plan_.barriers_ = std::move(barriers);
  plan_.events_ = flattenCanonicalEventBundles(bundles);
}

CanonicalMechanismPlanScore CanonicalSyncPlanBuilder::scoreCandidatePlan(
    ArrayRef<CanonicalBarrier> barriers,
    ArrayRef<CanonicalEventBundleCandidate> eventBundles) const {
  CanonicalMechanismPlanScore score;
  std::size_t maxLoopDepth = 0;
  funcOperation_->walk([&](Operation *operation) {
    maxLoopDepth =
        std::max(maxLoopDepth, getEnclosingLoopDepth(operation));
  });
  score.dynamicActionProfile.assign(maxLoopDepth + 1, 0);

  std::map<CanonicalEventDomainKey, std::vector<SyncInterval>> intervals;
  for (const CanonicalEventBundleCandidate &bundle : eventBundles) {
    score.candidateSignature.push_back(bundle.id);
    if (bundle.kind == CanonicalEventBundleKind::Ownership) {
      ++score.usefulOwnershipBundles;
      score.ownershipSignature.push_back(bundle.protocolIdentity);
    }
    for (const CanonicalEvent &event : bundle.events) {
      for (const CanonicalEventAction &action : event.actions) {
        const std::size_t depth = getEnclosingLoopDepth(action.anchor.operation);
        const std::size_t profileIndex = maxLoopDepth - depth;
        ++score.dynamicActionProfile[profileIndex];
      }
      const std::size_t intervalSpan =
          event.intervalEnd >= event.intervalBegin
              ? event.intervalEnd - event.intervalBegin
              : event.intervalBegin - event.intervalEnd;
      addSaturated(score.intervalSpan, intervalSpan);
      if (event.target < plan_.nodes_.size()) {
        const std::size_t waitPosition = getAnchorPosition(event.waitAnchor);
        const std::size_t targetPosition = plan_.nodes_[event.target].order * 2;
        addSaturated(score.waitDistance,
                     waitPosition >= targetPosition
                         ? waitPosition - targetPosition
                         : targetPosition - waitPosition);
      }
      auto &domain = intervals[{event.sourcePipe, event.targetPipe}];
      for (unsigned lane = 0; lane < event.width; ++lane) {
        domain.push_back({event.intervalBegin, event.intervalEnd});
      }
    }
  }

  llvm::sort(score.ownershipSignature);
  llvm::sort(score.candidateSignature);
  score.directedDomains = intervals.size();
  for (const auto &entry : intervals) {
    score.peakColorPressure =
        std::max(score.peakColorPressure,
                 colorSyncIntervals(entry.second).colorCount);
  }
  score.barrierCount = barriers.size();
  score.barrierActionProfile =
      buildCanonicalBarrierActionProfile(barriers, maxLoopDepth);
  score.candidateSignature.push_back(std::numeric_limits<std::size_t>::max());
  for (const CanonicalBarrier &barrier : barriers) {
    score.candidateSignature.push_back(barrier.id);
  }
  return score;
}

bool CanonicalSyncPlanBuilder::bootstrapFeasibleMechanismPlan(
    std::vector<CanonicalBarrier> &barriers,
    std::vector<CanonicalEventBundleCandidate> &eventBundles) const {
  const auto isCoveredAndWellFormed =
      [&](ArrayRef<CanonicalBarrier> candidateBarriers,
          ArrayRef<CanonicalEventBundleCandidate> candidateBundles) {
        if (!isCandidatePlanWellFormed(candidateBarriers, candidateBundles,
                                       plan_.completionRequirements_)) {
          return false;
        }
        const std::vector<CanonicalEvent> events =
            flattenCanonicalEventBundles(candidateBundles);
        return countUncoveredRequirements(candidateBarriers, events,
                                          plan_.completionRequirements_) == 0;
      };

  const auto reduceCovered =
      [&](std::vector<CanonicalBarrier> &candidateBarriers,
          std::vector<CanonicalEventBundleCandidate> &candidateBundles,
          std::optional<std::size_t> protectedBarrier,
          std::optional<std::size_t> protectedBundle) {
        for (std::size_t index = 0; index < candidateBarriers.size();) {
          if (protectedBarrier &&
              candidateBarriers[index].id == *protectedBarrier) {
            ++index;
            continue;
          }
          std::vector<CanonicalBarrier> reduced = candidateBarriers;
          reduced.erase(reduced.begin() + index);
          if (isCoveredAndWellFormed(reduced, candidateBundles)) {
            candidateBarriers = std::move(reduced);
            continue;
          }
          ++index;
        }
        for (std::size_t index = 0; index < candidateBundles.size();) {
          if (protectedBundle &&
              candidateBundles[index].id == *protectedBundle) {
            ++index;
            continue;
          }
          std::vector<CanonicalEventBundleCandidate> reduced =
              candidateBundles;
          reduced.erase(reduced.begin() + index);
          if (isCoveredAndWellFormed(candidateBarriers, reduced)) {
            candidateBundles = std::move(reduced);
            continue;
          }
          ++index;
        }
      };

  struct BootstrapState {
    std::vector<CanonicalBarrier> barriers;
    std::vector<CanonicalEventBundleCandidate> bundles;
    std::size_t colorOverflow = 0;
    std::size_t eventLanes = 0;
    std::vector<std::size_t> signature;
  };

  const auto evaluate = [&](BootstrapState &state) {
    if (!isCoveredAndWellFormed(state.barriers, state.bundles)) {
      return false;
    }
    const std::vector<CanonicalEvent> events =
        flattenCanonicalEventBundles(state.bundles);
    for (const CanonicalEvent &event : events) {
      addSaturated(state.eventLanes, static_cast<std::size_t>(event.width));
    }
    state.colorOverflow = calculateCanonicalEventColorOverflow(
        events, eventIdMax_, reservedIds_);
    for (const CanonicalEventBundleCandidate &bundle : state.bundles) {
      state.signature.push_back(bundle.id);
    }
    llvm::sort(state.signature);
    state.signature.push_back(std::numeric_limits<std::size_t>::max());
    SmallVector<std::size_t, 8> barrierIds;
    llvm::transform(state.barriers, std::back_inserter(barrierIds),
                    [](const CanonicalBarrier &barrier) { return barrier.id; });
    llvm::sort(barrierIds);
    state.signature.insert(state.signature.end(), barrierIds.begin(),
                           barrierIds.end());
    return true;
  };

  const auto stateLess = [](const BootstrapState &first,
                            const BootstrapState &second) {
    const std::size_t firstMechanisms =
        first.barriers.size() + first.bundles.size();
    const std::size_t secondMechanisms =
        second.barriers.size() + second.bundles.size();
    return std::make_tuple(first.colorOverflow, first.eventLanes,
                           firstMechanisms, first.signature) <
           std::make_tuple(second.colorOverflow, second.eventLanes,
                           secondMechanisms, second.signature);
  };

  reduceCovered(barriers, eventBundles, std::nullopt, std::nullopt);
  if (isCandidatePlanFeasible(barriers, eventBundles,
                              plan_.completionRequirements_)) {
    return true;
  }

  BootstrapState initial{barriers, eventBundles};
  if (!evaluate(initial)) {
    return false;
  }
  std::vector<BootstrapState> frontier{std::move(initial)};
  std::set<std::vector<std::size_t>> seen{frontier.front().signature};
  const bool largeRequirementSet =
      plan_.completionRequirements_.size() > 1024;
  const std::size_t candidateLimit =
      largeRequirementSet ? 1 : kJointSelectionCandidateLimit;
  const std::size_t roundLimit =
      largeRequirementSet ? 1 : kJointSelectionRoundLimit;

  for (std::size_t round = 0; round < roundLimit; ++round) {
    std::vector<BootstrapState> next;
    std::optional<BootstrapState> bestFeasible;
    std::optional<CanonicalMechanismPlanScore> bestFeasibleScore;
    const auto consider = [&](BootstrapState candidate) {
      const bool invalidOrDuplicate =
          !evaluate(candidate) || !seen.insert(candidate.signature).second;
      if (invalidOrDuplicate) {
        return;
      }
      if (!isCandidatePlanFeasible(candidate.barriers, candidate.bundles,
                                   plan_.completionRequirements_)) {
        next.push_back(std::move(candidate));
        return;
      }
      CanonicalMechanismPlanScore score =
          scoreCandidatePlan(candidate.barriers, candidate.bundles);
      if (!bestFeasibleScore ||
          canonicalMechanismPlanScoreLess(score, *bestFeasibleScore)) {
        bestFeasible = std::move(candidate);
        bestFeasibleScore = std::move(score);
      }
    };

    for (const BootstrapState &state : frontier) {
      SmallVector<const CanonicalEventBundleCandidate *, 16> eventCandidates;
      for (const CanonicalEventBundleCandidate &candidate :
           mechanismUniverse_.eventBundles) {
        const bool selected = llvm::any_of(
            state.bundles, [&](const auto &bundle) {
              return bundle.id == candidate.id ||
                     candidateBundlesEquivalent(bundle, candidate);
            });
        if (!selected) {
          eventCandidates.push_back(&candidate);
        }
      }
      eventCandidates =
          selectCanonicalEventCandidateFrontier(eventCandidates, candidateLimit);
      for (const CanonicalEventBundleCandidate *candidate : eventCandidates) {
        BootstrapState successor{state.barriers, state.bundles};
        if (!exchangeCanonicalEventBundleCandidate(successor.bundles,
                                                   *candidate)) {
          continue;
        }
        reduceCovered(successor.barriers, successor.bundles, std::nullopt,
                      candidate->id);
        consider(std::move(successor));
      }

      std::size_t consideredBarriers = 0;
      for (const CanonicalBarrierCandidate &candidate :
           mechanismUniverse_.barriers) {
        const bool selected = llvm::any_of(
            state.barriers, [&](const CanonicalBarrier &barrier) {
              return candidateBarriersEquivalent(barrier, candidate.barrier);
            });
        if (selected || consideredBarriers == candidateLimit) {
          continue;
        }
        ++consideredBarriers;
        BootstrapState successor{state.barriers, state.bundles};
        successor.barriers.push_back(candidate.barrier);
        reduceCovered(successor.barriers, successor.bundles, candidate.id,
                      std::nullopt);
        consider(std::move(successor));
      }
    }

    if (bestFeasible) {
      barriers = std::move(bestFeasible->barriers);
      eventBundles = std::move(bestFeasible->bundles);
      return true;
    }
    llvm::stable_sort(next, stateLess);
    const bool exceedsBeamLimit =
        next.size() > kJointSelectionCandidateLimit;
    if (exceedsBeamLimit) {
      next.resize(kJointSelectionCandidateLimit);
    }
    frontier = std::move(next);
    if (frontier.empty()) {
      break;
    }
  }
  return false;
}

LogicalResult CanonicalSyncPlanBuilder::optimizeMechanismSelection() {
  std::vector<CanonicalBarrier> incumbentBarriers = plan_.barriers_;
  std::vector<CanonicalEventBundleCandidate> incumbentBundles =
      buildCanonicalEventBundles(plan_.events_);

  std::size_t nextBundleId = mechanismUniverse_.eventBundles.size();
  if (failed(restoreCanonicalEventBundleIdentities(
          incumbentBundles, mechanismUniverse_.eventBundles,
          nextBundleId))) {
    return func_.emitError(
        "canonical event bundle identity space is exhausted");
  }
  std::size_t nextBarrierId = mechanismUniverse_.barriers.size();
  for (CanonicalBarrier &barrier : incumbentBarriers) {
    auto match = llvm::find_if(
        mechanismUniverse_.barriers, [&](const auto &candidate) {
          return candidateBarriersEquivalent(barrier, candidate.barrier);
        });
    barrier.id = match == mechanismUniverse_.barriers.end()
                     ? nextBarrierId++
                     : match->id;
  }

  const bool hasFeasibleIncumbent = isCandidatePlanFeasible(
      incumbentBarriers, incumbentBundles, plan_.completionRequirements_);
  if (!hasFeasibleIncumbent) {
    if (!bootstrapFeasibleMechanismPlan(incumbentBarriers,
                                        incumbentBundles)) {
      selectedEventBundles_ = incumbentBundles;
      plan_.events_ = flattenCanonicalEventBundles(selectedEventBundles_);
      return success();
    }
    plan_.usedInfeasibleBootstrap_ = true;
  }

  const auto reduce = [&](std::vector<CanonicalBarrier> &barriers,
                          std::vector<CanonicalEventBundleCandidate> &bundles) {
    for (std::size_t index = 0; index < barriers.size();) {
      std::vector<CanonicalBarrier> candidate = barriers;
      candidate.erase(candidate.begin() + index);
      if (isCandidatePlanFeasible(candidate, bundles,
                                  plan_.completionRequirements_)) {
        barriers = std::move(candidate);
        continue;
      }
      ++index;
    }
    for (std::size_t index = 0; index < bundles.size();) {
      std::vector<CanonicalEventBundleCandidate> candidate = bundles;
      candidate.erase(candidate.begin() + index);
      if (isCandidatePlanFeasible(barriers, candidate,
                                  plan_.completionRequirements_)) {
        bundles = std::move(candidate);
        continue;
      }
      ++index;
    }
  };

  reduce(incumbentBarriers, incumbentBundles);
  CanonicalMechanismPlanScore incumbentScore =
      scoreCandidatePlan(incumbentBarriers, incumbentBundles);
  const bool largeRequirementSet =
      plan_.completionRequirements_.size() > 1024;
  const std::size_t candidateLimit =
      largeRequirementSet ? 1 : kJointSelectionCandidateLimit;
  const std::size_t roundLimit =
      largeRequirementSet ? 1 : kJointSelectionRoundLimit;

  for (std::size_t round = 0; round < roundLimit; ++round) {
    std::optional<std::vector<CanonicalBarrier>> bestBarriers;
    std::optional<std::vector<CanonicalEventBundleCandidate>> bestBundles;
    std::optional<CanonicalMechanismPlanScore> bestScore;

    SmallVector<const CanonicalEventBundleCandidate *, 16> eventCandidates;
    for (const CanonicalEventBundleCandidate &candidate :
         mechanismUniverse_.eventBundles) {
      const bool selected = llvm::any_of(
          incumbentBundles, [&](const auto &bundle) {
            return bundle.id == candidate.id ||
                   candidateBundlesEquivalent(bundle, candidate);
          });
      if (!selected) {
        eventCandidates.push_back(&candidate);
      }
    }
    eventCandidates =
        selectCanonicalEventCandidateFrontier(eventCandidates, candidateLimit);

    const auto consider = [&](std::vector<CanonicalBarrier> barriers,
                              std::vector<CanonicalEventBundleCandidate>
                                  bundles) {
      reduce(barriers, bundles);
      if (!isCandidatePlanFeasible(barriers, bundles,
                                   plan_.completionRequirements_)) {
        return;
      }
      CanonicalMechanismPlanScore score =
          scoreCandidatePlan(barriers, bundles);
      const bool improvesIncumbent =
          canonicalMechanismPlanScoreLess(score, incumbentScore);
      const bool improvesBest =
          !bestScore || canonicalMechanismPlanScoreLess(score, *bestScore);
      if (!improvesIncumbent || !improvesBest) {
        return;
      }
      bestScore = std::move(score);
      bestBarriers = std::move(barriers);
      bestBundles = std::move(bundles);
    };

    for (const CanonicalEventBundleCandidate *candidate : eventCandidates) {
      std::vector<CanonicalEventBundleCandidate> bundles = incumbentBundles;
      bundles.push_back(*candidate);
      if (canonicalEventBundlesHaveNoConflicts(bundles)) {
        consider(incumbentBarriers, std::move(bundles));
      }
    }

    std::size_t consideredBarriers = 0;
    for (const CanonicalBarrierCandidate &candidate :
         mechanismUniverse_.barriers) {
      const bool selected = llvm::any_of(
          incumbentBarriers, [&](const CanonicalBarrier &barrier) {
            return candidateBarriersEquivalent(barrier, candidate.barrier);
          });
      if (selected || consideredBarriers == candidateLimit) {
        continue;
      }
      ++consideredBarriers;
      std::vector<CanonicalBarrier> barriers = incumbentBarriers;
      barriers.push_back(candidate.barrier);
      consider(std::move(barriers), incumbentBundles);
    }

    if (!bestScore) {
      break;
    }
    incumbentBarriers = std::move(*bestBarriers);
    incumbentBundles = std::move(*bestBundles);
    incumbentScore = std::move(*bestScore);
  }

  for (auto [index, barrier] : llvm::enumerate(incumbentBarriers)) {
    barrier.id = index;
  }
  plan_.barriers_ = std::move(incumbentBarriers);
  selectedEventBundles_ = std::move(incumbentBundles);
  plan_.events_ = flattenCanonicalEventBundles(selectedEventBundles_);
  return success();
}

bool CanonicalSyncPlanBuilder::isCandidatePlanWellFormed(
    ArrayRef<CanonicalBarrier> barriers,
    ArrayRef<CanonicalEventBundleCandidate> eventBundles,
    ArrayRef<CanonicalDependency> requirements, bool diagnose) const {
  if (!canonicalEventBundlesHaveNoConflicts(eventBundles)) {
    if (diagnose) {
      llvm::errs() << "duplicate or conflicting canonical event bundles\n";
    }
    return false;
  }

  for (std::size_t first = 0; first < eventBundles.size(); ++first) {
    if (eventBundles[first].kind != CanonicalEventBundleKind::Ownership) {
      continue;
    }
    auto firstCycle = llvm::find_if(
        plan_.ownershipCycles_, [&](const CanonicalOwnershipCycle &cycle) {
          return cycle.id == eventBundles[first].protocolIdentity;
        });
    if (firstCycle == plan_.ownershipCycles_.end()) {
      return false;
    }
    for (std::size_t second = first + 1; second < eventBundles.size();
         ++second) {
      if (eventBundles[second].kind != CanonicalEventBundleKind::Ownership) {
        continue;
      }
      auto secondCycle = llvm::find_if(
          plan_.ownershipCycles_, [&](const CanonicalOwnershipCycle &cycle) {
            return cycle.id == eventBundles[second].protocolIdentity;
          });
      const bool invalidSecondCycle =
          secondCycle == plan_.ownershipCycles_.end() ||
          ownershipCyclesConflict(*firstCycle, *secondCycle);
      if (invalidSecondCycle) {
        return false;
      }
    }
  }

  for (const CanonicalEventBundleCandidate &bundle : eventBundles) {
    if (bundle.kind == CanonicalEventBundleKind::Standalone) {
      const bool malformedStandalone =
          bundle.events.size() != 1 ||
          bundle.events.front().ownershipProtocol ||
          bundle.events.front().protocolBundle != 0;
      if (malformedStandalone) {
        return false;
      }
      continue;
    }

    SmallVector<const CanonicalEvent *, 2> eventPointers;
    llvm::transform(bundle.events, std::back_inserter(eventPointers),
                    [](const CanonicalEvent &event) { return &event; });
    if (bundle.kind == CanonicalEventBundleKind::SyntheticRoundTrip) {
      const bool hasVerifiedWitness =
          bundle.completionWitness
              ? verifyCanonicalSyntheticRoundTripWitness(
                    eventPointers, *bundle.completionWitness)
              : llvm::any_of(requirements, [&](const auto &requirement) {
                  return verifyCanonicalSyntheticRoundTripWitness(
                      eventPointers, requirement);
                });
      const bool malformedSynthetic =
          !verifyCanonicalSyntheticRoundTripBundle(eventPointers) ||
          !hasVerifiedWitness;
      if (malformedSynthetic) {
        return false;
      }
      continue;
    }

    auto cycle = llvm::find_if(plan_.ownershipCycles_, [&](const auto &entry) {
      return entry.id == bundle.protocolIdentity;
    });
    const bool malformedOwnership =
        cycle == plan_.ownershipCycles_.end() ||
        !verifyCanonicalOwnershipEventPair(*cycle, eventPointers);
    if (malformedOwnership) {
      return false;
    }
  }

  for (const CanonicalBarrier &barrier : barriers) {
    Operation *anchor = barrier.anchor.operation;
    if (!anchor || barrier.pipe == PipelineType::PIPE_UNASSIGNED ||
        (anchor != funcOperation_ && !funcOperation_->isAncestor(anchor))) {
      return false;
    }
  }

  const std::vector<CanonicalEvent> events =
      flattenCanonicalEventBundles(eventBundles);
  if (failed(verifyEventProtocols(events, /*requireAllocation=*/false,
                                  diagnose))) {
    return false;
  }
  return true;
}

bool CanonicalSyncPlanBuilder::isCandidatePlanFeasible(
    ArrayRef<CanonicalBarrier> barriers,
    ArrayRef<CanonicalEventBundleCandidate> eventBundles,
    ArrayRef<CanonicalDependency> requirements, bool diagnose) const {
  if (!isCandidatePlanWellFormed(barriers, eventBundles, requirements,
                                 diagnose)) {
    return false;
  }
  const std::vector<CanonicalEvent> events =
      flattenCanonicalEventBundles(eventBundles);
  if (!eventsFitBudget(events)) {
    return false;
  }
  return countUncoveredRequirements(barriers, events, requirements,
                                    diagnose) == 0;
}
