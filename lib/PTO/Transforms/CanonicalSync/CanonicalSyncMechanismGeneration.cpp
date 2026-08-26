// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::pto;

namespace {

using BundleKey = std::pair<CanonicalEventBundleKind, std::size_t>;

BundleKey getBundleKey(const CanonicalEvent &event, std::size_t index) {
  if (event.ownershipProtocol && event.ownershipCycle != 0) {
    return {CanonicalEventBundleKind::Ownership, event.ownershipCycle};
  }
  return {CanonicalEventBundleKind::Standalone, index};
}

bool sameAnchor(const CanonicalAnchor &first, const CanonicalAnchor &second) {
  return first.operation == second.operation && first.before == second.before;
}

bool sameAction(const CanonicalEventAction &first,
                const CanonicalEventAction &second) {
  return first.kind == second.kind && first.phase == second.phase &&
         sameAnchor(first.anchor, second.anchor) &&
         first.lane.kind == second.lane.kind &&
         first.lane.index == second.lane.index &&
         first.lane.selector == second.lane.selector &&
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop;
}

bool sameCompletion(const CanonicalEventCompletion &first,
                    const CanonicalEventCompletion &second) {
  return first.source == second.source && first.target == second.target &&
         first.iterationDistance == second.iterationDistance &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.setAction == second.setAction &&
         first.waitAction == second.waitAction;
}

bool sameTrace(const CanonicalEventTrace &first,
               const CanonicalEventTrace &second) {
  return first.kind == second.kind && first.actions == second.actions &&
         first.controlRegion == second.controlRegion &&
         first.guard.kind == second.guard.kind &&
         first.guard.loop == second.guard.loop &&
         first.hasExplicitTokenState == second.hasExplicitTokenState &&
         first.initialTokens == second.initialTokens &&
         first.expectedTokens == second.expectedTokens;
}

bool sameEvent(const CanonicalEvent &first, const CanonicalEvent &second) {
  return first.source == second.source && first.target == second.target &&
         first.sourcePipe == second.sourcePipe &&
         first.targetPipe == second.targetPipe &&
         sameAnchor(first.setAnchor, second.setAnchor) &&
         sameAnchor(first.waitAnchor, second.waitAnchor) &&
         first.recurrenceLoop == second.recurrenceLoop &&
         first.forwardDrainLoop == second.forwardDrainLoop &&
         first.scopeLoop == second.scopeLoop &&
         first.resourceScopeLoop == second.resourceScopeLoop &&
         first.iterationDistance == second.iterationDistance &&
         first.setSlot == second.setSlot && first.waitSlot == second.waitSlot &&
         first.width == second.width && first.eventIds == second.eventIds &&
         first.intervalBegin == second.intervalBegin &&
         first.intervalEnd == second.intervalEnd &&
         llvm::equal(first.actions, second.actions, sameAction) &&
         llvm::equal(first.completions, second.completions, sameCompletion) &&
         llvm::equal(first.traces, second.traces, sameTrace) &&
         first.ownershipCycle == second.ownershipCycle &&
         first.ownershipProtocolKind == second.ownershipProtocolKind &&
         first.ownershipRole == second.ownershipRole &&
         first.ownershipProtocol == second.ownershipProtocol;
}

bool slotsOverlap(const CanonicalPhysicalSlot &first,
                  const CanonicalPhysicalSlot &second) {
  if (first.space != second.space || first.size == 0 || second.size == 0) {
    return false;
  }
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  if (first.address > maximum - first.size ||
      second.address > maximum - second.size) {
    return true;
  }
  return std::max(first.address, second.address) <
         std::min(first.address + first.size, second.address + second.size);
}

bool cyclesConflict(const CanonicalOwnershipCycle &first,
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
          if (slotsOverlap(firstSlot, secondSlot)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void addConflict(CanonicalEventBundleCandidate &first,
                 CanonicalEventBundleCandidate &second) {
  if (!llvm::is_contained(first.conflicts, second.id)) {
    first.conflicts.push_back(second.id);
  }
  if (!llvm::is_contained(second.conflicts, first.id)) {
    second.conflicts.push_back(first.id);
  }
}

SmallVector<const CanonicalOwnershipCycle *, 3> getCycles(
    const CanonicalEventBundleCandidate &bundle,
    ArrayRef<CanonicalOwnershipCycle> cycles) {
  SmallVector<const CanonicalOwnershipCycle *, 3> result;
  std::set<std::size_t> seen;
  for (const CanonicalEvent &event : bundle.events) {
    if (!event.ownershipProtocol || event.ownershipCycle == 0 ||
        !seen.insert(event.ownershipCycle).second) {
      continue;
    }
    auto cycle = llvm::find_if(cycles, [&](const auto &candidate) {
      return candidate.id == event.ownershipCycle;
    });
    if (cycle != cycles.end()) {
      result.push_back(&*cycle);
    }
  }
  return result;
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
      if (event.ownershipProtocol) {
        bundle.ownershipProtocol = event.ownershipProtocolKind;
      }
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
  return llvm::equal(flattenCanonicalEventBundles(bundles), events, sameEvent);
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
    if (source.pipe != target.pipe || hasHardwareCompletion(source.pipe) ||
        hasIntrinsicMmadAccumulatorOrdering(requirement)) {
      continue;
    }
    CanonicalBarrier barrier;
    barrier.pipe = source.pipe;
    barrier.anchor = requirement.iterationDistance == 0
                         ? getWaitAnchor(source.operation, target.operation)
                         : CanonicalAnchor{target.operation, true};
    auto nodes = operationNodes_.find(barrier.anchor.operation);
    if (nodes != operationNodes_.end()) {
      barrier.anchorNodes.append(nodes->second.begin(), nodes->second.end());
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
    candidate.barrier = std::move(barrier);
    mechanismUniverse_.barriers.push_back(std::move(candidate));
  }

  std::vector<CanonicalEvent> events;
  // Generate from the immutable conservative universe. Alias annotations may
  // deactivate demands, but must not remove a synchronization opportunity
  // that can still cover another active demand transitively.
  materializeEventsFrom(plan_.conservativeCompletionRequirements_, events);
  mechanismUniverse_.eventBundles = buildCanonicalEventBundles(events);
  for (const CanonicalOwnershipCycle &cycle : plan_.ownershipCycles_) {
    std::optional<CanonicalEventBundleCandidate> bundle =
        buildOwnershipEventBundle(cycle, cycle.protocol);
    if (bundle) {
      bundle->id = mechanismUniverse_.eventBundles.size();
      mechanismUniverse_.eventBundles.push_back(std::move(*bundle));
    }
  }
  if (std::optional<CanonicalEventBundleCandidate> composite =
          buildCompositeOwnershipEventBundle()) {
    composite->id = mechanismUniverse_.eventBundles.size();
    mechanismUniverse_.eventBundles.push_back(std::move(*composite));
  }

  for (std::size_t first = 0;
       first < mechanismUniverse_.eventBundles.size(); ++first) {
    CanonicalEventBundleCandidate &firstBundle =
        mechanismUniverse_.eventBundles[first];
    if (firstBundle.kind != CanonicalEventBundleKind::Ownership &&
        firstBundle.kind != CanonicalEventBundleKind::CompositeOwnership) {
      continue;
    }
    const auto firstCycles = getCycles(firstBundle, plan_.ownershipCycles_);
    for (std::size_t second = first + 1;
         second < mechanismUniverse_.eventBundles.size(); ++second) {
      CanonicalEventBundleCandidate &secondBundle =
          mechanismUniverse_.eventBundles[second];
      if (secondBundle.kind != CanonicalEventBundleKind::Ownership &&
          secondBundle.kind !=
              CanonicalEventBundleKind::CompositeOwnership) {
        continue;
      }
      const auto secondCycles = getCycles(secondBundle, plan_.ownershipCycles_);
      const bool conflict = llvm::any_of(firstCycles, [&](const auto *left) {
        return llvm::any_of(secondCycles, [&](const auto *right) {
          return left->id == right->id || cyclesConflict(*left, *right);
        });
      });
      if (conflict) {
        addConflict(firstBundle, secondBundle);
      }
    }
  }
}
