// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Internal ownership boundary for translating CanonicalSync mechanisms into
// the MLIR-free SyncCover universe and running direct selection.

#ifndef PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSELECTION_H
#define PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSELECTION_H

#include "CanonicalSyncCoveringSlotRecipe.h"
#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotProtocol.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSolver.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace mlir {
namespace pto {
namespace canonical_sync_covering {

using ProviderMap =
    std::map<CanonicalSelectionMechanismRef, SyncCoverMechanismId>;
using DomainMap = std::map<CanonicalEventDomainKey, SyncCoverResourceDomainId>;

std::optional<std::uint64_t>
encodeProviderIdentity(CanonicalSelectionMechanismKind kind, std::size_t id);

bool sameDescriptor(const SyncCoverMechanismDescriptor &first,
                    const SyncCoverMechanismDescriptor &second);

bool barriersEquivalent(const CanonicalBarrier &first,
                        const CanonicalBarrier &second);

std::optional<SyncCoverScopeId> getEndpointScope(const SyncCoverGraph &graph,
                                                 std::size_t source,
                                                 std::size_t target);

std::optional<SyncCoverScopeId> getAnchorOccurrenceScope(
    const CanonicalAnchor &anchor,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes);

bool isCanonicalForwardEvent(
    const CanonicalEventBundleCandidate &bundle, const SyncCoverGraph &graph,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition);

struct TranslatedEventBundleMechanism {
  SyncCoverMechanismDescriptor descriptor;
  std::vector<std::size_t> eventResourceUses;
};

std::optional<TranslatedEventBundleMechanism> translateVerifiedEventBundle(
    const CanonicalEventBundleCandidate &bundle, std::uint64_t provider,
    const DomainMap &domains, const SyncCoverMechanismUniverse &universe,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition);

bool verifyBundleShape(const CanonicalEventBundleCandidate &bundle,
                       ArrayRef<CanonicalOwnershipCycle> cycles,
                       ArrayRef<CanonicalSyncNode> nodes);

bool verifyTranslatedEventBundleCorrespondence(
    const CanonicalEventBundleCandidate &bundle,
    const TranslatedEventBundleMechanism &translated,
    const SyncCoverMechanismDescriptor &actual,
    const SyncCoverMechanismUniverse &universe, const DomainMap &domains,
    const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
        &regionScopes,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition);

class MechanismAdapter {
public:
  MechanismAdapter(
      func::FuncOp func, const CanonicalSyncPlan &plan,
      const CanonicalMechanismUniverse &candidateUniverse,
      ArrayRef<CanonicalEventBundleCandidate> selectedEventBundles,
      unsigned eventIdMax,
      const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds,
      SyncCoverGraph &graph, const SyncCoverCandidateIndex &candidateIndex,
      const SyncCoverSlotLifecycleResult &slotLifecycles,
      const SyncCoverSlotProtocolResult &slotProtocols,
      ArrayRef<SyncCoverDemandId> activeDemands,
      const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
          &regionScopes,
      const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
      std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
      std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
          getBarrierCompletionEdges,
      std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols);

  LogicalResult build(CanonicalSyncCoveringSnapshot &snapshot);

private:
  LogicalResult collectEventBundles();
  LogicalResult addEventDomains();
  std::optional<SyncCoverEdge>
  translateBarrierEdge(const SyncGraphEdge &legacy) const;
  LogicalResult addBarriers();
  LogicalResult addEventBundles();
  LogicalResult addSlotProtocols();
  LogicalResult addGeneratedColumns(
      CanonicalSyncCoveringSnapshot &snapshot);
  LogicalResult addConflicts();
  LogicalResult solve(CanonicalSyncCoveringSnapshot &snapshot);
  LogicalResult validateSelectedResourceUsesAgainstUniverse(
      const CanonicalSyncCoveringSnapshot &snapshot);
  LogicalResult emitMechanismError(StringRef context,
                                   const SyncCoverMechanismResult &result);

  func::FuncOp func_;
  const CanonicalSyncPlan &plan_;
  const CanonicalMechanismUniverse &candidateUniverse_;
  std::vector<CanonicalEventBundleCandidate> selectedEventBundles_;
  unsigned eventIdMax_ = 0;
  const std::map<CanonicalEventDomainKey, std::set<unsigned>> &reservedIds_;
  SyncCoverMechanismUniverse universe_;
  const SyncCoverCandidateIndex &candidateIndex_;
  const SyncCoverSlotLifecycleResult &slotLifecycles_;
  const SyncCoverSlotProtocolResult &slotProtocols_;
  const std::map<Region *, SyncCoverScopeId, std::less<Region *>>
      &regionScopes_;
  const DenseMap<Operation *, SyncCoverScopeId> &loopScopes_;
  std::function<std::size_t(const CanonicalAnchor &)> getAnchorPosition_;
  std::function<std::vector<SyncGraphEdge>(const CanonicalBarrier &)>
      getBarrierCompletionEdges_;
  std::function<bool(ArrayRef<CanonicalEvent>)> verifyEventProtocols_;
  DomainMap domains_;
  ProviderMap providers_;
  std::map<CanonicalSelectionMechanismRef, SlotProtocolRecipe>
      slotProtocolRecipes_;
  std::map<CanonicalSelectionMechanismRef, CanonicalBarrier> barrierRecipes_;
  std::map<std::pair<SyncCoverMechanismId, std::size_t>, std::size_t>
      eventResourceUses_;
  std::vector<CanonicalEventBundleCandidate> eventBundles_;
  std::set<CanonicalSelectionMechanismRef> incumbentProviders_;
  std::vector<SyncCoverDemandId> activeDemands_;
  std::size_t unmaterializableSlotProtocols_ = 0;
};

} // namespace canonical_sync_covering
} // namespace pto
} // namespace mlir

#endif // PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSELECTION_H
