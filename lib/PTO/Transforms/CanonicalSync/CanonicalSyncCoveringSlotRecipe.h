// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSLOTRECIPE_H
#define PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSLOTRECIPE_H

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotProtocol.h"

#include <optional>

namespace mlir {
namespace pto {
namespace canonical_sync_covering {

struct SlotProtocolRecipe {
  std::size_t resourceUse = 0;
  CanonicalEvent event;
};

bool canBuildSlotProtocolRecipe(
    const SyncCoverSlotProtocolCandidate &candidate,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes);

std::optional<SlotProtocolRecipe> buildSlotProtocolRecipe(
    ArrayRef<CanonicalSyncNode> nodes,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanism &mechanism,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition);

bool verifySlotProtocolRecipeCorrespondence(
    ArrayRef<CanonicalSyncNode> nodes,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanism &mechanism,
    const DenseMap<Operation *, SyncCoverScopeId> &loopScopes,
    llvm::function_ref<std::size_t(const CanonicalAnchor &)> getAnchorPosition,
    const SlotProtocolRecipe &recipe);

std::optional<CanonicalEventBundleCandidate> materializeSlotProtocolBundle(
    const CanonicalSyncCoveringSelectedSlotProtocol &recipe,
    const CanonicalSyncCoveringSelectedResourceUse &use,
    const CanonicalSyncCoveringResourceAllocation &allocation,
    std::size_t bundleId);

} // namespace canonical_sync_covering
} // namespace pto
} // namespace mlir

#endif // PTO_LIB_TRANSFORMS_CANONICALSYNC_COVERINGSLOTRECIPE_H
