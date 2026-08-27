// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncLifecycle.h - Exact slot lifecycles -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCLIFECYCLE_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCLIFECYCLE_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using CanonicalSyncSlotLifecycleId = std::size_t;

struct CanonicalSyncUnitSlotLifecycle {
  CanonicalSyncSlotLifecycleId id = 0;
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;
  SyncCoverStorageAccessId producerAccess = 0;
  SyncCoverStorageAccessId consumerAccess = 0;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  SyncCoverScopeId recurrenceScope = 0;
  SyncCoverDemandId releaseDemand = 0;
  std::vector<SyncCoverDemandId> readyDemands;
};

enum class CanonicalSyncLifecycleError : std::uint8_t {
  None,
  InvalidGraph,
};

struct CanonicalSyncLifecycleOptions {
  std::size_t maximumLifecycles = 4096;
  std::size_t maximumEvaluations = 8192;
};

struct CanonicalSyncLifecycleResult {
  CanonicalSyncLifecycleError error = CanonicalSyncLifecycleError::None;
  std::vector<CanonicalSyncUnitSlotLifecycle> lifecycles;
  std::size_t evaluations = 0;
  bool truncated = false;

  explicit operator bool() const {
    return error == CanonicalSyncLifecycleError::None;
  }
};

CanonicalSyncLifecycleResult discoverCanonicalSyncUnitSlotLifecycles(
    const SyncCoverGraph &graph,
    const std::vector<SyncCoverDemandId> &activeDemands,
    CanonicalSyncLifecycleOptions options = {});

bool verifyCanonicalSyncUnitSlotLifecycle(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle);

std::optional<CanonicalSyncMechanismDescriptor>
makeCanonicalSyncUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId domain);

bool verifyCanonicalSyncUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId domain,
    const CanonicalSyncMechanismDescriptor &descriptor);

/// Build one independently materializable ready/release lifecycle. Unlike the
/// release-only protocol above, this descriptor owns both directions and does
/// not require a multi-mechanism activation pattern.
std::optional<CanonicalSyncMechanismDescriptor>
makeCanonicalSyncAtomicUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain);

bool verifyCanonicalSyncAtomicUnitSlotProtocol(
    const SyncCoverGraph &graph,
    const CanonicalSyncUnitSlotLifecycle &lifecycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncMechanismDescriptor &descriptor);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCLIFECYCLE_H
