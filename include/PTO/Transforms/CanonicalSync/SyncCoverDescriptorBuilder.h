// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverDescriptorBuilder.h - Mechanism construction ---*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERDESCRIPTORBUILDER_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERDESCRIPTORBUILDER_H

#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

struct SyncCoverDescriptorActionRef {
  std::size_t index = 0;
};

struct SyncCoverProtocolSupply {
  SyncCoverEdge edge;
  SyncCoverDescriptorActionRef produceAction;
  SyncCoverDescriptorActionRef consumeAction;
};

/// Builds descriptors without exposing mechanism-local action, use, edge, and
/// binding index spaces to adapter call sites. Failed additions are atomic.
class SyncCoverMechanismDescriptorBuilder {
public:
  explicit SyncCoverMechanismDescriptorBuilder(
      SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle,
      std::uint64_t providerIdentity = 0);

  SyncCoverDescriptorActionRef
  addAction(SyncCoverResourceActionKind kind, std::uint32_t resource,
            SyncCoverAnchor anchor);

  /// Adds one canonical distance-zero event and its complete action/use/binding
  /// representation. The domain must be an EventId domain.
  bool addCanonicalEvent(const SyncCoverResourceDomain &domain,
                         SyncCoverNodeId source, SyncCoverNodeId target,
                         SyncCoverScopeId scope = 0, std::size_t width = 1);

  /// Adds one verified protocol lane. Existing action references allow a
  /// physical action to be shared by EventId and BufferToken resource uses.
  bool addProtocolLane(
      const SyncCoverResourceDomain &domain, SyncCoverScopeId scope,
      unsigned distance, std::size_t width,
      std::vector<SyncCoverDescriptorActionRef> actions,
      std::vector<SyncCoverProtocolSupply> supplies);

  const SyncCoverMechanismDescriptor &getDescriptor() const {
    return descriptor_;
  }
  SyncCoverMechanismDescriptor takeDescriptor() &&;

private:
  bool addLane(const SyncCoverResourceDomain &domain, SyncCoverScopeId scope,
               unsigned distance, std::size_t width,
               std::vector<SyncCoverDescriptorActionRef> actions,
               std::vector<SyncCoverProtocolSupply> supplies);

  SyncCoverMechanismDescriptor descriptor_;
};

std::optional<SyncCoverMechanismDescriptor> makeSyncCoverCanonicalEvent(
    const SyncCoverResourceDomain &domain, SyncCoverNodeId source,
    SyncCoverNodeId target, SyncCoverScopeId scope = 0,
    std::size_t width = 1, std::uint64_t providerIdentity = 0);

/// Builds the conservative stock recurrence protocol: prime one event at loop
/// entry, consume it before the target, produce it after the source, and drain
/// it at loop exit. It intentionally supports only distance one and width one.
/// Multi-distance or multi-lane protocols require an adapter-specific verifier
/// that proves their exact lane and control-flow semantics.
std::optional<SyncCoverMechanismDescriptor>
makeSyncCoverUnitRecurrenceEvent(const SyncCoverResourceDomain &domain,
                                 SyncCoverNodeId source,
                                 SyncCoverNodeId target,
                                 SyncCoverScopeId loop,
                                 std::uint64_t providerIdentity = 0);

/// Verifies the exact descriptor shape emitted by
/// makeSyncCoverUnitRecurrenceEvent against the completed universe.
bool verifySyncCoverUnitRecurrenceEvent(
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverMechanismDescriptor &descriptor);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERDESCRIPTORBUILDER_H
