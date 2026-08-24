// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverMechanism.h - Atomic synchronization supply ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverResourceDomainId = std::size_t;

enum class SyncCoverMechanismKind : std::uint8_t {
  EventBundle,
  Barrier,
  OwnershipProtocol,
};

enum class SyncCoverResourceKind : std::uint8_t {
  EventId,
  BufferToken,
};

struct SyncCoverResourceDomain {
  SyncCoverResourceDomainId id = 0;
  SyncCoverResourceKind kind = SyncCoverResourceKind::EventId;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::uint64_t poolIdentity = 0;
  unsigned budget = 0;
};

/// One periodic resource lifetime. distance == 0 is a forward interval;
/// positive distance denotes a loop-carried circular lifetime.
struct SyncCoverResourceUse {
  SyncCoverResourceDomainId domain = 0;
  SyncCoverNodeId begin = 0;
  SyncCoverNodeId end = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  unsigned width = 1;
  /// Descriptor indices are local to supplyEdges. Stored mechanism indices
  /// are rewritten to graph edge indices by the universe.
  std::vector<std::size_t> supplyEdges;
};

struct SyncCoverBarrierPlacement {
  std::uint32_t resource = 0;
  SyncCoverNodeId anchor = 0;
  SyncCoverScopeId scope = 0;
};

struct SyncCoverMechanismDescriptor {
  SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle;
  std::uint64_t providerIdentity = 0;
  std::optional<SyncCoverBarrierPlacement> barrier;
  std::vector<SyncCoverEdge> supplyEdges;
  std::vector<SyncCoverResourceUse> resourceUses;
};

struct SyncCoverMechanism {
  SyncCoverMechanismId id = 0;
  SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle;
  std::uint64_t providerIdentity = 0;
  bool protocolVerified = false;
  std::optional<SyncCoverBarrierPlacement> barrier;
  std::vector<std::size_t> supplyEdges;
  std::vector<SyncCoverResourceUse> resourceUses;
  std::vector<SyncCoverMechanismId> conflicts;
};

enum class SyncCoverMechanismError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidDomain,
  InvalidResourceUse,
  EmptySupply,
  InvalidSupply,
  UnverifiedProtocol,
  InvalidMechanism,
  InvalidConflict,
};

struct SyncCoverMechanismResult {
  SyncCoverMechanismError error = SyncCoverMechanismError::None;
  std::optional<std::size_t> index;

  explicit operator bool() const {
    return error == SyncCoverMechanismError::None;
  }
};

/// Owns mechanism identity and atomically attaches supplied completion edges
/// to a SyncCoverGraph. Failed additions leave both objects unchanged.
class SyncCoverMechanismUniverse {
public:
  explicit SyncCoverMechanismUniverse(SyncCoverGraph &graph) : graph_(graph) {}

  SyncCoverMechanismResult addResourceDomain(SyncCoverResourceKind kind,
                                             std::uint32_t sourceResource,
                                             std::uint32_t targetResource,
                                             unsigned budget,
                                             std::uint64_t poolIdentity = 0);
  SyncCoverMechanismResult
  addMechanism(const SyncCoverMechanismDescriptor &descriptor);
  SyncCoverMechanismResult addVerifiedProtocol(
      const SyncCoverMechanismDescriptor &descriptor,
      const std::function<bool(const SyncCoverMechanismDescriptor &)> &verify);
  SyncCoverMechanismResult addConflict(SyncCoverMechanismId first,
                                       SyncCoverMechanismId second);
  SyncCoverMechanismResult validate() const;

  const std::vector<SyncCoverResourceDomain> &getResourceDomains() const {
    return domains_;
  }
  const std::vector<SyncCoverMechanism> &getMechanisms() const {
    return mechanisms_;
  }

private:
  SyncCoverMechanismResult
  addMechanismImpl(const SyncCoverMechanismDescriptor &descriptor,
                   bool protocolVerified);
  SyncCoverMechanismError validateResourceUse(
      const SyncCoverResourceUse &use,
      const std::vector<SyncCoverEdge> *descriptorEdges = nullptr) const;
  SyncCoverMechanismError
  validateBarrier(const SyncCoverMechanismDescriptor &descriptor) const;

  SyncCoverGraph &graph_;
  std::vector<SyncCoverResourceDomain> domains_;
  std::vector<SyncCoverMechanism> mechanisms_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERMECHANISM_H
