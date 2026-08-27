// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSyncOwnership.h - Exact ownership cycles -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCOWNERSHIP_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCOWNERSHIP_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

enum class CanonicalSyncOwnershipKind : std::uint8_t {
  L0Operand,
  L1Tile,
  L0Accumulator,
};

enum class CanonicalSyncOwnershipProtocolKind : std::uint8_t {
  RoundTrip,
  AlternatingPrefetch,
  BoundaryGuardedRoundTrip,
};

struct CanonicalSyncOwnershipSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval extent;

  bool operator<(const CanonicalSyncOwnershipSlot &other) const;
  bool operator==(const CanonicalSyncOwnershipSlot &other) const;
};

struct CanonicalSyncOwnershipLane {
  unsigned id = 0;
  std::vector<CanonicalSyncOwnershipSlot> slots;
};

struct CanonicalSyncOwnershipUse {
  unsigned lane = 0;
  unsigned producerLane = 0;
  std::vector<SyncCoverNodeId> producers;
  std::vector<SyncCoverNodeId> consumers;
  SyncCoverAnchor writeAcquire;
  SyncCoverAnchor ready;
  SyncCoverAnchor readAcquire;
  SyncCoverAnchor release;
};

struct CanonicalSyncOwnershipPath {
  SyncCoverScopeId scope = 0;
  std::vector<CanonicalSyncOwnershipUse> uses;
};

struct CanonicalSyncOwnershipCycle {
  std::size_t id = 0;
  CanonicalSyncOwnershipKind kind = CanonicalSyncOwnershipKind::L0Operand;
  CanonicalSyncOwnershipProtocolKind protocol =
      CanonicalSyncOwnershipProtocolKind::RoundTrip;
  SyncCoverScopeId recurrenceScope = 0;
  std::uint32_t producerResource = 0;
  std::uint32_t consumerResource = 0;
  std::vector<CanonicalSyncOwnershipLane> lanes;
  std::vector<CanonicalSyncOwnershipPath> paths;
  std::vector<SyncCoverNodeId> initialProducers;
  SyncCoverAnchor initialWriteAcquire;
  SyncCoverAnchor initialReady;
  unsigned initialReadyLane = 0;
  std::vector<unsigned> initiallyFreeLanes;
};

enum class CanonicalSyncOwnershipError : std::uint8_t {
  None,
  InvalidProgram,
};

struct CanonicalSyncOwnershipOptions {
  std::size_t maximumCycles = 256;
  std::size_t maximumInspections = 1U << 18;
};

struct CanonicalSyncOwnershipResult {
  CanonicalSyncOwnershipError error = CanonicalSyncOwnershipError::None;
  std::vector<CanonicalSyncOwnershipCycle> cycles;
  std::size_t inspections = 0;
  bool truncated = false;

  explicit operator bool() const {
    return error == CanonicalSyncOwnershipError::None;
  }
};

CanonicalSyncOwnershipResult discoverCanonicalSyncOwnershipCycles(
    const CanonicalSyncProgram &program,
    CanonicalSyncOwnershipOptions options = {});

bool verifyCanonicalSyncOwnershipCycle(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle);

struct CanonicalSyncOwnershipProtocol {
  CanonicalSyncMechanismDescriptor ready;
  CanonicalSyncMechanismDescriptor release;
};

std::optional<CanonicalSyncOwnershipProtocol>
makeCanonicalSyncOwnershipProtocol(const CanonicalSyncProgram &program,
                                   const CanonicalSyncOwnershipCycle &cycle,
                                   CanonicalSyncEventDomainId readyDomain,
                                   CanonicalSyncEventDomainId releaseDomain);

bool verifyCanonicalSyncOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncOwnershipProtocol &protocol);

std::optional<CanonicalSyncMechanismDescriptor>
makeCanonicalSyncAtomicOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain);

bool verifyCanonicalSyncAtomicOwnershipProtocol(
    const CanonicalSyncProgram &program,
    const CanonicalSyncOwnershipCycle &cycle,
    CanonicalSyncEventDomainId readyDomain,
    CanonicalSyncEventDomainId releaseDomain,
    const CanonicalSyncMechanismDescriptor &descriptor);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCOWNERSHIP_H
