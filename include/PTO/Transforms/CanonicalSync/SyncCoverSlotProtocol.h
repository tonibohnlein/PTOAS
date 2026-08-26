// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverSlotProtocol.h - Slot protocol candidates ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTPROTOCOL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTPROTOCOL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverSlotLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverSlotProtocolCandidateId = std::size_t;

enum class SyncCoverSlotProtocolKind : std::uint8_t {
  UnitRelease,
  HierarchicalRelease,
};

enum class SyncCoverSlotProtocolError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidCandidateIndex,
  InvalidLifecycle,
};

struct SyncCoverSlotProtocolOptions {
  std::size_t maximumCandidates = 4096;
  /// Upper bound on precharged opportunity/access inspection units.
  std::size_t maximumEvaluations = 8192;
};

struct SyncCoverSlotProtocolPath {
  std::vector<SyncCoverNodeId> sources;
  SyncCoverNodeId waitTarget = 0;
};

/// A verified factory result for one unit-distance release handoff. A unit
/// release waits at its first producer node. A hierarchical release waits once
/// at a nested-loop entry and supplies every guarded producer in that loop.
/// Candidate verification proves physical-access closure and correspondence;
/// descriptor verification independently proves prime/body/drain structure.
struct SyncCoverSlotProtocolCandidate {
  SyncCoverSlotProtocolCandidateId id = 0;
  SyncCoverSlotProtocolKind kind = SyncCoverSlotProtocolKind::UnitRelease;
  SyncCoverSlotLifecycleId lifecycle = 0;
  std::vector<SyncCoverCandidateOpportunityId> releases;
  std::vector<std::pair<SyncCoverNodeId, SyncCoverNodeId>> completionEdges;
  std::vector<SyncCoverNodeId> sources;
  std::vector<unsigned> sourceLanes;
  /// Representative source retained for unit-release compatibility.
  SyncCoverNodeId source = 0;
  std::vector<SyncCoverNodeId> targets;
  /// When present, the release is consumed once before this nested loop.
  /// Otherwise targetWaits identifies the first operation on each guarded path.
  std::optional<SyncCoverScopeId> waitScope;
  std::vector<SyncCoverNodeId> targetWaits;
  std::vector<SyncCoverSlotProtocolPath> paths;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;
  unsigned width = 1;
};

struct SyncCoverSlotProtocolResult {
  SyncCoverSlotProtocolError error = SyncCoverSlotProtocolError::None;
  std::vector<SyncCoverSlotProtocolCandidate> candidates;
  std::size_t pathSensitiveLifecycles = 0;
  std::size_t accessOpenLifecycles = 0;
  std::size_t unsupportedEffectLifecycles = 0;
  std::size_t unsupportedDistanceReleases = 0;
  std::size_t partialSlotOpportunities = 0;
  std::size_t nonBoundaryReleases = 0;
  std::size_t evaluations = 0;
  bool truncated = false;

  explicit operator bool() const {
    return error == SyncCoverSlotProtocolError::None;
  }
};

bool verifySyncCoverSlotProtocolCandidate(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverSlotProtocolCandidate &candidate);

std::optional<SyncCoverMechanismDescriptor>
makeSyncCoverSlotProtocolDescriptor(
    const SyncCoverResourceDomain &domain,
    const SyncCoverSlotProtocolCandidate &candidate,
    std::uint64_t providerIdentity);

bool verifySyncCoverSlotProtocol(
    const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycle &lifecycle,
    const SyncCoverMechanismUniverse &universe,
    const SyncCoverSlotProtocolCandidate &candidate,
    const SyncCoverMechanismDescriptor &descriptor);

SyncCoverSlotProtocolResult buildSyncCoverSlotProtocolCandidates(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycleResult &lifecycles,
    const SyncCoverSlotProtocolOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTPROTOCOL_H
