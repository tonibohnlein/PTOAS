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

#include "PTO/Transforms/CanonicalSync/SyncCoverSlotLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverSlotProtocolCandidateId = std::size_t;

enum class SyncCoverSlotProtocolError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidCandidateIndex,
  InvalidLifecycle,
};

struct SyncCoverSlotProtocolOptions {
  std::size_t maximumCandidates = 4096;
  std::size_t maximumEvaluations = 8192;
};

/// A verified factory result for one unit-distance release handoff. The
/// adapter may lower this into the stock prime/body/drain recurrence event.
/// Candidate verification proves physical-access closure and correspondence;
/// the stock recurrence verifier remains responsible for token correctness.
struct SyncCoverSlotProtocolCandidate {
  SyncCoverSlotProtocolCandidateId id = 0;
  SyncCoverSlotLifecycleId lifecycle = 0;
  SyncCoverCandidateOpportunityId release = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverScopeId recurrenceScope = 0;
  unsigned distance = 0;
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

SyncCoverSlotProtocolResult buildSyncCoverSlotProtocolCandidates(
    const SyncCoverGraph &graph, const SyncCoverCandidateIndex &index,
    const SyncCoverSlotLifecycleResult &lifecycles,
    const SyncCoverSlotProtocolOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERSLOTPROTOCOL_H
