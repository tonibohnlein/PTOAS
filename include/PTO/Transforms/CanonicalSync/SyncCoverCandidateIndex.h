// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverCandidateIndex.h - Frozen candidate lookup -----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCANDIDATEINDEX_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCANDIDATEINDEX_H

#include "PTO/Transforms/CanonicalSync/SyncCoverGraph.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverCandidateTimelineId = std::size_t;

enum class SyncCoverCandidateIndexError : std::uint8_t {
  None,
  InvalidGraph,
  StructureNotFrozen,
  Stale,
  InvalidIndex,
};

struct SyncCoverCandidateContext {
  std::uint32_t resource = 0;
  SyncCoverScopeId timelineScope = 0;
};

struct SyncCoverCandidateTimeline {
  SyncCoverCandidateTimelineId id = 0;
  SyncCoverCandidateContext context;
  std::vector<SyncCoverNodeId> nodes;
};

/// A deterministic group of witnesses with the same exact overlap. Groups are
/// descriptive evidence, not disjoint storage slots or ownership lanes.
struct SyncCoverCandidateWitnessOverlapGroup {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval interval;
  std::vector<SyncCoverStorageWitnessId> witnesses;
  std::vector<std::size_t> demands;
};

struct SyncCoverCandidateNodePosition {
  SyncCoverCandidateTimelineId timeline = 0;
  std::size_t ordinal = 0;
};

template <typename T> struct SyncCoverCandidateLookup {
  SyncCoverCandidateIndexError error = SyncCoverCandidateIndexError::None;
  const T *value = nullptr;

  explicit operator bool() const {
    return error == SyncCoverCandidateIndexError::None && value != nullptr;
  }
};

/// Derived lookup data over the graph's frozen structural prefix. It does not
/// infer ownership, lane order, or protocol semantics. Every query checks the
/// structural epoch so mechanism-owned supply edges cannot stale the index.
class SyncCoverCandidateIndex {
public:
  explicit SyncCoverCandidateIndex(const SyncCoverGraph &graph);

  explicit operator bool() const;
  SyncCoverCandidateIndexError getError() const;

  SyncCoverCandidateLookup<std::vector<SyncCoverCandidateTimeline>>
  getTimelines() const;
  SyncCoverCandidateLookup<SyncCoverCandidateNodePosition>
  getNodePosition(SyncCoverNodeId node) const;
  SyncCoverCandidateLookup<std::vector<SyncCoverStorageAccessId>>
  getDomainAccesses(SyncCoverStorageDomainId domain) const;
  SyncCoverCandidateLookup<std::vector<SyncCoverStorageWitnessId>>
  getDemandWitnesses(std::size_t demand) const;
  SyncCoverCandidateLookup<
      std::vector<SyncCoverCandidateWitnessOverlapGroup>>
  getWitnessOverlapGroups() const;

private:
  SyncCoverCandidateIndexError currentError() const;

  const SyncCoverGraph &graph_;
  std::size_t structuralGeneration_ = 0;
  SyncCoverCandidateIndexError buildError_ =
      SyncCoverCandidateIndexError::None;
  std::vector<SyncCoverCandidateTimeline> timelines_;
  std::vector<std::optional<SyncCoverCandidateNodePosition>> nodePositions_;
  std::vector<std::vector<SyncCoverStorageAccessId>> domainAccesses_;
  std::vector<std::vector<SyncCoverStorageWitnessId>> demandWitnesses_;
  std::vector<SyncCoverCandidateWitnessOverlapGroup> witnessOverlapGroups_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCANDIDATEINDEX_H
