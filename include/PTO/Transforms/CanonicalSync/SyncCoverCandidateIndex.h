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
using SyncCoverCandidateOpportunityId = std::size_t;

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

struct SyncCoverCandidateSlot {
  SyncCoverStorageDomainId domain = 0;
  SyncCoverStorageInterval overlap;
  SyncCoverStorageInterval sourceExtent;
  SyncCoverStorageInterval targetExtent;
  SyncCoverStorageAccessId sourceAccess = 0;
  SyncCoverStorageAccessId targetAccess = 0;
  SyncCoverStorageAccessFamilyId sourceFamily = 0;
  SyncCoverStorageAccessFamilyId targetFamily = 0;
  SyncCoverStorageAccessMode sourceMode = SyncCoverStorageAccessMode::Read;
  SyncCoverStorageAccessMode targetMode = SyncCoverStorageAccessMode::Read;
  std::optional<unsigned> sourceAddressOrdinal;
  std::optional<unsigned> targetAddressOrdinal;
};

/// Selection-independent opportunity presented to candidate factories. An
/// exact memory demand has one opportunity per physical-overlap witness;
/// non-memory and conservatively modeled demands have one slot-less record.
struct SyncCoverCandidateOpportunity {
  SyncCoverCandidateOpportunityId id = 0;
  std::size_t demand = 0;
  SyncCoverDemandKind kind = SyncCoverDemandKind::SSA;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverCandidateNodePosition sourcePosition;
  SyncCoverCandidateNodePosition targetPosition;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
  SyncCoverStorageProvenance storageProvenance =
      SyncCoverStorageProvenance::NotApplicable;
  std::optional<SyncCoverCandidateSlot> slot;
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
  SyncCoverCandidateLookup<std::vector<SyncCoverCandidateOpportunity>>
  getOpportunities() const;
  SyncCoverCandidateLookup<std::vector<SyncCoverCandidateOpportunityId>>
  getDemandOpportunities(std::size_t demand) const;

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
  std::vector<SyncCoverCandidateOpportunity> opportunities_;
  std::vector<std::vector<SyncCoverCandidateOpportunityId>>
      demandOpportunities_;
};

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCANDIDATEINDEX_H
