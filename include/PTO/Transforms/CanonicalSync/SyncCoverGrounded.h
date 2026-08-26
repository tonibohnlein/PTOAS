// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverGrounded.h - Materialized covering instance ----*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGROUNDED_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGROUNDED_H

#include "PTO/Transforms/CanonicalSync/SyncCoverCoverage.h"
#include "PTO/Transforms/CanonicalSync/SyncCoverMechanism.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverGroundedColumnId = std::size_t;

/// Dense active-demand bitset used by the search without graph traversal.
class SyncCoverDemandSet {
public:
  SyncCoverDemandSet() = default;
  explicit SyncCoverDemandSet(std::size_t size);

  bool add(std::size_t demand);
  bool contains(std::size_t demand) const;
  void unite(const SyncCoverDemandSet &other);
  void subtract(const SyncCoverDemandSet &other);
  std::size_t count() const;
  std::size_t size() const { return size_; }
  const std::vector<std::uint64_t> &getWords() const { return words_; }

  bool operator==(const SyncCoverDemandSet &other) const {
    return size_ == other.size_ && words_ == other.words_;
  }

private:
  std::size_t size_ = 0;
  std::vector<std::uint64_t> words_;
};

struct SyncCoverGroundedResourceUse {
  std::size_t resourceUse = 0;
  SyncCoverResourceDomainId domain = 0;
  std::size_t width = 0;
  SyncCoverTimelineInterval lifetime;
};

/// Selection-independent mechanism data needed by decomposition and search.
struct SyncCoverGroundedMechanism {
  SyncCoverMechanismId id = 0;
  SyncCoverMechanismKind kind = SyncCoverMechanismKind::EventBundle;
  std::vector<std::size_t> actionProfile;
  std::vector<std::size_t> barrierActionProfile;
  std::vector<SyncCoverGroundedResourceUse> resourceUses;
  std::vector<SyncCoverMechanismId> conflicts;
};

/// A column is bookkeeping, not an emission unit. Cost and resources are
/// charged over the union of members selected by all columns.
struct SyncCoverGroundedColumn {
  SyncCoverGroundedColumnId id = 0;
  std::vector<SyncCoverMechanismId> members;
  SyncCoverDemandSet coverage;
};

enum class SyncCoverGroundingError : std::uint8_t {
  None,
  InvalidUniverse,
  InvalidDemand,
  InvalidOptions,
  InvalidMechanism,
  CoverageFailure,
  ArithmeticOverflow,
};

struct SyncCoverGroundingOptions {
  /// Hard cap on factory-declared columns. Hitting the cap is reported as
  /// incomplete grounding; it is never interpreted as infeasibility.
  std::size_t maximumColumns = 65536;
};

/// A candidate factory's independently verified incidence declaration. Members
/// identify the physical mechanisms charged and emitted together. Demands list
/// additional transitive coverage proved by that factory's structural verifier.
struct SyncCoverVerifiedFactoryColumn {
  std::vector<SyncCoverMechanismId> members;
  std::vector<SyncCoverDemandId> demands;
};

/// Coverage is a pure function of a demand's endpoints, scope, distance, and
/// guards over the frozen graph: two demands with equal keys are covered by
/// exactly the same selections. Demand kind and provenance do not enter
/// reachability.
struct SyncCoverDemandCoverageKey {
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::vector<SyncCoverGuardLiteral> sourceGuard;
  std::vector<SyncCoverGuardLiteral> targetGuard;

  bool operator<(const SyncCoverDemandCoverageKey &other) const {
    return std::tie(source, target, scope, distance, sourceGuard,
                    targetGuard) <
           std::tie(other.source, other.target, other.scope, other.distance,
                    other.sourceGuard, other.targetGuard);
  }
};

SyncCoverDemandCoverageKey
makeSyncCoverDemandCoverageKey(const SyncCoverGraph &graph,
                               SyncCoverDemandId demand);

struct SyncCoverGroundedInstance {
  std::size_t universeVersion = 0;
  std::size_t graphGeneration = 0;
  /// Skylined rows: one representative demand per distinct coverage key.
  /// Covering the representative covers every active demand sharing its key;
  /// the final oracle verification still spans the deduplicated originals.
  std::vector<SyncCoverDemandId> demands;
  std::vector<SyncCoverGroundedColumn> columns;
  std::vector<std::vector<SyncCoverGroundedColumnId>> demandColumns;
  std::vector<SyncCoverGroundedMechanism> mechanisms;
  std::vector<SyncCoverResourceDomain> resourceDomains;
  std::vector<SyncCoverDemandId> demandsNeedingPricing;
  bool columnsTruncated = false;

  bool isCurrent(const SyncCoverMechanismUniverse &universe) const;
  SyncCoverDemandSet
  coveredBy(const std::vector<SyncCoverMechanismId> &selected) const;
  bool coversAll(const std::vector<SyncCoverMechanismId> &selected) const;
};

struct SyncCoverGroundingResult {
  SyncCoverGroundingError error = SyncCoverGroundingError::None;
  std::optional<SyncCoverDemandId> failedDemand;
  SyncCoverCoverageError coverageError = SyncCoverCoverageError::None;
  SyncCoverCoverageStatistics statistics;
  SyncCoverGroundedInstance instance;

  explicit operator bool() const {
    return error == SyncCoverGroundingError::None;
  }
};

/// Grounds structural coverage once. Search consumes only the returned
/// bitsets and immutable mechanism metadata; it never invokes the graph oracle.
SyncCoverGroundingResult groundSyncCoverInstance(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const SyncCoverGroundingOptions &options = {});

/// Grounds independently verified shared-cost columns. Selection never expands
/// or re-proves these declarations. The independent final oracle still checks
/// the complete selected plan before it can be emitted.
SyncCoverGroundingResult groundSyncCoverInstance(
    const SyncCoverMechanismUniverse &universe,
    const std::vector<SyncCoverDemandId> &activeDemands,
    const std::vector<SyncCoverVerifiedFactoryColumn> &factoryColumns,
    const SyncCoverGroundingOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERGROUNDED_H
