// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncCoverCoverage.h - Completion coverage bitsets -------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H

#include "PTO/Transforms/CanonicalSync/SyncCoverExpansion.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace mlir {
namespace pto {

using SyncCoverMechanismId = std::size_t;

/// Bounds the dense bit matrices used by one coverage query. The defaults cap
/// the simultaneous result-plus-workspace payload at 32 MiB and keep result
/// and mechanism index metadata bounded independently, including with no
/// demand bits.
struct SyncCoverCoverageLimits {
  std::size_t maximumWorkspaceWords = 1U << 22;
  std::size_t maximumResultWords = 1U << 22;
  std::size_t maximumTotalWords = 1U << 22;
  std::size_t maximumResultRows = 1U << 16;
  std::size_t maximumMechanismRows = 1U << 16;
};

/// Shared counter for one optional exact-coverage query. Queries without a
/// budget pass null and retain their existing behavior. A bounded query stops
/// before executing work that would exceed the limit.
struct SyncCoverCoverageWorkBudget {
  explicit SyncCoverCoverageWorkBudget(
      std::size_t maximum = std::numeric_limits<std::size_t>::max())
      : maximumWorkUnits(maximum) {}

  bool consume(std::size_t amount = 1) {
    if (exhausted || workUnits > maximumWorkUnits ||
        amount > maximumWorkUnits - workUnits) {
      exhausted = true;
      return false;
    }
    workUnits += amount;
    return true;
  }

  std::size_t maximumWorkUnits = std::numeric_limits<std::size_t>::max();
  std::size_t workUnits = 0;
  bool exhausted = false;
};

class SyncCoverDemandSet {
public:
  explicit SyncCoverDemandSet(std::size_t size = 0);

  std::size_t size() const { return size_; }
  std::size_t count() const;
  bool empty() const { return count() == 0; }
  bool contains(SyncCoverDemandId demand) const;
  bool insert(SyncCoverDemandId demand);
  void unite(const SyncCoverDemandSet &other);
  void subtract(const SyncCoverDemandSet &other);
  bool containsAll(const SyncCoverDemandSet &other) const;
  const std::vector<std::uint64_t> &getWords() const { return words_; }

  bool operator==(const SyncCoverDemandSet &other) const {
    return size_ == other.size_ && words_ == other.words_;
  }

private:
  std::size_t size_ = 0;
  std::vector<std::uint64_t> words_;
};

/// A physical synchronization point. The point kind is part of the hardware
/// contract: an event cut contains exactly one source-pipe Set and one
/// target-pipe Wait, while a pipe cut contains one same-pipe barrier.
enum class SyncCoverCutPointKind : std::uint8_t {
  EventSet,
  EventWait,
  PipeBarrier,
};

struct SyncCoverCutPoint {
  SyncCoverCutPointKind kind = SyncCoverCutPointKind::EventSet;
  std::uint32_t resource = 0;
  SyncCoverAnchor anchor;
  SyncCoverGuard guard;
  /// Deterministic order among physical actions sharing one anchor.
  std::size_t ordinal = 0;
};

enum class SyncCoverDirectCutKind : std::uint8_t {
  Event,
  PipeBarrier,
};

/// One token-independent physical cut. Target legality and event lifecycle are
/// separate admission proofs: this descriptor records only the ordering effect
/// of an already-authorized event pair or barrier at concrete cut points.
struct SyncCoverDirectCut {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverDirectCutKind kind = SyncCoverDirectCutKind::Event;
  SyncCoverCutPoint source;
  SyncCoverCutPoint target;
  SyncCoverOrderingRequirementMask suppliedRequirements = 0;
};

/// One exact enabled-mechanism set. IDs are strictly increasing so the world
/// has one deterministic identity and membership is logarithmic.
struct SyncCoverExactWorld {
  std::vector<SyncCoverMechanismId> enabledMechanisms;
};

/// Compact semantic effect of one enabled physical cut. The two boundaries
/// denote a source-resource prefix and target-resource suffix. Guards qualify
/// the dynamic occurrences for which the rectangle exists.
struct SyncCoverCompletionOrigin {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverDirectCutKind kind = SyncCoverDirectCutKind::Event;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverTimelinePosition sourceBoundary = 0;
  SyncCoverTimelinePosition targetBoundary = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
  SyncCoverOrderingRequirementMask suppliedRequirements = 0;
  std::size_t sourceOrdinal = 0;
  std::size_t targetOrdinal = 0;
};

struct SyncCoverFlatWorldLimits {
  std::size_t maximumCuts = 1U << 16;
  std::size_t maximumEnabledMechanisms = 1U << 16;
  std::size_t maximumCompletionOrigins = 1U << 16;
  std::size_t maximumStates = 1U << 22;
  std::size_t maximumGuardLiteralsPerPoint = 1U << 16;
  std::size_t maximumTotalGuardLiterals = 1U << 20;
};

enum class SyncCoverFlatWorldError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidWorld,
  InvalidCut,
  UnsupportedStructure,
  LimitExceeded,
  WorkLimitExceeded,
};

struct SyncCoverFlatWorldResult {
  SyncCoverFlatWorldError error = SyncCoverFlatWorldError::None;
  SyncCoverDemandSet covered;
  std::vector<SyncCoverCompletionOrigin> completionOrigins;
  std::optional<std::size_t> invalidIndex;

  bool coversAll() const {
    return error == SyncCoverFlatWorldError::None &&
           covered.count() == covered.size();
  }
  explicit operator bool() const {
    return error == SyncCoverFlatWorldError::None;
  }
};

struct SyncCoverRegionWorldLimits {
  std::size_t maximumWorldsPerBatch = 256;
  std::size_t maximumCuts = 1U << 16;
  std::size_t maximumRegionEvaluations = 1U << 20;
  std::size_t maximumStateWords = 1U << 22;
  std::size_t maximumCutActions = 1U << 17;
};

struct SyncCoverRegionWorldStatistics {
  std::size_t regionsEvaluated = 0;
  std::size_t choiceIntersections = 0;
  std::size_t guardSpecializations = 0;
  std::size_t maximumLiveStateWords = 0;
};

struct SyncCoverRegionWorldResult {
  SyncCoverFlatWorldError error = SyncCoverFlatWorldError::None;
  std::vector<SyncCoverDemandSet> coveredByWorld;
  SyncCoverRegionWorldStatistics statistics;
  std::optional<std::size_t> invalidIndex;

  explicit operator bool() const {
    return error == SyncCoverFlatWorldError::None;
  }
};

/// A mechanism validator has already established that this completion edge is
/// implemented atomically. Coverage only evaluates its semantic consequence.
enum class SyncCoverSupplyApplicability : std::uint8_t {
  AllDemands,
  DistanceZeroOnly,
};

struct SyncCoverCompletionSupply {
  SyncCoverMechanismId mechanism = 0;
  SyncCoverEdge edge;
  std::vector<SyncCoverDemandId> allowedDemands;
  /// A verified recurrence protocol with a balanced scope-exit drain exports
  /// completion to the enclosing arena through its loop summary.
  bool exportsCompletionAtScopeExit = false;
  SyncCoverSupplyApplicability applicability =
      SyncCoverSupplyApplicability::AllDemands;
};

enum class SyncCoverCoverageError : std::uint8_t {
  None,
  InvalidGraph,
  InvalidSupply,
  ExpansionUnavailable,
  LimitExceeded,
  WorkLimitExceeded,
};

struct SyncCoverCoverageResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  SyncCoverDemandSet covered;
  std::vector<SyncCoverDemandId> unavailableDemands;

  bool coversAll() const {
    return error == SyncCoverCoverageError::None &&
           covered.count() == covered.size();
  }
  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

/// Exact singleton coverage for every mechanism in one batched propagation.
/// Each entry has graph.getDemands().size() bits. This is the construction
/// path for singleton patterns; it avoids one graph traversal per mechanism.
struct SyncCoverSingletonCoverageResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  SyncCoverDemandSet baseline;
  std::vector<SyncCoverDemandSet> mechanisms;
  std::vector<SyncCoverDemandId> unavailableDemands;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

struct SyncCoverMechanismPair {
  SyncCoverMechanismId first = 0;
  SyncCoverMechanismId second = 0;
};

/// Exact joint coverage for a bounded set of two-mechanism proposals. All
/// proposals are propagated together in each demand's owning expansion arena,
/// so pair construction does not perform one graph traversal per proposal.
struct SyncCoverPairCoverageResult {
  SyncCoverCoverageError error = SyncCoverCoverageError::None;
  std::vector<SyncCoverDemandSet> pairs;
  std::vector<SyncCoverDemandId> unavailableDemands;

  explicit operator bool() const {
    return error == SyncCoverCoverageError::None;
  }
};

/// Computes exact demand coverage for one mechanism set over the immutable
/// bounded expansion. Expansion construction validates and binds the frozen
/// graph once; this query checks that binding but does not repeat full graph
/// validation, unroll a loop, or mutate the graph.
SyncCoverCoverageResult computeSyncCoverCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    const std::vector<SyncCoverCompletionSupply> &supplies);
SyncCoverCoverageResult
computeSyncCoverCoverage(const SyncCoverGraph &graph,
                         const SyncCoverExpandedProgram &expansion,
                         const std::vector<SyncCoverCompletionSupply> &supplies,
                         const std::vector<SyncCoverDemandId> &activeDemands,
                         SyncCoverCoverageWorkBudget *workBudget = nullptr);

SyncCoverSingletonCoverageResult computeSyncCoverSingletonCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &supplies,
    SyncCoverCoverageLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);
SyncCoverSingletonCoverageResult computeSyncCoverSingletonCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &supplies,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

SyncCoverPairCoverageResult computeSyncCoverPairCoverage(
    const SyncCoverGraph &graph, const SyncCoverExpandedProgram &expansion,
    std::size_t mechanismCount,
    const std::vector<SyncCoverCompletionSupply> &supplies,
    const std::vector<SyncCoverMechanismPair> &pairs,
    const std::vector<SyncCoverDemandId> &activeDemands,
    SyncCoverCoverageLimits limits = {});

/// Computes exact distance-zero coverage for one enabled set of physical cuts.
/// This is the deliberately flat reference engine used before hierarchical
/// must summaries are admitted. A Set/Wait or pipe barrier contributes a typed
/// source-prefix x target-suffix rectangle. Rectangles and fixed issue order
/// are closed transitively, but issue order alone never creates completion.
SyncCoverFlatWorldResult computeSyncCoverFlatExactWorld(
    const SyncCoverGraph &graph, const std::vector<SyncCoverDirectCut> &cuts,
    const SyncCoverExactWorld &world, SyncCoverFlatWorldLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

/// Evaluates a bounded batch of exact worlds through the first-class region
/// tree. Immediate children return only completion, issued-source, and logical
/// event state. Choice regions intersect those states across every feasible
/// alternative unless the demand context proves one alternative. Repeated
/// regions remain unsupported until the recurrence protocol layer is active.
SyncCoverRegionWorldResult computeSyncCoverRegionExactWorlds(
    const SyncCoverGraph &graph, const std::vector<SyncCoverDirectCut> &cuts,
    const std::vector<SyncCoverExactWorld> &worlds,
    SyncCoverRegionWorldLimits limits = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERCOVERAGE_H
