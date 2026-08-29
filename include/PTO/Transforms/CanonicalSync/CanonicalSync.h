// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- CanonicalSync.h - Bounded pattern synchronization ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

/// Internal ablation controls for the restricted direct-mechanism catalog.
/// Repair frontiers are grounded only from a live allocation conflict core in
/// a separately owned repair problem.
struct CanonicalSyncPatternOptions {
  bool enableDirectPairs = true;
  bool enableConflictCoreRepair = true;
  bool enableSlotLifecycleBundles = false;
  std::size_t maximumRepairFrontierInspections = 1U << 16;
  std::size_t maximumRepairFrontierProposals = 4096;
  std::size_t maximumSourcePrefixInspections = 1U << 20;
  std::size_t maximumSourcePrefixCandidates = 1U << 14;
  std::size_t maximumSourcePrefixIncidences = 1U << 20;
  std::size_t maximumLoopCarryInspections = 1U << 20;
  std::size_t maximumLoopCarryCandidates = 1U << 14;
  std::size_t maximumLoopCarryIncidences = 1U << 20;
  std::size_t maximumLoopBoundaryProtocolInspections = 1U << 20;
  std::size_t maximumLoopBoundaryProtocolCandidates = 1U << 14;
  std::size_t maximumLoopBoundaryProtocolIncidences = 1U << 20;
  std::size_t maximumSlotLifecycleInspections = 1U << 20;
  std::size_t maximumSlotLifecycleCandidates = 1U << 12;
  std::size_t maximumSlotLifecycleConflictIncidences = 1U << 13;
};

struct CanonicalSyncStrategyReport {
  CanonicalSyncSelectionStrategy strategy =
      CanonicalSyncSelectionStrategy::PairLookahead;
  CanonicalSyncSelectionError error = CanonicalSyncSelectionError::None;
  CanonicalSyncSelectionError verificationError =
      CanonicalSyncSelectionError::None;
  /// The precise-plan result retained even when a separately verified
  /// localized backstop is materialized.
  CanonicalSyncSelectionError preciseError = CanonicalSyncSelectionError::None;
  bool verified = false;
  bool usedLocalizedPipeAll = false;
  bool repairFrontierTruncated = false;
  bool repairBudgetExhausted = false;
  bool backstopDeletionTruncated = false;
  std::size_t repairRounds = 0;
  std::size_t repairTrials = 0;
  std::size_t repairWorkUnits = 0;
  std::size_t backstopDeletionTrials = 0;
  std::size_t backstopDeletionWorkUnits = 0;
  std::size_t selectedEvents = 0;
  std::size_t selectedTargetedBarriers = 0;
  std::size_t selectedPipeAllBarriers = 0;
  std::size_t emittedEventSets = 0;
  std::size_t emittedEventWaits = 0;
  std::size_t emittedTargetedBarriers = 0;
  std::size_t emittedZeroDistanceTargetedBarriers = 0;
  std::size_t emittedRecurrenceTargetedBarriers = 0;
  std::size_t emittedZeroOnlyTargetedBarriers = 0;
  std::size_t emittedRecurrenceOnlyTargetedBarriers = 0;
  std::size_t emittedMixedDistanceTargetedBarriers = 0;
  std::size_t emittedTargetLocalPipeDrainBarriers = 0;
  std::size_t emittedLoopCarryPipeDrainBarriers = 0;
  std::size_t emittedSourceLocalPipeDrainBarriers = 0;
  std::size_t emittedSourcePrefixPipeDrainBarriers = 0;
  std::size_t emittedPipeAllBarriers = 0;
  std::size_t predictedSyncInstructions = 0;
  std::size_t verificationWorkUnits = 0;
  std::uint64_t selectionNanoseconds = 0;
  std::uint64_t repairNanoseconds = 0;
  std::uint64_t verificationNanoseconds = 0;
  std::uint64_t planSignature = 0;
  CanonicalSyncStructuralCost cost;
  CanonicalSyncGreedyStatistics search;
  CanonicalSyncResourceAllocation allocation;
  CanonicalSyncGreedyStatistics preciseSearch;
  CanonicalSyncResourceAllocation preciseAllocation;
};

struct CanonicalSyncComparisonReport {
  std::string function;
  CanonicalSyncGmAliasPolicy gmAliasPolicy =
      CanonicalSyncGmAliasPolicy::MayAlias;
  std::size_t graphNodes = 0;
  std::size_t graphEdges = 0;
  std::size_t certifiedCompletionFrontiers = 0;
  std::size_t demands = 0;
  std::size_t uniqueDemandRows = 0;
  std::size_t selectionBasisRows = 0;
  std::size_t basisReducedRows = 0;
  bool basisReductionTruncated = false;
  std::size_t zeroDistanceDemandRows = 0;
  std::size_t recurrenceDemandRows = 0;
  std::size_t sameResourceDemandRows = 0;
  std::size_t crossResourceDemandRows = 0;
  std::size_t ssaDemandRows = 0;
  std::size_t rawDemandRows = 0;
  std::size_t warDemandRows = 0;
  std::size_t wawDemandRows = 0;
  unsigned maximumRecurrenceDistance = 0;
  std::size_t directMechanisms = 0;
  std::size_t directPairProposals = 0;
  std::size_t directPairEvaluations = 0;
  std::size_t synergisticPairs = 0;
  bool pairGenerationTruncated = false;
  std::size_t sourcePrefixInspections = 0;
  std::size_t sourcePrefixCandidates = 0;
  std::size_t sourcePrefixIncidences = 0;
  bool sourcePrefixGenerationTruncated = false;
  std::size_t loopCarryInspections = 0;
  std::size_t loopCarryCandidates = 0;
  std::size_t loopCarryIncidences = 0;
  bool loopCarryGenerationTruncated = false;
  std::size_t loopBoundaryProtocolInspections = 0;
  std::size_t loopBoundaryProtocolCandidates = 0;
  std::size_t loopBoundaryProtocolIncidences = 0;
  bool loopBoundaryProtocolGenerationTruncated = false;
  std::size_t slotLifecycleInspections = 0;
  std::size_t slotLifecycleCandidates = 0;
  std::size_t slotLifecycleConflictIncidences = 0;
  bool slotLifecycleGenerationTruncated = false;
  std::uint64_t preparationNanoseconds = 0;
  std::vector<CanonicalSyncStrategyReport> strategies;
};

struct CanonicalSyncBuildOptions {
  unsigned eventIdBudget = 8;
  CanonicalSyncAnalysisOptions analysis;
  CanonicalSyncPatternOptions patterns;
  CanonicalSyncDirectPairOptions directPairs;
  CanonicalSyncPatternProblem::Limits problemLimits;
  SyncCoverExpansionLimits expansionLimits;
  CanonicalSyncGreedyOptions selection;
  std::size_t maximumRepairRounds = 8;
  std::size_t maximumRepairTrials = 256;
  std::size_t maximumRepairWorkUnits = 1U << 28;
  std::size_t maximumBackstopDeletionTrials = 4096;
  std::size_t maximumBackstopDeletionWorkUnits = 1U << 27;
  std::size_t maximumVerificationWorkUnits = 1U << 27;
  bool enableDemandBasisReduction = true;
  std::size_t maximumDemandBasisGroupEdges = 1U << 18;
  std::size_t maximumDemandBasisReachabilityWords = 1U << 20;
  std::size_t maximumDemandBasisReductionWork = 1U << 24;
  bool analysisOnly = false;
  bool compareSelectionStrategies = false;
  std::function<LogicalResult(const CanonicalSyncComparisonReport &)>
      reportCallback;
};

/// Result of building one immutable candidate catalog. A precise catalog must
/// freeze completely; uncoverable rows and construction limits fail closed.
struct CanonicalSyncProblemBuildResult {
  std::unique_ptr<CanonicalSyncPatternProblem> problem;
  CanonicalSyncProblemResult status;
  /// Repair-only mechanisms keyed by the precise pressure-core event whose
  /// removal admitted them. These candidates must remain hidden from every
  /// other individual repair trial.
  std::map<CanonicalSyncMechanismId, std::vector<CanonicalSyncMechanismId>>
      repairMechanismsByOwner;
  /// Multi-event frontier mechanisms exposed only by the collective core
  /// trial.
  std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms;

  explicit operator bool() const {
    return problem != nullptr && static_cast<bool>(status);
  }
};

CanonicalSyncProblemBuildResult
buildCanonicalSyncPreciseProblem(const CanonicalSyncProgram &program,
                                 const CanonicalSyncBuildOptions &options);

CanonicalSyncProblemBuildResult buildCanonicalSyncRepairProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &preciseProblem,
    const CanonicalSyncBuildOptions &options,
    const std::vector<CanonicalSyncMechanismId> &conflictCore,
    const std::vector<CanonicalSyncMechanismId> &selectedMechanisms = {});

CanonicalSyncProblemBuildResult
buildCanonicalSyncPipeAllProblem(const CanonicalSyncProgram &program,
                                 const CanonicalSyncBuildOptions &options);

/// Build and freeze the singleton candidate problem. The returned problem
/// retains a non-owning reference to program and must not outlive or move it.
FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
buildCanonicalSyncSingletonProblem(const CanonicalSyncProgram &program,
                                   const CanonicalSyncBuildOptions &options);

/// Validate every concrete anchor and allocation before modifying the IR, then
/// emit each selected atomic recipe exactly once.
LogicalResult
materializeCanonicalSyncPlan(const CanonicalSyncProgram &program,
                             const CanonicalSyncPatternProblem &problem,
                             const CanonicalSyncVerifiedPlan &plan);

/// Run analysis, bounded-pattern selection, bitset-based finalization, and
/// materialization while all referenced graph storage remains alive.
LogicalResult runCanonicalSync(func::FuncOp function,
                               const CanonicalSyncBuildOptions &options = {});

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
