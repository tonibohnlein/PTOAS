// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- CanonicalSync.h - Bounded pattern synchronization ------*- C++ -*-===//

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNC_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSyncSelection.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

/// Independently selectable derived-mechanism families. The direct
/// correctness atoms remain enabled for every mask.
enum class CanonicalSyncMechanismFamily : std::uint8_t {
  CompletionFrontier,
  TargetCompletionCertificate,
  TargetLocalFence,
  SourceLocalCompletion,
  SourceLocalDrain,
  SourcePrefixDrain,
  LoopCarryDrain,
  LoopBoundaryProtocol,
  L0OperandOwnership,
  BasicOwnership,
  BoundaryOwnership,
  HierarchicalOwnership,
  RepairSourceLocalDrain,
  RepairSourcePrefixDrain,
  RepairTargetLocalDrain,
  RepairFrontier,
  StorageCutEvent,
  Count,
};

using CanonicalSyncMechanismFamilyMask = std::uint32_t;

constexpr std::size_t kCanonicalSyncMechanismFamilyCount =
    static_cast<std::size_t>(CanonicalSyncMechanismFamily::Count);
static_assert(kCanonicalSyncMechanismFamilyCount <
                  sizeof(CanonicalSyncMechanismFamilyMask) * 8,
              "canonical sync mechanism families must fit their mask");

constexpr CanonicalSyncMechanismFamilyMask
canonicalSyncMechanismFamilyBit(CanonicalSyncMechanismFamily family) {
  return CanonicalSyncMechanismFamilyMask{1} << static_cast<unsigned>(family);
}

constexpr CanonicalSyncMechanismFamilyMask kAllCanonicalSyncMechanismFamilies =
    (CanonicalSyncMechanismFamilyMask{1}
     << kCanonicalSyncMechanismFamilyCount) -
    1;

/// Production-default derived families. Newly synthesized storage-cut events
/// remain opt-in until their emitted plans pass the device correctness gate.
constexpr CanonicalSyncMechanismFamilyMask
    kDefaultCanonicalSyncMechanismFamilies =
        kAllCanonicalSyncMechanismFamilies &
        ~canonicalSyncMechanismFamilyBit(
            CanonicalSyncMechanismFamily::StorageCutEvent);

constexpr bool
canonicalSyncMechanismFamilyEnabled(CanonicalSyncMechanismFamilyMask mask,
                                    CanonicalSyncMechanismFamily family) {
  return (mask & canonicalSyncMechanismFamilyBit(family)) != 0;
}

/// Catalog construction policy. StrictMinimalDirect is an explicit
/// correctness baseline rather than a special interpretation of an empty
/// derived-family mask.
enum class CanonicalSyncCatalogMode : std::uint8_t {
  Standard,
  StrictMinimalDirect,
};

/// Internal ablation controls for synchronization mechanism construction.
/// Repair frontiers are grounded only from a live allocation conflict core in
/// a separately owned repair problem.
struct CanonicalSyncPatternOptions {
  CanonicalSyncCatalogMode catalogMode = CanonicalSyncCatalogMode::Standard;
  CanonicalSyncMechanismFamilyMask enabledMechanismFamilies =
      kDefaultCanonicalSyncMechanismFamilies;
  bool enableDirectPairs = true;
  bool enableConflictCoreRepair = true;
  /// Optional same-round acceleration. Tests may disable it to exercise the
  /// ordinary multi-round changed-core path without changing its semantics.
  bool enableCollectiveRepairTrial = true;
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
};

struct CanonicalSyncSelectedMechanismReport {
  CanonicalSyncMechanismId mechanism = 0;
  CanonicalSyncMechanismKind kind = CanonicalSyncMechanismKind::Event;
  CanonicalSyncMechanismOriginMask originMask = 0;
  std::size_t supplies = 0;
  std::size_t groundedCoverageRows = 0;
  std::size_t exactSupplyDemandRows = 0;
  std::size_t additionalGroundedCoverageRows = 0;
  std::vector<SyncCoverDemandId> exactSupplyDemands;
  std::vector<SyncCoverDemandId> groundedCoveredDemands;
  bool demandDetailsTruncated = false;
  std::size_t eventUses = 0;
  std::size_t actions = 0;
  std::size_t eventSets = 0;
  std::size_t eventWaits = 0;
  std::size_t targetedBarriers = 0;
  std::size_t pipeAllBarriers = 0;
  unsigned maximumRecurrenceDistance = 0;
};

/// Bounded, stable graph provenance for one original synchronization row.
/// Operation names are intentionally omitted: node IDs remain authoritative
/// across macro phases and keep report construction independent of MLIR text.
struct CanonicalSyncDemandReport {
  SyncCoverDemandId demand = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  std::vector<SyncCoverDemandKind> provenanceKinds;
  std::vector<SyncCoverStorageWitnessId> storageWitnesses;
  std::size_t originalDemandCount = 0;
  std::size_t sourceGuardLiterals = 0;
  std::size_t targetGuardLiterals = 0;
};

struct CanonicalSyncStorageLifecycleComponentReport {
  SyncCoverStorageLifecycleComponentId component = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t slots = 0;
  std::size_t epochs = 0;
  std::size_t edges = 0;
  std::size_t demands = 0;
  std::size_t sccs = 0;
  std::size_t cyclicSccs = 0;
  std::size_t readyReleaseSccs = 0;
  std::size_t sccTransfers = 0;
  std::size_t transitionClasses = 0;
};

struct CanonicalSyncStorageLifecycleTransitionReport {
  SyncCoverStorageLifecycleComponentId component = 0;
  SyncCoverStorageLifecycleTransitionClassId transition = 0;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
  SyncCoverGuard sourceGuard;
  SyncCoverGuard targetGuard;
  std::size_t edges = 0;
};

struct CanonicalSyncStorageProtocolSeedReport {
  SyncCoverStorageProtocolSeedId seed = 0;
  SyncCoverStorageAccessFamilyId family = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t components = 0;
  std::size_t slots = 0;
  std::size_t readyReleaseSccs = 0;
  std::size_t demands = 0;
  SyncCoverStorageLifecycleEdgeKindMask kinds = 0;
  unsigned maximumDistance = 0;
};

struct CanonicalSyncStorageProtocolGroupReport {
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  SyncCoverStorageProtocolBehavior behavior =
      SyncCoverStorageProtocolBehavior::StableRoundTrip;
  std::uint32_t readySourceResource = 0;
  std::uint32_t readyTargetResource = 0;
  std::vector<std::uint64_t> behaviorSignature;
  std::vector<SyncCoverStorageProtocolSeedId> seeds;
  std::size_t periodicControls = 0;
  std::size_t demands = 0;
  unsigned maximumDistance = 0;
};

struct CanonicalSyncStorageProtocolAutomatonReport {
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t states = 0;
  std::size_t transfers = 0;
  std::size_t statePairIncidences = 0;
  unsigned maximumDistance = 0;
};

struct CanonicalSyncStorageProtocolCutPlanReport {
  SyncCoverStorageProtocolCutPlanId plan = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t lanes = 0;
  std::size_t directReadyTransfers = 0;
  std::size_t recurrenceReleaseTransfers = 0;
  std::size_t readyRectangles = 0;
};

struct CanonicalSyncStorageProtocolFrontierPlanReport {
  SyncCoverStorageProtocolFrontierPlanId plan = 0;
  SyncCoverStorageProtocolAutomatonId automaton = 0;
  SyncCoverStorageProtocolGroupId group = 0;
  SyncCoverScopeId owningScope = 0;
  std::size_t lanes = 0;
  std::size_t frontiers = 0;
  std::size_t readyFrontiers = 0;
  std::size_t reuseFrontiers = 0;
  std::size_t directFrontiers = 0;
  std::size_t completionCutFactFrontiers = 0;
  std::size_t certificateFrontiers = 0;
  std::size_t sameResourceRecurrenceReuses = 0;
};

struct CanonicalSyncSyntheticRectangleGroundingReport {
  SyncCoverStorageFactoredRectangleId rectangle = 0;
  SyncCoverStorageCutId completionCut = 0;
  SyncCoverStorageCutId acquisitionCut = 0;
  std::size_t coverageRows = 0;
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
  std::size_t repairCatalogRebuilds = 0;
  /// Cumulative shared work at the first committed catalog replacement.
  std::size_t firstRepairCatalogRebuildWorkUnits = 0;
  std::size_t repairTrials = 0;
  std::size_t repairWorkUnits = 0;
  std::size_t backstopDeletionTrials = 0;
  std::size_t backstopDeletionWorkUnits = 0;
  std::size_t selectedEvents = 0;
  std::size_t selectedTargetedBarriers = 0;
  std::size_t selectedPipeAllBarriers = 0;
  std::array<std::size_t, kCanonicalSyncMechanismOriginCount>
      selectedMechanismsByOrigin{};
  std::size_t activeDirectPairs = 0;
  std::size_t activeDirectPairExtraCoverage = 0;
  bool selectedMechanismDetailsTruncated = false;
  std::vector<CanonicalSyncSelectedMechanismReport> selectedMechanisms;
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
  CanonicalSyncTargetCapabilities targetCapabilities;
  CanonicalSyncSelectionObjective selectionObjective =
      CanonicalSyncSelectionObjective::ActionFirst;
  CanonicalSyncCatalogMode catalogMode = CanonicalSyncCatalogMode::Standard;
  CanonicalSyncMechanismFamilyMask enabledMechanismFamilies =
      kDefaultCanonicalSyncMechanismFamilies;
  bool directPairsEnabled = true;
  bool conflictCoreRepairEnabled = true;
  CanonicalSyncGmAliasPolicy gmAliasPolicy =
      CanonicalSyncGmAliasPolicy::MayAlias;
  std::size_t graphNodes = 0;
  std::size_t graphEdges = 0;
  std::size_t certifiedCompletionFrontiers = 0;
  std::size_t ownershipDiscoveryInspections = 0;
  std::array<std::size_t, kCanonicalSyncBasicOwnershipKindCount>
      ownershipCertificatesByKind{};
  std::size_t ownershipSlots = 0;
  std::size_t ownershipPaths = 0;
  std::size_t ownershipUses = 0;
  std::size_t ownershipNodeReferences = 0;
  std::size_t ownershipAccessIncidences = 0;
  bool ownershipDiscoveryTruncated = false;
  bool storageLifecycleAnalysisEnabled = false;
  std::size_t storageLifecycleWorkUnits = 0;
  std::size_t storageLifecycleEligibleWitnesses = 0;
  std::size_t storageLifecycleIneligibleWitnesses = 0;
  std::size_t storageLifecycleComponents = 0;
  std::size_t storageLifecycleSlots = 0;
  std::size_t storageLifecycleEpochs = 0;
  std::size_t storageLifecycleEdges = 0;
  std::size_t storageLifecycleDemandIncidences = 0;
  std::size_t storageLifecycleSccs = 0;
  std::size_t storageLifecycleCyclicSccs = 0;
  std::size_t storageLifecycleReadyReleaseSccs = 0;
  std::size_t storageLifecycleSccTransfers = 0;
  std::size_t storageLifecycleMaximumSccEpochs = 0;
  std::size_t storageLifecycleTransitionClasses = 0;
  std::size_t storageLifecycleTransitionGuardLiterals = 0;
  std::size_t storageLifecycleMaximumTransitionClassEdges = 0;
  bool storageLifecycleDetailsTruncated = false;
  std::vector<CanonicalSyncStorageLifecycleComponentReport>
      storageLifecycleComponentDetails;
  std::vector<CanonicalSyncStorageLifecycleTransitionReport>
      storageLifecycleTransitionDetails;
  bool storageLifecycleTruncated = false;
  bool storageProtocolSeedAnalysisEnabled = false;
  std::size_t storageProtocolSeedWorkUnits = 0;
  std::size_t storageProtocolSeeds = 0;
  std::size_t storageProtocolReadyReleaseSeeds = 0;
  std::size_t storageProtocolComponentIncidences = 0;
  std::size_t storageProtocolSlotIncidences = 0;
  std::size_t storageProtocolSccIncidences = 0;
  std::size_t storageProtocolDemandIncidences = 0;
  std::size_t storageProtocolMaximumSeedComponents = 0;
  std::size_t storageProtocolMaximumSeedSlots = 0;
  std::size_t storageProtocolMaximumSeedSccs = 0;
  std::vector<CanonicalSyncStorageProtocolSeedReport>
      storageProtocolSeedDetails;
  bool storageProtocolSeedDetailsTruncated = false;
  bool storageProtocolSeedTruncated = false;
  bool storageProtocolGroupAnalysisEnabled = false;
  std::size_t storageProtocolGroupWorkUnits = 0;
  std::size_t storageProtocolEligibleSeeds = 0;
  std::size_t storageProtocolIneligibleSeeds = 0;
  std::size_t storageProtocolStableSeeds = 0;
  std::size_t storageProtocolPhaseRotatingSeeds = 0;
  std::size_t storageProtocolGroups = 0;
  std::size_t storageProtocolGroupSeedIncidences = 0;
  std::size_t storageProtocolGroupControlIncidences = 0;
  std::size_t storageProtocolGroupDemandIncidences = 0;
  std::size_t storageProtocolGroupSlotIncidences = 0;
  std::size_t storageProtocolGroupJointStateIncidences = 0;
  std::size_t storageProtocolMaximumGroupSeeds = 0;
  std::vector<CanonicalSyncStorageProtocolGroupReport>
      storageProtocolGroupDetails;
  bool storageProtocolGroupDetailsTruncated = false;
  bool storageProtocolGroupTruncated = false;
  bool storageProtocolAutomatonAnalysisEnabled = false;
  std::size_t storageProtocolAutomatonWorkUnits = 0;
  std::size_t storageProtocolAutomatonEligibleGroups = 0;
  std::size_t storageProtocolAutomatonIneligibleGroups = 0;
  std::size_t storageProtocolAutomatonLaneLimitedGroups = 0;
  std::size_t storageProtocolAutomatonScopeRejectedGroups = 0;
  std::size_t storageProtocolAutomatonMembershipRejectedGroups = 0;
  std::size_t storageProtocolAutomatonDirectionRejectedGroups = 0;
  std::size_t storageProtocolAutomatonUnreachableTransferGroups = 0;
  std::size_t storageProtocolAutomatonUnreachableReadyTransferGroups = 0;
  std::size_t storageProtocolAutomatonUnreachableReleaseTransferGroups = 0;
  std::size_t storageProtocolAutomatonUnreachableExclusionTransferGroups = 0;
  std::size_t storageProtocolAutomatonDemandSetMismatchGroups = 0;
  std::size_t storageProtocolAutomatonDistanceMismatchGroups = 0;
  std::size_t storageProtocolAutomata = 0;
  std::size_t storageProtocolAutomatonStates = 0;
  std::size_t storageProtocolAutomatonTransfers = 0;
  std::size_t storageProtocolAutomatonStatePairIncidences = 0;
  std::size_t storageProtocolMaximumAutomatonTransfers = 0;
  std::size_t storageProtocolMaximumTransferStatePairs = 0;
  std::vector<CanonicalSyncStorageProtocolAutomatonReport>
      storageProtocolAutomatonDetails;
  bool storageProtocolAutomatonDetailsTruncated = false;
  bool storageProtocolAutomatonTruncated = false;
  bool storageProtocolCutPlanAnalysisEnabled = false;
  std::size_t storageProtocolCutPlanWorkUnits = 0;
  std::size_t storageProtocolCutPlanEligibleAutomata = 0;
  std::size_t storageProtocolCutPlanIneligibleAutomata = 0;
  std::size_t storageProtocolCutPlanMissingReadyCutAutomata = 0;
  std::size_t storageProtocolCutPlanMissingReleaseAutomata = 0;
  std::size_t storageProtocolCutPlans = 0;
  std::size_t storageProtocolCutPlanTransferInspections = 0;
  std::size_t storageProtocolCutPlanDirectReadyTransfers = 0;
  std::size_t storageProtocolCutPlanRecurrenceReleaseTransfers = 0;
  std::size_t storageProtocolCutPlanReadyRectangleIncidences = 0;
  std::size_t storageProtocolCutPlanMaximumReadyRectangles = 0;
  std::vector<CanonicalSyncStorageProtocolCutPlanReport>
      storageProtocolCutPlanDetails;
  bool storageProtocolCutPlanDetailsTruncated = false;
  bool storageProtocolCutPlanTruncated = false;
  bool storageProtocolFrontierAnalysisEnabled = false;
  std::size_t storageProtocolFrontierWorkUnits = 0;
  std::size_t storageProtocolFrontierEligibleAutomata = 0;
  std::size_t storageProtocolFrontierIneligibleAutomata = 0;
  std::size_t storageProtocolFrontierMissingReadyAutomata = 0;
  std::size_t storageProtocolFrontierMissingRecurrenceReuseAutomata = 0;
  std::size_t storageProtocolFrontierMissingCompletionAutomata = 0;
  std::size_t storageProtocolFrontierPlans = 0;
  std::size_t storageProtocolFrontiers = 0;
  std::size_t storageProtocolReadyFrontiers = 0;
  std::size_t storageProtocolReuseFrontiers = 0;
  std::size_t storageProtocolDirectFrontiers = 0;
  std::size_t storageProtocolCompletionCutFactFrontiers = 0;
  std::size_t storageProtocolCertificateFrontiers = 0;
  std::size_t storageProtocolSameResourceRecurrenceReuses = 0;
  std::size_t storageProtocolFrontierCertificateDemandIncidences = 0;
  std::size_t storageProtocolFrontierCompletionCutFactDemandIncidences = 0;
  std::size_t storageProtocolFrontierTransferInspections = 0;
  std::size_t storageProtocolFrontierStatePairInspections = 0;
  std::size_t storageProtocolFrontierPlanIncidences = 0;
  std::size_t storageProtocolMaximumPlanFrontiers = 0;
  std::vector<CanonicalSyncStorageProtocolFrontierPlanReport>
      storageProtocolFrontierDetails;
  bool storageProtocolFrontierDetailsTruncated = false;
  bool storageProtocolFrontierTruncated = false;
  bool storageCutAnalysisEnabled = false;
  std::size_t storageCutWorkUnits = 0;
  std::size_t storageCutEligibleEdges = 0;
  std::size_t storageCutIneligibleEdges = 0;
  std::size_t storageCompletionCuts = 0;
  std::size_t storageAcquisitionCuts = 0;
  std::size_t storageRectangles = 0;
  std::size_t storageCutIncidences = 0;
  std::size_t storageCutGuardLiterals = 0;
  std::size_t storageMaximumRectangleEdges = 0;
  bool storageCutTruncated = false;
  bool storageRectangleAnalysisEnabled = false;
  std::size_t storageRectangleWorkUnits = 0;
  std::size_t storageRectangleInspections = 0;
  std::size_t storageFactoredRectangles = 0;
  std::size_t storageDirectFactoredRectangles = 0;
  std::size_t storageSyntheticFactoredRectangles = 0;
  std::size_t storageRectangleGuardLiterals = 0;
  bool storageRectangleTruncated = false;
  bool storageSyntheticRectangleGroundingEnabled = false;
  std::size_t storageSyntheticRectangleGroundingWorkUnits = 0;
  std::size_t storageSyntheticRectangleGroundingEvaluated = 0;
  std::size_t storageSyntheticRectanglesWithCoverage = 0;
  std::size_t storageSyntheticRectanglesCoveringMultipleRows = 0;
  std::size_t storageSyntheticRectangleMaximumCoverageRows = 0;
  std::size_t storageSyntheticRectangleTotalCoverageRows = 0;
  SyncCoverSyntheticRectangleGroundingError
      storageSyntheticRectangleGroundingError =
          SyncCoverSyntheticRectangleGroundingError::None;
  std::vector<CanonicalSyncSyntheticRectangleGroundingReport>
      storageSyntheticRectangleGroundingDetails;
  bool storageSyntheticRectangleGroundingDetailsTruncated = false;
  bool storageSyntheticRectangleGroundingTruncated = false;
  std::size_t demands = 0;
  std::size_t uniqueDemandRows = 0;
  bool demandDetailsTruncated = false;
  std::vector<CanonicalSyncDemandReport> demandDetails;
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
  std::size_t singletonCandidatesCoveringMultipleRows = 0;
  std::size_t maximumSingletonCandidateCoverageRows = 0;
  std::size_t totalSingletonCandidateCoverageRows = 0;
  std::array<std::size_t, kCanonicalSyncMechanismOriginCount>
      candidateMechanismsByOrigin{};
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
  /// Stable graph-demand provenance for every precise owner represented in a
  /// repair catalog. Changed-core rebuilding carries these rows forward so an
  /// owner that was forbidden in an earlier round does not lose the certified
  /// replacement mechanisms that keep its obligations covered.
  std::map<CanonicalSyncMechanismId, std::vector<SyncCoverDemandId>>
      repairCriticalDemandsByOwner;
  /// Multi-event frontier mechanisms exposed only by the collective core
  /// trial.
  std::vector<CanonicalSyncMechanismId> collectiveRepairMechanisms;

  explicit operator bool() const {
    return problem != nullptr && static_cast<bool>(status);
  }
};

/// Non-owning changed-core seed for one previously forbidden precise owner.
/// The caller retains the owning build-result map for the duration of repair
/// catalog construction; only owners named in the new accumulated core may be
/// supplied.
struct CanonicalSyncRepairCriticalDemandSeed {
  CanonicalSyncMechanismId owner = 0;
  llvm::ArrayRef<SyncCoverDemandId> demands;
};

CanonicalSyncProblemBuildResult
buildCanonicalSyncPreciseProblem(const CanonicalSyncProgram &program,
                                 const CanonicalSyncBuildOptions &options);

CanonicalSyncProblemBuildResult buildCanonicalSyncRepairProblem(
    const CanonicalSyncProgram &program,
    const CanonicalSyncPatternProblem &preciseProblem,
    const CanonicalSyncBuildOptions &options,
    const std::vector<CanonicalSyncMechanismId> &conflictCore,
    const std::vector<CanonicalSyncMechanismId> &selectedMechanisms = {},
    SyncCoverCoverageWorkBudget *workBudget = nullptr,
    llvm::ArrayRef<CanonicalSyncRepairCriticalDemandSeed>
        retainedCriticalDemands = {});

/// Recompute repair-only visibility for one exclusion trial. Explicitly
/// forced repair IDs remain forbidden even when their owner is otherwise
/// allowed; policy exclusions left by an earlier trial are replaced.
bool prepareCanonicalSyncRepairTrial(
    CanonicalSyncGreedyOptions &trialOptions,
    llvm::ArrayRef<CanonicalSyncMechanismId> allRepairMechanisms,
    llvm::ArrayRef<const std::vector<CanonicalSyncMechanismId> *>
        repairMechanismsByOwner,
    llvm::ArrayRef<CanonicalSyncMechanismId> collectiveRepairMechanisms,
    llvm::ArrayRef<CanonicalSyncMechanismId> owners, bool collective,
    llvm::ArrayRef<CanonicalSyncMechanismId> forcedRepairExclusions,
    SyncCoverCoverageWorkBudget *workBudget = nullptr);

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
