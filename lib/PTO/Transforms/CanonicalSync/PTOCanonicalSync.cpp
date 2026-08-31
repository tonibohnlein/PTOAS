// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOCANONICALSYNC
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

constexpr std::int64_t kCompilerUsableEventIdCount = 6;

std::int64_t jsonInteger(std::uint64_t value) {
  constexpr std::uint64_t maximum =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  return static_cast<std::int64_t>(std::min(value, maximum));
}

std::string jsonSignature(std::uint64_t value) {
  return "0x" + llvm::utohexstr(value, /*LowerCase=*/true);
}

bool configurePatternMode(StringRef mode,
                          pto::CanonicalSyncPatternOptions &options) {
  if (mode == "direct") {
    options.enableDirectPairs = false;
    return true;
  }
  if (mode == "direct-pair") {
    options.enableDirectPairs = true;
    return true;
  }
  return false;
}

StringRef mechanismFamilyName(pto::CanonicalSyncMechanismFamily family) {
  using Family = pto::CanonicalSyncMechanismFamily;
  switch (family) {
  case Family::CompletionFrontier:
    return "completion-frontier";
  case Family::TargetCompletionCertificate:
    return "target-completion-certificate";
  case Family::TargetLocalFence:
    return "target-local-fence";
  case Family::SourceLocalCompletion:
    return "source-local-completion";
  case Family::SourceLocalDrain:
    return "source-local-drain";
  case Family::SourcePrefixDrain:
    return "source-prefix-drain";
  case Family::LoopCarryDrain:
    return "loop-carry-drain";
  case Family::LoopBoundaryProtocol:
    return "loop-boundary-protocol";
  case Family::L0OperandOwnership:
    return "l0-operand-ownership";
  case Family::BasicOwnership:
    return "basic-ownership";
  case Family::BoundaryOwnership:
    return "boundary-ownership";
  case Family::HierarchicalOwnership:
    return "hierarchical-ownership";
  case Family::RepairSourceLocalDrain:
    return "repair-source-local-drain";
  case Family::RepairSourcePrefixDrain:
    return "repair-source-prefix-drain";
  case Family::RepairTargetLocalDrain:
    return "repair-target-local-drain";
  case Family::RepairFrontier:
    return "repair-frontier";
  case Family::StorageCutEvent:
    return "storage-cut-event";
  case Family::Count:
    break;
  }
  return "unknown";
}

bool configureMechanismFamilies(StringRef value,
                                pto::CanonicalSyncPatternOptions &options) {
  if (value == "default") {
    options.enabledMechanismFamilies =
        pto::kDefaultCanonicalSyncMechanismFamilies;
    return true;
  }
  if (value == "all") {
    options.enabledMechanismFamilies = pto::kAllCanonicalSyncMechanismFamilies;
    return true;
  }
  if (value == "core") {
    options.enabledMechanismFamilies = 0;
    return true;
  }
  SmallVector<StringRef> names;
  value.split(names, '+', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (names.empty()) {
    return false;
  }
  pto::CanonicalSyncMechanismFamilyMask mask = 0;
  for (StringRef name : names) {
    bool found = false;
    for (std::size_t index = 0; index < pto::kCanonicalSyncMechanismFamilyCount;
         ++index) {
      const auto family = static_cast<pto::CanonicalSyncMechanismFamily>(index);
      if (name != mechanismFamilyName(family)) {
        continue;
      }
      const pto::CanonicalSyncMechanismFamilyMask bit =
          pto::canonicalSyncMechanismFamilyBit(family);
      if ((mask & bit) != 0) {
        return false;
      }
      mask |= bit;
      found = true;
      break;
    }
    if (!found) {
      return false;
    }
  }
  options.enabledMechanismFamilies = mask;
  return true;
}

bool configureCatalogMode(StringRef value,
                          pto::CanonicalSyncPatternOptions &options) {
  if (value == "standard") {
    options.catalogMode = pto::CanonicalSyncCatalogMode::Standard;
    return true;
  }
  if (value == "strict-direct") {
    options.catalogMode = pto::CanonicalSyncCatalogMode::StrictMinimalDirect;
    return true;
  }
  return false;
}

StringRef catalogModeName(pto::CanonicalSyncCatalogMode mode) {
  switch (mode) {
  case pto::CanonicalSyncCatalogMode::Standard:
    return "standard";
  case pto::CanonicalSyncCatalogMode::StrictMinimalDirect:
    return "strict-direct";
  }
  return "unknown";
}

bool configureSelectionStrategy(StringRef strategy,
                                pto::CanonicalSyncGreedyOptions &options) {
  if (strategy == "fixed-cover") {
    options.strategy = pto::CanonicalSyncSelectionStrategy::FixedCover;
    return true;
  }
  if (strategy == "action-aware-singleton") {
    options.strategy =
        pto::CanonicalSyncSelectionStrategy::ActionAwareSingleton;
    return true;
  }
  if (strategy == "pair-lookahead") {
    options.strategy = pto::CanonicalSyncSelectionStrategy::PairLookahead;
    return true;
  }
  return false;
}

bool configureSelectionObjective(StringRef objective,
                                 pto::CanonicalSyncGreedyOptions &options) {
  if (objective == "action-first") {
    options.objective = pto::CanonicalSyncSelectionObjective::ActionFirst;
    return true;
  }
  if (objective == "serialization-first") {
    options.objective =
        pto::CanonicalSyncSelectionObjective::SerializationFirst;
    return true;
  }
  return false;
}

StringRef strategyName(pto::CanonicalSyncSelectionStrategy strategy) {
  switch (strategy) {
  case pto::CanonicalSyncSelectionStrategy::FixedCover:
    return "fixed-cover";
  case pto::CanonicalSyncSelectionStrategy::ActionAwareSingleton:
    return "action-aware-singleton";
  case pto::CanonicalSyncSelectionStrategy::PairLookahead:
    return "pair-lookahead";
  }
  return "unknown";
}

StringRef objectiveName(pto::CanonicalSyncSelectionObjective objective) {
  switch (objective) {
  case pto::CanonicalSyncSelectionObjective::ActionFirst:
    return "action-first";
  case pto::CanonicalSyncSelectionObjective::SerializationFirst:
    return "serialization-first";
  }
  return "unknown";
}

StringRef targetProfileName(pto::CanonicalSyncTargetProfile profile) {
  switch (profile) {
  case pto::CanonicalSyncTargetProfile::Unsupported:
    return "unsupported";
  case pto::CanonicalSyncTargetProfile::A2V1:
    return "a2-v1";
  case pto::CanonicalSyncTargetProfile::A2A3IntersectionV1:
    return "a2a3-intersection-v1";
  case pto::CanonicalSyncTargetProfile::A3V1:
    return "a3-v1";
  case pto::CanonicalSyncTargetProfile::A5V1:
    return "a5-v1";
  }
  return "unknown";
}

StringRef coreDomainName(pto::CanonicalSyncCoreDomain core) {
  switch (core) {
  case pto::CanonicalSyncCoreDomain::Unresolved:
    return "unresolved";
  case pto::CanonicalSyncCoreDomain::AIC:
    return "aic";
  case pto::CanonicalSyncCoreDomain::AIV:
    return "aiv";
  case pto::CanonicalSyncCoreDomain::Conflict:
    return "conflict";
  }
  return "unknown";
}

StringRef syncSpecVersionName(pto::CanonicalSyncTargetSyncSpecVersion version) {
  switch (version) {
  case pto::CanonicalSyncTargetSyncSpecVersion::None:
    return "none";
  case pto::CanonicalSyncTargetSyncSpecVersion::Ascend2201V1:
    return "ascend-2201-v1";
  case pto::CanonicalSyncTargetSyncSpecVersion::Ascend3510PartialV1:
    return "ascend-3510-partial-v1";
  }
  return "unknown";
}

StringRef targetEvidenceName(pto::CanonicalSyncTargetEvidence evidence) {
  switch (evidence) {
  case pto::CanonicalSyncTargetEvidence::None:
    return "none";
  case pto::CanonicalSyncTargetEvidence::AscendIntraCoreSync7008190b:
    return "ascend-intra-core-sync@7008190b";
  case pto::CanonicalSyncTargetEvidence::AscendSetWait7008190b:
    return "ascend-set-wait@7008190b";
  case pto::CanonicalSyncTargetEvidence::AscendKeyFeatures7008190b:
    return "ascend-sync-key-features@7008190b";
  case pto::CanonicalSyncTargetEvidence::AscendPipeBarrier850:
    return "ascend-pipe-barrier@cann-8.5.0";
  case pto::CanonicalSyncTargetEvidence::PTOASInsertSyncAccRarFfe46c09:
    return "ptoas-insert-sync-acc-rar@ffe46c09";
  }
  return "unknown";
}

llvm::json::Array jsonUnsignedValues(ArrayRef<std::uint64_t> values) {
  llvm::json::Array result;
  for (std::uint64_t value : values) {
    result.push_back(jsonInteger(value));
  }
  return result;
}

llvm::json::Object
jsonResourceCapability(const pto::CanonicalSyncResourceCapability &capability) {
  llvm::json::Array resources;
  for (std::uint32_t resource : capability.resources) {
    resources.push_back(jsonInteger(resource));
  }
  return llvm::json::Object{{"version", jsonInteger(capability.version)},
                            {"resources", std::move(resources)}};
}

llvm::json::Object jsonDirectedResourceCapability(
    const pto::CanonicalSyncDirectedResourceCapability &capability) {
  llvm::json::Array resourcePairs;
  for (const auto &[source, target] : capability.resourcePairs) {
    resourcePairs.push_back(llvm::json::Object{
        {"source", jsonInteger(source)}, {"target", jsonInteger(target)}});
  }
  return llvm::json::Object{{"version", jsonInteger(capability.version)},
                            {"resource_pairs", std::move(resourcePairs)}};
}

llvm::json::Object jsonTargetCapabilities(
    const pto::CanonicalSyncTargetCapabilities &capabilities) {
  llvm::json::Array evidence;
  for (pto::CanonicalSyncTargetEvidence entry : capabilities.evidence) {
    evidence.push_back(targetEvidenceName(entry));
  }
  llvm::json::Array compilerUsableEventIds;
  for (unsigned eventId : capabilities.compilerUsableEventIds) {
    compilerUsableEventIds.push_back(jsonInteger(eventId));
  }
  llvm::json::Object result{
      {"profile", targetProfileName(capabilities.profile)},
      {"core_domain", coreDomainName(capabilities.coreDomain)},
      {"sync_spec_version", syncSpecVersionName(capabilities.syncSpecVersion)},
      {"evidence", std::move(evidence)},
      {"hardware_event_completion",
       jsonDirectedResourceCapability(capabilities.hardwareEventCompletion)},
      {"direct_event_completion",
       jsonDirectedResourceCapability(capabilities.directEventCompletion)},
      {"legal_pipe_barriers",
       jsonResourceCapability(capabilities.legalPipeBarriers)},
      {"compiler_usable_event_ids", std::move(compilerUsableEventIds)},
      {"same_resource_completion_ordering",
       jsonResourceCapability(capabilities.sameResourceCompletionOrdering)},
      {"targeted_barrier_drains_source_prefix",
       jsonResourceCapability(capabilities.targetedBarrierDrainsSourcePrefix)},
      {"cross_resource_targeted_barrier_completion",
       jsonDirectedResourceCapability(
           capabilities.crossResourceTargetedBarrierCompletion)},
      {"mte1_l0_ready_set_completes_prefix",
       jsonInteger(capabilities.mte1L0ReadySetCompletesPrefix.version)},
      {"m_l0_alternative_join_set_completes",
       jsonInteger(capabilities.mL0AlternativeJoinSetCompletes.version)},
      {"mte1_scope_exit_set_completes_prefix",
       jsonInteger(capabilities.mte1ScopeExitSetCompletesPrefix.version)},
      {"m_to_fix_accumulator_boundary_completes",
       jsonInteger(capabilities.mToFixAccumulatorBoundaryCompletes.version)},
      {"intrinsic_mmad_accumulator_ordering",
       jsonInteger(capabilities.intrinsicMmadAccumulatorOrdering.version)},
      {"cross_pipe_accumulator_read_read_hazard",
       jsonInteger(capabilities.crossPipeAccumulatorReadReadHazard.version)}};
  if (capabilities.targetCompletionResources) {
    result["target_completion_resources"] = llvm::json::Object{
        {"mte1", jsonInteger(capabilities.targetCompletionResources->mte1)},
        {"matrix", jsonInteger(capabilities.targetCompletionResources->matrix)},
        {"fix", jsonInteger(capabilities.targetCompletionResources->fix)}};
  }
  return result;
}

llvm::json::Array jsonGuard(const pto::SyncCoverGuard &guard) {
  llvm::json::Array result;
  for (const pto::SyncCoverGuardLiteral &literal : guard.literals) {
    result.push_back(
        llvm::json::Object{{"control", jsonInteger(literal.control)},
                           {"alternative", jsonInteger(literal.alternative)}});
  }
  return result;
}

llvm::json::Array jsonStorageLifecycleComponents(
    ArrayRef<pto::CanonicalSyncStorageLifecycleComponentReport> components) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageLifecycleComponentReport &component :
       components) {
    result.push_back(llvm::json::Object{
        {"component", jsonInteger(component.component)},
        {"family", jsonInteger(component.family)},
        {"owning_scope", jsonInteger(component.owningScope)},
        {"slots", jsonInteger(component.slots)},
        {"epochs", jsonInteger(component.epochs)},
        {"edges", jsonInteger(component.edges)},
        {"demands", jsonInteger(component.demands)},
        {"sccs", jsonInteger(component.sccs)},
        {"cyclic_sccs", jsonInteger(component.cyclicSccs)},
        {"ready_release_sccs", jsonInteger(component.readyReleaseSccs)},
        {"scc_transfers", jsonInteger(component.sccTransfers)},
        {"transition_classes", jsonInteger(component.transitionClasses)}});
  }
  return result;
}

llvm::json::Array jsonStorageLifecycleTransitions(
    ArrayRef<pto::CanonicalSyncStorageLifecycleTransitionReport>
        transitions) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageLifecycleTransitionReport &transition :
       transitions) {
    result.push_back(llvm::json::Object{
        {"component", jsonInteger(transition.component)},
        {"transition", jsonInteger(transition.transition)},
        {"kinds", jsonInteger(transition.kinds)},
        {"source_resource", jsonInteger(transition.sourceResource)},
        {"target_resource", jsonInteger(transition.targetResource)},
        {"scope", jsonInteger(transition.scope)},
        {"distance", jsonInteger(transition.distance)},
        {"source_guard", jsonGuard(transition.sourceGuard)},
        {"target_guard", jsonGuard(transition.targetGuard)},
        {"edges", jsonInteger(transition.edges)}});
  }
  return result;
}

llvm::json::Array jsonStorageProtocolSeeds(
    ArrayRef<pto::CanonicalSyncStorageProtocolSeedReport> seeds) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolSeedReport &seed : seeds) {
    result.push_back(llvm::json::Object{
        {"seed", jsonInteger(seed.seed)},
        {"family", jsonInteger(seed.family)},
        {"owning_scope", jsonInteger(seed.owningScope)},
        {"components", jsonInteger(seed.components)},
        {"slots", jsonInteger(seed.slots)},
        {"ready_release_sccs", jsonInteger(seed.readyReleaseSccs)},
        {"demands", jsonInteger(seed.demands)},
        {"kinds", jsonInteger(seed.kinds)},
        {"maximum_distance", jsonInteger(seed.maximumDistance)}});
  }
  return result;
}

StringRef
storageProtocolBehaviorName(pto::SyncCoverStorageProtocolBehavior behavior) {
  using Behavior = pto::SyncCoverStorageProtocolBehavior;
  switch (behavior) {
  case Behavior::StableRoundTrip:
    return "stable-round-trip";
  case Behavior::PhaseRotatingRoundTrip:
    return "phase-rotating-round-trip";
  }
  return "unknown";
}

llvm::json::Array jsonStorageProtocolGroups(
    ArrayRef<pto::CanonicalSyncStorageProtocolGroupReport> groups) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolGroupReport &group : groups) {
    llvm::json::Array seeds;
    for (pto::SyncCoverStorageProtocolSeedId seed : group.seeds) {
      seeds.push_back(jsonInteger(seed));
    }
    result.push_back(llvm::json::Object{
        {"group", jsonInteger(group.group)},
        {"owning_scope", jsonInteger(group.owningScope)},
        {"behavior", storageProtocolBehaviorName(group.behavior)},
        {"ready_source_resource", jsonInteger(group.readySourceResource)},
        {"ready_target_resource", jsonInteger(group.readyTargetResource)},
        {"behavior_signature", jsonUnsignedValues(group.behaviorSignature)},
        {"seeds", std::move(seeds)},
        {"periodic_controls", jsonInteger(group.periodicControls)},
        {"demands", jsonInteger(group.demands)},
        {"maximum_distance", jsonInteger(group.maximumDistance)}});
  }
  return result;
}

llvm::json::Array jsonStorageProtocolAutomata(
    ArrayRef<pto::CanonicalSyncStorageProtocolAutomatonReport> automata) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolAutomatonReport &automaton :
       automata) {
    result.push_back(llvm::json::Object{
        {"automaton", jsonInteger(automaton.automaton)},
        {"group", jsonInteger(automaton.group)},
        {"owning_scope", jsonInteger(automaton.owningScope)},
        {"states", jsonInteger(automaton.states)},
        {"transfers", jsonInteger(automaton.transfers)},
        {"state_pair_incidences", jsonInteger(automaton.statePairIncidences)},
        {"maximum_distance", jsonInteger(automaton.maximumDistance)}});
  }
  return result;
}

llvm::json::Array jsonStorageProtocolCutPlans(
    ArrayRef<pto::CanonicalSyncStorageProtocolCutPlanReport> plans) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolCutPlanReport &plan : plans) {
    result.push_back(llvm::json::Object{
        {"plan", jsonInteger(plan.plan)},
        {"automaton", jsonInteger(plan.automaton)},
        {"group", jsonInteger(plan.group)},
        {"owning_scope", jsonInteger(plan.owningScope)},
        {"lanes", jsonInteger(plan.lanes)},
        {"direct_ready_transfers", jsonInteger(plan.directReadyTransfers)},
        {"recurrence_release_transfers",
         jsonInteger(plan.recurrenceReleaseTransfers)},
        {"ready_rectangles", jsonInteger(plan.readyRectangles)}});
  }
  return result;
}

llvm::json::Array jsonStorageProtocolFrontierPlans(
    ArrayRef<pto::CanonicalSyncStorageProtocolFrontierPlanReport> plans) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolFrontierPlanReport &plan :
       plans) {
    result.push_back(llvm::json::Object{
        {"plan", jsonInteger(plan.plan)},
        {"automaton", jsonInteger(plan.automaton)},
        {"group", jsonInteger(plan.group)},
        {"owning_scope", jsonInteger(plan.owningScope)},
        {"lanes", jsonInteger(plan.lanes)},
        {"frontiers", jsonInteger(plan.frontiers)},
        {"ready_frontiers", jsonInteger(plan.readyFrontiers)},
        {"reuse_frontiers", jsonInteger(plan.reuseFrontiers)},
        {"direct_frontiers", jsonInteger(plan.directFrontiers)},
        {"completion_cut_fact_frontiers",
         jsonInteger(plan.completionCutFactFrontiers)},
        {"certificate_frontiers", jsonInteger(plan.certificateFrontiers)},
        {"same_resource_recurrence_reuses",
         jsonInteger(plan.sameResourceRecurrenceReuses)}});
  }
  return result;
}

llvm::json::Array jsonStorageProtocolRectangleGrounding(
    ArrayRef<pto::CanonicalSyncStorageProtocolRectangleGroundingReport>
        details) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncStorageProtocolRectangleGroundingReport &detail :
       details) {
    result.push_back(llvm::json::Object{
        {"rectangle", jsonInteger(detail.rectangle)},
        {"automaton", jsonInteger(detail.automaton)},
        {"frontier_count", jsonInteger(detail.frontierCount)},
        {"admitted_demands", jsonInteger(detail.admittedDemands)},
        {"coverage_rows", jsonInteger(detail.coverageRows)}});
  }
  return result;
}

llvm::json::Array jsonSyntheticStorageRectangleGrounding(
    ArrayRef<pto::CanonicalSyncSyntheticRectangleGroundingReport> details) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncSyntheticRectangleGroundingReport &detail :
       details) {
    result.push_back(llvm::json::Object{
        {"rectangle", jsonInteger(detail.rectangle)},
        {"completion_cut", jsonInteger(detail.completionCut)},
        {"acquisition_cut", jsonInteger(detail.acquisitionCut)},
        {"coverage_rows", jsonInteger(detail.coverageRows)}});
  }
  return result;
}

StringRef gmAliasPolicyName(pto::CanonicalSyncGmAliasPolicy policy) {
  switch (policy) {
  case pto::CanonicalSyncGmAliasPolicy::MayAlias:
    return "may-alias";
  case pto::CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias:
    return "distinct-arguments-noalias";
  case pto::CanonicalSyncGmAliasPolicy::AllAccessesNoAlias:
    return "all-accesses-noalias";
  }
  return "unknown";
}

StringRef mechanismOriginName(pto::CanonicalSyncMechanismOrigin origin) {
  using Origin = pto::CanonicalSyncMechanismOrigin;
  switch (origin) {
  case Origin::Unclassified:
    return "unclassified";
  case Origin::DirectTargetedBarrier:
    return "direct-targeted-barrier";
  case Origin::DirectDistanceZeroEvent:
    return "direct-distance-zero-event";
  case Origin::DirectForwardRecurrenceEvent:
    return "direct-forward-recurrence-event";
  case Origin::DirectReleaseRecurrenceProtocol:
    return "direct-release-recurrence-protocol";
  case Origin::DirectBalancedTargetFenceEvent:
    return "direct-balanced-target-fence-event";
  case Origin::StorageCutEvent:
    return "storage-cut-event";
  case Origin::CompletionFrontierEvent:
    return "completion-frontier-event";
  case Origin::TargetCompletionCertificateEvent:
    return "target-completion-certificate-event";
  case Origin::TargetLocalFenceEvent:
    return "target-local-fence-event";
  case Origin::SourceLocalCompletionEvent:
    return "source-local-completion-event";
  case Origin::SourceLocalPipeDrain:
    return "source-local-pipe-drain";
  case Origin::SourcePrefixPipeDrain:
    return "source-prefix-pipe-drain";
  case Origin::LoopCarryPipeDrain:
    return "loop-carry-pipe-drain";
  case Origin::LoopBoundarySourcePrefixProtocol:
    return "loop-boundary-source-prefix-protocol";
  case Origin::BasicOwnershipL0OperandProtocol:
    return "basic-ownership-l0-operand-protocol";
  case Origin::BasicOwnershipStableL1Protocol:
    return "basic-ownership-stable-l1-protocol";
  case Origin::BasicOwnershipAlternatingL1Protocol:
    return "basic-ownership-alternating-l1-protocol";
  case Origin::BasicOwnershipAccumulatorProtocol:
    return "basic-ownership-accumulator-protocol";
  case Origin::BoundaryGuardedAccumulatorProtocol:
    return "boundary-guarded-accumulator-protocol";
  case Origin::HierarchicalStableL1Protocol:
    return "hierarchical-stable-l1-protocol";
  case Origin::HierarchicalAlternatingL1Protocol:
    return "hierarchical-alternating-l1-protocol";
  case Origin::CompositeOwnershipProtocol:
    return "composite-ownership-protocol";
  case Origin::RepairTargetLocalPipeDrain:
    return "repair-target-local-pipe-drain";
  case Origin::RepairSourceLocalPipeDrain:
    return "repair-source-local-pipe-drain";
  case Origin::RepairSourcePrefixPipeDrain:
    return "repair-source-prefix-pipe-drain";
  case Origin::RepairFrontierBarrier:
    return "repair-frontier-barrier";
  case Origin::RepairFrontierEvent:
    return "repair-frontier-event";
  case Origin::LocalizedPipeAll:
    return "localized-pipe-all";
  case Origin::Count:
    break;
  }
  return "unknown";
}

StringRef mechanismKindName(pto::CanonicalSyncMechanismKind kind) {
  switch (kind) {
  case pto::CanonicalSyncMechanismKind::Event:
    return "event";
  case pto::CanonicalSyncMechanismKind::Barrier:
    return "barrier";
  case pto::CanonicalSyncMechanismKind::Protocol:
    return "protocol";
  }
  return "unknown";
}

llvm::json::Array
jsonMechanismFamilies(pto::CanonicalSyncMechanismFamilyMask mask) {
  llvm::json::Array families;
  for (std::size_t index = 0; index < pto::kCanonicalSyncMechanismFamilyCount;
       ++index) {
    const auto family = static_cast<pto::CanonicalSyncMechanismFamily>(index);
    if (pto::canonicalSyncMechanismFamilyEnabled(mask, family)) {
      families.push_back(mechanismFamilyName(family));
    }
  }
  return families;
}

llvm::json::Array
jsonMechanismOrigins(pto::CanonicalSyncMechanismOriginMask mask) {
  llvm::json::Array origins;
  for (std::size_t index = 0; index < pto::kCanonicalSyncMechanismOriginCount;
       ++index) {
    const auto origin = static_cast<pto::CanonicalSyncMechanismOrigin>(index);
    if ((mask & pto::canonicalSyncMechanismOriginBit(origin)) != 0) {
      origins.push_back(mechanismOriginName(origin));
    }
  }
  return origins;
}

llvm::json::Object jsonMechanismOriginCounts(
    const std::array<std::size_t, pto::kCanonicalSyncMechanismOriginCount>
        &counts) {
  llvm::json::Object result;
  for (std::size_t index = 0; index < counts.size(); ++index) {
    result[mechanismOriginName(static_cast<pto::CanonicalSyncMechanismOrigin>(
        index))] = jsonInteger(counts[index]);
  }
  return result;
}

llvm::json::Array jsonSelectedMechanisms(
    ArrayRef<pto::CanonicalSyncSelectedMechanismReport> mechanisms) {
  llvm::json::Array result;
  for (const pto::CanonicalSyncSelectedMechanismReport &mechanism :
       mechanisms) {
    result.push_back(llvm::json::Object{
        {"mechanism", jsonInteger(mechanism.mechanism)},
        {"kind", mechanismKindName(mechanism.kind)},
        {"origin_mask", jsonInteger(mechanism.originMask)},
        {"origins", jsonMechanismOrigins(mechanism.originMask)},
        {"supplies", jsonInteger(mechanism.supplies)},
        {"grounded_coverage_rows", jsonInteger(mechanism.groundedCoverageRows)},
        {"event_uses", jsonInteger(mechanism.eventUses)},
        {"actions", jsonInteger(mechanism.actions)},
        {"event_sets", jsonInteger(mechanism.eventSets)},
        {"event_waits", jsonInteger(mechanism.eventWaits)},
        {"targeted_barriers", jsonInteger(mechanism.targetedBarriers)},
        {"pipe_all_barriers", jsonInteger(mechanism.pipeAllBarriers)},
        {"maximum_recurrence_distance",
         jsonInteger(mechanism.maximumRecurrenceDistance)}});
  }
  return result;
}

llvm::json::Object
jsonAllocation(const pto::CanonicalSyncResourceAllocation &allocation) {
  llvm::json::Array domains;
  for (const pto::CanonicalSyncDomainAllocation &domain : allocation.domains) {
    llvm::json::Array uses;
    llvm::json::Array liveMechanisms;
    for (pto::CanonicalSyncMechanismId mechanism : domain.liveMechanisms) {
      liveMechanisms.push_back(jsonInteger(mechanism));
    }
    for (const pto::CanonicalSyncEventAllocation &use : domain.uses) {
      llvm::json::Array ids;
      for (unsigned id : use.ids) {
        ids.push_back(static_cast<std::int64_t>(id));
      }
      uses.push_back(
          llvm::json::Object{{"mechanism", jsonInteger(use.mechanism)},
                             {"event_use", jsonInteger(use.eventUse)},
                             {"ids", std::move(ids)}});
    }
    llvm::json::Object item{{"domain", jsonInteger(domain.domain)},
                            {"required", jsonInteger(domain.required)},
                            {"available", jsonInteger(domain.available)},
                            {"live_mechanisms", std::move(liveMechanisms)},
                            {"uses", std::move(uses)}};
    if (domain.maximumPressurePoint) {
      item["maximum_pressure_point"] =
          jsonInteger(*domain.maximumPressurePoint);
    }
    domains.push_back(std::move(item));
  }
  return llvm::json::Object{{"valid", allocation.valid},
                            {"feasible", allocation.feasible},
                            {"domains", std::move(domains)}};
}

llvm::json::Object
jsonReport(const pto::CanonicalSyncComparisonReport &report) {
  llvm::json::Array strategies;
  for (const pto::CanonicalSyncStrategyReport &strategy : report.strategies) {
    strategies.push_back(llvm::json::Object{
        {"strategy", strategyName(strategy.strategy)},
        {"error", jsonInteger(static_cast<std::uint8_t>(strategy.error))},
        {"verification_error",
         jsonInteger(static_cast<std::uint8_t>(strategy.verificationError))},
        {"precise_error",
         jsonInteger(static_cast<std::uint8_t>(strategy.preciseError))},
        {"verified", strategy.verified},
        {"used_localized_pipe_all", strategy.usedLocalizedPipeAll},
        {"repair_frontier_truncated", strategy.repairFrontierTruncated},
        {"repair_budget_exhausted", strategy.repairBudgetExhausted},
        {"backstop_deletion_truncated", strategy.backstopDeletionTruncated},
        {"repair_rounds", jsonInteger(strategy.repairRounds)},
        {"repair_catalog_rebuilds",
         jsonInteger(strategy.repairCatalogRebuilds)},
        {"first_repair_catalog_rebuild_work_units",
         jsonInteger(strategy.firstRepairCatalogRebuildWorkUnits)},
        {"repair_trials", jsonInteger(strategy.repairTrials)},
        {"repair_work_units", jsonInteger(strategy.repairWorkUnits)},
        {"backstop_deletion_trials",
         jsonInteger(strategy.backstopDeletionTrials)},
        {"backstop_deletion_work_units",
         jsonInteger(strategy.backstopDeletionWorkUnits)},
        {"selected_events", jsonInteger(strategy.selectedEvents)},
        {"selected_targeted_barriers",
         jsonInteger(strategy.selectedTargetedBarriers)},
        {"selected_pipe_all_barriers",
         jsonInteger(strategy.selectedPipeAllBarriers)},
        {"selected_mechanisms_by_origin",
         jsonMechanismOriginCounts(strategy.selectedMechanismsByOrigin)},
        {"active_direct_pairs", jsonInteger(strategy.activeDirectPairs)},
        {"active_direct_pair_extra_coverage",
         jsonInteger(strategy.activeDirectPairExtraCoverage)},
        {"selected_mechanism_details_truncated",
         strategy.selectedMechanismDetailsTruncated},
        {"selected_mechanisms",
         jsonSelectedMechanisms(strategy.selectedMechanisms)},
        {"emitted_event_sets", jsonInteger(strategy.emittedEventSets)},
        {"emitted_event_waits", jsonInteger(strategy.emittedEventWaits)},
        {"emitted_targeted_barriers",
         jsonInteger(strategy.emittedTargetedBarriers)},
        {"emitted_zero_distance_targeted_barriers",
         jsonInteger(strategy.emittedZeroDistanceTargetedBarriers)},
        {"emitted_recurrence_targeted_barriers",
         jsonInteger(strategy.emittedRecurrenceTargetedBarriers)},
        {"emitted_zero_only_targeted_barriers",
         jsonInteger(strategy.emittedZeroOnlyTargetedBarriers)},
        {"emitted_recurrence_only_targeted_barriers",
         jsonInteger(strategy.emittedRecurrenceOnlyTargetedBarriers)},
        {"emitted_mixed_distance_targeted_barriers",
         jsonInteger(strategy.emittedMixedDistanceTargetedBarriers)},
        {"emitted_target_local_pipe_drain_barriers",
         jsonInteger(strategy.emittedTargetLocalPipeDrainBarriers)},
        {"emitted_loop_carry_pipe_drain_barriers",
         jsonInteger(strategy.emittedLoopCarryPipeDrainBarriers)},
        {"emitted_source_local_pipe_drain_barriers",
         jsonInteger(strategy.emittedSourceLocalPipeDrainBarriers)},
        {"emitted_source_prefix_pipe_drain_barriers",
         jsonInteger(strategy.emittedSourcePrefixPipeDrainBarriers)},
        {"emitted_pipe_all_barriers",
         jsonInteger(strategy.emittedPipeAllBarriers)},
        {"predicted_sync_instructions",
         jsonInteger(strategy.predictedSyncInstructions)},
        {"barrier_action_profile",
         jsonUnsignedValues(strategy.cost.barrierActionProfile)},
        {"event_action_profile",
         jsonUnsignedValues(strategy.cost.eventActionProfile)},
        {"action_profile", jsonUnsignedValues(strategy.cost.actionProfile)},
        {"serialization_breadth",
         jsonInteger(strategy.cost.serializationBreadth)},
        {"event_lifetime_area", jsonInteger(strategy.cost.eventLifetimeArea)},
        {"mechanisms", jsonInteger(strategy.cost.mechanismCount)},
        {"pattern_evaluations",
         jsonInteger(strategy.search.patternEvaluations)},
        {"deletion_evaluations",
         jsonInteger(strategy.search.deletionEvaluations)},
        {"work_units", jsonInteger(strategy.search.workUnits)},
        {"arithmetic_overflow", strategy.search.arithmeticOverflow},
        {"verification_work_units",
         jsonInteger(strategy.verificationWorkUnits)},
        {"selection_time_ns", jsonInteger(strategy.selectionNanoseconds)},
        {"repair_time_ns", jsonInteger(strategy.repairNanoseconds)},
        {"verification_time_ns", jsonInteger(strategy.verificationNanoseconds)},
        {"plan_signature", jsonSignature(strategy.planSignature)},
        {"event_allocation", jsonAllocation(strategy.allocation)},
        {"precise_pattern_evaluations",
         jsonInteger(strategy.preciseSearch.patternEvaluations)},
        {"precise_deletion_evaluations",
         jsonInteger(strategy.preciseSearch.deletionEvaluations)},
        {"precise_work_units", jsonInteger(strategy.preciseSearch.workUnits)},
        {"precise_arithmetic_overflow",
         strategy.preciseSearch.arithmeticOverflow},
        {"precise_event_allocation",
         jsonAllocation(strategy.preciseAllocation)}});
  }
  return llvm::json::Object{
      {"schema", "ptoas.canonical_sync.v1"},
      {"function", report.function},
      {"target_capabilities",
       jsonTargetCapabilities(report.targetCapabilities)},
      {"selection_objective", objectiveName(report.selectionObjective)},
      {"catalog_mode", catalogModeName(report.catalogMode)},
      {"enabled_mechanism_family_mask",
       jsonInteger(report.enabledMechanismFamilies)},
      {"enabled_mechanism_families",
       jsonMechanismFamilies(report.enabledMechanismFamilies)},
      {"direct_pairs_enabled", report.directPairsEnabled},
      {"conflict_core_repair_enabled", report.conflictCoreRepairEnabled},
      {"gm_alias_policy", gmAliasPolicyName(report.gmAliasPolicy)},
      {"graph_nodes", jsonInteger(report.graphNodes)},
      {"graph_edges", jsonInteger(report.graphEdges)},
      {"certified_completion_frontiers",
       jsonInteger(report.certifiedCompletionFrontiers)},
      {"ownership_discovery_inspections",
       jsonInteger(report.ownershipDiscoveryInspections)},
      {"ownership_certificates_by_kind",
       llvm::json::Object{
           {"l0_operand",
            jsonInteger(
                report.ownershipCertificatesByKind[static_cast<std::size_t>(
                    pto::SyncCoverBasicOwnershipKind::L0Operand)])},
           {"l1_tile",
            jsonInteger(
                report.ownershipCertificatesByKind[static_cast<std::size_t>(
                    pto::SyncCoverBasicOwnershipKind::L1Tile)])},
           {"l0_accumulator",
            jsonInteger(
                report.ownershipCertificatesByKind[static_cast<std::size_t>(
                    pto::SyncCoverBasicOwnershipKind::L0Accumulator)])}}},
      {"ownership_slots", jsonInteger(report.ownershipSlots)},
      {"ownership_paths", jsonInteger(report.ownershipPaths)},
      {"ownership_uses", jsonInteger(report.ownershipUses)},
      {"ownership_node_references",
       jsonInteger(report.ownershipNodeReferences)},
      {"ownership_access_incidences",
       jsonInteger(report.ownershipAccessIncidences)},
      {"ownership_discovery_truncated", report.ownershipDiscoveryTruncated},
      {"storage_lifecycle_analysis_enabled",
       report.storageLifecycleAnalysisEnabled},
      {"storage_lifecycle_work_units",
       jsonInteger(report.storageLifecycleWorkUnits)},
      {"storage_lifecycle_eligible_witnesses",
       jsonInteger(report.storageLifecycleEligibleWitnesses)},
      {"storage_lifecycle_ineligible_witnesses",
       jsonInteger(report.storageLifecycleIneligibleWitnesses)},
      {"storage_lifecycle_components",
       jsonInteger(report.storageLifecycleComponents)},
      {"storage_lifecycle_slots", jsonInteger(report.storageLifecycleSlots)},
      {"storage_lifecycle_epochs", jsonInteger(report.storageLifecycleEpochs)},
      {"storage_lifecycle_edges", jsonInteger(report.storageLifecycleEdges)},
      {"storage_lifecycle_demand_incidences",
       jsonInteger(report.storageLifecycleDemandIncidences)},
      {"storage_lifecycle_sccs", jsonInteger(report.storageLifecycleSccs)},
      {"storage_lifecycle_cyclic_sccs",
       jsonInteger(report.storageLifecycleCyclicSccs)},
      {"storage_lifecycle_ready_release_sccs",
       jsonInteger(report.storageLifecycleReadyReleaseSccs)},
      {"storage_lifecycle_scc_transfers",
       jsonInteger(report.storageLifecycleSccTransfers)},
      {"storage_lifecycle_maximum_scc_epochs",
       jsonInteger(report.storageLifecycleMaximumSccEpochs)},
      {"storage_lifecycle_transition_classes",
       jsonInteger(report.storageLifecycleTransitionClasses)},
      {"storage_lifecycle_transition_guard_literals",
       jsonInteger(report.storageLifecycleTransitionGuardLiterals)},
      {"storage_lifecycle_maximum_transition_class_edges",
       jsonInteger(report.storageLifecycleMaximumTransitionClassEdges)},
      {"storage_lifecycle_details_truncated",
       report.storageLifecycleDetailsTruncated},
      {"storage_lifecycle_component_details",
       jsonStorageLifecycleComponents(report.storageLifecycleComponentDetails)},
      {"storage_lifecycle_transition_details",
       jsonStorageLifecycleTransitions(
           report.storageLifecycleTransitionDetails)},
      {"storage_lifecycle_truncated", report.storageLifecycleTruncated},
      {"storage_protocol_seed_analysis_enabled",
       report.storageProtocolSeedAnalysisEnabled},
      {"storage_protocol_seed_work_units",
       jsonInteger(report.storageProtocolSeedWorkUnits)},
      {"storage_protocol_seeds", jsonInteger(report.storageProtocolSeeds)},
      {"storage_protocol_ready_release_seeds",
       jsonInteger(report.storageProtocolReadyReleaseSeeds)},
      {"storage_protocol_component_incidences",
       jsonInteger(report.storageProtocolComponentIncidences)},
      {"storage_protocol_slot_incidences",
       jsonInteger(report.storageProtocolSlotIncidences)},
      {"storage_protocol_scc_incidences",
       jsonInteger(report.storageProtocolSccIncidences)},
      {"storage_protocol_demand_incidences",
       jsonInteger(report.storageProtocolDemandIncidences)},
      {"storage_protocol_maximum_seed_components",
       jsonInteger(report.storageProtocolMaximumSeedComponents)},
      {"storage_protocol_maximum_seed_slots",
       jsonInteger(report.storageProtocolMaximumSeedSlots)},
      {"storage_protocol_maximum_seed_sccs",
       jsonInteger(report.storageProtocolMaximumSeedSccs)},
      {"storage_protocol_seed_details",
       jsonStorageProtocolSeeds(report.storageProtocolSeedDetails)},
      {"storage_protocol_seed_details_truncated",
       report.storageProtocolSeedDetailsTruncated},
      {"storage_protocol_seed_truncated", report.storageProtocolSeedTruncated},
      {"storage_protocol_group_analysis_enabled",
       report.storageProtocolGroupAnalysisEnabled},
      {"storage_protocol_group_work_units",
       jsonInteger(report.storageProtocolGroupWorkUnits)},
      {"storage_protocol_eligible_seeds",
       jsonInteger(report.storageProtocolEligibleSeeds)},
      {"storage_protocol_ineligible_seeds",
       jsonInteger(report.storageProtocolIneligibleSeeds)},
      {"storage_protocol_stable_seeds",
       jsonInteger(report.storageProtocolStableSeeds)},
      {"storage_protocol_phase_rotating_seeds",
       jsonInteger(report.storageProtocolPhaseRotatingSeeds)},
      {"storage_protocol_groups", jsonInteger(report.storageProtocolGroups)},
      {"storage_protocol_group_seed_incidences",
       jsonInteger(report.storageProtocolGroupSeedIncidences)},
      {"storage_protocol_group_control_incidences",
       jsonInteger(report.storageProtocolGroupControlIncidences)},
      {"storage_protocol_group_demand_incidences",
       jsonInteger(report.storageProtocolGroupDemandIncidences)},
      {"storage_protocol_group_slot_incidences",
       jsonInteger(report.storageProtocolGroupSlotIncidences)},
      {"storage_protocol_group_joint_state_incidences",
       jsonInteger(report.storageProtocolGroupJointStateIncidences)},
      {"storage_protocol_maximum_group_seeds",
       jsonInteger(report.storageProtocolMaximumGroupSeeds)},
      {"storage_protocol_group_details",
       jsonStorageProtocolGroups(report.storageProtocolGroupDetails)},
      {"storage_protocol_group_details_truncated",
       report.storageProtocolGroupDetailsTruncated},
      {"storage_protocol_group_truncated",
       report.storageProtocolGroupTruncated},
      {"storage_protocol_automaton_analysis_enabled",
       report.storageProtocolAutomatonAnalysisEnabled},
      {"storage_protocol_automaton_work_units",
       jsonInteger(report.storageProtocolAutomatonWorkUnits)},
      {"storage_protocol_automaton_eligible_groups",
       jsonInteger(report.storageProtocolAutomatonEligibleGroups)},
      {"storage_protocol_automaton_ineligible_groups",
       jsonInteger(report.storageProtocolAutomatonIneligibleGroups)},
      {"storage_protocol_automaton_lane_limited_groups",
       jsonInteger(report.storageProtocolAutomatonLaneLimitedGroups)},
      {"storage_protocol_automaton_scope_rejected_groups",
       jsonInteger(report.storageProtocolAutomatonScopeRejectedGroups)},
      {"storage_protocol_automaton_membership_rejected_groups",
       jsonInteger(report.storageProtocolAutomatonMembershipRejectedGroups)},
      {"storage_protocol_automaton_direction_rejected_groups",
       jsonInteger(report.storageProtocolAutomatonDirectionRejectedGroups)},
      {"storage_protocol_automaton_unreachable_transfer_groups",
       jsonInteger(report.storageProtocolAutomatonUnreachableTransferGroups)},
      {"storage_protocol_automaton_unreachable_ready_transfer_groups",
       jsonInteger(
           report.storageProtocolAutomatonUnreachableReadyTransferGroups)},
      {"storage_protocol_automaton_unreachable_release_transfer_groups",
       jsonInteger(
           report.storageProtocolAutomatonUnreachableReleaseTransferGroups)},
      {"storage_protocol_automaton_unreachable_exclusion_transfer_groups",
       jsonInteger(
           report.storageProtocolAutomatonUnreachableExclusionTransferGroups)},
      {"storage_protocol_automaton_demand_set_mismatch_groups",
       jsonInteger(report.storageProtocolAutomatonDemandSetMismatchGroups)},
      {"storage_protocol_automaton_distance_mismatch_groups",
       jsonInteger(report.storageProtocolAutomatonDistanceMismatchGroups)},
      {"storage_protocol_automata",
       jsonInteger(report.storageProtocolAutomata)},
      {"storage_protocol_automaton_states",
       jsonInteger(report.storageProtocolAutomatonStates)},
      {"storage_protocol_automaton_transfers",
       jsonInteger(report.storageProtocolAutomatonTransfers)},
      {"storage_protocol_automaton_state_pair_incidences",
       jsonInteger(report.storageProtocolAutomatonStatePairIncidences)},
      {"storage_protocol_maximum_automaton_transfers",
       jsonInteger(report.storageProtocolMaximumAutomatonTransfers)},
      {"storage_protocol_maximum_transfer_state_pairs",
       jsonInteger(report.storageProtocolMaximumTransferStatePairs)},
      {"storage_protocol_automaton_details",
       jsonStorageProtocolAutomata(report.storageProtocolAutomatonDetails)},
      {"storage_protocol_automaton_details_truncated",
       report.storageProtocolAutomatonDetailsTruncated},
      {"storage_protocol_automaton_truncated",
       report.storageProtocolAutomatonTruncated},
      {"storage_protocol_cut_plan_analysis_enabled",
       report.storageProtocolCutPlanAnalysisEnabled},
      {"storage_protocol_cut_plan_work_units",
       jsonInteger(report.storageProtocolCutPlanWorkUnits)},
      {"storage_protocol_cut_plan_eligible_automata",
       jsonInteger(report.storageProtocolCutPlanEligibleAutomata)},
      {"storage_protocol_cut_plan_ineligible_automata",
       jsonInteger(report.storageProtocolCutPlanIneligibleAutomata)},
      {"storage_protocol_cut_plan_missing_ready_cut_automata",
       jsonInteger(report.storageProtocolCutPlanMissingReadyCutAutomata)},
      {"storage_protocol_cut_plan_missing_release_automata",
       jsonInteger(report.storageProtocolCutPlanMissingReleaseAutomata)},
      {"storage_protocol_cut_plans",
       jsonInteger(report.storageProtocolCutPlans)},
      {"storage_protocol_cut_plan_transfer_inspections",
       jsonInteger(report.storageProtocolCutPlanTransferInspections)},
      {"storage_protocol_cut_plan_direct_ready_transfers",
       jsonInteger(report.storageProtocolCutPlanDirectReadyTransfers)},
      {"storage_protocol_cut_plan_recurrence_release_transfers",
       jsonInteger(report.storageProtocolCutPlanRecurrenceReleaseTransfers)},
      {"storage_protocol_cut_plan_ready_rectangle_incidences",
       jsonInteger(report.storageProtocolCutPlanReadyRectangleIncidences)},
      {"storage_protocol_cut_plan_maximum_ready_rectangles",
       jsonInteger(report.storageProtocolCutPlanMaximumReadyRectangles)},
      {"storage_protocol_cut_plan_details",
       jsonStorageProtocolCutPlans(report.storageProtocolCutPlanDetails)},
      {"storage_protocol_cut_plan_details_truncated",
       report.storageProtocolCutPlanDetailsTruncated},
      {"storage_protocol_cut_plan_truncated",
       report.storageProtocolCutPlanTruncated},
      {"storage_protocol_frontier_analysis_enabled",
       report.storageProtocolFrontierAnalysisEnabled},
      {"storage_protocol_frontier_work_units",
       jsonInteger(report.storageProtocolFrontierWorkUnits)},
      {"storage_protocol_frontier_eligible_automata",
       jsonInteger(report.storageProtocolFrontierEligibleAutomata)},
      {"storage_protocol_frontier_ineligible_automata",
       jsonInteger(report.storageProtocolFrontierIneligibleAutomata)},
      {"storage_protocol_frontier_missing_ready_automata",
       jsonInteger(report.storageProtocolFrontierMissingReadyAutomata)},
      {"storage_protocol_frontier_missing_recurrence_reuse_automata",
       jsonInteger(
           report.storageProtocolFrontierMissingRecurrenceReuseAutomata)},
      {"storage_protocol_frontier_missing_completion_automata",
       jsonInteger(report.storageProtocolFrontierMissingCompletionAutomata)},
      {"storage_protocol_frontier_plans",
       jsonInteger(report.storageProtocolFrontierPlans)},
      {"storage_protocol_frontiers",
       jsonInteger(report.storageProtocolFrontiers)},
      {"storage_protocol_ready_frontiers",
       jsonInteger(report.storageProtocolReadyFrontiers)},
      {"storage_protocol_reuse_frontiers",
       jsonInteger(report.storageProtocolReuseFrontiers)},
      {"storage_protocol_direct_frontiers",
       jsonInteger(report.storageProtocolDirectFrontiers)},
      {"storage_protocol_completion_cut_fact_frontiers",
       jsonInteger(report.storageProtocolCompletionCutFactFrontiers)},
      {"storage_protocol_certificate_frontiers",
       jsonInteger(report.storageProtocolCertificateFrontiers)},
      {"storage_protocol_same_resource_recurrence_reuses",
       jsonInteger(report.storageProtocolSameResourceRecurrenceReuses)},
      {"storage_protocol_frontier_certificate_demand_incidences",
       jsonInteger(report.storageProtocolFrontierCertificateDemandIncidences)},
      {"storage_protocol_frontier_completion_cut_fact_demand_incidences",
       jsonInteger(
           report.storageProtocolFrontierCompletionCutFactDemandIncidences)},
      {"storage_protocol_frontier_transfer_inspections",
       jsonInteger(report.storageProtocolFrontierTransferInspections)},
      {"storage_protocol_frontier_state_pair_inspections",
       jsonInteger(report.storageProtocolFrontierStatePairInspections)},
      {"storage_protocol_frontier_plan_incidences",
       jsonInteger(report.storageProtocolFrontierPlanIncidences)},
      {"storage_protocol_maximum_plan_frontiers",
       jsonInteger(report.storageProtocolMaximumPlanFrontiers)},
      {"storage_protocol_frontier_details",
       jsonStorageProtocolFrontierPlans(report.storageProtocolFrontierDetails)},
      {"storage_protocol_frontier_details_truncated",
       report.storageProtocolFrontierDetailsTruncated},
      {"storage_protocol_frontier_truncated",
       report.storageProtocolFrontierTruncated},
      {"storage_protocol_rectangle_analysis_enabled",
       report.storageProtocolRectangleAnalysisEnabled},
      {"storage_protocol_rectangle_work_units",
       jsonInteger(report.storageProtocolRectangleWorkUnits)},
      {"storage_protocol_rectangle_plans",
       jsonInteger(report.storageProtocolRectanglePlans)},
      {"storage_protocol_rectangle_frontier_inspections",
       jsonInteger(report.storageProtocolRectangleFrontierInspections)},
      {"storage_protocol_rectangles",
       jsonInteger(report.storageProtocolRectangles)},
      {"storage_protocol_ready_rectangles",
       jsonInteger(report.storageProtocolReadyRectangles)},
      {"storage_protocol_reuse_rectangles",
       jsonInteger(report.storageProtocolReuseRectangles)},
      {"storage_protocol_merged_rectangles",
       jsonInteger(report.storageProtocolMergedRectangles)},
      {"storage_protocol_rectangle_frontier_incidences",
       jsonInteger(report.storageProtocolRectangleFrontierIncidences)},
      {"storage_protocol_maximum_rectangle_frontiers",
       jsonInteger(report.storageProtocolMaximumRectangleFrontiers)},
      {"storage_protocol_rectangle_truncated",
       report.storageProtocolRectangleTruncated},
      {"storage_protocol_rectangle_grounding_enabled",
       report.storageProtocolRectangleGroundingEnabled},
      {"storage_protocol_rectangle_grounding_work_units",
       jsonInteger(report.storageProtocolRectangleGroundingWorkUnits)},
      {"storage_protocol_rectangle_grounding_evaluated",
       jsonInteger(report.storageProtocolRectangleGroundingEvaluated)},
      {"storage_protocol_rectangle_grounding_batches",
       jsonInteger(report.storageProtocolRectangleGroundingBatches)},
      {"storage_protocol_rectangles_with_coverage",
       jsonInteger(report.storageProtocolRectanglesWithCoverage)},
      {"storage_protocol_rectangles_covering_multiple_rows",
       jsonInteger(report.storageProtocolRectanglesCoveringMultipleRows)},
      {"storage_protocol_rectangle_admitted_demand_incidences",
       jsonInteger(report.storageProtocolRectangleAdmittedDemandIncidences)},
      {"storage_protocol_rectangle_maximum_coverage_rows",
       jsonInteger(report.storageProtocolRectangleMaximumCoverageRows)},
      {"storage_protocol_rectangle_total_coverage_rows",
       jsonInteger(report.storageProtocolRectangleTotalCoverageRows)},
      {"storage_protocol_rectangle_grounding_error",
       jsonInteger(static_cast<unsigned>(
           report.storageProtocolRectangleGroundingError))},
      {"storage_protocol_rectangle_grounding_details",
       jsonStorageProtocolRectangleGrounding(
           report.storageProtocolRectangleGroundingDetails)},
      {"storage_protocol_rectangle_grounding_details_truncated",
       report.storageProtocolRectangleGroundingDetailsTruncated},
      {"storage_protocol_rectangle_grounding_truncated",
       report.storageProtocolRectangleGroundingTruncated},
      {"storage_cut_analysis_enabled", report.storageCutAnalysisEnabled},
      {"storage_cut_work_units", jsonInteger(report.storageCutWorkUnits)},
      {"storage_cut_eligible_edges",
       jsonInteger(report.storageCutEligibleEdges)},
      {"storage_cut_ineligible_edges",
       jsonInteger(report.storageCutIneligibleEdges)},
      {"storage_completion_cuts", jsonInteger(report.storageCompletionCuts)},
      {"storage_acquisition_cuts", jsonInteger(report.storageAcquisitionCuts)},
      {"storage_rectangles", jsonInteger(report.storageRectangles)},
      {"storage_cut_incidences", jsonInteger(report.storageCutIncidences)},
      {"storage_cut_guard_literals",
       jsonInteger(report.storageCutGuardLiterals)},
      {"storage_maximum_rectangle_edges",
       jsonInteger(report.storageMaximumRectangleEdges)},
      {"storage_cut_truncated", report.storageCutTruncated},
      {"storage_rectangle_analysis_enabled",
       report.storageRectangleAnalysisEnabled},
      {"storage_rectangle_work_units",
       jsonInteger(report.storageRectangleWorkUnits)},
      {"storage_rectangle_inspections",
       jsonInteger(report.storageRectangleInspections)},
      {"storage_factored_rectangles",
       jsonInteger(report.storageFactoredRectangles)},
      {"storage_direct_factored_rectangles",
       jsonInteger(report.storageDirectFactoredRectangles)},
      {"storage_synthetic_factored_rectangles",
       jsonInteger(report.storageSyntheticFactoredRectangles)},
      {"storage_rectangle_guard_literals",
       jsonInteger(report.storageRectangleGuardLiterals)},
      {"storage_rectangle_truncated", report.storageRectangleTruncated},
      {"storage_synthetic_rectangle_grounding_enabled",
       report.storageSyntheticRectangleGroundingEnabled},
      {"storage_synthetic_rectangle_grounding_work_units",
       jsonInteger(report.storageSyntheticRectangleGroundingWorkUnits)},
      {"storage_synthetic_rectangle_grounding_evaluated",
       jsonInteger(report.storageSyntheticRectangleGroundingEvaluated)},
      {"storage_synthetic_rectangles_with_coverage",
       jsonInteger(report.storageSyntheticRectanglesWithCoverage)},
      {"storage_synthetic_rectangles_covering_multiple_rows",
       jsonInteger(report.storageSyntheticRectanglesCoveringMultipleRows)},
      {"storage_synthetic_rectangle_maximum_coverage_rows",
       jsonInteger(report.storageSyntheticRectangleMaximumCoverageRows)},
      {"storage_synthetic_rectangle_total_coverage_rows",
       jsonInteger(report.storageSyntheticRectangleTotalCoverageRows)},
      {"storage_synthetic_rectangle_grounding_error",
       jsonInteger(static_cast<unsigned>(
           report.storageSyntheticRectangleGroundingError))},
      {"storage_synthetic_rectangle_grounding_details",
       jsonSyntheticStorageRectangleGrounding(
           report.storageSyntheticRectangleGroundingDetails)},
      {"storage_synthetic_rectangle_grounding_details_truncated",
       report.storageSyntheticRectangleGroundingDetailsTruncated},
      {"storage_synthetic_rectangle_grounding_truncated",
       report.storageSyntheticRectangleGroundingTruncated},
      {"demands", jsonInteger(report.demands)},
      {"unique_demand_keys", jsonInteger(report.uniqueDemandRows)},
      {"selection_basis_rows", jsonInteger(report.selectionBasisRows)},
      {"basis_reduced_rows", jsonInteger(report.basisReducedRows)},
      {"basis_reduction_truncated", report.basisReductionTruncated},
      {"zero_distance_demand_keys", jsonInteger(report.zeroDistanceDemandRows)},
      {"recurrence_demand_keys", jsonInteger(report.recurrenceDemandRows)},
      {"same_resource_demand_keys", jsonInteger(report.sameResourceDemandRows)},
      {"cross_resource_demand_keys",
       jsonInteger(report.crossResourceDemandRows)},
      {"ssa_demand_keys", jsonInteger(report.ssaDemandRows)},
      {"raw_demand_keys", jsonInteger(report.rawDemandRows)},
      {"war_demand_keys", jsonInteger(report.warDemandRows)},
      {"waw_demand_keys", jsonInteger(report.wawDemandRows)},
      {"hardware_acc_rar_demand_keys",
       jsonInteger(report.hardwareAccRarDemandRows)},
      {"maximum_recurrence_distance",
       jsonInteger(report.maximumRecurrenceDistance)},
      {"direct_mechanisms", jsonInteger(report.directMechanisms)},
      {"singleton_candidates_covering_multiple_rows",
       jsonInteger(report.singletonCandidatesCoveringMultipleRows)},
      {"maximum_singleton_candidate_coverage_rows",
       jsonInteger(report.maximumSingletonCandidateCoverageRows)},
      {"total_singleton_candidate_coverage_rows",
       jsonInteger(report.totalSingletonCandidateCoverageRows)},
      {"candidate_mechanisms_by_origin",
       jsonMechanismOriginCounts(report.candidateMechanismsByOrigin)},
      {"direct_pair_proposals", jsonInteger(report.directPairProposals)},
      {"direct_pair_evaluations", jsonInteger(report.directPairEvaluations)},
      {"synergistic_pairs", jsonInteger(report.synergisticPairs)},
      {"pair_generation_truncated", report.pairGenerationTruncated},
      {"source_prefix_inspections",
       jsonInteger(report.sourcePrefixInspections)},
      {"source_prefix_candidates", jsonInteger(report.sourcePrefixCandidates)},
      {"source_prefix_incidences", jsonInteger(report.sourcePrefixIncidences)},
      {"source_prefix_generation_truncated",
       report.sourcePrefixGenerationTruncated},
      {"loop_carry_inspections", jsonInteger(report.loopCarryInspections)},
      {"loop_carry_candidates", jsonInteger(report.loopCarryCandidates)},
      {"loop_carry_incidences", jsonInteger(report.loopCarryIncidences)},
      {"loop_carry_generation_truncated", report.loopCarryGenerationTruncated},
      {"loop_boundary_protocol_inspections",
       jsonInteger(report.loopBoundaryProtocolInspections)},
      {"loop_boundary_protocol_candidates",
       jsonInteger(report.loopBoundaryProtocolCandidates)},
      {"loop_boundary_protocol_incidences",
       jsonInteger(report.loopBoundaryProtocolIncidences)},
      {"loop_boundary_protocol_generation_truncated",
       report.loopBoundaryProtocolGenerationTruncated},
      {"preparation_time_ns", jsonInteger(report.preparationNanoseconds)},
      {"strategies", std::move(strategies)}};
}

LogicalResult emitReport(func::FuncOp function, StringRef path,
                         const pto::CanonicalSyncComparisonReport &report) {
  function.emitRemark() << "canonical sync: demands=" << report.demands
                        << ", unique-demand-rows=" << report.uniqueDemandRows
                        << ", recurrence-demand-rows="
                        << report.recurrenceDemandRows
                        << ", same-resource-demand-rows="
                        << report.sameResourceDemandRows
                        << ", graph-nodes=" << report.graphNodes
                        << ", graph-edges=" << report.graphEdges
                        << ", direct-mechanisms=" << report.directMechanisms
                        << ", pair-proposals=" << report.directPairProposals
                        << ", pair-evaluations=" << report.directPairEvaluations
                        << ", synergistic-pairs=" << report.synergisticPairs;
  for (const pto::CanonicalSyncStrategyReport &strategy : report.strategies) {
    std::size_t maximumOverlap = 0;
    for (const pto::CanonicalSyncDomainAllocation &domain :
         strategy.allocation.domains) {
      maximumOverlap = std::max(maximumOverlap, domain.required);
    }
    function.emitRemark()
        << "canonical sync strategy=" << strategyName(strategy.strategy)
        << ", verified=" << strategy.verified
        << ", error=" << static_cast<unsigned>(strategy.error)
        << ", mechanisms=" << strategy.cost.mechanismCount
        << ", events=" << strategy.selectedEvents
        << ", targeted-barriers=" << strategy.selectedTargetedBarriers
        << ", pipe-all-barriers=" << strategy.selectedPipeAllBarriers
        << ", repairs=" << strategy.repairRounds
        << ", repair-rebuilds=" << strategy.repairCatalogRebuilds
        << ", repair-trials=" << strategy.repairTrials
        << ", repair-work=" << strategy.repairWorkUnits
        << ", repair-frontier-truncated=" << strategy.repairFrontierTruncated
        << ", repair-budget-exhausted=" << strategy.repairBudgetExhausted
        << ", backstop-deletion-trials=" << strategy.backstopDeletionTrials
        << ", backstop-deletion-work=" << strategy.backstopDeletionWorkUnits
        << ", backstop-deletion-truncated="
        << strategy.backstopDeletionTruncated
        << ", verification-work=" << strategy.verificationWorkUnits
        << ", predicted-sync-instructions="
        << strategy.predictedSyncInstructions
        << ", serialization=" << strategy.cost.serializationBreadth
        << ", lifetime=" << strategy.cost.eventLifetimeArea
        << ", maximum-event-overlap=" << maximumOverlap;
  }
  if (path.empty()) {
    return success();
  }
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error) {
    return function.emitError() << "cannot write canonical sync report '"
                                << path << "': " << error.message();
  }
  output << llvm::formatv("{0:2}\n", llvm::json::Value(jsonReport(report)));
  return success();
}

bool isKernelDispatchWrapper(func::FuncOp function) {
  const bool isEntry = function->hasAttr("pto.entry");
  const bool hasSingleBlock = function.getBody().hasOneBlock();
  if (!isEntry || !hasSingleBlock) {
    return false;
  }
  bool hasDispatch = false;
  for (Operation &operation : function.getBody().front().without_terminator()) {
    auto call = dyn_cast<func::CallOp>(operation);
    if (!call) {
      return false;
    }
    func::FuncOp callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || !callee->hasAttr("pto.kernel_kind")) {
      return false;
    }
    hasDispatch = true;
  }
  return hasDispatch;
}

bool shouldSkip(func::FuncOp function) {
  return function.isExternal() || function->hasAttr("pto.tileop.helper") ||
         function->hasAttr("pto.ptodsl.subkernel_helper") ||
         isKernelDispatchWrapper(function);
}

struct PTOCanonicalSyncPass
    : public pto::impl::PTOCanonicalSyncBase<PTOCanonicalSyncPass> {
  using pto::impl::PTOCanonicalSyncBase<
      PTOCanonicalSyncPass>::PTOCanonicalSyncBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (assumeDistinctGmArgsNoAlias && assumeAllGmAccessesNoAlias) {
      function.emitError(
          "--assume-distinct-gm-args-noalias and "
          "--assume-all-gm-accesses-noalias are mutually exclusive");
      signalPassFailure();
      return;
    }
    if (shouldSkip(function)) {
      return;
    }
    if (eventIdNumMax <= 0 || eventIdNumMax > kCompilerUsableEventIdCount) {
      function.emitError() << "event-id-num-max must be in [1, "
                           << kCompilerUsableEventIdCount << ']';
      signalPassFailure();
      return;
    }
    if (maximumPeriodicRecurrenceStates >
        static_cast<std::int64_t>(
            pto::kCanonicalSyncMaximumPeriodicRecurrenceStates)) {
      function.emitError()
          << "maximum-periodic-recurrence-states must not exceed "
          << pto::kCanonicalSyncMaximumPeriodicRecurrenceStates;
      signalPassFailure();
      return;
    }
    if (maximumRecurrenceWitnessStates >
        static_cast<std::int64_t>(
            pto::kCanonicalSyncMaximumRecurrenceWitnessStates)) {
      function.emitError()
          << "maximum-recurrence-witness-states must not exceed "
          << pto::kCanonicalSyncMaximumRecurrenceWitnessStates;
      signalPassFailure();
      return;
    }
    const auto validBound = [](std::int64_t value) {
      return value > 0 && static_cast<std::uint64_t>(value) <=
                              std::numeric_limits<std::size_t>::max();
    };
    const std::pair<std::int64_t, StringRef> bounds[] = {
        {maximumPeriodicRecurrenceStates, "maximum-periodic-recurrence-states"},
        {maximumRecurrenceWitnessStates, "maximum-recurrence-witness-states"},
        {maximumBasicOwnershipInspections,
         "maximum-basic-ownership-inspections"},
        {maximumBasicOwnershipCertificates,
         "maximum-basic-ownership-certificates"},
        {maximumBasicOwnershipSlots, "maximum-basic-ownership-slots"},
        {maximumBasicOwnershipPaths, "maximum-basic-ownership-paths"},
        {maximumBasicOwnershipUses, "maximum-basic-ownership-uses"},
        {maximumBasicOwnershipNodeReferences,
         "maximum-basic-ownership-node-references"},
        {maximumBasicOwnershipAccessIncidences,
         "maximum-basic-ownership-access-incidences"},
        {maximumPairEvaluationsPerScope, "maximum-pair-evaluations-per-scope"},
        {maximumSelectionWorkUnits, "maximum-selection-work-units"},
        {maximumRepairRounds, "maximum-repair-rounds"},
        {maximumRepairTrials, "maximum-repair-trials"},
        {maximumRepairWorkUnits, "maximum-repair-work-units"},
        {maximumRepairFrontierInspections,
         "maximum-repair-frontier-inspections"},
        {maximumRepairFrontierProposals, "maximum-repair-frontier-proposals"},
        {maximumBackstopDeletionTrials, "maximum-backstop-deletion-trials"},
        {maximumBackstopDeletionWorkUnits,
         "maximum-backstop-deletion-work-units"},
        {maximumVerificationWorkUnits, "maximum-verification-work-units"},
        {maximumDemandBasisGroupEdges, "maximum-demand-basis-group-edges"},
        {maximumDemandBasisReachabilityWords,
         "maximum-demand-basis-reachability-words"},
        {maximumDemandBasisReductionWork,
         "maximum-demand-basis-reduction-work"},
        {maximumSourcePrefixInspections, "maximum-source-prefix-inspections"},
        {maximumSourcePrefixCandidates, "maximum-source-prefix-candidates"},
        {maximumSourcePrefixIncidences, "maximum-source-prefix-incidences"},
        {maximumLoopCarryInspections, "maximum-loop-carry-inspections"},
        {maximumLoopCarryCandidates, "maximum-loop-carry-candidates"},
        {maximumLoopCarryIncidences, "maximum-loop-carry-incidences"},
        {maximumLoopBoundaryProtocolInspections,
         "maximum-loop-boundary-protocol-inspections"},
        {maximumLoopBoundaryProtocolCandidates,
         "maximum-loop-boundary-protocol-candidates"},
        {maximumLoopBoundaryProtocolIncidences,
         "maximum-loop-boundary-protocol-incidences"}};
    const auto invalidBound = llvm::find_if(
        bounds, [&](const auto &bound) { return !validBound(bound.first); });
    if (invalidBound != std::end(bounds)) {
      function.emitError() << invalidBound->second << " must be positive";
      signalPassFailure();
      return;
    }
    pto::CanonicalSyncBuildOptions options;
    options.eventIdBudget = static_cast<unsigned>(eventIdNumMax);
    options.analysis.maximumPeriodicRecurrenceStates =
        static_cast<std::size_t>(maximumPeriodicRecurrenceStates);
    options.analysis.maximumRecurrenceWitnessStates =
        static_cast<std::size_t>(maximumRecurrenceWitnessStates);
    options.analysis.maximumBasicOwnershipInspections =
        static_cast<std::size_t>(maximumBasicOwnershipInspections);
    options.analysis.maximumBasicOwnershipCertificates =
        static_cast<std::size_t>(maximumBasicOwnershipCertificates);
    options.analysis.maximumBasicOwnershipSlots =
        static_cast<std::size_t>(maximumBasicOwnershipSlots);
    options.analysis.maximumBasicOwnershipPaths =
        static_cast<std::size_t>(maximumBasicOwnershipPaths);
    options.analysis.maximumBasicOwnershipUses =
        static_cast<std::size_t>(maximumBasicOwnershipUses);
    options.analysis.maximumBasicOwnershipNodeReferences =
        static_cast<std::size_t>(maximumBasicOwnershipNodeReferences);
    options.analysis.maximumBasicOwnershipAccessIncidences =
        static_cast<std::size_t>(maximumBasicOwnershipAccessIncidences);
    if (!configurePatternMode(patternMode, options.patterns)) {
      function.emitError("pattern-mode must be direct or direct-pair");
      signalPassFailure();
      return;
    }
    if (!configureMechanismFamilies(mechanismFamilies, options.patterns)) {
      function.emitError()
          << "mechanism-families must be default, all, core, or a nonempty "
             "'+'-separated list of known family names";
      signalPassFailure();
      return;
    }
    if (!configureCatalogMode(catalogMode, options.patterns)) {
      function.emitError("catalog-mode must be standard or strict-direct");
      signalPassFailure();
      return;
    }
    options.patterns.enableConflictCoreRepair = enableConflictCoreRepair;
    const pto::CanonicalSyncMechanismFamilyMask strictOptionalFamilies =
        pto::canonicalSyncMechanismFamilyBit(
            pto::CanonicalSyncMechanismFamily::StorageCutEvent);
    const bool invalidStrictDirectConfiguration =
        options.patterns.catalogMode ==
            pto::CanonicalSyncCatalogMode::StrictMinimalDirect &&
        ((options.patterns.enabledMechanismFamilies &
          ~strictOptionalFamilies) != 0 ||
         options.patterns.enableConflictCoreRepair);
    if (invalidStrictDirectConfiguration) {
      function.emitError(
          "strict-direct catalog mode permits only mechanism-families=core "
          "or storage-cut-event, and requires "
          "enable-conflict-core-repair=false");
      signalPassFailure();
      return;
    }
    if (!configureSelectionStrategy(selectionStrategy, options.selection)) {
      function.emitError() << "selection-strategy must be fixed-cover, "
                              "action-aware-singleton, or pair-lookahead";
      signalPassFailure();
      return;
    }
    if (!configureSelectionObjective(selectionObjective, options.selection)) {
      function.emitError() << "selection-objective must be action-first or "
                              "serialization-first";
      signalPassFailure();
      return;
    }
    options.directPairs.maximumEvaluationsPerScope =
        static_cast<std::size_t>(maximumPairEvaluationsPerScope);
    options.selection.maximumWorkUnits =
        static_cast<std::size_t>(maximumSelectionWorkUnits);
    options.maximumRepairRounds = static_cast<std::size_t>(maximumRepairRounds);
    options.maximumRepairTrials = static_cast<std::size_t>(maximumRepairTrials);
    options.maximumRepairWorkUnits =
        static_cast<std::size_t>(maximumRepairWorkUnits);
    options.patterns.maximumRepairFrontierInspections =
        static_cast<std::size_t>(maximumRepairFrontierInspections);
    options.patterns.maximumRepairFrontierProposals =
        static_cast<std::size_t>(maximumRepairFrontierProposals);
    options.maximumBackstopDeletionTrials =
        static_cast<std::size_t>(maximumBackstopDeletionTrials);
    options.maximumBackstopDeletionWorkUnits =
        static_cast<std::size_t>(maximumBackstopDeletionWorkUnits);
    options.maximumVerificationWorkUnits =
        static_cast<std::size_t>(maximumVerificationWorkUnits);
    options.enableDemandBasisReduction = enableDemandBasisReduction;
    options.maximumDemandBasisGroupEdges =
        static_cast<std::size_t>(maximumDemandBasisGroupEdges);
    options.maximumDemandBasisReachabilityWords =
        static_cast<std::size_t>(maximumDemandBasisReachabilityWords);
    options.maximumDemandBasisReductionWork =
        static_cast<std::size_t>(maximumDemandBasisReductionWork);
    options.patterns.maximumSourcePrefixInspections =
        static_cast<std::size_t>(maximumSourcePrefixInspections);
    options.patterns.maximumSourcePrefixCandidates =
        static_cast<std::size_t>(maximumSourcePrefixCandidates);
    options.patterns.maximumSourcePrefixIncidences =
        static_cast<std::size_t>(maximumSourcePrefixIncidences);
    options.patterns.maximumLoopCarryInspections =
        static_cast<std::size_t>(maximumLoopCarryInspections);
    options.patterns.maximumLoopCarryCandidates =
        static_cast<std::size_t>(maximumLoopCarryCandidates);
    options.patterns.maximumLoopCarryIncidences =
        static_cast<std::size_t>(maximumLoopCarryIncidences);
    options.patterns.maximumLoopBoundaryProtocolInspections =
        static_cast<std::size_t>(maximumLoopBoundaryProtocolInspections);
    options.patterns.maximumLoopBoundaryProtocolCandidates =
        static_cast<std::size_t>(maximumLoopBoundaryProtocolCandidates);
    options.patterns.maximumLoopBoundaryProtocolIncidences =
        static_cast<std::size_t>(maximumLoopBoundaryProtocolIncidences);
    options.analysisOnly = analysisOnly;
    options.compareSelectionStrategies = analysisOnly;
    options.reportCallback =
        [&](const pto::CanonicalSyncComparisonReport &report) {
          return emitReport(function, comparisonReport, report);
        };
    options.analysis.gmAliasPolicy =
        assumeAllGmAccessesNoAlias
            ? pto::CanonicalSyncGmAliasPolicy::AllAccessesNoAlias
        : assumeDistinctGmArgsNoAlias
            ? pto::CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias
            : pto::CanonicalSyncGmAliasPolicy::MayAlias;
    options.analysis.discoverStorageLifecycleComponents =
        analyzeStorageLifecycles;
    if (failed(pto::runCanonicalSync(function, options))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOCanonicalSyncPass(const PTOCanonicalSyncOptions &options) {
  return std::make_unique<PTOCanonicalSyncPass>(options);
}
