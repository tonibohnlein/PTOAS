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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
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

constexpr std::int64_t kHardwareEventIdCount = 8;

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

llvm::json::Array jsonUnsignedValues(ArrayRef<std::uint64_t> values) {
  llvm::json::Array result;
  for (std::uint64_t value : values) {
    result.push_back(jsonInteger(value));
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
        {"precise_event_allocation",
         jsonAllocation(strategy.preciseAllocation)}});
  }
  return llvm::json::Object{
      {"schema", "ptoas.canonical_sync.v1"},
      {"function", report.function},
      {"gm_alias_policy", gmAliasPolicyName(report.gmAliasPolicy)},
      {"graph_nodes", jsonInteger(report.graphNodes)},
      {"graph_edges", jsonInteger(report.graphEdges)},
      {"certified_completion_frontiers",
       jsonInteger(report.certifiedCompletionFrontiers)},
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
      {"maximum_recurrence_distance",
       jsonInteger(report.maximumRecurrenceDistance)},
      {"direct_mechanisms", jsonInteger(report.directMechanisms)},
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
    if (eventIdNumMax <= 0 || eventIdNumMax > kHardwareEventIdCount) {
      function.emitError() << "event-id-num-max must be in [1, "
                           << kHardwareEventIdCount << ']';
      signalPassFailure();
      return;
    }
    const auto validBound = [](std::int64_t value) {
      return value > 0 && static_cast<std::uint64_t>(value) <=
                              std::numeric_limits<std::size_t>::max();
    };
    const std::pair<std::int64_t, StringRef> bounds[] = {
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
    if (!configurePatternMode(patternMode, options.patterns)) {
      function.emitError("pattern-mode must be direct or direct-pair");
      signalPassFailure();
      return;
    }
    if (!configureSelectionStrategy(selectionStrategy, options.selection)) {
      function.emitError() << "selection-strategy must be fixed-cover, "
                              "action-aware-singleton, or pair-lookahead";
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
