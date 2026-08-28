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

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <system_error>

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
    result.push_back(static_cast<std::int64_t>(value));
  }
  return result;
}

llvm::json::Object
jsonAllocation(const pto::CanonicalSyncResourceAllocation &allocation) {
  llvm::json::Array domains;
  for (const pto::CanonicalSyncDomainAllocation &domain : allocation.domains) {
    llvm::json::Array uses;
    for (const pto::CanonicalSyncEventAllocation &use : domain.uses) {
      llvm::json::Array ids;
      for (unsigned id : use.ids) {
        ids.push_back(static_cast<std::int64_t>(id));
      }
      uses.push_back(llvm::json::Object{
          {"mechanism", static_cast<std::int64_t>(use.mechanism)},
          {"event_use", static_cast<std::int64_t>(use.eventUse)},
          {"ids", std::move(ids)}});
    }
    llvm::json::Object item{
        {"domain", static_cast<std::int64_t>(domain.domain)},
        {"required", static_cast<std::int64_t>(domain.required)},
        {"available", static_cast<std::int64_t>(domain.available)},
        {"uses", std::move(uses)}};
    if (domain.maximumPressurePoint) {
      item["maximum_pressure_point"] =
          static_cast<std::int64_t>(*domain.maximumPressurePoint);
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
        {"error", static_cast<std::int64_t>(strategy.error)},
        {"verified", strategy.verified},
        {"used_localized_pipe_all", strategy.usedLocalizedPipeAll},
        {"repair_frontier_truncated", strategy.repairFrontierTruncated},
        {"repair_budget_exhausted", strategy.repairBudgetExhausted},
        {"backstop_deletion_truncated", strategy.backstopDeletionTruncated},
        {"repair_rounds", static_cast<std::int64_t>(strategy.repairRounds)},
        {"repair_trials", static_cast<std::int64_t>(strategy.repairTrials)},
        {"repair_work_units",
         static_cast<std::int64_t>(strategy.repairWorkUnits)},
        {"backstop_deletion_trials",
         static_cast<std::int64_t>(strategy.backstopDeletionTrials)},
        {"backstop_deletion_work_units",
         static_cast<std::int64_t>(strategy.backstopDeletionWorkUnits)},
        {"selected_events", static_cast<std::int64_t>(strategy.selectedEvents)},
        {"selected_targeted_barriers",
         static_cast<std::int64_t>(strategy.selectedTargetedBarriers)},
        {"selected_pipe_all_barriers",
         static_cast<std::int64_t>(strategy.selectedPipeAllBarriers)},
        {"action_profile", jsonUnsignedValues(strategy.cost.actionProfile)},
        {"serialization_breadth",
         static_cast<std::int64_t>(strategy.cost.serializationBreadth)},
        {"event_lifetime_area",
         static_cast<std::int64_t>(strategy.cost.eventLifetimeArea)},
        {"mechanisms", static_cast<std::int64_t>(strategy.cost.mechanismCount)},
        {"pattern_evaluations",
         static_cast<std::int64_t>(strategy.search.patternEvaluations)},
        {"deletion_evaluations",
         static_cast<std::int64_t>(strategy.search.deletionEvaluations)},
        {"work_units", static_cast<std::int64_t>(strategy.search.workUnits)},
        {"event_allocation", jsonAllocation(strategy.allocation)}});
  }
  return llvm::json::Object{
      {"demands", static_cast<std::int64_t>(report.demands)},
      {"unique_demand_keys", static_cast<std::int64_t>(report.demands)},
      {"direct_mechanisms", static_cast<std::int64_t>(report.directMechanisms)},
      {"direct_pair_proposals",
       static_cast<std::int64_t>(report.directPairProposals)},
      {"direct_pair_evaluations",
       static_cast<std::int64_t>(report.directPairEvaluations)},
      {"synergistic_pairs", static_cast<std::int64_t>(report.synergisticPairs)},
      {"pair_generation_truncated", report.pairGenerationTruncated},
      {"strategies", std::move(strategies)}};
}

LogicalResult emitReport(func::FuncOp function, StringRef path,
                         const pto::CanonicalSyncComparisonReport &report) {
  function.emitRemark() << "canonical sync: demands=" << report.demands
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
    if (maximumRepairRounds <= 0) {
      function.emitError("maximum-repair-rounds must be positive");
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
    options.maximumRepairRounds = static_cast<std::size_t>(maximumRepairRounds);
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
