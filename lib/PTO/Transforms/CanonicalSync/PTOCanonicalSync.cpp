// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <mutex>

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOCANONICALSYNC
#include "PTO/Transforms/Passes.h.inc"

namespace {

using CanonicalSyncClock = std::chrono::steady_clock;

std::uint64_t elapsedMicroseconds(CanonicalSyncClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             CanonicalSyncClock::now() - start)
      .count();
}

void printCanonicalSyncStatistics(const CanonicalSyncStatistics &statistics,
                                  func::FuncOp function, bool succeeded,
                                  StringRef failureStage) {
  llvm::json::Object record;
  record["kind"] = "canonical-sync-statistics";
  record["function"] = function.getSymName().str();
  record["status"] = succeeded ? "ok" : "failed";
  if (!succeeded) {
    record["failure_stage"] = failureStage.str();
  }

  llvm::json::Object counts;
  counts["regions"] = static_cast<std::int64_t>(statistics.regions);
  counts["phases"] = static_cast<std::int64_t>(statistics.phases);
  counts["accesses"] = static_cast<std::int64_t>(statistics.accesses);
  counts["fence_effects"] = static_cast<std::int64_t>(statistics.fenceEffects);
  counts["demands"] = static_cast<std::int64_t>(statistics.demands);
  counts["fixed_covered_demands"] =
      static_cast<std::int64_t>(statistics.fixedCoveredDemands);
  counts["mechanisms"] = static_cast<std::int64_t>(statistics.mechanisms);
  counts["shared_event_frontiers"] =
      static_cast<std::int64_t>(statistics.sharedEventFrontiers);
  counts["shared_event_frontier_members"] =
      static_cast<std::int64_t>(statistics.sharedEventFrontierMembers);
  counts["selected_shared_event_frontiers"] =
      static_cast<std::int64_t>(statistics.selectedSharedEventFrontiers);
  counts["multi_demand_pipe_barrier_candidates"] =
      static_cast<std::int64_t>(statistics.multiDemandPipeBarrierCandidates);
  counts["multi_demand_pipe_barrier_covered_demands"] =
      static_cast<std::int64_t>(
          statistics.multiDemandPipeBarrierCoveredDemands);
  counts["selected_multi_demand_pipe_barriers"] =
      static_cast<std::int64_t>(statistics.selectedMultiDemandPipeBarriers);
  counts["coverage_worlds"] =
      static_cast<std::int64_t>(statistics.coverageWorlds);
  counts["cover_universe"] =
      static_cast<std::int64_t>(statistics.coverUniverse);
  counts["cover_candidates"] =
      static_cast<std::int64_t>(statistics.coverCandidates);
  counts["selected_mechanisms"] =
      static_cast<std::int64_t>(statistics.selectedMechanisms);
  counts["alias_pair_tests"] =
      static_cast<std::int64_t>(statistics.aliasPairTests);
  counts["alias_candidate_pairs"] =
      static_cast<std::int64_t>(statistics.aliasCandidatePairs);
  counts["local_interval_records"] =
      static_cast<std::int64_t>(statistics.localIntervalRecords);
  counts["sparse_incidence_entries"] =
      static_cast<std::int64_t>(statistics.sparseIncidenceEntries);
  counts["greedy_heap_pops"] =
      static_cast<std::int64_t>(statistics.greedyHeapPops);
  counts["greedy_incidence_visits"] =
      static_cast<std::int64_t>(statistics.greedyIncidenceVisits);
  counts["precomputed_prefix_entries"] =
      static_cast<std::int64_t>(statistics.precomputedPrefixEntries);
  counts["ssa_trace_visits"] =
      static_cast<std::int64_t>(statistics.ssaTraceVisits);
  counts["mechanism_intern_key_tests"] =
      static_cast<std::int64_t>(statistics.mechanismInternKeyTests);
  counts["mechanism_prefix_phase_tests"] =
      static_cast<std::int64_t>(statistics.mechanismPrefixPhaseTests);
  counts["coverage_fact_key_tests"] =
      static_cast<std::int64_t>(statistics.coverageFactKeyTests);
  counts["coverage_transfer_key_tests"] =
      static_cast<std::int64_t>(statistics.coverageTransferKeyTests);
  counts["coverage_transfer_compose_tests"] =
      static_cast<std::int64_t>(statistics.coverageTransferComposeTests);
  counts["coverage_propagation_fact_tests"] =
      static_cast<std::int64_t>(statistics.coveragePropagationFactTests);
  counts["coverage_boundary_phase_tests"] =
      static_cast<std::int64_t>(statistics.coverageBoundaryPhaseTests);
  counts["coverage_region_summaries"] =
      static_cast<std::int64_t>(statistics.coverageRegionSummaries);
  counts["coverage_summary_facts"] =
      static_cast<std::int64_t>(statistics.coverageSummaryFacts);
  counts["coverage_summary_transfers"] =
      static_cast<std::int64_t>(statistics.coverageSummaryTransfers);
  counts["coverage_oracle_worlds"] =
      static_cast<std::int64_t>(statistics.coverageOracleWorlds);
  counts["coverage_oracle_skipped_worlds"] =
      static_cast<std::int64_t>(statistics.coverageOracleSkippedWorlds);
  counts["coverage_oracle_state_operations"] =
      static_cast<std::int64_t>(statistics.coverageOracleStateOperations);
  counts["coverage_oracle_mechanism_tests"] =
      static_cast<std::int64_t>(statistics.coverageOracleMechanismTests);
  counts["coverage_oracle_demand_tests"] =
      static_cast<std::int64_t>(statistics.coverageOracleDemandTests);
  counts["coverage_oracle_source_instance_tests"] =
      static_cast<std::int64_t>(statistics.coverageOracleSourceInstanceTests);
  counts["verifier_loop_transfers"] =
      static_cast<std::int64_t>(statistics.verifierLoopTransfers);
  counts["max_verifier_loop_states"] =
      static_cast<std::int64_t>(statistics.maxVerifierLoopStates);
  record["counts"] = std::move(counts);

  llvm::json::Object timing;
  timing["structure"] = static_cast<std::int64_t>(statistics.structureUs);
  timing["demands"] = static_cast<std::int64_t>(statistics.demandsUs);
  timing["mechanisms"] = static_cast<std::int64_t>(statistics.mechanismsUs);
  timing["coverage"] = static_cast<std::int64_t>(statistics.coverageUs);
  timing["set_cover_build"] =
      static_cast<std::int64_t>(statistics.setCoverBuildUs);
  timing["selection"] = static_cast<std::int64_t>(statistics.selectionUs);
  timing["allocation"] = static_cast<std::int64_t>(statistics.allocationUs);
  timing["freeze"] = static_cast<std::int64_t>(statistics.freezeUs);
  timing["materialize_verify"] =
      static_cast<std::int64_t>(statistics.materializeVerifyUs);
  record["time_us"] = std::move(timing);

  static std::mutex statisticsMutex;
  const std::lock_guard<std::mutex> lock(statisticsMutex);
  llvm::errs() << llvm::json::Value(std::move(record)) << '\n';
}

struct PTOCanonicalSyncPass
    : public impl::PTOCanonicalSyncBase<PTOCanonicalSyncPass> {
  PTOCanonicalSyncPass() = default;

  explicit PTOCanonicalSyncPass(const CanonicalSyncOptions &options) {
    analysisOnly = options.analysisOnly;
    dump = options.dump || options.analysisOnly;
    statistics = options.statistics;
    gmAliasPolicy =
        stringifyCanonicalGmAliasPolicy(options.gmAliasPolicy).str();
  }

  void runOnOperation() override {
    CanonicalSyncOptions options;
    options.analysisOnly = analysisOnly;
    options.dump = dump || analysisOnly;
    options.statistics = statistics;
    const std::optional<CanonicalGmAliasPolicy> parsedPolicy =
        parseCanonicalGmAliasPolicy(gmAliasPolicy);
    if (!parsedPolicy) {
      getOperation().emitError("unknown canonical sync GM alias policy '")
          << gmAliasPolicy << "'";
      return signalPassFailure();
    }
    options.gmAliasPolicy = *parsedPolicy;
    if (failed(runCanonicalSync(getOperation(), options))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::optional<CanonicalGmAliasPolicy>
parseCanonicalGmAliasPolicy(StringRef value) {
  if (value == "conservative") {
    return CanonicalGmAliasPolicy::Conservative;
  }
  if (value == "distinct-roots-unsafe") {
    return CanonicalGmAliasPolicy::DistinctRootsUnsafe;
  }
  return std::nullopt;
}

FailureOr<std::unique_ptr<CanonicalSyncProgram>>
buildCanonicalSyncProgram(func::FuncOp function,
                          CanonicalGmAliasPolicy gmAliasPolicy,
                          CanonicalSyncStatistics *statistics) {
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(function);
  if (failed(target)) {
    return failure();
  }
  if (failed(canonical_sync_detail::rejectUnsupportedCanonicalSyncInput(
          function))) {
    return failure();
  }
  auto program = std::make_unique<CanonicalSyncProgram>(function, gmAliasPolicy,
                                                        statistics);
  CanonicalSyncClock::time_point start = CanonicalSyncClock::now();
  if (failed(canonical_sync_detail::buildCanonicalStructureAndAccesses(
          *program, *target))) {
    return failure();
  }
  if (statistics) {
    statistics->structureUs += elapsedMicroseconds(start);
    statistics->regions = program->getRegions().size();
    statistics->phases = program->getPhases().size();
    statistics->accesses = program->getAccesses().size();
    statistics->fenceEffects = program->getFenceEffects().size();
  }
  start = CanonicalSyncClock::now();
  if (failed(
          canonical_sync_detail::deriveCanonicalDemands(*program, *target))) {
    return failure();
  }
  if (failed(canonical_sync_detail::integrateCanonicalFixedBaseline(*program,
                                                                    *target))) {
    return failure();
  }
  if (statistics) {
    statistics->demandsUs += elapsedMicroseconds(start);
    statistics->demands = program->getDemands().size();
  }
  if (failed(program->freezeGraph())) {
    return failure();
  }
  return std::move(program);
}

LogicalResult runCanonicalSync(func::FuncOp function,
                               const CanonicalSyncOptions &options) {
  CanonicalSyncStatistics statistics;
  bool succeeded = false;
  std::string failureStage = "declaration";
  const auto report = llvm::make_scope_exit([&]() {
    if (options.statistics) {
      printCanonicalSyncStatistics(statistics, function, succeeded,
                                   failureStage);
    }
  });
  const auto timed = [](std::uint64_t &duration, auto &&callback) {
    const CanonicalSyncClock::time_point start = CanonicalSyncClock::now();
    const LogicalResult result = callback();
    duration += elapsedMicroseconds(start);
    return result;
  };
  // External declarations contain no scheduled physical work.  The function
  // pass is still invoked for them when a generated module contains private
  // runtime adapters, so leave them unchanged instead of asking the
  // structured-program builder to manufacture a body.
  if (function.isDeclaration()) {
    succeeded = true;
    return success();
  }
  failureStage = "graph";
  FailureOr<std::unique_ptr<CanonicalSyncProgram>> program =
      buildCanonicalSyncProgram(function, options.gmAliasPolicy, &statistics);
  if (failed(program)) {
    return failure();
  }
  failureStage = "mechanisms";
  if (failed(timed(statistics.mechanismsUs, [&]() {
        return buildCanonicalDirectMechanisms(**program);
      }))) {
    return failure();
  }
  statistics.mechanisms = (*program)->getMechanisms().size();
  failureStage = "coverage";
  if (failed(timed(statistics.coverageUs, [&]() {
        return evaluateCanonicalSyncCoverage(**program);
      }))) {
    return failure();
  }
  statistics.coverageWorlds = (*program)->getCoverageWorlds().size();
  failureStage = "set-cover-build";
  if (failed(timed(statistics.setCoverBuildUs, [&]() {
        return buildCanonicalSyncSetCoverInstance(**program);
      }))) {
    return failure();
  }
  if (const auto &instance = (*program)->getSetCoverInstance()) {
    statistics.coverUniverse = instance->universe.size();
    statistics.coverCandidates = instance->candidates.size();
  }
  failureStage = "selection";
  if (failed(timed(statistics.selectionUs,
                   [&]() { return solveCanonicalSyncSetCover(**program); }))) {
    return failure();
  }
  if (const auto &solution = (*program)->getSetCoverSolution()) {
    statistics.selectedMechanisms = solution->mechanisms.size();
    statistics.selectedSharedEventFrontiers =
        llvm::count_if(solution->mechanisms, [&](CanonicalMechanismId id) {
          return (*program)->getMechanism(id).synthesis ==
                 CanonicalMechanismSynthesis::SharedEventFrontier;
        });
  }
  failureStage = "allocation";
  if (failed(timed(statistics.allocationUs,
                   [&]() { return allocateCanonicalSyncEvents(**program); }))) {
    return failure();
  }
  failureStage = "freeze";
  if (failed(
          timed(statistics.freezeUs, [&]() { return (*program)->freeze(); }))) {
    return failure();
  }
  if (options.dump || options.analysisOnly) {
    printCanonicalSyncProgram(**program, llvm::errs());
  }
  if (options.analysisOnly) {
    llvm::errs() << "VERIFY skipped (analysis-only; IR unchanged)\n";
    succeeded = true;
    return success();
  }
  failureStage = "materialize-verify";
  if (failed(timed(statistics.materializeVerifyUs, [&]() {
        return materializeAndVerifyCanonicalSync(**program);
      }))) {
    return failure();
  }
  if (options.dump) {
    llvm::errs() << "VERIFY ok\n";
  }
  succeeded = true;
  return success();
}

std::unique_ptr<Pass>
createPTOCanonicalSyncPass(const CanonicalSyncOptions &options) {
  return std::make_unique<PTOCanonicalSyncPass>(options);
}

} // namespace pto
} // namespace mlir
