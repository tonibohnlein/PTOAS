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
  counts["coverage_worlds"] =
      static_cast<std::int64_t>(statistics.coverageWorlds);
  counts["cover_universe"] =
      static_cast<std::int64_t>(statistics.coverUniverse);
  counts["cover_candidates"] =
      static_cast<std::int64_t>(statistics.coverCandidates);
  counts["structural_proposals"] =
      static_cast<std::int64_t>(statistics.structuralProposals);
  counts["admitted_structural_proposals"] =
      static_cast<std::int64_t>(statistics.admittedStructuralProposals);
  counts["structural_mechanism_memberships"] =
      static_cast<std::int64_t>(statistics.structuralMechanismMemberships);
  counts["structural_additional_coverage_rows"] = static_cast<std::int64_t>(
      statistics.structuralAdditionalCoverageRows);
  counts["structural_set_cover_candidates"] =
      static_cast<std::int64_t>(statistics.structuralSetCoverCandidates);
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
    structuralCoverMode = stringifyCanonicalStructuralCoverFamilies(
        options.structuralCoverFamilies);
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
    const std::optional<CanonicalStructuralCoverFamilies> parsedFamilies =
        parseCanonicalStructuralCoverFamilies(structuralCoverMode);
    if (!parsedFamilies) {
      getOperation().emitError("unknown canonical sync structural cover mode '")
          << structuralCoverMode << "'";
      return signalPassFailure();
    }
    options.structuralCoverFamilies = *parsedFamilies;
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

std::optional<CanonicalStructuralCoverFamilies>
parseCanonicalStructuralCoverFamilies(StringRef value) {
  if (value == "none") {
    return 0U;
  }
  if (value == "all") {
    return kAllCanonicalStructuralCoverFamilies;
  }
  SmallVector<StringRef, 5> names;
  value.split(names, ',', -1, false);
  CanonicalStructuralCoverFamilies families = 0U;
  for (StringRef name : names) {
    if (name == "level") {
      families |= static_cast<CanonicalStructuralCoverFamilies>(
          CanonicalStructuralCoverFamily::Level);
    } else if (name == "transitive") {
      families |= static_cast<CanonicalStructuralCoverFamilies>(
          CanonicalStructuralCoverFamily::Transitive);
    } else if (name == "connector") {
      families |= static_cast<CanonicalStructuralCoverFamilies>(
          CanonicalStructuralCoverFamily::Connector);
    } else if (name == "semantic") {
      families |= static_cast<CanonicalStructuralCoverFamilies>(
          CanonicalStructuralCoverFamily::Semantic);
    } else if (name == "storage") {
      families |= static_cast<CanonicalStructuralCoverFamilies>(
          CanonicalStructuralCoverFamily::Storage);
    } else {
      return std::nullopt;
    }
  }
  if (families == 0U) {
    return std::nullopt;
  }
  return families;
}

std::string stringifyCanonicalStructuralCoverFamilies(
    CanonicalStructuralCoverFamilies families) {
  if (families == 0U) {
    return "none";
  }
  if (families == kAllCanonicalStructuralCoverFamilies) {
    return "all";
  }
  std::string result;
  const auto append = [&](StringRef name) {
    if (!result.empty()) {
      result.push_back(',');
    }
    result.append(name.data(), name.size());
  };
  if (hasCanonicalStructuralCoverFamily(
          families, CanonicalStructuralCoverFamily::Level)) {
    append("level");
  }
  if (hasCanonicalStructuralCoverFamily(
          families, CanonicalStructuralCoverFamily::Transitive)) {
    append("transitive");
  }
  if (hasCanonicalStructuralCoverFamily(
          families, CanonicalStructuralCoverFamily::Connector)) {
    append("connector");
  }
  if (hasCanonicalStructuralCoverFamily(
          families, CanonicalStructuralCoverFamily::Semantic)) {
    append("semantic");
  }
  if (hasCanonicalStructuralCoverFamily(
          families, CanonicalStructuralCoverFamily::Storage)) {
    append("storage");
  }
  return result;
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
  failureStage = "structural-proposals";
  if (failed(proposeCanonicalSyncStructuralGroups(
          **program, options.structuralCoverFamilies))) {
    return failure();
  }
  statistics.structuralProposals =
      (*program)->getStructuralProposals().size();
  for (const CanonicalStructuralProposal &proposal :
       (*program)->getStructuralProposals()) {
    statistics.structuralMechanismMemberships += proposal.mechanisms.size();
  }
  failureStage = "coverage";
  if (failed(timed(statistics.coverageUs, [&]() {
        return evaluateCanonicalSyncCoverage(**program);
      }))) {
    return failure();
  }
  statistics.coverageWorlds = (*program)->getCoverageWorlds().size();
  for (const CanonicalStructuralProposal &proposal :
       (*program)->getStructuralProposals()) {
    if (!proposal.admitted) {
      continue;
    }
    ++statistics.admittedStructuralProposals;
    statistics.structuralAdditionalCoverageRows +=
        proposal.additionalCoverage.size();
  }
  failureStage = "set-cover-build";
  if (failed(timed(statistics.setCoverBuildUs, [&]() {
        return buildCanonicalSyncSetCoverInstance(**program);
      }))) {
    return failure();
  }
  if (const auto &instance = (*program)->getSetCoverInstance()) {
    statistics.coverUniverse = instance->universe.size();
    statistics.coverCandidates = instance->candidates.size();
    statistics.structuralSetCoverCandidates = llvm::count_if(
        instance->candidates,
        [](const CanonicalSetCoverCandidate &candidate) {
          return candidate.structuralProposal.has_value();
        });
  }
  failureStage = "selection";
  if (failed(timed(statistics.selectionUs,
                   [&]() { return solveCanonicalSyncSetCover(**program); }))) {
    return failure();
  }
  if (const auto &solution = (*program)->getSetCoverSolution()) {
    statistics.selectedMechanisms = solution->mechanisms.size();
  }
  failureStage = "allocation";
  const LogicalResult allocationResult = timed(
      statistics.allocationUs,
      [&]() { return allocateCanonicalSyncEvents(**program); });
  if (failed(allocationResult)) {
    if (options.dump || options.analysisOnly) {
      printCanonicalSyncProgram(**program, llvm::errs());
    }
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
