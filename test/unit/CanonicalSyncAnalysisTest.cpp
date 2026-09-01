// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace {

using namespace mlir;
using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncAnalysisTest failure: " << message << '\n';
  }
  return condition;
}

template <typename Result>
std::size_t takeIndex(const Result &result, bool &passed,
                      std::string_view message) {
  passed &=
      check(static_cast<bool>(result) && result.index.has_value(), message);
  return result.index.value_or(0);
}

bool verifyExactAndOneLessProtocolWork(
    const CanonicalSyncPatternProblem &problem,
    CanonicalSyncMechanismId mechanism, std::string_view exactMessage,
    std::string_view oneLessMessage) {
  SyncCoverCoverageWorkBudget measured;
  const CanonicalSyncProblemResult reference =
      problem.verifyMechanism(mechanism, &measured);
  if (!check(reference && measured.workUnits != 0,
             "measure production protocol verification work")) {
    return false;
  }
  SyncCoverCoverageWorkBudget exact(measured.workUnits);
  const CanonicalSyncProblemResult exactResult =
      problem.verifyMechanism(mechanism, &exact);
  SyncCoverCoverageWorkBudget oneLess(measured.workUnits - 1);
  const CanonicalSyncProblemResult oneLessResult =
      problem.verifyMechanism(mechanism, &oneLess);
  return check(exactResult && exact.workUnits == measured.workUnits,
               exactMessage) &&
         check(oneLessResult.error == CanonicalSyncProblemError::LimitExceeded,
               oneLessMessage);
}

template <typename Predicate>
CanonicalSyncProtocolVerifier testProtocolVerifier(Predicate predicate) {
  return [predicate = std::move(predicate)](
             const CanonicalSyncMechanismDescriptor &descriptor,
             SyncCoverCoverageWorkBudget &work) {
    const std::size_t supplies = descriptor.supplies.size();
    const bool squareOverflows =
        supplies != 0 &&
        supplies > std::numeric_limits<std::size_t>::max() / supplies;
    if (squareOverflows) {
      work.exhausted = true;
      return CanonicalSyncProblemError::LimitExceeded;
    }
    const std::size_t square = supplies * supplies;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const bool fixedOverflows =
        descriptor.eventUses.size() > maximum - 1 ||
        descriptor.actions.size() > maximum - 1 - descriptor.eventUses.size();
    const std::size_t fixed = fixedOverflows ? 0
                                             : 1 + descriptor.eventUses.size() +
                                                   descriptor.actions.size();
    if (fixedOverflows ||
        square > std::numeric_limits<std::size_t>::max() - fixed ||
        !work.consume(fixed + square)) {
      work.exhausted = true;
      return CanonicalSyncProblemError::LimitExceeded;
    }
    return predicate(descriptor)
               ? CanonicalSyncProblemError::None
               : CanonicalSyncProblemError::UnverifiedProtocol;
  };
}

std::string printOperation(Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}

bool matchesDenseDistanceZeroStorageHazards(const SyncCoverGraph &graph) {
  using HazardKey =
      std::tuple<SyncCoverNodeId, SyncCoverNodeId, SyncCoverDemandKind>;
  std::set<HazardKey> expected;
  for (const SyncCoverStorageAccess &first : graph.getStorageAccesses()) {
    for (const SyncCoverStorageAccess &second : graph.getStorageAccesses()) {
      if (first.node >= second.node || first.domain != second.domain ||
          first.extent.begin >= second.extent.end ||
          second.extent.begin >= first.extent.end) {
        continue;
      }
      const bool raw = syncCoverStorageModeWrites(first.mode) &&
                       syncCoverStorageModeReads(second.mode);
      if (raw) {
        expected.emplace(first.node, second.node,
                         SyncCoverDemandKind::MemoryRAW);
      }
      const bool war = syncCoverStorageModeReads(first.mode) &&
                       syncCoverStorageModeWrites(second.mode);
      if (war) {
        expected.emplace(first.node, second.node,
                         SyncCoverDemandKind::MemoryWAR);
      }
      const bool waw = syncCoverStorageModeWrites(first.mode) &&
                       syncCoverStorageModeWrites(second.mode);
      if (waw) {
        expected.emplace(first.node, second.node,
                         SyncCoverDemandKind::MemoryWAW);
      }
    }
  }
  std::set<HazardKey> actual;
  for (const SyncCoverDemand &demand : graph.getDemands()) {
    if (demand.distance != 0) {
      continue;
    }
    for (SyncCoverDemandKind kind : demand.provenanceKinds) {
      if (kind != SyncCoverDemandKind::SSA) {
        actual.emplace(demand.source, demand.target, kind);
      }
    }
  }
  return actual == expected;
}

void loadDialects(MLIRContext &context) {
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
}

bool expectAnalysisFailure(std::string_view source, StringRef functionName,
                           std::string_view expectedDiagnostic,
                           const CanonicalSyncAnalysisOptions &options = {}) {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      StringRef(source.data(), source.size()), &context);
  if (!check(static_cast<bool>(module), "parse rejected analysis fixture")) {
    return false;
  }
  bool sawExpectedDiagnostic = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    const std::string text = diagnostic.str();
    sawExpectedDiagnostic |= expectedDiagnostic.empty() ||
                             text.find(expectedDiagnostic) != std::string::npos;
    return success();
  });
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>(functionName), options);
  const std::string rejectedMessage =
      "reject unsupported analysis input for " + functionName.str();
  const std::string diagnosticMessage =
      "emit useful rejection diagnostic for " + functionName.str();
  return check(failed(program), rejectedMessage) &&
         check(sawExpectedDiagnostic, diagnosticMessage);
}

bool expectAnalysisSuccess(std::string_view source, StringRef functionName) {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      StringRef(source.data(), source.size()), &context);
  if (!check(static_cast<bool>(module), "parse accepted analysis fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>(functionName));
  return check(succeeded(program),
               "accept fixed synchronization outside canonical ownership");
}

bool testBuildsOneFrozenGraph() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @entry(
          %gm: !pto.partition_tensor_view<16x16xf32>,
          %mid: !pto.tile_buf<vec, 16x16xf32>,
          %out: !pto.tile_buf<vec, 16x16xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<vector>} {
        pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                  outs(%mid : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%mid : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%out : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse basic adapter fixture")) {
    return false;
  }

  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("entry"));
  if (!check(succeeded(program), "build basic adapter graph")) {
    return false;
  }
  const SyncCoverGraph &graph = program->getGraph();
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("entry");
  TLoadOp tload = *function.getBody().front().getOps<TLoadOp>().begin();
  TAbsOp tabs = *function.getBody().front().getOps<TAbsOp>().begin();
  const std::uint32_t vectorPipe =
      static_cast<std::uint32_t>(PipelineType::PIPE_V);
  const std::string rawDump = graph.getDeterministicRawDump();
  const bool typedPhysicalNodes = graph.getNodes()[0].physicalOperation == 0 &&
                                  graph.getNodes()[1].physicalOperation == 1 &&
                                  graph.getNodes()[0].macroPhase == -1 &&
                                  graph.getNodes()[1].macroPhase == -1;
  const bool typedPhysicalAccesses = llvm::all_of(
      graph.getStorageAccesses(), [](const SyncCoverStorageAccess &access) {
        return access.path == SyncCoverStorageAccessPath::PhysicalPipeline;
      });
  const bool typedRawDump =
      rawDump == graph.getDeterministicRawDump() &&
      rawDump.find("node 0 op=0 phase=-1") != std::string::npos &&
      rawDump.find("region 1 parent=0 scope=0 kind=1") != std::string::npos &&
      rawDump.find("access 0 node=0") != std::string::npos &&
      rawDump.find("requirements=1") != std::string::npos;
  FailureOr<CanonicalSyncHazardParityReport> parity =
      compareCanonicalSyncRawHazardsWithInsertSync(*program);
  return check(graph.isStructureFrozen(), "freeze authoritative graph") &&
         check(static_cast<bool>(graph.validate()), "validate adapter graph") &&
         check(graph.getNodes().size() == 2, "extract two scheduled nodes") &&
         check(
             graph.getRegions().size() == 2 &&
                 graph.getRegions()[0].kind == SyncCoverRegionKind::Function &&
                 graph.getRegions()[1].kind == SyncCoverRegionKind::Sequence &&
                 graph.getNodes()[0].region == 1 &&
                 graph.getNodes()[1].region == 1 &&
                 graph.getDemands().front().ownerRegion == 1,
             "build a flat function/sequence ownership tree") &&
         check(program->getNodeBindings().size() == graph.getNodes().size(),
               "keep one minimal node side table") &&
         check(program->getScopeBindings().size() == graph.getScopes().size(),
               "keep one minimal scope side table") &&
         check(
             program->getNodeBindings()[0].operation == tload.getOperation() &&
                 program->getNodeBindings()[1].operation == tabs.getOperation(),
             "bind graph nodes to the original MLIR operations") &&
         check(program->getNodeBindings()[0].macroPhase == -1 &&
                   program->getNodeBindings()[1].macroPhase == -1,
               "mark ordinary operations as non-macro nodes") &&
         check(typedPhysicalNodes,
               "record deterministic physical operation and phase IDs") &&
         check(program->getScopeBindings()[0].owner ==
                       function.getOperation() &&
                   program->getScopeBindings()[0].region == &function.getBody(),
               "bind root scope to the original function body") &&
         check(llvm::is_contained(graph.getNodes()[0].completionTargets,
                                  vectorPipe),
               "retain explicit MTE completion capability") &&
         check(graph.getDemands().size() == 1,
               "construct local RAW completion demand") &&
         check(matchesDenseDistanceZeroStorageHazards(graph),
               "match dense distance-zero storage hazard enumeration") &&
         check(llvm::is_contained(graph.getDemands().front().provenanceKinds,
                                  SyncCoverDemandKind::MemoryRAW),
               "retain RAW memory provenance") &&
         check(graph.getDemands().front().orderingRequirements ==
                   syncCoverOrderingRequirementBit(
                       SyncCoverOrderingRequirement::
                           PipelineCompletionBeforeAccess),
               "type the local RAW ordering requirement") &&
         check(typedPhysicalAccesses,
               "type translated accesses as physical-pipeline occurrences") &&
         check(typedRawDump,
               "serialize the typed raw graph deterministically") &&
         check(succeeded(parity) && parity->complete &&
                   parity->canonicalOnly.size() == 1 &&
                   parity->canonicalOnly.front().find("kind=WAW") !=
                       std::string::npos &&
                   parity->insertSyncOnly.empty() &&
                   parity->canonicalRawHazards.find("kind=RAW") !=
                       std::string::npos &&
                   parity->insertSyncRawHazards.find("kind=RAW") !=
                       std::string::npos,
               "match InsertSync's flat RAW ledger and classify Canonical's "
               "conservative unknown-root WAW") &&
         check(llvm::none_of(graph.getStorageAccesses(),
                             [](const SyncCoverStorageAccess &access) {
                               return access.exactPhysical;
                             }),
               "keep unplanned arguments conservative");
}

bool testOptionalLifecycleSynthesisTruncatesToDirectCatalog() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @lifecycle_truncation(
          %gm: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %addr = arith.constant 0 : i64
        %tile = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf32>
        scf.for %iv = %c0 to %c2 step %c1 {
          pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                    outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
          pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                    outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse lifecycle-truncation fixture")) {
    return false;
  }
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("lifecycle_truncation");
  CanonicalSyncAnalysisOptions analysisOptions;
  analysisOptions.discoverStorageLifecycleComponents = true;
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(function, analysisOptions);
  if (!check(succeeded(program), "build lifecycle-truncation graph")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  options.enableDemandBasisReduction = false;
  const CanonicalSyncMechanismFamilyMask genericLifecycle =
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::GenericLifecycle);
  options.patterns.enabledMechanismFamilies = genericLifecycle;
  options.patterns.enableDirectPairs = false;
  options.maximumLifecycleSynthesisWorkUnits = 1;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  const bool truncationReported =
      check(succeeded(problem),
            "retain a direct catalog after optional lifecycle truncation") &&
      check((*problem)
                ->getPatternStatistics()
                .genericLifecycleGenerationTruncated,
            "report optional lifecycle synthesis truncation") &&
      check((*problem)
                    ->getPatternStatistics()
                    .genericLifecycleSynthesisWorkUnits > 0,
            "report bounded lifecycle synthesis work");
  if (!truncationReported) {
    return false;
  }

  CanonicalSyncBuildOptions firstSignature = options;
  firstSignature.maximumLifecycleSynthesisWorkUnits = 1U << 20;
  CanonicalSyncBuildOptions secondSignature = firstSignature;
  ++secondSignature.maximumLifecycleSynthesisWorkUnits;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> firstProblem =
      buildCanonicalSyncSingletonProblem(*program, firstSignature);
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> secondProblem =
      buildCanonicalSyncSingletonProblem(*program, secondSignature);
  const bool signatureBound =
      check((kDefaultCanonicalSyncMechanismFamilies & genericLifecycle) == 0,
            "keep generic lifecycle synthesis opt-in") &&
      check(succeeded(firstProblem) && succeeded(secondProblem),
            "build lifecycle catalogs with distinct synthesis bounds") &&
      check(!(*firstProblem)->hasSameCandidatePrefix(**secondProblem),
            "bind the lifecycle synthesis bound into catalog identity");
  return signatureBound;
}

bool testGenericLifecycleMaterializationIsBoundedAndBalanced() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @generic_lifecycle(
          %gm: !pto.partition_tensor_view<16x16xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<vector>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %addr = arith.constant 0 : i64
        %one = arith.constant 1.000000e+00 : f32
        %tile = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf32>
        %condition = arith.cmpi slt, %c0, %c4 : index
        scf.if %condition {
          scf.if %condition {
            scf.if %condition {
              scf.if %condition {
                scf.for %iv = %c0 to %c4 step %c1 {
                  pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                            outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
                  pto.tmuls ins(%tile, %one :
                               !pto.tile_buf<vec, 16x16xf32>, f32)
                             outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
                }
              }
            }
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse generic-lifecycle fixture")) {
    return false;
  }
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("generic_lifecycle");
  CanonicalSyncAnalysisOptions analysisOptions;
  analysisOptions.discoverStorageLifecycleComponents = true;
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(function, analysisOptions);
  if (!check(succeeded(program), "build generic-lifecycle graph")) {
    return false;
  }
  const bool hasDeeplyGuardedDemand =
      llvm::any_of(program->getGraph().getDemands(), [](const auto &demand) {
        return demand.sourceGuard.literals.size() >= 4 &&
               demand.targetGuard.literals.size() >= 4;
      });
  if (!check(hasDeeplyGuardedDemand,
             "retain deeply guarded lifecycle demand rows")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  options.enableDemandBasisReduction = false;
  options.patterns.enabledMechanismFamilies = canonicalSyncMechanismFamilyBit(
      CanonicalSyncMechanismFamily::GenericLifecycle);
  options.patterns.enableDirectPairs = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(problem), "build generic-lifecycle catalog")) {
    return false;
  }

  const CanonicalSyncMechanismOriginMask lifecycleOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::GenericLifecycleProtocol);
  const auto lifecycle =
      llvm::find_if((*problem)->getMechanisms(), [&](const auto &mechanism) {
        return (mechanism.originMask & lifecycleOrigin) != 0;
      });
  if (!check(lifecycle != (*problem)->getMechanisms().end(),
             "synthesize a selectable generic lifecycle protocol") ||
      !check(lifecycle->descriptor.supplies.size() > 1,
             "exercise bounded staging with multiple lifecycle supplies")) {
    return false;
  }
  CanonicalSyncGreedyOptions lifecycleOnly;
  lifecycleOnly.strategy = CanonicalSyncSelectionStrategy::ActionAwareSingleton;
  for (const CanonicalSyncMechanism &mechanism : (*problem)->getMechanisms()) {
    if (mechanism.id != lifecycle->id) {
      lifecycleOnly.forbiddenMechanisms.push_back(mechanism.id);
    }
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem, lifecycleOnly);
  const CanonicalSyncVerifiedPlan semantic =
      verifyCanonicalSyncSelection(**problem, selection);
  SyncCoverCoverageWorkBudget measuredWork;
  const CanonicalSyncMaterializedPlanVerification measured =
      verifyCanonicalSyncMaterializedPlan(*program, **problem, semantic,
                                          &measuredWork);
  if (!check(selection &&
                 selection.mechanisms ==
                     std::vector<CanonicalSyncMechanismId>{lifecycle->id},
             "select only the generic lifecycle protocol") ||
      !check(measured && measured.lifecycleProtocolMechanisms == 1 &&
                 measured.deepProtocolVerifiersRun == 1 &&
                 measured.lifecycleAutomataVerified,
             "deeply verify the selected lifecycle automaton") ||
      !check(measuredWork.workUnits != 0,
             "measure lifecycle materialization verification work")) {
    return false;
  }
  SyncCoverCoverageWorkBudget exactWork(measuredWork.workUnits);
  const CanonicalSyncMaterializedPlanVerification exact =
      verifyCanonicalSyncMaterializedPlan(*program, **problem, semantic,
                                          &exactWork);
  SyncCoverCoverageWorkBudget oneLessWork(measuredWork.workUnits - 1);
  const CanonicalSyncMaterializedPlanVerification oneLess =
      verifyCanonicalSyncMaterializedPlan(*program, **problem, semantic,
                                          &oneLessWork);
  if (!check(exact && exactWork.workUnits == measuredWork.workUnits,
             "verify lifecycle materialization at its exact work bound") ||
      !check(!oneLess && oneLessWork.exhausted &&
                 oneLess.plan.error ==
                     CanonicalSyncSelectionError::WorkLimitExceeded,
             "reject lifecycle materialization at one less work unit") ||
      !check(succeeded(materializeCanonicalSyncPlan(*program, **problem,
                                                    measured.plan)),
             "materialize the deeply verified lifecycle protocol")) {
    return false;
  }

  scf::ForOp loop;
  function.walk([&](scf::ForOp candidate) { loop = candidate; });
  std::size_t primeSets = 0;
  std::size_t bodySets = 0;
  std::size_t bodyWaits = 0;
  std::size_t drainWaits = 0;
  bool guardedPrime = false;
  bool guardedDrain = false;
  function.walk([&](Operation *operation) {
    if (!loop || !operation->hasAttr("pto.canonical_sync")) {
      return;
    }
    scf::IfOp boundaryGuard = operation->getParentOfType<scf::IfOp>();
    Operation *boundary =
        boundaryGuard && boundaryGuard->getBlock() == loop->getBlock()
            ? boundaryGuard.getOperation()
            : operation;
    const bool topLevel = boundary->getBlock() == loop->getBlock();
    const bool inBody = operation->getBlock() == loop.getBody();
    if (isa<SetFlagOp>(operation)) {
      const bool prime = topLevel && boundary->isBeforeInBlock(loop);
      primeSets += prime;
      guardedPrime = guardedPrime || (prime && boundaryGuard);
      bodySets += inBody;
    }
    if (isa<WaitFlagOp>(operation)) {
      bodyWaits += inBody;
      const bool drain = topLevel && loop->isBeforeInBlock(boundary);
      drainWaits += drain;
      guardedDrain = guardedDrain || (drain && boundaryGuard);
    }
  });
  return check(primeSets == 1 && bodySets == 2 && bodyWaits == 2 &&
                   drainWaits == 1 && guardedPrime && guardedDrain,
               "materialize a balanced prime/body/drain lifecycle whose "
               "zero-trip path suppresses both boundary actions: prime=" +
                   std::to_string(primeSets) +
                   ", body-sets=" + std::to_string(bodySets) +
                   ", body-waits=" + std::to_string(bodyWaits) +
                   ", drains=" + std::to_string(drainWaits));
}

bool testMaterializationRejectsTamperedEventAllocations() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @allocation_validation(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<vector>} {
        %addr0 = arith.constant 0 : i64
        %addr1024 = arith.constant 1024 : i64
        %one = arith.constant 1.000000e+00 : f32
        %firstTile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        %secondTile = pto.alloc_tile addr = %addr1024 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%first : !pto.partition_tensor_view<16x16xf32>)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%firstTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%secondTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse allocation-validation fixture")) {
    return false;
  }
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("allocation_validation");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build allocation-validation graph")) {
    return false;
  }

  CanonicalSyncBuildOptions options;
  options.enableDemandBasisReduction = false;
  options.eventIdBudget = 4;
  options.patterns.enabledMechanismFamilies = 0;
  options.patterns.enableDirectPairs = false;

  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> directProblem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(directProblem),
             "build allocation-validation direct catalog")) {
    return false;
  }
  const CanonicalSyncSelection directSelection =
      selectCanonicalSyncPatterns(**directProblem);
  const bool directShape =
      directSelection && directSelection.mechanisms.size() == 2 &&
      (*directProblem)->getDomains().size() == 1 &&
      llvm::all_of(
          directSelection.mechanisms, [&](CanonicalSyncMechanismId mechanism) {
            const CanonicalSyncMechanismDescriptor &descriptor =
                (*directProblem)->getMechanisms()[mechanism].descriptor;
            return descriptor.kind == CanonicalSyncMechanismKind::Event &&
                   descriptor.eventUses.size() == 1 &&
                   descriptor.eventUses.front().width == 1;
          });
  if (!check(directShape,
             "select two independent direct allocation-validation events")) {
    return false;
  }

  const CanonicalSyncEventDomain &directDomain =
      (*directProblem)->getDomains().front();
  CanonicalSyncPatternProblem isolated(program->getGraph(),
                                       (*directProblem)->getDemands());
  bool isolatedBuilt = static_cast<bool>(isolated.addEventDomain(directDomain));
  isolatedBuilt =
      isolatedBuilt && isolated.addEventDomain({1,
                                                directDomain.targetResource,
                                                directDomain.sourceResource,
                                                directDomain.budget,
                                                {}});
  for (CanonicalSyncMechanismId mechanism : directSelection.mechanisms) {
    isolatedBuilt =
        isolatedBuilt &&
        isolated.internMechanism(
            (*directProblem)->getMechanisms()[mechanism].descriptor);
  }
  isolatedBuilt = isolatedBuilt && isolated.freeze();
  if (!check(isolatedBuilt,
             "freeze allocation-validation catalog with an unused domain")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(isolated);
  const CanonicalSyncVerifiedPlan semanticPlan =
      verifyCanonicalSyncSelection(isolated, selection);
  const CanonicalSyncMaterializedPlanVerification physicalPlan =
      verifyCanonicalSyncMaterializedPlan(*program, isolated, semanticPlan);
  const CanonicalSyncVerifiedPlan &verified = physicalPlan.plan;
  const bool verifiedShape = selection && physicalPlan &&
                             verified.mechanisms.size() == 2 &&
                             verified.allocation.domains.size() == 2 &&
                             verified.allocation.domains[0].domain == 0 &&
                             verified.allocation.domains[0].uses.size() == 2 &&
                             verified.allocation.domains[1].domain == 1 &&
                             verified.allocation.domains[1].uses.empty();
  if (!check(verifiedShape,
             "freshly verify the allocation-validation reference plan")) {
    return false;
  }

  FailureOr<CanonicalSyncProgram> mismatchedProgram =
      buildCanonicalSyncProgram(function);
  const CanonicalSyncMaterializedPlanVerification mismatchedGraphPlan =
      succeeded(mismatchedProgram)
          ? verifyCanonicalSyncMaterializedPlan(*mismatchedProgram, isolated,
                                                semanticPlan)
          : CanonicalSyncMaterializedPlanVerification{};
  if (!check(succeeded(mismatchedProgram),
             "build an independent allocation-validation graph") ||
      !check(!mismatchedGraphPlan &&
                 mismatchedGraphPlan.plan.error ==
                     CanonicalSyncSelectionError::FinalValidationFailed,
             "reject physical verification against a different graph")) {
    return false;
  }

  SyncCoverCoverageWorkBudget measuredVerification;
  const CanonicalSyncMaterializedPlanVerification measuredPlan =
      verifyCanonicalSyncMaterializedPlan(*program, isolated, semanticPlan,
                                          &measuredVerification);
  if (!check(measuredPlan && measuredVerification.workUnits != 0,
             "measure bounded physical-plan verification work")) {
    return false;
  }
  SyncCoverCoverageWorkBudget exactVerification(measuredVerification.workUnits);
  const CanonicalSyncMaterializedPlanVerification exactPlan =
      verifyCanonicalSyncMaterializedPlan(*program, isolated, semanticPlan,
                                          &exactVerification);
  SyncCoverCoverageWorkBudget oneLessVerification(
      measuredVerification.workUnits - 1);
  const CanonicalSyncMaterializedPlanVerification oneLessPlan =
      verifyCanonicalSyncMaterializedPlan(*program, isolated, semanticPlan,
                                          &oneLessVerification);
  if (!check(exactPlan &&
                 exactVerification.workUnits == measuredVerification.workUnits,
             "admit exact bounded physical-plan verification work") ||
      !check(!oneLessPlan && oneLessVerification.exhausted &&
                 oneLessPlan.plan.error ==
                     CanonicalSyncSelectionError::WorkLimitExceeded,
             "reject one-less bounded physical-plan verification work")) {
    return false;
  }

  const std::string irBefore = printOperation(function);
  const auto rejectsPlan = [&](CanonicalSyncVerifiedPlan plan,
                               std::string_view message) {
    bool sawDiagnostic = false;
    LogicalResult status = success();
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &) {
        sawDiagnostic = true;
        return success();
      });
      status = materializeCanonicalSyncPlan(*program, isolated, plan);
    }
    return check(failed(status) && sawDiagnostic, message);
  };

  CanonicalSyncVerifiedPlan wrongDomain = verified;
  wrongDomain.allocation.domains[1].uses.push_back(
      wrongDomain.allocation.domains[0].uses.front());
  wrongDomain.allocation.domains[0].uses.erase(
      wrongDomain.allocation.domains[0].uses.begin());

  CanonicalSyncVerifiedPlan wrongWidth = verified;
  wrongWidth.allocation.domains[0].uses.front().ids.push_back(3);

  CanonicalSyncVerifiedPlan reusedId = verified;
  reusedId.allocation.domains[0].uses[1].ids =
      reusedId.allocation.domains[0].uses[0].ids;

  CanonicalSyncVerifiedPlan missingUse = verified;
  missingUse.allocation.domains[0].uses.pop_back();

  CanonicalSyncVerifiedPlan extraUse = verified;
  CanonicalSyncEventAllocation duplicateUse =
      extraUse.allocation.domains[0].uses.front();
  duplicateUse.ids = {3};
  extraUse.allocation.domains[0].uses.push_back(std::move(duplicateUse));

  CanonicalSyncVerifiedPlan missingDomain = verified;
  missingDomain.allocation.domains.pop_back();

  CanonicalSyncVerifiedPlan extraDomain = verified;
  extraDomain.allocation.domains.push_back(
      extraDomain.allocation.domains.back());

  CanonicalSyncVerifiedPlan reservedId = verified;
  reservedId.allocation.domains[0].uses.front().ids = {6};

  CanonicalSyncVerifiedPlan removedMechanism = verified;
  removedMechanism.mechanisms.pop_back();
  removedMechanism.allocation =
      allocateCanonicalSyncEvents(isolated, removedMechanism.mechanisms);

  const bool rejectedTampering =
      rejectsPlan(semanticPlan,
                  "reject a semantic-only plan without a physical token") &&
      rejectsPlan(std::move(wrongDomain),
                  "reject an allocation indexed under the wrong domain") &&
      rejectsPlan(std::move(wrongWidth),
                  "reject an allocation with the wrong event width") &&
      rejectsPlan(std::move(reusedId),
                  "reject physical event-ID reuse within one domain") &&
      rejectsPlan(std::move(missingUse),
                  "reject a missing selected event allocation") &&
      rejectsPlan(std::move(extraUse),
                  "reject an extraneous selected event allocation") &&
      rejectsPlan(std::move(missingDomain),
                  "reject a missing event-allocation domain") &&
      rejectsPlan(std::move(extraDomain),
                  "reject an extraneous event-allocation domain") &&
      rejectsPlan(std::move(reservedId),
                  "reject a compiler-reserved event ID") &&
      rejectsPlan(std::move(removedMechanism),
                  "reject a mechanism set changed after token issuance") &&
      check(printOperation(function) == irBefore,
            "reject every tampered allocation before mutating IR");
  if (!rejectedTampering) {
    return false;
  }

  func::ReturnOp returnOp =
      *function.getBody().front().getOps<func::ReturnOp>().begin();
  OpBuilder builder(returnOp);
  arith::ConstantIndexOp injected =
      builder.create<arith::ConstantIndexOp>(returnOp.getLoc(), 7);
  const std::string mutatedIr = printOperation(function);
  const bool rejectedOperationMutation =
      rejectsPlan(verified,
                  "reject a physically verified plan after IR mutation") &&
      check(printOperation(function) == mutatedIr,
            "reject a stale materialization token before further IR "
            "mutation");
  injected.erase();
  if (!rejectedOperationMutation ||
      !check(printOperation(function) == irBefore,
             "restore the allocation-validation fixture after IR mutation")) {
    return false;
  }

  SmallVector<Block *> injectedBlocks;
  constexpr std::size_t extraEmptyBlocks = 64;
  injectedBlocks.reserve(extraEmptyBlocks);
  for (std::size_t index = 0; index < extraEmptyBlocks; ++index) {
    Block *block = new Block();
    function.getBody().push_back(block);
    injectedBlocks.push_back(block);
  }
  const bool rejectedIntraOperationMutation = rejectsPlan(
      verified,
      "reject a stale token after bounded intra-operation block growth");
  for (Block *block : injectedBlocks) {
    block->erase();
  }
  if (!rejectedIntraOperationMutation ||
      !check(printOperation(function) == irBefore,
             "restore the fixture after intra-operation block growth")) {
    return false;
  }

  program->getGraph() = std::move(mismatchedProgram->getGraph());
  return rejectsPlan(verified,
                     "reject a stale token after in-place graph replacement") &&
         check(printOperation(function) == irBefore,
               "reject graph replacement before mutating IR");
}

bool testMacroBindingsAndHiddenReservations() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @macro(
          %dst: !pto.partition_tensor_view<128xf32>,
          %src: !pto.partition_tensor_view<128xf32>,
          %ping: !pto.tile_buf<vec, 1x128xf32>,
          %pong: !pto.tile_buf<vec, 1x128xf32>) {
        pto.comm.tput(%dst, %src, buf(%ping, %pong) :
          !pto.partition_tensor_view<128xf32>,
          !pto.partition_tensor_view<128xf32>,
          !pto.tile_buf<vec, 1x128xf32>,
          !pto.tile_buf<vec, 1x128xf32>)
          {atomicType = #pto<atomic_type atomic_none>}
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse macro binding fixture")) {
    return false;
  }
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("macro");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build macro binding graph")) {
    return false;
  }
  TPutOp tput = *function.getBody().front().getOps<TPutOp>().begin();
  Operation *macro = tput.getOperation();
  const std::vector<CanonicalSyncNodeBinding> &bindings =
      program->getNodeBindings();
  const std::vector<SyncCoverNode> &nodes = program->getGraph().getNodes();
  const std::vector<Value> sourcePhaseOperands{tput.getSrc()};
  const std::vector<Value> destinationPhaseOperands{tput.getPing(),
                                                    tput.getPong()};
  const auto &reservations = program->getEventReservations();
  const std::pair<std::uint32_t, std::uint32_t> forward{
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE2),
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE3)};
  const std::pair<std::uint32_t, std::uint32_t> reverse{forward.second,
                                                        forward.first};
  return check(bindings.size() == 2, "expand one macro into two graph nodes") &&
         check(bindings[0].operation == macro && bindings[1].operation == macro,
               "bind both macro phases to one original operation") &&
         check(bindings[0].macroPhase == 0 && bindings[1].macroPhase == 1,
               "preserve deterministic macro phase identities") &&
         check(nodes[0].physicalOperation == nodes[1].physicalOperation &&
                   nodes[0].macroPhase == 0 && nodes[1].macroPhase == 1,
               "record authoritative macro phases on one physical op") &&
         check(bindings[0].ssaOperands == sourcePhaseOperands &&
                   bindings[1].ssaOperands == destinationPhaseOperands,
               "bind only phase-local synchronization macro operands") &&
         check(reservations.count(forward) == 1 &&
                   reservations.at(forward) == std::vector<unsigned>({0, 1}),
               "reserve hidden forward macro event IDs") &&
         check(reservations.count(reverse) == 1 &&
                   reservations.at(reverse) == std::vector<unsigned>({0, 1}),
               "reserve hidden reverse macro event IDs");
}

bool testEmptyNestedLoopTimelineIsClamped() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @nested(%limit: index,
                        %src: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %base = arith.constant 0 : i64
        %slot = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 16x16xf32>
        scf.for %outer = %c0 to %limit step %c1 {
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          scf.for %inner = %c0 to %limit step %c1 {
            %unused = arith.addi %inner, %c1 : index
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse empty nested-loop fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("nested"));
  if (!check(succeeded(program), "build empty nested-loop graph")) {
    return false;
  }
  const std::vector<SyncCoverScope> &scopes = program->getGraph().getScopes();
  const bool matchingTimelines =
      scopes.size() == 3 && scopes[1].timeline && scopes[2].timeline &&
      scopes[1].timeline->begin == scopes[2].timeline->begin &&
      scopes[1].timeline->end == scopes[2].timeline->end;
  return check(scopes.size() == 3, "retain both nested loop scopes") &&
         check(matchingTimelines,
               "empty inner loop inherits the refined parent timeline") &&
         check(static_cast<bool>(program->getGraph().validate()),
               "validate clamped nested-loop timeline");
}

FailureOr<CanonicalSyncProgram>
buildAliasingFixture(ModuleOp module, CanonicalSyncGmAliasPolicy policy) {
  CanonicalSyncAnalysisOptions options;
  options.gmAliasPolicy = policy;
  return buildCanonicalSyncProgram(
      module.lookupSymbol<func::FuncOp>("aliasing"), options);
}

bool testGmAliasPolicies() {
  constexpr std::string_view source = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @aliasing(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>) {
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 1024 : i64
        %src = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf32>
        %dst = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 16x16xf32>
        pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%first : !pto.partition_tensor_view<16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir";

  MLIRContext mayAliasContext;
  loadDialects(mayAliasContext);
  OwningOpRef<ModuleOp> mayAliasModule =
      parseSourceString<ModuleOp>(source, &mayAliasContext);
  if (!check(static_cast<bool>(mayAliasModule), "parse GM alias fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> mayAlias = buildAliasingFixture(
      *mayAliasModule, CanonicalSyncGmAliasPolicy::MayAlias);
  const bool validMayAlias =
      check(succeeded(mayAlias), "build may-alias graph") &&
      check(mayAlias->getGraph().getDemands().size() == 1,
            "conservative GM arguments may alias") &&
      check(mayAlias->getGraph().getDemands().front().orderingRequirements ==
                (syncCoverOrderingRequirementBit(
                     SyncCoverOrderingRequirement::
                         PipelineCompletionBeforeAccess) |
                 syncCoverOrderingRequirementBit(
                     SyncCoverOrderingRequirement::MemoryOrderBeforeAccess)),
            "type GM hazards as completion plus memory-order obligations");
  if (!validMayAlias) {
    return false;
  }

  MLIRContext noAliasContext;
  loadDialects(noAliasContext);
  OwningOpRef<ModuleOp> noAliasModule =
      parseSourceString<ModuleOp>(source, &noAliasContext);
  if (!check(static_cast<bool>(noAliasModule),
             "parse distinct GM argument fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> noAlias = buildAliasingFixture(
      *noAliasModule, CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias);
  return check(succeeded(noAlias), "build distinct-GM graph") &&
         check(noAlias->getGraph().getDemands().empty(),
               "distinct GM arguments suppress only cross-argument hazards");
}

bool testGmAliasContracts() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @same(%gm: !pto.partition_tensor_view<16x16xf32>) {
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 1024 : i64
        %src = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf32>
        %dst = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 16x16xf32>
        pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%gm : !pto.partition_tensor_view<16x16xf32>)
        pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @disjoint_slices(%gm: !pto.ptr<f32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 1024 : i64
        %view = pto.make_tensor_view %gm,
          shape = [%c32, %c16], strides = [%c16, %c1]
          : !pto.tensor_view<?x?xf32>
        %first = pto.partition_view %view,
          offsets = [%c0, %c0], sizes = [%c16, %c16]
          : !pto.tensor_view<?x?xf32>
            -> !pto.partition_tensor_view<16x16xf32>
        %second = pto.partition_view %view,
          offsets = [%c16, %c0], sizes = [%c16, %c16]
          : !pto.tensor_view<?x?xf32>
            -> !pto.partition_tensor_view<16x16xf32>
        %src = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf32>
        %dst = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 16x16xf32>
        pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%first : !pto.partition_tensor_view<16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @annotated(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>)
          attributes {pto.noalias_pairs = array<i64: 0, 1>} {
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 1024 : i64
        %src = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf32>
        %dst = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 16x16xf32>
        pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%first : !pto.partition_tensor_view<16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @recurrence(
          %gm: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 1024 : i64
        %src = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf32>
        %dst = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 16x16xf32>
        scf.for %iv = %c0 to %c2 step %c1 {
          pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%gm : !pto.partition_tensor_view<16x16xf32>)
          pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                    outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse GM contract fixture")) {
    return false;
  }
  CanonicalSyncAnalysisOptions distinct;
  distinct.gmAliasPolicy = CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias;
  FailureOr<CanonicalSyncProgram> same = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("same"), distinct);
  CanonicalSyncAnalysisOptions all;
  all.gmAliasPolicy = CanonicalSyncGmAliasPolicy::AllAccessesNoAlias;
  FailureOr<CanonicalSyncProgram> allAccesses = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("same"), all);
  FailureOr<CanonicalSyncProgram> disjointSlices = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("disjoint_slices"));
  FailureOr<CanonicalSyncProgram> recurrenceDistinct =
      buildCanonicalSyncProgram(
          module->lookupSymbol<func::FuncOp>("recurrence"), distinct);
  FailureOr<CanonicalSyncProgram> recurrenceAll = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("recurrence"), all);
  const auto hasLoopCarriedGmDemand = [](const CanonicalSyncProgram &program) {
    const SyncCoverGraph &graph = program.getGraph();
    return llvm::any_of(graph.getDemands(), [&](const SyncCoverDemand &demand) {
      return demand.distance > 0 &&
             llvm::any_of(
                 demand.storageWitnesses,
                 [&](SyncCoverStorageWitnessId witnessId) {
                   const SyncCoverStorageWitness &witness =
                       graph.getStorageWitnesses()[witnessId];
                   const SyncCoverStorageAccess &access =
                       graph.getStorageAccesses()[witness.sourceAccess];
                   return program.getStorageSpaces()[access.domain] ==
                          AddressSpace::GM;
                 });
    });
  };
  const auto hasLoopCarriedLocalDemand =
      [](const CanonicalSyncProgram &program) {
        const SyncCoverGraph &graph = program.getGraph();
        return llvm::any_of(
            graph.getDemands(), [&](const SyncCoverDemand &demand) {
              return demand.distance > 0 &&
                     llvm::any_of(
                         demand.storageWitnesses,
                         [&](SyncCoverStorageWitnessId witnessId) {
                           const SyncCoverStorageWitness &witness =
                               graph.getStorageWitnesses()[witnessId];
                           const SyncCoverStorageAccess &access =
                               graph.getStorageAccesses()[witness.sourceAccess];
                           return program.getStorageSpaces()[access.domain] !=
                                  AddressSpace::GM;
                         });
            });
      };
  FailureOr<CanonicalSyncProgram> annotated = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("annotated"));
  return check(succeeded(same), "build same-argument GM graph") &&
         check(!same->getGraph().getDemands().empty(),
               "preserve same-argument GM hazards") &&
         check(succeeded(allAccesses), "build all-GM-noalias graph") &&
         check(allAccesses->getGraph().getDemands().empty(),
               "strong GM contract suppresses same-argument hazards") &&
         check(succeeded(disjointSlices), "build disjoint GM slices graph") &&
         check(disjointSlices->getGraph().getDemands().empty(),
               "preserve disjoint root-relative GM slice ranges") &&
         check(succeeded(recurrenceDistinct),
               "build distinct-argument recurrence graph") &&
         check(hasLoopCarriedGmDemand(*recurrenceDistinct),
               "preserve same-argument loop-carried GM hazards") &&
         check(succeeded(recurrenceAll),
               "build all-GM-noalias recurrence graph") &&
         check(!hasLoopCarriedGmDemand(*recurrenceAll),
               "strong GM contract suppresses loop-carried GM hazards") &&
         check(hasLoopCarriedLocalDemand(*recurrenceAll),
               "strong GM contract preserves loop-carried local hazards") &&
         check(succeeded(annotated), "build annotated GM graph") &&
         check(annotated->getGraph().getDemands().empty(),
               "honor explicit distinct-argument noalias contract");
}

bool testStructuredIssueFrontier() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @branch(
          %condition: i1,
          %a: !pto.tile_buf<vec, 16x16xf32>,
          %b: !pto.tile_buf<vec, 16x16xf32>,
          %c: !pto.tile_buf<vec, 16x16xf32>,
          %d: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%b : !pto.tile_buf<vec, 16x16xf32>)
        scf.if %condition {
          pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%c : !pto.tile_buf<vec, 16x16xf32>)
        } else {
          pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%d : !pto.tile_buf<vec, 16x16xf32>)
        }
        pto.tabs ins(%c : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%d : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse structured frontier fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("branch"));
  if (!check(succeeded(program), "build structured frontier graph")) {
    return false;
  }
  std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>> issueEdges;
  for (const SyncCoverEdge &edge : program->getGraph().getEdges()) {
    if (edge.distance == 0) {
      issueEdges.insert({edge.source, edge.target});
    }
  }
  return check(issueEdges ==
                   std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>>{
                       {0, 1}, {0, 2}, {1, 3}, {2, 3}},
               "encode only structured immediate issue frontiers");
}

bool testDistanceTwoPhysicalSlotRecurrence() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @reuse(
          %src: !pto.partition_tensor_view<16x16xf32>, %limit: index)
          attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %base = arith.constant 0 : i64
        %one = arith.constant 1.000000e+00 : f32
        %buffer = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
        scf.for %i = %c0 to %limit step %c1 {
          %slot_index = arith.remui %i, %c2 : index
          %slot = pto.multi_tile_get %buffer[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
             -> !pto.tile_buf<vec, 16x16xf32>
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          pto.tmuls ins(%slot, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          pto.tadds ins(%slot, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse slot recurrence fixture")) {
    return false;
  }
  CanonicalSyncAnalysisOptions lifecycleOptions;
  lifecycleOptions.discoverStorageLifecycleComponents = true;
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("reuse"), lifecycleOptions);
  if (!check(succeeded(program), "build slot recurrence graph")) {
    return false;
  }
  const SyncCoverGraph &graph = program->getGraph();
  const auto recurrence =
      llvm::find_if(graph.getDemands(), [](const SyncCoverDemand &demand) {
        return demand.distance == 2 &&
               llvm::is_contained(demand.provenanceKinds,
                                  SyncCoverDemandKind::MemoryWAR);
      });
  const bool graphChecks =
      check(graph.getScopes().size() == 2, "construct explicit loop scope") &&
      check(graph.getScopes()[1].isLoop, "mark recurrence scope as loop") &&
      check(recurrence != graph.getDemands().end(),
            "recover distance-two slot reuse") &&
      check(recurrence->scope == 1, "attach recurrence to the loop timeline") &&
      check(llvm::any_of(graph.getStorageAccesses(),
                         [](const SyncCoverStorageAccess &access) {
                           return access.exactPhysical &&
                                  access.addressOrdinal.has_value();
                         }),
            "retain exact physical slot ordinals");
  if (!graphChecks) {
    return false;
  }
  const std::optional<SyncCoverStorageLifecycleIndex> &lifecycle =
      program->getStorageLifecycleIndex();
  const bool hasDistanceTwoRelease =
      lifecycle && lifecycle->isComplete() &&
      llvm::any_of(
          lifecycle->getComponents(),
          [](const SyncCoverStorageLifecycleComponent &component) {
            return llvm::any_of(
                component.edges, [](const SyncCoverStorageLifecycleEdge &edge) {
                  return edge.distance == 2 &&
                         (edge.kinds &
                          syncCoverStorageLifecycleEdgeKindBit(
                              SyncCoverStorageLifecycleEdgeKind::Release)) != 0;
                });
          });
  if (!check(hasDistanceTwoRelease,
             "index the phase-aware distance-two release lifecycle")) {
    return false;
  }

  CanonicalSyncBuildOptions options;
  // Inspect every distance-two recipe independently of optional selection-
  // basis reduction.
  options.enableDemandBasisReduction = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(problem),
             "build distance-two generic recurrence problem")) {
    return false;
  }
  const auto isDistanceTwoRing = [](const CanonicalSyncMechanism &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
           mechanism.descriptor.eventUses.size() == 1 &&
           mechanism.descriptor.eventUses.front().width == 2 &&
           llvm::any_of(mechanism.descriptor.supplies,
                        [](const CanonicalSyncSupplyBinding &binding) {
                          return binding.edge.distance == 2;
                        });
  };
  const auto distanceTwoRing =
      llvm::find_if((*problem)->getMechanisms(), isDistanceTwoRing);
  if (!check(distanceTwoRing != (*problem)->getMechanisms().end(),
             "generate a two-lane generic recurrence ring") ||
      !verifyExactAndOneLessProtocolWork(
          **problem, distanceTwoRing->id,
          "verify a direct recurrence at its exact work bound",
          "reject a direct recurrence at its one-less work bound")) {
    return false;
  }
  const auto isDistanceTwoLoopBoundaryPrefix =
      [](const CanonicalSyncMechanism &mechanism) {
        return mechanism.descriptor.kind ==
                   CanonicalSyncMechanismKind::Protocol &&
               mechanism.descriptor.eventUses.size() == 1 &&
               mechanism.descriptor.eventUses.front().width == 2 &&
               llvm::any_of(
                   mechanism.descriptor.actions,
                   [](const CanonicalSyncAction &action) {
                     return action.kind == CanonicalSyncActionKind::Barrier &&
                            action.anchor.kind ==
                                SyncCoverAnchorKind::LoopBodyExit &&
                            action.resource == static_cast<std::uint32_t>(
                                                   PipelineType::PIPE_V);
                   }) &&
               llvm::all_of(mechanism.descriptor.supplies,
                            [](const CanonicalSyncSupplyBinding &binding) {
                              return binding.edge.distance == 2 &&
                                     binding.proof ==
                                         CanonicalSyncSupplyProof::
                                             LoopBoundarySourcePrefixProtocol &&
                                     binding.attestedDemand &&
                                     binding.allowedDemands ==
                                         std::vector<SyncCoverDemandId>{
                                             *binding.attestedDemand};
                            });
      };
  const auto loopBoundaryProtocol = llvm::find_if(
      (*problem)->getMechanisms(), isDistanceTwoLoopBoundaryPrefix);
  if (!check(loopBoundaryProtocol != (*problem)->getMechanisms().end(),
             "generate a balanced two-lane A3 V loop-boundary prefix "
             "protocol") ||
      !verifyExactAndOneLessProtocolWork(
          **problem, loopBoundaryProtocol->id,
          "verify a loop-boundary protocol at its exact work bound",
          "reject a loop-boundary protocol at its one-less work bound")) {
    return false;
  }
  const CanonicalSyncPatternStatistics &protocolStatistics =
      (*problem)->getPatternStatistics();
  if (!check(protocolStatistics.loopBoundaryProtocolInspections ==
                     (*problem)->getObligationDemands().size() &&
                 protocolStatistics.loopBoundaryProtocolCandidates != 0 &&
                 protocolStatistics.loopBoundaryProtocolIncidences >= 2 &&
                 !protocolStatistics.loopBoundaryProtocolGenerationTruncated,
             "report complete loop-boundary protocol preparation")) {
    return false;
  }
  using LoopBoundaryGroup =
      std::tuple<SyncCoverScopeId, unsigned, std::uint32_t, std::uint32_t>;
  const auto collectLoopBoundaryGroups =
      [&](const CanonicalSyncPatternProblem &candidateProblem) {
        std::set<LoopBoundaryGroup> groups;
        std::size_t incidenceCount = 0;
        for (const CanonicalSyncMechanism &mechanism :
             candidateProblem.getMechanisms()) {
          if (mechanism.descriptor.kind !=
                  CanonicalSyncMechanismKind::Protocol ||
              mechanism.descriptor.supplies.empty() ||
              !llvm::all_of(mechanism.descriptor.supplies,
                            [](const CanonicalSyncSupplyBinding &binding) {
                              return binding.proof ==
                                     CanonicalSyncSupplyProof::
                                         LoopBoundarySourcePrefixProtocol;
                            })) {
            continue;
          }
          const CanonicalSyncSupplyBinding &binding =
              mechanism.descriptor.supplies.front();
          groups.insert({binding.edge.scope, binding.edge.distance,
                         graph.getNodes()[binding.edge.source].resource,
                         graph.getNodes()[binding.edge.target].resource});
          incidenceCount += mechanism.descriptor.supplies.size();
        }
        return std::make_pair(std::move(groups), incidenceCount);
      };
  const auto fullProtocolGroups = collectLoopBoundaryGroups(**problem);
  if (!check(fullProtocolGroups.first.size() ==
                     protocolStatistics.loopBoundaryProtocolCandidates &&
                 fullProtocolGroups.second ==
                     protocolStatistics.loopBoundaryProtocolIncidences,
             "count every loop-boundary descriptor and exact incidence")) {
    return false;
  }
  CanonicalSyncBuildOptions oneLessCandidate = options;
  oneLessCandidate.patterns.maximumLoopBoundaryProtocolCandidates =
      protocolStatistics.loopBoundaryProtocolCandidates - 1;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> oneLessProblem =
      buildCanonicalSyncSingletonProblem(*program, oneLessCandidate);
  std::set<LoopBoundaryGroup> omittedGroups;
  if (succeeded(oneLessProblem)) {
    const auto oneLessGroups = collectLoopBoundaryGroups(**oneLessProblem);
    std::set_difference(fullProtocolGroups.first.begin(),
                        fullProtocolGroups.first.end(),
                        oneLessGroups.first.begin(), oneLessGroups.first.end(),
                        std::inserter(omittedGroups, omittedGroups.end()));
  }
  if (!check(succeeded(oneLessProblem) &&
                 (*oneLessProblem)
                     ->getPatternStatistics()
                     .loopBoundaryProtocolGenerationTruncated &&
                 (*oneLessProblem)
                         ->getPatternStatistics()
                         .loopBoundaryProtocolCandidates ==
                     protocolStatistics.loopBoundaryProtocolCandidates - 1 &&
                 omittedGroups.size() == 1,
             "deterministically omit exactly one complete group at the "
             "one-less candidate bound")) {
    return false;
  }
  const auto boundedProtocolBuild =
      [&](std::size_t inspections, std::size_t candidates,
          std::size_t incidences, bool requireDistanceTwoProtocolAbsent,
          std::optional<std::size_t> maximumActions = std::nullopt) {
        CanonicalSyncBuildOptions bounded = options;
        bounded.patterns.maximumLoopBoundaryProtocolInspections = inspections;
        bounded.patterns.maximumLoopBoundaryProtocolCandidates = candidates;
        bounded.patterns.maximumLoopBoundaryProtocolIncidences = incidences;
        if (maximumActions) {
          bounded.problemLimits.maximumActionsPerMechanism = *maximumActions;
        }
        FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> built =
            buildCanonicalSyncSingletonProblem(*program, bounded);
        if (failed(built)) {
          return false;
        }
        const CanonicalSyncPatternStatistics &statistics =
            (*built)->getPatternStatistics();
        return statistics.loopBoundaryProtocolGenerationTruncated &&
               statistics.loopBoundaryProtocolInspections <= inspections &&
               statistics.loopBoundaryProtocolCandidates <= candidates &&
               statistics.loopBoundaryProtocolIncidences <= incidences &&
               (!requireDistanceTwoProtocolAbsent ||
                !llvm::any_of((*built)->getMechanisms(),
                              isDistanceTwoLoopBoundaryPrefix));
      };
  if (!check(boundedProtocolBuild(
                 protocolStatistics.loopBoundaryProtocolInspections - 1,
                 protocolStatistics.loopBoundaryProtocolCandidates,
                 protocolStatistics.loopBoundaryProtocolIncidences, true),
             "truncate the optional protocol family at its inspection bound") ||
      !check(boundedProtocolBuild(
                 protocolStatistics.loopBoundaryProtocolInspections,
                 protocolStatistics.loopBoundaryProtocolCandidates - 1,
                 protocolStatistics.loopBoundaryProtocolIncidences, false),
             "truncate the optional protocol family at its candidate bound") ||
      !check(boundedProtocolBuild(
                 protocolStatistics.loopBoundaryProtocolInspections,
                 protocolStatistics.loopBoundaryProtocolCandidates,
                 protocolStatistics.loopBoundaryProtocolIncidences - 1, false),
             "truncate the optional protocol family at its incidence bound") ||
      !check(boundedProtocolBuild(
                 protocolStatistics.loopBoundaryProtocolInspections,
                 protocolStatistics.loopBoundaryProtocolCandidates,
                 protocolStatistics.loopBoundaryProtocolIncidences, true, 6),
             "skip a distance-two protocol that exceeds its action bound")) {
    return false;
  }
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
  FailureOr<CanonicalSyncProgram> a5Program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("reuse"));
  if (!check(succeeded(a5Program),
             "build the distance-two recurrence graph for A5")) {
    return false;
  }
  bool sawA5IncompleteCatalog = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> a5Problem = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawA5IncompleteCatalog |=
          diagnostic.str().find("required singleton catalog is incomplete") !=
          std::string::npos;
      return success();
    });
    a5Problem = buildCanonicalSyncSingletonProblem(*a5Program, options);
  }
  const bool a5RecurrenceRejected =
      failed(a5Problem) && sawA5IncompleteCatalog &&
      a5Program->getTargetCapabilities()
          .directEventCompletion.resourcePairs.empty();
  if (!check(a5RecurrenceRejected,
             "fail closed instead of guessing an A5 recurrence event")) {
    return false;
  }
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
  const auto isDistanceTwoCrossResourceLoopCarry = [&](const auto &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier &&
           mechanism.descriptor.actions.size() == 1 &&
           llvm::any_of(
               mechanism.descriptor.supplies,
               [&](const CanonicalSyncSupplyBinding &binding) {
                 return binding.edge.distance == 2 &&
                        binding.proof ==
                            CanonicalSyncSupplyProof::LoopCarryPipeDrain &&
                        graph.getNodes()[binding.edge.source].resource !=
                            graph.getNodes()[binding.edge.target].resource;
               });
  };
  if (!check(!llvm::any_of((*problem)->getMechanisms(),
                           isDistanceTwoCrossResourceLoopCarry),
             "reject naked cross-resource distance-two loop-carry drains")) {
    return false;
  }
  const auto crossResourceDemand =
      llvm::find_if(graph.getDemands(), [&](const SyncCoverDemand &demand) {
        return demand.distance == 2 &&
               graph.getNodes()[demand.source].resource !=
                   graph.getNodes()[demand.target].resource;
      });
  if (!check(crossResourceDemand != graph.getDemands().end(),
             "find the cross-resource distance-two recurrence")) {
    return false;
  }
  const SyncCoverDemandId crossResourceDemandId =
      std::distance(graph.getDemands().begin(), crossResourceDemand);
  const SyncCoverNode &carrySource =
      graph.getNodes()[crossResourceDemand->source];
  CanonicalSyncMechanismDescriptor invalidCrossResourceCarry;
  invalidCrossResourceCarry.kind = CanonicalSyncMechanismKind::Barrier;
  invalidCrossResourceCarry.actions.push_back(
      {CanonicalSyncActionKind::Barrier,
       carrySource.resource,
       {SyncCoverAnchorKind::LoopBodyEntry, 0, crossResourceDemand->scope, 0},
       std::nullopt,
       0,
       {carrySource.resource},
       CanonicalSyncBarrierKind::Targeted,
       CanonicalSyncActionGuardKind::NotFirstIteration,
       crossResourceDemand->scope});
  CanonicalSyncSupplyBinding invalidBinding;
  invalidBinding.edge = {
      crossResourceDemand->source,         crossResourceDemand->target,
      SyncCoverEdgeKind::CompletionSupply, crossResourceDemand->scope,
      crossResourceDemand->distance,       crossResourceDemand->sourceGuard,
      crossResourceDemand->targetGuard};
  invalidBinding.barrierAction = 0;
  invalidBinding.proof = CanonicalSyncSupplyProof::LoopCarryPipeDrain;
  invalidBinding.attestedDemand = crossResourceDemandId;
  invalidBinding.allowedDemands = {crossResourceDemandId};
  invalidCrossResourceCarry.supplies.push_back(std::move(invalidBinding));
  CanonicalSyncPatternProblem invalidCarryProblem(
      graph, std::vector<SyncCoverDemandId>{crossResourceDemandId});
  if (!check(!invalidCarryProblem.internMechanism(
                 std::move(invalidCrossResourceCarry)),
             "reject a manually constructed naked cross-resource carry")) {
    return false;
  }
  std::vector<SyncCoverDemandId> protocolDemands;
  for (const CanonicalSyncSupplyBinding &binding :
       loopBoundaryProtocol->descriptor.supplies) {
    if (binding.attestedDemand) {
      protocolDemands.push_back(*binding.attestedDemand);
    }
  }
  llvm::sort(protocolDemands);
  protocolDemands.erase(
      std::unique(protocolDemands.begin(), protocolDemands.end()),
      protocolDemands.end());
  CanonicalSyncPatternProblem isolatedProtocolProblem(graph, protocolDemands);
  bool isolatedProblemBuilt = !protocolDemands.empty();
  for (const CanonicalSyncEventDomain &domain : (*problem)->getDomains()) {
    isolatedProblemBuilt =
        isolatedProblemBuilt && isolatedProtocolProblem.addEventDomain(domain);
  }
  const CanonicalSyncProblemResult isolatedProtocol =
      isolatedProtocolProblem.internVerifiedProtocol(
          loopBoundaryProtocol->descriptor,
          testProtocolVerifier(
              [&](const CanonicalSyncMechanismDescriptor &descriptor) {
                CanonicalSyncMechanism mechanism;
                mechanism.descriptor = descriptor;
                return isDistanceTwoLoopBoundaryPrefix(mechanism);
              }));
  isolatedProblemBuilt = isolatedProblemBuilt && isolatedProtocol &&
                         isolatedProtocolProblem.freeze();
  if (!check(isolatedProblemBuilt,
             "freeze the isolated distance-two loop-boundary cover")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(isolatedProtocolProblem);
  const CanonicalSyncVerifiedPlan semanticPlan =
      verifyCanonicalSyncSelection(isolatedProtocolProblem, selection);
  const CanonicalSyncMaterializedPlanVerification physicalPlan =
      verifyCanonicalSyncMaterializedPlan(*program, isolatedProtocolProblem,
                                          semanticPlan);
  const CanonicalSyncVerifiedPlan &verified = physicalPlan.plan;
  const bool selectedLoopBoundaryProtocol =
      selection &&
      selection.mechanisms == std::vector<CanonicalSyncMechanismId>{0};
  const bool recurrenceSelected =
      check(selectedLoopBoundaryProtocol && physicalPlan,
            "select and freshly verify only the distance-two loop-boundary "
            "recurrence protocol");
  const bool recurrenceMaterialized =
      check(physicalPlan && succeeded(materializeCanonicalSyncPlan(
                                *program, isolatedProtocolProblem, verified)),
            "materialize the selected distance-two loop-boundary protocol");
  if (!recurrenceSelected || !recurrenceMaterialized) {
    return false;
  }
  const auto rejectsProtocol = [&](CanonicalSyncMechanismDescriptor
                                       descriptor) {
    CanonicalSyncPatternProblem tampered(graph, protocolDemands);
    bool domainsAdded = true;
    for (const CanonicalSyncEventDomain &domain : (*problem)->getDomains()) {
      domainsAdded = domainsAdded && tampered.addEventDomain(domain);
    }
    const CanonicalSyncProblemResult added = tampered.internVerifiedProtocol(
        std::move(descriptor),
        testProtocolVerifier(
            [](const CanonicalSyncMechanismDescriptor &) { return true; }));
    return domainsAdded && !added;
  };
  CanonicalSyncMechanismDescriptor wrongWidth =
      loopBoundaryProtocol->descriptor;
  wrongWidth.eventUses.front().width = 1;
  CanonicalSyncMechanismDescriptor missingBarrier =
      loopBoundaryProtocol->descriptor;
  const auto barrierPosition = llvm::find_if(
      missingBarrier.actions, [](const CanonicalSyncAction &action) {
        return action.kind == CanonicalSyncActionKind::Barrier;
      });
  missingBarrier.actions.erase(barrierPosition);
  CanonicalSyncMechanismDescriptor wrongResource =
      loopBoundaryProtocol->descriptor;
  CanonicalSyncAction &wrongBarrier = *llvm::find_if(
      wrongResource.actions, [](const CanonicalSyncAction &action) {
        return action.kind == CanonicalSyncActionKind::Barrier;
      });
  wrongBarrier.resource = static_cast<std::uint32_t>(PipelineType::PIPE_MTE1);
  wrongBarrier.drainedResources = {wrongBarrier.resource};
  CanonicalSyncMechanismDescriptor unrestricted =
      loopBoundaryProtocol->descriptor;
  unrestricted.supplies.front().allowedDemands.clear();
  CanonicalSyncMechanismDescriptor wrongScope =
      loopBoundaryProtocol->descriptor;
  wrongScope.eventUses.front().recurrenceScope = 0;
  CanonicalSyncMechanismDescriptor wrongLane = loopBoundaryProtocol->descriptor;
  CanonicalSyncAction &bodyWait =
      *llvm::find_if(wrongLane.actions, [](const CanonicalSyncAction &action) {
        return action.kind == CanonicalSyncActionKind::EventWait &&
               action.anchor.kind == SyncCoverAnchorKind::LoopBodyEntry;
      });
  bodyWait.eventLaneKind = CanonicalSyncEventLaneKind::Static;
  bodyWait.eventLaneScope.reset();
  if (!check(rejectsProtocol(std::move(wrongWidth)),
             "reject a loop-boundary protocol with the wrong width") ||
      !check(rejectsProtocol(std::move(missingBarrier)),
             "reject a loop-boundary protocol without its V barrier") ||
      !check(rejectsProtocol(std::move(wrongResource)),
             "reject a loop-boundary protocol with the wrong resource") ||
      !check(rejectsProtocol(std::move(unrestricted)),
             "reject an unrestricted loop-boundary protocol binding") ||
      !check(rejectsProtocol(std::move(wrongScope)),
             "reject a loop-boundary protocol with the wrong scope") ||
      !check(rejectsProtocol(std::move(wrongLane)),
             "reject a loop-boundary protocol with the wrong dynamic lane")) {
    return false;
  }
  std::size_t dynamicSets = 0;
  std::size_t dynamicWaits = 0;
  std::vector<Operation *> staticPrimes;
  std::vector<Operation *> staticDrains;
  Operation *dynamicSet = nullptr;
  Operation *dynamicWait = nullptr;
  Operation *vectorBarrier = nullptr;
  std::size_t targetedBarriers = 0;
  std::size_t pipeAllBarriers = 0;
  bool unownedDynamicSync = false;
  const PIPE protocolTarget = static_cast<PIPE>(
      (*problem)
          ->getDomains()[loopBoundaryProtocol->descriptor.eventUses.front()
                             .domain]
          .targetResource);
  module->walk([&](Operation *operation) {
    dynamicSets += operation->getName().getStringRef() == "pto.set_flag_dyn";
    dynamicWaits += operation->getName().getStringRef() == "pto.wait_flag_dyn";
    const bool generated = operation->hasAttr("pto.canonical_sync");
    const bool dynamic = isa<SetFlagDynOp, WaitFlagDynOp>(operation);
    unownedDynamicSync = unownedDynamicSync || (dynamic && !generated);
    const auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
    const auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
    const bool protocolDomain = generated && source && target &&
                                source.getPipe() == PIPE::PIPE_V &&
                                target.getPipe() == protocolTarget;
    if (protocolDomain && isa<SetFlagOp>(operation)) {
      staticPrimes.push_back(operation);
    } else if (protocolDomain && isa<WaitFlagOp>(operation)) {
      staticDrains.push_back(operation);
    } else if (protocolDomain && isa<SetFlagDynOp>(operation)) {
      dynamicSet = operation;
    } else if (protocolDomain && isa<WaitFlagDynOp>(operation)) {
      dynamicWait = operation;
    }
    if (auto barrier = dyn_cast<BarrierOp>(operation);
        barrier && generated &&
        !barrier->hasAttr("pto.auto_sync_tail_barrier")) {
      if (barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
        ++pipeAllBarriers;
      } else {
        ++targetedBarriers;
        if (barrier.getPipe().getPipe() == PIPE::PIPE_V) {
          vectorBarrier = operation;
        }
      }
    }
  });
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("reuse");
  scf::ForOp loop;
  function.walk([&](scf::ForOp candidate) {
    if (!loop) {
      loop = candidate;
    }
  });
  const bool boundaryPlacement =
      loop && staticPrimes.size() == 2 && staticDrains.size() == 2 &&
      dynamicSet && dynamicWait && vectorBarrier &&
      llvm::all_of(staticPrimes,
                   [&](Operation *prime) {
                     return prime->getBlock() == loop->getBlock() &&
                            prime->isBeforeInBlock(loop);
                   }) &&
      llvm::all_of(staticDrains,
                   [&](Operation *drain) {
                     return drain->getBlock() == loop->getBlock() &&
                            loop->isBeforeInBlock(drain);
                   }) &&
      dynamicWait->getBlock() == loop.getBody() &&
      vectorBarrier->getBlock() == loop.getBody() &&
      dynamicSet->getBlock() == loop.getBody() &&
      dynamicWait->isBeforeInBlock(vectorBarrier) &&
      vectorBarrier->isBeforeInBlock(dynamicSet) &&
      dynamicSet->isBeforeInBlock(loop.getBody()->getTerminator());
  if (!check(dynamicSets == 1 && dynamicWaits == 1 && !unownedDynamicSync &&
                 targetedBarriers == 1 && pipeAllBarriers == 0,
             "emit only the selected distance-two protocol without PIPE_ALL") ||
      !check(boundaryPlacement,
             "place two primes, body-entry wait, V barrier plus body-exit set, "
             "and two drains at the exact loop boundaries")) {
    return false;
  }

  const std::string materialized = printOperation(function);
  CanonicalSyncBuildOptions analysisOnlyOptions = options;
  analysisOnlyOptions.analysisOnly = true;
  if (!check(succeeded(runCanonicalSync(function, analysisOnlyOptions)),
             "reanalyze pass-owned synchronization without mutation") ||
      !check(printOperation(function) == materialized,
             "preserve pass-owned IR in analysis-only mode")) {
    return false;
  }

  CanonicalSyncBuildOptions invalidOptions = options;
  invalidOptions.analysis.maximumNodes = 0;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    if (!check(failed(runCanonicalSync(function, invalidOptions)),
               "fail a rerun before replacement materialization")) {
      return false;
    }
  }
  const bool retainedAfterFailure =
      check(printOperation(function) == materialized,
            "retain the previous plan after a failed rerun");
  const bool replacementSucceeded =
      check(succeeded(runCanonicalSync(function, options)),
            "replace pass-owned synchronization on rerun");
  const std::string replacement = printOperation(function);
  const bool secondReplacementSucceeded =
      succeeded(runCanonicalSync(function, options));
  const std::string secondReplacement = printOperation(function);
  const bool replacementIsIdempotent =
      check(replacementSucceeded && secondReplacementSucceeded &&
                secondReplacement == replacement,
            "make pass-owned replacement idempotent");
  if (!retainedAfterFailure || !replacementSucceeded ||
      !replacementIsIdempotent) {
    return false;
  }
  return true;
}

bool testA5MatrixLoopBoundaryProtocolRequiresSourcedEventContract() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @matrix_reuse(
          %lhs: !pto.tile_buf<mat, 64x32xf16,
                              blayout=col_major, slayout=row_major>,
          %rhs: !pto.tile_buf<mat, 32x64xf16,
                              blayout=col_major, slayout=row_major>,
          %limit: index) attributes {
          pto.kernel_kind = #pto.kernel_kind<cube>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 8192 : i64
        %a2 = arith.constant 16384 : i64
        %a3 = arith.constant 24576 : i64
        scf.for %i = %c0 to %limit step %c1 {
          %left0 = pto.alloc_tile addr = %a0 :
            !pto.tile_buf<left, 32x32xf16,
                          blayout=col_major, slayout=row_major>
          pto.textract ins(%lhs, %c0, %c0 :
            !pto.tile_buf<mat, 64x32xf16,
                          blayout=col_major, slayout=row_major>, index, index)
            outs(%left0 : !pto.tile_buf<left, 32x32xf16,
                                        blayout=col_major,
                                        slayout=row_major>)
          %right0 = pto.alloc_tile addr = %a1 :
            !pto.tile_buf<right, 32x32xf16,
                          blayout=row_major, slayout=col_major>
          pto.textract ins(%rhs, %c0, %c0 :
            !pto.tile_buf<mat, 32x64xf16,
                          blayout=col_major, slayout=row_major>, index, index)
            outs(%right0 : !pto.tile_buf<right, 32x32xf16,
                                         blayout=row_major,
                                         slayout=col_major>)
          %acc0 = pto.alloc_tile addr = %a0 :
            !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                          slayout=row_major, fractal=1024>
          pto.tmatmul ins(%left0, %right0 :
            !pto.tile_buf<left, 32x32xf16,
                          blayout=col_major, slayout=row_major>,
            !pto.tile_buf<right, 32x32xf16,
                          blayout=row_major, slayout=col_major>)
            outs(%acc0 : !pto.tile_buf<acc, 32x32xf32,
                                       blayout=col_major, slayout=row_major,
                                       fractal=1024>)
          %left1 = pto.alloc_tile addr = %a2 :
            !pto.tile_buf<left, 32x32xf16,
                          blayout=col_major, slayout=row_major>
          pto.textract ins(%lhs, %c0, %c0 :
            !pto.tile_buf<mat, 64x32xf16,
                          blayout=col_major, slayout=row_major>, index, index)
            outs(%left1 : !pto.tile_buf<left, 32x32xf16,
                                        blayout=col_major,
                                        slayout=row_major>)
          %right1 = pto.alloc_tile addr = %a3 :
            !pto.tile_buf<right, 32x32xf16,
                          blayout=row_major, slayout=col_major>
          pto.textract ins(%rhs, %c0, %c0 :
            !pto.tile_buf<mat, 32x64xf16,
                          blayout=col_major, slayout=row_major>, index, index)
            outs(%right1 : !pto.tile_buf<right, 32x32xf16,
                                         blayout=row_major,
                                         slayout=col_major>)
          %acc1 = pto.alloc_tile addr = %a1 :
            !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                          slayout=row_major, fractal=1024>
          pto.tmatmul ins(%left1, %right1 :
            !pto.tile_buf<left, 32x32xf16,
                          blayout=col_major, slayout=row_major>,
            !pto.tile_buf<right, 32x32xf16,
                          blayout=row_major, slayout=col_major>)
            outs(%acc1 : !pto.tile_buf<acc, 32x32xf32,
                                       blayout=col_major, slayout=row_major,
                                       fractal=1024>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse A5 matrix loop-boundary protocol fixture")) {
    return false;
  }
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("matrix_reuse");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program),
             "build A5 matrix loop-boundary protocol graph")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  options.enableDemandBasisReduction = false;
  const std::uint32_t matrix = static_cast<std::uint32_t>(PipelineType::PIPE_M);
  bool sawIncompleteCatalog = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawIncompleteCatalog |=
          diagnostic.str().find("required singleton catalog is incomplete") !=
          std::string::npos;
      return success();
    });
    problem = buildCanonicalSyncSingletonProblem(*program, options);
  }
  const bool rejectedWithoutA5EventContract =
      failed(problem) && sawIncompleteCatalog &&
      program->getTargetCapabilities()
          .directEventCompletion.resourcePairs.empty();

  (*module)->removeAttr("pto.target_arch");
  FailureOr<CanonicalSyncProgram> unsupported =
      buildCanonicalSyncProgram(function);
  const bool unsupportedRejected =
      succeeded(unsupported) &&
      !unsupported->getGraph().supportsBlockingTargetedBarrier(matrix);
  return check(rejectedWithoutA5EventContract,
               "reject the A5 protocol catalog without a sourced directed "
               "event contract") &&
         check(unsupportedRejected,
               "do not authorize the non-V protocol without target barrier "
               "capabilities");
}

bool testDistanceTwoCrossRootSlotRecurrence() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @reuse(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %dst: !pto.partition_tensor_view<16x16xf32>, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %base = arith.constant 0 : i64
        %first = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
        %second = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
        scf.for %i = %c0 to %limit step %c1 {
          %slot_index = arith.remui %i, %c2 : index
          %read_slot = pto.multi_tile_get %first[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
             -> !pto.tile_buf<vec, 16x16xf32>
          %write_slot = pto.multi_tile_get %second[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
             -> !pto.tile_buf<vec, 16x16xf32>
          pto.tstore ins(%read_slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.partition_tensor_view<16x16xf32>)
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%write_slot : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse cross-root recurrence fixture")) {
    return false;
  }
  CanonicalSyncAnalysisOptions options;
  options.gmAliasPolicy = CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias;
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("reuse"), options);
  if (!check(succeeded(program), "build cross-root recurrence graph")) {
    return false;
  }
  return check(llvm::any_of(program->getGraph().getDemands(),
                            [](const SyncCoverDemand &demand) {
                              return demand.distance == 2 &&
                                     llvm::is_contained(
                                         demand.provenanceKinds,
                                         SyncCoverDemandKind::MemoryWAR);
                            }),
               "discover periodic aliasing across distinct allocation roots");
}

bool testTargetCapabilityProfilesAreVersionedAndConservative() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2"} {
      func.func @profile(
          %tile: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.partition_tensor_view<16x16xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<vector>} {
        pto.tabs ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
          outs(%dst : !pto.partition_tensor_view<16x16xf32>)
        return
      }
      func.func @cube_profile() attributes {
          pto.kernel_kind = #pto.kernel_kind<cube>} {
        return
      }
      func.func @unresolved_profile() {
        return
      }
      func.func @section_profile() {
        pto.section.cube {
        }
        return
      }
      func.func @mixed_section_profile() {
        pto.section.cube {
        }
        pto.section.vector {
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse target-capability profile fixture")) {
    return false;
  }
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("profile");
  const auto pipe = [](PipelineType resource) {
    return static_cast<std::uint32_t>(resource);
  };
  const auto checkKnownProfile = [&](StringRef arch,
                                     CanonicalSyncTargetProfile profile,
                                     bool vectorCompletionOrdered,
                                     bool vectorBarrierDrain,
                                     bool a3Ownership) {
    (*module)->setAttr("pto.target_arch", StringAttr::get(&context, arch));
    FailureOr<CanonicalSyncProgram> program =
        buildCanonicalSyncProgram(function);
    if (!check(succeeded(program), "build a known target capability profile")) {
      return false;
    }
    const CanonicalSyncTargetCapabilities &capabilities =
        program->getTargetCapabilities();
    const std::uint32_t vector = pipe(PipelineType::PIPE_V);
    const std::uint32_t store = pipe(PipelineType::PIPE_MTE3);
    const auto carry =
        program->getGraph().getResourceRecurrenceCarryKinds().find(vector);
    const auto vectorNode = llvm::find_if(
        program->getGraph().getNodes(),
        [&](const SyncCoverNode &node) { return node.resource == vector; });
    const bool commonContracts =
        capabilities.profile == profile &&
        capabilities.coreDomain == CanonicalSyncCoreDomain::AIV &&
        capabilities.syncSpecVersion !=
            CanonicalSyncTargetSyncSpecVersion::None &&
        capabilities.evidence.size() ==
            (profile == CanonicalSyncTargetProfile::A5V1 ? 4 : 5) &&
        capabilities.compilerUsableEventIds ==
            std::vector<unsigned>({0, 1, 2, 3, 4, 5}) &&
        capabilities.legalPipeBarriers.supports(pipe(PipelineType::PIPE_V)) ==
            vectorBarrierDrain &&
        !capabilities.legalPipeBarriers.supports(pipe(PipelineType::PIPE_S)) &&
        capabilities.sameResourceCompletionOrdering.version == 1 &&
        capabilities.sameResourceCompletionOrdering.supports(
            pipe(PipelineType::PIPE_S)) &&
        capabilities.sameResourceCompletionOrdering.supports(
            pipe(PipelineType::PIPE_V)) == vectorCompletionOrdered &&
        capabilities.targetedBarrierDrainsSourcePrefix.version == 1 &&
        capabilities.targetedBarrierDrainsSourcePrefix.supports(
            pipe(PipelineType::PIPE_M)) &&
        capabilities.targetedBarrierDrainsSourcePrefix.supports(
            pipe(PipelineType::PIPE_V)) == vectorBarrierDrain &&
        capabilities.crossResourceTargetedBarrierCompletion.version == 0 &&
        capabilities.crossResourceTargetedBarrierCompletion.resourcePairs
            .empty() &&
        capabilities.directEventCompletion.supports(vector, store) ==
            (profile != CanonicalSyncTargetProfile::A5V1) &&
        capabilities.directEventCompletion.resourcePairs.size() ==
            (profile != CanonicalSyncTargetProfile::A5V1 ? 12 : 0) &&
        capabilities.directEventCompletesSourcePrefix.isEnabled() ==
            (profile != CanonicalSyncTargetProfile::A5V1) &&
        (capabilities.directEventOrderingRequirements != 0) ==
            (profile != CanonicalSyncTargetProfile::A5V1) &&
        capabilities.pipeBarrierOrderingRequirements != 0 &&
        capabilities.hardwareEventCompletion.supports(vector, store) ==
            (profile != CanonicalSyncTargetProfile::A5V1) &&
        capabilities.crossPipeAccumulatorReadReadHazard.isEnabled() ==
            (profile != CanonicalSyncTargetProfile::A5V1);
    const bool graphContracts =
        carry != program->getGraph().getResourceRecurrenceCarryKinds().end() &&
        carry->second ==
            (vectorCompletionOrdered
                 ? SyncCoverEdgeKind::CompletionPreservingIssueOrder
                 : SyncCoverEdgeKind::NonCompletionPreservingIssueOrder) &&
        vectorNode != program->getGraph().getNodes().end() &&
        vectorNode->completionSignalCoversIssuedPrefix ==
            vectorCompletionOrdered &&
        llvm::is_contained(vectorNode->completionTargets, store) ==
            (profile != CanonicalSyncTargetProfile::A5V1) &&
        program->getGraph().supportsBlockingTargetedBarrier(vector) ==
            vectorBarrierDrain &&
        !program->getGraph().supportsCrossResourceTargetedBarrier(
            pipe(PipelineType::PIPE_MTE1), pipe(PipelineType::PIPE_MTE2));
    const bool ownershipContracts =
        capabilities.mte1L0ReadySetCompletesPrefix.isEnabled() == a3Ownership &&
        capabilities.mL0AlternativeJoinSetCompletes.isEnabled() ==
            a3Ownership &&
        capabilities.mte1ScopeExitSetCompletesPrefix.isEnabled() ==
            a3Ownership &&
        capabilities.mToFixAccumulatorBoundaryCompletes.isEnabled() ==
            a3Ownership &&
        capabilities.intrinsicMmadAccumulatorOrdering.isEnabled() ==
            a3Ownership &&
        capabilities.targetCompletionResources.has_value() == a3Ownership;
    return check(commonContracts,
                 "map the target to its versioned common contracts") &&
           check(graphContracts,
                 "apply the profile to graph completion and barrier facts") &&
           check(ownershipContracts,
                 "keep A3-only completion contracts target-qualified");
  };
  if (!checkKnownProfile("a2", CanonicalSyncTargetProfile::A2V1,
                         /*vectorCompletionOrdered=*/false,
                         /*vectorBarrierDrain=*/true,
                         /*a3Ownership=*/false) ||
      !checkKnownProfile("a2a3", CanonicalSyncTargetProfile::A2A3IntersectionV1,
                         /*vectorCompletionOrdered=*/false,
                         /*vectorBarrierDrain=*/true,
                         /*a3Ownership=*/false) ||
      !checkKnownProfile("a3", CanonicalSyncTargetProfile::A3V1,
                         /*vectorCompletionOrdered=*/false,
                         /*vectorBarrierDrain=*/true,
                         /*a3Ownership=*/true) ||
      !checkKnownProfile("a5", CanonicalSyncTargetProfile::A5V1,
                         /*vectorCompletionOrdered=*/true,
                         /*vectorBarrierDrain=*/false,
                         /*a3Ownership=*/false)) {
    return false;
  }

  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
  FailureOr<CanonicalSyncProgram> cube = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("cube_profile"));
  if (!check(succeeded(cube), "build the A3 AIC event profile")) {
    return false;
  }
  const CanonicalSyncTargetCapabilities &cubeCapabilities =
      cube->getTargetCapabilities();
  const CanonicalSyncDirectedResourceCapability &aicHardware =
      cubeCapabilities.hardwareEventCompletion;
  const CanonicalSyncDirectedResourceCapability &aicExposed =
      cubeCapabilities.directEventCompletion;
  const std::uint32_t matrix = pipe(PipelineType::PIPE_M);
  const std::uint32_t mte1 = pipe(PipelineType::PIPE_MTE1);
  const std::uint32_t mte2 = pipe(PipelineType::PIPE_MTE2);
  const std::uint32_t mte3 = pipe(PipelineType::PIPE_MTE3);
  const std::uint32_t fix = pipe(PipelineType::PIPE_FIX);
  const bool aicTableIsExact =
      cubeCapabilities.coreDomain == CanonicalSyncCoreDomain::AIC &&
      aicHardware.resourcePairs.size() == 18 &&
      aicExposed.resourcePairs.size() == 14 &&
      aicExposed.supports(matrix, mte1) && aicExposed.supports(mte1, matrix) &&
      aicExposed.supports(mte2, mte3) && aicExposed.supports(mte3, mte2) &&
      aicExposed.supports(fix, matrix) && aicHardware.supports(mte2, fix) &&
      aicHardware.supports(mte3, fix) && aicHardware.supports(fix, mte2) &&
      aicHardware.supports(fix, mte3) && !aicExposed.supports(mte2, fix) &&
      !aicExposed.supports(mte3, fix) && !aicExposed.supports(fix, mte2) &&
      !aicExposed.supports(fix, mte3) && !aicHardware.supports(mte3, matrix) &&
      !aicHardware.supports(pipe(PipelineType::PIPE_S), matrix);
  if (!check(aicTableIsExact,
             "separate exposed AIC events from hardware-only pairs")) {
    return false;
  }

  FailureOr<CanonicalSyncProgram> section = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("section_profile"));
  FailureOr<CanonicalSyncProgram> mixedSection = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("mixed_section_profile"));
  const bool cubeSectionIsAIC =
      succeeded(section) &&
      section->getTargetCapabilities().coreDomain ==
          CanonicalSyncCoreDomain::AIC &&
      section->getTargetCapabilities()
              .directEventCompletion.resourcePairs.size() == 14;
  const bool mixedSectionConflicts =
      succeeded(mixedSection) &&
      mixedSection->getTargetCapabilities().coreDomain ==
          CanonicalSyncCoreDomain::Conflict &&
      mixedSection->getTargetCapabilities()
          .directEventCompletion.resourcePairs.empty();
  if (!check(cubeSectionIsAIC,
             "resolve an explicit cube section as authoritative AIC") ||
      !check(mixedSectionConflicts,
             "reject conflicting explicit section domains")) {
    return false;
  }

  FailureOr<CanonicalSyncProgram> unresolved = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("unresolved_profile"));
  const bool unresolvedHasNoEvents =
      succeeded(unresolved) &&
      unresolved->getTargetCapabilities().coreDomain ==
          CanonicalSyncCoreDomain::Unresolved &&
      unresolved->getTargetCapabilities()
          .directEventCompletion.resourcePairs.empty();
  if (!check(unresolvedHasNoEvents,
             "refuse event authorization without a kernel kind")) {
    return false;
  }

  (*module)->setAttr(
      FunctionKernelKindAttr::name,
      FunctionKernelKindAttr::get(&context, FunctionKernelKind::Cube));
  FailureOr<CanonicalSyncProgram> conflict =
      buildCanonicalSyncProgram(function);
  (*module)->removeAttr(FunctionKernelKindAttr::name);
  const bool conflictHasNoEvents =
      succeeded(conflict) &&
      conflict->getTargetCapabilities().coreDomain ==
          CanonicalSyncCoreDomain::Conflict &&
      conflict->getTargetCapabilities()
          .directEventCompletion.resourcePairs.empty();
  if (!check(conflictHasNoEvents,
             "refuse event authorization for conflicting kernel kinds")) {
    return false;
  }

  (*module)->removeAttr("pto.target_arch");
  FailureOr<CanonicalSyncProgram> missing = buildCanonicalSyncProgram(function);
  const bool missingIsUnsupported =
      succeeded(missing) && missing->getTargetCapabilities().profile ==
                                CanonicalSyncTargetProfile::Unsupported;
  if (!check(missingIsUnsupported,
             "fail closed when the target declaration is missing")) {
    return false;
  }
  (*module)->setAttr("pto.device-spec", StringAttr::get(&context, "Ascend950"));
  FailureOr<CanonicalSyncProgram> deviceSpecified =
      buildCanonicalSyncProgram(function);
  const bool deviceSpecIsA5 =
      succeeded(deviceSpecified) &&
      deviceSpecified->getTargetCapabilities().profile ==
          CanonicalSyncTargetProfile::A5V1;
  if (!check(deviceSpecIsA5,
             "resolve an A5 device specification authoritatively")) {
    return false;
  }
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
  FailureOr<CanonicalSyncProgram> conflicting =
      buildCanonicalSyncProgram(function);
  const bool conflictIsUnsupported =
      succeeded(conflicting) && conflicting->getTargetCapabilities().profile ==
                                    CanonicalSyncTargetProfile::Unsupported;
  if (!check(conflictIsUnsupported,
             "fail closed on conflicting target declarations")) {
    return false;
  }
  (*module)->removeAttr("pto.device-spec");

  (*module)->setAttr("pto.target_arch",
                     IntegerAttr::get(IntegerType::get(&context, 64), 3));
  FailureOr<CanonicalSyncProgram> malformedArch =
      buildCanonicalSyncProgram(function);
  const bool malformedArchIsUnsupported =
      succeeded(malformedArch) &&
      malformedArch->getTargetCapabilities().profile ==
          CanonicalSyncTargetProfile::Unsupported;
  if (!check(malformedArchIsUnsupported,
             "fail closed on a malformed target architecture")) {
    return false;
  }
  (*module)->removeAttr("pto.target_arch");
  (*module)->setAttr("pto.device-spec",
                     IntegerAttr::get(IntegerType::get(&context, 64), 3));
  FailureOr<CanonicalSyncProgram> malformedDevice =
      buildCanonicalSyncProgram(function);
  const bool malformedDeviceIsUnsupported =
      succeeded(malformedDevice) &&
      malformedDevice->getTargetCapabilities().profile ==
          CanonicalSyncTargetProfile::Unsupported;
  if (!check(malformedDeviceIsUnsupported,
             "fail closed on a malformed device specification")) {
    return false;
  }
  (*module)->removeAttr("pto.device-spec");

  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "future"));
  FailureOr<CanonicalSyncProgram> unsupported =
      buildCanonicalSyncProgram(function);
  if (!check(succeeded(unsupported),
             "build a direct-cover graph for an unsupported target")) {
    return false;
  }
  const CanonicalSyncTargetCapabilities &capabilities =
      unsupported->getTargetCapabilities();
  const std::uint32_t vector = pipe(PipelineType::PIPE_V);
  const auto vectorNode = llvm::find_if(
      unsupported->getGraph().getNodes(),
      [&](const SyncCoverNode &node) { return node.resource == vector; });
  return check(
             capabilities.profile == CanonicalSyncTargetProfile::Unsupported &&
                 capabilities.sameResourceCompletionOrdering.version == 0 &&
                 capabilities.targetedBarrierDrainsSourcePrefix.version == 0 &&
                 capabilities.hardwareEventCompletion.version == 0 &&
                 capabilities.directEventCompletion.version == 0 &&
                 !capabilities.directEventCompletesSourcePrefix.isEnabled() &&
                 capabilities.legalPipeBarriers.version == 0 &&
                 capabilities.compilerUsableEventIds.empty() &&
                 capabilities.crossResourceTargetedBarrierCompletion.version ==
                     0 &&
                 !capabilities.targetCompletionResources &&
                 !capabilities.mte1L0ReadySetCompletesPrefix.isEnabled() &&
                 !capabilities.intrinsicMmadAccumulatorOrdering.isEnabled() &&
                 !capabilities.crossPipeAccumulatorReadReadHazard.isEnabled(),
             "default every unsupported-target capability to false") &&
         check(vectorNode != unsupported->getGraph().getNodes().end() &&
                   !vectorNode->completionSignalCoversIssuedPrefix &&
                   vectorNode->completionTargets.empty() &&
                   !unsupported->getGraph().supportsBlockingTargetedBarrier(
                       vector),
               "retain only the exact-operation direct event basis");
}

bool testAccumulatorReadReadHardwareHazardIsRawDemand() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @acc_rar(
          %lhs: !pto.tile_buf<left, 32x32xf32,
                              blayout=row_major, slayout=row_major>,
          %rhs: !pto.tile_buf<right, 32x32xf32,
                              blayout=row_major, slayout=col_major>) attributes {
          pto.kernel_kind = #pto.kernel_kind<cube>} {
        %addr0 = arith.constant 0 : i64
        %addr1 = arith.constant 8192 : i64
        %acc = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>
        %mat = pto.alloc_tile addr = %addr1 :
          !pto.tile_buf<mat, 32x32xf32, blayout=col_major,
                        slayout=row_major>
        pto.tmov ins(%acc :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>)
          outs(%mat : !pto.tile_buf<mat, 32x32xf32,
                                    blayout=col_major, slayout=row_major>)
        pto.tmatmul.acc ins(%acc, %lhs, %rhs :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>,
          !pto.tile_buf<left, 32x32xf32,
                        blayout=row_major, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf32,
                        blayout=row_major, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32,
                                    blayout=col_major, slayout=row_major,
                                    fractal=1024>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse accumulator read/read hardware-hazard fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("acc_rar"));
  if (!check(succeeded(program),
             "build accumulator read/read hardware-hazard graph")) {
    return false;
  }
  const SyncCoverGraph &graph = program->getGraph();
  const auto demand =
      llvm::find_if(graph.getDemands(), [](const SyncCoverDemand &candidate) {
        return llvm::is_contained(candidate.provenanceKinds,
                                  SyncCoverDemandKind::HardwareAccRAR);
      });
  if (!check(demand != graph.getDemands().end(),
             "retain the InsertSync ACC read/read hardware hazard")) {
    return false;
  }
  const SyncCoverOrderingRequirementMask expectedRequirements =
      syncCoverOrderingRequirementBit(
          SyncCoverOrderingRequirement::PipelineCompletionBeforeAccess) |
      syncCoverOrderingRequirementBit(
          SyncCoverOrderingRequirement::HardwareSpecialOrder);
  const bool exactAccumulatorWitness =
      demand->storageWitnesses.size() == 1 &&
      graph.getStorageDomains()
              [graph
                   .getStorageAccesses()
                       [graph
                            .getStorageWitnesses()[demand->storageWitnesses
                                                       .front()]
                            .sourceAccess]
                   .domain]
                  .role == SyncCoverStorageDomainRole::Accumulator;
  FailureOr<CanonicalSyncHazardParityReport> parity =
      compareCanonicalSyncRawHazardsWithInsertSync(*program);
  FailureOr<CanonicalSyncHazardParityReport> boundedParity =
      compareCanonicalSyncRawHazardsWithInsertSync(*program, 0);
  return check(demand->distance == 0 &&
                   graph.getNodes()[demand->source].resource !=
                       graph.getNodes()[demand->target].resource,
               "restrict the ACC read/read rule to cross-pipeline accesses") &&
         check(demand->orderingRequirements == expectedRequirements,
               "type the ACC read/read row as a hardware-special order") &&
         check(exactAccumulatorWitness,
               "retain the exact ACC overlap witness on the raw row") &&
         check(succeeded(parity) && parity->complete &&
                   parity->canonicalOnly.empty() &&
                   parity->insertSyncOnly.empty() &&
                   parity->canonicalRawHazards.find("kind=ACC_RAR") !=
                       std::string::npos &&
                   parity->insertSyncRawHazards.find("kind=ACC_RAR") !=
                       std::string::npos,
               "match InsertSync's flat pre-pruning ACC hazard ledger") &&
         check(succeeded(boundedParity) && !boundedParity->complete &&
                   boundedParity->incompleteReason.find("budget") !=
                       std::string::npos,
               "bound the diagnostic InsertSync parity oracle");
}

bool testMmadIntrinsicRequiresExactAccumulator() {
  MLIRContext exactContext;
  loadDialects(exactContext);
  OwningOpRef<ModuleOp> exactModule =
      parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @exact() {
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 16384 : i64
        %lhs = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<left, 32x32xf16, blayout=row_major, slayout=row_major>
        %rhs = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<right, 32x32xf16, blayout=row_major, slayout=col_major>
        %acc = pto.alloc_tile addr = %a1 :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>
        pto.tmatmul ins(%lhs, %rhs :
          !pto.tile_buf<left, 32x32xf16, blayout=row_major, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf16, blayout=row_major, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                                    slayout=row_major, fractal=1024>)
        pto.tmatmul.acc ins(%acc, %lhs, %rhs :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>,
          !pto.tile_buf<left, 32x32xf16, blayout=row_major, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf16, blayout=row_major, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                                    slayout=row_major, fractal=1024>)
        return
      }
    }
  )mlir",
                                  &exactContext);
  if (!check(static_cast<bool>(exactModule), "parse exact MMAD fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> exact = buildCanonicalSyncProgram(
      exactModule->lookupSymbol<func::FuncOp>("exact"));
  const bool validExact =
      check(succeeded(exact), "build exact MMAD graph") &&
      check(exact->getGraph().getDemands().empty(),
            "accept silicon-proven exact L0C accumulation order") &&
      check(llvm::all_of(exact->getGraph().getNodes(),
                         [](const SyncCoverNode &node) {
                           return node.completionTargets.empty() &&
                                  node.completionDominatedSources.empty();
                         }),
            "do not promote MMAD issue order to completion evidence");
  if (!validExact) {
    return false;
  }
  (*exactModule)
      ->setAttr("pto.target_arch", StringAttr::get(&exactContext, "a2"));
  FailureOr<CanonicalSyncProgram> a2Exact = buildCanonicalSyncProgram(
      exactModule->lookupSymbol<func::FuncOp>("exact"));
  const bool a2Built =
      check(succeeded(a2Exact), "build exact MMAD graph for A2");
  if (!a2Built) {
    return false;
  }
  if (!check(!a2Exact->getGraph().getDemands().empty(),
             "do not apply the A3 MMAD contract to A2")) {
    return false;
  }

  MLIRContext conservativeContext;
  loadDialects(conservativeContext);
  OwningOpRef<ModuleOp> conservativeModule =
      parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @conservative(
          %lhs: !pto.tile_buf<left, 32x32xf16,
                              blayout=row_major, slayout=row_major>,
          %rhs: !pto.tile_buf<right, 32x32xf16,
                              blayout=row_major, slayout=col_major>,
          %acc: !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                              slayout=row_major, fractal=1024>) {
        pto.tmatmul ins(%lhs, %rhs :
          !pto.tile_buf<left, 32x32xf16, blayout=row_major, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf16, blayout=row_major, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                                    slayout=row_major, fractal=1024>)
        pto.tmatmul.acc ins(%acc, %lhs, %rhs :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>,
          !pto.tile_buf<left, 32x32xf16, blayout=row_major, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf16, blayout=row_major, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                                    slayout=row_major, fractal=1024>)
        return
      }
    }
  )mlir",
                                  &conservativeContext);
  if (!check(static_cast<bool>(conservativeModule),
             "parse conservative MMAD fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> conservative = buildCanonicalSyncProgram(
      conservativeModule->lookupSymbol<func::FuncOp>("conservative"));
  return check(succeeded(conservative), "build conservative MMAD graph") &&
         check(!conservative->getGraph().getDemands().empty(),
               "do not apply MMAD intrinsic rule without exact L0C proof");
}

bool testA3TargetCompletionCertificatesAreArchitectureQualified() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @certified(
          %dst: !pto.partition_tensor_view<32x32xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<cube>} {
        %c0 = arith.constant 0 : index
        %a0 = arith.constant 0 : i64
        %a1 = arith.constant 8192 : i64
        %mat_a = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<mat, 64x32xf32, blayout=col_major, slayout=row_major>
        %mat_b = pto.alloc_tile addr = %a1 :
          !pto.tile_buf<mat, 32x64xf32, blayout=col_major, slayout=row_major>
        %left = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<left, 32x32xf32, slayout=row_major>
        %right = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<right, 32x32xf32, slayout=col_major>
        %acc = pto.alloc_tile addr = %a0 :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>
        pto.textract ins(%mat_a, %c0, %c0 :
          !pto.tile_buf<mat, 64x32xf32, blayout=col_major, slayout=row_major>,
          index, index)
          outs(%left : !pto.tile_buf<left, 32x32xf32, slayout=row_major>)
        pto.textract ins(%mat_b, %c0, %c0 :
          !pto.tile_buf<mat, 32x64xf32, blayout=col_major, slayout=row_major>,
          index, index)
          outs(%right : !pto.tile_buf<right, 32x32xf32, slayout=col_major>)
        pto.tmatmul ins(%left, %right :
          !pto.tile_buf<left, 32x32xf32, slayout=row_major>,
          !pto.tile_buf<right, 32x32xf32, slayout=col_major>)
          outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                                    slayout=row_major, fractal=1024>)
        pto.tstore ins(%acc :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>)
          outs(%dst : !pto.partition_tensor_view<32x32xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse target-completion certificate fixture")) {
    return false;
  }
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("certified");
  FailureOr<CanonicalSyncProgram> a3 = buildCanonicalSyncProgram(function);
  if (!check(succeeded(a3), "build A3 target-completion certificates")) {
    return false;
  }
  const auto countKind = [](const CanonicalSyncProgram &program,
                            SyncCoverTargetCompletionKind kind) {
    return llvm::count_if(
        program.getGraph().getTargetCompletionCertificates(),
        [&](const SyncCoverTargetCompletionCertificate &certificate) {
          return certificate.kind == kind;
        });
  };
  const bool a3Qualified =
      check(countKind(*a3, SyncCoverTargetCompletionKind::Mte1L0ReadyPrefix) ==
                1,
            "group two exact A3 L0 producers behind one ready certificate") &&
      check(countKind(
                *a3,
                SyncCoverTargetCompletionKind::MToFixAccumulatorBoundary) == 1,
            "certify the exact A3 accumulator-to-FIX boundary") &&
      check(a3->getGraph().getCompletionCutFacts().size() == 1,
            "publish the A3 accumulator boundary as a target-neutral cut");
  CanonicalSyncAnalysisOptions targetDiscoveryDisabled;
  targetDiscoveryDisabled.discoverTargetCompletionCertificates = false;
  targetDiscoveryDisabled.discoverBasicOwnershipCertificates = false;
  FailureOr<CanonicalSyncProgram> disabled =
      buildCanonicalSyncProgram(function, targetDiscoveryDisabled);
  const bool disabledSkipped =
      check(succeeded(disabled),
            "build A3 graph with target certificate discovery disabled") &&
      check(disabled->getGraph().getTargetCompletionCertificates().empty(),
            "skip disabled target completion certificate discovery") &&
      check(disabled->getGraph().getCompletionCutFacts().empty(),
            "skip disabled provider completion-cut discovery");
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a2"));
  FailureOr<CanonicalSyncProgram> a2 = buildCanonicalSyncProgram(function);
  const bool a2Rejected =
      check(succeeded(a2), "build the same graph for A2") &&
      check(a2->getGraph().getTargetCompletionCertificates().empty(),
            "do not infer A3 target certificates on A2") &&
      check(a2->getGraph().getCompletionCutFacts().empty(),
            "do not infer A3 provider completion cuts on A2");
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a2a3"));
  FailureOr<CanonicalSyncProgram> a2a3 = buildCanonicalSyncProgram(function);
  const bool a2a3Rejected =
      check(succeeded(a2a3), "build the same graph for A2/A3 intersection") &&
      check(
          a2a3->getGraph().getTargetCompletionCertificates().empty(),
          "do not infer A3 target certificates from the A2/A3 intersection") &&
      check(a2a3->getGraph().getCompletionCutFacts().empty(),
            "do not infer A3 provider cuts from the A2/A3 intersection");
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
  FailureOr<CanonicalSyncProgram> a5 = buildCanonicalSyncProgram(function);
  const bool a5Rejected =
      check(succeeded(a5), "build the same graph for A5") &&
      check(a5->getGraph().getTargetCompletionCertificates().empty(),
            "do not infer A3 target certificates on A5") &&
      check(a5->getGraph().getCompletionCutFacts().empty(),
            "do not infer A3 provider completion cuts on A5");
  if (!a3Qualified || !disabledSkipped || !a2Rejected || !a2a3Rejected ||
      !a5Rejected) {
    return false;
  }

  CanonicalSyncBuildOptions a5Options;
  bool sawA5IncompleteCatalog = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> a5Problem = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawA5IncompleteCatalog |=
          diagnostic.str().find("required singleton catalog is incomplete") !=
          std::string::npos;
      return success();
    });
    a5Problem = buildCanonicalSyncSingletonProblem(*a5, a5Options);
  }
  const bool a5MatrixToFixRejected =
      failed(a5Problem) && sawA5IncompleteCatalog;
  if (!check(a5MatrixToFixRejected,
             "reject unsourced A5 matrix-to-FIX event candidates during "
             "catalog construction")) {
    return false;
  }

  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
  if (!check(succeeded(runCanonicalSync(function)),
             "materialize A3 target-completion certificates")) {
    return false;
  }
  std::size_t mte1ToMatrixSets = 0;
  std::size_t mte1ToMatrixWaits = 0;
  std::size_t matrixToFixSets = 0;
  std::size_t matrixToFixWaits = 0;
  std::size_t bodyBarriers = 0;
  function.walk([&](Operation *operation) {
    const bool generated = operation->hasAttr("pto.canonical_sync");
    const auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
    const auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
    if (generated && source && target) {
      const bool mte1ToMatrix = source.getPipe() == PIPE::PIPE_MTE1 &&
                                target.getPipe() == PIPE::PIPE_M;
      const bool matrixToFix = source.getPipe() == PIPE::PIPE_M &&
                               target.getPipe() == PIPE::PIPE_FIX;
      mte1ToMatrixSets += isa<SetFlagOp>(operation) && mte1ToMatrix;
      mte1ToMatrixWaits += isa<WaitFlagOp>(operation) && mte1ToMatrix;
      matrixToFixSets += isa<SetFlagOp>(operation) && matrixToFix;
      matrixToFixWaits += isa<WaitFlagOp>(operation) && matrixToFix;
    }
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      bodyBarriers +=
          generated && !barrier->hasAttr("pto.auto_sync_tail_barrier");
    }
  });
  return check(mte1ToMatrixSets == 1 && mte1ToMatrixWaits == 1,
               "materialize one typed MTE1-to-M certificate event") &&
         check(matrixToFixSets == 1 && matrixToFixWaits == 1,
               "materialize one typed M-to-FIX certificate event") &&
         check(bodyBarriers == 0,
               "materialize certified A3 boundaries without body drains");
}

bool testAnalysisLimitFailsClosed() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @entry(
          %gm: !pto.partition_tensor_view<16x16xf32>,
          %mid: !pto.tile_buf<vec, 16x16xf32>,
          %out: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                  outs(%mid : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%mid : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%out : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse bounded adapter fixture")) {
    return false;
  }
  CanonicalSyncAnalysisOptions options;
  options.maximumPairInspections = 1;
  bool sawLimit = false;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
    sawLimit |= diagnostic.str().find("pair-inspection limit exceeded") !=
                std::string::npos;
    return success();
  });
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("entry"), options);
  return check(failed(program), "fail closed at the analysis work bound") &&
         check(sawLimit, "diagnose the exhausted analysis bound");
}

bool testCanonicalHazardsAggregateBeforeGraphMutation() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @aggregate_access_batches(
          %gm: !pto.partition_tensor_view<1x512xf32>,
          %executed: vector<4xi16>) {
        %base = arith.constant 0 : i64
        %src0 = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x128xf32>
        %src1 = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x128xf32>
        %src2 = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x128xf32>
        %src3 = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x128xf32>
        %tmp = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x512xf32>
        %dst = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x512xf32>
        %target = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 1x512xf32>
        pto.tmrgsort ins(%src0, %src1, %src2, %src3, %tmp
            {exhausted = false} :
            !pto.tile_buf<vec, 1x128xf32>,
            !pto.tile_buf<vec, 1x128xf32>,
            !pto.tile_buf<vec, 1x128xf32>,
            !pto.tile_buf<vec, 1x128xf32>,
            !pto.tile_buf<vec, 1x512xf32>)
          outs(%dst, %executed : !pto.tile_buf<vec, 1x512xf32>,
                                vector<4xi16>)
        pto.tload ins(%gm : !pto.partition_tensor_view<1x512xf32>)
                  outs(%target : !pto.tile_buf<vec, 1x512xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse canonical hazard aggregation fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("aggregate_access_batches"));
  if (!check(succeeded(program), "build aggregated canonical hazards")) {
    return false;
  }
  const std::vector<SyncCoverDemand> &demands =
      program->getGraph().getDemands();
  const bool oneCanonicalDemand = demands.size() == 1;
  if (!check(oneCanonicalDemand,
             "coalesce all access batches into one canonical demand")) {
    return false;
  }
  const SyncCoverDemand &demand = demands.front();
  const bool hasWar = llvm::is_contained(demand.provenanceKinds,
                                         SyncCoverDemandKind::MemoryWAR);
  const bool hasWaw = llvm::is_contained(demand.provenanceKinds,
                                         SyncCoverDemandKind::MemoryWAW);
  return check(hasWar && hasWaw && demand.provenanceKinds.size() == 2,
               "retain every aggregated hazard kind") &&
         check(demand.storageWitnesses.size() == 6,
               "retain every physical witness from the access batches") &&
         check(demand.originalDemandCount == 2,
               "mutate the graph once with complete hazard provenance");
}

bool testFailClosedInputs() {
  constexpr std::string_view malformedNoAlias = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @bad_noalias(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>)
          attributes {pto.noalias_pairs = array<i64: 0>} {
        return
      }
    }
  )mlir";
  constexpr std::string_view manualSync = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @manual_sync() {
        pto.syncall() mode = #pto.sync_all_mode<hard>,
          core_type = #pto.sync_core_type<aiv_only>
        return
      }
    }
  )mlir";
  constexpr std::string_view unmodeledEffect = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func private @opaque()
      func.func @unmodeled() {
        func.call @opaque() : () -> ()
        return
      }
    }
  )mlir";
  constexpr std::string_view resultIf = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @result_if(%condition: i1) {
        %result = scf.if %condition -> (i32) {
          %zero = arith.constant 0 : i32
          scf.yield %zero : i32
        } else {
          %one = arith.constant 1 : i32
          scf.yield %one : i32
        }
        return
      }
    }
  )mlir";
  constexpr std::string_view iterArgs = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @iter_args(%limit: index) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %result = scf.for %i = %zero to %limit step %one
            iter_args(%carried = %zero) -> (index) {
          scf.yield %carried : index
        }
        return
      }
    }
  )mlir";
  constexpr std::string_view asynchronousControl = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @async_control(%condition_ptr: !pto.ptr<i1>) {
        %zero = arith.constant 0 : index
        %condition = pto.load_scalar %condition_ptr[%zero] :
          !pto.ptr<i1> -> i1
        scf.if %condition {
        }
        return
      }
    }
  )mlir";
  constexpr std::string_view hiddenScheduledProducer = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func private @provenance_barrier(i1) -> i1
          attributes {pto.tileop.helper, pto.tileop.kind = "vector"}
      func.func @hidden_producer(%condition_ptr: !pto.ptr<i1>) {
        %zero = arith.constant 0 : index
        %loaded = pto.load_scalar %condition_ptr[%zero] :
          !pto.ptr<i1> -> i1
        %condition = func.call @provenance_barrier(%loaded) : (i1) -> i1
        scf.if %condition {
        }
        return
      }
    }
  )mlir";

  return expectAnalysisFailure(malformedNoAlias, "bad_noalias",
                               "even dense i64 array") &&
         expectAnalysisSuccess(manualSync, "manual_sync") &&
         expectAnalysisFailure(unmodeledEffect, "unmodeled",
                               "unrecognized helper") &&
         expectAnalysisFailure(resultIf, "result_if",
                               "result-carrying scf.if") &&
         expectAnalysisFailure(iterArgs, "iter_args", "scf.for iter_args") &&
         expectAnalysisFailure(asynchronousControl, "async_control",
                               "asynchronous scf.if condition") &&
         expectAnalysisFailure(hiddenScheduledProducer, "hidden_producer",
                               "cannot trace SSA provenance");
}

bool testAcceptsDeclaredStorageProvenanceRoots() {
  constexpr std::string_view declaredTile = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @declared_tile(
          %src: !pto.partition_tensor_view<16x16xf32>) {
        %dst = pto.declare_tile -> !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir";
  constexpr std::string_view declaredGlobal = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @declared_global(%dst: !pto.tile_buf<vec, 16x16xf32>) {
        %src = pto.declare_global -> !pto.tensor_view<16x16xf32>
        %zero = arith.constant 0 : index
        %size = arith.constant 16 : index
        %part = pto.partition_view %src,
          offsets = [%zero, %zero], sizes = [%size, %size]
          : !pto.tensor_view<16x16xf32>
            -> !pto.partition_tensor_view<16x16xf32>
        pto.tload ins(%part : !pto.partition_tensor_view<16x16xf32>)
                  outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir";
  return expectAnalysisSuccess(declaredTile, "declared_tile") &&
         expectAnalysisSuccess(declaredGlobal, "declared_global");
}

bool testRejectsOwnedSyncAndAcceptsFixedFence() {
  constexpr std::string_view source = R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @set_cross() {
        pto.set_cross_block <PIPE_FIX>, 0
        return
      }
      func.func @wait_cross() {
        pto.wait_cross_block <PIPE_MTE3>, 0
        return
      }
      func.func @set_intra() {
        pto.set_intra_block <PIPE_MTE1>, 17
        return
      }
      func.func @wait_intra() {
        pto.wait_intra_block <PIPE_V>, 17
        return
      }
      func.func @fence() {
        pto.fence.barrier_all #pto.fence_scope<gm>
        return
      }
    }
  )mlir";
  constexpr std::string_view diagnostic = "pre-existing pipe synchronization";
  return expectAnalysisFailure(source, "set_cross", diagnostic) &&
         expectAnalysisFailure(source, "wait_cross", diagnostic) &&
         expectAnalysisFailure(source, "set_intra", diagnostic) &&
         expectAnalysisFailure(source, "wait_intra", diagnostic) &&
         expectAnalysisSuccess(source, "fence");
}

bool testRejectsMalformedOwnedSynchronization() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @wrong_type() {
        %zero = arith.constant 0 : index
        return
      }
      func.func @unsupported() {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %product = arith.muli %zero, %one : index
        return
      }
      func.func @partial_tree(%condition: i1) {
        scf.if %condition {
          %zero = arith.constant 0 : index
        }
        return
      }
      func.func @escaping_helper() {
        %zero = arith.constant 0 : index
        %sum = arith.addi %zero, %zero : index
        return
      }
      func.func @preserve_previous(
          %input: !pto.partition_tensor_view<16x16xf32>,
          %output: !pto.partition_tensor_view<16x16xf32>) attributes {
          pto.kernel_kind = #pto.kernel_kind<vector>} {
        %addr = arith.constant 0 : i64
        %tile = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
                  outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%output : !pto.partition_tensor_view<16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse malformed pass-ownership fixtures")) {
    return false;
  }

  Builder builder(&context);
  func::FuncOp wrongType = module->lookupSymbol<func::FuncOp>("wrong_type");
  arith::ConstantOp wrongMarker =
      *wrongType.getBody().front().getOps<arith::ConstantOp>().begin();
  wrongMarker->setAttr("pto.canonical_sync", builder.getStringAttr("invalid"));
  func::FuncOp unsupported = module->lookupSymbol<func::FuncOp>("unsupported");
  arith::MulIOp unsupportedOwned =
      *unsupported.getBody().front().getOps<arith::MulIOp>().begin();
  unsupportedOwned->setAttr("pto.canonical_sync", builder.getUnitAttr());
  func::FuncOp partial = module->lookupSymbol<func::FuncOp>("partial_tree");
  scf::IfOp partialOwned =
      *partial.getBody().front().getOps<scf::IfOp>().begin();
  partialOwned->setAttr("pto.canonical_sync", builder.getUnitAttr());
  func::FuncOp escaping = module->lookupSymbol<func::FuncOp>("escaping_helper");
  arith::ConstantOp escapingOwned =
      *escaping.getBody().front().getOps<arith::ConstantOp>().begin();
  escapingOwned->setAttr("pto.canonical_sync", builder.getUnitAttr());

  const auto expectMalformed = [&](func::FuncOp function,
                                   StringRef expected) -> bool {
    bool sawExpected = false;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawExpected |= diagnostic.str().find(expected.str()) != std::string::npos;
      return success();
    });
    FailureOr<CanonicalSyncProgram> program =
        buildCanonicalSyncProgram(function);
    return check(failed(program), "reject malformed pass-owned IR") &&
           check(sawExpected, "diagnose malformed pass-owned IR");
  };
  const bool malformedRejected =
      expectMalformed(wrongType, "malformed pass-owned") &&
      expectMalformed(unsupported, "malformed pass-owned") &&
      expectMalformed(partial, "malformed pass-owned") &&
      expectMalformed(escaping, "pass-owned helper escapes");
  if (!malformedRejected) {
    return false;
  }

  func::FuncOp preserve =
      module->lookupSymbol<func::FuncOp>("preserve_previous");
  CanonicalSyncBuildOptions options;
  if (!check(succeeded(runCanonicalSync(preserve, options)),
             "materialize a plan before malformed-rerun rejection")) {
    return false;
  }
  TLoadOp malformedOwned =
      *preserve.getBody().front().getOps<TLoadOp>().begin();
  malformedOwned->setAttr("pto.canonical_sync", builder.getUnitAttr());
  const std::string beforeFailure = printOperation(preserve);
  bool failedRerun = false;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    failedRerun = failed(runCanonicalSync(preserve, options));
  }
  return check(failedRerun, "reject a malformed pass-owned rerun") &&
         check(printOperation(preserve) == beforeFailure,
               "preserve the previous plan after malformed ownership");
}

bool testFixedBarriersSupplyCompletionAndRemainUnowned() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<vector>} {
      func.func @fixed(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.barrier <PIPE_MTE2>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @fixed_cross(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %out: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.barrier <PIPE_MTE2>
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%out : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @fixed_branch(
          %condition: i1,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        scf.if %condition {
          pto.barrier <PIPE_MTE2>
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
      func.func @fixed_one_path(
          %condition: i1,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        scf.if %condition {
          pto.barrier <PIPE_MTE2>
        }
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @fixed_before_branch(
          %condition: i1,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.barrier <PIPE_MTE2>
        scf.if %condition {
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        } else {
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
      func.func @fixed_after_join(
          %condition: i1,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        scf.if %condition {
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        } else {
          pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                    outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        }
        pto.barrier <PIPE_MTE2>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @fixed_all(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %out: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.barrier <PIPE_ALL>
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%out : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @fixed_all_multi(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %dst: !pto.partition_tensor_view<16x16xf32>)
          attributes {pto.noalias_pairs = array<i64: 0, 1>} {
        %first_addr = arith.constant 0 : i64
        %second_addr = arith.constant 4096 : i64
        %slot = pto.alloc_tile addr = %first_addr :
          !pto.tile_buf<vec, 16x16xf32>
        %out = pto.alloc_tile addr = %second_addr :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%out : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%dst : !pto.partition_tensor_view<16x16xf32>)
        pto.barrier <PIPE_ALL>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%out : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%dst : !pto.partition_tensor_view<16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse fixed-barrier fixtures")) {
    return false;
  }

  const auto checkFunction = [&](StringRef name, bool expectGuardedSupply,
                                 bool expectMultipleResources = false,
                                 bool expectFixedSupply = true) -> bool {
    func::FuncOp function = module->lookupSymbol<func::FuncOp>(name);
    FailureOr<CanonicalSyncProgram> program =
        buildCanonicalSyncProgram(function);
    if (!check(succeeded(program), "build fixed-barrier completion graph")) {
      return false;
    }
    const SyncCoverGraph &graph = program->getGraph();
    const bool hasFixedSupply =
        llvm::any_of(graph.getEdges(), [&](const SyncCoverEdge &edge) {
          return edge.kind == SyncCoverEdgeKind::CompletionSupply &&
                 (!expectGuardedSupply || !edge.sourceGuard.literals.empty() ||
                  !edge.targetGuard.literals.empty());
        });
    const bool exactTargetedEndpoint =
        name != "fixed" ||
        llvm::any_of(graph.getEdges(), [&](const SyncCoverEdge &edge) {
          return edge.kind == SyncCoverEdgeKind::CompletionSupply &&
                 edge.source == 0 && edge.target == 1 &&
                 graph.getNodes()[edge.source].resource ==
                     static_cast<std::uint32_t>(PipelineType::PIPE_MTE2) &&
                 graph.getNodes()[edge.target].resource ==
                     static_cast<std::uint32_t>(PipelineType::PIPE_MTE2) &&
                 edge.sourceGuard.literals.empty() &&
                 edge.targetGuard.literals.empty();
        });
    std::set<std::uint32_t> fixedSourceResources;
    std::set<std::uint32_t> fixedTargetResources;
    for (const SyncCoverEdge &edge : graph.getEdges()) {
      if (edge.kind != SyncCoverEdgeKind::CompletionSupply) {
        continue;
      }
      fixedSourceResources.insert(graph.getNodes()[edge.source].resource);
      fixedTargetResources.insert(graph.getNodes()[edge.target].resource);
    }
    const bool expectedResources =
        !expectMultipleResources ||
        (fixedSourceResources.size() >= 2 && fixedTargetResources.size() >= 2);
    CanonicalSyncBuildOptions options;
    CanonicalSyncProblemBuildResult precise =
        buildCanonicalSyncPreciseProblem(*program, options);
    const bool baselineComplete =
        precise && precise.problem->getBaselineCoverage().count() ==
                       graph.getDemands().size();
    const bool credited = check(
        !graph.getDemands().empty() && hasFixedSupply == expectFixedSupply &&
            exactTargetedEndpoint && expectedResources,
        "credit only supported fixed-barrier completion supplies");
    const bool covered =
        check(baselineComplete == expectFixedSupply,
              "distinguish fixed coverage from required candidate coverage: " +
                  name.str());
    const bool materialized =
        check(succeeded(runCanonicalSync(function, options)),
              "materialize around a fixed user barrier");
    if (!credited || !covered || !materialized) {
      return false;
    }

    std::size_t userBarriers = 0;
    std::size_t generatedNonTailSync = 0;
    function.walk([&](Operation *operation) {
      const bool generated = operation->hasAttr("pto.canonical_sync");
      if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp>(operation)) {
        generatedNonTailSync += generated;
      }
      if (auto barrier = dyn_cast<BarrierOp>(operation)) {
        if (!generated) {
          ++userBarriers;
        } else if (!barrier->hasAttr("pto.auto_sync_tail_barrier")) {
          ++generatedNonTailSync;
        }
      }
    });
    return check(userBarriers == 1,
                 "preserve the unowned fixed barrier exactly once") &&
           check(expectFixedSupply ? generatedNonTailSync == 0
                                   : generatedNonTailSync != 0,
                 "materialize synchronization only when fixed supply is "
                 "insufficient");
  };

  const bool fixedCasesCovered =
      checkFunction("fixed", false) &&
      checkFunction("fixed_cross", false, false, false) &&
      checkFunction("fixed_branch", true) &&
      checkFunction("fixed_before_branch", true) &&
      checkFunction("fixed_after_join", true) &&
      checkFunction("fixed_all", false) &&
      checkFunction("fixed_all_multi", false, true);
  if (!fixedCasesCovered) {
    return false;
  }

  func::FuncOp onePath = module->lookupSymbol<func::FuncOp>("fixed_one_path");
  FailureOr<CanonicalSyncProgram> onePathProgram =
      buildCanonicalSyncProgram(onePath);
  if (!check(succeeded(onePathProgram),
             "build one-path fixed-barrier completion graph")) {
    return false;
  }
  const SyncCoverGraph &onePathGraph = onePathProgram->getGraph();
  const bool hasGuardedFixedSupply =
      llvm::any_of(onePathGraph.getEdges(), [](const SyncCoverEdge &edge) {
        return edge.kind == SyncCoverEdgeKind::CompletionSupply &&
               (!edge.sourceGuard.literals.empty() ||
                !edge.targetGuard.literals.empty());
      });
  CanonicalSyncBuildOptions onePathOptions;
  CanonicalSyncProblemBuildResult onePathPrecise =
      buildCanonicalSyncPreciseProblem(*onePathProgram, onePathOptions);
  const bool missingPathRemainsUncovered =
      onePathPrecise && onePathPrecise.problem->getBaselineCoverage().count() <
                            onePathGraph.getDemands().size();
  if (!check(hasGuardedFixedSupply && missingPathRemainsUncovered,
             "keep a one-branch barrier from covering the missing path") ||
      !check(succeeded(runCanonicalSync(onePath, onePathOptions)),
             "synchronize the path not covered by a conditional barrier")) {
    return false;
  }
  std::size_t generatedNonTailSync = 0;
  onePath.walk([&](Operation *operation) {
    const bool generated = operation->hasAttr("pto.canonical_sync");
    generatedNonTailSync +=
        generated &&
        isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp>(operation);
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      generatedNonTailSync +=
          generated && !barrier->hasAttr("pto.auto_sync_tail_barrier");
    }
  });
  return check(generatedNonTailSync != 0,
               "materialize synchronization for the missing branch path");
}

bool testRejectsFixedBarrierInsideLoop() {
  constexpr std::string_view source = R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @loop_barrier(%limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %limit step %c1 {
          pto.barrier <PIPE_MTE2>
        }
        return
      }
    }
  )mlir";
  return expectAnalysisFailure(source, "loop_barrier",
                               "fixed pipe barriers inside loops");
}

bool testFixedBarrierInspectionBoundsAndPersistentControlState() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @fixed_limit(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>,
          %first_slot: !pto.tile_buf<vec, 16x16xf32>,
          %second_slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%first : !pto.partition_tensor_view<16x16xf32>)
                  outs(%first_slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.barrier <PIPE_MTE2>
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
                  outs(%second_slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse fixed-barrier inspection-bound fixture")) {
    return false;
  }
  func::FuncOp fixed = module->lookupSymbol<func::FuncOp>("fixed_limit");
  std::size_t lowerBound = 1;
  std::size_t upperBound = 1U << 18;
  while (lowerBound < upperBound) {
    const std::size_t middle = lowerBound + (upperBound - lowerBound) / 2;
    CanonicalSyncAnalysisOptions options;
    options.maximumPairInspections = middle;
    FailureOr<CanonicalSyncProgram> trial = failure();
    {
      ScopedDiagnosticHandler handler(&context,
                                      [](Diagnostic &) { return success(); });
      trial = buildCanonicalSyncProgram(fixed, options);
    }
    if (succeeded(trial)) {
      upperBound = middle;
    } else {
      lowerBound = middle + 1;
    }
  }
  CanonicalSyncBuildOptions exactOptions;
  exactOptions.analysis.maximumPairInspections = lowerBound;
  if (!check(succeeded(runCanonicalSync(fixed, exactOptions)),
             "complete fixed-barrier analysis at its exact work bound")) {
    return false;
  }
  const std::string materialized = printOperation(fixed);
  CanonicalSyncBuildOptions belowOptions = exactOptions;
  --belowOptions.analysis.maximumPairInspections;
  bool failedBelow = false;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    failedBelow = failed(runCanonicalSync(fixed, belowOptions));
  }
  if (!check(failedBelow,
             "reject fixed-barrier analysis one work unit below its bound") ||
      !check(printOperation(fixed) == materialized,
             "retain generated IR after fixed-barrier bound exhaustion")) {
    return false;
  }

  constexpr std::size_t prefixNodes = 64;
  constexpr std::size_t controlDepth = 128;
  std::string deepSource = R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @deep(%condition: i1, %source: !pto.ptr<i32>) {
        %zero = arith.constant 0 : index
  )mlir";
  for (std::size_t index = 0; index < prefixNodes; ++index) {
    deepSource += "        %value" + std::to_string(index) +
                  " = pto.load_scalar %source[%zero] : !pto.ptr<i32> -> i32\n";
  }
  for (std::size_t depth = 0; depth < controlDepth; ++depth) {
    deepSource += "        scf.if %condition {\n";
  }
  for (std::size_t depth = 0; depth < controlDepth; ++depth) {
    deepSource += "        }\n";
  }
  deepSource += R"mlir(
        return
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> deepModule =
      parseSourceString<ModuleOp>(deepSource, &context);
  if (!check(static_cast<bool>(deepModule),
             "parse deep persistent-issue-state fixture")) {
    return false;
  }
  func::FuncOp deep = deepModule->lookupSymbol<func::FuncOp>("deep");
  std::size_t deepLowerBound = 1;
  std::size_t deepUpperBound = 1U << 18;
  while (deepLowerBound < deepUpperBound) {
    const std::size_t middle =
        deepLowerBound + (deepUpperBound - deepLowerBound) / 2;
    CanonicalSyncAnalysisOptions options;
    options.maximumPairInspections = middle;
    FailureOr<CanonicalSyncProgram> trial = failure();
    {
      ScopedDiagnosticHandler handler(&context,
                                      [](Diagnostic &) { return success(); });
      trial = buildCanonicalSyncProgram(deep, options);
    }
    if (succeeded(trial)) {
      deepUpperBound = middle;
    } else {
      deepLowerBound = middle + 1;
    }
  }
  CanonicalSyncAnalysisOptions deepExact;
  deepExact.maximumPairInspections = deepLowerBound;
  FailureOr<CanonicalSyncProgram> exact =
      buildCanonicalSyncProgram(deep, deepExact);
  CanonicalSyncAnalysisOptions deepBelow = deepExact;
  --deepBelow.maximumPairInspections;
  FailureOr<CanonicalSyncProgram> below = failure();
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    below = buildCanonicalSyncProgram(deep, deepBelow);
  }
  if (!check(succeeded(exact),
             "bound deep no-barrier state and control provenance") ||
      !check(failed(below),
             "account every persistent no-barrier issue-state update")) {
    return false;
  }

  constexpr std::size_t shallowGuardDepth = 1;
  constexpr std::size_t deepGuardDepth = 32;
  constexpr std::size_t barrierCount = 16;
  const auto makeGuardedBarrierFixture = [](StringRef name,
                                            std::size_t guardDepth) {
    std::string result = R"mlir(
      module attributes {pto.target_arch = "a5"} {
        func.func @)mlir";
    result += name.str();
    result += R"mlir((
            %condition: i1,
            %first: !pto.partition_tensor_view<16x16xf32>,
            %second: !pto.partition_tensor_view<16x16xf32>,
            %first_slot: !pto.tile_buf<vec, 16x16xf32>,
            %second_slot: !pto.tile_buf<vec, 16x16xf32>) {
)mlir";
    for (std::size_t depth = 0; depth < guardDepth; ++depth) {
      result += "          scf.if %condition {\n";
    }
    result += R"mlir(
            pto.tload ins(%first : !pto.partition_tensor_view<16x16xf32>)
                      outs(%first_slot : !pto.tile_buf<vec, 16x16xf32>)
)mlir";
    for (std::size_t barrier = 0; barrier < barrierCount; ++barrier) {
      result += "            pto.barrier <PIPE_MTE2>\n";
    }
    result += R"mlir(
            pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
                      outs(%second_slot : !pto.tile_buf<vec, 16x16xf32>)
)mlir";
    for (std::size_t depth = 0; depth < guardDepth; ++depth) {
      result += "          }\n";
    }
    result += R"mlir(
          return
        }
      }
)mlir";
    return result;
  };
  OwningOpRef<ModuleOp> guardedModule = parseSourceString<ModuleOp>(
      makeGuardedBarrierFixture("guarded_shallow", shallowGuardDepth),
      &context);
  OwningOpRef<ModuleOp> deeplyGuardedModule = parseSourceString<ModuleOp>(
      makeGuardedBarrierFixture("guarded_deep", deepGuardDepth), &context);
  const bool parsedGuardedModules = static_cast<bool>(guardedModule) &&
                                    static_cast<bool>(deeplyGuardedModule);
  if (!check(parsedGuardedModules,
             "parse guarded fixed-barrier accounting fixtures")) {
    return false;
  }
  const auto minimumAcceptedBound = [&](func::FuncOp function) {
    std::size_t lower = 1;
    std::size_t upper = 1U << 18;
    {
      CanonicalSyncAnalysisOptions options;
      options.maximumPairInspections = upper;
      if (failed(buildCanonicalSyncProgram(function, options))) {
        return std::size_t{0};
      }
    }
    while (lower < upper) {
      const std::size_t middle = lower + (upper - lower) / 2;
      CanonicalSyncAnalysisOptions options;
      options.maximumPairInspections = middle;
      FailureOr<CanonicalSyncProgram> trial = failure();
      {
        ScopedDiagnosticHandler handler(&context,
                                        [](Diagnostic &) { return success(); });
        trial = buildCanonicalSyncProgram(function, options);
      }
      if (succeeded(trial)) {
        upper = middle;
      } else {
        lower = middle + 1;
      }
    }
    return lower;
  };
  func::FuncOp shallow =
      guardedModule->lookupSymbol<func::FuncOp>("guarded_shallow");
  func::FuncOp deeplyGuarded =
      deeplyGuardedModule->lookupSymbol<func::FuncOp>("guarded_deep");
  const std::size_t shallowMinimum = minimumAcceptedBound(shallow);
  const std::size_t deepMinimum = minimumAcceptedBound(deeplyGuarded);
  if (!check(shallowMinimum != 0 && deepMinimum != 0,
             "find guarded fixed-barrier accounting bounds")) {
    return false;
  }
  const std::size_t chargedGuardMetadata =
      3 * barrierCount * (deepGuardDepth - shallowGuardDepth);
  const bool guardMetadataCharged =
      deepMinimum >= shallowMinimum &&
      deepMinimum - shallowMinimum >= chargedGuardMetadata;
  CanonicalSyncAnalysisOptions deepGuardExact;
  deepGuardExact.maximumPairInspections = deepMinimum;
  CanonicalSyncAnalysisOptions deepGuardBelow = deepGuardExact;
  --deepGuardBelow.maximumPairInspections;
  FailureOr<CanonicalSyncProgram> deepGuardExactResult =
      buildCanonicalSyncProgram(deeplyGuarded, deepGuardExact);
  FailureOr<CanonicalSyncProgram> deepGuardBelowResult = failure();
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    deepGuardBelowResult =
        buildCanonicalSyncProgram(deeplyGuarded, deepGuardBelow);
  }
  return check(
             guardMetadataCharged,
             "charge shared guard metadata before retaining and copying it") &&
         check(succeeded(deepGuardExactResult),
               "accept deeply guarded barriers at the exact work bound") &&
         check(failed(deepGuardBelowResult),
               "reject deeply guarded barriers one unit below the bound");
}

bool testStructuralLimitsFailClosed() {
  constexpr std::string_view basic = R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @bounded(
          %gm: !pto.partition_tensor_view<16x16xf32>,
          %mid: !pto.tile_buf<vec, 16x16xf32>,
          %out: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
                  outs(%mid : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%mid : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%out : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir";
  CanonicalSyncAnalysisOptions nodeLimit;
  nodeLimit.maximumNodes = 1;
  CanonicalSyncAnalysisOptions storageLimit;
  storageLimit.maximumStorageAccesses = 1;
  CanonicalSyncAnalysisOptions conflictLimit;
  conflictLimit.maximumStorageConflictEdges = 0;
  const bool basicLimits =
      expectAnalysisFailure(basic, "bounded", "node limit exceeded",
                            nodeLimit) &&
      expectAnalysisFailure(basic, "bounded", "storage-access limit exceeded",
                            storageLimit) &&
      expectAnalysisFailure(basic, "bounded",
                            "storage-conflict edge limit exceeded",
                            conflictLimit);
  if (!basicLimits) {
    return false;
  }

  constexpr std::string_view scope = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @bounded_scope(%limit: index) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %i = %zero to %limit step %one {
        }
        return
      }
    }
  )mlir";
  CanonicalSyncAnalysisOptions scopeLimit;
  scopeLimit.maximumScopes = 1;
  if (!expectAnalysisFailure(scope, "bounded_scope", "scope limit exceeded",
                             scopeLimit)) {
    return false;
  }

  constexpr std::string_view controls = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @bounded_controls(%condition: i1) {
        scf.if %condition {
        }
        scf.if %condition {
        }
        return
      }
    }
  )mlir";
  CanonicalSyncAnalysisOptions controlLimit;
  controlLimit.maximumControls = 1;
  return expectAnalysisFailure(controls, "bounded_controls",
                               "control limit exceeded", controlLimit);
}

bool testPeriodicBranchEvidence() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @periodic(%limit: index,
                          %a: !pto.tile_buf<vec, 16x16xf32>,
                          %b: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c0 : index
          scf.if %even {
            pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          } else {
            pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%a : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @negative_unsigned(%limit: index,
                                   %a: !pto.tile_buf<vec, 16x16xf32>,
                                   %b: !pto.tile_buf<vec, 16x16xf32>) {
        %cm1 = arith.constant -1 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %cm1 to %limit step %c1 {
          %phase = arith.remui %i, %c2 : index
          %selected = arith.cmpi eq, %phase, %c1 : index
          scf.if %selected {
            pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @negative(%limit: index,
                          %a: !pto.tile_buf<vec, 16x16xf32>,
                          %b: !pto.tile_buf<vec, 16x16xf32>) {
        %cm1 = arith.constant -1 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %cm1 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c1 : index
          scf.if %even {
            pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse periodic branch fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("periodic"));
  if (!check(succeeded(program), "build periodic branch graph")) {
    return false;
  }
  const std::vector<SyncCoverControl> &controls =
      program->getGraph().getControls();
  const bool hasPeriodicControl =
      check(controls.size() == 1 && controls[0].phaseRelation.has_value(),
            "recover one periodic control relation");
  if (!hasPeriodicControl) {
    return false;
  }
  const SyncCoverControlPhaseRelation &relation = *controls[0].phaseRelation;
  const bool validRelation =
      check(relation.initialPhase == 0, "recover initial phase") &&
      check(relation.nextPhase == std::vector<std::size_t>({1, 0}),
            "recover modulo phase transition") &&
      check(relation.activeAlternative == std::vector<unsigned>({0, 1}),
            "recover phase-to-alternative mapping");
  if (!validRelation) {
    return false;
  }
  const auto rejectsUnsupportedPeriodic = [&](StringRef name) {
    bool sawDiagnostic = false;
    FailureOr<CanonicalSyncProgram> rejected = failure();
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        sawDiagnostic |=
            diagnostic.str().find("cannot model this periodic loop control") !=
            std::string::npos;
        return success();
      });
      rejected =
          buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>(name));
    }
    return failed(rejected) && sawDiagnostic;
  };
  return check(rejectsUnsupportedPeriodic("negative"),
               "fail closed for signed remainder with a negative lower") &&
         check(rejectsUnsupportedPeriodic("negative_unsigned"),
               "fail closed for unsigned remainder with a negative lower");
}

bool testPhaseAwareRecurrenceDistances() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @same_parity(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c0 : index
          scf.if %even {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @opposite_parity(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c0 : index
          scf.if %even {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          } else {
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @unreachable_phase(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %limit step %c2 {
          %phase = arith.remsi %i, %c4 : index
          %selected = arith.cmpi eq, %phase, %c1 : index
          scf.if %selected {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @phase_restricted_slots(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %base = arith.constant 0 : i64
        %buffer = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=4>
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c0 : index
          %slot_index = arith.remui %i, %c4 : index
          %slot = pto.multi_tile_get %buffer[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=4>
             -> !pto.tile_buf<vec, 16x16xf32>
          scf.if %even {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @multiple_successor_gaps(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c3 : index
          %active = arith.cmpi ne, %phase, %c0 : index
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @period_sixteen(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c16 : index
          %active = arith.cmpi eq, %phase, %c0 : index
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @joint_controls(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase3 = arith.remsi %i, %c3 : index
          %active3 = arith.cmpi eq, %phase3, %c0 : index
          scf.if %active3 {
            %phase4 = arith.remsi %i, %c4 : index
            %active4 = arith.cmpi eq, %phase4, %c0 : index
            scf.if %active4 {
              pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                        outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
              pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                       outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
            }
          }
        }
        return
      }
      func.func @joint_phase_slot_horizon(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c15 = arith.constant 15 : index
        %c16 = arith.constant 16 : index
        %base = arith.constant 0 : i64
        %buffer = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=16>
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c15 : index
          %active = arith.cmpi eq, %phase, %c0 : index
          %slot_index = arith.remui %i, %c16 : index
          %slot = pto.multi_tile_get %buffer[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=16>
             -> !pto.tile_buf<vec, 16x16xf32>
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @phase_slot_successor_gaps(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %base = arith.constant 0 : i64
        %buffer = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c4 : index
          %active = arith.cmpi ne, %phase, %c0 : index
          %slot_index = arith.remui %i, %c2 : index
          %slot = pto.multi_tile_get %buffer[%slot_index]
              : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
             -> !pto.tile_buf<vec, 16x16xf32>
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @unsupported_periodic(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c17 = arith.constant 17 : index
        scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c17 : index
          %active = arith.cmpi eq, %phase, %c0 : index
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @same_parity_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @multiple_successor_gaps_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>,
          %dst: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @phase_slot_successor_gaps_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %base = arith.constant 0 : i64
        %buffer = pto.alloc_multi_tile addr = %base
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
        %slot0 = pto.multi_tile_get %buffer[%c0]
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
           -> !pto.tile_buf<vec, 16x16xf32>
        %slot1 = pto.multi_tile_get %buffer[%c1]
            : !pto.multi_tile_buf<vec, 16x16xf32, count=2>
           -> !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot0 : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse phase-aware recurrence fixture")) {
    return false;
  }

  FailureOr<CanonicalSyncProgram> same = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("same_parity"));
  FailureOr<CanonicalSyncProgram> opposite = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("opposite_parity"));
  FailureOr<CanonicalSyncProgram> unreachable = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("unreachable_phase"));
  FailureOr<CanonicalSyncProgram> phaseRestrictedSlots =
      buildCanonicalSyncProgram(
          module->lookupSymbol<func::FuncOp>("phase_restricted_slots"));
  FailureOr<CanonicalSyncProgram> multipleGaps = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("multiple_successor_gaps"));
  FailureOr<CanonicalSyncProgram> periodSixteen = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("period_sixteen"));
  FailureOr<CanonicalSyncProgram> jointControls = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("joint_controls"));
  FailureOr<CanonicalSyncProgram> jointHorizon = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("joint_phase_slot_horizon"));
  FailureOr<CanonicalSyncProgram> phaseSlotGaps = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("phase_slot_successor_gaps"));
  FailureOr<CanonicalSyncProgram> unrolled = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("same_parity_unrolled"));
  FailureOr<CanonicalSyncProgram> multipleGapsUnrolled =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>(
          "multiple_successor_gaps_unrolled"));
  FailureOr<CanonicalSyncProgram> phaseSlotGapsUnrolled =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>(
          "phase_slot_successor_gaps_unrolled"));
  const bool allBuilt =
      succeeded(same) && succeeded(opposite) && succeeded(unreachable) &&
      succeeded(phaseRestrictedSlots) && succeeded(multipleGaps) &&
      succeeded(periodSixteen) && succeeded(jointControls) &&
      succeeded(jointHorizon) && succeeded(phaseSlotGaps) &&
      succeeded(unrolled) && succeeded(multipleGapsUnrolled) &&
      succeeded(phaseSlotGapsUnrolled);
  if (!check(allBuilt, "build phase-aware recurrence fixtures")) {
    return false;
  }

  const auto hasDistance = [](const CanonicalSyncProgram &program,
                              unsigned distance) {
    return llvm::any_of(program.getGraph().getDemands(),
                        [&](const SyncCoverDemand &demand) {
                          return demand.distance == distance;
                        });
  };
  const bool basicChecks =
      check(!hasDistance(*same, 1),
            "exclude phantom same-parity distance-one recurrence") &&
      check(hasDistance(*same, 2),
            "retain real same-parity distance-two recurrence") &&
      check(hasDistance(*opposite, 1),
            "retain opposite-parity distance-one recurrence") &&
      check(hasDistance(*multipleGaps, 1) && hasDistance(*multipleGaps, 2),
            "retain both reachable successor gaps for one witness") &&
      check(hasDistance(*periodSixteen, 16),
            "accept and model the exact sixteen-state boundary") &&
      check(hasDistance(*jointControls, 12),
            "correlate coprime periodic controls in one joint orbit") &&
      check(hasDistance(*jointHorizon, 240),
            "inspect the complete phase-by-slot recurrence horizon") &&
      check(!hasDistance(*phaseSlotGaps, 1) && hasDistance(*phaseSlotGaps, 2) &&
                hasDistance(*phaseSlotGaps, 4),
            "retain distinct phase-to-slot successor distances") &&
      check(llvm::none_of(unreachable->getGraph().getDemands(),
                          [](const SyncCoverDemand &demand) {
                            return demand.distance != 0;
                          }),
            "exclude recurrences in an unreachable modulo alternative");
  if (!basicChecks) {
    return false;
  }
  const auto witnessOrdinalsAtDistance = [](const CanonicalSyncProgram &program,
                                            unsigned distance) {
    std::set<std::pair<std::uint32_t, std::uint32_t>> result;
    const SyncCoverGraph &graph = program.getGraph();
    for (const SyncCoverDemand &demand : graph.getDemands()) {
      if (demand.distance != distance) {
        continue;
      }
      for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
        const SyncCoverStorageWitness &witness =
            graph.getStorageWitnesses()[witnessId];
        const SyncCoverStorageAccess &sourceAccess =
            graph.getStorageAccesses()[witness.sourceAccess];
        const SyncCoverStorageAccess &targetAccess =
            graph.getStorageAccesses()[witness.targetAccess];
        if (sourceAccess.addressOrdinal && targetAccess.addressOrdinal) {
          result.insert(
              {*sourceAccess.addressOrdinal, *targetAccess.addressOrdinal});
        }
      }
    }
    return result;
  };
  const std::set<std::pair<std::uint32_t, std::uint32_t>> slotOne = {{1, 1}};
  const std::set<std::pair<std::uint32_t, std::uint32_t>> slotZero = {{0, 0}};
  const bool witnessesPreserveSourcePhases =
      witnessOrdinalsAtDistance(*phaseSlotGaps, 2) == slotOne &&
      witnessOrdinalsAtDistance(*phaseSlotGaps, 4) == slotZero;
  if (!check(witnessesPreserveSourcePhases,
             "preserve the source-phase mask of each physical witness")) {
    return false;
  }

  bool sawUnsupportedPeriodic = false;
  FailureOr<CanonicalSyncProgram> unsupportedPeriodic = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawUnsupportedPeriodic |=
          diagnostic.str().find("exceeds the supported phase bound") !=
          std::string::npos;
      return success();
    });
    unsupportedPeriodic = buildCanonicalSyncProgram(
        module->lookupSymbol<func::FuncOp>("unsupported_periodic"));
  }
  const bool unsupportedPeriodicRejected =
      check(failed(unsupportedPeriodic) && sawUnsupportedPeriodic,
            "fail closed for a recognized periodic control above the hard "
            "bound");
  if (!unsupportedPeriodicRejected) {
    return false;
  }

  CanonicalSyncAnalysisOptions phaseLimit;
  phaseLimit.maximumPeriodicRecurrenceStates = 1;
  bool sawPhaseLimit = false;
  FailureOr<CanonicalSyncProgram> limited = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawPhaseLimit |=
          diagnostic.str().find("periodic recurrence state limit exceeded") !=
          std::string::npos;
      return success();
    });
    limited = buildCanonicalSyncProgram(
        module->lookupSymbol<func::FuncOp>("same_parity"), phaseLimit);
  }
  const bool limitRejected = failed(limited) && sawPhaseLimit;
  if (!check(limitRejected,
             "fail closed at the periodic recurrence state bound")) {
    return false;
  }

  CanonicalSyncAnalysisOptions witnessLimit;
  witnessLimit.maximumRecurrenceWitnessStates = 1;
  bool sawWitnessLimit = false;
  FailureOr<CanonicalSyncProgram> witnessLimited = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawWitnessLimit |=
          diagnostic.str().find("recurrence witness-state limit exceeded") !=
          std::string::npos;
      return success();
    });
    witnessLimited = buildCanonicalSyncProgram(
        module->lookupSymbol<func::FuncOp>("phase_restricted_slots"),
        witnessLimit);
  }
  const bool witnessLimitRejected =
      check(failed(witnessLimited) && sawWitnessLimit,
            "fail closed before recurrence witness-state storage exceeds "
            "its bound");
  if (!witnessLimitRejected) {
    return false;
  }

  std::size_t lowerWorkBound = 1;
  std::size_t upperWorkBound =
      CanonicalSyncAnalysisOptions{}.maximumPairInspections;
  func::FuncOp jointHorizonFunction =
      module->lookupSymbol<func::FuncOp>("joint_phase_slot_horizon");
  while (lowerWorkBound < upperWorkBound) {
    const std::size_t middle =
        lowerWorkBound + (upperWorkBound - lowerWorkBound) / 2;
    CanonicalSyncAnalysisOptions options;
    options.maximumPairInspections = middle;
    FailureOr<CanonicalSyncProgram> trial = failure();
    {
      ScopedDiagnosticHandler handler(&context,
                                      [](Diagnostic &) { return success(); });
      trial = buildCanonicalSyncProgram(jointHorizonFunction, options);
    }
    if (succeeded(trial)) {
      upperWorkBound = middle;
    } else {
      lowerWorkBound = middle + 1;
    }
  }
  CanonicalSyncAnalysisOptions exactWork;
  exactWork.maximumPairInspections = lowerWorkBound;
  CanonicalSyncAnalysisOptions belowExactWork = exactWork;
  --belowExactWork.maximumPairInspections;
  FailureOr<CanonicalSyncProgram> exactWorkResult =
      buildCanonicalSyncProgram(jointHorizonFunction, exactWork);
  FailureOr<CanonicalSyncProgram> belowExactWorkResult = failure();
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    belowExactWorkResult =
        buildCanonicalSyncProgram(jointHorizonFunction, belowExactWork);
  }
  const bool exactWorkBounded =
      check(succeeded(exactWorkResult) && failed(belowExactWorkResult),
            "bound all joint-phase and slot-residue work exactly");
  if (!exactWorkBounded) {
    return false;
  }

  const SyncCoverGraph &slotGraph = phaseRestrictedSlots->getGraph();
  const bool slotDistances =
      check(llvm::none_of(slotGraph.getDemands(),
                          [](const SyncCoverDemand &demand) {
                            return demand.distance > 0 && demand.distance < 4;
                          }),
            "exclude pre-reuse distances for phase-restricted slots") &&
      check(hasDistance(*phaseRestrictedSlots, 4),
            "retain the first real phase-restricted slot reuse");
  const bool onlyReachableSlotWitnesses =
      llvm::all_of(slotGraph.getDemands(), [&](const SyncCoverDemand &demand) {
        if (demand.distance == 0) {
          return true;
        }
        return llvm::all_of(
            demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
              const SyncCoverStorageWitness &witness =
                  slotGraph.getStorageWitnesses()[witnessId];
              const SyncCoverStorageAccess &sourceAccess =
                  slotGraph.getStorageAccesses()[witness.sourceAccess];
              const SyncCoverStorageAccess &targetAccess =
                  slotGraph.getStorageAccesses()[witness.targetAccess];
              return sourceAccess.addressOrdinal &&
                     targetAccess.addressOrdinal &&
                     *sourceAccess.addressOrdinal % 2 == 0 &&
                     *targetAccess.addressOrdinal % 2 == 0;
            });
      });
  if (!slotDistances ||
      !check(onlyReachableSlotWitnesses,
             "retain only physically reachable periodic slot witnesses")) {
    return false;
  }

  using RecurrenceKey =
      std::tuple<unsigned, unsigned, SyncCoverDemandKind, unsigned,
                 AddressSpace, std::uint64_t, std::uint64_t, std::uint64_t,
                 std::uint64_t, SyncCoverStorageAccessMode,
                 SyncCoverStorageAccessMode>;
  struct ObligationSnapshot {
    std::map<RecurrenceKey, unsigned> minimumDistances;
    std::map<RecurrenceKey, bool> covered;
  };
  const auto operationRole =
      [](Operation *operation) -> std::optional<unsigned> {
    if (isa<TLoadOp>(operation)) {
      return 0;
    }
    if (isa<TAbsOp>(operation)) {
      return 1;
    }
    return std::nullopt;
  };
  const auto buildIterationMap = [&](const CanonicalSyncProgram &program,
                                     ArrayRef<unsigned> iterations) {
    std::map<Operation *, unsigned> result;
    std::array<std::size_t, 2> occurrences{};
    for (const CanonicalSyncNodeBinding &binding : program.getNodeBindings()) {
      const bool alreadyRecorded =
          binding.operation && result.count(binding.operation) != 0;
      if (!binding.operation || alreadyRecorded) {
        continue;
      }
      const std::optional<unsigned> role = operationRole(binding.operation);
      if (!role || occurrences[*role] >= iterations.size()) {
        continue;
      }
      result[binding.operation] = iterations[occurrences[*role]++];
    }
    return result;
  };
  const auto oracleCoverage = [&](const CanonicalSyncProgram &program)
      -> std::optional<SyncCoverDemandSet> {
    const SyncCoverGraph &graph = program.getGraph();
    SyncCoverExpandedProgram expansion(graph);
    SyncCoverDemandSet result(graph.getDemands().size());
    for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
      SyncCoverEdge edge;
      edge.source = demand.source;
      edge.target = demand.target;
      edge.kind = SyncCoverEdgeKind::CompletionSupply;
      edge.scope = demand.scope;
      edge.distance = demand.distance;
      edge.sourceGuard = demand.sourceGuard;
      edge.targetGuard = demand.targetGuard;
      SyncCoverCompletionSupply supply;
      supply.mechanism = demandId;
      supply.edge = std::move(edge);
      supply.allowedDemands = {demandId};
      const SyncCoverCoverageResult coverage = computeSyncCoverCoverage(
          graph, expansion, {std::move(supply)}, {demandId});
      if (!coverage) {
        return std::nullopt;
      }
      if (coverage.covered.contains(demandId)) {
        result.insert(demandId);
      }
    }
    return result;
  };
  using WitnessPhaseFilter =
      std::function<bool(const SyncCoverStorageAccess &,
                         const SyncCoverStorageAccess &, unsigned, unsigned)>;
  const auto collectSnapshot = [&](const CanonicalSyncProgram &program,
                                   unsigned period,
                                   const std::map<Operation *, unsigned>
                                       *unrolledIterations,
                                   const SyncCoverDemandSet &covered,
                                   const WitnessPhaseFilter &phaseFilter) {
    ObligationSnapshot result;
    const SyncCoverGraph &graph = program.getGraph();
    const auto guardMatches = [&](const SyncCoverGuard &guard,
                                  SyncCoverScopeId loopScope,
                                  unsigned iteration) {
      for (const SyncCoverGuardLiteral &literal : guard.literals) {
        const SyncCoverControl &control = graph.getControls()[literal.control];
        if (!control.phaseRelation ||
            control.phaseRelation->loopScope != loopScope) {
          continue;
        }
        std::size_t state = control.phaseRelation->initialPhase;
        for (unsigned step = 0; step < iteration; ++step) {
          state = control.phaseRelation->nextPhase[state];
        }
        if (control.phaseRelation->activeAlternative[state] !=
            literal.alternative) {
          return false;
        }
      }
      return true;
    };
    const auto supportsKind = [](const SyncCoverStorageAccess &sourceAccess,
                                 const SyncCoverStorageAccess &targetAccess,
                                 SyncCoverDemandKind kind) {
      switch (kind) {
      case SyncCoverDemandKind::MemoryRAW:
        return syncCoverStorageModeWrites(sourceAccess.mode) &&
               syncCoverStorageModeReads(targetAccess.mode);
      case SyncCoverDemandKind::MemoryWAR:
        return syncCoverStorageModeReads(sourceAccess.mode) &&
               syncCoverStorageModeWrites(targetAccess.mode);
      case SyncCoverDemandKind::MemoryWAW:
        return syncCoverStorageModeWrites(sourceAccess.mode) &&
               syncCoverStorageModeWrites(targetAccess.mode);
      case SyncCoverDemandKind::HardwareAccRAR:
        return syncCoverStorageModeReads(sourceAccess.mode) &&
               syncCoverStorageModeReads(targetAccess.mode);
      case SyncCoverDemandKind::SSA:
        return false;
      }
      return false;
    };
    for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
      Operation *sourceOperation =
          program.getNodeBindings()[demand.source].operation;
      Operation *targetOperation =
          program.getNodeBindings()[demand.target].operation;
      const std::optional<unsigned> sourceRole = operationRole(sourceOperation);
      const std::optional<unsigned> targetRole = operationRole(targetOperation);
      if (!sourceRole || !targetRole) {
        continue;
      }
      std::vector<unsigned> sourcePhases;
      unsigned distance = demand.distance;
      if (unrolledIterations) {
        const auto sourcePosition = unrolledIterations->find(sourceOperation);
        const auto targetPosition = unrolledIterations->find(targetOperation);
        const bool missingIteration =
            sourcePosition == unrolledIterations->end() ||
            targetPosition == unrolledIterations->end();
        const bool reversedIteration =
            !missingIteration &&
            sourcePosition->second >= targetPosition->second;
        if (missingIteration || reversedIteration) {
          continue;
        }
        distance = targetPosition->second - sourcePosition->second;
        sourcePhases.push_back(sourcePosition->second % period);
      } else {
        if (distance == 0) {
          continue;
        }
        for (unsigned phase = 0; phase < period; ++phase) {
          const unsigned targetPhase = (phase + distance) % period;
          const bool reachable =
              guardMatches(demand.sourceGuard, demand.scope, phase) &&
              guardMatches(demand.targetGuard, demand.scope, targetPhase);
          if (reachable) {
            sourcePhases.push_back(phase);
          }
        }
      }
      for (SyncCoverDemandKind kind : demand.provenanceKinds) {
        for (SyncCoverStorageWitnessId witnessId : demand.storageWitnesses) {
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[witnessId];
          const SyncCoverStorageAccess &sourceAccess =
              graph.getStorageAccesses()[witness.sourceAccess];
          const SyncCoverStorageAccess &targetAccess =
              graph.getStorageAccesses()[witness.targetAccess];
          if (!supportsKind(sourceAccess, targetAccess, kind)) {
            continue;
          }
          const AddressSpace space =
              program.getStorageSpaces()[sourceAccess.domain];
          for (unsigned sourcePhase : sourcePhases) {
            if (phaseFilter && !phaseFilter(sourceAccess, targetAccess,
                                            sourcePhase, distance)) {
              continue;
            }
            const RecurrenceKey key{*sourceRole,
                                    *targetRole,
                                    kind,
                                    sourcePhase,
                                    space,
                                    sourceAccess.extent.begin,
                                    sourceAccess.extent.end,
                                    targetAccess.extent.begin,
                                    targetAccess.extent.end,
                                    sourceAccess.mode,
                                    targetAccess.mode};
            auto [position, inserted] =
                result.minimumDistances.insert({key, distance});
            if (inserted || distance < position->second) {
              position->second = distance;
              result.covered[key] = covered.contains(demandId);
            } else if (distance == position->second) {
              result.covered[key] &= covered.contains(demandId);
            }
          }
        }
      }
    }
    return result;
  };

  const std::optional<SyncCoverDemandSet> sameCoverage = oracleCoverage(*same);
  const std::optional<SyncCoverDemandSet> unrolledCoverage =
      oracleCoverage(*unrolled);
  const std::optional<SyncCoverDemandSet> multipleGapCoverage =
      oracleCoverage(*multipleGaps);
  const std::optional<SyncCoverDemandSet> multipleGapUnrolledCoverage =
      oracleCoverage(*multipleGapsUnrolled);
  const std::optional<SyncCoverDemandSet> phaseSlotGapCoverage =
      oracleCoverage(*phaseSlotGaps);
  const std::optional<SyncCoverDemandSet> phaseSlotGapUnrolledCoverage =
      oracleCoverage(*phaseSlotGapsUnrolled);
  if (!check(sameCoverage && unrolledCoverage && multipleGapCoverage &&
                 multipleGapUnrolledCoverage && phaseSlotGapCoverage &&
                 phaseSlotGapUnrolledCoverage,
             "ground exact semantic coverage for explicit unroll checks")) {
    return false;
  }
  const std::array<unsigned, 3> sameIterations = {0, 2, 4};
  const std::array<unsigned, 4> multipleGapIterations = {1, 2, 4, 5};
  const std::array<unsigned, 5> phaseSlotGapIterations = {1, 2, 3, 5, 6};
  const std::map<Operation *, unsigned> sameUnrolledIterations =
      buildIterationMap(*unrolled, sameIterations);
  const std::map<Operation *, unsigned> multipleGapUnrolledIterations =
      buildIterationMap(*multipleGapsUnrolled, multipleGapIterations);
  const std::map<Operation *, unsigned> phaseSlotGapUnrolledIterations =
      buildIterationMap(*phaseSlotGapsUnrolled, phaseSlotGapIterations);
  const WitnessPhaseFilter allWitnessPhases;
  const WitnessPhaseFilter alternatingSlotPhase =
      [](const SyncCoverStorageAccess &sourceAccess,
         const SyncCoverStorageAccess &targetAccess, unsigned sourcePhase,
         unsigned distance) {
        const std::uint64_t sourceSize =
            sourceAccess.extent.end - sourceAccess.extent.begin;
        const std::uint64_t targetSize =
            targetAccess.extent.end - targetAccess.extent.begin;
        const std::uint64_t expectedSource = (sourcePhase % 2) * sourceSize;
        const std::uint64_t expectedTarget =
            ((sourcePhase + distance) % 2) * targetSize;
        return sourceAccess.extent.begin == expectedSource &&
               targetAccess.extent.begin == expectedTarget;
      };
  const ObligationSnapshot sameLoop =
      collectSnapshot(*same, 2, nullptr, *sameCoverage, allWitnessPhases);
  const ObligationSnapshot sameReference =
      collectSnapshot(*unrolled, 2, &sameUnrolledIterations, *unrolledCoverage,
                      allWitnessPhases);
  const ObligationSnapshot multipleGapLoop = collectSnapshot(
      *multipleGaps, 3, nullptr, *multipleGapCoverage, allWitnessPhases);
  const ObligationSnapshot multipleGapReference =
      collectSnapshot(*multipleGapsUnrolled, 3, &multipleGapUnrolledIterations,
                      *multipleGapUnrolledCoverage, allWitnessPhases);
  const ObligationSnapshot phaseSlotGapLoop = collectSnapshot(
      *phaseSlotGaps, 4, nullptr, *phaseSlotGapCoverage, alternatingSlotPhase);
  const ObligationSnapshot phaseSlotGapReference = collectSnapshot(
      *phaseSlotGapsUnrolled, 4, &phaseSlotGapUnrolledIterations,
      *phaseSlotGapUnrolledCoverage, alternatingSlotPhase);
  const auto allCovered = [](const ObligationSnapshot &snapshot) {
    return llvm::all_of(snapshot.covered,
                        [](const auto &entry) { return entry.second; });
  };
  return check(!sameLoop.minimumDistances.empty() &&
                   !multipleGapLoop.minimumDistances.empty() &&
                   !phaseSlotGapLoop.minimumDistances.empty(),
               "construct witness-level phase-aware loop obligations") &&
         check(sameLoop.minimumDistances == sameReference.minimumDistances &&
                   multipleGapLoop.minimumDistances ==
                       multipleGapReference.minimumDistances &&
                   phaseSlotGapLoop.minimumDistances ==
                       phaseSlotGapReference.minimumDistances,
               "match witness-level successor obligations from explicit "
               "unrollings") &&
         check(sameLoop.covered == sameReference.covered &&
                   multipleGapLoop.covered == multipleGapReference.covered &&
                   phaseSlotGapLoop.covered == phaseSlotGapReference.covered &&
                   allCovered(sameLoop) && allCovered(multipleGapLoop) &&
                   allCovered(phaseSlotGapLoop),
               "match grounded semantic coverage from explicit unrollings");
}

bool testFirstIterationRecurrenceSuppression() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @first(%limit: index,
                       %a: !pto.tile_buf<vec, 16x16xf32>,
                       %b: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %limit step %c1 {
          pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          %is_first = arith.cmpi eq, %i, %c0 : index
          scf.if %is_first {
            pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @near_miss(%limit: index,
                           %a: !pto.tile_buf<vec, 16x16xf32>,
                           %b: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %limit step %c1 {
          pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          %is_second = arith.cmpi eq, %i, %c1 : index
          scf.if %is_second {
            pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @nested(%outer_limit: index, %inner_limit: index,
                        %a: !pto.tile_buf<vec, 16x16xf32>,
                        %b: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %outer_limit step %c1 {
          pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                   outs(%b : !pto.tile_buf<vec, 16x16xf32>)
          scf.for %j = %c0 to %inner_limit step %c1 {
            %inner_first = arith.cmpi eq, %j, %c0 : index
            scf.if %inner_first {
              pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf32>)
                       outs(%b : !pto.tile_buf<vec, 16x16xf32>)
            }
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse first-iteration recurrence fixture")) {
    return false;
  }
  const auto build = [&](StringRef name) {
    return buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>(name));
  };
  FailureOr<CanonicalSyncProgram> first = build("first");
  bool sawLoopVaryingControl = false;
  FailureOr<CanonicalSyncProgram> nearMiss = failure();
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawLoopVaryingControl |=
          diagnostic.str().find("cannot model this loop-varying control") !=
          std::string::npos;
      return success();
    });
    nearMiss = build("near_miss");
  }
  FailureOr<CanonicalSyncProgram> nested = build("nested");
  const bool builtPrograms =
      check(succeeded(first) && failed(nearMiss) && sawLoopVaryingControl &&
                succeeded(nested),
            "accept exact first-iteration controls and reject an unmodeled "
            "loop-varying control");
  if (!builtPrograms) {
    return false;
  }
  const auto guardedTarget = [](const CanonicalSyncProgram &program) {
    for (auto [node, binding] : llvm::enumerate(program.getNodeBindings())) {
      if (binding.operation &&
          binding.operation->getParentOfType<scf::IfOp>()) {
        return std::optional<SyncCoverNodeId>(node);
      }
    }
    return std::optional<SyncCoverNodeId>{};
  };
  const auto recurrenceTo = [](const CanonicalSyncProgram &program,
                               SyncCoverNodeId target,
                               std::optional<SyncCoverScopeId> scope) {
    return llvm::any_of(
        program.getGraph().getDemands(), [&](const SyncCoverDemand &demand) {
          return demand.target == target && demand.distance != 0 &&
                 (!scope || demand.scope == *scope);
        });
  };
  const std::optional<SyncCoverNodeId> firstTarget = guardedTarget(*first);
  const std::optional<SyncCoverNodeId> nestedTarget = guardedTarget(*nested);
  std::optional<SyncCoverScopeId> outerScope;
  for (const SyncCoverScope &scope : nested->getGraph().getScopes()) {
    Operation *owner = nested->getScopeBindings()[scope.id].owner;
    auto loop = dyn_cast_or_null<scf::ForOp>(owner);
    if (loop && !loop->getParentOfType<scf::ForOp>()) {
      outerScope = scope.id;
      break;
    }
  }
  return check(firstTarget && !recurrenceTo(*first, *firstTarget, std::nullopt),
               "remove recurrence into a first-iteration-only target") &&
         check(nestedTarget && outerScope &&
                   recurrenceTo(*nested, *nestedTarget, outerScope),
               "retain enclosing-loop recurrence for an inner first iteration");
}

bool testUnmodeledLoopVaryingControlsFailClosed() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @gap_two(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %limit step %c1 {
          %active = arith.cmpi ne, %i, %c1 : index
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @shifted_modulo(
          %limit: index, %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        scf.for %i = %c0 to %limit step %c1 {
          %shifted = arith.addi %i, %c1 : index
          %phase = arith.remui %shifted, %c3 : index
          %active = arith.cmpi eq, %phase, %c0 : index
          scf.if %active {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @nested_outer_varying(
          %outer_limit: index, %inner_limit: index,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %outer_limit step %c1 {
          scf.for %j = %c0 to %inner_limit step %c1 {
            %active = arith.cmpi ne, %i, %c1 : index
            scf.if %active {
              pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                        outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
              pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                       outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            }
          }
        }
        return
      }
      func.func @nested_varying_trip_count(
          %outer_limit: index,
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %i = %c0 to %outer_limit step %c1 {
          %skip = arith.cmpi eq, %i, %c1 : index
          %inner_ub = arith.select %skip, %c0, %c1 : index
          scf.for %j = %c0 to %inner_ub step %c1 {
            pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                      outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
            pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                     outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
          }
        }
        return
      }
      func.func @gap_two_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @shifted_modulo_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @nested_outer_varying_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @nested_varying_trip_count_unrolled(
          %src: !pto.partition_tensor_view<16x16xf32>,
          %slot: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%src : !pto.partition_tensor_view<16x16xf32>)
                  outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%slot : !pto.tile_buf<vec, 16x16xf32>)
                 outs(%slot : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse unmodeled loop-varying control fixtures")) {
    return false;
  }
  const auto build = [&](StringRef name) {
    return buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>(name));
  };
  const auto rejectsWith = [&](StringRef name, StringRef expectedDiagnostic) {
    bool sawDiagnostic = false;
    FailureOr<CanonicalSyncProgram> rejected = failure();
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        sawDiagnostic |= diagnostic.str().find(expectedDiagnostic.str()) !=
                         std::string::npos;
        return success();
      });
      rejected = build(name);
    }
    return failed(rejected) && sawDiagnostic;
  };
  FailureOr<CanonicalSyncProgram> gapTwoUnrolled = build("gap_two_unrolled");
  FailureOr<CanonicalSyncProgram> shiftedUnrolled =
      build("shifted_modulo_unrolled");
  FailureOr<CanonicalSyncProgram> nestedUnrolled =
      build("nested_outer_varying_unrolled");
  FailureOr<CanonicalSyncProgram> nestedTripUnrolled =
      build("nested_varying_trip_count_unrolled");
  const bool unrolledBuilt =
      succeeded(gapTwoUnrolled) && succeeded(shiftedUnrolled) &&
      succeeded(nestedUnrolled) && succeeded(nestedTripUnrolled);
  if (!check(unrolledBuilt,
             "build explicit unrolls for rejected varying controls")) {
    return false;
  }
  const auto mappedHazardGaps = [](const CanonicalSyncProgram &program,
                                   ArrayRef<unsigned> iterations) {
    std::map<Operation *, unsigned> mappedIterations;
    std::array<std::size_t, 2> occurrences{};
    for (const CanonicalSyncNodeBinding &binding : program.getNodeBindings()) {
      Operation *operation = binding.operation;
      const bool alreadyMapped =
          !operation || mappedIterations.count(operation) != 0;
      if (alreadyMapped) {
        continue;
      }
      const unsigned role = isa<TLoadOp>(operation) ? 0 : 1;
      const bool supportedRole = isa<TLoadOp, TAbsOp>(operation);
      if (!supportedRole || occurrences[role] >= iterations.size()) {
        continue;
      }
      mappedIterations[operation] = iterations[occurrences[role]++];
    }
    std::set<unsigned> gaps;
    for (const SyncCoverDemand &demand : program.getGraph().getDemands()) {
      Operation *source = program.getNodeBindings()[demand.source].operation;
      Operation *target = program.getNodeBindings()[demand.target].operation;
      const auto sourceIteration = mappedIterations.find(source);
      const auto targetIteration = mappedIterations.find(target);
      const bool mapped = sourceIteration != mappedIterations.end() &&
                          targetIteration != mappedIterations.end();
      const bool memoryHazard =
          llvm::any_of(demand.provenanceKinds, [](SyncCoverDemandKind kind) {
            return kind != SyncCoverDemandKind::SSA;
          });
      if (mapped && memoryHazard &&
          sourceIteration->second < targetIteration->second) {
        gaps.insert(targetIteration->second - sourceIteration->second);
      }
    }
    return gaps;
  };
  const std::set<unsigned> gapTwo = {2};
  const std::set<unsigned> gapThree = {3};
  const bool unrollsExposeOmittedGaps =
      mappedHazardGaps(*gapTwoUnrolled, {0, 2}) == gapTwo &&
      mappedHazardGaps(*shiftedUnrolled, {2, 5}) == gapThree &&
      mappedHazardGaps(*nestedUnrolled, {0, 2}) == gapTwo &&
      mappedHazardGaps(*nestedTripUnrolled, {0, 2}) == gapTwo;
  return check(unrollsExposeOmittedGaps,
               "expose the real successor gaps in explicit unrolls") &&
         check(rejectsWith("gap_two", "cannot model this loop-varying control"),
               "reject a transient direct induction guard") &&
         check(rejectsWith("shifted_modulo",
                           "cannot model this loop-varying control"),
               "reject a transformed periodic guard") &&
         check(rejectsWith("nested_outer_varying",
                           "cannot model this loop-varying control"),
               "reject a nested guard that varies with an outer induction") &&
         check(rejectsWith("nested_varying_trip_count",
                           "scf.for upper bound that varies with an enclosing "
                           "loop"),
               "reject a nested trip count that varies with an outer "
               "induction");
}

bool testUnsupportedBlockArgumentProvenanceFailsClosed() {
  return expectAnalysisFailure(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @unsupported_block_argument() {
        pto.section.vector {
        ^bb0(%condition: i1):
          scf.if %condition {
          }
        }
        return
      }
    }
  )mlir",
                               "unsupported_block_argument",
                               "cannot trace SSA provenance through this "
                               "block argument");
}

bool testSsaProvenanceTraversalIsBoundedAndIterative() {
  constexpr std::size_t deepLength = 2048;
  constexpr std::size_t wideLeaves = 256;
  constexpr std::size_t sharedLevels = 512;
  std::string source = R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @deep(%seed: i1, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %true = arith.constant 1 : i1
)mlir";
  std::string previous = "%seed";
  for (std::size_t index = 0; index < deepLength; ++index) {
    const std::string result = "%deep" + std::to_string(index);
    source +=
        "        " + result + " = arith.xori " + previous + ", %true : i1\n";
    previous = result;
  }
  source += "        scf.for %i = %c0 to %limit step %c1 {\n"
            "          scf.if " +
            previous + " {\n          }\n        }\n        return\n      }\n";
  source += R"mlir(
      func.func @wide(%seed: i1, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %true = arith.constant 1 : i1
)mlir";
  std::vector<std::string> frontier;
  frontier.reserve(wideLeaves);
  for (std::size_t index = 0; index < wideLeaves; ++index) {
    const std::string result = "%wide" + std::to_string(index);
    source += "        " + result +
              " = arith.select %seed, %true, %seed : "
              "i1\n";
    frontier.push_back(result);
  }
  std::size_t join = 0;
  bool hasMultipleFrontierNodes = frontier.size() > 1;
  while (hasMultipleFrontierNodes) {
    std::vector<std::string> next;
    next.reserve((frontier.size() + 1) / 2);
    for (std::size_t index = 0; index < frontier.size(); index += 2) {
      if (index + 1 == frontier.size()) {
        next.push_back(frontier[index]);
        continue;
      }
      const std::string result = "%wide_join" + std::to_string(join++);
      source += "        " + result + " = arith.ori " + frontier[index] + ", " +
                frontier[index + 1] + " : i1\n";
      next.push_back(result);
    }
    frontier = std::move(next);
    hasMultipleFrontierNodes = frontier.size() > 1;
  }
  source += "        scf.for %i = %c0 to %limit step %c1 {\n"
            "          scf.if " +
            frontier.front() +
            " {\n          }\n        }\n        return\n      }\n";
  source += R"mlir(
      func.func @shared(%seed: i1, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
)mlir";
  previous = "%seed";
  for (std::size_t level = 0; level < sharedLevels; ++level) {
    const std::string left = "%shared_left" + std::to_string(level);
    const std::string right = "%shared_right" + std::to_string(level);
    const std::string result = "%shared_join" + std::to_string(level);
    source +=
        "        " + left + " = arith.andi " + previous + ", %seed : i1\n";
    source +=
        "        " + right + " = arith.ori " + previous + ", %seed : i1\n";
    source += "        " + result + " = arith.xori " + left + ", " + right +
              " : i1\n";
    previous = result;
  }
  source += "        scf.for %i = %c0 to %limit step %c1 {\n"
            "          scf.if " +
            previous +
            " {\n          }\n        }\n        return\n      }\n    }\n";

  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
  if (!check(static_cast<bool>(module),
             "parse bounded SSA-provenance fixtures")) {
    return false;
  }
  const auto minimumAcceptedBound = [&](func::FuncOp function) {
    std::size_t lower = 1;
    std::size_t upper = 1U << 20;
    {
      CanonicalSyncAnalysisOptions options;
      options.maximumPairInspections = upper;
      if (failed(buildCanonicalSyncProgram(function, options))) {
        return std::size_t{0};
      }
    }
    while (lower < upper) {
      const std::size_t middle = lower + (upper - lower) / 2;
      CanonicalSyncAnalysisOptions options;
      options.maximumPairInspections = middle;
      FailureOr<CanonicalSyncProgram> trial = failure();
      {
        ScopedDiagnosticHandler handler(&context,
                                        [](Diagnostic &) { return success(); });
        trial = buildCanonicalSyncProgram(function, options);
      }
      if (succeeded(trial)) {
        upper = middle;
      } else {
        lower = middle + 1;
      }
    }
    return lower;
  };
  const auto verifyExactAndBelow = [&](StringRef name, StringRef description,
                                       std::size_t &minimum) {
    func::FuncOp function = module->lookupSymbol<func::FuncOp>(name);
    minimum = minimumAcceptedBound(function);
    if (minimum <= 1) {
      return check(false, ("find the exact " + description + " bound").str());
    }
    CanonicalSyncAnalysisOptions exactOptions;
    exactOptions.maximumPairInspections = minimum;
    FailureOr<CanonicalSyncProgram> exact =
        buildCanonicalSyncProgram(function, exactOptions);
    CanonicalSyncAnalysisOptions belowOptions = exactOptions;
    --belowOptions.maximumPairInspections;
    bool sawLimit = false;
    FailureOr<CanonicalSyncProgram> below = failure();
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        sawLimit |= diagnostic.str().find("pair-inspection limit exceeded") !=
                    std::string::npos;
        return success();
      });
      below = buildCanonicalSyncProgram(function, belowOptions);
    }
    return check(succeeded(exact),
                 ("accept " + description + " at its exact bound").str()) &&
           check(failed(below) && sawLimit,
                 ("reject " + description + " one unit below its bound").str());
  };
  std::size_t deepMinimum = 0;
  std::size_t wideMinimum = 0;
  std::size_t sharedMinimum = 0;
  const bool exactBounds =
      verifyExactAndBelow("deep", "deep SSA provenance", deepMinimum) &&
      verifyExactAndBelow("wide", "wide SSA provenance", wideMinimum) &&
      verifyExactAndBelow("shared", "shared-diamond SSA provenance",
                          sharedMinimum);
  const std::size_t deepEdgeFloor = 2 * (deepLength + 1);
  const std::size_t wideEdgeFloor = 2 * (5 * wideLeaves - 1);
  const std::size_t sharedEdgeFloor = 2 * (6 * sharedLevels + 1);
  return exactBounds &&
         check(deepMinimum >= deepEdgeFloor,
               "charge every deep-chain provenance edge") &&
         check(wideMinimum >= wideEdgeFloor,
               "charge every wide-frontier provenance edge") &&
         check(sharedMinimum >= sharedEdgeFloor,
               "charge repeated edges in a shared provenance DAG");
}

bool testGuardedOwnershipVerificationWorkIsBounded() {
  bool passed = true;
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @guarded_ownership_host() {
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse guarded ownership host function")) {
    return false;
  }

  SyncCoverGraph graph;
  constexpr std::uint32_t producerResource = 1;
  constexpr std::uint32_t consumerResource = 2;
  const SyncCoverScopeId loop =
      takeIndex(graph.addScope(0, true, SyncCoverTimelineInterval{0, 31}, true),
                passed, "add guarded ownership loop");
  constexpr std::size_t guardDepth = 8;
  SyncCoverScopeId guardedScope = loop;
  SyncCoverGuard guard;
  for (std::size_t depth = 0; depth < guardDepth; ++depth) {
    const SyncCoverControlId control =
        takeIndex(graph.addControl(2, guardedScope), passed,
                  "add guarded ownership control");
    guard.literals.push_back({control, 0});
    guardedScope = takeIndex(
        graph.addScope(guardedScope, true, std::nullopt, false, guard), passed,
        "add guarded ownership path scope");
  }

  const SyncCoverNodeId producer0 =
      takeIndex(graph.addNode(producerResource, 1, guardedScope, 0, guard,
                              {consumerResource}),
                passed, "add first guarded producer");
  const SyncCoverNodeId consumer0 =
      takeIndex(graph.addNode(consumerResource, 1, guardedScope, 1, guard,
                              {producerResource}),
                passed, "add first guarded consumer");
  const SyncCoverNodeId producer1 =
      takeIndex(graph.addNode(producerResource, 1, guardedScope, 2, guard,
                              {consumerResource}),
                passed, "add second guarded producer");
  const SyncCoverNodeId consumer1 =
      takeIndex(graph.addNode(consumerResource, 1, guardedScope, 3, guard,
                              {producerResource}),
                passed, "add second guarded consumer");
  const SyncCoverStorageDomainId domain0 =
      takeIndex(graph.addStorageDomain(SyncCoverStorageDomainRole::L1Tile),
                passed, "add first guarded ownership domain");
  const SyncCoverStorageDomainId domain1 =
      takeIndex(graph.addStorageDomain(SyncCoverStorageDomainRole::L1Tile),
                passed, "add second guarded ownership domain");
  const SyncCoverStorageAccessId producer0Access =
      takeIndex(graph.addStorageAccess(producer0, domain0, 1, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add first guarded ownership write");
  const SyncCoverStorageAccessId consumer0Access =
      takeIndex(graph.addStorageAccess(consumer0, domain0, 1, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add first guarded ownership read");
  const SyncCoverStorageAccessId producer1Access =
      takeIndex(graph.addStorageAccess(producer1, domain1, 2, {0, 64},
                                       SyncCoverStorageAccessMode::Write,
                                       std::nullopt, true),
                passed, "add second guarded ownership write");
  const SyncCoverStorageAccessId consumer1Access =
      takeIndex(graph.addStorageAccess(consumer1, domain1, 2, {0, 64},
                                       SyncCoverStorageAccessMode::Read,
                                       std::nullopt, true),
                passed, "add second guarded ownership read");
  const auto addMemoryDemand =
      [&](SyncCoverNodeId source, SyncCoverNodeId target,
          SyncCoverScopeId scope, unsigned distance, SyncCoverDemandKind kind,
          SyncCoverStorageAccessId sourceAccess,
          SyncCoverStorageAccessId targetAccess, std::string_view message) {
        const SyncCoverStorageWitnessId witness =
            takeIndex(graph.addStorageWitness(sourceAccess, targetAccess),
                      passed, "add guarded ownership witness");
        SyncCoverDemand demand;
        demand.source = source;
        demand.target = target;
        demand.scope = scope;
        demand.distance = distance;
        demand.provenanceKinds = {kind};
        demand.storageWitnesses = {witness};
        passed &= check(static_cast<bool>(graph.addDemand(std::move(demand))),
                        message);
      };
  addMemoryDemand(producer0, consumer0, guardedScope, 0,
                  SyncCoverDemandKind::MemoryRAW, producer0Access,
                  consumer0Access, "add first guarded ready demand");
  addMemoryDemand(consumer0, producer0, loop, 1, SyncCoverDemandKind::MemoryWAR,
                  consumer0Access, producer0Access,
                  "add first guarded release demand");
  addMemoryDemand(producer1, consumer1, guardedScope, 0,
                  SyncCoverDemandKind::MemoryRAW, producer1Access,
                  consumer1Access, "add second guarded ready demand");
  addMemoryDemand(consumer1, producer1, loop, 1, SyncCoverDemandKind::MemoryWAR,
                  consumer1Access, producer1Access,
                  "add second guarded release demand");

  SyncCoverBasicOwnershipCertificate certificate;
  certificate.kind = SyncCoverBasicOwnershipKind::L1Tile;
  certificate.loopScope = loop;
  certificate.producerResource = producerResource;
  certificate.consumerResource = consumerResource;
  certificate.lanes = {
      {0, {{domain0, {0, 64}, {producer0Access, consumer0Access}}}},
      {1, {{domain1, {0, 64}, {producer1Access, consumer1Access}}}},
  };
  certificate.paths = {
      {guardedScope,
       {{0,
         0,
         {producer0},
         {consumer0},
         {SyncCoverAnchorKind::BeforeNode, producer0, 0, 0},
         {SyncCoverAnchorKind::AfterNode, producer0, 0, 0},
         {SyncCoverAnchorKind::BeforeNode, consumer0, 0, 0},
         {SyncCoverAnchorKind::AfterNode, consumer0, 0, 0}},
        {1,
         1,
         {producer1},
         {consumer1},
         {SyncCoverAnchorKind::BeforeNode, producer1, 0, 0},
         {SyncCoverAnchorKind::AfterNode, producer1, 0, 0},
         {SyncCoverAnchorKind::BeforeNode, consumer1, 0, 0},
         {SyncCoverAnchorKind::AfterNode, consumer1, 0, 0}}}},
  };
  passed &=
      check(static_cast<bool>(graph.addBasicOwnershipCertificate(certificate)),
            "register guarded ownership certificate");
  passed &= check(static_cast<bool>(graph.freezeStructure()),
                  "freeze guarded ownership graph");
  if (!passed) {
    return false;
  }

  SyncCoverStorageLifecycleIndex lifecycle =
      buildSyncCoverStorageLifecycleIndex(graph);
  SyncCoverStorageProtocolSeedIndex protocolSeeds =
      buildSyncCoverStorageProtocolSeedIndex(graph, lifecycle);
  SyncCoverStorageCutIndex cuts =
      buildSyncCoverStorageCutIndex(graph, lifecycle);
  SyncCoverStorageFactoredRectangleIndex rectangles =
      buildSyncCoverStorageFactoredRectangleIndex(graph, cuts);
  const bool hasCompleteSyntheticCuts =
      lifecycle.isComplete() && protocolSeeds.isComplete() &&
      cuts.isComplete() && rectangles.isComplete() &&
      rectangles.getStatistics().syntheticRectangles != 0;
  if (!check(hasCompleteSyntheticCuts,
             "build complete synthetic cuts for mixed-family limit test")) {
    return false;
  }

  CanonicalSyncTargetCapabilities syntheticCapabilities;
  syntheticCapabilities.directEventCompletion.version = 1;
  syntheticCapabilities.directEventCompletion.resourcePairs = {
      {producerResource, consumerResource},
      {consumerResource, producerResource},
  };
  syntheticCapabilities.compilerUsableEventIds = {0, 1, 2, 3, 4, 5};
  CanonicalSyncProgram program(
      module->lookupSymbol<func::FuncOp>("guarded_ownership_host"),
      std::move(graph), {}, {}, {}, {}, std::move(lifecycle),
      std::move(protocolSeeds), {}, {}, {}, {}, std::move(cuts), {},
      std::move(rectangles), std::move(syntheticCapabilities), {}, {});
  CanonicalSyncBuildOptions options;
  options.enableDemandBasisReduction = false;
  options.patterns.enabledMechanismFamilies = canonicalSyncMechanismFamilyBit(
      CanonicalSyncMechanismFamily::BasicOwnership);
  options.patterns.enableDirectPairs = false;
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(program, options);
  if (!check(precise && precise.problem,
             "build guarded ownership candidate catalog")) {
    return false;
  }
  CanonicalSyncBuildOptions mixedOptions = options;
  mixedOptions.patterns.enabledMechanismFamilies |=
      canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::StorageCutEvent);
  mixedOptions.problemLimits.maximumMechanisms =
      precise.problem->getMechanisms().size();
  CanonicalSyncProblemBuildResult mixed =
      buildCanonicalSyncPreciseProblem(program, mixedOptions);
  const CanonicalSyncMechanismOriginMask storageCutOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::StorageCutEvent);
  const CanonicalSyncMechanismOriginMask ownershipOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::BasicOwnershipStableL1Protocol);
  const bool mixedRetainsOwnershipBeforeCutTruncation =
      mixed && mixed.problem && mixed.problem->wasPatternGenerationTruncated() &&
      mixed.problem->getMechanisms().size() ==
          precise.problem->getMechanisms().size() &&
      llvm::any_of(mixed.problem->getMechanisms(), [&](const auto &candidate) {
        return (candidate.originMask & ownershipOrigin) != 0;
      }) &&
      llvm::none_of(mixed.problem->getMechanisms(), [&](const auto &candidate) {
        return (candidate.originMask & storageCutOrigin) != 0;
      });
  const auto mechanism = llvm::find_if(
      precise.problem->getMechanisms(), [&](const auto &candidate) {
        return (candidate.originMask & ownershipOrigin) != 0;
      });
  return check(mixedRetainsOwnershipBeforeCutTruncation,
               "truncate optional storage cuts after retaining ownership") &&
         check(mechanism != precise.problem->getMechanisms().end(),
               "synthesize guarded ownership protocol") &&
         verifyExactAndOneLessProtocolWork(
             *precise.problem, mechanism->id,
             "verify deep-guard ownership at its exact work bound",
             "reject deep-guard ownership at its one-less work bound");
}

bool testBasicL0OwnershipSharesExhaustiveBranchBoundaries() {
  MLIRContext context;
  loadDialects(context);
  const std::string ownershipSource = R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<cube>} {
      func.func @branch_l0(%limit: index, %condition: i1,
          %left_source: !pto.tile_buf<mat, 128x512xf16,
            blayout=col_major, slayout=row_major>,
          %right_source: !pto.tile_buf<mat, 256x256xf16,
            slayout=col_major>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %addr0 = arith.constant 0 : i64
        %addr16384 = arith.constant 16384 : i64
        %addr32768 = arith.constant 32768 : i64
        %left0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        %left1 = pto.alloc_tile addr = %addr16384 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        %right0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<right, 64x256xf16, slayout=col_major>
        %right1 = pto.alloc_tile addr = %addr32768 :
          !pto.tile_buf<right, 64x256xf16, slayout=col_major>
        %acc = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
            slayout=row_major, fractal=1024>
        scf.for %i = %c0 to %limit step %c1 {
          pto.textract ins(%left_source, %c0, %c0 :
            !pto.tile_buf<mat, 128x512xf16, blayout=col_major,
              slayout=row_major>, index, index)
            outs(%left0 : !pto.tile_buf<left, 128x64xf16,
              slayout=row_major>)
          pto.textract ins(%right_source, %c0, %c0 :
            !pto.tile_buf<mat, 256x256xf16, slayout=col_major>, index, index)
            outs(%right0 : !pto.tile_buf<right, 64x256xf16,
              slayout=col_major>)
          scf.if %condition {
            pto.tmatmul ins(%left0, %right0 :
              !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
              !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
              outs(%acc : !pto.tile_buf<acc, 128x256xf32,
                blayout=col_major, slayout=row_major, fractal=1024>)
          } else {
            pto.tmatmul.acc ins(%acc, %left0, %right0 :
              !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
                slayout=row_major, fractal=1024>,
              !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
              !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
              outs(%acc : !pto.tile_buf<acc, 128x256xf32,
                blayout=col_major, slayout=row_major, fractal=1024>)
          }
          pto.textract ins(%left_source, %c0, %c64 :
            !pto.tile_buf<mat, 128x512xf16, blayout=col_major,
              slayout=row_major>, index, index)
            outs(%left1 : !pto.tile_buf<left, 128x64xf16,
              slayout=row_major>)
          pto.textract ins(%right_source, %c64, %c0 :
            !pto.tile_buf<mat, 256x256xf16, slayout=col_major>, index, index)
            outs(%right1 : !pto.tile_buf<right, 64x256xf16,
              slayout=col_major>)
          pto.tmatmul.acc ins(%acc, %left1, %right1 :
            !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
              slayout=row_major, fractal=1024>,
            !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
            !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
            outs(%acc : !pto.tile_buf<acc, 128x256xf32,
              blayout=col_major, slayout=row_major, fractal=1024>)
        }
        return
      }
    }
  )mlir";
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(ownershipSource, &context);
  if (!check(static_cast<bool>(module),
             "parse exhaustive L0 ownership fixture")) {
    return false;
  }
  func::FuncOp function = module->lookupSymbol<func::FuncOp>("branch_l0");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "discover exhaustive L0 ownership")) {
    return false;
  }
  const auto certificate = llvm::find_if(
      program->getGraph().getBasicOwnershipCertificates(),
      [](const SyncCoverBasicOwnershipCertificate &candidate) {
        return candidate.kind == SyncCoverBasicOwnershipKind::L0Operand;
      });
  const bool exactCertificate =
      certificate !=
          program->getGraph().getBasicOwnershipCertificates().end() &&
      certificate->paths.size() == 1 &&
      certificate->paths[0].uses.size() == 2 &&
      certificate->paths[0].uses[0].consumers.size() == 2 &&
      certificate->paths[0].uses[0].readAcquireAnchor.kind ==
          SyncCoverAnchorKind::ControlEntry &&
      certificate->paths[0].uses[0].releaseAnchor.kind ==
          SyncCoverAnchorKind::ControlExit;
  if (!check(exactCertificate,
             "certify one wait and release at the exhaustive branch cut")) {
    return false;
  }

  CanonicalSyncAnalysisOptions ownershipDiscoveryDisabled;
  ownershipDiscoveryDisabled.discoverBasicOwnershipCertificates = false;
  FailureOr<CanonicalSyncProgram> ownershipDisabledProgram =
      buildCanonicalSyncProgram(function, ownershipDiscoveryDisabled);
  if (!check(succeeded(ownershipDisabledProgram),
             "build graph with ownership discovery disabled") ||
      !check(ownershipDisabledProgram->getGraph()
                 .getBasicOwnershipCertificates()
                 .empty(),
             "skip disabled ownership discovery") ||
      !check(ownershipDisabledProgram->getOwnershipDiscoveryStatistics()
                     .inspections == 0,
             "perform no disabled ownership inspections")) {
    return false;
  }

  CanonicalSyncAnalysisOptions boundedAnalysis;
  boundedAnalysis.maximumBasicOwnershipInspections = 1;
  FailureOr<CanonicalSyncProgram> boundedProgram =
      buildCanonicalSyncProgram(function, boundedAnalysis);
  if (!check(succeeded(boundedProgram),
             "truncate optional ownership discovery without rejecting the "
             "graph")) {
    return false;
  }
  const CanonicalSyncOwnershipDiscoveryStatistics &boundedStatistics =
      boundedProgram->getOwnershipDiscoveryStatistics();
  if (!check(boundedProgram->getGraph().getBasicOwnershipCertificates().empty(),
             "omit ownership certificates after discovery truncation") ||
      !check(boundedStatistics.truncated,
             "report bounded ownership-discovery truncation") ||
      !check(boundedStatistics.inspections == 1,
             "charge the complete bounded ownership-discovery allowance")) {
    return false;
  }

  CanonicalSyncAnalysisOptions aggregateBoundedAnalysis;
  aggregateBoundedAnalysis.maximumBasicOwnershipNodeReferences = 1;
  FailureOr<CanonicalSyncProgram> aggregateBoundedProgram =
      buildCanonicalSyncProgram(function, aggregateBoundedAnalysis);
  if (!check(succeeded(aggregateBoundedProgram),
             "truncate aggregate ownership storage without rejecting the "
             "graph")) {
    return false;
  }
  const CanonicalSyncOwnershipDiscoveryStatistics &aggregateStatistics =
      aggregateBoundedProgram->getOwnershipDiscoveryStatistics();
  if (!check(aggregateBoundedProgram->getGraph()
                 .getBasicOwnershipCertificates()
                 .empty(),
             "omit ownership certificates beyond the aggregate bound") ||
      !check(aggregateStatistics.truncated,
             "report aggregate ownership-discovery truncation") ||
      !check(aggregateStatistics.nodeReferences == 0,
             "retain no partial ownership node-reference census")) {
    return false;
  }

  std::string deepSource = ownershipSource;
  const std::size_t deepName = deepSource.find("@branch_l0");
  const std::string loopMarker =
      "        scf.for %i = %c0 to %limit step %c1 {\n";
  if (!check(deepName != std::string::npos,
             "locate deep ownership fixture insertion points")) {
    return false;
  }
  deepSource.replace(deepName, std::string("@branch_l0").size(),
                     "@deep_branch_l0");
  const std::size_t deepLoop = deepSource.find(loopMarker);
  if (!check(deepLoop != std::string::npos,
             "locate deep ownership loop insertion point")) {
    return false;
  }
  std::string nestedControls;
  constexpr std::size_t ownershipControlDepth = 24;
  for (std::size_t depth = 0; depth < ownershipControlDepth; ++depth) {
    nestedControls.append(10 + depth * 2, ' ');
    nestedControls += "scf.if %condition {\n";
  }
  for (std::size_t depth = ownershipControlDepth; depth != 0; --depth) {
    nestedControls.append(10 + (depth - 1) * 2, ' ');
    nestedControls += "}\n";
  }
  deepSource.insert(deepLoop + loopMarker.size(), nestedControls);
  OwningOpRef<ModuleOp> deepModule =
      parseSourceString<ModuleOp>(deepSource, &context);
  if (!check(static_cast<bool>(deepModule),
             "parse deeply controlled ownership fixture")) {
    return false;
  }
  func::FuncOp deepFunction =
      deepModule->lookupSymbol<func::FuncOp>("deep_branch_l0");
  CanonicalSyncAnalysisOptions deepReferenceOptions;
  deepReferenceOptions.maximumBasicOwnershipInspections = 1U << 28;
  FailureOr<CanonicalSyncProgram> deepReference =
      buildCanonicalSyncProgram(deepFunction, deepReferenceOptions);
  if (!check(succeeded(deepReference) &&
                 !deepReference->getGraph()
                      .getBasicOwnershipCertificates()
                      .empty() &&
                 !deepReference->getOwnershipDiscoveryStatistics().truncated,
             "recognize ownership with a complete deep traversal budget")) {
    return false;
  }
  CanonicalSyncAnalysisOptions deepBoundedOptions;
  deepBoundedOptions.maximumBasicOwnershipInspections =
      program->getOwnershipDiscoveryStatistics().inspections;
  FailureOr<CanonicalSyncProgram> deepBounded =
      buildCanonicalSyncProgram(deepFunction, deepBoundedOptions);
  if (!check(
          succeeded(deepBounded) &&
              deepBounded->getGraph().getBasicOwnershipCertificates().empty() &&
              deepBounded->getOwnershipDiscoveryStatistics().truncated &&
              deepBounded->getOwnershipDiscoveryStatistics().inspections ==
                  deepBoundedOptions.maximumBasicOwnershipInspections,
          "charge independent control/scope depth before recognition")) {
    return false;
  }

  std::string siblingSource = ownershipSource;
  const std::size_t siblingName = siblingSource.find("@branch_l0");
  if (!check(siblingName != std::string::npos,
             "locate unscheduled-sibling ownership fixture")) {
    return false;
  }
  siblingSource.replace(siblingName, std::string("@branch_l0").size(),
                        "@sibling_branch_l0");
  const std::size_t siblingLoop = siblingSource.find(loopMarker);
  if (!check(siblingLoop != std::string::npos,
             "locate unscheduled-sibling loop insertion point")) {
    return false;
  }
  std::string unscheduledSiblings;
  constexpr std::size_t ownershipSiblingCount = 64;
  for (std::size_t index = 0; index < ownershipSiblingCount; ++index) {
    unscheduledSiblings += "        %unused" + std::to_string(index) +
                           " = arith.constant " + std::to_string(index + 2) +
                           " : index\n";
  }
  siblingSource.insert(siblingLoop, unscheduledSiblings);
  OwningOpRef<ModuleOp> siblingModule =
      parseSourceString<ModuleOp>(siblingSource, &context);
  if (!check(static_cast<bool>(siblingModule),
             "parse unscheduled-sibling ownership fixture")) {
    return false;
  }
  func::FuncOp siblingFunction =
      siblingModule->lookupSymbol<func::FuncOp>("sibling_branch_l0");
  siblingFunction.getBody().front().invalidateOpOrder();
  FailureOr<CanonicalSyncProgram> siblingReference =
      buildCanonicalSyncProgram(siblingFunction);
  const bool siblingRecognized =
      succeeded(siblingReference) &&
      !siblingReference->getGraph().getBasicOwnershipCertificates().empty() &&
      !siblingReference->getOwnershipDiscoveryStatistics().truncated;
  if (!check(siblingRecognized,
             "recognize ownership across unscheduled block siblings")) {
    return false;
  }
  siblingFunction.getBody().front().invalidateOpOrder();
  CanonicalSyncAnalysisOptions siblingBoundedOptions;
  siblingBoundedOptions.maximumBasicOwnershipInspections =
      program->getOwnershipDiscoveryStatistics().inspections;
  FailureOr<CanonicalSyncProgram> siblingBounded =
      buildCanonicalSyncProgram(siblingFunction, siblingBoundedOptions);
  if (!check(
          succeeded(siblingBounded) &&
              siblingBounded->getGraph()
                  .getBasicOwnershipCertificates()
                  .empty() &&
              siblingBounded->getOwnershipDiscoveryStatistics().truncated &&
              siblingBounded->getOwnershipDiscoveryStatistics().inspections ==
                  siblingBoundedOptions.maximumBasicOwnershipInspections,
          "charge unscheduled block siblings before ownership recognition")) {
    return false;
  }

  std::string overlapSource = ownershipSource;
  const std::size_t overlapName = overlapSource.find("@branch_l0");
  const std::string allocationMarker =
      "        %acc = pto.alloc_tile addr = %addr0 :\n";
  if (!check(overlapName != std::string::npos,
             "locate overlapping ownership fixture insertion points")) {
    return false;
  }
  overlapSource.replace(overlapName, std::string("@branch_l0").size(),
                        "@overlap_branch_l0");
  const auto eraseOverlapMarker = [&](StringRef marker) {
    const std::size_t position = overlapSource.find(marker.str());
    if (position == std::string::npos) {
      return false;
    }
    overlapSource.erase(position, marker.size());
    return true;
  };
  const bool flattenedBranch =
      eraseOverlapMarker("          scf.if %condition {\n") &&
      eraseOverlapMarker("          } else {\n");
  if (!check(flattenedBranch,
             "flatten the overlap fixture branch for a direct-only control")) {
    return false;
  }
  const std::string flattenedExtract =
      "          pto.textract ins(%left_source, %c0, %c64";
  const std::string closingBranch = "          }\n" + flattenedExtract;
  const std::size_t closingPosition = overlapSource.find(closingBranch);
  if (!check(closingPosition != std::string::npos,
             "locate flattened overlap branch exit")) {
    return false;
  }
  overlapSource.replace(closingPosition, closingBranch.size(),
                        flattenedExtract);
  const std::size_t overlapAllocation = overlapSource.find(allocationMarker);
  if (!check(overlapAllocation != std::string::npos,
             "locate overlapping ownership allocation insertion point")) {
    return false;
  }
  overlapSource.insert(
      overlapAllocation,
      "        %addr8192 = arith.constant 8192 : i64\n"
      "        %left_overlap = pto.alloc_tile addr = %addr8192 :\n"
      "          !pto.tile_buf<left, 128x64xf16, slayout=row_major>\n");
  const std::size_t adjustedLoop = overlapSource.find(loopMarker);
  overlapSource.insert(
      adjustedLoop + loopMarker.size(),
      "          pto.textract ins(%left_source, %c0, %c0 :\n"
      "            !pto.tile_buf<mat, 128x512xf16, blayout=col_major,\n"
      "              slayout=row_major>, index, index)\n"
      "            outs(%left_overlap : !pto.tile_buf<left, 128x64xf16,\n"
      "              slayout=row_major>)\n");
  OwningOpRef<ModuleOp> overlapModule =
      parseSourceString<ModuleOp>(overlapSource, &context);
  if (!check(static_cast<bool>(overlapModule),
             "parse overlapping ownership fixture")) {
    return false;
  }
  func::FuncOp overlapFunction =
      overlapModule->lookupSymbol<func::FuncOp>("overlap_branch_l0");
  FailureOr<CanonicalSyncProgram> overlapProgram =
      buildCanonicalSyncProgram(overlapFunction);
  if (!check(succeeded(overlapProgram),
             "retain the direct graph for overlapping ownership storage")) {
    return false;
  }

  std::string directOverlapSource = overlapSource;
  const std::size_t directOverlapName =
      directOverlapSource.find("@overlap_branch_l0");
  const std::size_t directOverlapLoop = directOverlapSource.find(loopMarker);
  const std::string loopExitMarker = "        }\n        return\n";
  const std::size_t directOverlapExit =
      directOverlapSource.find(loopExitMarker);
  if (!check(directOverlapName != std::string::npos &&
                 directOverlapLoop != std::string::npos &&
                 directOverlapExit != std::string::npos,
             "locate direct-only overlap control markers")) {
    return false;
  }
  directOverlapSource.replace(directOverlapName,
                              std::string("@overlap_branch_l0").size(),
                              "@direct_overlap_l0");
  directOverlapSource.erase(directOverlapLoop, loopMarker.size());
  const std::size_t adjustedDirectExit =
      directOverlapSource.find(loopExitMarker);
  directOverlapSource.replace(adjustedDirectExit, loopExitMarker.size(),
                              "        return\n");
  OwningOpRef<ModuleOp> directOverlapModule =
      parseSourceString<ModuleOp>(directOverlapSource, &context);
  if (!check(static_cast<bool>(directOverlapModule),
             "parse direct-only overlap control")) {
    return false;
  }
  func::FuncOp directOverlapFunction =
      directOverlapModule->lookupSymbol<func::FuncOp>("direct_overlap_l0");
  FailureOr<CanonicalSyncProgram> directOverlapProgram =
      buildCanonicalSyncProgram(directOverlapFunction);
  if (!check(succeeded(directOverlapProgram),
             "build direct-only overlap control graph")) {
    return false;
  }
  CanonicalSyncBuildOptions directFallbackOptions;
  directFallbackOptions.patterns.catalogMode =
      CanonicalSyncCatalogMode::StrictMinimalDirect;
  directFallbackOptions.patterns.enabledMechanismFamilies = 0;
  directFallbackOptions.patterns.enableDirectPairs = false;
  directFallbackOptions.patterns.enableConflictCoreRepair = false;
  CanonicalSyncProblemBuildResult directFallback =
      buildCanonicalSyncPreciseProblem(*directOverlapProgram,
                                       directFallbackOptions);
  bool hasPartialOverlap = false;
  const auto &overlapAccesses = overlapProgram->getGraph().getStorageAccesses();
  for (std::size_t first = 0; first < overlapAccesses.size(); ++first) {
    for (std::size_t second = first + 1; second < overlapAccesses.size();
         ++second) {
      const SyncCoverStorageAccess &left = overlapAccesses[first];
      const SyncCoverStorageAccess &right = overlapAccesses[second];
      const bool differentExtent = left.extent.begin != right.extent.begin ||
                                   left.extent.end != right.extent.end;
      hasPartialOverlap |= left.domain == right.domain && differentExtent &&
                           left.extent.begin < right.extent.end &&
                           right.extent.begin < left.extent.end;
    }
  }
  const CanonicalSyncMechanismOriginMask directOrigins =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectTargetedBarrier) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectForwardRecurrenceEvent) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectReleaseRecurrenceProtocol) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectBalancedTargetFenceEvent);
  const bool directOnlyCatalog =
      directFallback && directFallback.problem &&
      llvm::all_of(directFallback.problem->getMechanisms(),
                   [&](const CanonicalSyncMechanism &mechanism) {
                     return mechanism.originMask != 0 &&
                            (mechanism.originMask & ~directOrigins) == 0;
                   });
  CanonicalSyncSelection directSelection;
  CanonicalSyncVerifiedPlan directVerification;
  if (directOnlyCatalog) {
    directSelection = selectCanonicalSyncPatterns(*directFallback.problem);
    directVerification =
        verifyCanonicalSyncSelection(*directFallback.problem, directSelection);
  }
  if (!check(hasPartialOverlap &&
                 overlapProgram->getGraph()
                     .getBasicOwnershipCertificates()
                     .empty() &&
                 !overlapProgram->getGraph().getDemands().empty() &&
                 !overlapProgram->getOwnershipDiscoveryStatistics().truncated &&
                 !directOverlapProgram->getGraph().getDemands().empty() &&
                 directOnlyCatalog && directSelection && directVerification,
             "reject partial ownership overlap while retaining direct "
             "cover and fresh verification")) {
    return false;
  }

  CanonicalSyncBuildOptions options;
  options.patterns.enabledMechanismFamilies = canonicalSyncMechanismFamilyBit(
      CanonicalSyncMechanismFamily::BasicOwnership);
  options.patterns.enableDirectPairs = false;
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  if (!check(precise && precise.problem,
             "build exhaustive L0 ownership candidate catalog")) {
    return false;
  }
  const CanonicalSyncMechanismOriginMask l0Origin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::BasicOwnershipL0OperandProtocol);
  const auto mechanism = llvm::find_if(precise.problem->getMechanisms(),
                                       [&](const CanonicalSyncMechanism &m) {
                                         return (m.originMask & l0Origin) != 0;
                                       });
  if (!check(mechanism != precise.problem->getMechanisms().end() &&
                 mechanism->descriptor.actions.size() == 12 &&
                 llvm::count_if(mechanism->descriptor.actions,
                                [](const CanonicalSyncAction &action) {
                                  return action.anchor.kind ==
                                         SyncCoverAnchorKind::ControlEntry;
                                }) == 1 &&
                 llvm::count_if(mechanism->descriptor.actions,
                                [](const CanonicalSyncAction &action) {
                                  return action.anchor.kind ==
                                         SyncCoverAnchorKind::ControlExit;
                                }) == 1,
             "synthesize one shared wait/set pair for branch alternatives")) {
    return false;
  }
  if (!verifyExactAndOneLessProtocolWork(
          *precise.problem, mechanism->id,
          "verify basic ownership at its exact work bound",
          "reject basic ownership at its one-less work bound")) {
    return false;
  }
  CanonicalSyncMechanismDescriptor reordered = mechanism->descriptor;
  std::reverse(reordered.supplies.begin(), reordered.supplies.end());
  SyncCoverCoverageWorkBudget reorderedWork;
  const CanonicalSyncProblemResult reorderedResult =
      precise.problem->verifyMechanismDescriptor(mechanism->id, reordered,
                                                 &reorderedWork);
  if (!check(static_cast<bool>(reorderedResult),
             "accept reordered ownership supplies as the same certificate")) {
    return false;
  }
  SyncCoverCoverageWorkBudget reorderedExact(reorderedWork.workUnits);
  SyncCoverCoverageWorkBudget reorderedOneLess(reorderedWork.workUnits - 1);
  if (!check(static_cast<bool>(precise.problem->verifyMechanismDescriptor(
                 mechanism->id, reordered, &reorderedExact)),
             "sort reordered ownership supplies at the exact work bound") ||
      !check(precise.problem
                     ->verifyMechanismDescriptor(mechanism->id, reordered,
                                                 &reorderedOneLess)
                     .error == CanonicalSyncProblemError::LimitExceeded,
             "stop reordered ownership sorting one work unit below")) {
    return false;
  }
  CanonicalSyncMechanismDescriptor alteredBinding = reordered;
  const bool hasComparisonFixture = alteredBinding.supplies.size() > 1 &&
                                    program->getGraph().getDemands().size() > 1;
  if (!check(hasComparisonFixture,
             "build a multi-supply ownership comparison fixture")) {
    return false;
  }
  const SyncCoverDemandId originalDemand =
      alteredBinding.supplies.front().allowedDemands.front();
  const SyncCoverDemandId replacementDemand =
      originalDemand + 1 < program->getGraph().getDemands().size()
          ? originalDemand + 1
          : 0;
  alteredBinding.supplies.front().allowedDemands = {replacementDemand};
  SyncCoverCoverageWorkBudget alteredWork;
  const CanonicalSyncProblemResult alteredResult =
      precise.problem->verifyMechanismDescriptor(mechanism->id, alteredBinding,
                                                 &alteredWork);
  if (!check(
          alteredResult.error ==
                  CanonicalSyncProblemError::UnverifiedProtocol &&
              alteredWork.workUnits != 0,
          "reject an unmatched ownership binding after exhaustive matching")) {
    return false;
  }
  SyncCoverCoverageWorkBudget alteredOneLess(alteredWork.workUnits - 1);
  if (!check(precise.problem
                     ->verifyMechanismDescriptor(mechanism->id, alteredBinding,
                                                 &alteredOneLess)
                     .error == CanonicalSyncProblemError::LimitExceeded,
             "stop an unmatched ownership comparison at its one-less bound")) {
    return false;
  }
  CanonicalSyncMechanismDescriptor alteredAction = mechanism->descriptor;
  const std::size_t actionLane = alteredAction.actions.front().eventLane;
  alteredAction.actions.front().eventLane =
      actionLane + 1 < alteredAction.eventUses.front().width ? actionLane + 1
                                                             : 0;
  if (!check(precise.problem
                     ->verifyMechanismDescriptor(mechanism->id, alteredAction)
                     .error == CanonicalSyncProblemError::UnverifiedProtocol,
             "reject an altered ownership action before common admission")) {
    return false;
  }
  SyncCoverCoverageWorkBudget baseOwnershipWork;
  if (!check(static_cast<bool>(precise.problem->verifyMechanism(
                 mechanism->id, &baseOwnershipWork)),
             "measure base ownership factory work")) {
    return false;
  }

  std::string unrelatedSource = ownershipSource;
  const std::size_t unrelatedName = unrelatedSource.find("@branch_l0");
  const std::string accumulatorMarker =
      "        %acc = pto.alloc_tile addr = %addr0 :\n";
  if (!check(unrelatedName != std::string::npos,
             "locate unrelated ownership function") ||
      !check(unrelatedSource.find(accumulatorMarker) != std::string::npos,
             "locate unrelated ownership allocation") ||
      !check(unrelatedSource.find(loopMarker) != std::string::npos,
             "locate unrelated ownership loop entry")) {
    return false;
  }
  unrelatedSource.replace(unrelatedName, std::string("@branch_l0").size(),
                          "@unrelated_branch_l0");
  unrelatedSource.insert(
      unrelatedSource.find(accumulatorMarker),
      "        %addr49152 = arith.constant 49152 : i64\n"
      "        %left_unrelated = pto.alloc_tile addr = %addr49152 :\n"
      "          !pto.tile_buf<left, 128x64xf16, slayout=row_major>\n");
  unrelatedSource.insert(
      unrelatedSource.find(loopMarker),
      "        scf.if %condition {\n"
      "          pto.textract ins(%left_source, %c0, %c0 :\n"
      "            !pto.tile_buf<mat, 128x512xf16, blayout=col_major,\n"
      "              slayout=row_major>, index, index)\n"
      "            outs(%left_unrelated : !pto.tile_buf<left, 128x64xf16,\n"
      "              slayout=row_major>)\n"
      "        } else {\n"
      "          pto.textract ins(%left_source, %c0, %c0 :\n"
      "            !pto.tile_buf<mat, 128x512xf16, blayout=col_major,\n"
      "              slayout=row_major>, index, index)\n"
      "            outs(%left_unrelated : !pto.tile_buf<left, 128x64xf16,\n"
      "              slayout=row_major>)\n"
      "        }\n");
  OwningOpRef<ModuleOp> unrelatedModule =
      parseSourceString<ModuleOp>(unrelatedSource, &context);
  if (!check(static_cast<bool>(unrelatedModule),
             "parse ownership with unrelated scheduled storage")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> unrelatedProgram = buildCanonicalSyncProgram(
      unrelatedModule->lookupSymbol<func::FuncOp>("unrelated_branch_l0"));
  CanonicalSyncProblemBuildResult unrelatedPrecise;
  if (succeeded(unrelatedProgram)) {
    unrelatedPrecise =
        buildCanonicalSyncPreciseProblem(*unrelatedProgram, options);
  }
  const bool builtUnrelatedProblem = succeeded(unrelatedProgram) &&
                                     unrelatedPrecise &&
                                     unrelatedPrecise.problem;
  if (!check(builtUnrelatedProblem,
             "build ownership with unrelated scheduled storage")) {
    return false;
  }
  const auto unrelatedMechanism =
      llvm::find_if(unrelatedPrecise.problem->getMechanisms(),
                    [&](const CanonicalSyncMechanism &candidate) {
                      return (candidate.originMask & l0Origin) != 0 &&
                             candidate.descriptor.actions.size() ==
                                 mechanism->descriptor.actions.size();
                    });
  if (!check(unrelatedMechanism !=
                 unrelatedPrecise.problem->getMechanisms().end(),
             "retain the original ownership recipe with unrelated storage")) {
    return false;
  }
  SyncCoverCoverageWorkBudget unrelatedOwnershipWork;
  const CanonicalSyncProblemResult unrelatedVerified =
      unrelatedPrecise.problem->verifyMechanism(unrelatedMechanism->id,
                                                &unrelatedOwnershipWork);
  const auto anchorDimension = [](const SyncCoverGraph &graph) {
    std::size_t guardIncidences = 0;
    for (const SyncCoverNode &node : graph.getNodes()) {
      guardIncidences += node.guard.literals.size();
    }
    std::size_t controlAlternatives = 0;
    for (const SyncCoverControl &control : graph.getControls()) {
      controlAlternatives += control.alternatives;
    }
    return graph.getNodes().size() * graph.getScopes().size() +
           guardIncidences + controlAlternatives + graph.getNodes().size() +
           4 * graph.getScopes().size();
  };
  const std::size_t baseAnchorDimension = anchorDimension(program->getGraph());
  const std::size_t unrelatedAnchorDimension =
      anchorDimension(unrelatedProgram->getGraph());
  const std::size_t minimumAnchorGrowth =
      mechanism->descriptor.actions.size() *
      (unrelatedAnchorDimension - baseAnchorDimension);
  if (!check(unrelatedVerified &&
                 unrelatedProgram->getGraph().getNodes().size() >
                     program->getGraph().getNodes().size() &&
                 unrelatedProgram->getGraph().getStorageAccesses().size() >
                     program->getGraph().getStorageAccesses().size() &&
                 unrelatedProgram->getGraph().getControls().size() >
                     program->getGraph().getControls().size() &&
                 unrelatedOwnershipWork.workUnits >=
                     baseOwnershipWork.workUnits + minimumAnchorGrowth,
             "charge unrelated guarded nodes, controls, accesses, and anchor "
             "resolution")) {
    return false;
  }
  if (!verifyExactAndOneLessProtocolWork(
          *unrelatedPrecise.problem, unrelatedMechanism->id,
          "verify guarded ControlEntry/Exit anchors at the exact work bound",
          "reject guarded ControlEntry/Exit anchors one work unit below")) {
    return false;
  }

  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(*precise.problem);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(*precise.problem, selection);
  return check(selection && verified,
               "select and freshly verify exhaustive L0 ownership");
}

bool testOwnershipDoesNotHideProducerOverwrite() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<cube>} {
      func.func @overwrite(
          %limit: index,
          %lhs: !pto.tile_buf<left, 32x32xf16,
                              blayout=row_major, slayout=row_major>,
          %rhs: !pto.tile_buf<right, 32x32xf16,
                              blayout=row_major, slayout=col_major>,
          %dst: !pto.partition_tensor_view<32x32xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %addr0 = arith.constant 0 : i64
        %acc = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<acc, 32x32xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>
        scf.for %outer = %c0 to %limit step %c1 {
          scf.for %inner = %c0 to %limit step %c1 {
            pto.tmatmul ins(%lhs, %rhs :
              !pto.tile_buf<left, 32x32xf16,
                            blayout=row_major, slayout=row_major>,
              !pto.tile_buf<right, 32x32xf16,
                            blayout=row_major, slayout=col_major>)
              outs(%acc : !pto.tile_buf<acc, 32x32xf32,
                                        blayout=col_major,
                                        slayout=row_major, fractal=1024>)
            pto.tmatmul ins(%lhs, %rhs :
              !pto.tile_buf<left, 32x32xf16,
                            blayout=row_major, slayout=row_major>,
              !pto.tile_buf<right, 32x32xf16,
                            blayout=row_major, slayout=col_major>)
              outs(%acc : !pto.tile_buf<acc, 32x32xf32,
                                        blayout=col_major,
                                        slayout=row_major, fractal=1024>)
          }
          pto.tstore ins(%acc : !pto.tile_buf<acc, 32x32xf32,
                                             blayout=col_major,
                                             slayout=row_major,
                                             fractal=1024>)
            outs(%dst : !pto.partition_tensor_view<32x32xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse producer-overwrite ownership fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("overwrite"));
  if (!check(succeeded(program), "build producer-overwrite ownership graph")) {
    return false;
  }
  const std::uint32_t matrix = static_cast<std::uint32_t>(PipelineType::PIPE_M);
  std::optional<SyncCoverDemandId> overwriteDemand;
  std::optional<SyncCoverDemandId> recurrenceOverwriteDemand;
  for (auto [demandId, demand] :
       llvm::enumerate(program->getGraph().getDemands())) {
    if (program->getGraph().getNodes()[demand.source].resource == matrix &&
        program->getGraph().getNodes()[demand.target].resource == matrix &&
        llvm::is_contained(demand.provenanceKinds,
                           SyncCoverDemandKind::MemoryWAW)) {
      if (demand.distance == 0 && !overwriteDemand) {
        overwriteDemand = demandId;
      } else if (demand.distance != 0 && !recurrenceOverwriteDemand) {
        recurrenceOverwriteDemand = demandId;
      }
    }
  }
  if (!check(overwriteDemand.has_value(),
             "retain a same-slot matrix overwrite demand") ||
      !check(recurrenceOverwriteDemand.has_value(),
             "retain the cross-iteration matrix overwrite demand")) {
    return false;
  }

  CanonicalSyncBuildOptions options;
  options.patterns.enabledMechanismFamilies = canonicalSyncMechanismFamilyBit(
      CanonicalSyncMechanismFamily::BasicOwnership);
  options.patterns.enableDirectPairs = false;
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  if (!check(precise && precise.problem,
             "build producer-overwrite ownership catalog")) {
    return false;
  }
  const CanonicalSyncMechanismOriginMask ownershipOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::BasicOwnershipAccumulatorProtocol);
  const CanonicalSyncMechanismOriginMask barrierOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectTargetedBarrier);
  bool sawOwnership = false;
  bool ownershipClaimsOverwrite = false;
  bool ownershipClaimsRecurrenceOverwrite = false;
  bool directBarrierClaimsOverwrite = false;
  const SyncCoverDemand &overwrite =
      program->getGraph().getDemands()[*overwriteDemand];
  const SyncCoverDemand &recurrenceOverwrite =
      program->getGraph().getDemands()[*recurrenceOverwriteDemand];
  for (const CanonicalSyncMechanism &mechanism :
       precise.problem->getMechanisms()) {
    const bool ownership = (mechanism.originMask & ownershipOrigin) != 0;
    const bool barrier = (mechanism.originMask & barrierOrigin) != 0;
    sawOwnership |= ownership;
    for (const CanonicalSyncSupplyBinding &supply :
         mechanism.descriptor.supplies) {
      const bool sameEdge = supply.edge.source == overwrite.source &&
                            supply.edge.target == overwrite.target &&
                            supply.edge.scope == overwrite.scope &&
                            supply.edge.distance == overwrite.distance;
      const bool admitsOverwrite =
          sameEdge &&
          (supply.allowedDemands.empty() ||
           llvm::is_contained(supply.allowedDemands, *overwriteDemand));
      ownershipClaimsOverwrite |= ownership && admitsOverwrite;
      directBarrierClaimsOverwrite |= barrier && admitsOverwrite;
      const bool sameRecurrenceEdge =
          supply.edge.source == recurrenceOverwrite.source &&
          supply.edge.target == recurrenceOverwrite.target &&
          supply.edge.scope == recurrenceOverwrite.scope &&
          supply.edge.distance == recurrenceOverwrite.distance;
      const bool admitsRecurrenceOverwrite =
          sameRecurrenceEdge &&
          (supply.allowedDemands.empty() ||
           llvm::is_contained(supply.allowedDemands,
                              *recurrenceOverwriteDemand));
      ownershipClaimsRecurrenceOverwrite |=
          ownership && admitsRecurrenceOverwrite;
    }
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(*precise.problem);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(*precise.problem, selection);
  const bool selectedDirectBarrier =
      selection &&
      llvm::any_of(
          selection.mechanisms, [&](CanonicalSyncMechanismId mechanismId) {
            return mechanismId < precise.problem->getMechanisms().size() &&
                   (precise.problem->getMechanisms()[mechanismId].originMask &
                    barrierOrigin) != 0;
          });
  return check(sawOwnership,
               "recognize the surrounding accumulator ownership lifecycle") &&
         check(!ownershipClaimsOverwrite,
               "do not supply producer-to-producer WAW through ownership") &&
         check(ownershipClaimsRecurrenceOverwrite,
               "retain lifecycle coverage for cross-iteration overwrites") &&
         check(directBarrierClaimsOverwrite,
               "retain a direct barrier fallback for the overwrite") &&
         check(selection && verified,
               "select and freshly verify the overwrite-safe plan") &&
         check(selectedDirectBarrier,
               "select the targeted matrix barrier for the overwrite");
}

bool testGenericRecurrenceWithoutOwnershipDiscovery() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @generic_recurrence(
          %input: !pto.partition_tensor_view<16x16xf32>, %limit: index)
          attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %addr0 = arith.constant 0 : i64
        %one = arith.constant 1.000000e+00 : f32
        %shared = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        scf.for %i = %c0 to %limit step %c1 {
          pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
            outs(%shared : !pto.tile_buf<vec, 16x16xf32>)
          pto.tmuls ins(%shared, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
            outs(%shared : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse generic recurrence ownership ablation")) {
    return false;
  }
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("generic_recurrence");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build generic recurrence program")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  // This test inspects the complete recurrence mechanism catalog itself;
  // demand-basis behavior is covered independently below.
  options.enableDemandBasisReduction = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(problem),
             "build generic recurrence problem without ownership")) {
    return false;
  }
  const auto isGenericRecurrence = [](const CanonicalSyncMechanism &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
           mechanism.descriptor.eventUses.size() == 1 &&
           llvm::any_of(mechanism.descriptor.supplies,
                        [](const CanonicalSyncSupplyBinding &supply) {
                          return supply.edge.distance == 1;
                        });
  };
  if (!check(llvm::any_of((*problem)->getMechanisms(), isGenericRecurrence),
             "generate a generic prime-body-drain recurrence event")) {
    return false;
  }
  const SyncCoverGraph &graph = program->getGraph();
  const auto isExactDrain = [&](const CanonicalSyncMechanism &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier &&
           llvm::any_of(
               mechanism.descriptor.supplies,
               [&](const CanonicalSyncSupplyBinding &supply) {
                 return supply.edge.distance == 1 &&
                        graph.getNodes()[supply.edge.source].resource !=
                            graph.getNodes()[supply.edge.target].resource &&
                        (supply.proof == CanonicalSyncSupplyProof::
                                             TargetLocalPipeDrainAction ||
                         supply.proof == CanonicalSyncSupplyProof::
                                             SourceLocalPipeDrainAction ||
                         supply.proof ==
                             CanonicalSyncSupplyProof::LoopCarryPipeDrain);
               });
  };
  if (!check(!llvm::any_of((*problem)->getMechanisms(), isExactDrain),
             "exclude naked cross-resource distance-one drains")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem);
  const bool selectedExactRecurrence =
      selection && llvm::any_of(selection.mechanisms, [&](auto mechanism) {
        const CanonicalSyncMechanism &selected =
            (*problem)->getMechanisms()[mechanism];
        return isExactDrain(selected) || isGenericRecurrence(selected);
      });
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(**problem, selection);
  if (!check(selectedExactRecurrence,
             "select an exact distance-one recurrence cover") ||
      !check(static_cast<bool>(verified),
             "finalize generic recurrence from certified coverage") ||
      !check(succeeded(runCanonicalSync(function, options)),
             "materialize generic recurrence without ownership")) {
    return false;
  }
  std::size_t setCount = 0;
  std::size_t waitCount = 0;
  std::size_t targetedBarrierCount = 0;
  std::size_t pipeAllBarrierCount = 0;
  function.walk([&](Operation *operation) {
    setCount += operation->getName().getStringRef() == "pto.set_flag";
    waitCount += operation->getName().getStringRef() == "pto.wait_flag";
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      const bool bodyBarrier = barrier->hasAttr("pto.canonical_sync") &&
                               !barrier->hasAttr("pto.auto_sync_tail_barrier");
      targetedBarrierCount +=
          bodyBarrier && barrier.getPipe().getPipe() != PIPE::PIPE_ALL;
      pipeAllBarrierCount +=
          bodyBarrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
    }
  });
  return check(setCount == waitCount &&
                   (setCount != 0 || targetedBarrierCount != 0) &&
                   pipeAllBarrierCount == 0,
               "emit a balanced recurrence plan without PIPE_ALL");
}

bool testGuardedEndpointUsesSourceLocalCompletionEvent() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @uncoverable_guarded_endpoint(
          %input: !pto.partition_tensor_view<16x16xf32>, %condition: i1)
          attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
        %addr0 = arith.constant 0 : i64
        %one = arith.constant 1.000000e+00 : f32
        %shared = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
          outs(%shared : !pto.tile_buf<vec, 16x16xf32>)
        scf.if %condition {
          pto.tmuls ins(%shared, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
            outs(%shared : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse uncoverable guarded kernel")) {
    return false;
  }
  OwningOpRef<Operation *> exactVerificationClone(module->clone());
  OwningOpRef<Operation *> belowVerificationClone(module->clone());
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("uncoverable_guarded_endpoint");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build uncoverable guarded program")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  if (!check(precise && precise.problem,
             "cover the guarded cross-pipe demand in the precise catalog")) {
    return false;
  }
  const CanonicalSyncMechanismOriginMask targetLocalOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::TargetLocalFenceEvent);
  const bool hasTargetLocalOrigin =
      llvm::any_of(precise.problem->getMechanisms(),
                   [&](const CanonicalSyncMechanism &mechanism) {
                     return (mechanism.originMask & targetLocalOrigin) != 0;
                   });
  CanonicalSyncBuildOptions explicitDefaultOptions = options;
  explicitDefaultOptions.patterns.enabledMechanismFamilies =
      kDefaultCanonicalSyncMechanismFamilies;
  CanonicalSyncProblemBuildResult explicitDefault =
      buildCanonicalSyncPreciseProblem(*program, explicitDefaultOptions);
  if (!check(hasTargetLocalOrigin,
             "classify guarded completeness as a target-local fence") ||
      !check(explicitDefault && explicitDefault.problem &&
                 precise.problem->hasSameCandidatePrefix(
                     *explicitDefault.problem),
             "make the default catalog identical to its explicit mask")) {
    return false;
  }

  CanonicalSyncBuildOptions withoutTargetLocalOptions = options;
  withoutTargetLocalOptions.patterns.enabledMechanismFamilies &=
      ~canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::TargetLocalFence);
  CanonicalSyncBuildOptions coreOptions = options;
  coreOptions.patterns.catalogMode =
      CanonicalSyncCatalogMode::StrictMinimalDirect;
  coreOptions.patterns.enabledMechanismFamilies = 0;
  coreOptions.patterns.enableDirectPairs = false;
  coreOptions.patterns.enableConflictCoreRepair = false;
  CanonicalSyncProblemBuildResult withoutTargetLocal;
  CanonicalSyncProblemBuildResult core;
  CanonicalSyncProblemBuildResult mechanical;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    withoutTargetLocal =
        buildCanonicalSyncPreciseProblem(*program, withoutTargetLocalOptions);
    core = buildCanonicalSyncPreciseProblem(*program, coreOptions);
    CanonicalSyncBuildOptions mechanicalOptions = coreOptions;
    mechanicalOptions.patterns.catalogMode =
        CanonicalSyncCatalogMode::MechanicalDirect;
    mechanical = buildCanonicalSyncPreciseProblem(*program, mechanicalOptions);
  }
  const CanonicalSyncMechanismOriginMask sourceLocalOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::SourceLocalCompletionEvent);
  const bool hasSourceLocalFallback =
      withoutTargetLocal && withoutTargetLocal.problem &&
      llvm::any_of(withoutTargetLocal.problem->getMechanisms(),
                   [&](const CanonicalSyncMechanism &mechanism) {
                     return (mechanism.originMask & sourceLocalOrigin) != 0;
                   });
  if (!check(hasSourceLocalFallback,
             "fall back to a balanced source-local completion event") ||
      !check(!core && !core.problem,
             "fail strict-direct when no exact direct cut exists") ||
      !check(!mechanical && !mechanical.problem,
             "fail mechanical-direct when no exact direct cut exists")) {
    return false;
  }
  CanonicalSyncComparisonReport report;
  options.reportCallback = [&](const CanonicalSyncComparisonReport &actual) {
    report = actual;
    return success();
  };
  if (!check(succeeded(runCanonicalSync(function, options)),
             "materialize guarded target-local completeness")) {
    return false;
  }
  std::size_t generatedPipeAllBackstops = 0;
  std::size_t generatedTargetedDrains = 0;
  std::size_t generatedSets = 0;
  std::size_t generatedWaits = 0;
  function.walk([&](Operation *operation) {
    const StringRef name = operation->getName().getStringRef();
    generatedSets += name == "pto.set_flag";
    generatedWaits += name == "pto.wait_flag";
  });
  function.walk([&](BarrierOp barrier) {
    const bool generated = barrier->hasAttr("pto.canonical_sync") &&
                           !barrier->hasAttr("pto.auto_sync_tail_barrier");
    generatedPipeAllBackstops +=
        generated && barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
    generatedTargetedDrains +=
        generated && barrier.getPipe().getPipe() != PIPE::PIPE_ALL;
  });
  if (!check(generatedPipeAllBackstops == 0 && generatedTargetedDrains == 1 &&
                 generatedSets == 1 && generatedWaits == 1,
             "drain and publish one balanced target-local event") ||
      !check(report.function == "uncoverable_guarded_endpoint" &&
                 report.graphNodes != 0 && report.uniqueDemandRows != 0 &&
                 report.strategies.size() == 1 &&
                 report.strategies.front().verified &&
                 !report.strategies.front().usedLocalizedPipeAll &&
                 report.strategies.front().emittedEventSets == 1 &&
                 report.strategies.front().emittedEventWaits == 1 &&
                 report.strategies.front().emittedTargetedBarriers == 1 &&
                 report.strategies.front().emittedPipeAllBarriers == 0 &&
                 report.strategies.front().verificationWorkUnits != 0 &&
                 report.strategies.front().predictedSyncInstructions != 0 &&
                 report.strategies.front().planSignature != 0,
             "report the freshly verified target-local event plan")) {
    return false;
  }
  const std::size_t exactVerificationWork =
      report.strategies.front().verificationWorkUnits;
  ModuleOp exactModule = cast<ModuleOp>(*exactVerificationClone);
  func::FuncOp exactFunction =
      exactModule.lookupSymbol<func::FuncOp>("uncoverable_guarded_endpoint");
  CanonicalSyncBuildOptions exactOptions = options;
  exactOptions.reportCallback = {};
  exactOptions.maximumVerificationWorkUnits = exactVerificationWork;
  if (!check(succeeded(runCanonicalSync(exactFunction, exactOptions)),
             "accept final verification at its reported exact work bound")) {
    return false;
  }
  ModuleOp belowModule = cast<ModuleOp>(*belowVerificationClone);
  func::FuncOp belowFunction =
      belowModule.lookupSymbol<func::FuncOp>("uncoverable_guarded_endpoint");
  CanonicalSyncBuildOptions belowOptions = exactOptions;
  --belowOptions.maximumVerificationWorkUnits;
  const std::string belowBefore = printOperation(belowFunction);
  bool rejectedBelow = false;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    rejectedBelow = failed(runCanonicalSync(belowFunction, belowOptions));
  }
  return check(rejectedBelow,
               "reject final verification one unit below its exact bound") &&
         check(printOperation(belowFunction) == belowBefore,
               "preserve IR when bounded final verification fails");
}

bool testMinimalDirectCatalogIsCompleteAndNeverFallsBack() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<vector>} {
      func.func @direct_chain(
          %input: !pto.partition_tensor_view<16x16xf32>,
          %secondInput: !pto.partition_tensor_view<16x16xf32>,
          %output: !pto.partition_tensor_view<16x16xf32>) {
        %addr0 = arith.constant 0 : i64
        %addr1024 = arith.constant 1024 : i64
        %firstTile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        %secondTile = pto.alloc_tile addr = %addr1024 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%secondInput : !pto.partition_tensor_view<16x16xf32>)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tadd ins(%firstTile, %secondTile :
          !pto.tile_buf<vec, 16x16xf32>, !pto.tile_buf<vec, 16x16xf32>)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
          outs(%output : !pto.partition_tensor_view<16x16xf32>)
        return
      }
      func.func @direct_scarcity(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>) {
        %addr0 = arith.constant 0 : i64
        %addr1024 = arith.constant 1024 : i64
        %one = arith.constant 1.000000e+00 : f32
        %firstTile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        %secondTile = pto.alloc_tile addr = %addr1024 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%first : !pto.partition_tensor_view<16x16xf32>)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%firstTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%secondTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @direct_loop(
          %input: !pto.partition_tensor_view<16x16xf32>, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %addr0 = arith.constant 0 : i64
        %one = arith.constant 1.000000e+00 : f32
        %tile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        scf.for %iv = %c0 to %limit step %c1 {
          pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
            outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
          pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
            outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module),
             "parse minimal direct catalog fixtures")) {
    return false;
  }

  CanonicalSyncBuildOptions options;
  options.patterns.catalogMode = CanonicalSyncCatalogMode::StrictMinimalDirect;
  options.patterns.enabledMechanismFamilies = 0;
  options.patterns.enableDirectPairs = false;
  options.patterns.enableConflictCoreRepair = false;
  options.selection.strategy =
      CanonicalSyncSelectionStrategy::ActionAwareSingleton;
  func::FuncOp chain = module->lookupSymbol<func::FuncOp>("direct_chain");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(chain);
  if (!check(succeeded(program), "build minimal direct catalog graph")) {
    return false;
  }
  CanonicalSyncProblemBuildResult problem =
      buildCanonicalSyncPreciseProblem(*program, options);
  if (!check(problem && problem.problem,
             "build complete minimal direct catalog")) {
    return false;
  }
  CanonicalSyncProblemBuildResult forbiddenRepair;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    forbiddenRepair = buildCanonicalSyncRepairProblem(
        *program, *problem.problem, options, {});
  }
  if (!check(!forbiddenRepair && !forbiddenRepair.problem &&
                 forbiddenRepair.status.error ==
                     CanonicalSyncProblemError::InvalidPattern,
             "reject repair-catalog extension of strict-direct problems")) {
    return false;
  }

  const CanonicalSyncMechanismOriginMask directOrigins =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectTargetedBarrier) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectDistanceZeroEvent) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectForwardRecurrenceEvent) |
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::DirectReleaseRecurrenceProtocol);
  const bool fullDemandUniverse =
      problem.problem->getDemands().size() ==
          program->getGraph().getDemands().size() &&
      problem.problem->getObligationDemands().size() ==
          program->getGraph().getDemands().size();
  const bool singletonOnly =
      problem.problem->getPatterns().size() ==
          problem.problem->getMechanisms().size() &&
      llvm::all_of(problem.problem->getPatterns(),
                   [](const CanonicalSyncPattern &pattern) {
                     return pattern.kind ==
                                CanonicalSyncPatternKind::Singleton &&
                            pattern.members.size() == 1;
                   });
  const bool directOnly =
      llvm::all_of(problem.problem->getMechanisms(),
                   [&](const CanonicalSyncMechanism &mechanism) {
                     return mechanism.originMask != 0 &&
                            (mechanism.originMask & ~directOrigins) == 0;
                   });
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(*problem.problem, options.selection);
  CanonicalSyncBuildOptions mechanicalOptions = options;
  mechanicalOptions.patterns.catalogMode =
      CanonicalSyncCatalogMode::MechanicalDirect;
  CanonicalSyncProblemBuildResult mechanicalProblem =
      buildCanonicalSyncPreciseProblem(*program, mechanicalOptions);
  CanonicalSyncSelection mechanicalSelection;
  if (mechanicalProblem && mechanicalProblem.problem) {
    mechanicalSelection = selectAllCanonicalSyncSingletonMechanisms(
        *mechanicalProblem.problem,
        mechanicalOptions.selection.maximumWorkUnits);
  }
  if (!check(fullDemandUniverse,
             "retain every unique hazard row in minimal direct mode") ||
      !check(singletonOnly && directOnly,
             "restrict minimal direct mode to direct singleton columns") ||
      !check(mechanicalProblem && mechanicalProblem.problem &&
                 problem.problem->hasSameCandidatePrefix(
                     *mechanicalProblem.problem) &&
                 mechanicalProblem.problem->hasSameCandidatePrefix(
                     *problem.problem),
             "ground the same direct catalog in strict and mechanical modes") ||
      !check(selection &&
                 verifyCanonicalSyncSelection(*problem.problem, selection),
             "select and freshly verify the minimal direct cover") ||
      !check(mechanicalProblem && mechanicalProblem.problem &&
                 mechanicalSelection &&
                 mechanicalSelection.mechanisms.size() ==
                     mechanicalProblem.problem->getMechanisms().size() &&
                 mechanicalSelection.statistics.deletionEvaluations == 0 &&
                 verifyCanonicalSyncSelection(*mechanicalProblem.problem,
                                              mechanicalSelection),
             "retain and freshly verify every mechanical direct recipe") ||
      !check(selection.mechanisms.size() <
                 mechanicalSelection.mechanisms.size(),
             "remove a redundant direct cut through grounded rectangle "
             "coverage")) {
    return false;
  }

  func::FuncOp loop = module->lookupSymbol<func::FuncOp>("direct_loop");
  FailureOr<CanonicalSyncProgram> loopProgram = buildCanonicalSyncProgram(loop);
  if (!check(succeeded(loopProgram), "build repeated direct-event graph")) {
    return false;
  }
  CanonicalSyncProblemBuildResult strictLoopProblem;
  CanonicalSyncProblemBuildResult mechanicalLoopProblem;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    strictLoopProblem = buildCanonicalSyncPreciseProblem(*loopProgram, options);
    mechanicalLoopProblem =
        buildCanonicalSyncPreciseProblem(*loopProgram, mechanicalOptions);
  }
  if (!check(!strictLoopProblem && !strictLoopProblem.problem &&
                 !mechanicalLoopProblem && !mechanicalLoopProblem.problem,
             "reject the same token-unsafe loop catalog in both direct-only "
             "modes")) {
    return false;
  }

  func::FuncOp scarcity = module->lookupSymbol<func::FuncOp>("direct_scarcity");
  const std::string before = printOperation(scarcity);
  const auto rejectsScarcity = [&](CanonicalSyncBuildOptions scarcityOptions,
                                   StringRef mode) {
    scarcityOptions.eventIdBudget = 1;
    CanonicalSyncComparisonReport scarcityReport;
    scarcityOptions.reportCallback =
        [&](const CanonicalSyncComparisonReport &report) {
          scarcityReport = report;
          return success();
        };
    bool sawScarcityDiagnostic = false;
    LogicalResult scarcityResult = success();
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        sawScarcityDiagnostic |=
            diagnostic.str().find("exhausted the event-ID budget") !=
            std::string::npos;
        return success();
      });
      scarcityResult = runCanonicalSync(scarcity, scarcityOptions);
    }
    const bool reportedInfeasible =
        scarcityReport.strategies.size() == 1 &&
        scarcityReport.strategies.front().error ==
            CanonicalSyncSelectionError::ResourceInfeasible &&
        !scarcityReport.strategies.front().verified &&
        !scarcityReport.strategies.front().usedLocalizedPipeAll;
    return check(failed(scarcityResult) && sawScarcityDiagnostic,
                 std::string("fail ") + mode.str() +
                     " explicitly when direct events exceed the ID budget") &&
           check(printOperation(scarcity) == before,
                 std::string("leave IR unchanged after ") + mode.str() +
                     " event scarcity") &&
           check(reportedInfeasible,
                 std::string("report ") + mode.str() +
                     " scarcity without a verified fallback plan");
  };
  return rejectsScarcity(options, "strict-direct") &&
         rejectsScarcity(mechanicalOptions, "mechanical-direct");
}

bool testDemandBasisReductionIsBoundedAndTruncating() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
      func.func @basis(%tile: !pto.tile_buf<vec, 16x16xf32>) {
        %one = arith.constant 1.000000e+00 : f32
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @basis_large(%tile: !pto.tile_buf<vec, 16x16xf32>) {
        %one = arith.constant 1.000000e+00 : f32
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile, %one : !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse demand-basis fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("basis"));
  if (!check(succeeded(program), "build demand-basis fixture")) {
    return false;
  }
  CanonicalSyncBuildOptions reducedOptions;
  CanonicalSyncProblemBuildResult reduced =
      buildCanonicalSyncPreciseProblem(*program, reducedOptions);
  if (!check(reduced && reduced.problem, "build reduced demand basis")) {
    return false;
  }
  CanonicalSyncBuildOptions boundedOptions = reducedOptions;
  boundedOptions.maximumDemandBasisGroupEdges = 2;
  CanonicalSyncProblemBuildResult bounded =
      buildCanonicalSyncPreciseProblem(*program, boundedOptions);
  if (!check(bounded && bounded.problem, "truncate bounded demand basis")) {
    return false;
  }
  CanonicalSyncBuildOptions exactOptions = reducedOptions;
  exactOptions.maximumDemandBasisGroupEdges = 3;
  exactOptions.maximumDemandBasisReachabilityWords = 6;
  exactOptions.maximumDemandBasisReductionWork = 151;
  CanonicalSyncProblemBuildResult exact =
      buildCanonicalSyncPreciseProblem(*program, exactOptions);
  if (!check(exact && exact.problem, "accept exact demand-basis bounds")) {
    return false;
  }
  CanonicalSyncBuildOptions wordBelowOptions = exactOptions;
  --wordBelowOptions.maximumDemandBasisReachabilityWords;
  CanonicalSyncProblemBuildResult wordBelow =
      buildCanonicalSyncPreciseProblem(*program, wordBelowOptions);
  if (!check(wordBelow && wordBelow.problem,
             "truncate one word below the demand-basis bound")) {
    return false;
  }
  CanonicalSyncBuildOptions workBelowOptions = exactOptions;
  --workBelowOptions.maximumDemandBasisReductionWork;
  CanonicalSyncProblemBuildResult workBelow =
      buildCanonicalSyncPreciseProblem(*program, workBelowOptions);
  if (!check(workBelow && workBelow.problem,
             "truncate one unit below the demand-basis work bound")) {
    return false;
  }
  const std::size_t obligations =
      reduced.problem->getObligationDemands().size();
  FailureOr<CanonicalSyncProgram> largeProgram = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("basis_large"));
  if (!check(succeeded(largeProgram), "build large demand-basis fixture")) {
    return false;
  }
  CanonicalSyncBuildOptions largeReferenceOptions;
  CanonicalSyncProblemBuildResult largeReference =
      buildCanonicalSyncPreciseProblem(*largeProgram, largeReferenceOptions);
  if (!check(largeReference && largeReference.problem,
             "build large demand-basis reference")) {
    return false;
  }
  const std::size_t largeEdges =
      largeReference.problem->getObligationDemands().size();
  const std::size_t maximumNodes = largeEdges * 2;
  const std::size_t wordsPerRow = maximumNodes / 64 + (maximumNodes % 64 != 0);
  const std::size_t globalWork =
      4 * largeEdges + largeProgram->getGraph().getScopes().size();
  const std::size_t groupWork =
      2 * maximumNodes * maximumNodes + largeEdges * largeEdges +
      largeEdges * maximumNodes + largeEdges * (wordsPerRow + 2) +
      maximumNodes * wordsPerRow + 3 * maximumNodes + 2 * largeEdges;
  CanonicalSyncBuildOptions largeExactOptions = largeReferenceOptions;
  largeExactOptions.maximumDemandBasisReductionWork = globalWork + groupWork;
  CanonicalSyncProblemBuildResult largeExact =
      buildCanonicalSyncPreciseProblem(*largeProgram, largeExactOptions);
  CanonicalSyncBuildOptions largeBelowOptions = largeExactOptions;
  --largeBelowOptions.maximumDemandBasisReductionWork;
  CanonicalSyncProblemBuildResult largeBelow =
      buildCanonicalSyncPreciseProblem(*largeProgram, largeBelowOptions);
  return check(obligations == 3,
               "construct the three dense distance-zero obligations") &&
         check(reduced.problem->getDemands().size() == 2 &&
                   !reduced.problem->wasBasisReductionTruncated(),
               "remove only the transitive selection row") &&
         check(exact.problem->getDemands().size() == 2 &&
                   !exact.problem->wasBasisReductionTruncated(),
               "reduce at the exact word and work bounds") &&
         check(bounded.problem->getDemands().size() == obligations &&
                   bounded.problem->wasBasisReductionTruncated(),
               "retain all rows when the pre-allocation edge cap is hit") &&
         check(wordBelow.problem->getDemands().size() == obligations &&
                   wordBelow.problem->wasBasisReductionTruncated(),
               "retain all rows one word below the bound") &&
         check(workBelow.problem->getDemands().size() == obligations &&
                   workBelow.problem->wasBasisReductionTruncated(),
               "retain all rows one work unit below the bound") &&
         check(largeEdges > obligations && largeExact && largeExact.problem &&
                   !largeExact.problem->wasBasisReductionTruncated() &&
                   largeExact.problem->getDemands().size() < largeEdges,
               "reduce a larger adversarial group at its exact work bound") &&
         check(largeBelow && largeBelow.problem &&
                   largeBelow.problem->wasBasisReductionTruncated() &&
                   largeBelow.problem->getDemands().size() == largeEdges,
               "retain the larger group one work unit below its bound");
}

bool testSourcePrefixGenerationIsBoundedAndTruncating() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<vector>} {
      func.func @prefix(
          %gm0: !pto.partition_tensor_view<16x16xf32>,
          %gm1: !pto.partition_tensor_view<16x16xf32>,
          %gm2: !pto.partition_tensor_view<16x16xf32>,
          %gm3: !pto.partition_tensor_view<16x16xf32>,
          %a0: !pto.tile_buf<vec, 16x16xf32>,
          %a1: !pto.tile_buf<vec, 16x16xf32>,
          %a2: !pto.tile_buf<vec, 16x16xf32>,
          %a3: !pto.tile_buf<vec, 16x16xf32>,
          %o0: !pto.tile_buf<vec, 16x16xf32>,
          %o1: !pto.tile_buf<vec, 16x16xf32>,
          %o2: !pto.tile_buf<vec, 16x16xf32>,
          %o3: !pto.tile_buf<vec, 16x16xf32>) {
        pto.tload ins(%gm0 : !pto.partition_tensor_view<16x16xf32>)
          outs(%a0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%gm1 : !pto.partition_tensor_view<16x16xf32>)
          outs(%a1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%gm2 : !pto.partition_tensor_view<16x16xf32>)
          outs(%a2 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%gm3 : !pto.partition_tensor_view<16x16xf32>)
          outs(%a3 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%a0 : !pto.tile_buf<vec, 16x16xf32>)
          outs(%o0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%a1 : !pto.tile_buf<vec, 16x16xf32>)
          outs(%o1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%a2 : !pto.tile_buf<vec, 16x16xf32>)
          outs(%o2 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%a3 : !pto.tile_buf<vec, 16x16xf32>)
          outs(%o3 : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @basis_ineligible(
          %gm: !pto.partition_tensor_view<16x16xf32>, %condition: i1) {
        %address = arith.constant 0 : i64
        %tile = pto.alloc_tile addr = %address :
          !pto.tile_buf<vec, 16x16xf32>
        scf.if %condition {
          pto.tload ins(%gm : !pto.partition_tensor_view<16x16xf32>)
            outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
          pto.tabs ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
            outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse source-prefix fixture")) {
    return false;
  }
  CanonicalSyncAnalysisOptions analysis;
  analysis.gmAliasPolicy = CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias;
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("prefix"), analysis);
  if (!check(succeeded(program), "build source-prefix fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> ineligibleProgram = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("basis_ineligible"), analysis);
  if (!check(succeeded(ineligibleProgram),
             "build all-ineligible demand-basis fixture")) {
    return false;
  }
  const std::size_t ineligibleRows =
      ineligibleProgram->getGraph().getDemands().size();
  CanonicalSyncBuildOptions ineligibleExactOptions;
  ineligibleExactOptions.analysis = analysis;
  ineligibleExactOptions.maximumDemandBasisReductionWork =
      4 * ineligibleRows + ineligibleProgram->getGraph().getScopes().size();
  CanonicalSyncProblemBuildResult ineligibleExact =
      buildCanonicalSyncPreciseProblem(*ineligibleProgram,
                                       ineligibleExactOptions);
  CanonicalSyncBuildOptions ineligibleBelowOptions = ineligibleExactOptions;
  --ineligibleBelowOptions.maximumDemandBasisReductionWork;
  CanonicalSyncProblemBuildResult ineligibleBelow =
      buildCanonicalSyncPreciseProblem(*ineligibleProgram,
                                       ineligibleBelowOptions);
  if (!check(ineligibleExact && ineligibleExact.problem,
             "build the exact-bound all-ineligible basis problem") ||
      !check(!ineligibleExact.problem->wasBasisReductionTruncated(),
             "finish all-ineligible basis accounting at the exact bound") ||
      !check(ineligibleExact.problem->getDemands().size() == ineligibleRows,
             "retain every all-ineligible row at the exact bound") ||
      !check(ineligibleBelow && ineligibleBelow.problem,
             "build the below-bound all-ineligible basis problem") ||
      !check(ineligibleBelow.problem->wasBasisReductionTruncated(),
             "truncate all-ineligible accounting one work unit below") ||
      !check(ineligibleBelow.problem->getDemands().size() == ineligibleRows,
             "retain all-ineligible rows one work unit below the bound")) {
    return false;
  }
  CanonicalSyncBuildOptions referenceOptions;
  referenceOptions.analysis = analysis;
  // Exercise source-prefix catalog bounds against the complete demand set;
  // selection-basis reduction has separate exact-bound coverage above.
  referenceOptions.enableDemandBasisReduction = false;
  CanonicalSyncProblemBuildResult preciseReference =
      buildCanonicalSyncPreciseProblem(*program, referenceOptions);
  if (!check(preciseReference && preciseReference.problem,
             "build precise source-prefix fixture")) {
    return false;
  }
  const bool preciseHasCrossSourceDrain = llvm::any_of(
      preciseReference.problem->getMechanisms(),
      [&](const CanonicalSyncMechanism &mechanism) {
        return llvm::any_of(
            mechanism.descriptor.supplies,
            [&](const CanonicalSyncSupplyBinding &binding) {
              const bool sourceDrain =
                  binding.proof ==
                      CanonicalSyncSupplyProof::SourceLocalPipeDrainAction ||
                  binding.proof ==
                      CanonicalSyncSupplyProof::SourcePrefixPipeDrainAction;
              return sourceDrain &&
                     program->getGraph()
                             .getNodes()[binding.edge.source]
                             .resource != program->getGraph()
                                              .getNodes()[binding.edge.target]
                                              .resource;
            });
      });
  const auto buildRepair = [&](const CanonicalSyncBuildOptions &options) {
    CanonicalSyncProblemBuildResult precise =
        buildCanonicalSyncPreciseProblem(*program, options);
    if (!precise || !precise.problem) {
      return precise;
    }
    std::vector<CanonicalSyncMechanismId> conflictCore;
    for (const CanonicalSyncMechanism &mechanism :
         precise.problem->getMechanisms()) {
      if (llvm::any_of(mechanism.descriptor.supplies,
                       [](const CanonicalSyncSupplyBinding &binding) {
                         return binding.eventUse.has_value();
                       })) {
        conflictCore.push_back(mechanism.id);
      }
    }
    return buildCanonicalSyncRepairProblem(*program, *precise.problem, options,
                                           conflictCore);
  };
  CanonicalSyncProblemBuildResult reference = buildRepair(referenceOptions);
  if (!check(reference && reference.problem,
             "build reference source-prefix catalog")) {
    return false;
  }
  const CanonicalSyncPatternStatistics &statistics =
      reference.problem->getPatternStatistics();
  if (!check(statistics.sourcePrefixInspections != 0 &&
                 statistics.sourcePrefixCandidates != 0 &&
                 statistics.sourcePrefixIncidences != 0 &&
                 !statistics.sourcePrefixGenerationTruncated,
             "record complete source-prefix preparation")) {
    return false;
  }

  const auto buildLimited = [&](std::size_t inspections, std::size_t candidates,
                                std::size_t incidences) {
    CanonicalSyncBuildOptions options = referenceOptions;
    options.patterns.maximumSourcePrefixInspections = inspections;
    options.patterns.maximumSourcePrefixCandidates = candidates;
    options.patterns.maximumSourcePrefixIncidences = incidences;
    return buildRepair(options);
  };
  CanonicalSyncProblemBuildResult exact = buildLimited(
      statistics.sourcePrefixInspections, statistics.sourcePrefixCandidates,
      statistics.sourcePrefixIncidences);
  CanonicalSyncProblemBuildResult belowInspection = buildLimited(
      statistics.sourcePrefixInspections - 1, statistics.sourcePrefixCandidates,
      statistics.sourcePrefixIncidences);
  CanonicalSyncProblemBuildResult belowCandidate = buildLimited(
      statistics.sourcePrefixInspections, statistics.sourcePrefixCandidates - 1,
      statistics.sourcePrefixIncidences);
  CanonicalSyncProblemBuildResult belowIncidence = buildLimited(
      statistics.sourcePrefixInspections, statistics.sourcePrefixCandidates,
      statistics.sourcePrefixIncidences - 1);
  const auto truncatedAndVerified = [](const auto &built) {
    if (!built || !built.problem ||
        !built.problem->getPatternStatistics()
             .sourcePrefixGenerationTruncated) {
      std::cerr << "source-prefix bounded build: built="
                << static_cast<bool>(built)
                << ", problem=" << static_cast<bool>(built.problem)
                << ", truncated="
                << (built.problem ? built.problem->getPatternStatistics()
                                        .sourcePrefixGenerationTruncated
                                  : false)
                << '\n';
      return false;
    }
    const CanonicalSyncSelection selection =
        selectCanonicalSyncPatterns(*built.problem);
    const CanonicalSyncVerifiedPlan verified =
        verifyCanonicalSyncSelection(*built.problem, selection);
    if (!selection || !verified) {
      std::cerr << "source-prefix bounded selection error="
                << static_cast<unsigned>(selection.error)
                << ", verification error="
                << static_cast<unsigned>(verified.error) << '\n';
    }
    return selection && verified;
  };
  return check(!preciseHasCrossSourceDrain,
               "exclude cross-pipeline source drains from precise covers") &&
         check(exact && exact.problem &&
                   !exact.problem->getPatternStatistics()
                        .sourcePrefixGenerationTruncated,
               "accept exact source-prefix preparation limits") &&
         check(truncatedAndVerified(belowInspection),
               "truncate safely below the source-prefix inspection limit") &&
         check(truncatedAndVerified(belowCandidate),
               "truncate safely below the source-prefix candidate limit") &&
         check(truncatedAndVerified(belowIncidence),
               "truncate safely below the source-prefix incidence limit");
}

bool testConflictCoreRepairAvoidsPipeAll() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3",
                       pto.kernel_kind = #pto.kernel_kind<vector>} {
      func.func @scarcity_frontier(
          %first: !pto.partition_tensor_view<16x16xf32>,
          %second: !pto.partition_tensor_view<16x16xf32>) {
        %addr0 = arith.constant 0 : i64
        %addr1024 = arith.constant 1024 : i64
        %one = arith.constant 1.000000e+00 : f32
        %firstTile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        %secondTile = pto.alloc_tile addr = %addr1024 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%first : !pto.partition_tensor_view<16x16xf32>)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%second : !pto.partition_tensor_view<16x16xf32>)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%firstTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%firstTile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%secondTile, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%secondTile : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
      func.func @repair_pair_overlap(
          %input: !pto.partition_tensor_view<16x16xf32>,
          %output: !pto.partition_tensor_view<16x16xf32>) {
        %addr0 = arith.constant 0 : i64
        %tile = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%input : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tabs ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
          outs(%tile : !pto.tile_buf<vec, 16x16xf32>)
        pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf32>)
          outs(%output : !pto.partition_tensor_view<16x16xf32>)
        return
      }
      func.func @changed_core_driver(
          %input0: !pto.partition_tensor_view<16x16xf32>,
          %input1: !pto.partition_tensor_view<16x16xf32>,
          %input2: !pto.partition_tensor_view<16x16xf32>,
          %input3: !pto.partition_tensor_view<16x16xf32>,
          %input4: !pto.partition_tensor_view<16x16xf32>,
          %input5: !pto.partition_tensor_view<16x16xf32>,
          %input6: !pto.partition_tensor_view<16x16xf32>,
          %input7: !pto.partition_tensor_view<16x16xf32>) {
        %addr0 = arith.constant 0 : i64
        %addr1024 = arith.constant 1024 : i64
        %addr2048 = arith.constant 2048 : i64
        %addr3072 = arith.constant 3072 : i64
        %addr4096 = arith.constant 4096 : i64
        %addr5120 = arith.constant 5120 : i64
        %addr6144 = arith.constant 6144 : i64
        %addr7168 = arith.constant 7168 : i64
        %one = arith.constant 1.000000e+00 : f32
        %tile0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile1 = pto.alloc_tile addr = %addr1024 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile2 = pto.alloc_tile addr = %addr2048 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile3 = pto.alloc_tile addr = %addr3072 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile4 = pto.alloc_tile addr = %addr4096 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile5 = pto.alloc_tile addr = %addr5120 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile6 = pto.alloc_tile addr = %addr6144 :
          !pto.tile_buf<vec, 16x16xf32>
        %tile7 = pto.alloc_tile addr = %addr7168 :
          !pto.tile_buf<vec, 16x16xf32>
        pto.tload ins(%input0 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input1 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input2 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile2 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input3 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile3 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input4 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile4 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input5 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile5 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input6 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile6 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tload ins(%input7 : !pto.partition_tensor_view<16x16xf32>)
          outs(%tile7 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile0, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile0 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile1, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile1 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile2, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile2 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile3, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile3 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile4, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile4 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile5, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile5 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile6, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile6 : !pto.tile_buf<vec, 16x16xf32>)
        pto.tmuls ins(%tile7, %one :
          !pto.tile_buf<vec, 16x16xf32>, f32)
          outs(%tile7 : !pto.tile_buf<vec, 16x16xf32>)
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse scarcity-frontier kernel")) {
    return false;
  }
  OwningOpRef<Operation *> forcedFallbackClone(module->clone());
  OwningOpRef<Operation *> analysisOnlyClone(module->clone());
  OwningOpRef<Operation *> exactRepairClone(module->clone());
  OwningOpRef<Operation *> belowTrialRepairClone(module->clone());
  OwningOpRef<Operation *> belowWorkRepairClone(module->clone());
  OwningOpRef<Operation *> cleanupReferenceClone(module->clone());
  OwningOpRef<Operation *> cleanupExactClone(module->clone());
  OwningOpRef<Operation *> cleanupBelowClone(module->clone());
  OwningOpRef<Operation *> cleanupRetainedClone(module->clone());
  OwningOpRef<Operation *> changedCoreClone(module->clone());
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("scarcity_frontier");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build scarcity-frontier program")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  options.eventIdBudget = 1;
  options.patterns.maximumRepairFrontierInspections = 1;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(problem), "build scarcity-frontier problem")) {
    return false;
  }

  const CanonicalSyncSelection precise = selectCanonicalSyncPatterns(**problem);
  if (!check(precise.error == CanonicalSyncSelectionError::ResourceInfeasible,
             "one event ID rejects the overlapping precise plan")) {
    return false;
  }
  const std::vector<CanonicalSyncMechanismId> &conflictCore =
      precise.allocation.domains.front().liveMechanisms;
  CanonicalSyncProblemBuildResult partialRepair =
      buildCanonicalSyncRepairProblem(*program, **problem, options,
                                      {conflictCore.front()});
  const bool partialRepairStayedWithinCore =
      partialRepair &&
      partialRepair.problem->getPatternStatistics().repairFrontierInspections ==
          0 &&
      partialRepair.problem->getPatternStatistics().repairFrontierProposals ==
          0;
  if (!check(partialRepairStayedWithinCore,
             "do not generate frontiers from events outside the live core")) {
    return false;
  }

  ModuleOp changedCoreModule = cast<ModuleOp>(*changedCoreClone);
  changedCoreModule->setAttr("pto.target_arch",
                             StringAttr::get(&context, "a3"));
  func::FuncOp changedCoreFunction =
      changedCoreModule.lookupSymbol<func::FuncOp>("scarcity_frontier");
  FailureOr<CanonicalSyncProgram> changedCoreProgram =
      buildCanonicalSyncProgram(changedCoreFunction);
  CanonicalSyncProblemBuildResult changedCorePrecise =
      succeeded(changedCoreProgram)
          ? buildCanonicalSyncPreciseProblem(*changedCoreProgram, options)
          : CanonicalSyncProblemBuildResult{};
  CanonicalSyncSelection changedCoreSelection;
  if (changedCorePrecise) {
    changedCoreSelection =
        selectCanonicalSyncPatterns(*changedCorePrecise.problem);
  }
  const std::vector<CanonicalSyncMechanismId> changedConflictCore =
      changedCoreSelection.allocation.domains.empty()
          ? std::vector<CanonicalSyncMechanismId>{}
          : changedCoreSelection.allocation.domains.front().liveMechanisms;
  if (!check(changedCoreSelection.error ==
                     CanonicalSyncSelectionError::ResourceInfeasible &&
                 changedConflictCore.size() >= 2,
             "expose a second precise owner for changed-core repair")) {
    return false;
  }
  const CanonicalSyncMechanismId retainedOwner = changedConflictCore.front();
  CanonicalSyncProblemBuildResult retainedOwnerRepair =
      buildCanonicalSyncRepairProblem(
          *changedCoreProgram, *changedCorePrecise.problem, options,
          {retainedOwner}, changedCoreSelection.mechanisms);
  if (!check(retainedOwnerRepair &&
                 retainedOwnerRepair.repairCriticalDemandsByOwner.count(
                     retainedOwner) == 1 &&
                 !retainedOwnerRepair.repairCriticalDemandsByOwner
                      .at(retainedOwner)
                      .empty(),
             "retain the first owner's critical-demand provenance")) {
    return false;
  }
  std::vector<CanonicalSyncMechanismId> changedSelected =
      changedCoreSelection.mechanisms;
  changedSelected.erase(std::remove(changedSelected.begin(),
                                    changedSelected.end(), retainedOwner),
                        changedSelected.end());
  const CanonicalSyncMechanismId changedOwner = changedConflictCore.back();
  const std::vector<CanonicalSyncMechanismId> accumulatedCore = {retainedOwner,
                                                                 changedOwner};
  const auto retainedCriticalDemandMap =
      retainedOwnerRepair.repairCriticalDemandsByOwner;
  const std::vector<CanonicalSyncRepairCriticalDemandSeed>
      retainedCriticalDemands = {
          {retainedOwner, retainedCriticalDemandMap.at(retainedOwner)}};
  SyncCoverCoverageWorkBudget changedCoreWork;
  CanonicalSyncProblemBuildResult changedCoreRepair =
      buildCanonicalSyncRepairProblem(
          *changedCoreProgram, *changedCorePrecise.problem, options,
          accumulatedCore, changedSelected, &changedCoreWork,
          retainedCriticalDemands);
  const std::size_t exactChangedCoreWork = changedCoreWork.workUnits;
  SyncCoverCoverageWorkBudget exactChangedCoreBudget(exactChangedCoreWork);
  CanonicalSyncProblemBuildResult exactChangedCoreRepair =
      buildCanonicalSyncRepairProblem(
          *changedCoreProgram, *changedCorePrecise.problem, options,
          accumulatedCore, changedSelected, &exactChangedCoreBudget,
          retainedCriticalDemands);
  SyncCoverCoverageWorkBudget belowChangedCoreBudget(
      exactChangedCoreWork == 0 ? 0 : exactChangedCoreWork - 1);
  CanonicalSyncProblemBuildResult belowChangedCoreRepair =
      buildCanonicalSyncRepairProblem(
          *changedCoreProgram, *changedCorePrecise.problem, options,
          accumulatedCore, changedSelected, &belowChangedCoreBudget,
          retainedCriticalDemands);
  const bool changedCoreRetainsOwners =
      changedCoreRepair && exactChangedCoreRepair &&
      exactChangedCoreWork != 0 &&
      exactChangedCoreBudget.workUnits == exactChangedCoreWork &&
      !exactChangedCoreBudget.exhausted && !belowChangedCoreRepair &&
      belowChangedCoreRepair.status.error ==
          CanonicalSyncProblemError::LimitExceeded &&
      belowChangedCoreBudget.exhausted &&
      changedCoreRepair.repairCriticalDemandsByOwner.count(retainedOwner) ==
          1 &&
      changedCoreRepair.repairCriticalDemandsByOwner.at(retainedOwner) ==
          retainedCriticalDemandMap.at(retainedOwner) &&
      changedCoreRepair.repairCriticalDemandsByOwner.count(changedOwner) == 1 &&
      !changedCoreRepair.repairCriticalDemandsByOwner.at(changedOwner).empty();
  if (!check(changedCoreRetainsOwners,
             "carry forbidden-owner provenance into a changed repair core")) {
    return false;
  }

  const std::vector<CanonicalSyncMechanismId> syntheticRepairMechanisms = {
      10, 11, 12};
  const std::vector<CanonicalSyncMechanismId> syntheticOwnerRepairs = {10, 11};
  const std::vector<const std::vector<CanonicalSyncMechanismId> *>
      syntheticOwnerIndex = {&syntheticOwnerRepairs};
  const auto prepareSyntheticTrial = [&](SyncCoverCoverageWorkBudget *budget) {
    CanonicalSyncGreedyOptions trial;
    trial.forbiddenMechanisms = {5, 11, 12};
    const bool prepared = prepareCanonicalSyncRepairTrial(
        trial, syntheticRepairMechanisms, syntheticOwnerIndex, {}, {0}, false,
        {10}, budget);
    return std::make_pair(prepared, trial.forbiddenMechanisms);
  };
  SyncCoverCoverageWorkBudget syntheticRepairWork;
  const auto syntheticReference = prepareSyntheticTrial(&syntheticRepairWork);
  SyncCoverCoverageWorkBudget syntheticExact(syntheticRepairWork.workUnits);
  const auto syntheticAtExact = prepareSyntheticTrial(&syntheticExact);
  SyncCoverCoverageWorkBudget syntheticBelow(
      syntheticRepairWork.workUnits == 0 ? 0
                                         : syntheticRepairWork.workUnits - 1);
  const auto syntheticBelowBound = prepareSyntheticTrial(&syntheticBelow);
  if (!check(syntheticReference.first && syntheticAtExact.first &&
                 syntheticReference.second ==
                     std::vector<CanonicalSyncMechanismId>({5, 10, 12}) &&
                 syntheticAtExact.second == syntheticReference.second &&
                 !syntheticExact.exhausted && !syntheticBelowBound.first &&
                 syntheticBelow.exhausted,
             "preserve explicit repair-local exclusions at exact work bound")) {
    return false;
  }

  bool sawStaleSelectionDiagnostic = false;
  bool sawMismatchedProvenanceDiagnostic = false;
  CanonicalSyncProblemBuildResult staleSelectionRepair;
  CanonicalSyncProblemBuildResult mismatchedProvenanceRepair;
  const std::vector<CanonicalSyncMechanismId> changedOnlyCore = {changedOwner};
  {
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
      sawStaleSelectionDiagnostic |=
          diagnostic.str().find("does not match the precise catalog") !=
          std::string::npos;
      sawMismatchedProvenanceDiagnostic |=
          diagnostic.str().find("provenance does not match the repair core") !=
          std::string::npos;
      return success();
    });
    staleSelectionRepair = buildCanonicalSyncRepairProblem(
        *changedCoreProgram, *changedCorePrecise.problem, options,
        accumulatedCore, {changedCorePrecise.problem->getMechanisms().size()},
        nullptr, retainedCriticalDemands);
    mismatchedProvenanceRepair = buildCanonicalSyncRepairProblem(
        *changedCoreProgram, *changedCorePrecise.problem, options,
        changedOnlyCore, changedSelected, nullptr, retainedCriticalDemands);
  }
  if (!check(!staleSelectionRepair &&
                 staleSelectionRepair.status.error ==
                     CanonicalSyncProblemError::InvalidPattern &&
                 sawStaleSelectionDiagnostic && !mismatchedProvenanceRepair &&
                 mismatchedProvenanceRepair.status.error ==
                     CanonicalSyncProblemError::InvalidPattern &&
                 sawMismatchedProvenanceDiagnostic,
             "reject stale repair-local IDs and provenance outside the core")) {
    return false;
  }
  const auto rejectsMismatchedPrefix = [&](const CanonicalSyncBuildOptions
                                               &staleOptions,
                                           std::string_view message) {
    bool sawStaleCoreDiagnostic = false;
    CanonicalSyncProblemBuildResult staleRepair;
    {
      ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diagnostic) {
        sawStaleCoreDiagnostic |=
            diagnostic.str().find("does not match the precise catalog") !=
            std::string::npos;
        return success();
      });
      staleRepair = buildCanonicalSyncRepairProblem(*program, **problem,
                                                    staleOptions, conflictCore);
    }
    return check(!staleRepair &&
                     staleRepair.status.error ==
                         CanonicalSyncProblemError::InvalidPattern &&
                     sawStaleCoreDiagnostic,
                 message);
  };
  CanonicalSyncBuildOptions staleOptions = options;
  staleOptions.eventIdBudget = 2;
  if (!rejectsMismatchedPrefix(
          staleOptions,
          "reject a conflict core against a differently configured prefix")) {
    return false;
  }
  staleOptions = options;
  ++staleOptions.directPairs.pairCoverageLimits.maximumTotalWords;
  if (!rejectsMismatchedPrefix(
          staleOptions,
          "reject a repair prefix with a different pair total-word cap")) {
    return false;
  }
  staleOptions = options;
  ++staleOptions.directPairs.maximumPreparationWords;
  if (!rejectsMismatchedPrefix(
          staleOptions,
          "reject a repair prefix with a different pair preparation cap")) {
    return false;
  }
  staleOptions = options;
  ++staleOptions.problemLimits.maximumSingletonCoverageWords;
  if (!rejectsMismatchedPrefix(
          staleOptions,
          "reject a repair prefix with a different singleton coverage cap")) {
    return false;
  }
  for (CanonicalSyncMechanismFamily family :
       {CanonicalSyncMechanismFamily::L0OperandOwnership,
        CanonicalSyncMechanismFamily::BoundaryOwnership,
        CanonicalSyncMechanismFamily::HierarchicalOwnership}) {
    staleOptions = options;
    staleOptions.patterns.enabledMechanismFamilies ^=
        canonicalSyncMechanismFamilyBit(family);
    if (!rejectsMismatchedPrefix(
            staleOptions,
            "reject a repair prefix with a different ownership family")) {
      return false;
    }
  }
  CanonicalSyncBuildOptions truncatedOptions = options;
  truncatedOptions.patterns.maximumRepairFrontierInspections = 0;
  CanonicalSyncProblemBuildResult truncatedRepair =
      buildCanonicalSyncRepairProblem(*program, **problem, truncatedOptions,
                                      conflictCore);
  if (!check(static_cast<bool>(truncatedRepair),
             "retain the precise catalog when repair generation truncates")) {
    return false;
  }
  const CanonicalSyncPatternStatistics &truncatedStatistics =
      truncatedRepair.problem->getPatternStatistics();
  if (!check(
          truncatedRepair.problem->hasSameCandidatePrefix(**problem) &&
              truncatedStatistics.repairFrontierTruncated &&
              truncatedStatistics.repairFrontierInspections == 0 &&
              truncatedStatistics.repairFrontierProposals == 0,
          "truncate the optional repair batch before its first inspection")) {
    return false;
  }
  const CanonicalSyncProblemResult frozenMetadataWrite =
      truncatedRepair.problem->recordRepairFrontierGeneration(9, 9, false);
  if (!check(frozenMetadataWrite.error == CanonicalSyncProblemError::Frozen &&
                 truncatedRepair.problem->getPatternStatistics()
                     .repairFrontierTruncated,
             "keep repair metadata immutable after problem freeze")) {
    return false;
  }
  CanonicalSyncBuildOptions proposalTruncatedOptions = options;
  proposalTruncatedOptions.patterns.maximumRepairFrontierProposals = 0;
  CanonicalSyncProblemBuildResult proposalTruncatedRepair =
      buildCanonicalSyncRepairProblem(*program, **problem,
                                      proposalTruncatedOptions, conflictCore);
  if (!check(static_cast<bool>(proposalTruncatedRepair),
             "retain the precise catalog at the proposal bound")) {
    return false;
  }
  const CanonicalSyncPatternStatistics &proposalTruncatedStatistics =
      proposalTruncatedRepair.problem->getPatternStatistics();
  if (!check(proposalTruncatedStatistics.repairFrontierTruncated &&
                 proposalTruncatedStatistics.repairFrontierInspections == 1 &&
                 proposalTruncatedStatistics.repairFrontierProposals == 0,
             "reject the first proposal beyond a zero proposal budget")) {
    return false;
  }
  options.patterns.maximumRepairFrontierProposals = 1;
  CanonicalSyncProblemBuildResult repair = buildCanonicalSyncRepairProblem(
      *program, **problem, options, conflictCore);
  if (!check(static_cast<bool>(repair),
             "build repair candidates from the live conflict core")) {
    return false;
  }
  SyncCoverCoverageWorkBudget preparationReference;
  CanonicalSyncProblemBuildResult meteredRepair =
      buildCanonicalSyncRepairProblem(*program, **problem, options,
                                      conflictCore, {}, &preparationReference);
  const std::size_t exactPreparationWork = preparationReference.workUnits;
  SyncCoverCoverageWorkBudget exactPreparationBudget(exactPreparationWork);
  CanonicalSyncProblemBuildResult exactMeteredRepair =
      buildCanonicalSyncRepairProblem(*program, **problem, options,
                                      conflictCore, {},
                                      &exactPreparationBudget);
  SyncCoverCoverageWorkBudget belowPreparationBudget(
      exactPreparationWork == 0 ? 0 : exactPreparationWork - 1);
  CanonicalSyncProblemBuildResult belowMeteredRepair =
      buildCanonicalSyncRepairProblem(*program, **problem, options,
                                      conflictCore, {},
                                      &belowPreparationBudget);
  if (!check(meteredRepair && exactMeteredRepair && exactPreparationWork != 0,
             "meter incremental repair-catalog preparation") ||
      !check(meteredRepair.problem->hasSameCandidatePrefix(**problem),
             "preserve the frozen precise candidate prefix during repair") ||
      !check(&meteredRepair.problem->getExpansion() ==
                 &(*problem)->getExpansion(),
             "share the immutable expansion across repair catalogs") ||
      !check(!exactPreparationBudget.exhausted &&
                 exactPreparationBudget.workUnits == exactPreparationWork,
             "accept repair preparation at its exact shared-work bound") ||
      !check(!belowMeteredRepair && belowPreparationBudget.exhausted &&
                 belowMeteredRepair.status.error ==
                     CanonicalSyncProblemError::LimitExceeded,
             "stop repair preparation one shared-work unit below its bound")) {
    return false;
  }
  const CanonicalSyncPatternStatistics &repairStatistics =
      repair.problem->getPatternStatistics();
  if (!check(!repairStatistics.repairFrontierTruncated &&
                 repairStatistics.repairFrontierInspections == 1 &&
                 repairStatistics.repairFrontierProposals == 1,
             "admit the exact-bound repair proposal deterministically")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(*repair.problem);
  const std::size_t frontierMembers = static_cast<std::size_t>(std::count_if(
      selection.mechanisms.begin(), selection.mechanisms.end(),
      [&](CanonicalSyncMechanismId mechanism) {
        return repair.problem->getMechanisms()[mechanism].descriptor.kind ==
               CanonicalSyncMechanismKind::Barrier;
      }));
  const bool usesPipeAll = std::any_of(
      selection.mechanisms.begin(), selection.mechanisms.end(),
      [&](CanonicalSyncMechanismId mechanism) {
        return llvm::any_of(
            repair.problem->getMechanisms()[mechanism].descriptor.actions,
            [](const CanonicalSyncAction &action) {
              return action.kind == CanonicalSyncActionKind::Barrier &&
                     action.barrierKind == CanonicalSyncBarrierKind::All;
            });
      });
  CanonicalSyncProblemBuildResult fallback =
      buildCanonicalSyncPipeAllProblem(*program, options);
  const bool fallbackOnlyContainsPipeAll =
      fallback &&
      llvm::all_of(fallback.problem->getMechanisms(),
                   [](const CanonicalSyncMechanism &mechanism) {
                     return mechanism.descriptor.kind ==
                                CanonicalSyncMechanismKind::Barrier &&
                            llvm::all_of(
                                mechanism.descriptor.actions,
                                [](const CanonicalSyncAction &action) {
                                  return action.kind ==
                                             CanonicalSyncActionKind::Barrier &&
                                         action.barrierKind ==
                                             CanonicalSyncBarrierKind::All;
                                });
                   });
  const bool preciseHasRepairPattern = llvm::any_of(
      (*problem)->getPatterns(), [](const CanonicalSyncPattern &pattern) {
        return pattern.kind == CanonicalSyncPatternKind::RepairFrontier;
      });
  const bool repairHasRepairPattern = llvm::any_of(
      repair.problem->getPatterns(), [](const CanonicalSyncPattern &pattern) {
        return pattern.kind == CanonicalSyncPatternKind::RepairFrontier;
      });
  const bool catalogChecks =
      check(!preciseHasRepairPattern && repairHasRepairPattern,
            "keep repair candidates out of the precise catalog") &&
      check(fallbackOnlyContainsPipeAll,
            "keep PIPE_ALL in a barrier-only fallback problem") &&
      check(selection && frontierMembers == 1 && !usesPipeAll,
            "select one targeted barrier plus one frontier event") &&
      check(selection.allocation.domains.size() == 1 &&
                selection.allocation.domains.front().required == 1,
            "frontier fits the one-ID budget") &&
      check(static_cast<bool>(
                verifyCanonicalSyncSelection(*repair.problem, selection)),
            "finalize conflict-core repair from certified coverage");
  if (!catalogChecks) {
    return false;
  }

  CanonicalSyncComparisonReport repairReport;
  options.reportCallback = [&](const CanonicalSyncComparisonReport &report) {
    repairReport = report;
    return success();
  };
  if (!check(succeeded(runCanonicalSync(function, options)),
             "run conflict-core repair through production orchestration")) {
    return false;
  }
  const bool repairReported =
      repairReport.strategies.size() == 1 &&
      repairReport.strategies.front().verified &&
      !repairReport.strategies.front().usedLocalizedPipeAll &&
      repairReport.strategies.front().repairRounds == 1 &&
      repairReport.strategies.front().repairTrials != 0 &&
      repairReport.strategies.front().repairTrials <=
          options.maximumRepairTrials &&
      repairReport.strategies.front().repairWorkUnits != 0 &&
      repairReport.strategies.front().selectedEvents == 1 &&
      repairReport.strategies.front().emittedTargetedBarriers == 1 &&
      repairReport.strategies.front().emittedPipeAllBarriers == 0;
  if (!check(repairReported,
             "report the bounded, freshly verified production repair")) {
    return false;
  }
  std::size_t generatedTargetedBarriers = 0;
  function.walk([&](BarrierOp barrier) {
    const bool generated = barrier->hasAttr("pto.canonical_sync");
    const bool tail = barrier->hasAttr("pto.auto_sync_tail_barrier");
    generatedTargetedBarriers +=
        generated && !tail && barrier.getPipe().getPipe() != PIPE::PIPE_ALL;
  });
  if (!check(generatedTargetedBarriers == 1,
             "materialize the verified repair rather than PIPE_ALL")) {
    return false;
  }

  const CanonicalSyncStrategyReport &referenceRepair =
      repairReport.strategies.front();
  const auto runCloneWithReport =
      [&](OwningOpRef<Operation *> &clone, CanonicalSyncBuildOptions runOptions,
          CanonicalSyncComparisonReport &runReport) {
        ModuleOp cloneModule = cast<ModuleOp>(*clone);
        func::FuncOp cloneFunction =
            cloneModule.lookupSymbol<func::FuncOp>("scarcity_frontier");
        runOptions.reportCallback =
            [&](const CanonicalSyncComparisonReport &actual) {
              runReport = actual;
              return success();
            };
        return runCanonicalSync(cloneFunction, runOptions);
      };

  CanonicalSyncBuildOptions exactRepairOptions = options;
  exactRepairOptions.maximumRepairTrials = referenceRepair.repairTrials;
  exactRepairOptions.maximumRepairWorkUnits = referenceRepair.repairWorkUnits;
  CanonicalSyncComparisonReport exactRepairReport;
  if (!check(succeeded(runCloneWithReport(exactRepairClone, exactRepairOptions,
                                          exactRepairReport)),
             "accept aggregate repair work at the exact bound")) {
    return false;
  }
  const CanonicalSyncStrategyReport &exactRepair =
      exactRepairReport.strategies.front();
  if (!check(!exactRepair.repairBudgetExhausted &&
                 !exactRepair.usedLocalizedPipeAll &&
                 exactRepair.repairTrials == referenceRepair.repairTrials &&
                 exactRepair.repairWorkUnits == referenceRepair.repairWorkUnits,
             "complete all repair trials at their exact aggregate limits")) {
    return false;
  }

  CanonicalSyncBuildOptions belowTrialOptions = exactRepairOptions;
  belowTrialOptions.maximumRepairTrials = referenceRepair.repairTrials - 1;
  CanonicalSyncComparisonReport belowTrialReport;
  if (!check(succeeded(runCloneWithReport(belowTrialRepairClone,
                                          belowTrialOptions, belowTrialReport)),
             "stop repair at one trial below the exact bound") ||
      !check(belowTrialReport.strategies.front().repairBudgetExhausted &&
                 belowTrialReport.strategies.front().repairTrials ==
                     belowTrialOptions.maximumRepairTrials,
             "report the first repair trial rejected by the aggregate bound")) {
    return false;
  }

  CanonicalSyncBuildOptions belowWorkOptions = exactRepairOptions;
  belowWorkOptions.maximumRepairWorkUnits = referenceRepair.repairWorkUnits - 1;
  CanonicalSyncComparisonReport belowWorkReport;
  if (!check(succeeded(runCloneWithReport(belowWorkRepairClone,
                                          belowWorkOptions, belowWorkReport)),
             "stop repair at one work unit below the exact bound") ||
      !check(belowWorkReport.strategies.front().repairBudgetExhausted &&
                 belowWorkReport.strategies.front().repairWorkUnits <
                     referenceRepair.repairWorkUnits,
             "report aggregate repair work exhaustion exactly")) {
    return false;
  }

  CanonicalSyncBuildOptions cleanupReferenceOptions;
  cleanupReferenceOptions.eventIdBudget = 1;
  cleanupReferenceOptions.maximumRepairTrials = 0;
  CanonicalSyncComparisonReport cleanupReferenceReport;
  if (!check(succeeded(runCloneWithReport(cleanupReferenceClone,
                                          cleanupReferenceOptions,
                                          cleanupReferenceReport)),
             "measure the bounded backstop cleanup reference")) {
    return false;
  }
  const CanonicalSyncStrategyReport &cleanupReference =
      cleanupReferenceReport.strategies.front();
  if (!check(!cleanupReference.backstopDeletionTruncated &&
                 cleanupReference.backstopDeletionTrials != 0 &&
                 cleanupReference.backstopDeletionWorkUnits != 0,
             "complete the reference backstop cleanup")) {
    return false;
  }

  CanonicalSyncBuildOptions cleanupExactOptions = cleanupReferenceOptions;
  cleanupExactOptions.maximumBackstopDeletionTrials =
      cleanupReference.backstopDeletionTrials;
  cleanupExactOptions.maximumBackstopDeletionWorkUnits =
      cleanupReference.backstopDeletionWorkUnits;
  CanonicalSyncComparisonReport cleanupExactReport;
  if (!check(succeeded(runCloneWithReport(
                 cleanupExactClone, cleanupExactOptions, cleanupExactReport)),
             "accept backstop cleanup at the exact work bound")) {
    return false;
  }
  const CanonicalSyncStrategyReport &cleanupExact =
      cleanupExactReport.strategies.front();
  if (!check(!cleanupExact.backstopDeletionTruncated &&
                 cleanupExact.backstopDeletionTrials ==
                     cleanupReference.backstopDeletionTrials &&
                 cleanupExact.backstopDeletionWorkUnits ==
                     cleanupReference.backstopDeletionWorkUnits,
             "complete backstop cleanup at exact trial and work limits")) {
    return false;
  }

  CanonicalSyncBuildOptions cleanupBelowOptions = cleanupExactOptions;
  cleanupBelowOptions.maximumBackstopDeletionWorkUnits =
      cleanupReference.backstopDeletionWorkUnits - 1;
  CanonicalSyncComparisonReport cleanupBelowReport;
  if (!check(succeeded(runCloneWithReport(
                 cleanupBelowClone, cleanupBelowOptions, cleanupBelowReport)),
             "retain a verified backstop below the cleanup work bound") ||
      !check(
          cleanupBelowReport.strategies.front().backstopDeletionTruncated &&
              cleanupBelowReport.strategies.front().backstopDeletionTrials ==
                  cleanupReference.backstopDeletionTrials &&
              cleanupBelowReport.strategies.front().backstopDeletionWorkUnits ==
                  cleanupBelowOptions.maximumBackstopDeletionWorkUnits,
          "reject the first cleanup trial beyond the work bound")) {
    return false;
  }

  CanonicalSyncBuildOptions cleanupRetainedOptions = cleanupReferenceOptions;
  cleanupRetainedOptions.maximumBackstopDeletionTrials =
      cleanupReference.backstopDeletionTrials - 1;
  CanonicalSyncComparisonReport cleanupRetainedReport;
  if (!check(succeeded(runCloneWithReport(cleanupRetainedClone,
                                          cleanupRetainedOptions,
                                          cleanupRetainedReport)),
             "materialize the last plan verified before cleanup exhaustion") ||
      !check(printOperation(cast<ModuleOp>(*cleanupBelowClone)) ==
                 printOperation(cast<ModuleOp>(*cleanupRetainedClone)),
             "retain exactly the last verified backstop plan")) {
    return false;
  }

  ModuleOp forcedFallbackModule = cast<ModuleOp>(*forcedFallbackClone);
  func::FuncOp forcedFallback =
      forcedFallbackModule.lookupSymbol<func::FuncOp>("scarcity_frontier");
  CanonicalSyncBuildOptions forcedOptions;
  forcedOptions.eventIdBudget = 1;
  forcedOptions.maximumRepairTrials = 0;
  forcedOptions.maximumBackstopDeletionTrials = 0;
  CanonicalSyncComparisonReport forcedReport;
  forcedOptions.reportCallback =
      [&](const CanonicalSyncComparisonReport &report) {
        forcedReport = report;
        return success();
      };
  if (!check(succeeded(runCanonicalSync(forcedFallback, forcedOptions)),
             "fall back when the aggregate repair-trial budget is zero") ||
      !check(forcedReport.strategies.size() == 1 &&
                 forcedReport.strategies.front().usedLocalizedPipeAll &&
                 forcedReport.strategies.front().repairBudgetExhausted &&
                 forcedReport.strategies.front().repairTrials == 0 &&
                 forcedReport.strategies.front().backstopDeletionTruncated &&
                 forcedReport.strategies.front().backstopDeletionTrials == 0,
             "report bounded repair and backstop-deletion exhaustion")) {
    return false;
  }
  std::size_t generatedPipeAllBackstops = 0;
  forcedFallback.walk([&](BarrierOp barrier) {
    generatedPipeAllBackstops +=
        barrier->hasAttr("pto.canonical_sync") &&
        !barrier->hasAttr("pto.auto_sync_tail_barrier") &&
        barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
  });
  if (!check(generatedPipeAllBackstops == 2,
             "retain the verified full backstop when cleanup is bounded")) {
    return false;
  }

  ModuleOp analysisOnlyModule = cast<ModuleOp>(*analysisOnlyClone);
  func::FuncOp analysisOnly =
      analysisOnlyModule.lookupSymbol<func::FuncOp>("scarcity_frontier");
  CanonicalSyncBuildOptions analysisOptions = forcedOptions;
  analysisOptions.analysisOnly = true;
  CanonicalSyncComparisonReport analysisReport;
  analysisOptions.reportCallback =
      [&](const CanonicalSyncComparisonReport &report) {
        analysisReport = report;
        return success();
      };
  const std::string irBefore = printOperation(analysisOnly);
  if (!check(succeeded(runCanonicalSync(analysisOnly, analysisOptions)),
             "analyze complete strategy feasibility without mutation")) {
    return false;
  }
  const std::string irAfter = printOperation(analysisOnly);
  return check(irBefore == irAfter,
               "leave IR unchanged in analysis-only mode") &&
         check(analysisReport.strategies.size() == 3 &&
                   llvm::all_of(analysisReport.strategies,
                                [](const CanonicalSyncStrategyReport &report) {
                                  return report.verified &&
                                         report.usedLocalizedPipeAll;
                                }),
               "report the complete backstop path for every compared strategy");
}

} // namespace

int main() {
  const bool passed =
      testBuildsOneFrozenGraph() &&
      testOptionalLifecycleSynthesisTruncatesToDirectCatalog() &&
      testGenericLifecycleMaterializationIsBoundedAndBalanced() &&
      testMaterializationRejectsTamperedEventAllocations() &&
      testMacroBindingsAndHiddenReservations() &&
      testEmptyNestedLoopTimelineIsClamped() && testGmAliasPolicies() &&
      testGmAliasContracts() && testStructuredIssueFrontier() &&
      testDistanceTwoPhysicalSlotRecurrence() &&
      testA5MatrixLoopBoundaryProtocolRequiresSourcedEventContract() &&
      testDistanceTwoCrossRootSlotRecurrence() &&
      testTargetCapabilityProfilesAreVersionedAndConservative() &&
      testAccumulatorReadReadHardwareHazardIsRawDemand() &&
      testMmadIntrinsicRequiresExactAccumulator() &&
      testA3TargetCompletionCertificatesAreArchitectureQualified() &&
      testAnalysisLimitFailsClosed() &&
      testCanonicalHazardsAggregateBeforeGraphMutation() &&
      testFailClosedInputs() && testAcceptsDeclaredStorageProvenanceRoots() &&
      testRejectsOwnedSyncAndAcceptsFixedFence() &&
      testRejectsMalformedOwnedSynchronization() &&
      testFixedBarriersSupplyCompletionAndRemainUnowned() &&
      testRejectsFixedBarrierInsideLoop() &&
      testFixedBarrierInspectionBoundsAndPersistentControlState() &&
      testStructuralLimitsFailClosed() && testPeriodicBranchEvidence() &&
      testPhaseAwareRecurrenceDistances() &&
      testFirstIterationRecurrenceSuppression() &&
      testUnmodeledLoopVaryingControlsFailClosed() &&
      testUnsupportedBlockArgumentProvenanceFailsClosed() &&
      testSsaProvenanceTraversalIsBoundedAndIterative() &&
      testGuardedOwnershipVerificationWorkIsBounded() &&
      testBasicL0OwnershipSharesExhaustiveBranchBoundaries() &&
      testOwnershipDoesNotHideProducerOverwrite() &&
      testGenericRecurrenceWithoutOwnershipDiscovery() &&
      testGuardedEndpointUsesSourceLocalCompletionEvent() &&
      testMinimalDirectCatalogIsCompleteAndNeverFallsBack() &&
      testDemandBasisReductionIsBoundedAndTruncating() &&
      testSourcePrefixGenerationIsBoundedAndTruncating() &&
      testConflictCoreRepairAvoidsPipeAll();
  return passed ? 0 : 1;
}
