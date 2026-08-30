// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

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
#include <iostream>
#include <iterator>
#include <limits>
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
  return check(graph.isStructureFrozen(), "freeze authoritative graph") &&
         check(static_cast<bool>(graph.validate()), "validate adapter graph") &&
         check(graph.getNodes().size() == 2, "extract two scheduled nodes") &&
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
         check(llvm::none_of(graph.getStorageAccesses(),
                             [](const SyncCoverStorageAccess &access) {
                               return access.exactPhysical;
                             }),
               "keep unplanned arguments conservative");
}

bool testMaterializationRejectsTamperedEventAllocations() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @allocation_validation(
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
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(isolated, selection);
  const bool verifiedShape = selection && verified &&
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

  return rejectsPlan(std::move(wrongDomain),
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
         check(printOperation(function) == irBefore,
               "reject every tampered allocation before mutating IR");
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
            "conservative GM arguments may alias");
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
          %src: !pto.partition_tensor_view<16x16xf32>, %limit: index) {
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
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("reuse"));
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
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> a5Problem =
      buildCanonicalSyncSingletonProblem(*a5Program, options);
  const bool a5HasLoopBoundaryPrefix =
      succeeded(a5Problem) && llvm::any_of((*a5Problem)->getMechanisms(),
                                           isDistanceTwoLoopBoundaryPrefix);
  if (!check(succeeded(a5Problem),
             "build the distance-two recurrence catalog for A5") ||
      !check(!a5HasLoopBoundaryPrefix,
             "do not infer the targeted-V loop-boundary protocol on A5")) {
    return false;
  }
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
  const auto isDistanceTwoExactDrain = [](const auto &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier &&
           llvm::any_of(
               mechanism.descriptor.supplies,
               [](const CanonicalSyncSupplyBinding &binding) {
                 return binding.edge.distance == 2 &&
                        (binding.proof == CanonicalSyncSupplyProof::
                                              TargetLocalPipeDrainAction ||
                         binding.proof == CanonicalSyncSupplyProof::
                                              SourceLocalPipeDrainAction ||
                         binding.proof ==
                             CanonicalSyncSupplyProof::LoopCarryPipeDrain);
               });
  };
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
  if (!check(llvm::any_of((*problem)->getMechanisms(), isDistanceTwoExactDrain),
             "generate an exact distance-two target drain") ||
      !check(llvm::any_of((*problem)->getMechanisms(),
                          isDistanceTwoCrossResourceLoopCarry),
             "generate an exact cross-resource distance-two loop-carry "
             "drain")) {
    return false;
  }
  const auto crossResourceLoopCarry = llvm::find_if(
      (*problem)->getMechanisms(), isDistanceTwoCrossResourceLoopCarry);
  std::vector<SyncCoverDemandId> loopCarryDemands;
  for (const CanonicalSyncSupplyBinding &binding :
       crossResourceLoopCarry->descriptor.supplies) {
    if (binding.attestedDemand) {
      loopCarryDemands.push_back(*binding.attestedDemand);
    }
  }
  llvm::sort(loopCarryDemands);
  loopCarryDemands.erase(
      std::unique(loopCarryDemands.begin(), loopCarryDemands.end()),
      loopCarryDemands.end());
  CanonicalSyncPatternProblem isolatedLoopCarry(graph, loopCarryDemands);
  const bool isolatedLoopCarryBuilt =
      !loopCarryDemands.empty() &&
      isolatedLoopCarry.internMechanism(crossResourceLoopCarry->descriptor) &&
      isolatedLoopCarry.freeze();
  CanonicalSyncSelection loopCarrySelection;
  CanonicalSyncVerifiedPlan verifiedLoopCarry;
  if (isolatedLoopCarryBuilt) {
    loopCarrySelection = selectCanonicalSyncPatterns(isolatedLoopCarry);
    verifiedLoopCarry =
        verifyCanonicalSyncSelection(isolatedLoopCarry, loopCarrySelection);
  }
  if (!check(isolatedLoopCarryBuilt && loopCarrySelection &&
                 loopCarrySelection.mechanisms ==
                     std::vector<CanonicalSyncMechanismId>{0} &&
                 verifiedLoopCarry,
             "select and freshly verify only the cross-resource "
             "distance-two loop-carry drain")) {
    return false;
  }
  const auto rejectsLoopCarry = [&](CanonicalSyncMechanismDescriptor value) {
    CanonicalSyncPatternProblem tampered(graph, loopCarryDemands);
    return !tampered.internMechanism(std::move(value));
  };
  CanonicalSyncMechanismDescriptor wrongLoopCarryPipe =
      crossResourceLoopCarry->descriptor;
  wrongLoopCarryPipe.actions.front().resource =
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1);
  wrongLoopCarryPipe.actions.front().drainedResources = {
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1)};
  CanonicalSyncMechanismDescriptor wrongLoopCarryScope =
      crossResourceLoopCarry->descriptor;
  wrongLoopCarryScope.actions.front().anchor.scope = 0;
  wrongLoopCarryScope.actions.front().guardScope = 0;
  CanonicalSyncMechanismDescriptor wrongLoopCarryGuard =
      crossResourceLoopCarry->descriptor;
  wrongLoopCarryGuard.actions.front().guard =
      CanonicalSyncActionGuardKind::None;
  wrongLoopCarryGuard.actions.front().guardScope.reset();
  CanonicalSyncMechanismDescriptor unverifiedLoopCarry =
      crossResourceLoopCarry->descriptor;
  unverifiedLoopCarry.supplies.front().attestedDemand.reset();
  if (!check(rejectsLoopCarry(std::move(wrongLoopCarryPipe)),
             "reject a loop-carry drain for the wrong source pipe") ||
      !check(rejectsLoopCarry(std::move(wrongLoopCarryScope)),
             "reject a loop-carry drain at the wrong loop scope") ||
      !check(rejectsLoopCarry(std::move(wrongLoopCarryGuard)),
             "reject an unguarded loop-carry drain") ||
      !check(rejectsLoopCarry(std::move(unverifiedLoopCarry)),
             "reject a loop-carry drain without exact attestation")) {
    return false;
  }
  if (!check(succeeded(materializeCanonicalSyncPlan(*program, isolatedLoopCarry,
                                                    verifiedLoopCarry)),
             "materialize only the cross-resource distance-two loop-carry "
             "drain")) {
    return false;
  }
  std::size_t loopCarryBarriers = 0;
  std::size_t loopCarryEvents = 0;
  bool exactNotFirstIterationGuard = false;
  module->walk([&](Operation *operation) {
    const bool generated = operation->hasAttr("pto.canonical_sync");
    loopCarryEvents +=
        generated &&
        isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp>(operation);
    auto barrier = dyn_cast<BarrierOp>(operation);
    if (!generated || !barrier ||
        barrier->hasAttr("pto.auto_sync_tail_barrier")) {
      return;
    }
    ++loopCarryBarriers;
    auto guard = barrier->getParentOfType<scf::IfOp>();
    auto loop = barrier->getParentOfType<scf::ForOp>();
    auto comparison =
        guard ? guard.getCondition().getDefiningOp<arith::CmpIOp>() : nullptr;
    exactNotFirstIterationGuard =
        exactNotFirstIterationGuard ||
        (loop && comparison &&
         comparison.getPredicate() == arith::CmpIPredicate::ne &&
         comparison.getLhs() == loop.getInductionVar() &&
         comparison.getRhs() == loop.getLowerBound());
  });
  if (!check(loopCarryBarriers == 1 && loopCarryEvents == 0 &&
                 exactNotFirstIterationGuard,
             "emit one targeted barrier that is inactive for zero/one-trip "
             "carry and active on every later iteration")) {
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
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(isolatedProtocolProblem, selection);
  const bool selectedLoopBoundaryProtocol =
      selection &&
      selection.mechanisms == std::vector<CanonicalSyncMechanismId>{0};
  const bool recurrenceSelected =
      check(selectedLoopBoundaryProtocol && verified,
            "select and freshly verify only the distance-two loop-boundary "
            "recurrence protocol");
  const bool recurrenceMaterialized =
      check(verified && succeeded(materializeCanonicalSyncPlan(
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

bool testA5MatrixLoopBoundaryProtocol() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a5"} {
      func.func @matrix_reuse(
          %lhs: !pto.tile_buf<mat, 64x32xf16,
                              blayout=col_major, slayout=row_major>,
          %rhs: !pto.tile_buf<mat, 32x64xf16,
                              blayout=col_major, slayout=row_major>,
          %limit: index) {
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
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  if (!check(succeeded(problem),
             "build A5 matrix loop-boundary protocol catalog")) {
    return false;
  }
  const std::uint32_t matrix = static_cast<std::uint32_t>(PipelineType::PIPE_M);
  const std::uint32_t mte1 =
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1);
  const auto isMatrixProtocol = [&](const CanonicalSyncMechanism &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
           mechanism.descriptor.eventUses.size() == 1 &&
           mechanism.descriptor.supplies.size() >= 2 &&
           llvm::all_of(mechanism.descriptor.supplies,
                        [&](const CanonicalSyncSupplyBinding &binding) {
                          return binding.edge.distance == 1 &&
                                 binding.proof ==
                                     CanonicalSyncSupplyProof::
                                         LoopBoundarySourcePrefixProtocol &&
                                 program->getGraph()
                                         .getNodes()[binding.edge.source]
                                         .resource == matrix &&
                                 program->getGraph()
                                         .getNodes()[binding.edge.target]
                                         .resource == mte1;
                        }) &&
           llvm::any_of(
               mechanism.descriptor.actions,
               [&](const CanonicalSyncAction &action) {
                 return action.kind == CanonicalSyncActionKind::Barrier &&
                        action.resource == matrix &&
                        action.anchor.kind == SyncCoverAnchorKind::LoopBodyExit;
               });
  };
  const auto protocol =
      llvm::find_if((*problem)->getMechanisms(), isMatrixProtocol);
  if (!check(protocol != (*problem)->getMechanisms().end(),
             "generate a supported non-V A5 loop-boundary protocol") ||
      !verifyExactAndOneLessProtocolWork(
          **problem, protocol->id,
          "verify a multi-supply loop-boundary protocol at its exact bound",
          "reject a multi-supply loop-boundary protocol at its one-less "
          "bound")) {
    return false;
  }
  std::vector<SyncCoverDemandId> demands;
  for (const CanonicalSyncSupplyBinding &binding :
       protocol->descriptor.supplies) {
    demands.push_back(*binding.attestedDemand);
  }
  llvm::sort(demands);
  demands.erase(std::unique(demands.begin(), demands.end()), demands.end());
  CanonicalSyncPatternProblem isolated(program->getGraph(), demands);
  for (const CanonicalSyncEventDomain &domain : (*problem)->getDomains()) {
    if (!isolated.addEventDomain(domain)) {
      return check(false, "copy A5 protocol event domains");
    }
  }
  const CanonicalSyncProblemResult added = isolated.internVerifiedProtocol(
      protocol->descriptor,
      testProtocolVerifier([&](const CanonicalSyncMechanismDescriptor &value) {
        CanonicalSyncMechanism mechanism;
        mechanism.descriptor = value;
        return isMatrixProtocol(mechanism);
      }));
  if (!check(added && isolated.freeze(),
             "freeze the isolated A5 matrix loop-boundary cover")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(isolated);
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(isolated, selection);
  if (!check(selection &&
                 selection.mechanisms ==
                     std::vector<CanonicalSyncMechanismId>{0} &&
                 verified,
             "select and freshly verify only the A5 matrix loop-boundary "
             "protocol") ||
      !check(
          succeeded(materializeCanonicalSyncPlan(*program, isolated, verified)),
          "materialize only the A5 matrix loop-boundary protocol")) {
    return false;
  }
  std::size_t sets = 0;
  std::size_t waits = 0;
  std::size_t matrixBarriers = 0;
  std::size_t generatedPipeAll = 0;
  function.walk([&](Operation *operation) {
    const bool generated = operation->hasAttr("pto.canonical_sync");
    sets += generated && isa<SetFlagOp>(operation);
    waits += generated && isa<WaitFlagOp>(operation);
    if (auto barrier = dyn_cast<BarrierOp>(operation);
        barrier && generated &&
        !barrier->hasAttr("pto.auto_sync_tail_barrier")) {
      matrixBarriers += barrier.getPipe().getPipe() == PIPE::PIPE_M;
      generatedPipeAll += barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
    }
  });
  const bool materializedShape =
      sets == 2 && waits == 2 && matrixBarriers == 1 && generatedPipeAll == 0;

  (*module)->removeAttr("pto.target_arch");
  FailureOr<CanonicalSyncProgram> unsupported =
      buildCanonicalSyncProgram(function);
  const bool unsupportedRejected =
      succeeded(unsupported) &&
      !unsupported->getGraph().supportsBlockingTargetedBarrier(matrix);
  return check(materializedShape,
               "emit one A5 M barrier with balanced prime/body/drain flags") &&
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
          %dst: !pto.partition_tensor_view<32x32xf32>) {
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
            "certify the exact A3 accumulator-to-FIX boundary");
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
  FailureOr<CanonicalSyncProgram> a5 = buildCanonicalSyncProgram(function);
  const bool a5Rejected =
      check(succeeded(a5), "build the same graph for A5") &&
      check(a5->getGraph().getTargetCompletionCertificates().empty(),
            "do not infer A3 target certificates on A5");
  if (!a3Qualified || !a5Rejected) {
    return false;
  }

  CanonicalSyncBuildOptions a5Options;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> a5Problem =
      buildCanonicalSyncSingletonProblem(*a5, a5Options);
  const std::uint32_t matrixResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_M);
  const std::uint32_t fixResource =
      static_cast<std::uint32_t>(PipelineType::PIPE_FIX);
  const bool hasBarrierBackedMatrixEvent =
      succeeded(a5Problem) &&
      llvm::any_of((*a5Problem)->getMechanisms(), [&](const auto &mechanism) {
        const auto &descriptor = mechanism.descriptor;
        const bool hasMatrixToFixSupply = llvm::any_of(
            descriptor.supplies, [&](const CanonicalSyncSupplyBinding &supply) {
              return supply.proof == CanonicalSyncSupplyProof::
                                         SourceLocalCompletionAction &&
                     (*a5Problem)
                             ->getGraph()
                             .getNodes()[supply.edge.source]
                             .resource == matrixResource &&
                     (*a5Problem)
                             ->getGraph()
                             .getNodes()[supply.edge.target]
                             .resource == fixResource;
            });
        return hasMatrixToFixSupply && descriptor.actions.size() == 3 &&
               descriptor.actions[0].kind == CanonicalSyncActionKind::Barrier &&
               descriptor.actions[0].resource == matrixResource &&
               descriptor.actions[1].kind ==
                   CanonicalSyncActionKind::EventSet &&
               descriptor.actions[2].kind == CanonicalSyncActionKind::EventWait;
      });
  if (!check(hasBarrierBackedMatrixEvent,
             "build an A5 barrier-backed M-to-FIX event, not a drain-only "
             "normal candidate") ||
      !check(succeeded(runCanonicalSync(function, a5Options)),
             "materialize the A5 non-direct source event")) {
    return false;
  }
  std::size_t a5MatrixBarriers = 0;
  std::size_t a5MatrixToFixSets = 0;
  std::size_t a5MatrixToFixWaits = 0;
  std::size_t a5BodyPipeAll = 0;
  Operation *a5MatrixBarrierOp = nullptr;
  Operation *a5MatrixToFixSetOp = nullptr;
  Operation *a5MatrixToFixWaitOp = nullptr;
  function.walk([&](Operation *operation) {
    const bool generated = operation->hasAttr("pto.canonical_sync");
    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      const bool body =
          generated && !barrier->hasAttr("pto.auto_sync_tail_barrier");
      a5MatrixBarriers += body && barrier.getPipe().getPipe() == PIPE::PIPE_M;
      if (body && barrier.getPipe().getPipe() == PIPE::PIPE_M) {
        a5MatrixBarrierOp = operation;
      }
      a5BodyPipeAll += body && barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
    }
    const auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
    const auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
    const bool matrixToFix = generated && source && target &&
                             source.getPipe() == PIPE::PIPE_M &&
                             target.getPipe() == PIPE::PIPE_FIX;
    a5MatrixToFixSets += matrixToFix && isa<SetFlagOp>(operation);
    a5MatrixToFixWaits += matrixToFix && isa<WaitFlagOp>(operation);
    if (matrixToFix && isa<SetFlagOp>(operation)) {
      a5MatrixToFixSetOp = operation;
    }
    if (matrixToFix && isa<WaitFlagOp>(operation)) {
      a5MatrixToFixWaitOp = operation;
    }
  });
  const bool orderedA5Recipe =
      a5MatrixBarrierOp && a5MatrixToFixSetOp && a5MatrixToFixWaitOp &&
      a5MatrixBarrierOp->getBlock() == a5MatrixToFixSetOp->getBlock() &&
      a5MatrixToFixSetOp->getBlock() == a5MatrixToFixWaitOp->getBlock() &&
      a5MatrixBarrierOp->isBeforeInBlock(a5MatrixToFixSetOp) &&
      a5MatrixToFixSetOp->isBeforeInBlock(a5MatrixToFixWaitOp);
  if (!check(a5MatrixBarriers == 1 && a5MatrixToFixSets == 1 &&
                 a5MatrixToFixWaits == 1 && a5BodyPipeAll == 0,
             "emit one balanced A5 barrier-backed M-to-FIX event without "
             "PIPE_ALL") ||
      !check(orderedA5Recipe,
             "emit the A5 source barrier, set, and wait in order")) {
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
    module attributes {pto.target_arch = "a5"} {
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
          %output: !pto.partition_tensor_view<16x16xf32>) {
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
    module attributes {pto.target_arch = "a5"} {
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
                                 bool expectMultipleResources = false) -> bool {
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
    const bool credited =
        check(!graph.getDemands().empty() && hasFixedSupply &&
                  exactTargetedEndpoint && expectedResources,
              "credit an unowned barrier as fixed completion supply");
    const bool covered =
        check(baselineComplete,
              "cover fixed-barrier hazards without a candidate mechanism: " +
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
           check(generatedNonTailSync == 0,
                 "avoid redundant synchronization around fixed supply");
  };

  const bool fixedCasesCovered = checkFunction("fixed", false) &&
                                 checkFunction("fixed_cross", false) &&
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
  CanonicalSyncBuildOptions exactOptions;
  exactOptions.analysis.maximumPairInspections = 27;
  if (!check(succeeded(runCanonicalSync(fixed, exactOptions)),
             "complete fixed-barrier analysis at its exact work bound")) {
    return false;
  }
  const std::string materialized = printOperation(fixed);
  CanonicalSyncBuildOptions belowOptions = exactOptions;
  belowOptions.analysis.maximumPairInspections = 26;
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
  CanonicalSyncAnalysisOptions deepExact;
  deepExact.maximumPairInspections = 7 * prefixNodes - 4;
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
             "bound deep no-barrier state independently of control depth") ||
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
  FailureOr<CanonicalSyncProgram> negative =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("negative"));
  FailureOr<CanonicalSyncProgram> negativeUnsigned = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("negative_unsigned"));
  return check(succeeded(negative), "build negative-lower graph") &&
         check(negative->getGraph().getControls().size() == 1 &&
                   !negative->getGraph().getControls()[0].phaseRelation,
               "do not mis-model signed remainder with negative lower bound") &&
         check(succeeded(negativeUnsigned),
               "build unsigned negative-lower graph") &&
         check(negativeUnsigned->getGraph().getControls().size() == 1 &&
                   !negativeUnsigned->getGraph().getControls()[0].phaseRelation,
               "do not mis-model unsigned remainder with negative lower");
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
  FailureOr<CanonicalSyncProgram> nearMiss = build("near_miss");
  FailureOr<CanonicalSyncProgram> nested = build("nested");
  const bool builtPrograms =
      check(succeeded(first) && succeeded(nearMiss) && succeeded(nested),
            "build first-iteration recurrence graphs");
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
  const std::optional<SyncCoverNodeId> nearMissTarget =
      guardedTarget(*nearMiss);
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
         check(nearMissTarget &&
                   recurrenceTo(*nearMiss, *nearMissTarget, std::nullopt),
               "retain recurrence for an unrecognized loop condition") &&
         check(nestedTarget && outerScope &&
                   recurrenceTo(*nested, *nestedTarget, outerScope),
               "retain enclosing-loop recurrence for an inner first iteration");
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

  CanonicalSyncProgram program(
      module->lookupSymbol<func::FuncOp>("guarded_ownership_host"),
      std::move(graph), {}, {}, {}, {}, {}, {}, {});
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
  const CanonicalSyncMechanismOriginMask origin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::BasicOwnershipStableL1Protocol);
  const auto mechanism = llvm::find_if(
      precise.problem->getMechanisms(), [&](const auto &candidate) {
        return (candidate.originMask & origin) != 0;
      });
  return check(mechanism != precise.problem->getMechanisms().end(),
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
    module attributes {pto.target_arch = "a3"} {
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
  directFallbackOptions.patterns.enabledMechanismFamilies = 0;
  directFallbackOptions.patterns.enableDirectPairs = false;
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
          CanonicalSyncMechanismOrigin::DirectReleaseRecurrenceProtocol);
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
    module attributes {pto.target_arch = "a3"} {
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
          %input: !pto.partition_tensor_view<16x16xf32>, %limit: index) {
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
  const auto isExactDrain = [](const CanonicalSyncMechanism &mechanism) {
    return mechanism.descriptor.kind == CanonicalSyncMechanismKind::Barrier &&
           llvm::any_of(
               mechanism.descriptor.supplies,
               [](const CanonicalSyncSupplyBinding &supply) {
                 return supply.edge.distance == 1 &&
                        (supply.proof == CanonicalSyncSupplyProof::
                                             TargetLocalPipeDrainAction ||
                         supply.proof == CanonicalSyncSupplyProof::
                                             SourceLocalPipeDrainAction ||
                         supply.proof ==
                             CanonicalSyncSupplyProof::LoopCarryPipeDrain);
               });
  };
  if (!check(llvm::any_of((*problem)->getMechanisms(), isExactDrain),
             "generate an exact distance-one target drain")) {
    return false;
  }
  const SyncCoverGraph &graph = program->getGraph();
  const auto isLoopCarryDrain = [&](const CanonicalSyncMechanism &mechanism) {
    if (mechanism.descriptor.kind != CanonicalSyncMechanismKind::Barrier ||
        mechanism.descriptor.actions.size() != 1 ||
        mechanism.descriptor.supplies.empty()) {
      return false;
    }
    const CanonicalSyncAction &action = mechanism.descriptor.actions.front();
    std::set<std::uint32_t> targetResources;
    bool crossesResources = false;
    const bool exactBindings = llvm::all_of(
        mechanism.descriptor.supplies,
        [&](const CanonicalSyncSupplyBinding &supply) {
          if (supply.edge.source >= graph.getNodes().size() ||
              supply.edge.target >= graph.getNodes().size()) {
            return false;
          }
          const std::uint32_t sourceResource =
              graph.getNodes()[supply.edge.source].resource;
          const std::uint32_t targetResource =
              graph.getNodes()[supply.edge.target].resource;
          targetResources.insert(targetResource);
          crossesResources |= sourceResource != targetResource;
          return sourceResource == action.resource &&
                 supply.edge.distance != 0 &&
                 supply.proof == CanonicalSyncSupplyProof::LoopCarryPipeDrain &&
                 supply.attestedDemand &&
                 supply.allowedDemands ==
                     std::vector<SyncCoverDemandId>{*supply.attestedDemand};
        });
    return action.anchor.kind == SyncCoverAnchorKind::LoopBodyEntry &&
           action.guard == CanonicalSyncActionGuardKind::NotFirstIteration &&
           action.guardScope ==
               std::optional<SyncCoverScopeId>(action.anchor.scope) &&
           exactBindings && crossesResources && targetResources.size() >= 2;
  };
  if (!check(llvm::any_of((*problem)->getMechanisms(), isLoopCarryDrain),
             "share one exact loop-entry carry drain across same- and "
             "cross-pipe targets")) {
    return false;
  }
  const auto loopCarry =
      llvm::find_if((*problem)->getMechanisms(), isLoopCarryDrain);
  std::vector<SyncCoverDemandId> loopCarryDemands;
  for (const CanonicalSyncSupplyBinding &binding :
       loopCarry->descriptor.supplies) {
    loopCarryDemands.push_back(*binding.attestedDemand);
  }
  llvm::sort(loopCarryDemands);
  loopCarryDemands.erase(
      std::unique(loopCarryDemands.begin(), loopCarryDemands.end()),
      loopCarryDemands.end());
  CanonicalSyncPatternProblem isolatedLoopCarry(graph, loopCarryDemands);
  const bool isolatedLoopCarryBuilt =
      isolatedLoopCarry.internMechanism(loopCarry->descriptor) &&
      isolatedLoopCarry.freeze();
  CanonicalSyncSelection isolatedLoopCarrySelection;
  CanonicalSyncVerifiedPlan isolatedLoopCarryPlan;
  if (isolatedLoopCarryBuilt) {
    isolatedLoopCarrySelection = selectCanonicalSyncPatterns(isolatedLoopCarry);
    isolatedLoopCarryPlan = verifyCanonicalSyncSelection(
        isolatedLoopCarry, isolatedLoopCarrySelection);
  }
  if (!check(isolatedLoopCarryBuilt && isolatedLoopCarrySelection &&
                 isolatedLoopCarrySelection.mechanisms ==
                     std::vector<CanonicalSyncMechanismId>{0} &&
                 isolatedLoopCarryPlan,
             "select and freshly verify only the mixed-target "
             "distance-one loop-carry drain")) {
    return false;
  }
  using LoopCarryGroup = std::pair<SyncCoverScopeId, std::uint32_t>;
  const auto collectLoopCarryGroups =
      [](const CanonicalSyncPatternProblem &candidateProblem) {
        std::map<LoopCarryGroup, std::size_t> groups;
        for (const CanonicalSyncMechanism &mechanism :
             candidateProblem.getMechanisms()) {
          if (mechanism.descriptor.supplies.empty() ||
              !llvm::all_of(
                  mechanism.descriptor.supplies,
                  [](const CanonicalSyncSupplyBinding &binding) {
                    return binding.proof ==
                           CanonicalSyncSupplyProof::LoopCarryPipeDrain;
                  })) {
            continue;
          }
          const CanonicalSyncAction &action =
              mechanism.descriptor.actions.front();
          groups[{action.anchor.scope, action.resource}] =
              mechanism.descriptor.supplies.size();
        }
        return groups;
      };
  const CanonicalSyncPatternStatistics &carryStatistics =
      (*problem)->getPatternStatistics();
  const std::map<LoopCarryGroup, std::size_t> fullCarryGroups =
      collectLoopCarryGroups(**problem);
  std::size_t fullCarryIncidences = 0;
  for (const auto &[group, incidences] : fullCarryGroups) {
    (void)group;
    fullCarryIncidences += incidences;
  }
  if (!check(!fullCarryGroups.empty() &&
                 carryStatistics.loopCarryInspections ==
                     (*problem)->getDemands().size() &&
                 carryStatistics.loopCarryCandidates ==
                     fullCarryGroups.size() &&
                 carryStatistics.loopCarryIncidences == fullCarryIncidences &&
                 carryStatistics.loopCarryIncidences > 1 &&
                 !carryStatistics.loopCarryGenerationTruncated,
             "record every complete loop-carry group")) {
    return false;
  }
  const CanonicalSyncAction &mixedTargetCarryAction =
      loopCarry->descriptor.actions.front();
  const LoopCarryGroup mixedTargetCarryKey{mixedTargetCarryAction.anchor.scope,
                                           mixedTargetCarryAction.resource};
  const std::size_t carryGroupSize = fullCarryGroups.at(mixedTargetCarryKey);
  const auto firstLoopCarry = llvm::find_if(
      (*problem)->getMechanisms(), [](const CanonicalSyncMechanism &mechanism) {
        return !mechanism.descriptor.supplies.empty() &&
               llvm::all_of(
                   mechanism.descriptor.supplies,
                   [](const CanonicalSyncSupplyBinding &binding) {
                     return binding.proof ==
                            CanonicalSyncSupplyProof::LoopCarryPipeDrain;
                   });
      });
  std::size_t requiredPrefixSupplies = 0;
  for (const CanonicalSyncMechanism &mechanism : (*problem)->getMechanisms()) {
    if (mechanism.id >= firstLoopCarry->id) {
      break;
    }
    requiredPrefixSupplies += mechanism.descriptor.supplies.size();
  }
  const std::size_t firstCarryGroupSize =
      firstLoopCarry->descriptor.supplies.size();
  const auto buildBoundedCarry = [&](std::size_t inspections,
                                     std::size_t candidates,
                                     std::size_t incidences,
                                     std::size_t suppliesPerMechanism,
                                     std::size_t totalSupplies) {
    CanonicalSyncBuildOptions bounded = options;
    bounded.patterns.maximumLoopCarryInspections = inspections;
    bounded.patterns.maximumLoopCarryCandidates = candidates;
    bounded.patterns.maximumLoopCarryIncidences = incidences;
    bounded.patterns.maximumLoopBoundaryProtocolInspections = 0;
    bounded.problemLimits.maximumSuppliesPerMechanism = suppliesPerMechanism;
    bounded.problemLimits.maximumTotalSupplies = totalSupplies;
    return buildCanonicalSyncSingletonProblem(*program, bounded);
  };
  const std::size_t defaultPerMechanism =
      options.problemLimits.maximumSuppliesPerMechanism;
  const std::size_t defaultTotal = options.problemLimits.maximumTotalSupplies;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> exactCarry =
      buildBoundedCarry(carryStatistics.loopCarryInspections,
                        carryStatistics.loopCarryCandidates,
                        carryStatistics.loopCarryIncidences,
                        defaultPerMechanism, defaultTotal);
  const auto truncatedAndSingletonVerified =
      [&](const FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>>
              &built) {
        if (failed(built) ||
            !(*built)->getPatternStatistics().loopCarryGenerationTruncated) {
          return false;
        }
        const std::map<LoopCarryGroup, std::size_t> retainedGroups =
            collectLoopCarryGroups(**built);
        if (retainedGroups == fullCarryGroups ||
            llvm::any_of(retainedGroups, [&](const auto &retained) {
              const auto full = fullCarryGroups.find(retained.first);
              return full == fullCarryGroups.end() ||
                     full->second != retained.second;
            })) {
          return false;
        }
        const CanonicalSyncSelection boundedSelection =
            selectCanonicalSyncPatterns(**built);
        const CanonicalSyncVerifiedPlan boundedPlan =
            verifyCanonicalSyncSelection(**built, boundedSelection);
        return boundedSelection && boundedPlan;
      };
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> belowInspection =
      buildBoundedCarry(carryStatistics.loopCarryInspections - 1,
                        carryStatistics.loopCarryCandidates,
                        carryStatistics.loopCarryIncidences,
                        defaultPerMechanism, defaultTotal);
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> belowCandidate =
      buildBoundedCarry(carryStatistics.loopCarryInspections,
                        carryStatistics.loopCarryCandidates - 1,
                        carryStatistics.loopCarryIncidences,
                        defaultPerMechanism, defaultTotal);
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> belowIncidence =
      buildBoundedCarry(carryStatistics.loopCarryInspections,
                        carryStatistics.loopCarryCandidates,
                        carryStatistics.loopCarryIncidences - 1,
                        defaultPerMechanism, defaultTotal);
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> oversizedGroup =
      buildBoundedCarry(carryStatistics.loopCarryInspections,
                        carryStatistics.loopCarryCandidates,
                        carryStatistics.loopCarryIncidences, carryGroupSize - 1,
                        defaultTotal);
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> aggregateLimited =
      buildBoundedCarry(carryStatistics.loopCarryInspections,
                        carryStatistics.loopCarryCandidates,
                        carryStatistics.loopCarryIncidences,
                        defaultPerMechanism,
                        requiredPrefixSupplies + firstCarryGroupSize - 1);
  const std::map<LoopCarryGroup, std::size_t> oversizedRetainedGroups =
      succeeded(oversizedGroup) ? collectLoopCarryGroups(**oversizedGroup)
                                : std::map<LoopCarryGroup, std::size_t>{};
  if (!check(succeeded(exactCarry) &&
                 !(*exactCarry)
                      ->getPatternStatistics()
                      .loopCarryGenerationTruncated &&
                 collectLoopCarryGroups(**exactCarry) == fullCarryGroups,
             "retain every loop-carry group at the exact bounds") ||
      !check(truncatedAndSingletonVerified(belowInspection),
             "truncate loop-carry generation one inspection below") ||
      !check(truncatedAndSingletonVerified(belowCandidate),
             "truncate loop-carry generation one candidate below") ||
      !check(truncatedAndSingletonVerified(belowIncidence),
             "truncate loop-carry generation one incidence below") ||
      !check(truncatedAndSingletonVerified(oversizedGroup),
             "skip an oversized cross-target loop-carry group all-or-none") ||
      !check(succeeded(oversizedGroup) &&
                 oversizedRetainedGroups.find(mixedTargetCarryKey) ==
                     oversizedRetainedGroups.end(),
             "omit the complete oversized mixed-target carry group") ||
      !check(truncatedAndSingletonVerified(aggregateLimited),
             "retain singleton correctness when the aggregate supply cap "
             "rejects loop-carry consolidation")) {
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
          %input: !pto.partition_tensor_view<16x16xf32>, %condition: i1) {
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
  const CanonicalSyncMechanismOriginMask sourceLocalOrigin =
      canonicalSyncMechanismOriginBit(
          CanonicalSyncMechanismOrigin::SourceLocalCompletionEvent);
  const bool hasSourceLocalOrigin =
      llvm::any_of(precise.problem->getMechanisms(),
                   [&](const CanonicalSyncMechanism &mechanism) {
                     return (mechanism.originMask & sourceLocalOrigin) != 0;
                   });
  CanonicalSyncBuildOptions explicitAllOptions = options;
  explicitAllOptions.patterns.enabledMechanismFamilies =
      kAllCanonicalSyncMechanismFamilies;
  CanonicalSyncProblemBuildResult explicitAll =
      buildCanonicalSyncPreciseProblem(*program, explicitAllOptions);
  if (!check(hasSourceLocalOrigin,
             "classify guarded completeness as a source-local event") ||
      !check(explicitAll && explicitAll.problem &&
                 precise.problem->hasSameCandidatePrefix(*explicitAll.problem),
             "make the default catalog identical to an explicit ALL mask")) {
    return false;
  }

  CanonicalSyncBuildOptions withoutSourceLocalOptions = options;
  withoutSourceLocalOptions.patterns.enabledMechanismFamilies &=
      ~canonicalSyncMechanismFamilyBit(
          CanonicalSyncMechanismFamily::SourceLocalCompletion);
  CanonicalSyncBuildOptions coreOptions = options;
  coreOptions.patterns.enabledMechanismFamilies = 0;
  bool withoutSourceLocalRejected = false;
  bool coreRejected = false;
  {
    ScopedDiagnosticHandler handler(&context,
                                    [](Diagnostic &) { return success(); });
    withoutSourceLocalRejected =
        !buildCanonicalSyncPreciseProblem(*program, withoutSourceLocalOptions);
    coreRejected = !buildCanonicalSyncPreciseProblem(*program, coreOptions);
  }
  if (!check(withoutSourceLocalRejected,
             "fail closed when the required source-local family is disabled") ||
      !check(coreRejected,
             "fail closed for a core-only catalog without a balanced recipe")) {
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
  if (!check(generatedPipeAllBackstops == 0 && generatedTargetedDrains == 0 &&
                 generatedSets == 1 && generatedWaits == 1,
             "use one balanced source-local event for the guarded endpoint") ||
      !check(report.function == "uncoverable_guarded_endpoint" &&
                 report.graphNodes != 0 && report.uniqueDemandRows != 0 &&
                 report.strategies.size() == 1 &&
                 report.strategies.front().verified &&
                 !report.strategies.front().usedLocalizedPipeAll &&
                 report.strategies.front().emittedEventSets == 1 &&
                 report.strategies.front().emittedEventWaits == 1 &&
                 report.strategies.front().emittedTargetedBarriers == 0 &&
                 report.strategies.front().emittedPipeAllBarriers == 0 &&
                 report.strategies.front().verificationWorkUnits != 0 &&
                 report.strategies.front().predictedSyncInstructions != 0 &&
                 report.strategies.front().planSignature != 0,
             "report the freshly verified source-local event plan")) {
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
    module attributes {pto.target_arch = "a5"} {
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
    module {
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
  CanonicalSyncBuildOptions staleOptions = options;
  staleOptions.eventIdBudget = 2;
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
  if (!check(
          !staleRepair &&
              staleRepair.status.error ==
                  CanonicalSyncProblemError::InvalidPattern &&
              sawStaleCoreDiagnostic,
          "reject a conflict core against a differently configured prefix")) {
    return false;
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
      testMaterializationRejectsTamperedEventAllocations() &&
      testMacroBindingsAndHiddenReservations() &&
      testEmptyNestedLoopTimelineIsClamped() && testGmAliasPolicies() &&
      testGmAliasContracts() && testStructuredIssueFrontier() &&
      testDistanceTwoPhysicalSlotRecurrence() &&
      testA5MatrixLoopBoundaryProtocol() &&
      testDistanceTwoCrossRootSlotRecurrence() &&
      testMmadIntrinsicRequiresExactAccumulator() &&
      testA3TargetCompletionCertificatesAreArchitectureQualified() &&
      testAnalysisLimitFailsClosed() && testFailClosedInputs() &&
      testAcceptsDeclaredStorageProvenanceRoots() &&
      testRejectsOwnedSyncAndAcceptsFixedFence() &&
      testRejectsMalformedOwnedSynchronization() &&
      testFixedBarriersSupplyCompletionAndRemainUnowned() &&
      testRejectsFixedBarrierInsideLoop() &&
      testFixedBarrierInspectionBoundsAndPersistentControlState() &&
      testStructuralLimitsFailClosed() && testPeriodicBranchEvidence() &&
      testFirstIterationRecurrenceSuppression() &&
      testGuardedOwnershipVerificationWorkIsBounded() &&
      testBasicL0OwnershipSharesExhaustiveBranchBoundaries() &&
      testOwnershipDoesNotHideProducerOverwrite() &&
      testGenericRecurrenceWithoutOwnershipDiscovery() &&
      testGuardedEndpointUsesSourceLocalCompletionEvent() &&
      testDemandBasisReductionIsBoundedAndTruncating() &&
      testSourcePrefixGenerationIsBoundedAndTruncating() &&
      testConflictCoreRepairAvoidsPipeAll();
  return passed ? 0 : 1;
}
