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
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

namespace {

using namespace mlir;
using namespace mlir::pto;

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "CanonicalSyncAnalysisTest failure: " << message << '\n';
  }
  return condition;
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
  if (!check(llvm::any_of((*problem)->getMechanisms(), isDistanceTwoRing),
             "generate a two-lane generic recurrence ring")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem);
  const bool selectedRing =
      selection && llvm::any_of(selection.mechanisms, [&](auto mechanism) {
        return isDistanceTwoRing((*problem)->getMechanisms()[mechanism]);
      });
  const bool ringSelected =
      check(selectedRing, "select the distance-two recurrence ring");
  const bool ringMaterialized =
      check(succeeded(runCanonicalSync(
                module->lookupSymbol<func::FuncOp>("reuse"), options)),
            "materialize the distance-two recurrence ring");
  if (!ringSelected || !ringMaterialized) {
    return false;
  }
  std::size_t dynamicSets = 0;
  std::size_t dynamicWaits = 0;
  std::size_t generatedHelpers = 0;
  bool unownedDynamicSync = false;
  module->walk([&](Operation *operation) {
    dynamicSets += operation->getName().getStringRef() == "pto.set_flag_dyn";
    dynamicWaits += operation->getName().getStringRef() == "pto.wait_flag_dyn";
    const bool generated = operation->hasAttr("pto.canonical_sync");
    const bool dynamic = isa<SetFlagDynOp, WaitFlagDynOp>(operation);
    unownedDynamicSync = unownedDynamicSync || (dynamic && !generated);
    generatedHelpers +=
        generated && operation->getDialect() ==
                         context.getLoadedDialect<arith::ArithDialect>();
  });
  if (!check(dynamicSets == 1 && dynamicWaits == 1 && !unownedDynamicSync,
             "emit one owned modulo-selected body set and wait") ||
      !check(generatedHelpers != 0,
             "mark dynamic-lane and guard helpers as pass-owned")) {
    return false;
  }

  func::FuncOp function = module->lookupSymbol<func::FuncOp>("reuse");
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
  const bool replacementIsIdempotent =
      check(printOperation(function) == materialized,
            "make pass-owned replacement idempotent");
  if (!retainedAfterFailure || !replacementSucceeded ||
      !replacementIsIdempotent) {
    return false;
  }
  return true;
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
  exactOptions.analysis.maximumPairInspections = 24;
  if (!check(succeeded(runCanonicalSync(fixed, exactOptions)),
             "complete fixed-barrier analysis at its exact work bound")) {
    return false;
  }
  const std::string materialized = printOperation(fixed);
  CanonicalSyncBuildOptions belowOptions = exactOptions;
  belowOptions.analysis.maximumPairInspections = 23;
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
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem);
  const bool selectedGeneric =
      selection && llvm::any_of(selection.mechanisms, [&](auto mechanism) {
        return isGenericRecurrence((*problem)->getMechanisms()[mechanism]);
      });
  const CanonicalSyncVerifiedPlan verified =
      verifyCanonicalSyncSelection(**problem, selection);
  if (!check(selectedGeneric,
             "select the generic recurrence event without PIPE_ALL") ||
      !check(static_cast<bool>(verified),
             "finalize generic recurrence from certified coverage") ||
      !check(succeeded(runCanonicalSync(function, options)),
             "materialize generic recurrence without ownership")) {
    return false;
  }
  std::size_t setCount = 0;
  std::size_t waitCount = 0;
  function.walk([&](Operation *operation) {
    setCount += operation->getName().getStringRef() == "pto.set_flag";
    waitCount += operation->getName().getStringRef() == "pto.wait_flag";
  });
  return check(setCount == 3 && waitCount == 3,
               "emit direct ready plus prime-body-drain recurrence actions");
}

bool testUncoverablePreciseCatalogUsesBackstop() {
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
  func::FuncOp function =
      module->lookupSymbol<func::FuncOp>("uncoverable_guarded_endpoint");
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(function);
  if (!check(succeeded(program), "build uncoverable guarded program")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  CanonicalSyncProblemBuildResult precise =
      buildCanonicalSyncPreciseProblem(*program, options);
  const bool preciselyUncoverable =
      !precise && precise.problem &&
      precise.status.error == CanonicalSyncProblemError::UncoverableDemand;
  if (!check(preciselyUncoverable,
             "identify the guarded cross-pipe demand as uncoverable")) {
    return false;
  }
  CanonicalSyncComparisonReport report;
  options.reportCallback = [&](const CanonicalSyncComparisonReport &actual) {
    report = actual;
    return success();
  };
  if (!check(succeeded(runCanonicalSync(function, options)),
             "materialize the backstop for an uncoverable precise catalog")) {
    return false;
  }
  std::size_t generatedPipeAllBackstops = 0;
  function.walk([&](BarrierOp barrier) {
    generatedPipeAllBackstops +=
        barrier->hasAttr("pto.canonical_sync") &&
        !barrier->hasAttr("pto.auto_sync_tail_barrier") &&
        barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
  });
  return check(generatedPipeAllBackstops == 1,
               "emit one localized PIPE_ALL for the uncovered demand") &&
         check(report.strategies.size() == 1 &&
                   report.strategies.front().verified &&
                   report.strategies.front().usedLocalizedPipeAll,
               "report the verified uncoverable-demand backstop");
}

bool testConflictCoreRepairAvoidsPipeAll() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a3"} {
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
  const bool partialRepairStayedPrecise =
      partialRepair && partialRepair.problem->getMechanisms().size() ==
                           (*problem)->getMechanisms().size();
  if (!check(partialRepairStayedPrecise,
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
  const bool catalogChecks =
      check(repair.problem->getMechanisms().size() ==
                (*problem)->getMechanisms().size() + 2,
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
      repairReport.strategies.front().repairTrials == 3 &&
      repairReport.strategies.front().repairWorkUnits != 0 &&
      repairReport.strategies.front().selectedEvents == 1 &&
      repairReport.strategies.front().selectedTargetedBarriers == 1;
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
      testBuildsOneFrozenGraph() && testMacroBindingsAndHiddenReservations() &&
      testEmptyNestedLoopTimelineIsClamped() && testGmAliasPolicies() &&
      testGmAliasContracts() && testStructuredIssueFrontier() &&
      testDistanceTwoPhysicalSlotRecurrence() &&
      testDistanceTwoCrossRootSlotRecurrence() &&
      testMmadIntrinsicRequiresExactAccumulator() &&
      testAnalysisLimitFailsClosed() && testFailClosedInputs() &&
      testAcceptsDeclaredStorageProvenanceRoots() &&
      testRejectsOwnedSyncAndAcceptsFixedFence() &&
      testRejectsMalformedOwnedSynchronization() &&
      testFixedBarriersSupplyCompletionAndRemainUnowned() &&
      testRejectsFixedBarrierInsideLoop() &&
      testFixedBarrierInspectionBoundsAndPersistentControlState() &&
      testStructuralLimitsFailClosed() && testPeriodicBranchEvidence() &&
      testFirstIterationRecurrenceSuppression() &&
      testGenericRecurrenceWithoutOwnershipDiscovery() &&
      testUncoverablePreciseCatalogUsesBackstop() &&
      testConflictCoreRepairAvoidsPipeAll();
  return passed ? 0 : 1;
}
