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
#include "PTO/Transforms/CanonicalSync/CanonicalSyncOwnership.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"

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

void loadDialects(MLIRContext &context) {
  context.loadDialect<PTODialect, arith::ArithDialect, func::FuncDialect,
                      scf::SCFDialect>();
}

bool sameEdge(const SyncCoverEdge &left, const SyncCoverEdge &right) {
  return std::tie(left.source, left.target, left.kind, left.scope,
                  left.distance, left.sourceGuard.literals,
                  left.targetGuard.literals) ==
         std::tie(right.source, right.target, right.kind, right.scope,
                  right.distance, right.sourceGuard.literals,
                  right.targetGuard.literals);
}

bool sameDescriptor(const CanonicalSyncMechanismDescriptor &left,
                    const CanonicalSyncMechanismDescriptor &right) {
  if (left.kind != right.kind ||
      left.supplies.size() != right.supplies.size() ||
      left.eventUses.size() != right.eventUses.size() ||
      left.actions.size() != right.actions.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.supplies.size(); ++index) {
    const CanonicalSyncSupplyBinding &first = left.supplies[index];
    const CanonicalSyncSupplyBinding &second = right.supplies[index];
    const bool different =
        !sameEdge(first.edge, second.edge) ||
        std::tie(first.eventUse, first.barrierAction, first.produceAction,
                 first.consumeAction, first.proof, first.allowedDemands) !=
            std::tie(second.eventUse, second.barrierAction,
                     second.produceAction, second.consumeAction, second.proof,
                     second.allowedDemands);
    if (different) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.eventUses.size(); ++index) {
    const CanonicalSyncEventUse &first = left.eventUses[index];
    const CanonicalSyncEventUse &second = right.eventUses[index];
    if (std::tie(first.domain, first.width, first.recurrenceScope,
                 first.lifetimeScope) !=
        std::tie(second.domain, second.width, second.recurrenceScope,
                 second.lifetimeScope)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.actions.size(); ++index) {
    const CanonicalSyncAction &first = left.actions[index];
    const CanonicalSyncAction &second = right.actions[index];
    if (std::tie(first.kind, first.resource, first.anchor.kind,
                 first.anchor.node, first.anchor.scope, first.anchor.position,
                 first.eventUse, first.eventLane, first.drainedResources,
                 first.barrierKind, first.guard, first.guardScope) !=
        std::tie(second.kind, second.resource, second.anchor.kind,
                 second.anchor.node, second.anchor.scope,
                 second.anchor.position, second.eventUse, second.eventLane,
                 second.drainedResources, second.barrierKind, second.guard,
                 second.guardScope)) {
      return false;
    }
  }
  return true;
}

bool sameMechanismCatalog(const CanonicalSyncPatternProblem &left,
                          const CanonicalSyncPatternProblem &right) {
  const bool differentSizes =
      left.getDomains().size() != right.getDomains().size() ||
      left.getMechanisms().size() != right.getMechanisms().size();
  if (differentSizes) {
    return false;
  }
  for (std::size_t index = 0; index < left.getDomains().size(); ++index) {
    const CanonicalSyncEventDomain &first = left.getDomains()[index];
    const CanonicalSyncEventDomain &second = right.getDomains()[index];
    if (std::tie(first.id, first.sourceResource, first.targetResource,
                 first.budget, first.reservedIds) !=
        std::tie(second.id, second.sourceResource, second.targetResource,
                 second.budget, second.reservedIds)) {
      return false;
    }
  }
  for (std::size_t index = 0; index < left.getMechanisms().size(); ++index) {
    const CanonicalSyncMechanism &first = left.getMechanisms()[index];
    const CanonicalSyncMechanism &second = right.getMechanisms()[index];
    if (first.id != second.id ||
        !sameDescriptor(first.descriptor, second.descriptor) ||
        first.conflicts != second.conflicts ||
        first.cost.barrierActions != second.cost.barrierActions ||
        first.cost.eventActions != second.cost.eventActions ||
        first.eventLifetimes.size() != second.eventLifetimes.size()) {
      return false;
    }
    for (std::size_t lifetime = 0;
         lifetime < first.eventLifetimes.size(); ++lifetime) {
      if (std::tie(first.eventLifetimes[lifetime].begin,
                   first.eventLifetimes[lifetime].end) !=
          std::tie(second.eventLifetimes[lifetime].begin,
                   second.eventLifetimes[lifetime].end)) {
        return false;
      }
    }
  }
  return true;
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
  return check(failed(program), "reject unsupported analysis input") &&
         check(sawExpectedDiagnostic, "emit useful rejection diagnostic");
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
  return check(graph.getScopes().size() == 2,
               "construct explicit loop scope") &&
         check(graph.getScopes()[1].isLoop, "mark recurrence scope as loop") &&
         check(recurrence != graph.getDemands().end(),
               "recover distance-two slot reuse") &&
         check(recurrence->scope == 1,
               "attach recurrence to the loop timeline") &&
         check(llvm::any_of(graph.getStorageAccesses(),
                            [](const SyncCoverStorageAccess &access) {
                              return access.exactPhysical &&
                                     access.addressOrdinal.has_value();
                            }),
               "retain exact physical slot ordinals");
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

  return expectAnalysisFailure(malformedNoAlias, "bad_noalias",
                               "even dense i64 array") &&
         expectAnalysisFailure(manualSync, "manual_sync",
                               "pre-existing pipe synchronization") &&
         expectAnalysisFailure(unmodeledEffect, "unmodeled",
                               "unrecognized helper") &&
         expectAnalysisFailure(resultIf, "result_if",
                               "result-carrying scf.if") &&
         expectAnalysisFailure(iterArgs, "iter_args", "scf.for iter_args") &&
         expectAnalysisFailure(asynchronousControl, "async_control",
                               "asynchronous scf.if condition");
}

bool testRejectsAllExplicitSyncForms() {
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
         expectAnalysisFailure(source, "fence", diagnostic);
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
  const bool basicLimits =
      expectAnalysisFailure(basic, "bounded", "node limit exceeded",
                            nodeLimit) &&
      expectAnalysisFailure(basic, "bounded", "storage-access limit exceeded",
                            storageLimit);
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

bool testL0OwnershipProtocolTrustBoundary() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2a3"} {
      func.func @ownership(%limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %addr0 = arith.constant 0 : i64
        %addr16384 = arith.constant 16384 : i64
        %addr32768 = arith.constant 32768 : i64
        %leftSource = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %rightSource = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<mat, 128x256xf16, slayout=col_major>
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
          pto.textract ins(%leftSource, %c0, %c0 :
            !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index, index)
            outs(%left0 :
              !pto.tile_buf<left, 128x64xf16, slayout=row_major>)
          pto.textract ins(%rightSource, %c0, %c0 :
            !pto.tile_buf<mat, 128x256xf16, slayout=col_major>, index, index)
            outs(%right0 :
              !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
          pto.tmatmul.acc ins(%acc, %left0, %right0 :
            !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
                          slayout=row_major, fractal=1024>,
            !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
            !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
            outs(%acc : !pto.tile_buf<acc, 128x256xf32,
              blayout=col_major, slayout=row_major, fractal=1024>)
          pto.textract ins(%leftSource, %c0, %c64 :
            !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index, index)
            outs(%left1 :
              !pto.tile_buf<left, 128x64xf16, slayout=row_major>)
          pto.textract ins(%rightSource, %c64, %c0 :
            !pto.tile_buf<mat, 128x256xf16, slayout=col_major>, index, index)
            outs(%right1 :
              !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
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
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse L0 ownership fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("ownership"));
  if (!check(succeeded(program), "build L0 ownership graph")) {
    return false;
  }
  CanonicalSyncOwnershipResult result =
      discoverCanonicalSyncOwnershipCycles(*program);
  if (!check(!result.truncated &&
                 result.error == CanonicalSyncOwnershipError::None &&
                 result.cycles.size() == 1,
             "discover one bounded L0 ownership cycle")) {
    return false;
  }
  const CanonicalSyncOwnershipCycle &cycle = result.cycles.front();
  CanonicalSyncOwnershipOptions truncatedOptions;
  truncatedOptions.maximumInspections = 0;
  const CanonicalSyncOwnershipResult truncated =
      discoverCanonicalSyncOwnershipCycles(*program, truncatedOptions);
  std::optional<CanonicalSyncOwnershipProtocol> protocol =
      makeCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1);
  if (!check(truncated && truncated.truncated && truncated.cycles.empty(),
             "bound ownership discovery without partial protocols") ||
      !check(verifyCanonicalSyncOwnershipCycle(*program, cycle),
             "independently verify L0 ownership cycle") ||
      !check(protocol.has_value(), "build L0 ownership protocol") ||
      !check(verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                  *protocol),
             "independently verify L0 ownership protocol")) {
    return false;
  }
  if (!check(
          program->getTargetCapabilities().mte1L0ReadySetCompletesPrefix &&
              program->getTargetCapabilities().mL0AlternativeJoinSetCompletes &&
              protocol->ready.eventUses.size() == 1,
          "use target-qualified L0 ready prefix and alternative join")) {
    return false;
  }

  const std::optional<CanonicalSyncMechanismDescriptor> expectedAtomic =
      makeCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0, 1);
  const std::size_t splitActionLimit = std::max(
      protocol->ready.actions.size(), protocol->release.actions.size());
  if (!check(expectedAtomic &&
                 expectedAtomic->actions.size() > splitActionLimit,
             "fixture distinguishes split and atomic action limits")) {
    return false;
  }
  CanonicalSyncBuildOptions splitLimitOptions;
  splitLimitOptions.problemLimits.maximumActionsPerMechanism = splitActionLimit;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> splitLimitProblem =
      buildCanonicalSyncSingletonProblem(*program, splitLimitOptions);
  if (!check(succeeded(splitLimitProblem),
             "retain split L0 protocols when the atomic recipe is oversized")) {
    return false;
  }
  const std::uint32_t limitedCube =
      static_cast<std::uint32_t>(PipelineType::PIPE_M);
  const std::uint32_t limitedMte1 =
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1);
  std::optional<CanonicalSyncEventDomainId> limitedReadyDomain;
  std::optional<CanonicalSyncEventDomainId> limitedReleaseDomain;
  for (const CanonicalSyncEventDomain &domain :
       (*splitLimitProblem)->getDomains()) {
    if (domain.sourceResource == limitedMte1 &&
        domain.targetResource == limitedCube) {
      limitedReadyDomain = domain.id;
    }
    if (domain.sourceResource == limitedCube &&
        domain.targetResource == limitedMte1) {
      limitedReleaseDomain = domain.id;
    }
  }
  if (!check(limitedReadyDomain && limitedReleaseDomain,
             "find split L0 event domains")) {
    return false;
  }
  const std::optional<CanonicalSyncOwnershipProtocol> expectedSplit =
      makeCanonicalSyncOwnershipProtocol(*program, cycle, *limitedReadyDomain,
                                         *limitedReleaseDomain);
  const std::optional<CanonicalSyncMechanismDescriptor> limitedAtomic =
      makeCanonicalSyncAtomicOwnershipProtocol(
          *program, cycle, *limitedReadyDomain, *limitedReleaseDomain);
  if (!check(expectedSplit && limitedAtomic,
             "rebuild split L0 recipes for the admitted domains")) {
    return false;
  }
  const bool hasSplitReady = llvm::any_of(
      (*splitLimitProblem)->getMechanisms(), [&](const auto &mechanism) {
        return sameDescriptor(mechanism.descriptor, expectedSplit->ready);
      });
  const bool hasSplitRelease = llvm::any_of(
      (*splitLimitProblem)->getMechanisms(), [&](const auto &mechanism) {
        return sameDescriptor(mechanism.descriptor, expectedSplit->release);
      });
  const bool hasOversizedAtomic = llvm::any_of(
      (*splitLimitProblem)->getMechanisms(), [&](const auto &mechanism) {
        return sameDescriptor(mechanism.descriptor, *limitedAtomic);
      });
  if (!check(hasSplitReady && hasSplitRelease && !hasOversizedAtomic,
             "admit both split alternatives independently of atomic limits")) {
    return false;
  }

  std::vector<SyncCoverCompletionSupply> releaseSupplies;
  SyncCoverDemandSet qualified(program->getGraph().getDemands().size());
  for (const CanonicalSyncSupplyBinding &binding : protocol->release.supplies) {
    releaseSupplies.push_back({0, binding.edge, binding.allowedDemands});
    for (SyncCoverDemandId demand : binding.allowedDemands) {
      qualified.insert(demand);
    }
  }
  SyncCoverExpandedProgram expansion(program->getGraph());
  const SyncCoverCoverageResult releaseCoverage =
      computeSyncCoverCoverage(program->getGraph(), expansion, releaseSupplies);
  const SyncCoverSingletonCoverageResult singletonRelease =
      computeSyncCoverSingletonCoverage(program->getGraph(), expansion, 1,
                                        releaseSupplies);
  std::vector<SyncCoverCompletionSupply> ownershipSupplies = releaseSupplies;
  for (SyncCoverCompletionSupply &supply : ownershipSupplies) {
    supply.mechanism = 1;
  }
  for (const CanonicalSyncSupplyBinding &binding : protocol->ready.supplies) {
    ownershipSupplies.push_back({0, binding.edge, binding.allowedDemands});
  }
  const SyncCoverCoverageResult ownershipCoverage = computeSyncCoverCoverage(
      program->getGraph(), expansion, ownershipSupplies);
  if (!check(releaseCoverage && !qualified.empty(),
             "compute qualified release coverage") ||
      !check(singletonRelease &&
                 singletonRelease.mechanisms.front() == releaseCoverage.covered,
             "batched singleton coverage preserves demand qualifications") ||
      !check(qualified.containsAll(releaseCoverage.covered),
             "release covers only its exact storage demands") ||
      !check(ownershipCoverage &&
                 ownershipCoverage.covered.containsAll(qualified),
             "ready and release compose over every exact ownership demand")) {
    return false;
  }
  CanonicalSyncBuildOptions buildOptions;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, buildOptions);
  if (!check(succeeded(problem), "build ownership pattern problem")) {
    return false;
  }
  const std::uint32_t cube = static_cast<std::uint32_t>(PipelineType::PIPE_M);
  const std::uint32_t mte1 =
      static_cast<std::uint32_t>(PipelineType::PIPE_MTE1);
  std::optional<CanonicalSyncMechanismId> atomicOwnership;
  for (const CanonicalSyncMechanism &mechanism : (*problem)->getMechanisms()) {
    if (mechanism.descriptor.kind != CanonicalSyncMechanismKind::Protocol ||
        mechanism.descriptor.eventUses.size() < 2) {
      continue;
    }
    bool hasReady = false;
    bool hasRelease = false;
    for (const CanonicalSyncEventUse &use : mechanism.descriptor.eventUses) {
      const CanonicalSyncEventDomain &domain =
          (*problem)->getDomains()[use.domain];
      hasReady = hasReady || (domain.sourceResource == mte1 &&
                              domain.targetResource == cube);
      hasRelease = hasRelease || (domain.sourceResource == cube &&
                                  domain.targetResource == mte1);
    }
    if (hasReady && hasRelease) {
      atomicOwnership = mechanism.id;
      break;
    }
  }
  if (!check(atomicOwnership.has_value(),
             "retain one atomic L0 ownership mechanism")) {
    return false;
  }
  if (!check((*problem)->getPatterns()[*atomicOwnership].coverage.containsAll(
                 qualified),
             "ground complete L0 ownership in one singleton pattern")) {
    return false;
  }
  const auto familyAblationKeepsMechanisms =
      [&](CanonicalSyncPatternKind disabled) {
        CanonicalSyncBuildOptions ablationOptions;
        switch (disabled) {
        case CanonicalSyncPatternKind::PipelineScope:
          ablationOptions.patterns.enablePipelineScope = false;
          break;
        case CanonicalSyncPatternKind::RoundTrip:
          ablationOptions.patterns.enableRoundTrip = false;
          break;
        case CanonicalSyncPatternKind::Singleton:
          return false;
        }
        FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> ablated =
            buildCanonicalSyncSingletonProblem(*program, ablationOptions);
        return succeeded(ablated) &&
               sameMechanismCatalog(**problem, **ablated) &&
               llvm::none_of((*ablated)->getPatterns(),
                             [&](const CanonicalSyncPattern &pattern) {
                               return pattern.kind == disabled;
                             });
      };
  if (!check(familyAblationKeepsMechanisms(
                 CanonicalSyncPatternKind::PipelineScope),
             "pipeline-pattern ablation keeps the mechanism catalog") ||
      !check(familyAblationKeepsMechanisms(CanonicalSyncPatternKind::RoundTrip),
             "round-trip ablation keeps the mechanism catalog")) {
    return false;
  }
  CanonicalSyncBuildOptions atomicOptions;
  atomicOptions.patterns.enablePipelineScope = false;
  atomicOptions.patterns.enableRoundTrip = false;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> atomicProblem =
      buildCanonicalSyncSingletonProblem(*program, atomicOptions);
  const bool onlySingletonPatterns =
      succeeded(atomicProblem) &&
      llvm::all_of((*atomicProblem)->getPatterns(),
                   [](const CanonicalSyncPattern &pattern) {
                     return pattern.kind == CanonicalSyncPatternKind::Singleton;
                   });
  if (!check(onlySingletonPatterns,
             "disable every composite family for atomic-only analysis")) {
    return false;
  }
  const CanonicalSyncSelection selection =
      selectCanonicalSyncPatterns(**problem);
  bool selectedReadyProtocol = false;
  bool selectedReleaseProtocol = false;
  bool selectedBarrier = false;
  if (selection) {
    for (CanonicalSyncMechanismId mechanism : selection.mechanisms) {
      const CanonicalSyncMechanismDescriptor &descriptor =
          (*problem)->getMechanisms()[mechanism].descriptor;
      selectedBarrier = selectedBarrier ||
                        descriptor.kind == CanonicalSyncMechanismKind::Barrier;
      if (descriptor.kind != CanonicalSyncMechanismKind::Protocol) {
        continue;
      }
      for (const CanonicalSyncEventUse &use : descriptor.eventUses) {
        const CanonicalSyncEventDomain &domain =
            (*problem)->getDomains()[use.domain];
        selectedReadyProtocol =
            selectedReadyProtocol ||
            (domain.sourceResource == mte1 && domain.targetResource == cube);
        selectedReleaseProtocol =
            selectedReleaseProtocol ||
            (domain.sourceResource == cube && domain.targetResource == mte1);
      }
    }
  }
  const bool selectedOwnership =
      selectedReadyProtocol && selectedReleaseProtocol && !selectedBarrier;
  if (!check(static_cast<bool>(selection),
             "select ownership pattern problem") ||
      !check(selectedOwnership,
             "select verified ownership instead of barrier fallbacks")) {
    return false;
  }

  CanonicalSyncOwnershipProtocol missingAction = *protocol;
  missingAction.ready.actions.pop_back();
  CanonicalSyncOwnershipProtocol missingBinding = *protocol;
  missingBinding.release.supplies.pop_back();
  CanonicalSyncOwnershipProtocol wrongLane = *protocol;
  wrongLane.ready.actions.front().eventLane = cycle.lanes.size();
  CanonicalSyncOwnershipProtocol wrongQualifiedDemand = *protocol;
  if (!check(!wrongQualifiedDemand.release.supplies.empty() &&
                 !wrongQualifiedDemand.release.supplies.front()
                      .allowedDemands.empty(),
             "release protocol carries exact demand qualifications")) {
    return false;
  }
  const std::vector<SyncCoverDemandId> &allowed =
      wrongQualifiedDemand.release.supplies.front().allowedDemands;
  std::optional<SyncCoverDemandId> unrelated;
  for (SyncCoverDemandId demand = 0;
       demand < program->getGraph().getDemands().size(); ++demand) {
    if (!std::binary_search(allowed.begin(), allowed.end(), demand)) {
      unrelated = demand;
      break;
    }
  }
  if (!check(unrelated.has_value(),
             "fixture contains an unrelated completion demand")) {
    return false;
  }
  wrongQualifiedDemand.release.supplies.front().allowedDemands = {*unrelated};
  CanonicalSyncOwnershipCycle wrongProducerLane = cycle;
  wrongProducerLane.paths.front().uses.front().producerLane =
      (wrongProducerLane.paths.front().uses.front().lane + 1) %
      wrongProducerLane.lanes.size();
  CanonicalSyncOwnershipCycle wrongRole = cycle;
  wrongRole.paths.front().uses.front().producers.front() =
      wrongRole.paths.front().uses.front().consumers.front();
  CanonicalSyncOwnershipProtocol wrongPrefixAnchor = *protocol;
  wrongPrefixAnchor.ready.actions.front().anchor.node =
      cycle.paths.front().uses.front().producers.front();
  module->getOperation()->removeAttr("pto.target_arch");
  FailureOr<CanonicalSyncProgram> conservativeProgram =
      buildCanonicalSyncProgram(
          module->lookupSymbol<func::FuncOp>("ownership"));
  CanonicalSyncOwnershipResult conservativeCycles =
      succeeded(conservativeProgram)
          ? discoverCanonicalSyncOwnershipCycles(*conservativeProgram)
          : CanonicalSyncOwnershipResult{};
  const std::optional<CanonicalSyncOwnershipProtocol> conservativeProtocol =
      conservativeCycles.cycles.size() == 1
          ? makeCanonicalSyncOwnershipProtocol(
                *conservativeProgram, conservativeCycles.cycles.front(), 0, 1)
          : std::nullopt;
  const bool conservativeReady = succeeded(conservativeProgram) &&
                                 conservativeCycles &&
                                 !conservativeProgram->getTargetCapabilities()
                                      .mte1L0ReadySetCompletesPrefix &&
                                 !conservativeProgram->getTargetCapabilities()
                                      .mL0AlternativeJoinSetCompletes &&
                                 conservativeProtocol &&
                                 conservativeProtocol->ready.eventUses.size() ==
                                     conservativeCycles.cycles.front()
                                         .paths.front()
                                         .uses.front()
                                         .producers.size();
  const bool rejectsPrefixWithoutEvidence =
      conservativeReady &&
      !verifyCanonicalSyncOwnershipProtocol(*conservativeProgram,
                                            conservativeCycles.cycles.front(),
                                            0, 1, *protocol);
  return check(!verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                     missingAction),
               "reject ownership protocol with a missing action") &&
         check(!verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                     missingBinding),
               "reject ownership protocol with a missing binding") &&
         check(!verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                     wrongLane),
               "reject ownership protocol with a wrong event lane") &&
         check(!verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                     wrongQualifiedDemand),
               "reject ownership release qualified for an unrelated demand") &&
         check(!verifyCanonicalSyncOwnershipCycle(*program, wrongProducerLane),
               "reject ownership cycle with a mismatched producer lane") &&
         check(!verifyCanonicalSyncOwnershipCycle(*program, wrongRole),
               "reject ownership cycle with a reversed access role") &&
         check(!verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                     wrongPrefixAnchor),
               "reject a prefix-ready protocol with an early set") &&
         check(conservativeReady,
               "retain per-producer L0 readiness without target evidence") &&
         check(rejectsPrefixWithoutEvidence,
               "reject prefix-ready coverage without target evidence");
}

bool testStableL1OwnershipProtocol() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2a3"} {
      func.func @ownership(%source0: !pto.ptr<f16, gm>,
                           %source1: !pto.ptr<f16, gm>, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %addr0 = arith.constant 0 : i64
        %addr16384 = arith.constant 16384 : i64
        %addr32768 = arith.constant 32768 : i64
        %view0 = pto.make_tensor_view %source0, shape = [%c128, %c128],
          strides = [%c128, %c1] {layout = #pto.layout<nd>} :
          !pto.tensor_view<?x?xf16>
        %view1 = pto.make_tensor_view %source1, shape = [%c128, %c128],
          strides = [%c128, %c1] {layout = #pto.layout<nd>} :
          !pto.tensor_view<?x?xf16>
        %part0 = pto.partition_view %view0, offsets = [%c0, %c0],
          sizes = [%c128, %c128] : !pto.tensor_view<?x?xf16>
        %part1 = pto.partition_view %view1, offsets = [%c0, %c0],
          sizes = [%c128, %c128] : !pto.tensor_view<?x?xf16>
        %mat0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %mat1 = pto.alloc_tile addr = %addr32768 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %left0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        %left1 = pto.alloc_tile addr = %addr16384 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        scf.for %i = %c0 to %limit step %c1 {
          pto.tload ins(%part0 : !pto.partition_tensor_view<128x128xf16>)
            outs(%mat0 : !pto.tile_buf<mat, 128x128xf16,
                                    slayout=row_major>)
          pto.tload ins(%part1 : !pto.partition_tensor_view<128x128xf16>)
            outs(%mat1 : !pto.tile_buf<mat, 128x128xf16,
                                    slayout=row_major>)
          scf.for %k = %c0 to %c2 step %c1 {
            pto.textract ins(%mat0, %c0, %c0 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left0 : !pto.tile_buf<left, 128x64xf16,
                                               slayout=row_major>)
            pto.textract ins(%mat1, %c0, %c64 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left1 : !pto.tile_buf<left, 128x64xf16,
                                               slayout=row_major>)
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse stable L1 fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("ownership"));
  if (!check(succeeded(program), "build stable L1 graph")) {
    return false;
  }
  CanonicalSyncOwnershipResult result =
      discoverCanonicalSyncOwnershipCycles(*program);
  if (!check(result && !result.truncated && result.cycles.size() == 1 &&
                 result.cycles.front().kind ==
                     CanonicalSyncOwnershipKind::L1Tile,
             "discover one exact stable L1 ownership cycle")) {
    return false;
  }
  const CanonicalSyncOwnershipCycle &cycle = result.cycles.front();
  const bool boundaryAnchors = llvm::all_of(
      cycle.paths.front().uses, [](const CanonicalSyncOwnershipUse &use) {
        return use.producers.size() == 1 && !use.consumers.empty() &&
               use.writeAcquire.kind == SyncCoverAnchorKind::BeforeNode &&
               use.ready.kind == SyncCoverAnchorKind::AfterNode &&
               use.readAcquire.kind == SyncCoverAnchorKind::ScopeEntry &&
               use.release.kind == SyncCoverAnchorKind::ScopeExit;
      });
  std::optional<CanonicalSyncOwnershipProtocol> protocol =
      makeCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1);
  std::optional<CanonicalSyncMechanismDescriptor> atomicProtocol =
      makeCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0, 1);
  if (!check(boundaryAnchors,
             "anchor L1 ownership around the complete consumer loop") ||
      !check(protocol.has_value(), "build stable L1 ownership protocol") ||
      !check(verifyCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1,
                                                  *protocol),
             "independently verify stable L1 ownership protocol") ||
      !check(atomicProtocol.has_value() &&
                 verifyCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0,
                                                            1, *atomicProtocol),
             "verify the indivisible stable L1 handshake")) {
    return false;
  }
  SyncCoverDemandSet qualified(program->getGraph().getDemands().size());
  for (const CanonicalSyncSupplyBinding &binding : protocol->release.supplies) {
    for (SyncCoverDemandId demand : binding.allowedDemands) {
      qualified.insert(demand);
    }
  }
  CanonicalSyncBuildOptions options;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  std::optional<CanonicalSyncMechanismId> ownershipMechanism;
  if (succeeded(problem)) {
    for (const CanonicalSyncMechanism &mechanism :
         (*problem)->getMechanisms()) {
      if (mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
          mechanism.descriptor.eventUses.size() == 2) {
        ownershipMechanism = mechanism.id;
        break;
      }
    }
  }
  if (!check(!qualified.empty(), "qualify exact MAT ownership demands") ||
      !check(succeeded(problem), "build stable L1 pattern problem") ||
      !check(ownershipMechanism.has_value() &&
                 (*problem)
                     ->getPatterns()[*ownershipMechanism]
                     .coverage.containsAll(qualified),
             "ground stable L1 ownership as one atomic mechanism")) {
    return false;
  }
  module->getOperation()->setAttr("pto.target_arch",
                                  StringAttr::get(&context, "a5"));
  FailureOr<CanonicalSyncProgram> a5Program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("ownership"));
  CanonicalSyncOwnershipResult a5Cycles =
      succeeded(a5Program) ? discoverCanonicalSyncOwnershipCycles(*a5Program)
                           : CanonicalSyncOwnershipResult{};
  const bool a5FailsClosed =
      succeeded(a5Program) && a5Cycles && a5Cycles.cycles.size() == 1 &&
      !makeCanonicalSyncAtomicOwnershipProtocol(*a5Program,
                                                a5Cycles.cycles.front(), 0, 1);
  if (!check(a5FailsClosed,
             "gate scope-exit L1 release on explicit target evidence")) {
    return false;
  }
  CanonicalSyncOwnershipCycle wrongAnchor = cycle;
  wrongAnchor.paths.front().uses.front().release.kind =
      SyncCoverAnchorKind::AfterNode;
  CanonicalSyncMechanismDescriptor partial = *atomicProtocol;
  partial.actions.pop_back();
  CanonicalSyncOwnershipCycle malformed;
  return check(!verifyCanonicalSyncOwnershipCycle(*program, wrongAnchor),
               "reject L1 release before the complete consumer loop") &&
         check(!verifyCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0,
                                                           1, partial),
               "reject a partial stable L1 handshake") &&
         check(!verifyCanonicalSyncAtomicOwnershipProtocol(
                   *program, malformed, 0, 1, *atomicProtocol),
               "fail closed for a malformed ownership cycle");
}

bool testHierarchicalL1OwnershipProtocol() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2a3"} {
      func.func @hierarchical(%source0: !pto.ptr<f16, gm>,
                              %source1: !pto.ptr<f16, gm>,
                              %outer_limit: index, %inner_limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %addr0 = arith.constant 0 : i64
        %addr16384 = arith.constant 16384 : i64
        %addr32768 = arith.constant 32768 : i64
        %view0 = pto.make_tensor_view %source0, shape = [%c128, %c128],
          strides = [%c128, %c1] {layout = #pto.layout<nd>} :
          !pto.tensor_view<?x?xf16>
        %view1 = pto.make_tensor_view %source1, shape = [%c128, %c128],
          strides = [%c128, %c1] {layout = #pto.layout<nd>} :
          !pto.tensor_view<?x?xf16>
        %part0 = pto.partition_view %view0, offsets = [%c0, %c0],
          sizes = [%c128, %c128] : !pto.tensor_view<?x?xf16>
        %part1 = pto.partition_view %view1, offsets = [%c0, %c0],
          sizes = [%c128, %c128] : !pto.tensor_view<?x?xf16>
        %mat0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %mat1 = pto.alloc_tile addr = %addr32768 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %left0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        %left1 = pto.alloc_tile addr = %addr16384 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        scf.for %i = %c0 to %outer_limit step %c1 {
          scf.for %k = %c0 to %inner_limit step %c1 {
            pto.tload ins(%part0 : !pto.partition_tensor_view<128x128xf16>)
              outs(%mat0 : !pto.tile_buf<mat, 128x128xf16,
                                      slayout=row_major>)
            pto.tload ins(%part1 : !pto.partition_tensor_view<128x128xf16>)
              outs(%mat1 : !pto.tile_buf<mat, 128x128xf16,
                                      slayout=row_major>)
            pto.textract ins(%mat0, %c0, %c0 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left0 : !pto.tile_buf<left, 128x64xf16,
                                               slayout=row_major>)
            pto.textract ins(%mat1, %c0, %c64 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left1 : !pto.tile_buf<left, 128x64xf16,
                                               slayout=row_major>)
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse hierarchical L1 fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("hierarchical"));
  if (!check(succeeded(program), "build hierarchical L1 graph")) {
    return false;
  }
  CanonicalSyncOwnershipResult ownership =
      discoverCanonicalSyncOwnershipCycles(*program);
  auto cycle = llvm::find_if(ownership.cycles, [&](const auto &candidate) {
    return candidate.kind == CanonicalSyncOwnershipKind::L1Tile &&
           program->getGraph().getScopes()[candidate.recurrenceScope].parent !=
               0;
  });
  if (!check(ownership && cycle != ownership.cycles.end(),
             "discover nested stable L1 ownership")) {
    return false;
  }
  const SyncCoverScopeId outer =
      program->getGraph().getScopes()[cycle->recurrenceScope].parent;
  std::optional<CanonicalSyncMechanismDescriptor> descriptor =
      makeCanonicalSyncHierarchicalL1Protocol(*program, *cycle, outer, 0, 1);
  const bool builtDescriptor =
      check(descriptor.has_value(), "build hierarchical L1 protocol");
  const bool verifiedDescriptor =
      builtDescriptor && check(verifyCanonicalSyncHierarchicalL1Protocol(
                                   *program, *cycle, outer, 0, 1, *descriptor),
                               "independently verify hierarchical L1 protocol");
  if (!verifiedDescriptor) {
    return false;
  }
  const bool innerRecurrence = llvm::all_of(
      descriptor->eventUses, [&](const CanonicalSyncEventUse &use) {
        return use.recurrenceScope == cycle->recurrenceScope;
      });
  const auto outerLifetime = llvm::find_if(
      descriptor->eventUses, [&](const CanonicalSyncEventUse &use) {
        return use.lifetimeScope == outer;
      });
  const bool outerSupply = llvm::any_of(
      descriptor->supplies, [&](const CanonicalSyncSupplyBinding &binding) {
        return binding.edge.scope == outer && binding.edge.distance == 1;
      });
  CanonicalSyncMechanismDescriptor wrongLifetime = *descriptor;
  const std::size_t lifetimeIndex =
      static_cast<std::size_t>(outerLifetime - descriptor->eventUses.begin());
  if (outerLifetime != descriptor->eventUses.end()) {
    wrongLifetime.eventUses[lifetimeIndex].lifetimeScope =
        cycle->recurrenceScope;
  }
  CanonicalSyncMechanismDescriptor wrongSupply = *descriptor;
  auto outerBinding = llvm::find_if(
      wrongSupply.supplies, [&](const CanonicalSyncSupplyBinding &binding) {
        return binding.edge.scope == outer && binding.edge.distance == 1;
      });
  if (outerBinding != wrongSupply.supplies.end()) {
    ++outerBinding->edge.distance;
  }
  CanonicalSyncMechanismDescriptor missingAction = *descriptor;
  missingAction.actions.pop_back();
  return check(innerRecurrence && outerLifetime != descriptor->eventUses.end(),
               "hold hierarchical event IDs across the enclosing loop") &&
         check(outerSupply, "supply an enclosing-loop ownership handoff") &&
         check(outerLifetime != descriptor->eventUses.end() &&
                   !verifyCanonicalSyncHierarchicalL1Protocol(
                       *program, *cycle, outer, 0, 1, wrongLifetime),
               "reject a narrowed hierarchical lifetime") &&
         check(outerBinding != wrongSupply.supplies.end() &&
                   !verifyCanonicalSyncHierarchicalL1Protocol(
                       *program, *cycle, outer, 0, 1, wrongSupply),
               "reject a tampered outer recurrence supply") &&
         check(!verifyCanonicalSyncHierarchicalL1Protocol(
                   *program, *cycle, outer, 0, 1, missingAction),
               "reject a partial hierarchical protocol");
}

bool testAlternatingL1OwnershipProtocol() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2a3"} {
      func.func @prefetch(%source: !pto.ptr<f16, gm>, %outer_limit: index,
                          %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c64 = arith.constant 64 : index
        %c128 = arith.constant 128 : index
        %addr0 = arith.constant 0 : i64
        %addr32768 = arith.constant 32768 : i64
        %addr65536 = arith.constant 65536 : i64
        %addr98304 = arith.constant 98304 : i64
        %view = pto.make_tensor_view %source, shape = [%c128, %c128],
          strides = [%c128, %c1] {layout = #pto.layout<nd>} :
          !pto.tensor_view<?x?xf16>
        %part = pto.partition_view %view, offsets = [%c0, %c0],
          sizes = [%c128, %c128] : !pto.tensor_view<?x?xf16>
        %mat0 = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %mat1 = pto.alloc_tile addr = %addr32768 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %stable0 = pto.alloc_tile addr = %addr65536 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %stable1 = pto.alloc_tile addr = %addr98304 :
          !pto.tile_buf<mat, 128x128xf16, slayout=row_major>
        %left = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        scf.for %outer = %c0 to %outer_limit step %c1 {
          pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
            outs(%mat0 : !pto.tile_buf<mat, 128x128xf16,
                                    slayout=row_major>)
          scf.for %i = %c0 to %limit step %c1 {
          %phase = arith.remsi %i, %c2 : index
          %even = arith.cmpi eq, %phase, %c0 : index
          scf.if %even {
            pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
              outs(%stable0 : !pto.tile_buf<mat, 128x128xf16,
                                               slayout=row_major>)
            pto.textract ins(%stable0, %c0, %c0 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
              outs(%stable1 : !pto.tile_buf<mat, 128x128xf16,
                                               slayout=row_major>)
            pto.textract ins(%stable1, %c0, %c64 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            pto.textract ins(%mat0, %c0, %c0 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            %next = arith.addi %i, %c1 : index
            %has_next = arith.cmpi slt, %next, %limit : index
            scf.if %has_next {
              pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
                outs(%mat1 : !pto.tile_buf<mat, 128x128xf16,
                                            slayout=row_major>)
            }
          } else {
            pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
              outs(%stable0 : !pto.tile_buf<mat, 128x128xf16,
                                               slayout=row_major>)
            pto.textract ins(%stable0, %c0, %c0 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
              outs(%stable1 : !pto.tile_buf<mat, 128x128xf16,
                                               slayout=row_major>)
            pto.textract ins(%stable1, %c0, %c64 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            pto.textract ins(%mat1, %c0, %c64 :
              !pto.tile_buf<mat, 128x128xf16, slayout=row_major>, index,
              index) outs(%left : !pto.tile_buf<left, 128x64xf16,
                                             slayout=row_major>)
            %next = arith.addi %i, %c1 : index
            %has_next = arith.cmpi slt, %next, %limit : index
            scf.if %has_next {
              pto.tload ins(%part : !pto.partition_tensor_view<128x128xf16>)
                outs(%mat0 : !pto.tile_buf<mat, 128x128xf16,
                                            slayout=row_major>)
            }
          }
          }
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse alternating L1 fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("prefetch"));
  if (!check(succeeded(program), "build alternating L1 graph")) {
    return false;
  }
  CanonicalSyncOwnershipResult result =
      discoverCanonicalSyncOwnershipCycles(*program);
  auto alternating = llvm::find_if(result.cycles, [](const auto &cycle) {
    return cycle.protocol ==
           CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
  });
  const std::size_t stableCycles =
      llvm::count_if(result.cycles, [](const auto &cycle) {
        return cycle.kind == CanonicalSyncOwnershipKind::L1Tile &&
               cycle.protocol == CanonicalSyncOwnershipProtocolKind::RoundTrip;
      });
  if (!check(result && !result.truncated && stableCycles == 1 &&
                 alternating != result.cycles.end(),
             "discover alternating-prefetch ownership")) {
    return false;
  }
  const CanonicalSyncOwnershipCycle &cycle = *alternating;
  std::optional<CanonicalSyncOwnershipProtocol> protocol =
      makeCanonicalSyncOwnershipProtocol(*program, cycle, 0, 1);
  std::optional<CanonicalSyncMechanismDescriptor> atomic =
      makeCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0, 1);
  const SyncCoverScopeId outerScope =
      program->getGraph().getScopes()[cycle.recurrenceScope].parent;
  std::optional<CanonicalSyncMechanismDescriptor> hierarchical =
      makeCanonicalSyncHierarchicalL1Protocol(*program, cycle, outerScope, 0,
                                              1);
  if (!check(verifyCanonicalSyncOwnershipCycle(*program, cycle),
             "independently verify alternating ownership") ||
      !check(protocol && verifyCanonicalSyncOwnershipProtocol(*program, cycle,
                                                              0, 1, *protocol),
             "verify alternating ready/release lifecycle") ||
      !check(atomic && verifyCanonicalSyncAtomicOwnershipProtocol(
                           *program, cycle, 0, 1, *atomic),
             "verify atomic alternating lifecycle") ||
      !check(hierarchical &&
                 verifyCanonicalSyncHierarchicalL1Protocol(
                     *program, cycle, outerScope, 0, 1, *hierarchical),
             "verify hierarchical alternating lifecycle")) {
    return false;
  }
  CanonicalSyncBuildOptions options;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  std::optional<CanonicalSyncMechanismId> alternatingMechanism;
  if (succeeded(problem)) {
    for (const CanonicalSyncMechanism &mechanism :
         (*problem)->getMechanisms()) {
      if (mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
          mechanism.descriptor.actions.size() == atomic->actions.size()) {
        alternatingMechanism = mechanism.id;
      }
    }
  }
  SyncCoverDemandSet managedDemands(program->getGraph().getDemands().size());
  std::set<SyncCoverNodeId> managedNodes(cycle.initialProducers.begin(),
                                         cycle.initialProducers.end());
  for (const CanonicalSyncOwnershipPath &path : cycle.paths) {
    for (const CanonicalSyncOwnershipUse &use : path.uses) {
      managedNodes.insert(use.producers.begin(), use.producers.end());
      managedNodes.insert(use.consumers.begin(), use.consumers.end());
    }
  }
  const SyncCoverGraph &graph = program->getGraph();
  for (auto [demandId, demand] : llvm::enumerate(graph.getDemands())) {
    if (demand.scope != cycle.recurrenceScope ||
        demand.storageWitnesses.empty() ||
        managedNodes.count(demand.source) == 0 ||
        managedNodes.count(demand.target) == 0) {
      continue;
    }
    const bool exactManaged = llvm::all_of(
        demand.storageWitnesses, [&](SyncCoverStorageWitnessId witnessId) {
          if (witnessId >= graph.getStorageWitnesses().size()) {
            return false;
          }
          const SyncCoverStorageWitness &witness =
              graph.getStorageWitnesses()[witnessId];
          if (witness.sourceAccess >= graph.getStorageAccesses().size() ||
              witness.targetAccess >= graph.getStorageAccesses().size()) {
            return false;
          }
          const SyncCoverStorageAccess &source =
              graph.getStorageAccesses()[witness.sourceAccess];
          const SyncCoverStorageAccess &target =
              graph.getStorageAccesses()[witness.targetAccess];
          return source.exactPhysical && target.exactPhysical &&
                 llvm::any_of(
                     cycle.lanes, [&](const CanonicalSyncOwnershipLane &lane) {
                       return llvm::any_of(
                           lane.slots,
                           [&](const CanonicalSyncOwnershipSlot &slot) {
                             return source.domain == slot.domain &&
                                    target.domain == slot.domain &&
                                    slot.extent.begin <=
                                        witness.overlap.begin &&
                                    witness.overlap.end <= slot.extent.end;
                           });
                     });
        });
    if (exactManaged) {
      managedDemands.insert(demandId);
    }
  }
  const CanonicalSyncPattern *alternatingPattern =
      alternatingMechanism && succeeded(problem)
          ? &(*problem)->getPatterns()[*alternatingMechanism]
          : nullptr;
  CanonicalSyncSelection selection;
  if (succeeded(problem)) {
    selection = selectCanonicalSyncPatterns(**problem);
  }
  if (!check(succeeded(problem), "build alternating pattern problem") ||
      !check(alternatingMechanism.has_value(),
             "admit alternating ownership mechanism") ||
      !check(!managedDemands.empty() && alternatingPattern &&
                 alternatingPattern->coverage.containsAll(managedDemands),
             "cover every exact managed MAT demand with the lifecycle") ||
      !check(selection && llvm::is_contained(selection.mechanisms,
                                             *alternatingMechanism),
             "select alternating ownership mechanism")) {
    return false;
  }
  const std::size_t guarded =
      llvm::count_if(atomic->actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::LoopNonEmpty;
      });
  CanonicalSyncMechanismDescriptor missingGuard = *atomic;
  auto guardedAction = llvm::find_if(
      missingGuard.actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::LoopNonEmpty;
      });
  guardedAction->guard = CanonicalSyncActionGuardKind::None;
  guardedAction->guardScope.reset();
  CanonicalSyncOwnershipCycle wrongTransition = cycle;
  wrongTransition.paths[1].uses.front().producerLane =
      wrongTransition.paths[1].uses.front().lane;
  CanonicalSyncOwnershipCycle wrongInitial = cycle;
  wrongInitial.initialReadyLane = wrongInitial.initiallyFreeLanes.front();
  CanonicalSyncMechanismDescriptor wrongComposite = *atomic;
  auto composite = llvm::find_if(
      wrongComposite.supplies, [](const CanonicalSyncSupplyBinding &binding) {
        return binding.proof ==
               CanonicalSyncSupplyProof::VerifiedCompositeProtocol;
      });
  const bool hasComposite = composite != wrongComposite.supplies.end();
  if (hasComposite) {
    ++composite->edge.distance;
  }
  const bool rejectsMissingGuard = !verifyCanonicalSyncAtomicOwnershipProtocol(
      *program, cycle, 0, 1, missingGuard);
  const bool rejectsWrongTransition =
      !verifyCanonicalSyncOwnershipCycle(*program, wrongTransition);
  const bool rejectsWrongInitial =
      !verifyCanonicalSyncOwnershipCycle(*program, wrongInitial);
  const bool rejectsWrongComposite =
      hasComposite && !verifyCanonicalSyncAtomicOwnershipProtocol(
                          *program, cycle, 0, 1, wrongComposite);

  const auto tokenTraceBalances = [&](unsigned trips) {
    std::vector<int> ready(cycle.lanes.size());
    std::vector<int> release(cycle.lanes.size());
    if (trips == 0) {
      return ready == std::vector<int>({0, 0}) &&
             release == std::vector<int>({0, 0});
    }
    ++ready[cycle.initialReadyLane];
    ++release[cycle.initiallyFreeLanes.front()];
    for (unsigned iteration = 0; iteration < trips; ++iteration) {
      const CanonicalSyncOwnershipUse &use =
          cycle.paths[iteration % cycle.paths.size()].uses.front();
      if (--ready[use.lane] < 0) {
        return false;
      }
      ++release[use.lane];
      if (iteration + 1 < trips) {
        if (--release[use.producerLane] < 0) {
          return false;
        }
        ++ready[use.producerLane];
      }
    }
    for (int &token : release) {
      if (--token != 0) {
        return false;
      }
    }
    return ready == std::vector<int>({0, 0});
  };
  const bool tokenTraces =
      llvm::all_of(std::vector<unsigned>({0, 1, 2, 3, 8}), tokenTraceBalances);
  const auto hierarchicalTokenTraceBalances = [&](unsigned outerTrips,
                                                  unsigned innerTrips) {
    std::vector<int> ready(cycle.lanes.size());
    std::vector<int> release(cycle.lanes.size(), 1);
    for (unsigned outer = 0; outer < outerTrips; ++outer) {
      ++ready[cycle.initialReadyLane];
      if (--release[cycle.initialReadyLane] < 0) {
        return false;
      }
      if (innerTrips == 0) {
        if (--ready[cycle.initialReadyLane] < 0) {
          return false;
        }
        ++release[cycle.initialReadyLane];
        continue;
      }
      for (unsigned iteration = 0; iteration < innerTrips; ++iteration) {
        const CanonicalSyncOwnershipUse &use =
            cycle.paths[iteration % cycle.paths.size()].uses.front();
        if (--ready[use.lane] < 0) {
          return false;
        }
        ++release[use.lane];
        if (iteration + 1 < innerTrips) {
          if (--release[use.producerLane] < 0) {
            return false;
          }
          ++ready[use.producerLane];
        }
      }
    }
    for (int &token : release) {
      if (--token != 0) {
        return false;
      }
    }
    return ready == std::vector<int>({0, 0});
  };
  bool hierarchicalTokenTraces = true;
  for (unsigned outerTrips : {0U, 1U, 2U}) {
    for (unsigned innerTrips : {0U, 1U, 2U, 3U}) {
      hierarchicalTokenTraces &=
          hierarchicalTokenTraceBalances(outerTrips, innerTrips);
    }
  }
  const std::size_t loopEmptyActions = llvm::count_if(
      hierarchical->actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::LoopEmpty;
      });
  const bool hasNestedSummary = llvm::any_of(
      hierarchical->supplies, [](const CanonicalSyncSupplyBinding &binding) {
        return binding.proof ==
               CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary;
      });
  CanonicalSyncMechanismDescriptor wrongNestedSummary = *hierarchical;
  auto nestedSummary = llvm::find_if(
      wrongNestedSummary.supplies,
      [](const CanonicalSyncSupplyBinding &binding) {
        return binding.proof ==
               CanonicalSyncSupplyProof::VerifiedNestedRecurrenceSummary;
      });
  if (nestedSummary != wrongNestedSummary.supplies.end()) {
    ++nestedSummary->edge.distance;
  }
  const bool rejectsWrongNestedSummary =
      hasNestedSummary &&
      !verifyCanonicalSyncHierarchicalL1Protocol(*program, cycle, outerScope, 0,
                                                 1, wrongNestedSummary);
  const bool hasOwnershipClosure = llvm::any_of(
      hierarchical->supplies, [](const CanonicalSyncSupplyBinding &binding) {
        return binding.proof ==
               CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
      });
  CanonicalSyncMechanismDescriptor wrongOwnershipClosure = *hierarchical;
  auto ownershipClosure =
      llvm::find_if(wrongOwnershipClosure.supplies,
                    [](const CanonicalSyncSupplyBinding &binding) {
                      return binding.proof ==
                             CanonicalSyncSupplyProof::VerifiedOwnershipClosure;
                    });
  if (ownershipClosure != wrongOwnershipClosure.supplies.end()) {
    ++ownershipClosure->edge.distance;
  }
  const bool rejectsWrongOwnershipClosure =
      hasOwnershipClosure &&
      !verifyCanonicalSyncHierarchicalL1Protocol(*program, cycle, outerScope, 0,
                                                 1, wrongOwnershipClosure);
  module->getOperation()->removeAttr("pto.target_arch");
  FailureOr<CanonicalSyncProgram> noPrefixProgram =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("prefetch"));
  CanonicalSyncOwnershipResult noPrefixCycles =
      succeeded(noPrefixProgram)
          ? discoverCanonicalSyncOwnershipCycles(*noPrefixProgram)
          : CanonicalSyncOwnershipResult{};
  auto noPrefixAlternating =
      llvm::find_if(noPrefixCycles.cycles, [](const auto &candidate) {
        return candidate.protocol ==
               CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
      });
  const bool noPrefixAccepted =
      succeeded(noPrefixProgram) && noPrefixCycles &&
      !noPrefixProgram->getTargetCapabilities()
           .mte1ScopeExitSetCompletesPrefix &&
      noPrefixAlternating != noPrefixCycles.cycles.end() &&
      makeCanonicalSyncAtomicOwnershipProtocol(*noPrefixProgram,
                                               *noPrefixAlternating, 0, 1)
          .has_value();

  auto stable = llvm::find_if(result.cycles, [](const auto &candidate) {
    return candidate.kind == CanonicalSyncOwnershipKind::L1Tile &&
           candidate.protocol == CanonicalSyncOwnershipProtocolKind::RoundTrip;
  });
  Operation *intervening =
      stable != result.cycles.end()
          ? program->getNodeBindings()[stable->paths[0].uses[0].producers[0]]
                .operation->clone()
          : nullptr;
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program->getScopeBindings()[cycle.recurrenceScope].owner);
  bool rejectsInterveningProducer = false;
  if (intervening && loop) {
    loop->getBlock()->getOperations().insert(loop->getIterator(), intervening);
    FailureOr<CanonicalSyncProgram> interveningProgram =
        buildCanonicalSyncProgram(
            module->lookupSymbol<func::FuncOp>("prefetch"));
    if (succeeded(interveningProgram)) {
      CanonicalSyncOwnershipResult interveningCycles =
          discoverCanonicalSyncOwnershipCycles(*interveningProgram);
      rejectsInterveningProducer =
          llvm::none_of(interveningCycles.cycles, [](const auto &candidate) {
            return candidate.protocol ==
                   CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
          });
    }
    intervening->erase();
  }

  Operation *consumer =
      program->getNodeBindings()[cycle.paths[0].uses[0].consumers[0]].operation;
  Operation *duplicateConsumer = consumer->clone();
  consumer->getBlock()->getOperations().insert(
      std::next(consumer->getIterator()), duplicateConsumer);
  FailureOr<CanonicalSyncProgram> duplicateConsumerProgram =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("prefetch"));
  bool rejectsMultipleConsumers = false;
  if (succeeded(duplicateConsumerProgram)) {
    CanonicalSyncOwnershipResult duplicateConsumerCycles =
        discoverCanonicalSyncOwnershipCycles(*duplicateConsumerProgram);
    auto duplicateAlternating = llvm::find_if(
        duplicateConsumerCycles.cycles, [](const auto &candidate) {
          return candidate.protocol ==
                 CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
        });
    rejectsMultipleConsumers =
        duplicateAlternating == duplicateConsumerCycles.cycles.end() ||
        !makeCanonicalSyncAtomicOwnershipProtocol(*duplicateConsumerProgram,
                                                  *duplicateAlternating, 0, 1);
  }
  duplicateConsumer->erase();

  Operation *continuationProducer =
      program->getNodeBindings()[cycle.paths[0].uses[0].producers[0]].operation;
  Operation *continuationGuard = continuationProducer->getParentOp();
  continuationProducer->moveBefore(continuationGuard);
  FailureOr<CanonicalSyncProgram> unguardedProgram =
      buildCanonicalSyncProgram(module->lookupSymbol<func::FuncOp>("prefetch"));
  bool rejectsUnguardedContinuation = false;
  if (succeeded(unguardedProgram)) {
    CanonicalSyncOwnershipResult unguardedCycles =
        discoverCanonicalSyncOwnershipCycles(*unguardedProgram);
    rejectsUnguardedContinuation =
        llvm::none_of(unguardedCycles.cycles, [](const auto &candidate) {
          return candidate.protocol ==
                 CanonicalSyncOwnershipProtocolKind::AlternatingPrefetch;
        });
  }

  return check(guarded == 4,
               "guard every zero-trip-sensitive prime and drain") &&
         check(rejectsMissingGuard, "reject an unguarded alternating prime") &&
         check(rejectsWrongTransition,
               "reject a non-alternating lane transition") &&
         check(rejectsWrongInitial, "reject a mismatched initial owner") &&
         check(rejectsWrongComposite,
               "reject a wrong-distance composite WAW supply") &&
         check(tokenTraces,
               "balance alternating tokens for zero and positive trips") &&
         check(loopEmptyActions == 2,
               "balance both hierarchical zero-trip token domains") &&
         check(hierarchicalTokenTraces,
               "balance hierarchical tokens across outer iterations") &&
         check(hasNestedSummary,
               "summarize successor-guarded ready completion") &&
         check(rejectsWrongNestedSummary,
               "reject a malformed nested recurrence summary") &&
         check(rejectsWrongOwnershipClosure,
               "reject a malformed outer ownership closure") &&
         check(noPrefixAccepted,
               "admit direct releases without scope-prefix capability") &&
         check(rejectsInterveningProducer,
               "reject an intervening producer-pipe operation") &&
         check(succeeded(duplicateConsumerProgram),
               "build duplicate-consumer ownership graph") &&
         check(rejectsMultipleConsumers,
               "reject multiple consumers without prefix completion") &&
         check(rejectsUnguardedContinuation,
               "reject an unguarded continuation producer");
}

bool testAccumulatorOwnershipProtocol() {
  MLIRContext context;
  loadDialects(context);
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {pto.target_arch = "a2a3"} {
      func.func @accumulator(
          %output: !pto.partition_tensor_view<128x256xf32>, %limit: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %addr0 = arith.constant 0 : i64
        %left = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<left, 128x64xf16, slayout=row_major>
        %right = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<right, 64x256xf16, slayout=col_major>
        %acc = pto.alloc_tile addr = %addr0 :
          !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
                        slayout=row_major, fractal=1024>
        scf.for %i = %c0 to %limit step %c1 {
          scf.for %k = %c0 to %c2 step %c1 {
            %first = arith.cmpi eq, %k, %c0 : index
            scf.if %first {
              pto.tmatmul ins(%left, %right :
                !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
                !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
                outs(%acc : !pto.tile_buf<acc, 128x256xf32,
                     blayout=col_major, slayout=row_major, fractal=1024>)
            } else {
              pto.tmatmul.acc ins(%acc, %left, %right :
                !pto.tile_buf<acc, 128x256xf32, blayout=col_major,
                              slayout=row_major, fractal=1024>,
                !pto.tile_buf<left, 128x64xf16, slayout=row_major>,
                !pto.tile_buf<right, 64x256xf16, slayout=col_major>)
                outs(%acc : !pto.tile_buf<acc, 128x256xf32,
                     blayout=col_major, slayout=row_major, fractal=1024>)
            }
          }
          pto.tstore ins(%acc : !pto.tile_buf<acc, 128x256xf32,
                           blayout=col_major, slayout=row_major, fractal=1024>)
            outs(%output : !pto.partition_tensor_view<128x256xf32>)
        }
        return
      }
    }
  )mlir",
                                                             &context);
  if (!check(static_cast<bool>(module), "parse accumulator fixture")) {
    return false;
  }
  FailureOr<CanonicalSyncProgram> program = buildCanonicalSyncProgram(
      module->lookupSymbol<func::FuncOp>("accumulator"));
  if (!check(succeeded(program), "build accumulator graph")) {
    return false;
  }
  CanonicalSyncOwnershipResult result =
      discoverCanonicalSyncOwnershipCycles(*program);
  auto found = llvm::find_if(result.cycles, [](const auto &cycle) {
    return cycle.kind == CanonicalSyncOwnershipKind::L0Accumulator;
  });
  if (!check(result && !result.truncated && found != result.cycles.end(),
             "discover accumulator ownership")) {
    return false;
  }
  const CanonicalSyncOwnershipCycle &cycle = *found;
  const bool shape =
      cycle.protocol ==
          CanonicalSyncOwnershipProtocolKind::BoundaryGuardedRoundTrip &&
      cycle.lanes.size() == 1 && cycle.paths.size() == 1 &&
      cycle.paths.front().uses.size() == 1 &&
      cycle.paths.front().uses.front().producers.size() == 2;
  std::optional<CanonicalSyncMechanismDescriptor> atomic =
      makeCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0, 1);
  if (!check(shape, "recognize the nested MMAD accumulator boundary") ||
      !check(verifyCanonicalSyncOwnershipCycle(*program, cycle),
             "independently verify accumulator ownership") ||
      !check(atomic && verifyCanonicalSyncAtomicOwnershipProtocol(
                           *program, cycle, 0, 1, *atomic),
             "verify the atomic accumulator lifecycle")) {
    return false;
  }

  const std::size_t notFirst =
      llvm::count_if(atomic->actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::NotFirstIteration;
      });
  const std::size_t hasSuccessor =
      llvm::count_if(atomic->actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::HasSuccessor;
      });
  CanonicalSyncMechanismDescriptor missingGuard = *atomic;
  auto guarded = llvm::find_if(
      missingGuard.actions, [](const CanonicalSyncAction &action) {
        return action.guard == CanonicalSyncActionGuardKind::NotFirstIteration;
      });
  guarded->guard = CanonicalSyncActionGuardKind::None;
  guarded->guardScope.reset();
  CanonicalSyncMechanismDescriptor wrongComposite = *atomic;
  auto composite = llvm::find_if(
      wrongComposite.supplies, [](const CanonicalSyncSupplyBinding &binding) {
        return binding.proof ==
               CanonicalSyncSupplyProof::VerifiedCompositeProtocol;
      });
  const bool hasComposite = composite != wrongComposite.supplies.end();
  if (hasComposite) {
    composite->edge.distance = 0;
  }
  CanonicalSyncOwnershipCycle multipleConsumers = cycle;
  multipleConsumers.paths.front().uses.front().consumers.push_back(
      multipleConsumers.paths.front().uses.front().consumers.front());

  CanonicalSyncBuildOptions options;
  FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> problem =
      buildCanonicalSyncSingletonProblem(*program, options);
  std::optional<CanonicalSyncMechanismId> accumulatorMechanism;
  if (succeeded(problem)) {
    for (const CanonicalSyncMechanism &mechanism :
         (*problem)->getMechanisms()) {
      const bool boundaryGuarded = llvm::any_of(
          mechanism.descriptor.actions, [](const CanonicalSyncAction &action) {
            return action.guard ==
                   CanonicalSyncActionGuardKind::NotFirstIteration;
          });
      if (mechanism.descriptor.kind == CanonicalSyncMechanismKind::Protocol &&
          boundaryGuarded) {
        accumulatorMechanism = mechanism.id;
        break;
      }
    }
  }
  CanonicalSyncSelection selection;
  if (succeeded(problem)) {
    selection = selectCanonicalSyncPatterns(**problem);
  }
  bool coversManagedAccumulatorDemands = false;
  if (succeeded(problem) && accumulatorMechanism &&
      *accumulatorMechanism < (*problem)->getPatterns().size()) {
    const CanonicalSyncPattern &singleton =
        (*problem)->getPatterns()[*accumulatorMechanism];
    const CanonicalSyncOwnershipUse &use = cycle.paths.front().uses.front();
    bool sawReady = false;
    bool sawRelease = false;
    bool sawRecurrence = false;
    coversManagedAccumulatorDemands = llvm::all_of(
        llvm::enumerate(program->getGraph().getDemands()), [&](auto entry) {
          const SyncCoverDemand &demand = entry.value();
          const bool ready = demand.distance == 0 &&
                             llvm::is_contained(use.producers, demand.source) &&
                             llvm::is_contained(use.consumers, demand.target);
          const bool release =
              demand.scope == cycle.recurrenceScope && demand.distance == 1 &&
              llvm::is_contained(use.consumers, demand.source) &&
              llvm::is_contained(use.producers, demand.target);
          const bool recurrence =
              demand.scope == cycle.recurrenceScope && demand.distance == 1 &&
              llvm::is_contained(use.producers, demand.source) &&
              llvm::is_contained(use.producers, demand.target);
          if ((!ready && !release && !recurrence) ||
              demand.storageWitnesses.empty()) {
            return true;
          }
          const bool exactAccumulator = llvm::all_of(
              demand.storageWitnesses,
              [&](SyncCoverStorageWitnessId witnessId) {
                const SyncCoverStorageWitness &witness =
                    program->getGraph().getStorageWitnesses()[witnessId];
                const SyncCoverStorageAccess &source =
                    program->getGraph()
                        .getStorageAccesses()[witness.sourceAccess];
                const SyncCoverStorageAccess &target =
                    program->getGraph()
                        .getStorageAccesses()[witness.targetAccess];
                return llvm::any_of(
                    cycle.lanes.front().slots,
                    [&](const CanonicalSyncOwnershipSlot &slot) {
                      return source.domain == slot.domain &&
                             target.domain == slot.domain &&
                             slot.extent.begin <= witness.overlap.begin &&
                             witness.overlap.end <= slot.extent.end;
                    });
              });
          if (!exactAccumulator) {
            return true;
          }
          sawReady = sawReady || ready;
          sawRelease = sawRelease || release;
          sawRecurrence = sawRecurrence || recurrence;
          auto position = llvm::find((*problem)->getDemands(), entry.index());
          return position != (*problem)->getDemands().end() &&
                 ((*problem)->getBaselineCoverage().contains(
                      position - (*problem)->getDemands().begin()) ||
                  singleton.coverage.contains(
                      position - (*problem)->getDemands().begin()));
        });
    coversManagedAccumulatorDemands = coversManagedAccumulatorDemands &&
                                      sawReady && sawRelease && sawRecurrence;
  }
  (*module)->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
  FailureOr<CanonicalSyncProgram> unsupportedProgram =
      buildCanonicalSyncProgram(
          module->lookupSymbol<func::FuncOp>("accumulator"));
  bool rejectsUnsupportedTarget = false;
  bool unsupportedTargetFallsBack = false;
  if (succeeded(unsupportedProgram)) {
    CanonicalSyncOwnershipResult unsupportedCycles =
        discoverCanonicalSyncOwnershipCycles(*unsupportedProgram);
    auto unsupportedAccumulator =
        llvm::find_if(unsupportedCycles.cycles, [](const auto &candidate) {
          return candidate.kind == CanonicalSyncOwnershipKind::L0Accumulator;
        });
    if (unsupportedAccumulator != unsupportedCycles.cycles.end()) {
      std::optional<CanonicalSyncMechanismDescriptor> unsupportedAtomic =
          makeCanonicalSyncAtomicOwnershipProtocol(
              *unsupportedProgram, *unsupportedAccumulator, 0, 1);
      rejectsUnsupportedTarget = !unsupportedAtomic;
    }
    FailureOr<std::unique_ptr<CanonicalSyncPatternProblem>> unsupportedProblem =
        buildCanonicalSyncSingletonProblem(*unsupportedProgram, options);
    unsupportedTargetFallsBack =
        succeeded(unsupportedProblem) &&
        llvm::none_of(
            (*unsupportedProblem)->getMechanisms(),
            [](const CanonicalSyncMechanism &mechanism) {
              return llvm::any_of(
                  mechanism.descriptor.actions,
                  [](const CanonicalSyncAction &action) {
                    return action.guard ==
                           CanonicalSyncActionGuardKind::NotFirstIteration;
                  });
            });
  }
  const auto tokenTraceBalances = [](unsigned trips) {
    int token = 0;
    for (unsigned iteration = 0; iteration < trips; ++iteration) {
      if (iteration != 0 && --token != 0) {
        return false;
      }
      if (iteration + 1 < trips) {
        ++token;
      }
    }
    return token == 0;
  };
  return check(notFirst == 1 && hasSuccessor == 1,
               "guard the accumulator release boundaries") &&
         check(hasComposite,
               "derive exact accumulator WAW coverage from both legs") &&
         check(!verifyCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0,
                                                           1, missingGuard),
               "reject an unguarded accumulator acquire") &&
         check(!verifyCanonicalSyncAtomicOwnershipProtocol(*program, cycle, 0,
                                                           1, wrongComposite),
               "reject a wrong accumulator composite supply") &&
         check(!verifyCanonicalSyncOwnershipCycle(*program, multipleConsumers),
               "reject multiple accumulator consumers") &&
         check(tokenTraceBalances(0) && tokenTraceBalances(1) &&
                   tokenTraceBalances(2) && tokenTraceBalances(3) &&
                   tokenTraceBalances(8),
               "balance accumulator release tokens") &&
         check(succeeded(problem) && accumulatorMechanism.has_value(),
               "admit the accumulator ownership mechanism") &&
         check(selection && llvm::is_contained(selection.mechanisms,
                                               *accumulatorMechanism),
               "select the accumulator ownership mechanism") &&
         check(coversManagedAccumulatorDemands,
               "cover every managed accumulator lifecycle demand") &&
         check(rejectsUnsupportedTarget,
               "reject accumulator completion without target evidence") &&
         check(unsupportedTargetFallsBack,
               "fall back safely without accumulator target evidence");
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
      testRejectsAllExplicitSyncForms() && testStructuralLimitsFailClosed() &&
      testPeriodicBranchEvidence() &&
      testFirstIterationRecurrenceSuppression() &&
      testL0OwnershipProtocolTrustBoundary() &&
      testStableL1OwnershipProtocol() &&
      testHierarchicalL1OwnershipProtocol() &&
      testAlternatingL1OwnershipProtocol() &&
      testAccumulatorOwnershipProtocol();
  return passed ? 0 : 1;
}
