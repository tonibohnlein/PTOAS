// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect shared inputs; the separate existing-check mode runs InsertSync.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRecognition.h"
#include "PTO/Transforms/FrontierSynch/ClosedCallees.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/IR/PTO.h"
#include "SyncPhaseCopyChecks.h"
#include "SyncLogicalInsertionChecks.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
LogicalResult dumpProgramRecognition(func::FuncOp, const pto::SyncInput &,
    const pto::frontiersynch::ProgramRecognition &);
LogicalResult verifyProgramStructure(func::FuncOp, const pto::SyncInput &,
    const pto::frontiersynch::ProgramRecognition &);
void dumpRegionalArithmetic(func::FuncOp function, const pto::frontiersynch::PhaseIndex& index,
                            const pto::SyncInput& input);
void dumpArithmeticJSON(func::FuncOp function, const pto::frontiersynch::ArithmeticProgram& program);
int runSyncRegionContractChecks(func::FuncOp function, const pto::SyncInput &input);
int runPeriodicDemandChecks(llvm::StringRef path);
int runGuardedPeriodicChecks(llvm::StringRef path);
int runSyncAliasChecks(func::FuncOp function, const pto::SyncInput &input);
LogicalResult auditSyncStep0(func::FuncOp function, const pto::SyncInput &input, bool envelopes);
namespace {
LogicalResult checkVaryingBoundarySession(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  if (function->hasAttr("test.varying_boundary_rejected")) {
    for (std::size_t id = 0; id < session.result()->nodes.size(); ++id) {
      if (session.analyzeVaryingBoundary({id}).mathematical) {
        return function.emitError("unsupported carried prerequisite acquired repeating-boundary demands");
      }
    }
    llvm::outs() << "repeating-boundary-session: unsupported-carried-prerequisites retained-obligation\n";
    return success();
  }
  auto node = llvm::find_if(session.result()->nodes, [](const auto& candidate) {
    return candidate.varyingRotating && candidate.varyingRotating->result.state == RecognitionState::Applicable;
  });
  if (node == session.result()->nodes.end()) { return function.emitError("repeating-boundary form unavailable"); }
  const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
  auto child = session.analyzeVaryingBoundary({id});
  const auto owner = child.mathematical;
  const bool exact = child.status == AnalysisStatus::Ready && owner && owner->varyingBoundaryDemands &&
      owner->varyingNode == id && !session.hasWholeFunctionMinimumDemands() &&
      session.constructionCounts().logicalPreparations == 0 && session.constructionCounts().allocationExports == 0 &&
      session.constructionCounts().varyingQueryBuilds == 0 && session.constructionCounts().varyingSelectorBuilds == 0;
  if (!exact) { return function.emitError("repeating-boundary child did not retain exact demands"); }
  const auto& certificate = *owner->varyingBoundaryDemands;
  const bool complete = certificate.error.empty() && certificate.child.quotient.error.empty() &&
      certificate.startupCrossings.size() == certificate.startup &&
      certificate.suffixCrossings.size() == certificate.period && !certificate.child.fragments.empty();
  if (!complete) { return function.emitError("repeating-boundary child/crossing certificate incomplete"); }
  std::string originalIR;
  llvm::raw_string_ostream originalStream(originalIR);
  function.print(originalStream);
  AnalysisRequest queryRequest;
  queryRequest.region = id;
  queryRequest.needs.queries = true;
  auto queries = session.analyzeVaryingBoundary(queryRequest);
  const bool queryOnly = queries.status == AnalysisStatus::Ready && queries.mathematical == owner &&
      queries.regionalExports && queries.available.queries && !queries.available.selectors &&
      queries.regionalExports->storageBoundary.empty() && queries.regionalExports->lastPayloads.empty() &&
      session.constructionCounts().varyingQueryBuilds == 1 && session.constructionCounts().varyingSelectorBuilds == 0;
  if (!queryOnly) { return function.emitError("repeating-boundary query request built selectors or lost demands"); }
  std::shared_ptr<const RegionalAnalysis> selectors;
  for (unsigned capability = 0; capability < 6; ++capability) {
    AnalysisRequest request;
    request.region = id;
    request.needs.queries = capability % 3 == 0;
    request.needs.selectors = capability % 3 == 1;
    request.needs.synchronization = capability % 3 == 2;
    auto exported = session.analyzeVaryingBoundary(request);
    const bool retained = exported.mathematical == owner && exported.status == AnalysisStatus::Ready &&
        session.constructionCounts().varyingBoundaryReductions == 1 &&
        session.constructionCounts().varyingQueryBuilds == 1;
    if (!retained) { return function.emitError("repeating-boundary export discarded or recomputed demands"); }
    if (request.needs.queries && exported.regionalExports != queries.regionalExports) { return failure(); }
    if (request.needs.selectors) {
      if (selectors && selectors != exported.regionalExports) { return failure(); }
      selectors = exported.regionalExports;
      const bool completeSelectors = selectors && !selectors->lastPayloads.empty() &&
          !selectors->storageBoundary.empty();
      if (!completeSelectors) {
        return function.emitError("repeating-boundary selectors omitted native or storage boundary");
      }
    }
  }
  if (session.constructionCounts().varyingSelectorBuilds != 1 ||
      !queries.regionalExports->storageBoundary.empty() || !queries.regionalExports->lastPayloads.empty()) {
    return function.emitError("selector extension changed an immutable query result");
  }
  AnalysisRequest logicalRequest;
  logicalRequest.region = id;
  logicalRequest.needs.synchronization = true;
  auto logical = session.analyzeVaryingBoundary(logicalRequest);
  auto firstPlan = session.prepareLogical(logical);
  auto secondPlan = session.prepareLogical(logical);
  const bool distinctPlans = succeeded(firstPlan) && succeeded(secondPlan) &&
      firstPlan->get() != secondPlan->get() && !(*firstPlan)->preparation.empty() &&
      !(*secondPlan)->preparation.empty();
  if (!distinctPlans) { return function.emitError("varying preparation did not instantiate fresh detached plans"); }
  llvm::DenseSet<Operation*> firstOperations;
  for (const auto& preparation : (*firstPlan)->preparation) {
    if (preparation.code->getParent()) { return failure(); }
    preparation.code->walk([&](Operation* operation) { firstOperations.insert(operation); });
  }
  for (const auto& preparation : (*secondPlan)->preparation) {
    if (preparation.code->getParent()) { return failure(); }
    bool sharedOperation = false;
    preparation.code->walk([&](Operation* operation) { sharedOperation |= firstOperations.contains(operation); });
    if (sharedOperation) { return function.emitError("varying plans share detached operations"); }
  }
  auto printPlan = [](const PreparedLogicalPlan& plan) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const auto& preparation : plan.preparation) {
      for (auto& operation : *preparation.code) { operation.print(stream); }
    }
    return text;
  };
  const auto secondText = printPlan(**secondPlan);
  firstPlan->reset();
  std::string unchangedIR;
  llvm::raw_string_ostream unchangedStream(unchangedIR);
  function.print(unchangedStream);
  const bool freshOwnership = printPlan(**secondPlan) == secondText && unchangedIR == originalIR &&
      session.constructionCounts().varyingBoundaryReductions == 1 &&
      session.constructionCounts().varyingQueryBuilds == 1 && session.constructionCounts().varyingSelectorBuilds == 1 &&
      session.constructionCounts().allocationExports == 0;
  if (!freshOwnership) { return function.emitError("varying plan lifetime changed IR or repeated mathematical work"); }
  auto root = session.analyzeVaryingBoundary({});
  const bool childOnly = function->hasAttr("test.varying_boundary_child");
  if (childOnly) {
    if (root.mathematical || session.hasWholeFunctionMinimumDemands()) {
      return function.emitError("repeating-boundary child promoted across external payloads");
    }
  } else if (!root.mathematical || root.mathematical->varyingBoundaryDemands != owner->varyingBoundaryDemands) {
    return function.emitError("repeating-boundary whole-region owner was not reused");
  }
  const bool unchanged = session.analyzeVaryingBoundary({id}).mathematical == owner &&
      session.constructionCounts().structuralIndices == 1 &&
      session.constructionCounts().varyingBoundaryReductions == 1 &&
      session.constructionCounts().allocationExports == 0;
  if (!unchanged) { return failure(); }
  std::string exportError;
  auto privateArena = std::make_shared<RegionExpressions>();
  PhaseIndex exportIndex;
  if (failed(exportIndex.build(function, *session.input()))) { return failure(); }
  auto provider = buildVaryingQueries(function, *node->varyingRotating, exportIndex, *session.input(),
      privateArena, owner->varyingBoundaryDemands, exportError, owner->input);
  if (!provider) { return function.emitError(exportError); }
  const auto oldQuery = varyingQueryResult(*provider);
  const auto otherPolicy = policy == pto::GMAliasPolicy::MayAlias ?
      pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
  if (failed(session.initialize(otherPolicy))) { return failure(); }
  auto isolated = session.analyzeVaryingBoundary({id});
  const bool independent = isolated.mathematical && isolated.mathematical->input != owner->input &&
      isolated.mathematical->varyingBoundaryDemands != owner->varyingBoundaryDemands &&
      certificate.error.empty() && certificate.child.quotient.error.empty();
  if (!independent) { return function.emitError("repeating-boundary owner did not survive isolated context reset"); }
  auto foreign = varyingSelectedResult(*provider, *session.input(), exportError);
  const bool contextRejected = !foreign && !exportError.empty() && varyingQueryResult(*provider) == oldQuery;
  if (!contextRejected) {
    return function.emitError("varying selectors accepted or poisoned a foreign modeled input");
  }
  auto originalSelectors = varyingSelectedResult(*provider, *owner->input, exportError);
  const bool recovered = originalSelectors && exportError.empty() && !originalSelectors->lastPayloads.empty() &&
      oldQuery->lastPayloads.empty();
  if (!recovered) {
    return function.emitError("foreign context rejection poisoned original selector export");
  }
  llvm::outs() << "repeating-boundary-session: owned-child-startup-seam-suffix cached-exports original-scope\n";
  return success();
}
LogicalResult checkAllocationSession(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  auto demands = session.minimumDemands();
  if (demands.status != AnalysisStatus::Ready) { return function.emitError("exact demands unavailable"); }
  AnalysisRequest request;
  request.needs.synchronization = true;
  auto logical = session.analyze(request);
  auto left = session.prepareLogical(logical), right = session.prepareLogical(logical);
  const bool prepared = succeeded(left) && succeeded(right);
  if (logical.status != AnalysisStatus::Ready || !prepared) {
    return function.emitError("logical preparation unavailable");
  }
  if ((*left)->allocationCertificate || (*right)->allocationCertificate ||
      session.constructionCounts().allocationExports) {
    return function.emitError("logical preparation constructed physical allocation");
  }
  if ((*left)->regionalAllocation || (*right)->regionalAllocation) {
    return function.emitError("logical preparation constructed a regional allocation summary");
  }
  const auto bounded = logical.mathematical->boundedDemands;
  if (bounded && bounded->allocationRecipe) {
    return function.emitError("bounded logical preparation constructed an allocation proof");
  }
  const auto work = session.constructionCounts().mathematicalAttempts;
  const auto endpoints = (*left)->endpoints.size();
  const auto allocation = session.attachAllocation(logical, **left);
  const auto retry = session.attachAllocation(logical, **right);
  const bool sameOutcome = succeeded(allocation) == succeeded(retry);
  if (!sameOutcome ||
      (*left)->allocationCertificate != (*right)->allocationCertificate ||
      session.constructionCounts().allocationExports != 1 || (*left)->endpoints.size() != endpoints ||
      session.constructionCounts().mathematicalAttempts != work ||
      session.minimumDemands().mathematical != demands.mathematical) {
    return function.emitError("allocation retry repeated work, changed commands or lost exact demands");
  }
  llvm::outs() << "allocation-session: " << logical.mathematical->backend
               << " available=" << succeeded(allocation) << " retained-demands cached-export\n";
  return success();
}
LogicalResult checkArithmeticExportFailure(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  pto::SyncInput input(policy);
  PhaseIndex index;
  const bool ready = succeeded(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) &&
      succeeded(index.build(function, input));
  if (!ready) { return failure(); }
  scf::ForOp root;
  for (auto loop : function.getOps<scf::ForOp>()) { root = loop; break; }
  if (!root) { return failure(); }
  auto form = recognizeArithmeticProgram({function, root}, index, input, input.accesses(), {8, 8, 1, 4096});
  std::shared_ptr<const ArithmeticRegionalRelations> retained;
  std::string error;
  auto exported = analyzeArithmeticRegionRetained({function, root}, index, input,
      std::make_shared<RegionExpressions>(), retained, error, &form);
  const bool lostDemands = succeeded(exported) || !retained || !retained->analysis.exactMinimum ||
      error.find("entry origin") == std::string::npos;
  if (lostDemands) {
    return function.emitError("ordinal export failure did not retain exact arithmetic demands");
  }
  pto::SyncInput other(policy == pto::GMAliasPolicy::MayAlias ?
                       pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias);
  if (failed(other.build(function, pto::SyncInstructionView::PipeEnvelopes))) { return failure(); }
  retained.reset(); error.clear();
  exported = analyzeArithmeticRegionRetained({function, root}, index, other,
      std::make_shared<RegionExpressions>(), retained, error, &form);
  const bool mixedContext = succeeded(exported) || retained ||
      error.find("different entry context") == std::string::npos;
  if (mixedContext) {
    return function.emitError("arithmetic certificate crossed modeled input or alias context");
  }
  auto specialized = recognizeArithmeticProgram({function, root}, index, input, input.accesses(),
      {8, 8, 1, 4096}, [&](Value value) -> std::optional<int64_t> {
        return value == function.getArgument(0) ? std::optional<int64_t>(4) : std::nullopt;
      });
  retained.reset(); error.clear();
  exported = analyzeArithmeticRegionRetained({function, root}, index, input,
      std::make_shared<RegionExpressions>(), retained, error, &specialized);
  const bool unboundSpecialization = succeeded(exported) || retained ||
      error.find("different entry context") == std::string::npos;
  if (unboundSpecialization) {
    return function.emitError("specialized arithmetic certificate lost its entry bindings");
  }
  FrontierAnalysis direct(function);
  if (failed(direct.initialize(policy))) { return failure(); }
  (void)direct.analyzeSequenceFunction();
  const ArithmeticLimits profile{8, 8, 1, 4096};
  if (succeeded(direct.configureArithmeticProfiles(ArrayRef<ArithmeticLimits>(&profile, 1),
                                                  ArrayRef<ArithmeticLimits>(&profile, 1)))) {
    return function.emitError("direct sequence construction did not freeze its arithmetic context");
  }
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  auto node = llvm::find_if(session.result()->nodes, [&](const auto& n) { return n.anchor == root; });
  if (node == session.result()->nodes.end()) { return failure(); }
  const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
  auto demands = session.analyzeArithmeticRegional({id});
  if (!demands.mathematical || !demands.mathematical->arithmeticRegionalDemands) { return failure(); }
  const auto constructions = session.arithmeticRegionConstructions();
  if (session.constructionCounts().arithmeticQueryBuilds || session.constructionCounts().arithmeticSelectorBuilds) {
    return function.emitError("demands-only regional arithmetic constructed optional exports");
  }
  AnalysisNeeds needs; needs.queries = needs.selectors = true;
  auto unavailable = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  auto retry = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  if (unavailable.status != AnalysisStatus::UnmetObligation ||
      retry.mathematical != demands.mathematical || unavailable.mathematical != demands.mathematical ||
      session.arithmeticRegionConstructions() != constructions) {
    return function.emitError("arithmetic export retry lost demands or repeated the producer");
  }
  llvm::outs() << "arithmetic-export-failure: exact-demands-retained cached-retry isolated-context\n";
  return success();
}
LogicalResult checkArithmeticRequests(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  std::string originalIR;
  llvm::raw_string_ostream originalStream(originalIR);
  function.print(originalStream);
  auto input = std::make_shared<pto::SyncInput>(policy);
  PhaseIndex index;
  const bool invalidInput = failed(input->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
      failed(index.build(function, *input));
  if (invalidInput) { return failure(); }
  auto root = *function.getOps<scf::ForOp>().begin();
  auto form = recognizeArithmeticProgram({function, root}, index, *input, input->accesses(), {8, 8, 1, 4096});
  std::string error;
  auto demands = analyzeArithmeticRegionDemands({function, root}, index, *input, &form, error);
  const bool impureDemands = !demands || !demands->analysis.exactMinimum || !demands->parameters.empty() ||
      !demands->occurrences.empty() || !demands->selectors.boundaries.empty() || !demands->enclosing.empty();
  if (impureDemands) {
    return function.emitError("demands-only arithmetic constructed optional export state");
  }
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  auto node = llvm::find_if(session.result()->nodes, [&](const auto& n) { return n.anchor == root; });
  if (node == session.result()->nodes.end()) { return failure(); }
  const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
  auto retained = session.analyzeArithmeticRegional({id});
  if (!retained.mathematical || !retained.mathematical->arithmeticRegionalDemands ||
      session.constructionCounts().arithmeticQueryBuilds || session.constructionCounts().arithmeticSelectorBuilds) {
    return function.emitError("session did not retain pure regional arithmetic mathematics");
  }
  const auto reductions = session.arithmeticRegionConstructions();
  AnalysisNeeds needs; needs.queries = true;
  auto query = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  needs.selectors = true;
  auto storage = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  auto retry = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  if (query.status != AnalysisStatus::Ready || storage.status != AnalysisStatus::Ready ||
      retry.regionalExports != storage.regionalExports || retry.mathematical != retained.mathematical ||
      query.mathematical != retained.mathematical || storage.mathematical != retained.mathematical ||
      session.arithmeticRegionConstructions() != reductions ||
      session.constructionCounts().arithmeticQueryBuilds != 1 ||
      session.constructionCounts().arithmeticSelectorBuilds != 1) {
    return function.emitError("session arithmetic export retry reconstructed mathematics or exports");
  }
  needs.evaluation = AnalysisEvaluation::Stateful;
  auto stateful = session.analyzeArithmeticRegional({id, AnalysisMode::MinimumExact, needs});
  if (stateful.status != AnalysisStatus::UnmetObligation || stateful.mathematical != retained.mathematical) {
    return function.emitError("uniform arithmetic exports incorrectly satisfied stateful evaluation");
  }
  auto queryArena = query.regionalExports->expressions;
  auto zero = queryArena->constant(0);
  RegionalEvent cachedEvent{0, zero, PeriodicEventKind::Start, {zero}};
  auto cachedPresence = query.regionalExports->presence(cachedEvent);
  session.invalidate();
  const bool expiredSessionQuery = !cachedPresence || query.regionalExports->presence(cachedEvent) != cachedPresence;
  if (expiredSessionQuery) { return function.emitError("arithmetic queries lost their owner after session reset"); }
  auto arena = std::make_shared<RegionExpressions>();
  auto queries = exportArithmeticRegion(*demands, arena, false, error, input);
  const bool invalidQueries =
      failed(queries) || !queries->capabilities.exactQueries || queries->capabilities.exactSelectors ||
      queries->storageSelectors || queries->prepare || queries->capabilities.endpointRecipes ||
      !queries->arithmeticRelations->selectors.boundaries.empty();
  if (invalidQueries) {
    return function.emitError("query-only arithmetic constructed storage or endpoint exports");
  }
  RegionalEvent event{0, arena->constant(0), PeriodicEventKind::Start, {arena->constant(0)}};
  auto presence = queries->presence(event);
  if (!presence) { return failure(); }
  // A deliberately unsupported access adapter must not obstruct occurrence
  // queries or poison already published IDs. No source IR is modified.
  auto unsupported = *demands;
  bool changed = false;
  for (auto& relation : unsupported.program.primitives.relations) {
    if (relation.kind == PrimitiveKind::Reads || relation.kind == PrimitiveKind::Writes) {
      relation.storageSpace.reset(); changed = true;
    }
  }
  if (!changed) { return failure(); }
  const auto nodes = arena->size();
  error.clear();
  auto failedSelectors = exportArithmeticRegion(unsupported, arena, true, error, input);
  const bool damagedQueries = succeeded(failedSelectors) || error.empty() || arena->size() != nodes ||
      !arena->constructionError().empty() || queries->presence(event) != presence;
  if (damagedQueries) { return function.emitError("failed arithmetic selectors corrupted retained queries"); }
  error.clear();
  auto weaker = exportArithmeticRegion(unsupported, arena, false, error, input);
  auto selected = exportArithmeticRegion(*demands, arena, true, error, input);
  const bool inconsistentExports =
      failed(weaker) || failed(selected) || selected->expressions != queries->expressions ||
      !selected->capabilities.exactSelectors || queries->capabilities.exactSelectors ||
      selected->presence(event) != presence;
  if (inconsistentExports) {
    return function.emitError("arithmetic stronger and weaker exports disagree in the common arena");
  }
  auto first = prepareArithmeticRegion(*demands, arena, {}, error, input);
  auto second = prepareArithmeticRegion(*demands, arena, {}, error, input);
  const bool preparation = succeeded(first) && succeeded(second) && first->get() != second->get() &&
      !(*first)->preparation.empty() && !(*second)->preparation.empty();
  if (!preparation) { return function.emitError("arithmetic preparation did not create distinct owned plans"); }
  llvm::DenseSet<Operation*> firstOperations;
  for (const auto& fragment : (*first)->preparation) {
    if (fragment.code->getParent()) { return failure(); }
    fragment.code->walk([&](Operation* operation) { firstOperations.insert(operation); });
  }
  for (const auto& fragment : (*second)->preparation) {
    if (fragment.code->getParent()) { return failure(); }
    bool shared = false;
    fragment.code->walk([&](Operation* operation) { shared |= firstOperations.contains(operation); });
    if (shared) { return function.emitError("arithmetic plans share detached operations"); }
  }
  auto printPlan = [](const PreparedLogicalPlan& plan) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const auto& fragment : plan.preparation) {
      for (auto& operation : *fragment.code) { operation.print(stream); }
    }
    return text;
  };
  const auto secondText = printPlan(**second);
  first->reset();
  auto invalidContext = prepareArithmeticRegion(*demands, arena, {root}, error, input);
  std::string unchangedIR;
  llvm::raw_string_ostream unchangedStream(unchangedIR);
  function.print(unchangedStream);
  const bool changedPreparation =
      succeeded(invalidContext) || printPlan(**second) != secondText || unchangedIR != originalIR;
  if (changedPreparation) {
    return function.emitError("arithmetic failed preparation or fragment lifetime changed original IR");
  }
  const auto beforeAllocation = printPlan(**second);
  prepareAllocationSupport(**second);
  const bool invalidAllocation = !(*second)->regionalAllocation || printPlan(**second) != beforeAllocation ||
      !arena->constructionError().empty() || queries->presence(event) != presence;
  if (invalidAllocation) {
    return function.emitError("successful arithmetic allocation damaged logical fragments or queries");
  }
  auto noAllocation = *demands;
  noAllocation.analysis.requiredOrder.clear(); // Deliberately unavailable handoff-reuse certificate.
  auto unavailablePlan = prepareArithmeticRegion(noAllocation, arena, {}, error, input);
  if (failed(unavailablePlan)) { return function.emitError("allocation refusal prevented logical preparation"); }
  const auto unavailableText = printPlan(**unavailablePlan);
  const auto allocationNodes = arena->size();
  prepareAllocationSupport(**unavailablePlan);
  prepareAllocationSupport(**unavailablePlan); // The one-shot attempt does not run twice.
  const bool damagedAllocation =
      (*unavailablePlan)->regionalAllocation || printPlan(**unavailablePlan) != unavailableText ||
      arena->size() != allocationNodes || !arena->constructionError().empty() ||
      queries->presence(event) != presence;
  if (damagedAllocation) {
    return function.emitError("arithmetic allocation refusal corrupted logical fragments or queries");
  }
  unavailablePlan->reset();
  second->reset(); selected = failure(); weaker = failure();
  std::weak_ptr<const pto::SyncInput> lifetime = input;
  input.reset(); demands.reset();
  const bool expiredOwner = lifetime.expired() || queries->presence(event) != presence;
  if (expiredOwner) {
    return function.emitError("arithmetic query lost its modeled input owner");
  }
  llvm::outs() << "arithmetic-requests: demands-only query-only isolated-selector-failure fresh-plans retained-owner\n";
  return success();
}
LogicalResult checkFiniteExpansionSession(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  bool checked = false, bodyChecked = false, runChecked = false;
  std::shared_ptr<const MathematicalResult> retainedOwner;
  std::shared_ptr<const RegionalAnalysis> retainedQueries, retainedStorage;
  for (auto [id, node] : llvm::enumerate(session.result()->nodes)) {
    if (id == 0) { continue; }
    auto first = session.analyzeFiniteExpansion({id});
    if (!first.mathematical) { continue; }
    SmallVector<Operation*> expected;
    if (node.kind == StructureKind::ExplicitRun) { expected = node.operations; }
    else if (node.kind == StructureKind::Sequence && node.region) {
      for (auto& operation : node.region->front()) {
        if (!operation.hasTrait<OpTrait::IsTerminator>()) { expected.push_back(&operation); }
      }
    } else { expected.push_back(node.anchor); }
    const auto& expanded = first.mathematical->finiteGuardedDemands->expandedProgram;
    if (!expanded || expanded->context.roots != expected) {
      return function.emitError("expanded adapter changed regional root list");
    }
    for (const auto& site : expanded->sites) {
      const bool enclosed = llvm::any_of(expected, [&](Operation* root) {
        return root->isAncestor(site.phase->elementOp);
      });
      if (!enclosed) { return function.emitError("expanded adapter included an enclosing visit or sibling"); }
      for (auto coordinate : site.fixedCoordinates) {
        const bool internal = llvm::any_of(expected, [&](Operation* root) {
          return root->isAncestor(coordinate.loop.getOperation());
        });
        if (!internal) { return function.emitError("expanded adapter enumerated a parent loop"); }
      }
    }
    checked = true;
    bodyChecked |= node.kind == StructureKind::Sequence;
    runChecked |= node.kind == StructureKind::ExplicitRun && node.payloadCount > 0;
    const bool invalidExports = session.hasWholeFunctionMinimumDemands() ||
        first.available.queries || first.available.selectors;
    if (invalidExports) {
      return function.emitError("expanded child advertised unsupported coverage or exports");
    }
    const auto work = session.constructionCounts().mathematicalAttempts;
    std::shared_ptr<const RegionalAnalysis> queries;
    for (unsigned capability = 0; capability < 3; ++capability) {
      AnalysisRequest stronger{id};
      stronger.needs.queries = true;
      stronger.needs.selectors = capability == 1;
      stronger.needs.synchronization = capability == 2;
      const auto selectorsBefore = session.constructionCounts().expandedSelectorBuilds;
      const auto checksBefore = session.constructionCounts().expandedSelectorChecks;
      auto outcome = session.analyzeFiniteExpansion(stronger);
      const auto selectorsAfter = session.constructionCounts().expandedSelectorBuilds;
      const auto checksAfter = session.constructionCounts().expandedSelectorChecks;
      auto retry = session.analyzeFiniteExpansion(stronger);
      const auto expectedSelectorCount = selectorsBefore + (capability == 1 ? 1 : 0);
      if (selectorsAfter != expectedSelectorCount ||
          session.constructionCounts().expandedSelectorBuilds != selectorsAfter ||
          session.constructionCounts().expandedSelectorChecks != checksAfter ||
          (capability != 1 && checksAfter != checksBefore)) {
        return function.emitError("expanded selector request repeated construction or polluted weaker requests");
      }
      const auto expectedStatus = capability == 0 || (capability == 1 && outcome.available.selectors) ?
          AnalysisStatus::Ready : AnalysisStatus::UnmetObligation;
      const bool retained = outcome.status == expectedStatus && outcome.mathematical == first.mathematical &&
          retry.mathematical == first.mathematical && outcome.available.queries &&
          outcome.regionalExports && outcome.regionalExports == retry.regionalExports &&
          session.constructionCounts().mathematicalAttempts == work;
      if (!retained) {
        return function.emitError("finite expansion export retry discarded or reconstructed demands or queries");
      }
      if (capability != 1 && queries && queries != outcome.regionalExports) { return failure(); }
      if (capability != 1) { queries = outcome.regionalExports; }
      if (capability == 1 && outcome.available.selectors) {
        const auto& selected = *outcome.regionalExports;
        const bool complete = selected.capabilities.completeStorageModel && selected.capabilities.exactSelectors &&
            selected.storageSelectors && selected.symbolicStorage && !selected.prepare &&
            !selected.capabilities.endpointRecipes;
        retainedStorage = outcome.regionalExports;
        if (!complete) {
          return function.emitError("expanded storage export has incomplete support or unexpected recipes");
        }
      }
      const bool queryOnly = !queries->capabilities.exactSelectors && !queries->capabilities.endpointRecipes &&
          !queries->capabilities.completeStorageModel && queries->storageBoundary.empty() && !queries->prepare;
      if (!queryOnly) {
        return function.emitError("expanded queries advertised unmapped selector or endpoint exports");
      }
    }
    for (auto [siteID, site] : llvm::enumerate(expanded->sites)) {
      const auto& coordinates = queries->anchors[siteID].coordinates;
      const bool sameCoordinateCount = coordinates.size() == site.fixedCoordinates.size();
      if (!sameCoordinateCount) { return failure(); }
      for (auto [position, coordinate] : llvm::enumerate(coordinates)) {
        auto expectedCoordinate = site.fixedCoordinates[position];
        if (coordinate.loop != expectedCoordinate.loop || coordinate.induction != expectedCoordinate.induction) {
          return function.emitError("expanded query lost original fixed loop coordinates");
        }
      }
      const auto zero = queries->expressions->constant(0);
      RegionalEvent external{static_cast<uint32_t>(siteID), zero, PeriodicEventKind::Start, {zero}};
      const bool acceptedExternal = queries->presence(external) || queries->reachability(external, external) ||
          queries->referenceBefore(external, external);
      if (acceptedExternal) {
        return function.emitError("expanded query accepted an unbound external visit");
      }
    }
    if (!expanded->sites.empty()) {
      retainedOwner = first.mathematical;
      retainedQueries = queries;
      auto& e = *queries->expressions;
      const auto zero = e.constant(0);
      RegionalEvent missing{0, e.constant(1), PeriodicEventKind::Start};
      RegionalEvent target{static_cast<uint32_t>(expanded->sites.size() - 1), zero, PeriodicEventKind::Completion};
      auto absent = queries->presence(missing);
      auto unreachable = queries->reachability(missing, target);
      auto unordered = queries->referenceBefore(missing, target);
      const bool missingAbsent = absent && unreachable && unordered &&
          e.constantValue(*absent) == 0 && e.constantValue(*unreachable) == 0 && e.constantValue(*unordered) == 0;
      if (!missingAbsent) { return function.emitError("expanded nonzero ordinal aliases an actual occurrence"); }
      for (auto malformed : {RegionalEvent{0, RegionExpressions::invalid, PeriodicEventKind::Start},
                            RegionalEvent{0, zero, static_cast<PeriodicEventKind>(255)}}) {
        const bool accepted = queries->presence(malformed) || queries->reachability(malformed, target) ||
            queries->referenceBefore(malformed, target);
        if (accepted) { return function.emitError("expanded query accepted a malformed event identity"); }
      }
    }
  }
  if (!checked || !bodyChecked || !runChecked) {
    return function.emitError("finite expansion fixture requires successful body and run requests");
  }
  if (!retainedOwner || !retainedQueries) { return failure(); }
  RegionalEvent saved{0, retainedQueries->expressions->constant(0), PeriodicEventKind::Start};
  auto savedPresence = retainedQueries->presence(saved);
  auto savedReachability = retainedQueries->reachability(saved, saved);
  const auto otherPolicy = policy == pto::GMAliasPolicy::MayAlias ?
      pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
  if (failed(session.initialize(otherPolicy))) { return failure(); }
  auto foreign = expandedFiniteRegionalQueries(*retainedOwner->finiteGuardedDemands, session.sharedInput());
  if (foreign.capabilities.exactQueries) { return function.emitError("expanded query accepted a foreign input owner"); }
  std::string foreignError;
  auto foreignStorage = expandedFiniteRegionalSelectors(*retainedOwner->finiteGuardedDemands,
      *retainedQueries, foreignError, session.sharedInput());
  if (succeeded(foreignStorage)) { return function.emitError("expanded selectors accepted a foreign input owner"); }
  if (!retainedStorage) { return function.emitError("finite expansion fixture produced no storage selectors"); }
  RegionalByteAddress savedByte{pto::AddressSpace::LEFT, {}, retainedStorage->expressions->constant(512)};
  auto savedSelectors = retainedStorage->storageSelectors(savedByte);
  retainedOwner.reset();
  session.invalidate();
  auto survivingSelectors = retainedStorage->storageSelectors(savedByte);
  const bool sameSelectors = savedSelectors && survivingSelectors &&
      savedSelectors->firstWriters.size() == survivingSelectors->firstWriters.size() &&
      savedSelectors->lastWriters.size() == survivingSelectors->lastWriters.size();
  if (!sameSelectors) { return function.emitError("expanded storage owner did not survive session reset"); }
  const bool surviving = savedPresence && savedReachability && retainedQueries->presence(saved) == savedPresence &&
      retainedQueries->reachability(saved, saved) == savedReachability;
  if (!surviving) { return function.emitError("expanded query handle did not survive isolated session reset"); }
  llvm::outs() << "finite-expansion-session: original-scope retained-demands cached-exports no-child-promotion\n";
  return success();
}
LogicalResult checkRegionalSession(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  unsigned checked = 0;
  for (auto [id, node] : llvm::enumerate(session.result()->nodes)) {
    if (node.kind != StructureKind::Loop) { continue; }
    auto demands = session.minimumDemands(id);
    if (demands.status != AnalysisStatus::Ready || !demands.mathematical ||
        session.constructionCounts().logicalPreparations || session.constructionCounts().allocationExports) {
      return function.emitError("regional demands-only request constructed code or lost demands");
    }
    AnalysisNeeds interfaces;
    interfaces.queries = interfaces.selectors = true;
    auto first = session.minimumDemands(id, interfaces);
    if (first.status != AnalysisStatus::Ready || !first.mathematical || first.mathematical->region != id ||
        session.hasWholeFunctionMinimumDemands()) {
      return function.emitError("regional success lost its identity or established whole-function evidence");
    }
    const auto work = session.constructionCounts().mathematicalAttempts;
    const auto arithmetic = session.arithmeticRegionConstructions();
    auto retry = session.minimumDemands(id, interfaces);
    if (retry.mathematical != first.mathematical || session.constructionCounts().mathematicalAttempts != work ||
        session.arithmeticRegionConstructions() != arithmetic) {
      return function.emitError("regional request repeated mathematical construction");
    }
    auto retained = session.minimumDemands(id);
    if (retained.mathematical != demands.mathematical) {
      return function.emitError("stronger regional request replaced an earlier exact result");
    }
    ++checked;
  }
  const auto rotating = session.constructionCounts().rotatingReductions;
  const auto guarded = session.constructionCounts().guardedRotatingReductions;
  auto root = session.minimumDemands();
  if (session.constructionCounts().rotatingReductions != rotating ||
      session.constructionCounts().guardedRotatingReductions != guarded) {
    return function.emitError("composition repeated a cached loop reduction");
  }
  if (!checked || root.status != AnalysisStatus::Ready || !root.mathematical || root.mathematical->region ||
      !session.hasWholeFunctionMinimumDemands()) {
    return function.emitError("whole-function request did not establish independent exact evidence");
  }
  AnalysisRequest logical;
  logical.needs.queries = logical.needs.selectors = logical.needs.synchronization = true;
  auto complete = session.analyze(logical);
  if (complete.status != AnalysisStatus::Ready) {
    return function.emitError("whole-region logical export did not complete");
  }
  const auto preparations = session.constructionCounts().logicalPreparations;
  for (auto [id, node] : llvm::enumerate(session.result()->nodes)) {
    const bool arm = function->hasAttr("test.cached_arm_exports") &&
                     node.kind == StructureKind::Sequence && node.guard && node.payloadCount;
    const bool selected = (node.kind == StructureKind::Loop || arm) && node.loops.empty();
    if (!selected) { continue; }
    logical.region = id;
    auto child = session.analyze(logical);
    if (child.status != AnalysisStatus::Ready ||
        session.constructionCounts().logicalPreparations != preparations) {
      return function.emitError("whole-region preparation bypassed the child export cache");
    }
  }
  if (auto expected = function->getAttrOfType<IntegerAttr>("test.expected_arithmetic_regions")) {
    const bool expectedCount = session.arithmeticRegionConstructions() == expected.getValue().getLimitedValue();
    if (!expectedCount) {
      return function.emitError("unexpected arithmetic regional construction count: ")
          << session.arithmeticRegionConstructions();
    }
  }
  return success();
}
LogicalResult checkRetainedDemands(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  auto first = session.minimumDemands();
  if (first.status != AnalysisStatus::Ready || !first.mathematical || !session.hasWholeFunctionMinimumDemands()) {
    return function.emitError("uniform exact demands unavailable");
  }
  if (auto expected = function->getAttrOfType<StringAttr>("test.expected_demand_backend")) {
    if (first.mathematical->backend != expected.getValue()) {
      return function.emitError("unexpected exact demand backend: ") << first.mathematical->backend;
    }
  }
  const auto& counts = session.constructionCounts();
  if (counts.logicalPreparations || counts.allocationExports) {
    return function.emitError("mathematical request constructed logical or allocation exports");
  }
  for (const auto& node : session.result()->nodes) {
    if (node.logicalEndpoints || node.periodicAllocation) {
      return function.emitError("numerical analysis eagerly constructed exports");
    }
  }
  AnalysisRequest logical;
  logical.needs.synchronization = true;
  auto unsupported = session.analyze(logical);
  const auto work = counts.mathematicalAttempts;
  const auto estimates = session.costRecords().size();
  auto retry = session.analyze(logical);
  const bool stableEstimates = session.costRecords().size() == estimates;
  if (!stableEstimates) {
    return function.emitError("repeated request rebuilt cost selection");
  }
  const auto costs = session.costRecords();
  for (std::size_t i = 1; i < costs.size(); ++i) {
    if (costs[i].request == costs[i - 1].request &&
        estimatedCostLess(costs[i].estimate, costs[i - 1].estimate)) {
      return function.emitError("eligible arithmetic methods were not ordered by estimated work");
    }
  }
  auto again = session.minimumDemands();
  if (unsupported.status != AnalysisStatus::UnmetObligation || !unsupported.mathematical ||
      unsupported.mathematical != first.mathematical || retry.mathematical != first.mathematical ||
      again.mathematical != first.mathematical || counts.mathematicalAttempts != work) {
    return function.emitError("endpoint failure lost exact demands or repeated mathematical construction");
  }
  logical.mode = AnalysisMode::Fallback;
  auto fallback = session.analyze(logical);
  if (fallback.status != AnalysisStatus::UnmetObligation || fallback.mathematical != first.mathematical) {
    return function.emitError("fallback replaced the retained exact result");
  }
  return success();
}
LogicalResult checkAnalysisSession(func::FuncOp function, pto::GMAliasPolicy policy) {
  using namespace pto::frontiersynch;
  FrontierAnalysis session(function);
  if (failed(session.initialize(policy))) { return failure(); }
  auto first = session.minimumDemands();
  auto again = session.minimumDemands();
  const auto& counts = session.constructionCounts();
  if (first.status != AnalysisStatus::Ready || !first.mathematical ||
      first.mathematical != again.mathematical || counts.structuralIndices != 1 ||
      counts.explicitReductions != 1 || counts.logicalPreparations || counts.allocationExports) {
    return function.emitError("demand-only session rebuilt analysis or constructed exports");
  }
  AnalysisNeeds stateful;
  stateful.evaluation = AnalysisEvaluation::Stateful;
  auto unsupported = session.minimumDemands(0, stateful);
  if (unsupported.status != AnalysisStatus::UnmetObligation ||
      unsupported.mathematical != first.mathematical) {
    return function.emitError("unavailable evaluation discarded exact demands");
  }
  AnalysisRequest logical;
  logical.needs.synchronization = true;
  auto ready = session.analyze(logical);
  const auto preparations = counts.logicalPreparations;
  auto retry = session.analyze(logical);
  if (ready.status != AnalysisStatus::Ready || retry.mathematical != first.mathematical ||
      counts.explicitReductions != 1 || counts.logicalPreparations != preparations || counts.allocationExports) {
    return function.emitError("logical retry repeated mathematical or export construction");
  }
  auto left = session.prepareLogical(ready);
  if (counts.logicalPreparations != preparations) {
    return function.emitError("accepted synchronization fragment was prepared twice");
  }
  auto right = session.prepareLogical(ready);
  const bool bothPrepared = succeeded(left) && succeeded(right);
  if (!bothPrepared) { return failure(); }
  const bool sharedPlan = left->get() == right->get();
  if (sharedPlan || (*left)->allocationCertificate || (*right)->allocationCertificate) {
    return function.emitError("detached preparation ownership or allocation separation failed");
  }
  PreparedLogicalPlan unrelated(0);
  const auto unrelatedAllocation = session.attachAllocation(ready, unrelated);
  const auto allocation = session.attachAllocation(ready, **left);
  const bool allocationChecked = failed(unrelatedAllocation) && succeeded(allocation);
  if (!allocationChecked) {
    return function.emitError("allocation did not validate selected-order ownership");
  }
  auto otherPolicy = policy == pto::GMAliasPolicy::MayAlias ?
      pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
  const auto changedPolicy = session.initialize(otherPolicy);
  auto incompatible = session.prepareLogical(first);
  const bool policyIsolated = succeeded(changedPolicy) && failed(incompatible);
  if (!policyIsolated) {
    return function.emitError("alias contexts reused incompatible mathematical ownership");
  }
  session.invalidate();
  auto stale = session.prepareLogical(first);
  const auto reinitialized = session.initialize(policy);
  const bool invalidated = failed(stale) && succeeded(reinitialized);
  if (!invalidated) { return failure(); }
  auto renewed = session.minimumDemands();
  if (renewed.status != AnalysisStatus::Ready || renewed.mathematical == first.mathematical) { return failure(); }
  return success();
}
LogicalResult dumpStorageEffects(func::FuncOp function, const pto::SyncInput &input) {
  const auto& storage = input.accesses();
  llvm::outs() << "storage " << function.getSymName() << ": cells=" << storage.cells().size()
               << " all-materialized=" << storage.allAccessesMaterialized() << "\n";
  for (auto [id, cell] : llvm::enumerate(storage.cells())) {
    llvm::outs() << "  cell " << id << " space=" << static_cast<unsigned>(cell.space)
                 << " [" << cell.begin << "," << cell.end << ")\n";
  }
  for (auto [id, effect] : llvm::enumerate(storage.effects())) {
    StringRef representation = effect.rangesMaterialized ? "intervals" :
        (!effect.regions.empty() ? "symbolic" : "unresolved");
    llvm::outs() << "  effect " << id << " " << effect.phase->opName.getStringRef()
                 << (effect.mode == pto::SyncAccessMode::Read ? " read " : " write ")
                 << representation << " cells=";
    llvm::interleaveComma(effect.cells, llvm::outs());
    llvm::outs() << "\n";
    auto dumpRegion = [&](const pto::SyncAccessRegion& region, StringRef label) {
      llvm::outs() << "    " << label << " base=" << (region.base ? "pointer" : "absolute")
                   << " bytes=" << region.elementBytes << " map=";
      AffineMap::get(region.extents.size(), region.symbols.size(), region.byteOffset).print(llvm::outs());
      llvm::outs() << " extents=";
      llvm::interleaveComma(region.extents, llvm::outs());
      llvm::outs() << " loops=" << region.iterations.size() << "\n";
    };
    if (effect.descriptorRegion) {
      dumpRegion(*effect.descriptorRegion, "descriptor");
    }
    if (effect.region) {
      dumpRegion(*effect.region, "access");
    }
    if (effect.selection) {
      llvm::outs() << "    selected-addresses=";
      llvm::interleaveComma(effect.selection->addresses, llvm::outs());
      llvm::outs() << " dynamic=" << !effect.selection->selector.getDefiningOp<arith::ConstantOp>() << "\n";
    }
  }
  if (auto queries = function->getAttrOfType<DenseI64ArrayAttr>("test.overlap")) {
    if (queries.size() % 2 != 0) {
      return failure();
    }
    for (int64_t i = 0; i < queries.size(); i += 2) {
      if (queries[i] < 0 || queries[i + 1] < 0) {
        return failure();
      }
      llvm::outs() << "  overlap " << queries[i] << "," << queries[i + 1] << "="
                   << storage.mayOverlap(static_cast<std::size_t>(queries[i]),
                                         static_cast<std::size_t>(queries[i + 1])) << "\n";
    }
  }
  if (auto queries = function->getAttrOfType<DenseI64ArrayAttr>("test.conflict")) {
    if (queries.size() % 2 != 0) {
      return failure();
    }
    for (int64_t i = 0; i < queries.size(); i += 2) {
      if (queries[i] < 0 || queries[i + 1] < 0) {
        return failure();
      }
      llvm::outs() << "  conflict " << queries[i] << "," << queries[i + 1] << "="
                   << storage.mayConflict(queries[i], queries[i + 1]) << "\n";
    }
  }
  auto independenceQueries = function->getAttrOfType<DenseI64ArrayAttr>("test.independent_gm");
  if (independenceQueries) {
    for (auto id : independenceQueries.asArrayRef()) {
      if (id < 0) {
        return failure();
      }
      const bool independent = storage.independentOfOtherPhases(static_cast<std::size_t>(id));
      if (storage.independentOfOtherPhases(static_cast<std::size_t>(id)) != independent) {
        return failure();
      }
      llvm::outs() << "  independent-gm " << id << "=" << independent << "\n";
    }
  }
  // Rebuilding must neither accumulate cells nor retain prior phase mappings.
  auto cellCount = storage.cells().size(), effectCount = storage.effects().size();
  pto::SyncStorageEffects rebuilt;
  if (failed(rebuilt.build(input)) || rebuilt.cells().size() != cellCount || rebuilt.effects().size() != effectCount) {
    return failure();
  }
  if (independenceQueries) {
    for (auto id : independenceQueries.asArrayRef()) {
      if (rebuilt.independentOfOtherPhases(static_cast<std::size_t>(id)) !=
          storage.independentOfOtherPhases(static_cast<std::size_t>(id))) {
        return failure();
      }
    }
    if (failed(rebuilt.build(input))) {
      return failure();
    }
    for (auto id : independenceQueries.asArrayRef()) {
      if (rebuilt.independentOfOtherPhases(static_cast<std::size_t>(id)) !=
          storage.independentOfOtherPhases(static_cast<std::size_t>(id))) {
        return failure();
      }
    }
  }
  return success();
}
void dumpRecognition(StringRef label, const pto::frontiersynch::RecognitionResult &result) {
  namespace fs = pto::frontiersynch;
  llvm::outs() << "recognize " << label << ": " << fs::recognitionName(result.state)
               << " backend=not-run\n";
  for (const auto &diagnostic : result.diagnostics) {
    llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue);
    if (diagnostic.anchor) {
      llvm::outs() << " at " << diagnostic.anchor->getLoc();
    }
    llvm::outs() << "\n";
  }
  DenseMap<Value, std::size_t> families;
  for (const auto &access : result.accesses) {
    auto family = families.try_emplace(access.family, families.size()).first->second;
    llvm::outs() << "  rotation family=" << family << " slots=" << access.slots
                 << " stride=" << access.stride << " offset=";
    if (access.parameterOffset) {
      llvm::outs() << "(" << access.parameterOffset << ") parameters=" << access.parameters.size();
    } else {
      llvm::outs() << access.offset;
    }
    llvm::outs() << " refresh=" << access.refresh << " atom=";
    if (access.atom) {
      llvm::outs() << "[" << access.atom->first << "," << access.atom->second << ")";
    } else {
      llvm::outs() << "unknown";
    }
    if (access.guard) {
      llvm::outs() << " guard=" << *access.guard;
    }
    llvm::outs() << " effects=" << access.effects.size() << " reads=" << access.reads
                 << " writes=" << access.writes << "\n";
  }
}
void dumpGuarded(StringRef label, const pto::frontiersynch::GuardedRecognition &result) {
  dumpRecognition(label, result.result);
  llvm::outs() << "  guarded phases=" << result.phases.size() << " guards=" << result.guards.size()
               << " entry-guards=" << (result.entryGuardsAvailable ? "available" : "late")
               << " entry-expressions=" << result.entryExpressions.size() << "\n";
  for (auto [id, guard] : llvm::enumerate(result.guards)) {
    llvm::outs() << "  guard " << id << " parent=";
    if (guard.parent) {
      llvm::outs() << *guard.parent;
    } else {
      llvm::outs() << "true";
    }
    llvm::outs() << " arm=" << (guard.takeThen ? "then" : "else") << "\n";
  }
  for (const auto &phase : result.phases) {
    llvm::outs() << "  guarded-phase " << phase.phase->opName.getStringRef() << " guard=";
    if (phase.guard) {
      llvm::outs() << *phase.guard;
    } else {
      llvm::outs() << "true";
    }
    llvm::outs() << "\n";
  }
}
LogicalResult recognize(func::FuncOp function, const pto::SyncInput &input, bool arithmeticOnly,
                        const pto::frontiersynch::ProgramRecognition *cached = nullptr) {
  pto::frontiersynch::PhaseIndex index;
  const auto& effects = input.accesses();
  if (failed(index.build(function, input))) {
    return failure();
  }
  llvm::outs() << "recognition " << function.getSymName() << "\n";
  namespace fs = pto::frontiersynch;
  // Fixed test-client class: up to 8 pipes, 8 coordinates, period 2, coefficient 8.
  // Production callers choose their class limits, never observed input maxima.
  std::optional<fs::ArithmeticProgram> direct;
  if (!cached || !cached->arithmetic) {
    direct = fs::recognizeArithmeticProgram(function, index, input, effects, {8, 8, 2, 8});
  }
  const auto& arithmetic = direct ? *direct : *cached->arithmetic;
  auto status = arithmetic.extraction.state == fs::RecognitionState::Applicable ?
                arithmetic.recognition.state : arithmetic.extraction.state;
  llvm::outs() << "recognize arithmetic: " << fs::recognitionName(status) << " backend=not-run\n";
  for (const auto &diagnostic : arithmetic.extraction.diagnostics) {
    llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue) << "\n";
  }
  for (const auto &diagnostic : fs::summarizeArithmeticDiagnostics(arithmetic.recognition.diagnostics)) {
    llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue)
                 << " count=" << diagnostic.count << " first-relation=" << diagnostic.relation
                 << " first-piece=" << diagnostic.piece << "\n";
  }
  llvm::outs() << "  arithmetic sites=" << arithmetic.sites.size()
               << " parameters=" << arithmetic.parameters.size()
               << " primitives=" << arithmetic.primitives.relations.size()
               << " class=" << fs::recognitionName(arithmetic.recognition.arithmeticClass) << "\n";
  if (arithmeticOnly) {
    dumpArithmeticJSON(function, arithmetic);
    dumpRegionalArithmetic(function, index, input);
    return success();
  }
  if (!function.isDeclaration()) {
    dumpRecognition("explicit", pto::frontiersynch::recognizeExplicit(function.front(), index, effects));
    if (cached && cached->nodes.front().finiteGuardedResult) {
      dumpGuarded("finite-guarded", *cached->nodes.front().finiteGuardedResult);
    }
  }
  function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    llvm::outs() << "  loop " << loop.getLoc() << "\n";
    if (cached) {
      for (const auto& node : cached->nodes) {
        const bool hasLoopResult = node.anchor == loop.getOperation() &&
            node.rotatingResult && node.guardedRotatingResult;
        if (hasLoopResult) {
          dumpRecognition("rotating", *node.rotatingResult);
          dumpGuarded("guarded-rotating", *node.guardedRotatingResult);
          break;
        }
      }
    }
  });
  return success();
}
LogicalResult dumpPhaseIndex(func::FuncOp function, const pto::SyncInput &input) {
  namespace fs = pto::frontiersynch;
  fs::PhaseIndex index;
  if (failed(index.build(function, input))) {
    return failure();
  }
  for (const auto *phase : input.instructions()) {
    llvm::outs() << "phase-index " << phase->opName.getStringRef()
                 << " anchor-phases=" << index.phasesFor(phase->elementOp).size()
                 << " control-depth=" << index.controlPath(*phase).size() << "\n";
  }
  if (!function.isDeclaration()) {
    auto sequence = index.explicitSequence(function.front());
    llvm::outs() << "sequence " << function.getSymName() << ": ";
    if (succeeded(sequence)) {
      llvm::outs() << sequence->size() << "\n";
    } else {
      llvm::outs() << "structured\n";
    }
  }
  function.walk([&](Operation* operation) {
    auto label = operation->getAttrOfType<StringAttr>("test.prerequisite");
    if (!label) { return; }
    llvm::json::Array edges;
    for (const auto& edge : index.prerequisitesFor(operation)) {
      auto source = edge.producer->elementOp->getAttrOfType<StringAttr>("test.label");
      edges.push_back(llvm::json::Object{{"source", source ? source.getValue() : "unlabeled"},
          {"native", edge.native}, {"direct_ssa", edge.directSSA}});
    }
    llvm::outs() << "prerequisite-json " << llvm::json::Value(llvm::json::Object{
        {"function", function.getSymName()}, {"consumer", label.getValue()}, {"edges", std::move(edges)},
        {"mapping_obligation", index.needsValuePrerequisite(operation)}}) << "\n";
  });
  bool invalidProbe = false;
  function.walk([&](Operation *op) {
    if (auto probe = op->getAttrOfType<StringAttr>("test.probe")) {
      if (!op->getNumResults() || !op->getNumOperands()) {
        invalidProbe = true;
        return;
      }
      llvm::outs() << "availability " << probe.getValue()
                   << " before=" << index.valueAvailable(op->getResult(0), op, fs::Boundary::Before)
                   << " after=" << index.valueAvailable(op->getResult(0), op, fs::Boundary::After)
                   << " operand-before=" << index.valueAvailable(op->getOperand(0), op, fs::Boundary::Before)
                   << " operand-before-owner=" << index.valueAvailable(op->getOperand(0),
                          op->getParentOp(), fs::Boundary::Before) << "\n";
    }
  });
  return failure(invalidProbe);
}
void dumpCapabilities() {
  using namespace pto;
  auto dump = [](StringRef name, std::optional<bool> available, bool noScenario = false) {
    llvm::outs() << name << ": " << (available ? (*available ? "available" : "absent") : "unknown");
    if (noScenario) {
      llvm::outs() << " (no application scenario)";
    }
    llvm::outs() << "\n";
  };
  auto event = [&](StringRef name, StringRef architecture, SyncPhysicalCore core, PIPE source, PIPE target) {
    auto capability = getSyncEventAvailability(architecture, core, source, target);
    dump(name, capability.available, capability.noApplicationScenario);
  };
  event("aic-m-fix", "a3", SyncPhysicalCore::AIC, PIPE::PIPE_M, PIPE::PIPE_FIX);
  event("aic-m-mte3", "a3", SyncPhysicalCore::AIC, PIPE::PIPE_M, PIPE::PIPE_MTE3);
  event("aic-mte2-fix", "a3", SyncPhysicalCore::AIC, PIPE::PIPE_MTE2, PIPE::PIPE_FIX);
  event("aiv-s-v", "a2", SyncPhysicalCore::AIV, PIPE::PIPE_S, PIPE::PIPE_V);
  event("aiv-m-v", "a2", SyncPhysicalCore::AIV, PIPE::PIPE_M, PIPE::PIPE_V);
  dump("scalar-barrier", getSyncBarrierAvailability("a3", SyncPhysicalCore::AIV, PIPE::PIPE_S));
  dump("vector-barrier", getSyncBarrierAvailability("a3", SyncPhysicalCore::AIV, PIPE::PIPE_V));
  event("unknown-target", "a5", SyncPhysicalCore::AIV, PIPE::PIPE_V, PIPE::PIPE_MTE3);
  dump("unknown-core", getSyncBarrierAvailability("a3", SyncPhysicalCore::Unknown, PIPE::PIPE_V));
  event("same-pipe-event", "a3", SyncPhysicalCore::AIV, PIPE::PIPE_V, PIPE::PIPE_V);
}
std::string render(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  return text;
}
}
LogicalResult runRegionExpressionChecks(func::FuncOp function);
LogicalResult runRotatingAnalysisChecks(func::FuncOp function, const pto::SyncInput& input);
LogicalResult dumpExplicitAnalysis(func::FuncOp function, const pto::SyncInput& input);
int runArithmeticSelectorChecks();
int runArithmeticDemandChecks();
int runIntegerRelationChecks();
int runGeneralArithmeticSelectorChecks();
int runArithmeticStorageSelectorChecks();
int runGeneralArithmeticDemandChecks();
int runFiniteOverlayChecks();
int runRotatingBoundaryChecks();
int runNumericalWeightedRepetitionChecks();
int runNumericalRepeatedSquaringChecks();
bool runFiniteVisitChecks(MLIRContext*);
int runFiniteVisitInputChecks(func::FuncOp);
int runRotatingRegionChecks(func::FuncOp);
int runLifetimeStreamChecks();
int runArithmeticPeriodicConversionChecks();
int runArithmeticPeriodicInputChecks(func::FuncOp, const pto::SyncInput&);
int runResolvedLifetimeStreamChecks();
int runGuardedRankChecks();
int runGeneralArithmeticAllocationChecks();
int runMixedStrideChecks();
int runPhysicalExecutedCounterChecks();
int runStorageLaneAllocationChecks();
int runBoundedLifetimeProvenanceChecks();
int runBoundedLifetimeAllocationChecks();
int runSharedHandoffAllocationChecks();
int runFiniteAllocationQueryChecks();
int runNumericalHierarchyChecks(func::FuncOp function);
int runBoundingRegionalChecks(func::FuncOp function, const pto::SyncInput& input);
int runRequirementProvenanceChecks(func::FuncOp function, const pto::SyncInput& input);
int runCertifiedPartialReductionChecks();
int runControlOriginDistanceChecks();
int runCompactWriterReaderChecks();
int runCompactWriterReaderInputChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactLowerFactsChecks();
int runPeriodicExcessChecks();
int runCompactOrderBoundsChecks(func::FuncOp function, const pto::SyncInput& input);
int runBalancedCompactBodyChecks(func::FuncOp function, const pto::SyncInput& input);
int runGuardedCompactMatchingChecks(func::FuncOp function, const pto::SyncInput& input);
int runConditionalCompactInputChecks(func::FuncOp function, const pto::SyncInput& input);
int runBoundaryExcessChecks();
int runRepeatedExcessChecks();
int runBoundingRepetitionChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactClassRepetitionChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactBoundingPipelineChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactBoundingAllocationChecks(func::FuncOp function, const pto::SyncInput& input);
int runBoundingSequenceChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactStorageBoundaryChecks(func::FuncOp function, const pto::SyncInput& input);
int runCompactBoundaryRanksChecks(func::FuncOp function, const pto::SyncInput& input);
int runFiniteRequirementReplacementChecks(func::FuncOp function, const pto::SyncInput& input);
int runPeriodicSharedAllocationChecks();
bool runRepeatedReadOnlyStorageChecks(MLIRContext*);
bool runExpressionRelationChecks(MLIRContext*);
int runRegionalRelationChecks();
int runMixedSymbolicRegionChecks(func::FuncOp);
LogicalResult runFiniteOverlayInsertionChecks(func::FuncOp, pto::GMAliasPolicy);
int main(int argc, char **argv) {
  if (argc == 2 && StringRef(argv[1]) == "--repeated-excess-checks") {
    return runRepeatedExcessChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--boundary-excess-checks") {
    return runBoundaryExcessChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--compact-lower-facts-checks") {
    return runCompactLowerFactsChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--periodic-excess-checks") {
    return runPeriodicExcessChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--control-origin-distance-checks") {
    return runControlOriginDistanceChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--compact-writer-reader-checks") {
    return runCompactWriterReaderChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--certified-partial-reduction-checks") {
    return runCertifiedPartialReductionChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--finite-query-checks") {
    return runFiniteAllocationQueryChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--storage-lane-checks") {
    return runStorageLaneAllocationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--bounded-provenance-checks") {
    return runBoundedLifetimeProvenanceChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--physical-executed-counter-checks") {
    return runPhysicalExecutedCounterChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--mixed-stride-checks") { return runMixedStrideChecks(); }
  if (argc == 2 && StringRef(argv[1]) == "--periodic-shared-allocation-checks") {
    return runPeriodicSharedAllocationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--shared-handoff-allocation-checks") {
    return runSharedHandoffAllocationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--guarded-rank-checks") { return runGuardedRankChecks(); }
  if (argc == 2 && StringRef(argv[1]) == "--general-arithmetic-allocation-checks") {
    return runGeneralArithmeticAllocationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--bounded-lifetime-allocation-checks") {
    return runBoundedLifetimeAllocationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--resolved-lifetime-stream-checks") {
    return runResolvedLifetimeStreamChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--arithmetic-periodic-checks") {
    return runArithmeticPeriodicConversionChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--lifetime-stream-checks") { return runLifetimeStreamChecks(); }
  if (argc == 2 && StringRef(argv[1]) == "--numerical-repeated-squaring-checks") {
    return runNumericalRepeatedSquaringChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--numerical-weighted-repetition-checks") {
    return runNumericalWeightedRepetitionChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--rotating-boundary-checks") { return runRotatingBoundaryChecks(); }
  if (argc == 2 && StringRef(argv[1]) == "--finite-overlay-checks") {
    return runFiniteOverlayChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--general-arithmetic-demand-checks") {
    return runGeneralArithmeticDemandChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--integer-relations") {
    return runIntegerRelationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--arithmetic-storage-selector-checks") {
    return runArithmeticStorageSelectorChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--general-arithmetic-selector-checks") {
    return runGeneralArithmeticSelectorChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--arithmetic-demand-checks") {
    return runArithmeticDemandChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--arithmetic-selector-checks") {
    return runArithmeticSelectorChecks();
  }
  // Test-only numerical graph input; the production pass still consumes MLIR.
  if (argc == 3 && StringRef(argv[1]) == "--guarded-periodic-checks") {
    return runGuardedPeriodicChecks(argv[2]);
  }
  if (argc == 3 && StringRef(argv[1]) == "--periodic-checks") {
    return runPeriodicDemandChecks(argv[2]);
  }
  auto policy = pto::GMAliasPolicy::MayNotAlias;
  if (argc > 1 && StringRef(argv[1]).starts_with("--gm-alias=")) {
    auto name = StringRef(argv[1]).drop_front(StringRef("--gm-alias=").size());
    if (name != "may-alias" && name != "may-not-alias") {
      llvm::errs() << "unknown GM alias policy; expected may-alias or may-not-alias\n";
      return 1;
    }
    policy = name == "may-alias" ? pto::GMAliasPolicy::MayAlias : pto::GMAliasPolicy::MayNotAlias;
    for (int i = 1; i + 1 < argc; ++i) {
      argv[i] = argv[i + 1];
    }
    --argc;
  }
  std::optional<pto::frontiersynch::ArithmeticLimits> requestedProfile;
  if (argc > 1 && StringRef(argv[1]).starts_with("--arithmetic-profile=")) {
    auto profile = pto::frontiersynch::parseArithmeticProfile(
        StringRef(argv[1]).drop_front(StringRef("--arithmetic-profile=").size()));
    if (failed(profile)) {
      llvm::errs() << "invalid arithmetic profile; expected positive k:D:P:C with P<=INT64_MAX\n";
      return 1;
    }
    requestedProfile = *profile;
    for (int i = 1; i + 1 < argc; ++i) { argv[i] = argv[i + 1]; }
    --argc;
  }
  const bool allocationSessionChecks = argc == 3 && StringRef(argv[1]) == "--allocation-session-checks";
  const bool regionalSessionChecks = argc == 3 && StringRef(argv[1]) == "--regional-session-checks";
  const bool retainedChecks = argc == 3 && StringRef(argv[1]) == "--retained-demand-checks";
  const bool sessionChecks = argc == 3 && StringRef(argv[1]) == "--analysis-session-checks";
  const bool phaseCopies = argc == 3 && StringRef(argv[1]) == "--phase-copy-checks";
  const bool regionChecks = argc == 3 && StringRef(argv[1]) == "--region-contract-checks";
  const bool step1 = argc == 3 && StringRef(argv[1]) == "--step1-json";
  const bool step0 = step1 || (argc == 3 && StringRef(argv[1]) == "--step0-json");
  const bool existingDump = argc == 3 && StringRef(argv[1]) == "--existing-dump";
  const bool existing = existingDump || (argc == 3 && StringRef(argv[1]) == "--existing-check");
  const bool roundtrip = argc == 3 && StringRef(argv[1]) == "--roundtrip";
  const bool aliasChecks = argc == 3 && StringRef(argv[1]) == "--alias-contract";
  const bool expectFailure = argc == 3 && StringRef(argv[1]) == "--expect-failure";
  const bool capabilities = argc == 3 && StringRef(argv[1]) == "--capabilities";
  const bool phaseIndex = argc == 3 && StringRef(argv[1]) == "--phase-index";
  const bool storageEffects = argc == 3 && StringRef(argv[1]) == "--storage-effects";
  const bool rotatingAnalysis = argc == 3 && StringRef(argv[1]) == "--rotating-analysis";
  const bool explicitAnalysis = argc == 3 && StringRef(argv[1]) == "--explicit-analysis";
  const bool mixedSymbolicChecks = argc == 3 && StringRef(argv[1]) == "--mixed-symbolic-checks";
  const bool rotatingRegionChecks = argc == 3 && StringRef(argv[1]) == "--rotating-region-checks";
  const bool finiteVisitInput = argc == 3 && StringRef(argv[1]) == "--finite-visit-input-checks";
  const bool arithmeticPeriodicInput = argc == 3 && StringRef(argv[1]) == "--arithmetic-periodic-input-checks";
  const bool arithmetic = argc == 3 && StringRef(argv[1]) == "--arithmetic";
  const bool finiteExpansion = argc == 3 && StringRef(argv[1]) == "--finite-expansion";
  const bool certification = argc == 3 && StringRef(argv[1]) == "--certify-regions";
  const bool regionalRecognition = argc == 3 && StringRef(argv[1]) == "--recognize-regions";
  const bool recognition = certification || regionalRecognition ||
      (argc == 3 && StringRef(argv[1]) == "--recognize");
  if (requestedProfile && !recognition) {
    llvm::errs() << "arithmetic-profile requires recognize, recognize-regions or certify-regions\n";
    return 1;
  }
  const bool numericAnalysis = argc == 3 && StringRef(argv[1]) == "--numeric-analysis";
  const bool recognitionBoundary = argc == 3 && StringRef(argv[1]) == "--recognition-boundary";
  const bool insertLogical = argc == 3 && StringRef(argv[1]) == "--insert-logical";
  const bool insertLogicalLibrary = argc == 3 && StringRef(argv[1]) == "--insert-logical-library";
  const bool preparedInsertion = argc == 3 && StringRef(argv[1]) == "--prepared-insertion-checks";
  const bool insertionTrace = argc == 3 && StringRef(argv[1]) == "--insertion-trace";
  const bool expressionChecks = argc == 3 && StringRef(argv[1]) == "--region-expression-checks";
  const bool hierarchyChecks = argc == 3 && StringRef(argv[1]) == "--numerical-hierarchy-checks";
  const bool boundingChecks = argc == 3 && StringRef(argv[1]) == "--bounding-contract-checks";
  const bool compactInputChecks = argc == 3 && StringRef(argv[1]) == "--compact-input-checks";
  const bool boundingRepetitionChecks = argc == 3 && StringRef(argv[1]) == "--bounding-repetition-checks";
  const bool compactBoundingAllocationChecks =
      argc == 3 && StringRef(argv[1]) == "--compact-bounding-allocation-checks";
  const bool compactBoundingPipelineChecks = argc == 3 && StringRef(argv[1]) == "--compact-bounding-pipeline-checks";
  const bool compactClassRepetitionChecks = argc == 3 && StringRef(argv[1]) == "--compact-class-repetition-checks";
  const bool compactBoundaryRanksChecks = argc == 3 && StringRef(argv[1]) == "--compact-boundary-ranks-checks";
  const bool compactStorageBoundaryChecks = argc == 3 &&
      StringRef(argv[1]) == "--compact-storage-boundary-checks";
  const bool boundingSequenceChecks = argc == 3 && StringRef(argv[1]) == "--bounding-sequence-checks";
  const bool finiteReplacementChecks = argc == 3 && StringRef(argv[1]) == "--finite-replacement-checks";
  const bool conditionalCompactChecks = argc == 3 && StringRef(argv[1]) == "--conditional-compact-input-checks";
  const bool balancedCompactChecks = argc == 3 && StringRef(argv[1]) == "--balanced-compact-checks";
  const bool guardedCompactChecks = argc == 3 && StringRef(argv[1]) == "--guarded-compact-checks";
  const bool compactBoundsChecks = argc == 3 && StringRef(argv[1]) == "--compact-order-bounds-checks";
  const bool finiteOverlayInsertion = argc == 3 && StringRef(argv[1]) == "--finite-overlay-insertion";
  const bool finiteGuardedAnalysis = argc == 3 && StringRef(argv[1]) == "--finite-guarded-analysis";
  const bool sequenceAnalysis = argc == 3 && StringRef(argv[1]) == "--sequence-analysis";
  const bool structuredTrace = argc == 3 && StringRef(argv[1]) == "--structured-trace";
  const bool physicalTrace = argc == 3 && StringRef(argv[1]) == "--physical-trace";
  if (argc != 2 && !allocationSessionChecks && !regionalSessionChecks && !retainedChecks && !sessionChecks &&
      !rotatingAnalysis && !explicitAnalysis &&
      !arithmetic && !finiteExpansion && !recognition &&
      !numericAnalysis && !insertLogical && !insertLogicalLibrary && !recognitionBoundary &&
      !insertionTrace && !physicalTrace && !structuredTrace && !sequenceAnalysis && !finiteGuardedAnalysis &&
      !mixedSymbolicChecks && !rotatingRegionChecks && !finiteVisitInput && !arithmeticPeriodicInput &&
      !expressionChecks && !hierarchyChecks && !boundingChecks &&
      !compactInputChecks && !compactBoundsChecks &&
      !balancedCompactChecks && !guardedCompactChecks && !conditionalCompactChecks && !finiteReplacementChecks &&
      !boundingSequenceChecks && !compactStorageBoundaryChecks && !compactBoundaryRanksChecks &&
      !boundingRepetitionChecks && !compactClassRepetitionChecks &&
      !compactBoundingPipelineChecks && !compactBoundingAllocationChecks &&
      !preparedInsertion && !finiteOverlayInsertion &&
      !expectFailure && !capabilities && !phaseIndex && !storageEffects && !aliasChecks && !roundtrip &&
      !regionChecks && !phaseCopies && !step0 && !existing) {
    llvm::errs() << "usage: pto-sync-input-test "
                 << "[--gm-alias=may-alias|may-not-alias] "
                 << "[--alias-contract|--expect-failure|--capabilities|--phase-index|--storage-effects|"
                 "--allocation-session-checks|--retained-demand-checks|--analysis-session-checks|"
                 "--recognize|--recognition-boundary|--numeric-analysis|--insert-logical|--insert-logical-library|"
                 "--prepared-insertion-checks|--insertion-trace|"
                 "--finite-guarded-analysis|--finite-overlay-insertion|--region-expression-checks|--sequence-analysis|"
                 "--structured-trace|--physical-trace|--numerical-hierarchy-checks|--bounding-contract-checks|"
                 "--compact-input-checks|"
                 "--compact-order-bounds-checks|--balanced-compact-checks|--guarded-compact-checks|"
                 "--conditional-compact-input-checks|--finite-replacement-checks|--bounding-sequence-checks|"
                 "--compact-storage-boundary-checks|--compact-boundary-ranks-checks|"
                 "--bounding-repetition-checks|--compact-class-repetition-checks|"
                 "--compact-bounding-pipeline-checks|--compact-bounding-allocation-checks|"
                 "--arithmetic|--finite-expansion|--arithmetic-periodic-input-checks|"
                 "--explicit-analysis|--rotating-analysis|--roundtrip|"
                 "--region-contract-checks|"
                 "--step0-json|--step1-json|--existing-check|--existing-dump|--phase-copy-checks] input.pto\n";
    return 1;
  }
  DialectRegistry dialects;
  dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect,
                  scf::SCFDialect, LLVM::LLVMDialect, DLTIDialect>();
  MLIRContext context(dialects);
  if (argc == 2 && StringRef(argv[1]) == "--regional-relation-checks") {
    return runRegionalRelationChecks();
  }
  if (argc == 2 && StringRef(argv[1]) == "--expression-relation-checks") {
    context.disableMultithreading();
    return runExpressionRelationChecks(&context) ? 0 : 1;
  }
  if (argc == 2 && StringRef(argv[1]) == "--finite-visit-checks") {
    context.disableMultithreading();
    return runFiniteVisitChecks(&context) ? 0 : 1;
  }
  if (argc == 2 && StringRef(argv[1]) == "--repeated-readonly-storage-checks") {
    context.disableMultithreading();
    return runRepeatedReadOnlyStorageChecks(&context) ? 0 : 1;
  }
  context.disableMultithreading();
  const bool hasOption = allocationSessionChecks || regionalSessionChecks || retainedChecks || sessionChecks ||
                         mixedSymbolicChecks || rotatingAnalysis ||
                         explicitAnalysis || expectFailure ||
                         capabilities || phaseIndex ||
                         storageEffects || recognition || numericAnalysis || insertLogical || insertLogicalLibrary ||
                         recognitionBoundary ||
                         insertionTrace || physicalTrace ||
                         structuredTrace || sequenceAnalysis || finiteGuardedAnalysis || finiteOverlayInsertion ||
                         expressionChecks || hierarchyChecks || boundingChecks || compactInputChecks ||
                         compactBoundsChecks || balancedCompactChecks || guardedCompactChecks ||
                         conditionalCompactChecks || finiteReplacementChecks || boundingSequenceChecks ||
                         compactStorageBoundaryChecks || compactBoundaryRanksChecks ||
                         boundingRepetitionChecks || compactClassRepetitionChecks || compactBoundingPipelineChecks ||
                         compactBoundingAllocationChecks ||
                         preparedInsertion || arithmetic || finiteExpansion || arithmeticPeriodicInput ||
                         finiteVisitInput || rotatingRegionChecks ||
                         aliasChecks || roundtrip || regionChecks || phaseCopies || step0 || existing;
  const auto filename = argv[hasOption ? 2 : 1];
  auto module = parseSourceFile<ModuleOp>(filename, &context);
  if (!module || failed(verify(*module))) {
    return 1;
  }
  if (allocationSessionChecks) {
    const auto before = render(module->getOperation());
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(checkAllocationSession(function, policy))) { return 1; }
    }
    if (before != render(module->getOperation())) { return 1; }
    return 0;
  }
  if (regionalSessionChecks) {
    const auto before = render(module->getOperation());
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(checkRegionalSession(function, policy))) { return 1; }
    }
    if (before != render(module->getOperation())) { return 1; }
    llvm::outs() << "regional-session: cached original-identities independent-root-evidence source-unchanged\n";
    return 0;
  }
  if (retainedChecks) {
    const auto before = render(module->getOperation());
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(checkRetainedDemands(function, policy))) { return 1; }
    }
    const auto after = render(module->getOperation());
    if (before != after) { return 1; }
    llvm::outs() << "retained-demands: endpoint-failure cached-retry exact-fallback source-unchanged\n";
    return 0;
  }
  if (sessionChecks) {
    auto before = render(module->getOperation());
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(checkAnalysisSession(function, policy))) { return 1; }
    }
    const auto after = render(module->getOperation());
    if (after != before) { return 1; }
    llvm::outs() << "analysis-session: retained-demands fresh-fragments source-unchanged\n";
    return 0;
  }
  if (hierarchyChecks) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (runNumericalHierarchyChecks(function)) { return 1; }
    }
    return 0;
  }
  if (expressionChecks) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runRegionExpressionChecks(function))) {
        return 1;
      }
    }
    return 0;
  }
  if (finiteOverlayInsertion) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runFiniteOverlayInsertionChecks(function, policy))) { return 1; }
    }
    return 0;
  }
  if (finiteGuardedAnalysis) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runFiniteGuardedAnalysisChecks(function, policy))) { return 1; }
    }
    return 0;
  }
  if (sequenceAnalysis) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runSequenceAnalysisChecks(function, policy))) { return 1; }
    }
    return 0;
  }
  if (mixedSymbolicChecks) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (runMixedSymbolicRegionChecks(function)) { return 1; }
    }
    return 0;
  }
  if (rotatingRegionChecks) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (runRotatingRegionChecks(function)) { return 1; }
    }
    return 0;
  }
  if (finiteVisitInput) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (runFiniteVisitInputChecks(function)) { return 1; }
    }
    return 0;
  }
  if (structuredTrace) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runStructuredInsertionChecks(function, policy))) {
        return 1;
      }
    }
    return 0;
  }
  if (existing) {
    PassManager manager(&context);
    pto::PTOInsertSyncOptions options;
    options.gmAlias = policy == pto::GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias";
    manager.addNestedPass<func::FuncOp>(pto::createPTOInsertSyncPass(options));
    if (failed(manager.run(*module))) {
      return 1;
    }
    if (existingDump) {
      module->print(llvm::outs());
    }
    return 0;
  }
  if (roundtrip) {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  if (capabilities) {
    dumpCapabilities();
  }
  const auto before = render(module->getOperation());
  if (preparedInsertion) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runPreparedInsertionChecks(function))) {
        return 1;
      }
    }
    module->print(llvm::outs());
    return 0;
  }
  if (insertionTrace || physicalTrace) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (failed(runLogicalInsertionChecks(function, policy, physicalTrace))) {
        return 1;
      }
    }
    return 0;
  }
  // Library regressions may exercise detached endpoint recipes while the
  // production pass deliberately stops at mathematical certification.
  if (insertLogicalLibrary) {
    for (auto function : module->getOps<func::FuncOp>()) {
      if (function.isDeclaration()) { continue; }
      pto::frontiersynch::FrontierAnalysis analysis(function);
      if (failed(analysis.initialize(policy))) { return 1; }
      pto::frontiersynch::AnalysisRequest request;
      request.needs.synchronization = true;
      auto result = analysis.analyze(request);
      if (result.status != pto::frontiersynch::AnalysisStatus::Ready) { return 1; }
      auto prepared = analysis.prepareLogical(result);
      if (failed(prepared)) { return 1; }
      const bool inserted = succeeded(pto::frontiersynch::insertLogicalSynchronization(function, **prepared));
      analysis.invalidate();
      if (!inserted || failed(verify(function))) { return 1; }
    }
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  if (insertLogical || recognitionBoundary) {
    PassManager manager(&context);
    pto::PTOFrontierAnalysisOptions options;
    options.gmAlias = policy == pto::GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias";
    manager.addPass(pto::createPTOFrontierAnalysisPass(options));
    const bool rejected = failed(manager.run(*module));
    if (recognitionBoundary) {
      const bool preserved = render(module->getOperation()) == before;
      if (!rejected || !preserved) { return 1; }
      llvm::outs() << "recognition-boundary: rejected; source-unchanged\n";
      return 0;
    }
    if (rejected) { return 1; }
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  if (recognition || numericAnalysis) {
    auto delegation = pto::frontiersynch::recognizeClosedCallees(*module);
    for (auto function : module->getOps<func::FuncOp>()) {
      if (function.isDeclaration()) { continue; }
      if (pto::hasManualOnCoreSynchronization(function)) {
        llvm::outs() << "recognition-skipped " << function.getSymName() << ": manual-on-core-synchronization\n";
        continue;
      }
      if (auto found = delegation.wrappers.find(function); found != delegation.wrappers.end()) {
        llvm::outs() << "recognition " << function.getSymName() << "\n";
        dumpRecognition("closed-callee", found->second.result);
        llvm::outs() << "  callee-closure=requires-analysis-and-insertion\n";
        llvm::json::Array callees;
        for (auto callee : found->second.callees) { callees.push_back(callee.getSymName().str()); }
        llvm::outs() << "closed-callee-json " << llvm::json::Value(llvm::json::Object{
            {"function", function.getSymName()},
            {"state", pto::frontiersynch::recognitionName(found->second.result.state)},
            {"callees", std::move(callees)}, {"closure_established", false}}) << "\n";
      }
      pto::frontiersynch::FrontierAnalysis analysis(function);
      if (failed(analysis.initialize(policy))) {
        return 1;
      }
      if (requestedProfile && failed(analysis.configureArithmeticProfiles(
          ArrayRef<pto::frontiersynch::ArithmeticLimits>(&*requestedProfile, 1),
          ArrayRef<pto::frontiersynch::ArithmeticLimits>(&*requestedProfile, 1)))) { return 1; }
      // Insertion consumes the periodic result without constructing arithmetic.
      // The full recognition report explicitly requests and caches that route.
      if (analysis.result()->arithmetic || failed(analysis.recognizeArithmetic())) {
        return 1;
      }
      if (requestedProfile && succeeded(analysis.configureArithmeticProfiles(
          ArrayRef<pto::frontiersynch::ArithmeticLimits>(&*requestedProfile, 1),
          ArrayRef<pto::frontiersynch::ArithmeticLimits>(&*requestedProfile, 1)))) {
        function.emitError("an arithmetic request did not freeze its profile context");
        return 1;
      }
      const auto* arithmeticCandidate = analysis.result()->arithmetic ? &*analysis.result()->arithmetic : nullptr;
      if (failed(analysis.initialize(policy)) || failed(analysis.recognizeArithmetic()) ||
          (analysis.result()->arithmetic ? &*analysis.result()->arithmetic : nullptr) != arithmeticCandidate) {
        return 1;
      }
      if (llvm::any_of(analysis.result()->nodes, [](const auto& node) {
            return node.periodicAnalysis || node.logicalEndpoints || node.periodicAllocation;
          })) {
        function.emitError("recognition unexpectedly executed a numeric backend");
        return 1;
      }
      if (regionalRecognition && failed(analysis.recognizeRegionalArithmetic())) { return 1; }
      if (certification) {
        const bool exportProbeFailed = function->hasAttr("test.export_failure") &&
            failed(checkArithmeticExportFailure(function, policy));
        if (exportProbeFailed) {
          return 1;
        }
        const bool arithmeticProbeFailed = function->hasAttr("test.arithmetic_requests") &&
            failed(checkArithmeticRequests(function, policy));
        if (arithmeticProbeFailed) {
          return 1;
        }
        if (function->hasAttr("test.native_only")) {
          pto::frontiersynch::FrontierAnalysis native(function);
          if (failed(native.initialize(policy, false))) { return 1; }
          const auto nativeResults = native.certifyRegions();
          const bool mislabeled = nativeResults.empty() || !nativeResults.front().analysis.mathematical ||
              nativeResults.front().status != pto::frontiersynch::CertificationStatus::Unresolved ||
              !nativeResults.front().selectedClass.empty();
          if (mislabeled) {
            function.emitError("native-only demands falsely established finite-list membership");
            return 1;
          }
          llvm::outs() << "native-certification: retained-demands unresolved-class\n";
        }
        const bool expansionProbeFailed = function->hasAttr("test.finite_expansion_session") &&
            failed(checkFiniteExpansionSession(function, policy));
        if (expansionProbeFailed) { return 1; }
        const bool varyingProbeFailed = function->hasAttr("test.varying_boundary_session") &&
            failed(checkVaryingBoundarySession(function, policy));
        if (varyingProbeFailed) { return 1; }
        const auto results = analysis.certifyRegions();
        const bool childOnly = function->hasAttr("test.finite_visit_child");
        const bool finiteVisitProbe = function->hasAttr("test.finite_visit_session") || childOnly;
        if (finiteVisitProbe) {
          pto::frontiersynch::FrontierAnalysis probe(function);
          if (failed(probe.initialize(policy))) { return 1; }
          auto node = llvm::find_if(probe.result()->nodes, [](const auto& candidate) {
            return candidate.kind == pto::frontiersynch::StructureKind::Loop;
          });
          if (node == probe.result()->nodes.end()) { return 1; }
          const auto id = static_cast<std::size_t>(node - probe.result()->nodes.begin());
          auto child = probe.analyzeFiniteVisit({id});
          const bool childExact = child.mathematical && child.mathematical->finiteVisitDemands &&
              child.mathematical->region == id && !probe.hasWholeFunctionMinimumDemands();
          if (!childExact) {
            function.emitError("finite-visit probe: child did not retain finite-type demands independently");
            return 1;
          }
          auto retained = probe.analyzeFiniteVisit({});
          if (childOnly) {
            const bool leaked = retained.mathematical || probe.hasWholeFunctionMinimumDemands() ||
                probe.constructionCounts().structuralIndices != 1;
            if (leaked) {
              function.emitError("finite-visit probe: external payload allowed whole-region evidence");
              return 1;
            }
            llvm::outs() << "finite-visit-child: exact-demands no-whole-region-evidence\n";
          } else {
            const auto owner = retained.mathematical;
            const bool exact = owner && owner->finiteVisitDemands && owner->finiteVisitDemands->demands &&
                owner->finiteVisitDemands->demands->boundaries.size() == 16 &&
                owner->finiteVisitDemands == child.mathematical->finiteVisitDemands;
            if (!exact) {
              function.emitError("finite-visit probe: whole-loop owner was not reused");
              return 1;
            }
            for (unsigned capability = 0; capability < 6; ++capability) {
              pto::frontiersynch::AnalysisNeeds needs;
              needs.queries = capability % 3 == 0;
              needs.selectors = capability % 3 == 1;
              needs.synchronization = capability % 3 == 2;
              auto request = pto::frontiersynch::AnalysisRequest{};
              request.needs = needs;
              const auto beforeRetry = probe.constructionCounts().mathematicalAttempts;
              auto unavailable = probe.analyzeFiniteVisit(request);
              const bool preserved = unavailable.mathematical == owner &&
                  unavailable.status == pto::frontiersynch::AnalysisStatus::UnmetObligation;
              if (!preserved) {
                function.emitError("finite-visit probe: unsupported export did not preserve demands");
                return 1;
              }
              if (capability >= 3 && probe.constructionCounts().mathematicalAttempts != beforeRetry) { return 1; }
            }
            const auto afterExports = probe.constructionCounts().mathematicalAttempts;
            if (probe.analyzeFiniteVisit({}).mathematical != owner ||
                probe.constructionCounts().mathematicalAttempts != afterExports ||
                probe.constructionCounts().structuralIndices != 1) {
              return 1;
            }
            llvm::outs() << "finite-visit-session: exact-demands retained-exports cached-retry\n";
          }
        }
        llvm::json::Array regions;
        for (const auto& result : results) {
          llvm::json::Array obligations;
          for (const auto& obligation : result.analysis.obligations) {
            obligations.push_back(obligation.diagnostic);
          }
          regions.push_back(llvm::json::Object{{"node", result.region},
              {"status", result.status == pto::frontiersynch::CertificationStatus::Recognized ?
                  "recognized" : "unresolved"},
              {"class", result.selectedClass}, {"representation", result.representation},
              {"exact_demands", static_cast<bool>(result.analysis.mathematical)},
              {"queries", result.analysis.available.queries}, {"selectors", result.analysis.available.selectors},
              {"obligations", std::move(obligations)}});
        }
        const auto& counts = analysis.constructionCounts();
        if (counts.logicalPreparations || counts.allocationExports) {
          function.emitError("certification unexpectedly prepared code or allocation");
          return 1;
        }
        const auto attempts = counts.mathematicalAttempts;
        const auto forms = analysis.result()->arithmeticContracts.size();
        (void)analysis.certifyRegions();
        if (analysis.constructionCounts().mathematicalAttempts != attempts ||
            analysis.result()->arithmeticContracts.size() != forms) {
          function.emitError("repeated certification reconstructed analysis");
          return 1;
        }
        llvm::outs() << "certification-json " << llvm::json::Value(llvm::json::Object{
            {"function", function.getSymName()}, {"regions", std::move(regions)},
            {"mathematical_attempts", attempts}, {"logical_preparations", counts.logicalPreparations},
            {"allocation_exports", counts.allocationExports}}) << "\n";
      }
      if (numericAnalysis && failed(analysis.prepareNumericCandidateExports())) { return 1; }
      const auto& input = *analysis.input();
      const auto& program = *analysis.result();
      if (failed(verifyProgramStructure(function, input, program)) ||
          (!numericAnalysis && failed(recognize(function, input, false, &program))) ||
          failed(dumpProgramRecognition(function, input, program))) {
        return 1;
      }
    }
    if (render(module->getOperation()) != before) {
      return 1;
    }
    llvm::outs() << "source-unchanged; synchronization-insertion-not-run\n";
    return 0;
  }
  pto::SyncInput input(policy);
  for (auto function : module->getOps<func::FuncOp>()) {
    const auto view = (step1 || finiteExpansion) ? pto::SyncInstructionView::PipeEnvelopes :
        pto::SyncInstructionView::TranslatorStages;
    const bool translated = succeeded(input.build(function, view));
    if (expectFailure) {
      if (translated || !input.ir().empty() || !input.instructions().empty() || !input.buffers().empty()) {
        return 1;
      }
      llvm::outs() << "translation-failed; records-empty\n";
      continue;
    }
    if (!translated) {
      return 1;
    }
    if (finiteExpansion) {
      pto::frontiersynch::PhaseIndex index;
      if (failed(index.build(function, input))) { return 1; }
      const auto original = render(function);
      dumpRegionalArithmetic(function, index, input);
      const bool unchanged = render(function) == original;
      if (!unchanged) { return 1; }
      llvm::outs() << "finite-expansion: source-unchanged\n";
      continue;
    }
    if (arithmeticPeriodicInput) {
      if (runArithmeticPeriodicInputChecks(function, input)) { return 1; }
      continue;
    }
    if (boundingChecks) {
      if (runBoundingRegionalChecks(function, input) || runRequirementProvenanceChecks(function, input)) {
        return 1;
      }
      continue;
    }
    if (compactInputChecks) {
      if (runCompactWriterReaderInputChecks(function, input)) { return 1; }
      continue;
    }
    if (compactBoundingAllocationChecks) {
      if (runCompactBoundingAllocationChecks(function, input)) { return 1; }
      continue;
    }
    if (compactBoundingPipelineChecks) {
      if (runCompactBoundingPipelineChecks(function, input)) { return 1; }
      continue;
    }
    if (boundingRepetitionChecks) {
      if (runBoundingRepetitionChecks(function, input)) { return 1; }
      continue;
    }
    if (compactClassRepetitionChecks) {
      if (runCompactClassRepetitionChecks(function, input)) { return 1; }
      continue;
    }
    if (compactBoundaryRanksChecks) {
      if (runCompactBoundaryRanksChecks(function, input)) { return 1; }
      continue;
    }
    if (compactStorageBoundaryChecks) {
      if (runCompactStorageBoundaryChecks(function, input)) { return 1; }
      continue;
    }
    if (boundingSequenceChecks) {
      if (runBoundingSequenceChecks(function, input)) { return 1; }
      continue;
    }
    if (finiteReplacementChecks) {
      if (runFiniteRequirementReplacementChecks(function, input)) { return 1; }
      continue;
    }
    if (conditionalCompactChecks) {
      if (runConditionalCompactInputChecks(function, input)) { return 1; }
      continue;
    }
    if (balancedCompactChecks) {
      if (runBalancedCompactBodyChecks(function, input)) { return 1; }
      continue;
    }
    if (guardedCompactChecks) {
      if (runGuardedCompactMatchingChecks(function, input)) { return 1; }
      continue;
    }
    if (compactBoundsChecks) {
      if (runCompactOrderBoundsChecks(function, input)) { return 1; }
      continue;
    }
    if (rotatingAnalysis) {
      if (failed(runRotatingAnalysisChecks(function, input))) {
        return 1;
      }
      continue;
    }
    if (explicitAnalysis) {
      if (failed(dumpExplicitAnalysis(function, input))) {
        return 1;
      }
      continue;
    }
    if (step0) {
      if (failed(auditSyncStep0(function, input, step1))) {
        return 1;
      }
      continue;
    }
    if (phaseCopies) {
      if (!checkPhaseCopies(input) || !checkMacroEnvelopeView(function, input)) {
        llvm::errs() << "copied phase lost shared dependencies\n";
        return 1;
      }
      llvm::outs() << "phase-copy-check " << function.getSymName() << ": passed\n";
    }
    if (regionChecks && runSyncRegionContractChecks(function, input)) {
      return 1;
    }
    if (aliasChecks && runSyncAliasChecks(function, input)) {
      return 1;
    }
    if ((recognition || arithmetic) && failed(recognize(function, input, arithmetic))) {
      return 1;
    }
    if (phaseIndex && failed(dumpPhaseIndex(function, input))) {
      return 1;
    }
    if (storageEffects && failed(dumpStorageEffects(function, input))) {
      return 1;
    }
    llvm::outs() << function.getSymName() << ": phases=" << input.instructions().size() << "\n";
    for (const auto *phase : input.instructions()) {
      llvm::outs() << "  " << phase->opName.getStringRef()
                   << " pipe=" << static_cast<unsigned>(phase->kPipeValue)
                   << " reads=" << phase->useVec.size() << " writes=" << phase->defVec.size()
                   << " core=" << static_cast<unsigned>(phase->compoundCoreType)
                   << " macro=" << phase->macroOpInstanceId << "\n";
      for (const auto *memory : phase->useVec) {
        llvm::outs() << "    read-space=" << static_cast<unsigned>(memory->scope) << "\n";
      }
      for (const auto *memory : phase->defVec) {
        llvm::outs() << "    write-space=" << static_cast<unsigned>(memory->scope) << "\n";
      }
    }
  }
  if (arithmeticPeriodicInput || rotatingAnalysis || balancedCompactChecks || guardedCompactChecks ||
      compactBoundingPipelineChecks || compactBoundingAllocationChecks) {
    return 0;
  }
  if (before != render(module->getOperation())) {
    llvm::errs() << "shared extraction mutated source IR\n";
    return 1;
  }
  if (!step0) {
    llvm::outs() << "source-unchanged; dependency-analysis-not-run\n";
  }
  return 0;
}
