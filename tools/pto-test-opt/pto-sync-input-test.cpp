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
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ClosedCallees.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/IR/PTO.h"
#include "SyncPhaseCopyChecks.h"
#include "SyncLogicalInsertionChecks.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
LogicalResult auditSyncStep0(func::FuncOp function, const pto::SyncInput &input);
namespace {
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
  const bool phaseCopies = argc == 3 && StringRef(argv[1]) == "--phase-copy-checks";
  const bool regionChecks = argc == 3 && StringRef(argv[1]) == "--region-contract-checks";
  const bool step0 = argc == 3 && StringRef(argv[1]) == "--step0-json";
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
  const bool finiteVisitInput = argc == 3 && StringRef(argv[1]) == "--finite-visit-input-checks";
  const bool arithmeticPeriodicInput = argc == 3 && StringRef(argv[1]) == "--arithmetic-periodic-input-checks";
  const bool arithmetic = argc == 3 && StringRef(argv[1]) == "--arithmetic";
  const bool recognition = argc == 3 && StringRef(argv[1]) == "--recognize";
  const bool numericAnalysis = argc == 3 && StringRef(argv[1]) == "--numeric-analysis";
  const bool insertLogical = argc == 3 && StringRef(argv[1]) == "--insert-logical";
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
  if (argc != 2 && !rotatingAnalysis && !explicitAnalysis && !arithmetic && !recognition &&
      !numericAnalysis && !insertLogical &&
      !insertionTrace && !physicalTrace && !structuredTrace && !sequenceAnalysis && !finiteGuardedAnalysis &&
      !finiteVisitInput && !arithmeticPeriodicInput && !expressionChecks && !hierarchyChecks && !boundingChecks &&
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
                 "--recognize|--numeric-analysis|--insert-logical|--prepared-insertion-checks|--insertion-trace|"
                 "--finite-guarded-analysis|--finite-overlay-insertion|--region-expression-checks|--sequence-analysis|"
                 "--structured-trace|--physical-trace|--numerical-hierarchy-checks|--bounding-contract-checks|"
                 "--compact-input-checks|"
                 "--compact-order-bounds-checks|--balanced-compact-checks|--guarded-compact-checks|"
                 "--conditional-compact-input-checks|--finite-replacement-checks|--bounding-sequence-checks|"
                 "--compact-storage-boundary-checks|--compact-boundary-ranks-checks|"
                 "--bounding-repetition-checks|--compact-class-repetition-checks|"
                 "--compact-bounding-pipeline-checks|--compact-bounding-allocation-checks|"
                 "--arithmetic|--arithmetic-periodic-input-checks|--explicit-analysis|--rotating-analysis|--roundtrip|"
                 "--region-contract-checks|"
                 "--step0-json|--existing-check|--existing-dump|--phase-copy-checks] input.pto\n";
    return 1;
  }
  DialectRegistry dialects;
  dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect, LLVM::LLVMDialect>();
  MLIRContext context(dialects);
  if (argc == 2 && StringRef(argv[1]) == "--finite-visit-checks") {
    context.disableMultithreading();
    return runFiniteVisitChecks(&context) ? 0 : 1;
  }
  if (argc == 2 && StringRef(argv[1]) == "--repeated-readonly-storage-checks") {
    context.disableMultithreading();
    return runRepeatedReadOnlyStorageChecks(&context) ? 0 : 1;
  }
  context.disableMultithreading();
  const bool hasOption = rotatingAnalysis || explicitAnalysis || expectFailure || capabilities || phaseIndex ||
                         storageEffects || recognition || numericAnalysis || insertLogical ||
                         insertionTrace || physicalTrace ||
                         structuredTrace || sequenceAnalysis || finiteGuardedAnalysis || finiteOverlayInsertion ||
                         expressionChecks || hierarchyChecks || boundingChecks || compactInputChecks ||
                         compactBoundsChecks || balancedCompactChecks || guardedCompactChecks ||
                         conditionalCompactChecks || finiteReplacementChecks || boundingSequenceChecks ||
                         compactStorageBoundaryChecks || compactBoundaryRanksChecks ||
                         boundingRepetitionChecks || compactClassRepetitionChecks || compactBoundingPipelineChecks ||
                         compactBoundingAllocationChecks ||
                         preparedInsertion || arithmetic || arithmeticPeriodicInput || finiteVisitInput ||
                         aliasChecks || roundtrip || regionChecks || phaseCopies || step0 || existing;
  const auto filename = argv[hasOption ? 2 : 1];
  auto module = parseSourceFile<ModuleOp>(filename, &context);
  if (!module || failed(verify(*module))) {
    return 1;
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
  if (insertLogical) {
    PassManager manager(&context);
    pto::PTOFrontierAnalysisOptions options;
    options.gmAlias = policy == pto::GMAliasPolicy::MayAlias ? "may-alias" : "may-not-alias";
    manager.addPass(pto::createPTOFrontierAnalysisPass(options));
    if (failed(manager.run(*module))) {
      return 1;
    }
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
      // Insertion consumes the periodic result without constructing arithmetic.
      // The full recognition report explicitly requests and caches that route.
      if (analysis.result()->arithmetic || failed(analysis.recognizeArithmetic())) {
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
      if (numericAnalysis && failed(analysis.analyzeNumericCandidates())) { return 1; }
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
    const bool translated = succeeded(input.build(function));
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
      if (failed(auditSyncStep0(function, input))) {
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
