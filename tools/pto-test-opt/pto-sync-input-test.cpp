// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect the shared instruction contract without any dependency analyzer.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
void dumpArithmeticJSON(func::FuncOp function, const pto::frontiersynch::ArithmeticProgram& program);
int runSyncRegionContractChecks(func::FuncOp function, const pto::SyncInput &input);
int runSyncAliasChecks(func::FuncOp function, const pto::SyncInput &input);
namespace {
LogicalResult dumpStorageEffects(func::FuncOp function, const pto::SyncInput &input) {
  pto::SyncStorageEffects storage;
  if (failed(storage.build(input))) {
    return failure();
  }
  llvm::outs() << "storage " << function.getSymName() << ": cells=" << storage.cells().size()
               << " all-exact=" << storage.allAccessesExact() << "\n";
  for (auto [id, cell] : llvm::enumerate(storage.cells())) {
    llvm::outs() << "  cell " << id << " space=" << static_cast<unsigned>(cell.space)
                 << " [" << cell.begin << "," << cell.end << ")\n";
  }
  for (auto [id, effect] : llvm::enumerate(storage.effects())) {
    auto precision = effect.precision == pto::SyncAccessPrecision::Exact ? "exact" :
                     effect.precision == pto::SyncAccessPrecision::UpperBound ? "upper" : "unknown";
    llvm::outs() << "  effect " << id << " " << effect.phase->opName.getStringRef()
                 << (effect.mode == pto::SyncAccessMode::Read ? " read " : " write ")
                 << precision << " definite-write=" << effect.hasDefiniteWrites() << " cells=";
    llvm::interleaveComma(effect.cells, llvm::outs());
    llvm::outs() << "\n";
    if (effect.descriptorRegion) {
      const auto& region = *effect.descriptorRegion;
      llvm::outs() << "    descriptor base=" << (region.base ? "pointer" : "absolute")
                   << " bytes=" << region.elementBytes << " map=";
      AffineMap::get(region.extents.size(), region.symbols.size(), region.byteOffset).print(llvm::outs());
      llvm::outs() << " extents=";
      llvm::interleaveComma(region.extents, llvm::outs());
      llvm::outs() << " loops=" << region.iterations.size() << "\n";
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
  // Rebuilding must neither accumulate cells nor retain prior phase mappings.
  auto cellCount = storage.cells().size(), effectCount = storage.effects().size();
  if (failed(storage.build(input)) || storage.cells().size() != cellCount || storage.effects().size() != effectCount) {
    return failure();
  }
  return success();
}
void dumpRecognition(StringRef label, const pto::frontiersynch::RecognitionResult &result) {
  namespace fs = pto::frontiersynch;
  llvm::outs() << "recognize " << label << ": " << fs::recognitionName(result.state)
               << " backend=unavailable\n";
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
                 << " stride=" << access.stride << " offset=" << access.offset
                 << " refresh=" << access.refresh << " atom=";
    if (access.atom) {
      llvm::outs() << "[" << access.atom->first << "," << access.atom->second << ")";
    } else {
      llvm::outs() << "unknown";
    }
    if (access.guard) {
      llvm::outs() << " guard=" << *access.guard;
    }
    llvm::outs() << "\n";
  }
}
void dumpGuarded(StringRef label, const pto::frontiersynch::GuardedRecognition &result) {
  dumpRecognition(label, result.result);
  llvm::outs() << "  guarded phases=" << result.phases.size() << " guards=" << result.guards.size()
               << " entry-guards=" << (result.entryGuardsAvailable ? "available" : "late") << "\n";
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
LogicalResult recognize(func::FuncOp function, const pto::SyncInput &input, bool arithmeticOnly) {
  pto::frontiersynch::PhaseIndex index;
  pto::SyncStorageEffects effects;
  if (failed(index.build(function, input)) || failed(effects.build(input))) {
    return failure();
  }
  llvm::outs() << "recognition " << function.getSymName() << "\n";
  namespace fs = pto::frontiersynch;
  // Fixed test-client class: up to 8 pipes, 8 coordinates, period 2, coefficient 8.
  // Production callers choose their class limits, never observed input maxima.
  auto arithmetic = fs::recognizeArithmeticProgram(function, index, input, effects, {8, 8, 2, 8});
  auto status = arithmetic.extraction.state == fs::RecognitionState::Applicable ?
                arithmetic.recognition.state : arithmetic.extraction.state;
  llvm::outs() << "recognize arithmetic: " << fs::recognitionName(status) << " backend=unavailable\n";
  for (const auto &diagnostic : arithmetic.extraction.diagnostics) {
    llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue) << "\n";
  }
  for (const auto &diagnostic : arithmetic.recognition.diagnostics) {
    llvm::outs() << "  issue " << fs::recognitionName(diagnostic.issue) << "\n";
  }
  llvm::outs() << "  arithmetic sites=" << arithmetic.sites.size()
               << " parameters=" << arithmetic.parameters.size()
               << " primitives=" << arithmetic.primitives.relations.size()
               << " class=" << fs::recognitionName(arithmetic.recognition.arithmeticClass) << "\n";
  if (arithmeticOnly) {
    dumpArithmeticJSON(function, arithmetic);
    return success();
  }
  if (!function.isDeclaration()) {
    dumpRecognition("explicit", pto::frontiersynch::recognizeExplicit(function.front(), index, effects));
    dumpGuarded("finite-guarded", pto::frontiersynch::recognizeFiniteGuarded(function.getBody(), index, effects));
  }
  function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
    llvm::outs() << "  loop " << loop.getLoc() << "\n";
    dumpRecognition("rotating", pto::frontiersynch::recognizeRotating(loop, index, input, effects));
    dumpGuarded("guarded-rotating", pto::frontiersynch::recognizeGuardedRotating(loop, index, input, effects));
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
int main(int argc, char **argv) {
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
  const bool regionChecks = argc == 3 && StringRef(argv[1]) == "--region-contract-checks";
  const bool roundtrip = argc == 3 && StringRef(argv[1]) == "--roundtrip";
  const bool aliasChecks = argc == 3 && StringRef(argv[1]) == "--alias-contract";
  const bool expectFailure = argc == 3 && StringRef(argv[1]) == "--expect-failure";
  const bool capabilities = argc == 3 && StringRef(argv[1]) == "--capabilities";
  const bool phaseIndex = argc == 3 && StringRef(argv[1]) == "--phase-index";
  const bool storageEffects = argc == 3 && StringRef(argv[1]) == "--storage-effects";
  const bool arithmetic = argc == 3 && StringRef(argv[1]) == "--arithmetic";
  const bool recognition = argc == 3 && StringRef(argv[1]) == "--recognize";
  if (argc != 2 && !arithmetic && !recognition && !expectFailure &&
      !capabilities && !phaseIndex && !storageEffects && !aliasChecks && !roundtrip && !regionChecks) {
    llvm::errs() << "usage: pto-sync-input-test "
                 << "[--gm-alias=may-alias|may-not-alias] "
                 << "[--alias-contract|--expect-failure|--capabilities|--phase-index|--storage-effects|"
                 "--recognize|--arithmetic|--roundtrip|--region-contract-checks] input.pto\n";
    return 1;
  }
  DialectRegistry dialects;
  dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
  MLIRContext context(dialects);
  context.disableMultithreading();
  const bool hasOption = expectFailure || capabilities || phaseIndex || storageEffects ||
                         recognition || arithmetic || aliasChecks || roundtrip || regionChecks;
  const auto filename = argv[hasOption ? 2 : 1];
  auto module = parseSourceFile<ModuleOp>(filename, &context);
  if (!module || failed(verify(*module))) {
    return 1;
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
  if (before != render(module->getOperation())) {
    llvm::errs() << "shared extraction mutated source IR\n";
    return 1;
  }
  llvm::outs() << "source-unchanged; dependency-analysis-not-run\n";
  return 0;
}
