// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect the shared instruction contract without any dependency analyzer.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace {
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
  const bool expectFailure = argc == 3 && StringRef(argv[1]) == "--expect-failure";
  const bool capabilities = argc == 3 && StringRef(argv[1]) == "--capabilities";
  if (argc != 2 && !expectFailure && !capabilities) {
    llvm::errs() << "usage: pto-sync-input-test [--expect-failure|--capabilities] input.pto\n";
    return 1;
  }
  DialectRegistry dialects;
  dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
  MLIRContext context(dialects);
  context.disableMultithreading();
  auto module = parseSourceFile<ModuleOp>(argv[expectFailure || capabilities ? 2 : 1], &context);
  if (!module || failed(verify(*module))) {
    return 1;
  }
  if (capabilities) {
    dumpCapabilities();
  }
  const auto before = render(module->getOperation());
  pto::SyncInput input;
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
