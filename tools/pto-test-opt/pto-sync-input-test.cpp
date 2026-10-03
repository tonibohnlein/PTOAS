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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace {
std::string render(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  return text;
}
}
int main(int argc, char **argv) {
  const bool expectFailure = argc == 3 && StringRef(argv[1]) == "--expect-failure";
  if (argc != 2 && !expectFailure) {
    llvm::errs() << "usage: pto-sync-input-test [--expect-failure] input.pto\n";
    return 1;
  }
  DialectRegistry dialects;
  dialects.insert<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
  MLIRContext context(dialects);
  context.disableMultithreading();
  auto module = parseSourceFile<ModuleOp>(argv[expectFailure ? 2 : 1], &context);
  if (!module || failed(verify(*module))) {
    return 1;
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
