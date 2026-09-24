// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Corpus runner for the existing Phase A query boundary. Success means that
// preparation and queries ran, not that unresolved facts or Phase B are complete.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <string>

namespace {
namespace fs = mlir::pto::frontiersynch;

std::string printIR(mlir::Operation *operation) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  operation->print(stream);
  return result;
}

struct Counts {
  std::array<std::size_t, 3> hazards{};
  std::size_t interpreted = 0;
  std::size_t unknownOccurrence = 0;
  std::size_t decodedUnresolved = 0;
  std::size_t readers = 0;
};

mlir::LogicalResult inspectRequests(const fs::ProgramAnalysis &analysis, Counts &counts) {
  for (std::size_t site = 0; site < analysis.structure().operations.size(); ++site) {
    const auto &requests = analysis.requirementsAt(site);
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const auto &request = requests[index];
      if (request.source.operation >= analysis.structure().operations.size()) {
        return mlir::failure();
      }
      bool subscribed = false;
      for (const auto &subscription : analysis.subscriptionsAt(request.source.operation)) {
        subscribed |= subscription.deadlineOperation == site && subscription.requirementIndex == index;
      }
      if (!subscribed) {
        return mlir::failure();
      }
      const auto &answer = analysis.interpretAt(site, index);
      ++counts.interpreted;
      ++counts.hazards.at(static_cast<std::size_t>(request.relationship.kind));
      counts.unknownOccurrence += answer.occurrence.status == fs::OriginalOccurrenceInterpretation::Status::Unknown;
      counts.decodedUnresolved += !answer.decoded->unresolved.empty();
      counts.readers += answer.decoded->hasReaderFrontier;
    }
  }
  return mlir::success();
}

mlir::LogicalResult analyze(mlir::func::FuncOp function) {
  mlir::pto::SyncInput input;
  if (mlir::failed(input.build(function))) {
    return mlir::failure();
  }
  fs::OriginalStructure original;
  if (mlir::failed(fs::importOriginalStructure(function, input, original))) {
    return mlir::failure();
  }
  fs::ProgramAnalysis analysis(input, std::move(original));
  if (!analysis.complete()) {
    return function.emitError(analysis.reason());
  }
  Counts counts;
  if (mlir::failed(inspectRequests(analysis, counts))) {
    return function.emitError("missing original source subscription");
  }
  const auto &structure = analysis.structure();
  llvm::outs() << function.getName() << ": phases=" << structure.operations.size()
               << " cells=" << structure.cells.size() << " bank-relations=" << structure.physicalAddresses.size()
               << "\n  RAW=" << counts.hazards[0] << " WAR=" << counts.hazards[1] << " WAW=" << counts.hazards[2]
               << " interpreted=" << counts.interpreted << " readers=" << counts.readers
               << "\n  unknown-occurrence=" << counts.unknownOccurrence
               << " decoded-unresolved=" << counts.decodedUnresolved << "\n";
  return mlir::success();
}

mlir::LogicalResult runFile(llvm::StringRef path, mlir::MLIRContext &context, bool verifyOnly) {
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
  if (!module || mlir::failed(mlir::verify(*module))) {
    return mlir::failure();
  }
  if (verifyOnly) {
    std::size_t localSync = 0, crossSync = 0;
    module->walk([&](mlir::Operation *operation) {
      const auto name = operation->getName().getStringRef();
      localSync += name == "pto.set_flag_dyn" || name == "pto.wait_flag_dyn";
      crossSync += name == "pto.sync.set" || name == "pto.sync.wait";
    });
    llvm::outs() << "verified: local-sync=" << localSync << " cross-sync=" << crossSync << "\n";
    return mlir::success();
  }
  const auto before = printIR(module->getOperation());
  for (auto function : module->getOps<mlir::func::FuncOp>()) {
    if (mlir::failed(analyze(function))) {
      return mlir::failure();
    }
  }
  if (before != printIR(module->getOperation())) {
    return module->emitError("Phase A changed the original IR");
  }
  llvm::outs() << "original-ir-unchanged; construction-not-run\n";
  return mlir::success();
}
} // namespace

int main(int argc, char **argv) {
  const bool verifyOnly = argc == 3 && llvm::StringRef(argv[1]) == "--verify-only";
  if (argc != 2 && !verifyOnly) {
    llvm::errs() << "usage: pto-frontier-analysis-test [--verify-only] input.pto\n";
    return 1;
  }
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
  mlir::MLIRContext context(dialects);
  context.disableMultithreading();
  return mlir::failed(runFile(argv[verifyOnly ? 2 : 1], context, verifyOnly));
}
