// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOCANONICALSYNC
#include "PTO/Transforms/Passes.h.inc"

namespace {

struct PTOCanonicalSyncPass
    : public impl::PTOCanonicalSyncBase<PTOCanonicalSyncPass> {
  PTOCanonicalSyncPass() = default;

  explicit PTOCanonicalSyncPass(const CanonicalSyncOptions &options) {
    analysisOnly = options.analysisOnly;
    dump = options.dump || options.analysisOnly;
  }

  void runOnOperation() override {
    CanonicalSyncOptions options;
    options.analysisOnly = analysisOnly;
    options.dump = dump || analysisOnly;
    if (failed(runCanonicalSync(getOperation(), options))) {
      signalPassFailure();
    }
  }
};

} // namespace

FailureOr<std::unique_ptr<CanonicalSyncProgram>>
buildCanonicalSyncProgram(func::FuncOp function) {
  FailureOr<CanonicalSyncTarget> target =
      CanonicalSyncTarget::resolve(function);
  if (failed(target)) {
    return failure();
  }
  if (failed(canonical_sync_detail::rejectUnsupportedCanonicalSyncInput(
          function))) {
    return failure();
  }
  auto program = std::make_unique<CanonicalSyncProgram>(function);
  if (failed(canonical_sync_detail::buildCanonicalStructureAndAccesses(
          *program, *target)) ||
      failed(
          canonical_sync_detail::deriveCanonicalDemands(*program, *target)) ||
      failed(program->freezeGraph())) {
    return failure();
  }
  return std::move(program);
}

LogicalResult runCanonicalSync(func::FuncOp function,
                               const CanonicalSyncOptions &options) {
  // External declarations contain no scheduled physical work.  The function
  // pass is still invoked for them when a generated module contains private
  // runtime adapters, so leave them unchanged instead of asking the
  // structured-program builder to manufacture a body.
  if (function.isDeclaration()) {
    return success();
  }
  FailureOr<std::unique_ptr<CanonicalSyncProgram>> program =
      buildCanonicalSyncProgram(function);
  if (failed(program)) {
    return failure();
  }
  if (failed(buildCanonicalDirectMechanisms(**program))) {
    return failure();
  }
  if (failed(evaluateCanonicalSyncCoverage(**program))) {
    return failure();
  }
  if (failed(buildCanonicalSyncSetCoverInstance(**program))) {
    return failure();
  }
  if (failed(solveCanonicalSyncSetCover(**program))) {
    return failure();
  }
  if (failed(allocateCanonicalSyncEvents(**program))) {
    return failure();
  }
  if (failed((*program)->freeze())) {
    return failure();
  }
  if (options.dump || options.analysisOnly) {
    printCanonicalSyncProgram(**program, llvm::errs());
  }
  if (options.analysisOnly) {
    llvm::errs() << "VERIFY skipped (analysis-only; IR unchanged)\n";
    return success();
  }
  if (failed(materializeAndVerifyCanonicalSync(**program))) {
    return failure();
  }
  if (options.dump) {
    llvm::errs() << "VERIFY ok\n";
  }
  return success();
}

std::unique_ptr<Pass>
createPTOCanonicalSyncPass(const CanonicalSyncOptions &options) {
  return std::make_unique<PTOCanonicalSyncPass>(options);
}

} // namespace pto
} // namespace mlir
