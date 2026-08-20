// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSync.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOCANONICALSYNC
#define GEN_PASS_DEF_PRINTCANONICALSYNCPLAN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

constexpr std::int64_t kHardwareEventIdCount = 8;

bool isValidView(StringRef view) {
  return view == "all" || view == "dependencies" || view == "plan" ||
         view == "events";
}

bool shouldSkipCanonicalSync(func::FuncOp func) {
  return func.isExternal() || func->hasAttr("pto.tileop.helper") ||
         func->hasAttr("pto.ptodsl.subkernel_helper");
}

struct PTOCanonicalSyncPass
    : public pto::impl::PTOCanonicalSyncBase<PTOCanonicalSyncPass> {
  using pto::impl::PTOCanonicalSyncBase<
      PTOCanonicalSyncPass>::PTOCanonicalSyncBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (shouldSkipCanonicalSync(func)) {
      return;
    }
    if (eventIdNumMax <= 0 || eventIdNumMax > kHardwareEventIdCount) {
      func.emitError() << "event-id-num-max must be in [1, "
                       << kHardwareEventIdCount << ']';
      signalPassFailure();
      return;
    }
    FailureOr<pto::CanonicalSyncPlan> plan =
        pto::buildCanonicalSyncPlan(func, static_cast<unsigned>(eventIdNumMax));
    if (failed(plan) || failed(pto::emitCanonicalSyncPlan(func, *plan))) {
      signalPassFailure();
    }
  }
};

struct PrintCanonicalSyncPlanPass
    : public pto::impl::PrintCanonicalSyncPlanBase<PrintCanonicalSyncPlanPass> {
  using pto::impl::PrintCanonicalSyncPlanBase<
      PrintCanonicalSyncPlanPass>::PrintCanonicalSyncPlanBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if ((format != "text" && format != "dot") || !isValidView(view)) {
      module.emitError()
          << "unsupported canonical sync print selection: format='" << format
          << "', view='" << view << "'";
      signalPassFailure();
      return;
    }
    if (eventIdNumMax <= 0 || eventIdNumMax > kHardwareEventIdCount) {
      module.emitError() << "event-id-num-max must be in [1, "
                         << kHardwareEventIdCount << ']';
      signalPassFailure();
      return;
    }
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (shouldSkipCanonicalSync(func)) {
        continue;
      }
      FailureOr<pto::CanonicalSyncPlan> plan = pto::buildCanonicalSyncPlan(
          func, static_cast<unsigned>(eventIdNumMax));
      if (failed(plan)) {
        signalPassFailure();
        return;
      }
      if (format == "dot") {
        pto::printCanonicalSyncPlanDot(llvm::outs(), func, *plan, view);
      } else {
        pto::printCanonicalSyncPlan(llvm::outs(), func, *plan, view);
      }
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOCanonicalSyncPass(const PTOCanonicalSyncOptions &options) {
  return std::make_unique<PTOCanonicalSyncPass>(options);
}

std::unique_ptr<Pass> mlir::pto::createPrintCanonicalSyncPlanPass() {
  return std::make_unique<PrintCanonicalSyncPlanPass>();
}
