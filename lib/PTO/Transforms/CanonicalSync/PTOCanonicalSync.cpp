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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

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

std::optional<pto::CanonicalSyncSolver> parseCanonicalSyncSolver(
    StringRef solver) {
  if (solver == "legacy") {
    return pto::CanonicalSyncSolver::Legacy;
  }
  if (solver == "covering") {
    return pto::CanonicalSyncSolver::Covering;
  }
  return std::nullopt;
}

bool isValidView(StringRef view) {
  return view == "all" || view == "dependencies" || view == "plan" ||
         view == "events" || view == "ownership" || view == "selection" ||
         view == "covering";
}

bool isKernelDispatchWrapper(func::FuncOp func) {
  if (!func->hasAttr("pto.entry") || !func.getBody().hasOneBlock()) {
    return false;
  }

  bool hasDispatch = false;
  for (Operation &operation : func.getBody().front().without_terminator()) {
    auto call = dyn_cast<func::CallOp>(operation);
    if (!call) {
      return false;
    }
    func::FuncOp callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee || !callee->hasAttr("pto.kernel_kind")) {
      return false;
    }
    hasDispatch = true;
  }
  return hasDispatch;
}

bool shouldSkipCanonicalSync(func::FuncOp func) {
  return func.isExternal() || func->hasAttr("pto.tileop.helper") ||
         func->hasAttr("pto.ptodsl.subkernel_helper") ||
         isKernelDispatchWrapper(func);
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
    const auto gmAliasPolicy =
        assumeDistinctGmArgsNoAlias
            ? pto::CanonicalGMAliasPolicy::DistinctArgumentsNoAlias
            : pto::CanonicalGMAliasPolicy::MayAlias;
    const std::optional<pto::CanonicalSyncSolver> selectedSolver =
        parseCanonicalSyncSolver(solver);
    if (!selectedSolver) {
      func.emitError() << "solver must be 'legacy' or 'covering'";
      signalPassFailure();
      return;
    }
    pto::CanonicalSyncBuildOptions options;
    options.eventIdMax = static_cast<unsigned>(eventIdNumMax);
    options.gmAliasPolicy = gmAliasPolicy;
    options.solver = *selectedSolver;
    options.coveringShadow = coveringShadow;
    FailureOr<pto::CanonicalSyncPlan> plan =
        pto::buildCanonicalSyncPlan(func, options);
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
    const bool hasEvictionRequest = !evictCanonicalBarrierIds.empty() ||
                                    !evictCanonicalEventBundleIds.empty();
    const bool invalidSelectionFormat =
        view == "selection" && format != "text";
    const bool invalidCoveringFormat =
        view == "covering" && format != "text";
    const bool evictionWithoutSelectionView =
        hasEvictionRequest && view != "selection";
    if (invalidSelectionFormat || evictionWithoutSelectionView) {
      module.emitError()
          << "canonical sync eviction diagnostics require format='text' "
             "and view='selection'";
      signalPassFailure();
      return;
    }
    if (invalidCoveringFormat) {
      module.emitError()
          << "canonical sync covering diagnostics require format='text'";
      signalPassFailure();
      return;
    }
    if (eventIdNumMax <= 0 || eventIdNumMax > kHardwareEventIdCount) {
      module.emitError() << "event-id-num-max must be in [1, "
                         << kHardwareEventIdCount << ']';
      signalPassFailure();
      return;
    }
    const auto gmAliasPolicy =
        assumeDistinctGmArgsNoAlias
            ? pto::CanonicalGMAliasPolicy::DistinctArgumentsNoAlias
            : pto::CanonicalGMAliasPolicy::MayAlias;
    const std::optional<pto::CanonicalSyncSolver> selectedSolver =
        parseCanonicalSyncSolver(solver);
    if (!selectedSolver) {
      module.emitError() << "solver must be 'legacy' or 'covering'";
      signalPassFailure();
      return;
    }
    pto::CanonicalSelectionDiagnosticRequest diagnosticRequest;
    diagnosticRequest.barrierIds.append(evictCanonicalBarrierIds.begin(),
                                        evictCanonicalBarrierIds.end());
    diagnosticRequest.eventBundleIds.append(
        evictCanonicalEventBundleIds.begin(),
        evictCanonicalEventBundleIds.end());
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (shouldSkipCanonicalSync(func)) {
        continue;
      }
      pto::CanonicalSyncBuildOptions options;
      options.eventIdMax = static_cast<unsigned>(eventIdNumMax);
      options.gmAliasPolicy = gmAliasPolicy;
      options.solver = *selectedSolver;
      options.diagnosticRequest =
          view == "selection" ? &diagnosticRequest : nullptr;
      options.coveringShadow = coveringShadow || view == "covering";
      options.coveringMembershipProbe = view == "covering";
      FailureOr<pto::CanonicalSyncPlan> plan =
          pto::buildCanonicalSyncPlan(func, options);
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
