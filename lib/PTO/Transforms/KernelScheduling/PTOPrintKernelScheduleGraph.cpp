// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/KernelScheduleGraph.h"
#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PRINTKERNELSCHEDULEGRAPH
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PrintKernelScheduleGraphPass
    : public pto::impl::PrintKernelScheduleGraphBase<
          PrintKernelScheduleGraphPass> {
  using pto::impl::PrintKernelScheduleGraphBase<
      PrintKernelScheduleGraphPass>::PrintKernelScheduleGraphBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal()) {
      return;
    }
    if (format != "text" && format != "dot") {
      func.emitError() << "unsupported kernel schedule graph format '" << format
                       << "'; expected 'text' or 'dot'";
      signalPassFailure();
      return;
    }
    if (requireExactDurations && durationTable.empty()) {
      func.emitError("--require-exact-durations requires --duration-table");
      signalPassFailure();
      return;
    }

    std::optional<pto::PTOISADurationTable> durationTableData;
    if (!durationTable.empty()) {
      FailureOr<pto::PTOISADurationTable> loaded =
          pto::PTOISADurationTable::loadFromFile(durationTable);
      if (failed(loaded)) {
        func.emitError() << "failed to load PTO-ISA duration table '"
                         << durationTable << "'";
        signalPassFailure();
        return;
      }
      durationTableData = std::move(*loaded);
    }
    pto::KernelScheduleGraphBuildOptions options;
    options.durationTable = durationTableData ? &*durationTableData : nullptr;
    options.requireExactDurations = requireExactDurations;
    FailureOr<pto::KernelScheduleGraph> graph =
        pto::buildKernelScheduleGraph(func, options);
    if (failed(graph)) {
      signalPassFailure();
      return;
    }
    if (format == "dot") {
      pto::printKernelScheduleGraphDot(llvm::outs(), func, *graph);
    } else {
      pto::printKernelScheduleGraph(llvm::outs(), func, *graph);
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPrintKernelScheduleGraphPass() {
  return std::make_unique<PrintKernelScheduleGraphPass>();
}
