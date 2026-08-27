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

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOCANONICALSYNC
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

constexpr std::int64_t kHardwareEventIdCount = 8;

bool isKernelDispatchWrapper(func::FuncOp function) {
  const bool isEntry = function->hasAttr("pto.entry");
  const bool hasSingleBlock = function.getBody().hasOneBlock();
  if (!isEntry || !hasSingleBlock) {
    return false;
  }
  bool hasDispatch = false;
  for (Operation &operation : function.getBody().front().without_terminator()) {
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

bool shouldSkip(func::FuncOp function) {
  return function.isExternal() || function->hasAttr("pto.tileop.helper") ||
         function->hasAttr("pto.ptodsl.subkernel_helper") ||
         isKernelDispatchWrapper(function);
}

struct PTOCanonicalSyncPass
    : public pto::impl::PTOCanonicalSyncBase<PTOCanonicalSyncPass> {
  using pto::impl::PTOCanonicalSyncBase<
      PTOCanonicalSyncPass>::PTOCanonicalSyncBase;

  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (assumeDistinctGmArgsNoAlias && assumeAllGmAccessesNoAlias) {
      function.emitError(
          "--assume-distinct-gm-args-noalias and "
          "--assume-all-gm-accesses-noalias are mutually exclusive");
      signalPassFailure();
      return;
    }
    if (shouldSkip(function)) {
      return;
    }
    if (eventIdNumMax <= 0 || eventIdNumMax > kHardwareEventIdCount) {
      function.emitError() << "event-id-num-max must be in [1, "
                           << kHardwareEventIdCount << ']';
      signalPassFailure();
      return;
    }
    pto::CanonicalSyncBuildOptions options;
    options.eventIdBudget = static_cast<unsigned>(eventIdNumMax);
    options.analysis.gmAliasPolicy =
        assumeAllGmAccessesNoAlias
            ? pto::CanonicalSyncGmAliasPolicy::AllAccessesNoAlias
        : assumeDistinctGmArgsNoAlias
            ? pto::CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias
            : pto::CanonicalSyncGmAliasPolicy::MayAlias;
    if (failed(pto::runCanonicalSync(function, options))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::pto::createPTOCanonicalSyncPass(const PTOCanonicalSyncOptions &options) {
  return std::make_unique<PTOCanonicalSyncPass>(options);
}
