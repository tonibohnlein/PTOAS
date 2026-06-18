// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOUnrollSIMTForPass.cpp -------------------------------------------===//
//
// Unroll small constant-trip-count scf.for loops inside pto.simt_entry
// functions to eliminate divergent control flow before LLVM lowering.
//
// Workaround for: https://github.com/mouliangyu/PTOAS/issues/485
//
// When a SIMTVF kernel contains scf.for + scf.if with a constant-condition
// branch (e.g. for i in 0..16: if i < 10: store), the AICore backend may
// emit a predicated END instruction.  By fully unrolling the loop and
// letting canonicalize/sccp constant-fold the induction-variable-dependent
// conditions, we obtain straight-line code without branches, which avoids
// the bug.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOUNROLLSIMTFOR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "pto-unroll-simt-for"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Name of the unroll annotation placed on scf::ForOp by users.
static constexpr llvm::StringLiteral kUnrollAttrName = "pto.unroll";
static constexpr llvm::StringLiteral kUnrollFullValue = "full";

/// Try to compute a constant trip count for an scf::ForOp.
/// Returns std::nullopt if any bound or step is non-constant.
static std::optional<int64_t> computeConstantTripCount(scf::ForOp forOp) {
  std::optional<int64_t> lb = getConstantIntValue(forOp.getLowerBound());
  std::optional<int64_t> ub = getConstantIntValue(forOp.getUpperBound());
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0)
    return std::nullopt;
  if (*ub <= *lb)
    return 0;
  return (*ub - *lb + *step - 1) / *step; // ceil division
}

/// Check whether the loop has the explicit "full unroll" annotation.
static bool hasUnrollFullAttr(scf::ForOp forOp) {
  if (auto attr = forOp->getAttrOfType<StringAttr>(kUnrollAttrName))
    return attr.getValue() == kUnrollFullValue;
  return false;
}

/// Check whether this function is a SIMT entry.
static bool isSIMTEntry(func::FuncOp func) {
  return func->hasAttr(pto::kPTOSimtEntryAttrName);
}

// ---------------------------------------------------------------------------
// Rewrite pattern
// ---------------------------------------------------------------------------

namespace {

struct UnrollSIMTForPattern : public OpRewritePattern<scf::ForOp> {
  UnrollSIMTForPattern(MLIRContext *ctx, int64_t maxTripCount)
      : OpRewritePattern(ctx), maxTripCount(maxTripCount) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Only apply inside SIMT entry functions.
    auto func = forOp->getParentOfType<func::FuncOp>();
    if (!func || !isSIMTEntry(func))
      return failure();

    // Check explicit annotation first.
    if (hasUnrollFullAttr(forOp)) {
      return unrollFull(forOp, rewriter, /*isAnnotated=*/true);
    }

    // Auto-detect: constant bounds and small trip count.
    auto tripCount = computeConstantTripCount(forOp);
    if (!tripCount || *tripCount > maxTripCount)
      return failure();

    return unrollFull(forOp, rewriter, /*isAnnotated=*/false);
  }

private:
  LogicalResult unrollFull(scf::ForOp forOp, PatternRewriter &rewriter,
                           bool isAnnotated) const {
    auto tripCount = computeConstantTripCount(forOp);
    if (!tripCount || *tripCount <= 0)
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "PTOUnrollSIMTFor: unrolling scf.for "
               << (isAnnotated ? "(annotated)" : "(auto)") << " tripCount="
               << *tripCount << " at " << forOp.getLoc() << "\n");

    // loopUnrollByFactor returns failure if the loop carries iteration
    // arguments that have uses outside the loop (live-out values).  In that
    // case we cannot unroll.
    if (failed(loopUnrollByFactor(forOp, static_cast<uint64_t>(*tripCount))))
      return failure();

    return success();
  }

  int64_t maxTripCount;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass definition
// ---------------------------------------------------------------------------

namespace {

struct PTOUnrollSIMTFor : public pto::impl::PTOUnrollSIMTForBase<PTOUnrollSIMTFor> {
  using pto::impl::PTOUnrollSIMTForBase<
      PTOUnrollSIMTFor>::PTOUnrollSIMTForBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (!isSIMTEntry(func))
      return;

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<UnrollSIMTForPattern>(ctx, maxTripCount);

    GreedyRewriteConfig config;
    config.maxIterations = 10; // loops may nest
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

// ---------------------------------------------------------------------------
// Pass constructor
// ---------------------------------------------------------------------------

std::unique_ptr<Pass> mlir::pto::createPTOUnrollSIMTForPass() {
  return std::make_unique<PTOUnrollSIMTFor>();
}
