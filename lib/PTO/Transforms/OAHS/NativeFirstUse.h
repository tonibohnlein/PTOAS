// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_OAHS_NATIVE_FIRST_USE_H
#define PTO_OAHS_NATIVE_FIRST_USE_H
#include "SelectedInternal.h"
#include "PTO/Transforms/OAHS/ObservedPrograms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

namespace mlir::pto::oahs::native_detail {
// Qualification depends only on original scalar/control semantics, never on
// payload names or effects. Native import is the owner of this proof boundary.
inline std::optional<int64_t> firstUseInteger(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant) {
    return {};
  }
  auto attr = dyn_cast<IntegerAttr>(constant.getValue());
  const bool supported = attr && attr.getValue().getBitWidth() <= 64;
  if (!supported) {
    return {};
  }
  return attr.getInt();
}

inline bool firstUseTerms(Value condition, SmallVectorImpl<Value> &ivs) {
  if (auto conjunction = condition.getDefiningOp<arith::AndIOp>()) {
    return firstUseTerms(conjunction.getLhs(), ivs) && firstUseTerms(conjunction.getRhs(), ivs);
  }
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  const bool equality = cmp && cmp.getPredicate() == arith::CmpIPredicate::eq;
  if (!equality) {
    return false;
  }
  Value iv = cmp.getLhs(), literal = cmp.getRhs();
  if (firstUseInteger(iv)) {
    std::swap(iv, literal);
  }
  const bool first = firstUseInteger(literal) == std::optional<int64_t>(0);
  if (!first || llvm::is_contained(ivs, iv)) {
    return false;
  }
  ivs.push_back(iv);
  return true;
}

inline SmallVector<scf::ForOp> firstUseLoops(scf::IfOp choice) {
  SmallVector<Value> ivs;
  if (!firstUseTerms(choice.getCondition(), ivs)) {
    return {};
  }
  SmallVector<scf::ForOp> loops;
  for (auto *parent = choice->getParentOp(); parent && !ivs.empty(); parent = parent->getParentOp()) {
    if (isa<scf::WhileOp>(parent)) {
      return {};
    }
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    const auto found = llvm::find(ivs, loop.getInductionVar());
    const bool unsupported = found == ivs.end() || !isa<IndexType>(loop.getInductionVar().getType()) ||
        loop->hasAttr("unsignedCmp") || loop->hasAttr("unsigned_cmp") ||
        firstUseInteger(loop.getLowerBound()) != std::optional<int64_t>(0) ||
        firstUseInteger(loop.getStep()) != std::optional<int64_t>(1);
    if (unsupported) {
      return {};
    }
    ivs.erase(found);
    loops.push_back(loop);
  }
  return ivs.empty() ? loops : SmallVector<scf::ForOp>{};
}

inline void importFirstUse(func::FuncOp function, Program &program,
                          const DenseMap<mlir::Operation *, std::size_t> &ids,
                          std::vector<std::string> &notes) {
  function.walk([&](scf::IfOp choice) {
    auto loops = firstUseLoops(choice);
    if (loops.empty()) {
      return;
    }
    FirstUseRegion region;
    region.entry = ids.lookup(loops.back().getOperation());
    region.exit = ids.lookup(loops.back()->getNextNode());
    for (auto loop : loops) {
      region.backedgeOwners.push_back(ids.lookup(loop.getOperation()));
    }
    const auto &q = *program.observed;
    const auto anchor = ids.lookup(choice.getOperation());
    for (std::size_t site = 0; site < q.sites.size(); ++site) {
      const auto observation = q.sites[site].observation;
      if (observation != NoControlId && q.observations[observation].anchor == anchor &&
          q.sites[site].successors.size() == 2) {
        region.decisions.push_back(site);
      }
    }
    auto refined = refineFirstUse(program, region);
    // Prefix splitting can expose multiple entries to a remaining SCC. Keep
    // the conservative original graph when the constructor cannot represent
    // that interface; optional control qualification must not introduce a
    // construction refusal for an otherwise supported input.
    if (refined.success) {
      const selected::Control control(refined.program);
      if (!control.complete) {
        refined.success = false;
        refined.reason = control.reason;
      }
    }
    if (refined.success) {
      program = std::move(refined.program);
      notes.push_back("qualified original first-use region prefix");
    } else {
      notes.push_back("kept original first-use control: " + refined.reason);
    }
  });
}
} // namespace mlir::pto::oahs::native_detail
#endif
