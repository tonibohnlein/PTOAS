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
#include "PTO/Transforms/InsertSync/SyncSlotMapping.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <optional>

namespace mlir::pto::oahs::native_detail {
// Qualification depends only on original scalar/control semantics, never on
// payload names or effects. Use the same constant and counted-domain proofs
// as physical occurrence analysis.
inline bool firstUseTerms(Value condition, DenseMap<Value, uint64_t> &terms,
                          SyncSlotMapping::ConstantCache &constants) {
  if (auto conjunction = condition.getDefiningOp<arith::AndIOp>()) {
    return firstUseTerms(conjunction.getLhs(), terms, constants) &&
           firstUseTerms(conjunction.getRhs(), terms, constants);
  }
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  const bool equality = cmp && cmp.getPredicate() == arith::CmpIPredicate::eq;
  if (!equality) {
    return false;
  }
  const auto lhs = SyncSlotMapping::evaluateConstant(cmp.getLhs(), constants);
  const auto rhs = SyncSlotMapping::evaluateConstant(cmp.getRhs(), constants);
  const bool oneConstant = lhs.has_value() != rhs.has_value();
  if (!oneConstant) {
    return false;
  }
  const auto iv = lhs ? cmp.getRhs() : cmp.getLhs();
  const auto literal = lhs ? *lhs : *rhs;
  return terms.try_emplace(iv, literal).second;
}

inline SmallVector<scf::ForOp> firstUseLoops(scf::IfOp choice,
    SyncSlotMapping::ConstantCache &constants, SyncSlotMapping::RangeCache &ranges,
    DenseMap<mlir::Operation *, std::optional<SyncSlotMapping::LoopDomain>> &domains) {
  DenseMap<Value, uint64_t> terms;
  if (!firstUseTerms(choice.getCondition(), terms, constants)) {
    return {};
  }
  SmallVector<scf::ForOp> loops;
  for (auto *parent = choice->getParentOp(); parent && !terms.empty(); parent = parent->getParentOp()) {
    if (isa<scf::WhileOp>(parent)) {
      return {};
    }
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop) {
      continue;
    }
    const auto found = terms.find(loop.getInductionVar());
    if (found == terms.end()) { return {}; }
    auto domain = domains.find(loop.getOperation());
    if (domain == domains.end()) {
      domain = domains.try_emplace(loop.getOperation(),
          SyncSlotMapping::originalLoopDomain(loop, constants, ranges)).first;
    }
    const bool unsupported = !domain->second || found->second != domain->second->lower;
    if (unsupported) {
      return {};
    }
    terms.erase(found);
    loops.push_back(loop);
  }
  return terms.empty() ? loops : SmallVector<scf::ForOp>{};
}

inline void importFirstUse(func::FuncOp function, Program &program,
                          const DenseMap<mlir::Operation *, std::size_t> &ids,
                          std::vector<std::string> &notes) {
  SyncSlotMapping::ConstantCache constants;
  SyncSlotMapping::RangeCache ranges;
  DenseMap<mlir::Operation *, std::optional<SyncSlotMapping::LoopDomain>> domains;
  function.walk([&](scf::IfOp choice) {
    auto loops = firstUseLoops(choice, constants, ranges, domains);
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
