// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

using namespace mlir;
using namespace mlir::pto;
int main() {
  MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<func::FuncDialect, arith::ArithDialect>();
  auto type = FunctionType::get(&context, {IndexType::get(&context), IndexType::get(&context)}, {});
  OwningOpRef<func::FuncOp> function = func::FuncOp::create(UnknownLoc::get(&context), "alias", type);
  function->addEntryBlock();
  const auto root = function->getArgument(0), other = function->getArgument(1);
  const auto maximum = std::numeric_limits<uint64_t>::max();
  BaseMemInfo a(root, root, AddressSpace::GM, {maximum - 3}, 8, false, false);
  BaseMemInfo b(root, root, AddressSpace::GM, {maximum - 1}, 1, false, false);
  MemoryDependentAnalyzer aliases;
  unsigned checked = 0;
  auto check = [&](bool condition, const char *name) {
    ++checked;
    if (!condition) { llvm::errs() << "alias overflow regression: " << name << "\n"; }
    return condition;
  };
  bool passed = check(!MemoryDependentAnalyzer::storageCoordinates(a), "overflow has no coordinate certificate");
  passed &= check(aliases.MemAlias(&a, &b) && aliases.MemAlias(&b, &a), "GM overflow is conservative");
  b.rootBuffer = other;
  passed &= check(aliases.MemAlias(&a, &b), "overflow precedes different-root disjointness");
  b.scope = AddressSpace::VEC;
  passed &= check(!aliases.MemAlias(&a, &b), "different physical domains remain disjoint");
  a.scope = b.scope = AddressSpace::GM;
  a.baseAddresses = {0};
  a.allocateSize = 16;
  b.baseAddresses = {8};
  b.allocateSize = 16;
  passed &= check(!aliases.MemAlias(&a, &b), "ordinary distinct GM roots retain compatibility contract");
  b.rootBuffer = root;
  passed &= check(aliases.MemAlias(&a, &b), "ordinary same-root overlap");
  b.baseAddresses = {16};
  passed &= check(!aliases.MemAlias(&a, &b), "ordinary adjacent intervals");
  a.baseAddresses = {maximum - 4};
  a.allocateSize = 4;
  b.baseAddresses = {maximum - 1};
  b.allocateSize = 1;
  passed &= check(bool(MemoryDependentAnalyzer::storageCoordinates(a)) && aliases.MemAlias(&a, &b),
                  "exact maximum endpoint is representable");
  a.scope = b.scope = AddressSpace::VEC;
  a.hasKnownPhysicalAddresses = b.hasKnownPhysicalAddresses = true;
  a.allocateSize = 8;
  passed &= check(aliases.MemAlias(&a, &b), "known local overflow is conservative");
  // The legacy cross-root path derives absolute starts from integer roots.
  // Representable relative offsets alone do not certify those absolute ends.
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(&function->front());
  auto high = builder.create<arith::ConstantIntOp>(function->getLoc(), -4, 64);
  auto nearby = builder.create<arith::ConstantIntOp>(function->getLoc(), -3, 64);
  a.rootBuffer = high;
  b.rootBuffer = nearby;
  a.hasKnownPhysicalAddresses = b.hasKnownPhysicalAddresses = false;
  a.baseAddresses = {1};
  a.allocateSize = 4;
  b.baseAddresses = {0};
  b.allocateSize = 1;
  passed &= check(aliases.MemAlias(&a, &b) && aliases.MemAlias(&b, &a),
                  "derived absolute endpoint overflow is conservative");
  a.baseAddresses = {8};
  a.allocateSize = 1;
  passed &= check(aliases.MemAlias(&a, &b) && aliases.MemAlias(&b, &a),
                  "derived absolute start overflow is conservative");
  a.baseAddresses = {1};
  passed &= check(aliases.MemAlias(&a, &b), "representable derived absolute overlap");
  a.baseAddresses = {0};
  passed &= check(!aliases.MemAlias(&a, &b), "representable derived absolute adjacency");
  llvm::outs() << "shared alias overflow checks=" << checked << " passed=" << passed << "\n";
  return passed ? 0 : 1;
}
