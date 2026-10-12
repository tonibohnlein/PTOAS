// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check the shared alias contract against finite byte intersections.
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <limits>
using namespace mlir;
using namespace mlir::pto;

namespace {
bool checkAliasSelectChains(func::FuncOp original) {
  original.getContext()->getOrLoadDialect<arith::ArithDialect>();
  for (unsigned roots : {32U, 64U, 128U, 256U}) {
    OpBuilder builder(original.getContext());
    SmallVector<Type> arguments(roots, original.getArgument(0).getType());
    arguments.push_back(builder.getI1Type());
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(original.getLoc(),
        "alias_select_chain", builder.getFunctionType(arguments, {})));
    auto *body = function->addEntryBlock();
    builder.setInsertionPointToStart(body);
    Value current = body->getArgument(0);
    SmallVector<Value> results;
    for (unsigned i = 1; i < roots; ++i) {
      current = builder.create<arith::SelectOp>(original.getLoc(), body->getArgument(roots),
                                               current, body->getArgument(i)).getResult();
      results.push_back(current);
      // Both arms carry identical root sets. They must deduplicate without
      // changing first-seen order, even after a growing-prefix union.
      current = builder.create<arith::SelectOp>(original.getLoc(), body->getArgument(roots),
                                               current, current).getResult();
      results.push_back(current);
    }
    builder.create<func::ReturnOp>(original.getLoc());
    SyncIRs ir;
    Buffer2MemInfoMap memories;
    OpAccessMemInfos opAccessMemInfos;
    PTOIRTranslator translator(ir, memories, opAccessMemInfos, *function, SyncAnalysisMode::NORMALSYNC);
    if (failed(translator.Build())) { return false; }
    for (unsigned index = 0; index < results.size(); ++index) {
      auto found = memories.find(results[index]);
      const unsigned count = index / 2 + 2;
      if (found == memories.end() || found->second.size() != count) { return false; }
      for (unsigned root = 0; root < count; ++root) {
        const auto& info = found->second[root];
        if (info->rootBuffer != body->getArgument(root) || info->baseBuffer != results[index]) { return false; }
      }
    }
    const auto& stats = translator.aliasConstructionStats();
    const uint64_t prefixes = static_cast<uint64_t>(roots) * (roots + 1) - 2;
    if (stats.retainedClones != prefixes || stats.candidates != 3 * prefixes / 2 ||
        stats.indexedRecords != static_cast<uint64_t>(roots) * roots - 1 ||
        stats.equalityChecks > 16 * prefixes || stats.hashCalls > 24 * prefixes) { return false; }
    llvm::outs() << "alias-select-chain: roots=" << roots << " retained=" << stats.retainedClones
                 << " equality-checks=" << stats.equalityChecks << " hash-calls=" << stats.hashCalls << "\n";
  }
  return true;
}
} // namespace

int runSyncAliasChecks(func::FuncOp function, const SyncInput &input) {
  if (function.getNumArguments() < 2) {
    return 1;
  }
  if (!checkAliasSelectChains(function)) { return 1; }
  Value first = function.getArgument(0), second = function.getArgument(1);
  const auto &analyzer = input.memory();
  const bool mayAlias = analyzer.gmPolicy() == GMAliasPolicy::MayAlias;
  constexpr std::array<uint64_t, 5> offsets{0, 4, 8, 12, 16};
  for (auto a : offsets) {
    for (auto b : offsets) {
      bool overlap = false;
      for (uint64_t byte = a; byte < a + 8; ++byte) {
        overlap |= byte >= b && byte < b + 8;
      }
      BaseMemInfo left(first, first, AddressSpace::GM, {a}, 8);
      BaseMemInfo same(second, first, AddressSpace::GM, {b}, 8);
      BaseMemInfo distinct(second, second, AddressSpace::GM, {b}, 8);
      if (analyzer.MemAlias(&left, &same) != overlap ||
          analyzer.MemAlias(&left, &distinct) != mayAlias) {
        return 1;
      }
      left.scope = AddressSpace::VEC;
      distinct.scope = AddressSpace::VEC;
      left.hasKnownPhysicalAddresses = distinct.hasKnownPhysicalAddresses = true;
      if (analyzer.MemAlias(&left, &distinct) != overlap) {
        return 1;
      }
      distinct.scope = AddressSpace::MAT;
      if (analyzer.MemAlias(&left, &distinct)) {
        return 1;
      }
    }
  }
  BaseMemInfo unknown(first, first, AddressSpace::GM, {0}, 0);
  BaseMemInfo other(second, second, AddressSpace::GM, {0}, 0);
  if (analyzer.MemAlias(&unknown, &other) != mayAlias || !analyzer.MemAlias(&unknown, &unknown)) {
    return 1;
  }
  unknown.aliasesUnknownRange = true;
  if (!analyzer.MemAlias(&unknown, &other)) {
    return 1;
  }
  BaseMemInfo overflow(first, first, AddressSpace::GM, {std::numeric_limits<uint64_t>::max() - 3}, 8);
  BaseMemInfo same(second, first, AddressSpace::GM, {0}, 8);
  if (!analyzer.MemAlias(&overflow, &same)) {
    return 1;
  }
  llvm::outs() << "shared-alias-contract: passed\n";
  return 0;
}
