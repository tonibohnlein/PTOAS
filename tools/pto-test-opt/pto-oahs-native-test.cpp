// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/SyncOrdinaryExternalModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace mlir;
using namespace mlir::pto;
static void require(bool condition) { if (!condition) std::abort(); }
static std::string text(func::FuncOp function) {
  std::string result; llvm::raw_string_ostream out(result); function.print(out); out.flush(); return result;
}
int main() {
  DialectRegistry registry;
  registerSyncOrdinaryExternalModels(registry);
  MLIRContext context(registry);
  context.loadDialect<PTODialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  const char *source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @test(%src: !pto.partition_tensor_view<1x32xf32>, %b: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %c = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    scf.if %b {
      pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
    } else {
    }
    return
  }
})mlir";
  {
    auto module = parseSourceString<ModuleOp>(source, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("test");
    const std::string before = text(function);
    oahs::NativeAnalysis report;
    require(succeeded(oahs::analyzeHandoffSync(function, report)));
    require(report.analysis.complete && !report.analysis.verified());
    require(report.phases.size() == 2 && report.program.operations.size() == 2);
    require(isa<TLoadOp>(report.phases[0]) && isa<TAddOp>(report.phases[1]));
    bool raw = false;
    for (const auto &r : report.analysis.residuals)
      raw |= r.kind == oahs::CompletionRequirement::RAW &&
             r.demand.producer == 0 && r.demand.consumer == 1;
    require(raw && report.analysis.protocol.empty());
    const auto contextId = report.analysis.cuts[1].context;
    require(report.analysis.contexts[contextId].kind == oahs::AnalysisContext::ThenArm);
    require(!report.analysis.retirement.empty());
    require(text(function) == before);
    require(succeeded(oahs::analyzeHandoffSync(function)));
    require(text(function) == before);
    // M2 queries own the imported state and never change native IR. The load's
    // enclosing unconditional cut cannot pair with an optional branch consumer.
    oahs::PrefixQuery prefixes(report.program);
    const auto cover = prefixes.coverByPrefixes(1);
    require(cover.coversAll() && !cover.selected.empty());
    require(cover.candidates[cover.selected.front()].publication == 1);
    require(!prefixes.inspectPrefix(oahs::Pipe::MTE2, 0, 1).matchingEstablished);
    require(text(function) == before);
    // The same imported program/analysis explains and verifies the candidate.
    const auto plan = oahs::construct(report.program);
    require(plan.success && oahs::analyze(report.program, plan.commands).verified());
  }
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("test");
    const std::string before = text(function);
    bool changed = false;
    auto result = oahs::testing::runHandoffSyncWithMutation(function,
      [&](func::FuncOp working) {
        if (!mutation) return;
        if (mutation == 3) {
          // An extra global fence can remain memory-safe but is not the selected
          // packet word. Exact emission identity must reject it transactionally.
          auto *ret = working.getBody().front().getTerminator();
          OpBuilder builder(ret);
          builder.create<BarrierOp>(ret->getLoc(),
              PipeAttr::get(working.getContext(), PIPE::PIPE_ALL));
          changed = true;
          return;
        }
        scf::IfOp choice;
        working.walk([&](scf::IfOp op) { choice = op; });
        mlir::Operation *victim = nullptr;
        working.walk([&](mlir::Operation *op) {
          if (victim || !isa<BarrierOp, WaitFlagOp>(op)) return;
          if (mutation == 1 && op->getBlock() == &choice.getThenRegion().front()) victim = op;
          if (mutation == 2 && isa<BarrierOp>(op) &&
              cast<BarrierOp>(op).getPipe().getPipe() == PIPE::PIPE_ALL &&
              op->getBlock() == &working.getBody().front()) victim = op;
        });
        require(victim != nullptr);
        victim->moveBefore(choice.getElseRegion().front().getTerminator());
        changed = true;
      });
    if (!mutation) require(succeeded(result) && succeeded(verify(function)));
    else require(changed && failed(result) && text(function) == before);
  }
  {
    const char *earlySource = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @early_prefix(%src: !pto.partition_tensor_view<1x32xf32>, %b: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %z = arith.constant 256 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %other = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %z : !pto.tile_buf<vec, 1x32xf32>
    scf.if %b {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>)
      pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    }
    return
  }
})mlir";
    auto module = parseSourceString<ModuleOp>(earlySource, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("early_prefix");
    const auto before = text(function);
    oahs::NativeAnalysis imported;
    require(succeeded(oahs::analyzeHandoffSync(function, imported)));
    oahs::PrefixQuery query(imported.program);
    const auto cover = query.coverByPrefixes(2);
    require(cover.coversAll() && cover.selected.size() == 1);
    const auto &prefix = cover.candidates[cover.selected.front()];
    require(prefix.publication == 1 && prefix.acquisition == 2);
    require(prefix.sourceContext == prefix.targetContext);
    require(text(function) == before);
    require(succeeded(oahs::runHandoffSync(function)));
    // The analysis report's old phase pointers are invalid after replacement.
    llvm::SmallVector<TLoadOp> loads;
    function.walk([&](TLoadOp load) { loads.push_back(load); });
    require(loads.size() == 2);
    require(loads[1]->getPrevNode() && isa<SetFlagOp>(loads[1]->getPrevNode()));
    require(loads[1]->getNextNode() && isa<WaitFlagOp>(loads[1]->getNextNode()));
    require(succeeded(verify(function)));
  }
  // Repeated identical footprints must not cross a compiler-work threshold
  // and switch to ALL. This population exceeded the old million-pair cutoff.
  std::string many = R"mlir(module attributes {pto.target_arch = "a3"} {
    func.func @repeated_footprint(%src: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
      %address = arith.constant 0 : i64
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %a = pto.alloc_tile addr = %address : !pto.tile_buf<vec, 1x32xf32>
      scf.for %i = %zero to %two step %one {
  )mlir";
  for (unsigned i = 0; i < 730; ++i)
    many += "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)\n";
  many += "} return } }";
  auto module = parseSourceString<ModuleOp>(many, &context);
  require(bool(module));
  auto manyFunction = module->lookupSymbol<func::FuncOp>("repeated_footprint");
  require(succeeded(oahs::runHandoffSync(manyFunction)));
  unsigned loads = 0;
  manyFunction.walk([&](TLoadOp load) {
    auto *previous = load->getPrevNode();
    require(previous && isa<BarrierOp>(previous) &&
            cast<BarrierOp>(previous).getPipe().getPipe() == PIPE::PIPE_MTE2);
    ++loads;
  });
  require(loads == 730);
  unsigned drains = 0;
  manyFunction.walk([&](BarrierOp barrier) { drains += barrier.getPipe().getPipe() == PIPE::PIPE_ALL; });
  require(drains == 1); // original invocation retirement, not a budget fallback
  llvm::outs() << "OAHS native placement/atomicity/grouped-footprint/prefix checks passed\n";
}
