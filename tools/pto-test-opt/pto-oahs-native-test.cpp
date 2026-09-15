// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/OAHS/Native.h"
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
    require(succeeded(oahs::analyzeHandoffSync(function)));
    require(text(function) == before);
  }
  for (unsigned mutation = 0; mutation < 3; ++mutation) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("test");
    const std::string before = text(function);
    bool changed = false;
    auto result = oahs::testing::runHandoffSyncWithMutation(function,
      [&](func::FuncOp working) {
        if (!mutation) return;
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
  llvm::outs() << "OAHS native placement/atomicity/grouped-footprint checks passed\n";
}
