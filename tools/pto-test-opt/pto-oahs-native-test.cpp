// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: LicenseRef-CANN-Open-Software-License-2.0
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
  // Exercise the real native alias-query widening threshold, not a trusted
  // user attribute. 730 two-effect loads exceed one million effect pairs.
  std::string many = R"mlir(module attributes {pto.target_arch = "a3"} {
    func.func @budget(%src: !pto.partition_tensor_view<1x32xf32>)
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
  auto budgetFunction = module->lookupSymbol<func::FuncOp>("budget");
  require(succeeded(oahs::runHandoffSync(budgetFunction)));
  unsigned loads = 0;
  budgetFunction.walk([&](TLoadOp load) {
    auto *previous = load->getPrevNode();
    require(previous && isa<BarrierOp>(previous) &&
            cast<BarrierOp>(previous).getPipe().getPipe() == PIPE::PIPE_ALL);
    ++loads;
  });
  require(loads == 730);
  llvm::outs() << "OAHS native placement/atomicity/budget checks passed\n";
}
