// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;
namespace {
bool check(bool condition, StringRef message) {
  if (!condition) {
    llvm::errs() << "selected native test: " << message << "\n";
  }
  return condition;
}
std::string text(func::FuncOp function) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  function.print(stream);
  stream.flush();
  return result;
}
const char *ordinary = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ordinary(%src: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %z = arith.constant 256 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %other = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %z : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>)
    pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>)
      outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%out : !pto.tile_buf<vec, 1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
const char *loop = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @loop(%src: !pto.partition_tensor_view<1x32xf32>, %n: index, %choose: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    scf.for %i = %zero to %n step %one {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      scf.if %choose {
        pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    return
  }
})mlir";
const char *recurrence = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @recurrence(%src: !pto.partition_tensor_view<1x32xf32>, %n: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    scf.for %i = %zero to %n step %one {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
    }
    pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
const char *collective = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @collective(%src: !pto.partition_tensor_view<1x32xf32>,
                       %tile: !pto.tile_buf<vec, 1x32xf32>, %out: !pto.tile_buf<vec, 1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%tile : !pto.tile_buf<vec, 1x32xf32>)
    pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<mix>
    pto.tabs ins(%tile : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
const char *queue = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @queue(%gm: !pto.ptr<f32, gm>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %base = arith.constant 0 : i32
    %outaddr = arith.constant 1024 : i64
    %pipe = pto.initialize_l2g2l_pipe{dir_mask = 3, slot_size = 128,
      slot_num = 2, local_slot_num = 1, flag_base = 0, nosplit = true}
      (%gm : !pto.ptr<f32, gm>, %base : i32, %base : i32) -> !pto.pipe
    %a = pto.declare_tile -> !pto.tile_buf<vec, 1x32xf32>
    %b = pto.declare_tile -> !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %outaddr : !pto.tile_buf<vec, 1x32xf32>
    pto.tpop(%a, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
    pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    pto.tfree(%pipe : !pto.pipe) {split = 0}
    pto.tpop(%b, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
    pto.tpush(%out, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
    pto.tfree(%pipe : !pto.pipe) {split = 0}
    return
  }
})mlir";
bool positive(MLIRContext &context, const char *source, StringRef name) {
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!check(bool(module), "parse positive input")) { return false; }
  auto function = module->lookupSymbol<func::FuncOp>(name);
  oahs::SelectedPlan report;
  const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report);
  if (!check(succeeded(status) && report.success && succeeded(verify(function)), "new native construction")) {
    llvm::errs() << "case=" << name << " reason=" << report.reason << "\n";
    return false;
  }
  auto liveModule = parseSourceString<ModuleOp>(source, &context);
  if (!check(bool(liveModule), "parse live comparison input")) { return false; }
  auto live = liveModule->lookupSymbol<func::FuncOp>(name);
  if (!check(succeeded(oahs::runHandoffSync(live)) && text(live) == text(function),
             "live handoff must emit the selected constructor's exact word")) { return false; }
  if (name == "recurrence") {
    unsigned guards = 0;
    function.walk([&](scf::IfOp) { ++guards; });
    if (!check(report.channels.size() == 2 && guards != 0,
               "live normalized loop must use guarded ready/release roles")) return false;
  }
  unsigned waits = 0, retirements = 0;
  function.walk([&](WaitFlagOp) { ++waits; });
  function.walk([&](BarrierOp barrier) { retirements += barrier.getPipe().getPipe() == PIPE::PIPE_ALL; });
  llvm::outs() << "case=" << name << " selected_updates=" << report.work.selectedUpdates
               << " waits=" << waits << " retirements=" << retirements << "\n";
  return check(waits != 0 && retirements == 1, "selected handoffs and exact terminal drain");
}
bool mutations(MLIRContext &context) {
  for (unsigned mutation = 0; mutation < 3; ++mutation) {
    auto module = parseSourceString<ModuleOp>(ordinary, &context);
    if (!check(bool(module), "parse mutation input")) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("ordinary");
    const auto before = text(function);
    bool changed = false;
    ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) { return success(); });
    const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, [&](func::FuncOp working) {
      mlir::Operation *victim = nullptr;
      working.walk([&](mlir::Operation *op) {
        if (victim) { return; }
        if ((mutation == 0 && isa<WaitFlagOp>(op)) || (mutation == 1 && isa<BarrierOp>(op) &&
            cast<BarrierOp>(op).getPipe().getPipe() == PIPE::PIPE_ALL) || (mutation == 2 && isa<TAddOp>(op))) {
          victim = op;
        }
      });
      if (victim) {
        if (mutation == 2) { victim->setAttr("selected_test_mutation", UnitAttr::get(&context)); }
        else { victim->erase(); }
        changed = true;
      }
    });
    if (!check(changed && failed(status) && text(function) == before, "atomic reconstruction refusal")) {
      return false;
    }
  }
  return true;
}
bool runFile(MLIRContext &context, const char *path) {
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) { return false; }
  bool accepted = true;
  module->walk([&](func::FuncOp function) {
    if (function.isDeclaration()) { return; }
    oahs::SelectedPlan report;
    const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report);
    accepted &= succeeded(status);
    // Existing token order is preserved so earlier recorded logs stay comparable.
    // The added tokens are the report's own counters; `reason` stays last.
    const auto &work = report.work;
    llvm::errs() << "function=" << function.getSymName() << " construction=" << report.success
                 << " reconstruction=" << succeeded(status) << " failure=" << unsigned(report.failure)
                 << " cut=" << report.cut
                 << " updates=" << work.selectedUpdates << " replay=" << work.replaySiteEvaluations
                 << " microseconds=" << work.elapsedMicroseconds
                 << " forward=" << work.forwardSiteEvaluations << " visits=" << work.frontierVisits
                 << " key_queries=" << work.keyQueries << " invariant=" << work.invariantSiteEvaluations
                 << " prepare_microseconds=" << work.preparationMicroseconds
                 << " sites=" << work.constructedSites << " words=" << work.commandWords
                 << " cells=" << work.cells << " eligible_keys=" << work.eligibleKeys
                 << " components=" << work.components << " cyclic=" << work.cyclicComponents
                 << " recurring=" << work.recurringChannels
                 << " contextual=" << work.contextualReplays
                 << " unreused_updates=" << work.unreusedUpdates
                 << " sources=" << work.sourceHandles << " acknowledgments=" << work.acknowledgments
                 << " common_cut=" << work.commonCutTransfers
                 << " decisions=" << report.decisions.size() << " endpoints=" << report.ledger.size();
    // How much of the component prefix each update actually kept. A nonzero
    // reuse count says nothing on its own; the fraction of the prefix is what
    // distinguishes working reuse from a boundary that collapses to the entry.
    std::size_t reusedTotal = 0, reusedMax = 0, contextualUpdates = 0;
    for (const auto &update : report.updates) {
      reusedTotal += update.reusedComponents;
      reusedMax = std::max(reusedMax, update.reusedComponents);
      contextualUpdates += update.contextual;
    }
    llvm::errs() << " reused_total=" << reusedTotal << " reused_max=" << reusedMax
                 << " contextual_updates=" << contextualUpdates
                 << " periods=";
    for (std::size_t i = 0; i < report.channels.size(); ++i) {
      llvm::errs() << (i ? "," : "") << report.channels[i].period;
    }
    llvm::errs() << (report.channels.empty() ? "-" : "") << " reason=" << report.reason << "\n";
  });
  if (accepted) { module->print(llvm::outs()); }
  return accepted;
}
} // namespace
int main(int argc, char **argv) {
  MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<PTODialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  if (argc == 3 && StringRef(argv[1]) == "--construct") {
    return runFile(context, argv[2]) ? 0 : 1;
  }
  if (argc != 1) {
    llvm::errs() << "usage: pto-oahs-selected-test [--construct INPUT]\n";
    return 2;
  }
  const bool passed = positive(context, ordinary, "ordinary") && positive(context, loop, "loop") &&
                      positive(context, recurrence, "recurrence") &&
                      positive(context, collective, "collective") &&
                      positive(context, queue, "queue") && mutations(context);
  return passed ? 0 : 1;
}
