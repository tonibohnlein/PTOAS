// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/InsertSync/SyncSlotMapping.h"
#include "PTO/Transforms/InsertSync/SyncAccumulatorOrdering.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

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
// Scalar/address qualification is shared and independent of payload opcodes.
const char *slotInput = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @slots(%src: !pto.partition_tensor_view<1x32xf32>, %n: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %stride = arith.constant 128 : index
    %base = arith.constant 256 : index
    %out_addr = arith.constant 2048 : i64
    %out = pto.alloc_tile addr = %out_addr : !pto.tile_buf<vec, 1x32xf32>
    %result = scf.for %i = %zero to %n step %one iter_args(%slot = %two) -> index {
      %advance = arith.addi %slot, %two : index
      %next = arith.remsi %advance, %three : index
      %offset = arith.muli %next, %stride : index
      %address = arith.addi %offset, %base : index
      %cast = arith.index_cast %address : index to i64
      %bank = pto.alloc_tile addr = %cast : !pto.tile_buf<vec, 1x32xf32>
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
      pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
      scf.yield %next : index
    }
    return
  }
})mlir";
const char *matrixInput = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @matrix(%unknown: index) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %zero = arith.constant 0 : i64
    %m = arith.constant 128 : index
    %n = arith.constant 256 : index
    %k = arith.constant 64 : index
    %a = pto.alloc_tile addr = %zero valid_row = %m valid_col = %k : !pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>
    %b = pto.alloc_tile addr = %zero valid_row = %k valid_col = %n : !pto.tile_buf<right, 64x256xf16, valid=?x?, slayout=col_major>
    %c = pto.alloc_tile addr = %zero valid_row = %m valid_col = %n : !pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>
    pto.tmatmul ins(%a, %b : !pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>, !pto.tile_buf<right, 64x256xf16, valid=?x?, slayout=col_major>) outs(%c : !pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>)
    pto.tmatmul.acc ins(%c, %a, %b : !pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>, !pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>, !pto.tile_buf<right, 64x256xf16, valid=?x?, slayout=col_major>) outs(%c : !pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>)
    return
  }
})mlir";
bool accumulatorOrdering(MLIRContext &context) {
  for (unsigned mutation = 0; mutation < 5; ++mutation) {
    std::string source = matrixInput;
    if (mutation == 1) {
      for (const std::string size : {"128", "256"}) {
        std::size_t at = 0;
        while ((at = source.find(size, at)) != std::string::npos) {
          source.replace(at, size.size(), "16"); at += 2;
        }
      }
    }
    if (mutation == 2) {
      auto at = source.find("valid_row = %m");
      source.replace(at, std::string("valid_row = %m").size(), "valid_row = %unknown");
    }
    if (mutation == 3) {
      const auto at = source.find("    pto.tmatmul ins");
      source.insert(at, "    pto.set_validshape %a, %m, %k : !pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse ACC contract fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("matrix");
    if (mutation == 4) function.walk([&](TMatmulAccOp op) {
      op->setAttr("accPhase", AccPhaseAttr::get(&context, AccPhase::Final));
    });
    oahs::NativeAnalysis imported;
    if (mutation == 4) {
      bool rejected = true;
      function.walk([&](TMatmulAccOp op) { rejected &= !syncAccumulatorOrder(op); });
      if (!check(rejected && failed(oahs::analyzeHandoffSync(function, imported)),
                 "phase mode borrowed ordinary ACC credit")) return false;
      continue;
    }
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)), "ACC import")) return false;
    const bool qualified = llvm::any_of(imported.program.cells, [](const auto &c) { return c.nativeMmadAccOrder; });
    if (!check(qualified == (mutation == 0), "native ACC qualification scope")) return false;
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)), "construct ACC fixture")) return false;
    const auto barriers = std::count_if(plan.ledger.begin(), plan.ledger.end(), [](const auto &e) {
      return e.command.kind == oahs::Command::Barrier && e.command.source == oahs::Pipe::M;
    });
    if (!check((barriers == 0) == (mutation == 0), "ACC fence required outside qualified contract")) return false;
  }
  return true;
}
bool constantAddresses(MLIRContext &context) {
  const char *input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @addresses(%unknown: i64) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %z = arith.index_cast %zero : index to i64
    %base = arith.constant 24576 : i64
    %sum = arith.addi %base, %z : i64
    %a = pto.alloc_tile addr = %sum : !pto.tile_buf<vec, 1x16xf32>
    pto.tabs ins(%a : !pto.tile_buf<vec, 1x16xf32>) outs(%a : !pto.tile_buf<vec, 1x16xf32>)
    return
  }
})mlir";
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!check(bool(module), "parse constant address fixture")) {
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("addresses");
    AllocTileOp allocation;
    function.walk([&](AllocTileOp op) { allocation = op; });
    auto sum = allocation.getAddr().getDefiningOp<arith::AddIOp>();
    OpBuilder builder(sum);
    if (mutation == 1) {
      sum.setOperand(1, function.getArgument(0));
    }
    if (mutation == 2) {
      sum.setOperand(0, builder.create<arith::ConstantIntOp>(sum.getLoc(), INT64_MAX, 64));
      sum.setOperand(1, builder.create<arith::ConstantIntOp>(sum.getLoc(), 1, 64));
    }
    if (mutation == 3) {
      auto large = builder.create<arith::ConstantIndexOp>(sum.getLoc(), 256);
      auto narrow = builder.create<arith::IndexCastOp>(sum.getLoc(), builder.getI8Type(), large);
      auto widen = builder.create<arith::IndexCastOp>(sum.getLoc(), builder.getIndexType(), narrow);
      auto address = builder.create<arith::IndexCastOp>(sum.getLoc(), builder.getI64Type(), widen);
      allocation.getAddrMutable().assign(address);
    }
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)), "constant address import")) {
      return false;
    }
    bool known = false, unknown = false;
    for (const auto &cell : imported.program.cells) {
      unknown |= cell.unknownRange;
      known |= !cell.unknownRange && cell.ranges == std::vector<std::pair<uint64_t,uint64_t>>{{24576, 64}};
    }
    if (!check(mutation == 0 ? known && !unknown : !known && unknown,
               "constant address certainty or byte footprint incorrect")) {
      return false;
    }
  }
  return true;
}
bool slotMappings(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(slotInput, &context);
  if (!check(bool(module), "parse carried-slot fixture")) return false;
  auto function = module->lookupSymbol<func::FuncOp>("slots");
  const auto original = text(function);
  scf::ForOp loop;
  AllocTileOp bank;
  function.walk([&](scf::ForOp op) { loop = op; });
  loop.walk([&](AllocTileOp op) { bank = op; });
  auto mapping = SyncSlotMapping::derive(loop, 8);
  if (!check(mapping && mapping->period == 3, "derive stride-two modulo-three orbit")) return false;
  uint64_t slot = 2;
  for (unsigned i = 0; i < 30; ++i) {
    slot = (slot + 2) % 3;
    auto address = SyncSlotMapping::evaluate(bank.getAddr(), mapping->values[i % 3]);
    if (!check(address && *address == 256 + 128 * slot, "slot address differs from original recurrence")) return false;
  }
  if (!check(!SyncSlotMapping::derive(loop, 2), "finite vocabulary cannot silently truncate the orbit")) return false;
  oahs::NativeAnalysis imported;
  if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)) && text(function) == original,
             "periodic import must preserve original IR")) return false;
  bool distinctBanks[3] = {};
  for (const auto &cell : imported.program.cells) {
    if (cell.coordinateSpace != "physical-local" || cell.ranges.size() != 1) continue;
    for (unsigned i = 0; i < 3; ++i)
      distinctBanks[i] |= cell.ranges[0] == std::make_pair(uint64_t(256 + 128 * i), uint64_t(128));
  }
  if (!check(distinctBanks[0] && distinctBanks[1] && distinctBanks[2], "lost exact physical bank partition")) return false;
  oahs::SelectedPlan plan;
  if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)) &&
             plan.work.recurringChannels == 6, "generic three-bank native recurrence")) return false;

  std::string nestedSource = slotInput;
  const auto insertion = nestedSource.find("      scf.yield %next");
  nestedSource.insert(insertion, "      scf.for %child = %zero to %n step %one {\n      }\n");
  auto nestedModule = parseSourceString<ModuleOp>(nestedSource, &context);
  if (!check(bool(nestedModule), "parse outer finite footprint fixture")) return false;
  oahs::NativeAnalysis nested;
  auto nestedFunction = nestedModule->lookupSymbol<func::FuncOp>("slots");
  if (!check(succeeded(oahs::analyzeHandoffSync(nestedFunction, nested)), "outer footprint import")) return false;
  for (unsigned bankId = 0; bankId < 3; ++bankId) {
    if (!check(llvm::any_of(nested.program.cells, [&](const auto &c) {
      return !c.unknownRange && std::find(c.ranges.begin(), c.ranges.end(),
                 std::make_pair(uint64_t(256 + 128 * bankId), uint64_t(128))) != c.ranges.end();
    }), "non-leaf scalar orbit lost finite physical bank footprint")) return false;
  }
  // Bounds, initialization, arithmetic and narrowing must be proved, not guessed.
  for (unsigned mutation = 0; mutation < 6; ++mutation) {
    auto test = parseSourceString<ModuleOp>(slotInput, &context);
    scf::ForOp changed;
    test->walk([&](scf::ForOp op) { changed = op; });
    auto yield = cast<scf::YieldOp>(changed.getBody()->getTerminator());
    auto rem = yield.getOperand(0).getDefiningOp<arith::RemSIOp>();
    auto add = rem.getLhs().getDefiningOp<arith::AddIOp>();
    if (mutation == 0) changed.getInitArgsMutable().assign(ValueRange{changed.getUpperBound()});
    if (mutation == 1) changed.getStepMutable().assign(add.getRhs());
    if (mutation == 2) rem.setOperand(1, changed.getLowerBound());
    if (mutation == 3) add.setOperand(1, changed.getInductionVar());
    if (mutation == 4) changed.getInitArgsMutable().assign(ValueRange{rem.getRhs()});
    if (mutation == 5) {
      OpBuilder builder(changed);
      auto huge = builder.create<arith::ConstantIndexOp>(changed.getLoc(), std::numeric_limits<int64_t>::max());
      add.setOperand(1, huge);
    }
    if (!check(!SyncSlotMapping::derive(changed, 8), "unproved carried slot was admitted")) return false;
  }
  auto unresolved = parseSourceString<ModuleOp>(slotInput, &context);
  auto unknownFunction = unresolved->lookupSymbol<func::FuncOp>("slots");
  scf::ForOp unknownLoop;
  AllocTileOp unknownBank;
  unresolved->walk([&](scf::ForOp op) { unknownLoop = op; });
  unknownLoop.walk([&](AllocTileOp op) { unknownBank = op; });
  auto castAddress = unknownBank.getAddr().getDefiningOp<arith::IndexCastOp>();
  OpBuilder unknownBuilder(castAddress);
  Value unavailable = unknownLoop.getUpperBound();
  for (unsigned depth = 0; depth < 40; ++depth)
    unavailable = unknownBuilder.create<arith::AddIOp>(castAddress.getLoc(), unavailable, unavailable);
  castAddress->setOperand(0, unavailable);
  oahs::NativeAnalysis unknown;
  if (!check(succeeded(oahs::analyzeHandoffSync(unknownFunction, unknown)) &&
             unknown.program.operations.size() == 2 &&
             llvm::any_of(unknown.program.cells, [](const auto &c) {
               return c.unknownRange && c.addressSpace == std::to_string(unsigned(AddressSpace::VEC));
             }), "unknown pool base must retain conservative alias coverage")) return false;
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
                 << " recurring_trials=" << work.recurringTrials
                 << " recurring_removed=" << work.redundantRecurringChannels
                 << " recurring_analysis_sites=" << work.recurringAnalysisSites
                 << " loop_entry_transfers=" << work.loopEntryTransfers
                 << " loop_entry_analysis_sites=" << work.loopEntryAnalysisSites
                 << " loop_entry_preparation_sites=" << work.loopEntryPreparationSites
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
                      positive(context, queue, "queue") && mutations(context) && constantAddresses(context) &&
                      slotMappings(context) && accumulatorOrdering(context);
  return passed ? 0 : 1;
}
