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
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncAccumulatorOrdering.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <map>
#include <tuple>

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
    oahs::SelectedPlan report;
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
    }, &report);
    const bool atomicRefusal = changed && failed(status) && text(function) == before &&
                               !report.declinedFirstUse && !report.declinedObservation;
    if (!check(atomicRefusal,
               "atomic reconstruction refusal without admission retry")) {
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
// Variants exercise one semantic contract through the real native importer and
// constructor. They change representation, not the synchronization policy.
std::pair<std::string, bool> accumulatorVariant(unsigned variant) {
  std::string source = matrixInput;
  bool expected = true;
  auto replace = [&source](const std::string &from, const std::string &to) {
    std::size_t at = 0;
    while ((at = source.find(from, at)) != std::string::npos) {
      source.replace(at, from.size(), to);
      at += to.size();
    }
  };
  const std::string aType = "!pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>";
  const std::string cType = "!pto.tile_buf<acc, 128x256xf32, valid=?x?, "
                            "blayout=col_major, slayout=row_major, fractal=1024>";
  std::string update = "    pto.set_validshape %a, %m, %k : " + aType + "\n";
  if (variant == 1 || variant == 10) {
    replace("128", variant == 1 ? "48" : "16");
    replace("256", variant == 1 ? "48" : "160");
    expected = variant == 10;
  } else if (variant == 2) {
    replace("valid_row = %m", "valid_row = %unknown");
    expected = false;
  } else if (variant == 3) {
    source.insert(source.find("    pto.tmatmul ins"), update);
  } else if (variant == 4) {
    expected = false;
  } else if (variant == 5 || variant == 6) {
    replace("f16", variant == 5 ? "bf16" : "i8");
    if (variant == 6) { replace("f32", "i32"); }
  } else if (variant == 7) {
    // A smaller reduction tail changes K but preserves output coverage.
    std::string tail = source.substr(source.find("    %a ="), source.find("    %c =") - source.find("    %a ="));
    for (const auto &names : {std::pair<std::string, std::string>{"%a =", "%at ="}, {"%b =", "%bt ="}}) {
      tail.replace(tail.find(names.first), names.first.size(), names.second);
    }
    source.insert(source.find("    pto.tmatmul ins"), tail);
    const auto begin = source.find("    pto.tmatmul.acc");
    source.replace(source.find("%a, %b", begin), 6, "%at, %bt");
    const auto start = source.find("    %at =");
    source.insert(start, "    %kt = arith.constant 32 : index\n");
    for (const std::string name : {"%at =", "%bt ="}) {
      auto pos = source.find(name);
      pos = source.find("%k", pos);
      source.replace(pos, 2, "%kt");
    }
  } else if (variant == 8 || variant == 9) {
    replace("constant 128 : index", variant == 8 ? "constant 96 : index" : "constant 127 : index");
    replace("constant 256 : index", variant == 8 ? "constant 240 : index" : "constant 255 : index");
    if (variant == 9) { replace("constant 64 : index", "constant 63 : index"); }
  } else if (variant == 11) {
    replace("constant 128 : index", "constant 31 : index");
    replace("constant 256 : index", "constant 159 : index");
    expected = false;
  } else if (variant == 12) {
    const auto at = source.find("    pto.tmatmul ins");
    source.insert(at, "    %spare = pto.alloc_tile addr = %zero valid_row = %m valid_col = %k : " +
                      aType + "\n    pto.set_validshape %spare, %unknown, %k : " + aType + "\n");
  } else if (variant == 13 || variant == 14) {
    update.replace(update.find("%m"), 2, "%unknown");
    source.insert(source.find(variant == 13 ? "    pto.tmatmul.acc" : "    return"), update);
    expected = variant == 14;
  } else if (variant == 15 || variant == 16) {
    const auto at = source.find("    pto.tmatmul ins");
    const auto right = variant == 15 ? update : "    pto.set_validshape %a, %unknown, %k : " + aType + "\n";
    source.insert(at, "    %cond = arith.cmpi eq, %unknown, %m : index\n    scf.if %cond {\n" +
                      update + "    } else {\n" + right + "    }\n");
    expected = variant == 15;
  } else if (variant == 17) {
    // Distinct SSA descriptors of one physical accumulator, with static valid
    // shape on the view instead of explicit allocation operands.
    std::string viewType = cType;
    viewType.erase(viewType.find(", valid=?x?"), std::string(", valid=?x?").size());
    const auto at = source.find("    pto.tmatmul.acc");
    source.insert(at, "    %view = pto.treshape %c : " + cType + " -> " + viewType + "\n");
    const auto dest = source.find("outs(%c", source.find("    pto.tmatmul.acc"));
    source.replace(dest, std::string("outs(%c : " + cType).size(), "outs(%view : " + viewType);
    replace(cType, viewType);
    replace("%c = pto.alloc_tile addr = %zero valid_row = %m valid_col = %n",
            "%c = pto.alloc_tile addr = %zero");
  } else if (variant == 18) {
    // Overlap without identical output identity cannot borrow access ordering.
    const auto at = source.find("    pto.tmatmul.acc");
    source.insert(at, "    %offset = arith.constant 1024 : i64\n"
                      "    %other = pto.alloc_tile addr = %offset valid_row = %m valid_col = %n : " + cType + "\n");
    source.replace(source.find("outs(%c", source.find("    pto.tmatmul.acc")), 7, "outs(%other");
    expected = false;
  } else if (variant == 19) {
    replace("%m = arith.constant 128 : index", "%half = arith.constant 64 : index\n"
                                                "    %m = arith.addi %half, %half : index");
  } else if (variant == 20 || variant == 21) {
    const auto at = source.find("    pto.tmatmul.acc");
    const std::string offset = variant == 20 ? "%z" : "%unknown";
    source.insert(at, "    %z = arith.constant 0 : index\n"
                      "    %view = pto.subview %c[" + offset + ", %z] sizes [128, 256] : " +
                      cType + " -> " + cType + "\n");
    source.replace(source.find("outs(%c", source.find("    pto.tmatmul.acc")), 7, "outs(%view");
    std::string staticType = cType;
    staticType.erase(staticType.find(", valid=?x?"), std::string(", valid=?x?").size());
    replace(cType, staticType);
    replace("%c = pto.alloc_tile addr = %zero valid_row = %m valid_col = %n",
            "%c = pto.alloc_tile addr = %zero");
    expected = variant == 20;
  }
  if (variant == 22 || variant == 23) {
    // Argument descriptor dimensions are facts too, but set_validshape is
    // only legal on locally bound tiles. Exercise the supported argument
    // contract: unknown dynamic dimensions versus known static dimensions.
    std::string argumentType = aType;
    if (variant == 23) {
      argumentType.erase(argumentType.find(", valid=?x?"), std::string(", valid=?x?").size());
    }
    const auto begin = source.find("    %a =");
    source.erase(begin, source.find('\n', begin) + 1 - begin);
    replace("%unknown: index", "%unknown: index, %a: " + argumentType);
    replace(aType, argumentType);
    expected = variant == 23;
  }
  return {source, expected};
}

bool accumulatorAccessScope(MLIRContext &context) {
  std::string source = matrixInput;
  const std::string acc = "!pto.tile_buf<acc, 128x256xf32, valid=?x?, "
                          "blayout=col_major, slayout=row_major, fractal=1024>";
  const std::string left = "!pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>";
  const std::string mat = "!pto.tile_buf<mat, 128x64xf16, valid=?x?, slayout=row_major>";
  source.replace(source.find("%unknown: index"), std::string("%unknown: index").size(),
                 "%unknown: index, %out: !pto.partition_tensor_view<128x256xf32>");
  source.insert(source.find("    return"),
      "    pto.tstore ins(%c : " + acc + ") outs(%out : !pto.partition_tensor_view<128x256xf32>)\n"
      "    %z = arith.constant 0 : index\n"
      "    %mat = pto.alloc_tile addr = %zero valid_row = %m valid_col = %k : " + mat + "\n"
      "    pto.textract ins(%mat, %z, %z : " + mat + ", index, index) outs(%a : " + left + ")\n");
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!check(bool(module), "parse ACC access-scope fixture")) { return false; }
  auto function = module->lookupSymbol<func::FuncOp>("matrix");
  oahs::NativeAnalysis input;
  if (!check(succeeded(oahs::analyzeHandoffSync(function, input)), "import ACC access scope")) { return false; }
  oahs::CausalFrontier frontier(input.program);
  const auto first = frontier.issue(frontier.initial(), 0);
  const auto accumulated = frontier.issue(first.state, 1);
  if (!check(first.applied && accumulated.applied && !frontier.inspect(accumulated.state, 2).applied &&
             !frontier.inspect(accumulated.state, 3).applied,
             "ACC ordering released operands or established FIX readiness")) { return false; }
  oahs::SelectedPlan plan;
  if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
             "construct ACC access-scope fixture")) { return false; }
  // Removing either required transfer must still be rejected by the unchanged
  // independent causal checker, despite the new accumulator qualification.
  bool fix = false, operand = false;
  for (oahs::Cut cut = 0; cut < plan.commands.size(); ++cut) {
    for (std::size_t index = 0; index < plan.commands[cut].size(); ++index) {
      const auto &command = plan.commands[cut][index];
      if (command.kind != oahs::Command::Acquire || command.source != oahs::Pipe::M ||
          (command.observer != oahs::Pipe::FIX && command.observer != oahs::Pipe::MTE1)) { continue; }
      auto damaged = plan.commands;
      damaged[cut].erase(damaged[cut].begin() + index);
      if (!check(!oahs::checkCausalFrontier(input.program, damaged).accepted,
                 "missing matrix completion transfer was accepted")) { return false; }
      fix |= command.observer == oahs::Pipe::FIX;
      operand |= command.observer == oahs::Pipe::MTE1;
    }
  }
  return check(fix && operand, "ACC fixture lost required FIX/operand transfers");
}

bool accumulatorOrdering(MLIRContext &context) {
  for (unsigned variant = 0; variant < 24; ++variant) {
    const auto fixture = accumulatorVariant(variant);
    auto module = parseSourceString<ModuleOp>(fixture.first, &context);
    if (!check(bool(module), "parse ACC contract fixture")) {
      llvm::errs() << "ACC variant=" << variant << "\n";
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("matrix");
    if (variant == 4) {
      function.walk([&context](TMatmulAccOp op) {
        op->setAttr("accPhase", AccPhaseAttr::get(&context, AccPhase::Final));
      });
    }
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)), "ACC import")) { return false; }
    const bool qualified = llvm::any_of(imported.program.operations, [&imported](const auto &op) {
      if (!op.nativeMmadAccumulate) { return false; }
      return llvm::all_of(op.accesses, [&imported](const auto &access) {
        const auto &cell = imported.program.cells[access.cell];
        return cell.domain != oahs::Cell::Domain::Accumulator ||
               access.nativeAccumulatorClass != oahs::NoControlId;
      });
    });
    if (!check(qualified == fixture.second, "native ACC qualification scope")) {
      llvm::errs() << "ACC variant=" << variant << "\n";
      return false;
    }
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "construct ACC fixture")) { return false; }
    const auto barriers = std::count_if(plan.ledger.begin(), plan.ledger.end(), [](const auto &e) {
      return e.command.kind == oahs::Command::Barrier && e.command.source == oahs::Pipe::M;
    });
    if (!check((barriers == 0) == fixture.second, "ACC fence required outside qualified contract")) {
      llvm::errs() << "ACC variant=" << variant << "\n";
      return false;
    }
  }
  return accumulatorAccessScope(context);
}
bool accumulatorEpisodes(MLIRContext& context) {
  const std::string original = matrixInput;
  const std::string left = "!pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>";
  const std::string acc = "!pto.tile_buf<acc, 128x256xf32, valid=?x?, "
                          "blayout=col_major, slayout=row_major, fractal=1024>";
  const auto begin = original.find("    pto.tmatmul ins");
  const auto middle = original.find("    pto.tmatmul.acc");
  const auto end = original.find("    return");
  const auto initialize = original.substr(begin, middle - begin);
  const auto accumulate = original.substr(middle, end - middle);
  auto shape = [&](const std::string& rows) {
    return "    pto.set_validshape %a, " + rows + ", %k : " + left + "\n" +
           "    pto.set_validshape %c, " + rows + ", %n : " + acc + "\n";
  };
  for (unsigned variant = 0; variant < 5; ++variant) {
    const std::string other = variant == 1 ? "%unknown" : "%small";
    std::string body;
    if (variant == 2) {
      body = initialize + accumulate + shape(other) + initialize + accumulate;
    } else if (variant == 3) {
      body = "    %choose = arith.cmpi eq, %unknown, %m : index\n"
             "    scf.if %choose {\n" + shape(other) + initialize +
             "    } else {\n" + initialize + "    }\n" +
             shape("%m") + initialize + accumulate + accumulate;
    } else {
      body = shape(other) + initialize + shape("%m") +
             (variant == 4 ? accumulate : initialize) + accumulate + accumulate;
    }
    const auto source = original.substr(0, begin) + "    %small = arith.constant 16 : index\n" +
                        body + original.substr(end);
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse mixed ACC episode fixture")) {
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("matrix");
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)), "import mixed ACC episodes")) {
      return false;
    }
    const auto& last = imported.program.operations.back();
    if (!check(last.nativeMmadAccumulate && llvm::any_of(last.accesses, [](const auto& access) {
          return access.nativeAccumulatorClass != oahs::NoControlId;
        }), "unrelated ACC episode erased a qualified access contract")) {
      return false;
    }
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "construct mixed ACC episodes")) {
      return false;
    }
    const auto barriers = std::count_if(plan.ledger.begin(), plan.ledger.end(), [](const auto& endpoint) {
      return endpoint.command.kind == oahs::Command::Barrier && endpoint.command.source == oahs::Pipe::M;
    });
    if (!check(barriers == 1, "ACC episodes need their transition repair, not per-accumulation repairs")) {
      llvm::errs() << "episode variant=" << variant << " barriers=" << barriers << "\n";
      return false;
    }
    bool removed = false;
    auto damaged = plan.commands;
    for (auto& word : damaged) {
      word.erase(std::remove_if(word.begin(), word.end(), [&](const auto& command) {
        if (command.kind == oahs::Command::Barrier && command.source == oahs::Pipe::M) {
          removed = true;
          return true;
        }
        return false;
      }), word.end());
    }
    if (!check(removed && !oahs::analyze(imported.program, damaged).verified(),
               "mixed ACC transition lost its required independent-checker repair")) {
      return false;
    }
  }
  return true;
}

bool firstUseOrdering(MLIRContext &context) {
  std::string original = matrixInput;
  const auto argument = original.find("%unknown: index");
  original.replace(argument, std::string("%unknown: index").size(),
                   "%unknown: index, %out: !pto.partition_tensor_view<128x256xf32>");
  const auto begin = original.find("    pto.tmatmul ins");
  const auto middle = original.find("    pto.tmatmul.acc ins");
  const auto end = original.find("    return");
  for (unsigned variant = 0; variant < 10; ++variant) {
    std::string source = original.substr(0, begin);
    source += R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c5equiv = arith.subi %c9, %c4 : index
    scf.for %tile = %c0 to %c2 step %c1 {
)mlir";
    const bool shifted = variant >= 4;
    const auto lower = shifted ? "%c5" : "%c0";
    const auto upper = shifted ? "%c9" : "%c2";
    const auto step = shifted ? "%c2" : "%c1";
    const auto literal = variant == 7 ? "%c7" : variant >= 5 ? "%c5equiv" : lower;
    source += std::string("    scf.for %outer = ") + lower + " to " + upper +
        " step " + step + " {\n";
    source += std::string("      scf.for %inner = ") + lower + " to " + upper +
        " step " + step + " {\n";
    source += variant == 6 ? std::string("        %firstOuter = arith.cmpi eq, ") + literal +
        ", %outer : index\n" : std::string("        %firstOuter = arith.cmpi eq, %outer, ") +
        literal + " : index\n";
    source += std::string("        %firstInner = arith.cmpi eq, %inner, ") + literal + " : index\n";
    if (variant == 0 || variant == 3 || shifted) {
      source += variant != 3 ? "        %first = arith.andi %firstOuter, %firstInner : i1\n"
                             : "        %first = arith.ori %firstOuter, %firstInner : i1\n";
    }
    if (variant == 8) {
      source += "        %duplicate = arith.andi %first, %firstOuter : i1\n";
    }
    const char *condition = variant == 1 ? "%firstInner" : variant == 2 ? "%firstOuter" :
        variant == 8 ? "%duplicate" : "%first";
    source += std::string("        scf.if ") + condition + " {\n";
    source += original.substr(begin, middle - begin);
    source += "        } else {\n";
    source += original.substr(middle, end - middle);
    source += R"mlir(
        }
      }
    }
    pto.tstore
      ins(%c : !pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>)
      outs(%out : !pto.partition_tensor_view<128x256xf32>)
    }
)mlir" + original.substr(end);
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse nested first-use fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("matrix");
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    if (variant == 9) {
      loops[1]->setAttr("unsignedCmp", UnitAttr::get(&context));
    }
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
               "import nested first-use fixture")) return false;
    const bool qualifiedFirstUse = llvm::any_of(imported.observationNotes,
        [](const std::string &note) {
          return note == "qualified original first-use region prefix";
        });
    const bool expectedFirstUse = variant == 0 || (shifted && variant < 7);
    if (expectedFirstUse &&
        !check(qualifiedFirstUse, "shared loop domain did not qualify equivalent first-use role")) {
      return false;
    }
    const bool refusedFirstUse = variant == 3 || variant == 7 || variant == 8 || variant == 9;
    if (refusedFirstUse &&
        !check(!qualifiedFirstUse, "unsupported first-use predicate was incorrectly qualified")) {
      return false;
    }
    const auto &graph = *imported.program.observed;
    std::vector<bool> reached(graph.sites.size()), live(graph.observations.size());
    std::vector<oahs::Cut> pending{graph.entry};
    while (!pending.empty()) {
      const auto at = pending.back();
      pending.pop_back();
      if (reached[at]) {
        continue;
      }
      reached[at] = true;
      if (oahs::legalCommandCut(imported.program, at)) {
        live[graph.sites[at].observation] = true;
      }
      for (auto next : graph.sites[at].successors) {
        pending.push_back(next);
      }
    }
    for (oahs::Cut at = 0; at < graph.sites.size(); ++at) {
      const auto observation = graph.sites[at].observation;
      const bool available = observation != oahs::NoControlId && live[observation];
      if (!check(bool(imported.cuts[at]) == available,
                 "native anchors disagree with live command words, including canonical aliases")) {
        return false;
      }
    }
    oahs::Commands commands(oahs::commandCutCount(imported.program));
    for (oahs::Cut cut = 0; cut < commands.size(); ++cut) {
      if (imported.cuts[cut] == loops[1].getOperation()) {
        commands[cut] = {{oahs::Command::Publish, oahs::Pipe::FIX, oahs::Pipe::M, 0},
                         {oahs::Command::Acquire, oahs::Pipe::FIX, oahs::Pipe::M, 0}};
      } else if (isa_and_nonnull<TStoreOp>(imported.cuts[cut])) {
        commands[cut] = {{oahs::Command::Publish, oahs::Pipe::M, oahs::Pipe::FIX, 0},
                         {oahs::Command::Acquire, oahs::Pipe::M, oahs::Pipe::FIX, 0}};
      }
    }
    commands[oahs::invocationExitCut(imported.program)] = {{oahs::Command::BarrierAll}};
    const auto checked = oahs::checkCausalFrontier(imported.program, commands);
    if (!check(checked.accepted == expectedFirstUse,
               "first use or genuinely repeating initialization completion")) return false;
    if (!expectedFirstUse) {
      for (oahs::Cut cut = 0; cut < commands.size(); ++cut) {
        if (isa_and_nonnull<TMatmulOp>(imported.cuts[cut]))
          commands[cut].push_back({oahs::Command::Barrier, oahs::Pipe::M});
      }
      const auto repaired = oahs::checkCausalFrontier(imported.program, commands);
      if (!repaired.accepted) {
        llvm::errs() << "first-use variant=" << variant << " cut=" << repaired.cut
                     << " command=" << repaired.command << " reason=" << repaired.reason << "\n";
      }
      if (!check(repaired.accepted, "repeated initialization must retain a real completion repair")) {
        return false;
      }
    }
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
  {
    auto module = parseSourceString<ModuleOp>(input, &context);
    auto function = module->lookupSymbol<func::FuncOp>("addresses");
    OpBuilder builder(function.getBody().front().getTerminator());
    auto zero = builder.create<arith::ConstantIntOp>(function.getLoc(), 0, 64);
    std::vector<Value> chain{function.getArgument(0)};
    for (unsigned i = 0; i < 256; ++i)
      chain.push_back(builder.create<arith::AddIOp>(function.getLoc(), chain.back(), zero));
    for (bool reverse : {false, true}) {
      SyncSlotMapping::ConstantCache cache;
      for (unsigned i = 1; i < chain.size(); ++i) {
        const auto at = reverse ? chain.size() - i : i;
        if (!check(!SyncSlotMapping::evaluateConstant(chain[at], cache), "unknown address became constant")) return false;
      }
      if (!check(cache.evaluations == chain.size() && cache.notConstant.size() == chain.size(),
                 "failed address prefixes were repeatedly expanded")) return false;
      const auto work = cache.evaluations;
      llvm::DenseMap<Value, uint64_t> seeded;
      seeded[chain.front()] = 9;
      if (!check(SyncSlotMapping::evaluate(chain.back(), seeded) == std::optional<uint64_t>(9) &&
                 !SyncSlotMapping::evaluateConstant(chain.back(), cache) && cache.evaluations == work,
                 "unseeded negative facts leaked into a slot valuation")) return false;
      cache.clear();
      if (!check(cache.evaluations == 0 && cache.known.empty() && cache.notConstant.empty(),
                 "new import retained old scalar facts")) return false;
    }
    // Rejected arithmetic and unsupported expressions are memoized too.
    auto maximum = builder.create<arith::ConstantIntOp>(function.getLoc(), std::numeric_limits<int64_t>::max(), 64);
    auto one = builder.create<arith::ConstantIntOp>(function.getLoc(), 1, 64);
    Value overflow = builder.create<arith::AddIOp>(function.getLoc(), maximum, one);
    Value unsupported = builder.create<arith::DivUIOp>(function.getLoc(), one, zero);
    SyncSlotMapping::ConstantCache cache;
    if (!check(!SyncSlotMapping::evaluateConstant(overflow, cache) &&
               !SyncSlotMapping::evaluateConstant(unsupported, cache), "rejected scalar unexpectedly admitted")) return false;
    const auto work = cache.evaluations;
    for (unsigned repeat = 0; repeat < 32; ++repeat) {
      if (!check(!SyncSlotMapping::evaluateConstant(overflow, cache) &&
                 !SyncSlotMapping::evaluateConstant(unsupported, cache) && cache.evaluations == work,
                 "rejected expression was reevaluated")) return false;
    }
  }
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
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
    uint64_t expected = 24576;
    const bool expectKnown = mutation == 0 || (mutation >= 4 && mutation <= 6);
    if (mutation >= 4) {
      expected = mutation == 6 ? 64 : 256;
      auto input = builder.create<arith::ConstantIntOp>(sum.getLoc(),
          mutation >= 7 ? (mutation == 9 ? 256 : -1) : int64_t(expected), 32);
      Value converted;
      if (mutation == 4 || mutation == 7)
        converted = builder.create<arith::ExtSIOp>(sum.getLoc(), builder.getI64Type(), input);
      else if (mutation == 5 || mutation == 8)
        converted = builder.create<arith::ExtUIOp>(sum.getLoc(), builder.getI64Type(), input);
      else {
        auto narrow = builder.create<arith::TruncIOp>(sum.getLoc(), builder.getI8Type(), input);
        converted = builder.create<arith::ExtUIOp>(sum.getLoc(), builder.getI64Type(), narrow);
      }
      allocation.getAddrMutable().assign(converted);
    }
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)), "constant address import")) {
      return false;
    }
    bool known = false, unknown = false;
    for (const auto &cell : imported.program.cells) {
      unknown |= cell.unknownRange;
      known |= !cell.unknownRange && cell.ranges == std::vector<std::pair<uint64_t,uint64_t>>{{expected, 64}};
    }
    if (!check(expectKnown ? known && !unknown : !known && unknown,
               "constant address certainty or byte footprint incorrect")) {
      return false;
    }
  }
  return true;
}
bool exactCommandEmission(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>("module { func.func @exact() { return } }", &context);
  if (!check(bool(module), "exact-word emitter fixture failed to parse")) {
    return false;
  }
  auto function = *module->getOps<func::FuncOp>().begin();
  SyncIRs emission;
  SmallVector<std::unique_ptr<SyncOperation>> storage;
  auto anchor = std::make_unique<PlaceHolderInstanceElement>(0, 0);
  anchor->elementOp = function.getBody().front().getTerminator();
  const auto P = oahs::Pipe::MTE2, Q = oahs::Pipe::V;
  const std::vector<oahs::Command> expected{
      {oahs::Command::Publish, P, Q, 0}, {oahs::Command::Acquire, P, Q, 0},
      {oahs::Command::Publish, Q, P, 0}, {oahs::Command::Acquire, Q, P, 0},
      {oahs::Command::Publish, P, Q, 0}, {oahs::Command::Acquire, P, Q, 0},
      {oahs::Command::Publish, Q, P, 0}, {oahs::Command::Acquire, Q, P, 0},
      {oahs::Command::Barrier, Q}, {oahs::Command::Barrier, Q}};
  auto native = [&](oahs::Pipe pipe) {
    return pipe == P ? PipelineType::PIPE_MTE2 : PipelineType::PIPE_V;
  };
  for (const auto &command : expected) {
    const auto type = command.kind == oahs::Command::Publish ? SyncOperation::TYPE::SET_EVENT :
        command.kind == oahs::Command::Acquire ? SyncOperation::TYPE::WAIT_EVENT : SyncOperation::TYPE::PIPE_BARRIER;
    auto sync = std::make_unique<SyncOperation>(type, native(command.source), native(command.observer),
                                               storage.size(), 0, std::nullopt);
    sync->eventIds.push_back(command.key);
    anchor->pipeBefore.push_back(sync.get());
    storage.push_back(std::move(sync));
  }
  emission.push_back(std::move(anchor));
  SyncCodegen codegen(emission, function, SyncAnalysisMode::NORMALSYNC, SyncCodegen::CommandListPolicy::PreserveOrder);
  codegen.Run();
  std::vector<oahs::Command> actual;
  auto portable = [&](PIPE pipe) { return pipe == PIPE::PIPE_MTE2 ? P : Q; };
  function.walk([&](mlir::Operation *operation) {
    if (auto set = dyn_cast<SetFlagOp>(operation)) {
      actual.push_back({oahs::Command::Publish, portable(set.getSrcPipe().getPipe()),
          portable(set.getDstPipe().getPipe()), unsigned(set.getEventId().getEvent())});
    } else if (auto wait = dyn_cast<WaitFlagOp>(operation)) {
      actual.push_back({oahs::Command::Acquire, portable(wait.getSrcPipe().getPipe()),
          portable(wait.getDstPipe().getPipe()), unsigned(wait.getEventId().getEvent())});
    } else if (auto barrier = dyn_cast<BarrierOp>(operation)) {
      actual.push_back({oahs::Command::Barrier, portable(barrier.getPipe().getPipe())});
    }
  });
  return check(actual.size() == expected.size() &&
               std::equal(actual.begin(), actual.end(), expected.begin(), oahs::selected::identical) &&
               succeeded(verify(*module)), "shared emission merged or reordered a checked command word");
}

bool originalResidueDecisions(MLIRContext &context) {
  const char *source = R"mlir(module attributes {pto.target_arch = "a3"} {
    func.func @raw_residue(%src: !pto.partition_tensor_view<1x32xf32>)
        attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
      %lower = arith.constant 3 : index
      %upper = arith.constant 11 : index
      %step = arith.constant 2 : index
      %modulus = arith.constant 4 : index
      %a = arith.constant 256 : i64
      %b = arith.constant 4096 : i64
      %bank = pto.alloc_tile addr = %a : !pto.tile_buf<vec, 1x32xf32>
      %out = pto.alloc_tile addr = %b : !pto.tile_buf<vec, 1x32xf32>
      scf.for %i = %lower to %upper step %step {
        %residue = arith.remui %i, %modulus : index
        %first = arith.cmpi eq, %residue, %lower : index
        scf.if %first {
          pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
        } else {
          pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
        }
      }
      return
    }
  })mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!check(bool(module), "raw residue fixture failed to parse")) {
    return false;
  }
  auto function = *module->getOps<func::FuncOp>().begin();
  oahs::NativeAnalysis imported;
  if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
             "raw residue import failed")) {
    return false;
  }
  const auto &graph = *imported.program.observed;
  std::vector<bool> seen(graph.sites.size());
  std::vector<std::size_t> todo{graph.entry};
  bool witnessedLoad = false, witnessedRead = false;
  while (!todo.empty()) {
    const auto at = todo.back();
    todo.pop_back();
    if (seen[at]) {
      continue;
    }
    seen[at] = true;
    const auto &site = graph.sites[at];
    if (site.operation != oahs::NoControlId) {
      const auto pipe = imported.program.operations[site.operation].pipe;
      const auto &atoms = graph.observations[site.observation].atoms;
      const auto residue = std::find_if(atoms.begin(), atoms.end(), [](const auto &atom) {
        return atom.kind == oahs::ObservationAtom::LoopResidue;
      });
      if (!check(residue != atoms.end() && residue->parameter == 2 &&
                 residue->value == (pipe == oahs::Pipe::MTE2 ? 0u : 1u),
                 "raw IV residue was confused with logical iteration residue")) {
        return false;
      }
      witnessedLoad |= pipe == oahs::Pipe::MTE2;
      witnessedRead |= pipe == oahs::Pipe::V;
    }
    for (auto next : site.successors) {
      todo.push_back(next);
    }
  }
  oahs::SelectedPlan plan;
  return check(witnessedLoad && witnessedRead &&
               succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)) &&
               !plan.declinedObservation && !plan.declinedFirstUse,
               "normalized raw-residue protocol failed construction/reconstruction");
}

bool normalizedGuardReadback(MLIRContext &context) {
  const char *source = R"mlir(module {
    func.func @guards() {
      %lower = arith.constant 3 : index
      %upper = arith.constant 260 : index
      %step = arith.constant 128 : index
      scf.for %i = %lower to %upper step %step {}
      return
    }
  })mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!check(bool(module), "normalized guard fixture failed to parse")) {
    return false;
  }
  scf::ForOp loop;
  module->walk([&](scf::ForOp found) { loop = found; });
  auto *anchor = loop.getBody()->getTerminator();
  const SmallVector<std::pair<std::size_t, mlir::Operation *>> owners{{42, loop.getOperation()}};
  for (unsigned kind = 0; kind < 3; ++kind) {
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
      OpBuilder builder(anchor);
      const auto loc = anchor->getLoc();
      auto constant = [&](uint64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(loc, int64_t(value));
      };
      Value distance = builder.create<arith::SubIOp>(loc, loop.getInductionVar(),
          mutation == 1 ? loop.getStep() : loop.getLowerBound());
      Value ordinal = builder.create<arith::DivUIOp>(loc, distance,
          mutation == 2 ? loop.getLowerBound() : loop.getStep());
      const auto role = kind == 0 ? oahs::ObservationAtom::LoopResidue :
          kind == 1 ? oahs::ObservationAtom::LoopHasPrevious : oahs::ObservationAtom::LoopHasNext;
      oahs::OriginalObservation observation{0, {{role, 42, 2, 1}}, true};
      Value condition;
      if (kind == 0) {
        Value residue = builder.create<arith::RemUIOp>(loc, ordinal, constant(mutation == 3 ? 3 : 2));
        condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, residue, constant(1));
      } else if (kind == 1) {
        condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
            ordinal, constant(mutation == 3 ? 3 : 2));
      } else {
        Value remaining = builder.create<arith::SubIOp>(loc,
            mutation == 1 ? loop.getLowerBound() : loop.getUpperBound(),
            mutation == 2 ? ordinal : loop.getInductionVar());
        condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
            remaining, constant(mutation == 3 ? 2 : 256));
      }
      const auto status = oahs::testing::checkHandoffObservationPredicate(observation, anchor, owners, condition);
      if (!check(succeeded(status) == (mutation == 0), "normalized guard mutation was not discriminated")) {
        return false;
      }
    }
  }
  for (unsigned polarity = 0; polarity < 2; ++polarity) {
    oahs::OriginalObservation observation{0, {{oahs::ObservationAtom::LoopHasNext, 42,
        uint64_t(std::numeric_limits<int64_t>::max()), polarity}}, true};
    OpBuilder builder(anchor);
    Value good = builder.create<arith::ConstantIntOp>(anchor->getLoc(), !polarity, 1);
    Value bad = builder.create<arith::ConstantIntOp>(anchor->getLoc(), polarity, 1);
    if (!check(succeeded(oahs::testing::checkHandoffObservationPredicate(observation, anchor, owners, good)) &&
               failed(oahs::testing::checkHandoffObservationPredicate(observation, anchor, owners, bad)),
               "unrepresentable next distance did not preserve both predicate polarities")) {
      return false;
    }
  }
  return check(succeeded(verify(*module)), "guard tests produced invalid IR");
}

bool originalLoopDomains(MLIRContext &context) {
  struct Case {
    int64_t lower, upper, step;
    bool dynamic, bounded, accepted, nonempty;
  };
  const auto maximum = std::numeric_limits<int64_t>::max();
  const std::vector<Case> cases{
      {0, 256, 128, false, false, true, true},
      {3, 260, 128, false, false, true, true},
      {3, 3, 128, false, false, true, false},
      {3, 4, 128, false, false, true, true},
      {0, maximum, 2, false, false, false, false},
      {1, maximum, 2, false, false, true, true},
      {0, 0, 1, true, false, true, false},
      {0, 0, 2, true, false, false, false},
      {1, 0, 2, true, false, true, false},
      {0, 256, 128, true, true, true, false},
      {-1, 2, 1, false, false, false, false},
      {0, -1, 1, false, false, true, false}};
  for (const auto &c : cases) {
    std::string source = "module { func.func @domain(%n: index) {\n";
    source += "%lower = arith.constant " + std::to_string(c.lower) + " : index\n";
    source += "%upper = arith.constant " + std::to_string(c.upper) + " : index\n";
    source += "%step = arith.constant " + std::to_string(c.step) + " : index\n";
    source += "%bounded = arith.minsi %n, %upper : index\n";
    source += "scf.for %i = %lower to " + std::string(c.bounded ? "%bounded" : c.dynamic ? "%n" : "%upper") +
              " step %step {}\nreturn } }";
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "loop-domain fixture failed to parse")) {
      return false;
    }
    scf::ForOp loop;
    module->walk([&](scf::ForOp found) { loop = found; });
    SyncSlotMapping::ConstantCache constants;
    SyncSlotMapping::RangeCache ranges;
    const auto domain = SyncSlotMapping::originalLoopDomain(loop, constants, ranges);
    if (!check(bool(domain) == c.accepted, "original loop progression overflow proof differs")) {
      return false;
    }
    if (domain && !check(domain->lower == uint64_t(c.lower) && domain->step == uint64_t(c.step) &&
                         domain->atLeastOnce == c.nonempty, "original domain changed iteration coordinates")) {
      return false;
    }
    const auto work = ranges.evaluations;
    SyncSlotMapping::originalLoopDomain(loop, constants, ranges);
    if (!check(ranges.evaluations == work, "shared scalar ranges were recomputed")) {
      return false;
    }
    if (domain && c.step > 1 && !check(!domain->distance(uint64_t(maximum)),
                                      "overflowing future-visit distance accepted")) {
      return false;
    }
  }
  return true;
}

bool readerGenerations(MLIRContext &context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @reader_generations(%src: !pto.partition_tensor_view<1x32xf32>,
      %dst: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %stride = arith.constant 128 : index
    %base = arith.constant 256 : index
    %out_addr = arith.constant 4096 : i64
    %out = pto.alloc_tile addr = %out_addr : !pto.tile_buf<vec, 1x32xf32>
    scf.for %i = %zero to %four step %one {
      %slot = arith.remui %i, %two : index
      %offset = arith.muli %slot, %stride : index
      %address = arith.addi %offset, %base : index
      %cast = arith.index_cast %address : index to i64
      %bank = pto.alloc_tile addr = %cast : !pto.tile_buf<vec, 1x32xf32>
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
      scf.for %j = %zero to %two step %one {
        pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
        pto.tstore ins(%out : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)
      }
      // RELOAD
      scf.for %k = %zero to %two step %one {
        pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
        pto.tstore ins(%out : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)
      }
    }
    return
  }
})mlir";
  std::pair<unsigned, unsigned> typedCounts;
  for (unsigned variant = 0; variant < 12; ++variant) {
    std::string source = input;
    if (variant == 1) {
      source.replace(source.find("// RELOAD"), std::string("// RELOAD").size(),
          "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) "
          "outs(%bank : !pto.tile_buf<vec, 1x32xf32>)");
    }
    if (variant >= 2) {
      const std::string store = "        pto.tstore ins(%out : !pto.tile_buf<vec, 1x32xf32>) "
                                "outs(%dst : !pto.partition_tensor_view<1x32xf32>)\n";
      for (auto at = source.find(store); at != std::string::npos; at = source.find(store)) {
        source.erase(at, store.size());
      }
    }
    if (variant == 3) {
      const std::string loop = "scf.for %j = %zero to %two step %one {";
      const auto at = source.find(loop);
      source.replace(at, loop.size(),
          "%unused = scf.for %j = %zero to %two step %one iter_args(%counter = %zero) -> index {");
      const auto end = source.find("      }", at);
      source.insert(end, "        %next_counter = arith.addi %counter, %one : index\n"
                         "        scf.yield %next_counter : index\n");
    }
    if (variant >= 4) {
      const bool shifted = variant >= 5;
      const unsigned upper = variant == 6 ? 4 : variant == 7 ? 3 : shifted ? 260 : 256;
      const std::string declarations =
          "%child_lower_literal = arith.constant " + std::string(shifted ? "3" : "0") + " : index\n" +
          "%child_upper = arith.constant " + std::to_string(upper) + " : index\n" +
          "%child_step_literal = arith.constant 128 : index\n" +
          (variant == 8 ? "%child_lower = arith.addi %child_lower_literal, %zero : index\n"
                          "%child_step = arith.muli %child_step_literal, %one : index\n" : "");
      source.insert(source.find("    scf.for %i"), declarations);
      for (const std::string iv : {"j", "k"}) {
        const std::string old = "scf.for %" + iv + " = %zero to %two step %one";
        source.replace(source.find(old), old.size(), "scf.for %" + iv + " = " +
            (variant == 8 ? "%child_lower" : "%child_lower_literal") + " to %child_upper step " +
            (variant == 8 ? "%child_step" : "%child_step_literal"));
      }
    }
    if (variant >= 9) {
      const std::string selector = "%slot = arith.remui %i, %two : index";
      source.replace(source.find(selector), selector.size(), "%slot = arith.constant 0 : index");
    }
    if (variant == 10) {
      const auto at = source.find("      pto.tload");
      source.insert(at, "      %bank_view = pto.treshape %bank : !pto.tile_buf<vec, 1x32xf32> "
                        "-> !pto.tile_buf<vec, 1x32xf32>\n");
      const std::string use = "pto.tabs ins(%bank :";
      for (auto pos = source.find(use); pos != std::string::npos; pos = source.find(use, pos + 1)) {
        source.replace(pos, use.size(), "pto.tabs ins(%bank_view :");
      }
    }
    if (variant == 11) {
      const auto at = source.find("    scf.for %i");
      source.insert(at, "    %predicate = arith.cmpi eq, %zero, %one : index\n"
                        "    %width = arith.constant 32 : index\n"
                        "    %spare_addr = arith.constant 8192 : i64\n"
                        "    %spare = pto.alloc_tile addr = %spare_addr valid_row = %one valid_col = %width : "
                        "!pto.tile_buf<vec, 1x32xf32, valid=?x?>\n"
                        "    scf.if %predicate {\n"
                        "      pto.set_validshape %spare, %one, %width : !pto.tile_buf<vec, 1x32xf32, valid=?x?>\n"
                        "    }\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "reader-generation fixture failed to parse")) {
      return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    if (variant >= 9) {
      oahs::NativeAnalysis imported;
      if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
                 "typed reader fixture failed shared import")) { return false; }
      oahs::selected::Control control(imported.program);
      oahs::StorageFrontierAnalysis storage(imported.program);
      oahs::selected::RequirementFrontiers facts(imported.program, control, storage);
      unsigned first = 0, final = 0;
      for (const auto& loop : imported.program.observed->loops) {
        for (auto site : loop.sites) {
          const auto operation = control.graph.operations[site];
          if (operation == oahs::NoAnalysisId) { continue; }
          for (const auto& access : imported.program.operations[operation].accesses) {
            if (!access.read || access.write) { continue; }
            const auto& endpoints = facts.readerBoundaries(site, access.cell, loop.owner);
            if (endpoints.originalInterval) {
              first += endpoints.first.hit(); final += endpoints.final.hit();
            }
          }
        }
      }
      if (!check(first && final, "native enclosing readers did not consume typed endpoint frontiers")) { return false; }
      if (variant == 9) { typedCounts = {first, final}; }
      if (!check(typedCounts == std::make_pair(first, final),
                 "equivalent view or unrelated descriptor/control changed typed endpoint facts")) { return false; }
    }
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "reader generation failed native construction/reconstruction")) {
      return false;
    }
    if (!check(!plan.declinedObservation && !plan.declinedFirstUse && !plan.declinedRecurring,
               "reader generation depended on an optional fallback")) {
      return false;
    }
    const auto ready = std::count_if(plan.channels.begin(), plan.channels.end(), [](const auto& channel) {
      return channel.source == oahs::Pipe::MTE2 && channel.observer == oahs::Pipe::V && channel.period == 2;
    });
    const bool expectedReady = variant >= 9 || ready == (variant == 7 ? 0 : 2);
    if (!check(expectedReady, "physical generation across children lost per-bank readiness")) {
      llvm::errs() << "reader variant=" << variant << " ready=" << ready << "\n";
      return false;
    }
  }
  return true;
}

bool optionalReaderParticipation(MLIRContext& context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @optional_readers(%src: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %four = arith.constant 4 : index
    %address = arith.constant 0 : i64
    %out_address = arith.constant 4096 : i64
    %bank = pto.alloc_tile addr = %address : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %out_address : !pto.tile_buf<vec, 1x32xf32>
    scf.for %i = %zero to %four step %one {
      %optional = arith.remui %i, %three : index
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
      scf.for %j = %zero to %two step %one {
        pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
      }
      // LATE
      scf.for %k = %zero to %optional step %one {
        pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 6; ++variant) {
    auto source = input;
    if (variant == 1) {
      const std::string declaration = "%optional = arith.remui %i, %three : index";
      source.replace(source.find(declaration), declaration.size(),
          "%optional_raw = arith.remui %i, %three : index\n"
          "      %optional = arith.addi %optional_raw, %zero : index");
      source.insert(source.find("    scf.for %i"),
          "    %view = pto.treshape %bank : !pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x32xf32>\n");
      const std::string use = "pto.tabs ins(%bank :";
      for (auto at = source.find(use); at != std::string::npos; at = source.find(use, at + 1)) {
        source.replace(at, use.size(), "pto.tabs ins(%view :");
      }
    }
    if (variant == 2) {
      const std::string declaration = "      %optional = arith.remui %i, %three : index\n";
      source.erase(source.find(declaration), declaration.size());
      source.insert(source.find("      // LATE"), declaration);
    }
    if (variant == 3) {
      source.insert(source.find("    scf.for %i"),
          "    %spare_address = arith.constant 8192 : i64\n"
          "    %spare = pto.alloc_tile addr = %spare_address : !pto.tile_buf<vec, 1x32xf32>\n");
      source.insert(source.find("      scf.for %j"),
          "      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) "
          "outs(%spare : !pto.tile_buf<vec, 1x32xf32>)\n");
    }
    if (variant == 4) {
      source.insert(source.find("      // LATE"),
          "      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) "
          "outs(%bank : !pto.tile_buf<vec, 1x32xf32>)\n");
    }
    if (variant == 5) {
      source.insert(source.find("    }\n    return"),
          "      pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) "
          "outs(%out : !pto.tile_buf<vec, 1x32xf32>)\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "optional reader fixture parse")) { return false; }
    auto function = *module->getOps<func::FuncOp>().begin();
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
               "optional reader shared import")) { return false; }
    bool qualified = false;
    for (const auto& observation : imported.program.observed->observations) {
      for (const auto& atom : observation.atoms) { qualified |= atom.kind == oahs::ObservationAtom::LoopNonEmpty; }
    }
    const bool expectedParticipation = variant < 2 || variant == 3;
    if (!check(qualified == expectedParticipation, "optional reader original bound availability")) {
      return false;
    }
    SmallVector<scf::ForOp> loops;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
    auto sibling = loops[2];
    auto* anchor = loops[1].getOperation();
    OpBuilder builder(anchor);
    auto good = builder.create<arith::CmpIOp>(anchor->getLoc(), arith::CmpIPredicate::slt,
                                             sibling.getLowerBound(), sibling.getUpperBound());
    auto wrong = builder.create<arith::CmpIOp>(anchor->getLoc(), arith::CmpIPredicate::sge,
                                              sibling.getLowerBound(), sibling.getUpperBound());
    const oahs::OriginalObservation observation{0, {{oahs::ObservationAtom::LoopNonEmpty, 42, 0, 1}}, true};
    const SmallVector<std::pair<std::size_t, mlir::Operation*>> owners{{42, sibling.getOperation()}};
    bool guardChecks;
    {
      ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
      const bool exact = succeeded(oahs::testing::checkHandoffObservationPredicate(
          observation, anchor, owners, good));
      const bool mutation = failed(oahs::testing::checkHandoffObservationPredicate(
          observation, anchor, owners, wrong));
      guardChecks = exact == (variant != 2) && mutation;
    }
    good.erase(); wrong.erase();
    if (!check(guardChecks, "original sibling predicate readback accepted unavailable bounds or wrong polarity")) {
      return false;
    }
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "optional reader native reconstruction")) { return false; }
    if (expectedParticipation && !check(!plan.declinedObservation && !plan.declinedFirstUse &&
                                        !plan.declinedRecurring &&
                              !plan.activations.empty() && plan.channels.size() == 2,
                              "optional children require one activated readiness/return family")) { return false; }
  }
  return true;
}

bool sourceGapPlacement(MLIRContext &context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @source_gap(%src: !pto.partition_tensor_view<1x32xi32>,
                       %dst: !pto.partition_tensor_view<1x32xi32>, %predicate: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %a0 = arith.constant 0 : i64
    %a1 = arith.constant 256 : i64
    %a2 = arith.constant 512 : i64
    %a3 = arith.constant 768 : i64
    %zero = arith.constant 0 : i32
    %z = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 1x32xi32>
    %x = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 1x32xi32>
    %y = pto.alloc_tile addr = %a2 : !pto.tile_buf<vec, 1x32xi32>
    %w = pto.alloc_tile addr = %a3 : !pto.tile_buf<vec, 1x32xi32>
    pto.tci ins(%zero : i32) outs(%z : !pto.tile_buf<vec, 1x32xi32>) {descending = false}
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xi32>) outs(%y : !pto.tile_buf<vec, 1x32xi32>)
    pto.tadd ins(%z, %z : !pto.tile_buf<vec, 1x32xi32>, !pto.tile_buf<vec, 1x32xi32>)
      outs(%x : !pto.tile_buf<vec, 1x32xi32>)
    pto.tadd ins(%y, %y : !pto.tile_buf<vec, 1x32xi32>, !pto.tile_buf<vec, 1x32xi32>)
      outs(%w : !pto.tile_buf<vec, 1x32xi32>)
    pto.tstore ins(%x : !pto.tile_buf<vec, 1x32xi32>) outs(%dst : !pto.partition_tensor_view<1x32xi32>)
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto source = input;
    auto replace = [&](const std::string& from, const std::string& to) {
      source.replace(source.find(from), from.size(), to);
    };
    if (variant == 1) {
      replace("%x = pto.alloc_tile", "%allocation = pto.alloc_tile");
      replace("%y = pto.alloc_tile", "%x = pto.treshape %allocation : "
          "!pto.tile_buf<vec, 1x32xi32> -> !pto.tile_buf<vec, 1x32xi32>\n    %y = pto.alloc_tile");
    }
    if (variant == 2) {
      replace("%z = pto.alloc_tile", "scf.if %predicate { }\n"
          "    %unrelated = arith.addi %a0, %a1 : i64\n    %z = pto.alloc_tile");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    const bool parsed = check(bool(module) && succeeded(verify(*module)),
                              "source-gap native fixture failed to parse");
    if (!parsed) {
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("source_gap");
    oahs::SelectedPlan plan;
    const bool constructed = check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)) &&
                   plan.work.earlyPublications != 0 && !plan.declinedObservation &&
                   !plan.declinedFirstUse && !plan.declinedRecurring,
               "native exact-gap construction/reconstruction missed early publication");
    if (!constructed) {
      return false;
    }
  }
  return true;
}

bool milestoneKeyBinding(MLIRContext& context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @milestone_binding(%src: !pto.partition_tensor_view<1x32xf32>, %predicate: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %a0 = arith.constant 0 : i64
    %a1 = arith.constant 128 : i64
    %a2 = arith.constant 256 : i64
    %a3 = arith.constant 384 : i64
    %a4 = arith.constant 512 : i64
    %old = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 1x32xf32>
    %first = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 1x32xf32>
    %spare = pto.alloc_tile addr = %a2 : !pto.tile_buf<vec, 1x32xf32>
    %x = pto.alloc_tile addr = %a3 : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %a4 : !pto.tile_buf<vec, 1x32xf32>
    pto.tabs ins(%old : !pto.tile_buf<vec, 1x32xf32>) outs(%first : !pto.tile_buf<vec, 1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%old : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%spare : !pto.tile_buf<vec, 1x32xf32>) outs(%x : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%old : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%x : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto source = input;
    if (variant == 1) {
      const auto at = source.find("    pto.tabs ins(%old");
      source.insert(at, "    %view = pto.treshape %x : !pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x32xf32>\n");
      const auto use = source.find("outs(%x");
      source.replace(use, std::string("outs(%x").size(), "outs(%view");
    }
    if (variant == 2) {
      const auto at = source.find("    pto.tabs ins(%old");
      source.insert(at, "    %unused = arith.addi %a0, %a4 : i64\n    scf.if %predicate { }\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    const bool parsed = bool(module) && succeeded(verify(*module));
    if (!check(parsed, "native milestone fixture parse")) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("milestone_binding");
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "native milestone construction/reconstruction")) { return false; }
    const bool higher = std::any_of(plan.ledger.begin(), plan.ledger.end(), [](const auto& endpoint) {
      return endpoint.command.kind == oahs::Command::Publish && endpoint.command.source == oahs::Pipe::V &&
          endpoint.command.observer == oahs::Pipe::MTE2 && endpoint.command.key == 1;
    });
    if (!check(plan.work.earlyPublications != 0 && higher,
               "native key selection lost the useful pre-receipt source milestone")) { return false; }
  }
  return true;
}

bool atomicAcknowledgment(MLIRContext& context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @atomic_ack(%src: !pto.partition_tensor_view<1x32xf32>, %choose: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %a0 = arith.constant 0 : i64
    %a1 = arith.constant 128 : i64
    %a2 = arith.constant 256 : i64
    %a3 = arith.constant 384 : i64
    %x = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 1x32xf32>
    %y = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %a2 : !pto.tile_buf<vec, 1x32xf32>
    %spare = pto.alloc_tile addr = %a3 : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%x : !pto.tile_buf<vec, 1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%y : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%x : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%y : !pto.tile_buf<vec, 1x32xf32>) outs(%spare : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto source = input;
    if (variant == 1) {
      const auto at = source.find("    pto.tabs ins(%y");
      source.insert(at, "    %view = pto.treshape %y : !pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x32xf32>\n");
      const auto use = source.find("pto.tabs ins(%y");
      source.replace(use, std::string("pto.tabs ins(%y").size(), "pto.tabs ins(%view");
    }
    if (variant == 2) {
      const auto at = source.find("    pto.tload");
      source.insert(at, "    %unused = arith.addi %a1, %a2 : i64\n    scf.if %choose { }\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    const bool parsed = bool(module) && succeeded(verify(*module));
    if (!check(parsed, "native atomic acknowledgment parse")) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("atomic_ack");
    oahs::NativeAnalysis imported;
    const bool importedOK = succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported));
    if (!check(importedOK, "native atomic acknowledgment import")) { return false; }
    // Explicit scarce resource profile; retain the original imported effects.
    const auto P = oahs::Pipe::MTE2, Q = oahs::Pipe::V;
    imported.program.target.keys[unsigned(P)][unsigned(Q)] = {0};
    imported.program.target.keys[unsigned(Q)][unsigned(P)] = {0};
    const auto plan = oahs::constructSelectedPlan(imported.program);
    if (!check(plan.success && oahs::checkCausalFrontier(imported.program, plan.commands).accepted,
               "native atomic acknowledgment construction")) { return false; }
    const bool tracked = plan.work.publicationSupportSites != 0 &&
        std::any_of(plan.publicationSupport.begin(), plan.publicationSupport.end(), [](const auto& contract) {
          return contract.admittedRoot != oahs::NoAnalysisId && contract.preserved;
        });
    if (!check(tracked, "native equivalent views/control lost publication source contracts")) { return false; }
    const auto repair = std::find_if(plan.decisions.begin(), plan.decisions.end(), [](const auto& decision) {
      return decision.repairedAcquisition != oahs::NoAnalysisId;
    });
    const bool completeRepair = repair != plan.decisions.end() && repair->endpoints.size() == 4;
    if (!check(completeRepair, "native scarcity did not select the complete ordinary repair")) { return false; }
    const auto updates = std::count_if(plan.updates.begin(), plan.updates.end(), [&](const auto& update) {
      return update.version > repair->repairInputVersion && update.version <= repair->repairOutputVersion;
    });
    if (!check(updates == 1, "native acknowledgment selected an incomplete intermediate packet")) { return false; }
    auto missing = plan.commands;
    const auto& wait = plan.ledger[repair->endpoints[1]];
    auto& word = missing[wait.cut];
    const auto at = std::find_if(word.begin(), word.end(), [&](const auto& command) {
      return oahs::selected::identical(command, wait.command);
    });
    if (!check(at != word.end(), "native atomic receipt missing")) { return false; }
    word.erase(at);
    if (!check(!oahs::checkCausalFrontier(imported.program, missing).accepted,
               "native republication used anticipated acknowledgment credit")) { return false; }
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function)),
               "native atomic fixture reconstruction under real target pools")) { return false; }
  }
  return true;
}

bool restorationDeadlines(MLIRContext& context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @restore_deadline(%src: !pto.partition_tensor_view<1x32xf32>,
      %dst: !pto.partition_tensor_view<1x32xf32>, %choose: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %a0 = arith.constant 0 : i64
    %a1 = arith.constant 128 : i64
    %a2 = arith.constant 256 : i64
    %a3 = arith.constant 384 : i64
    %a4 = arith.constant 512 : i64
    %x = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 1x32xf32>
    %z = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 1x32xf32>
    %y = pto.alloc_tile addr = %a2 : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %a3 : !pto.tile_buf<vec, 1x32xf32>
    %spare = pto.alloc_tile addr = %a4 : !pto.tile_buf<vec, 1x32xf32>
    pto.tabs ins(%spare : !pto.tile_buf<vec, 1x32xf32>) outs(%spare : !pto.tile_buf<vec, 1x32xf32>)
    scf.if %choose {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%x : !pto.tile_buf<vec, 1x32xf32>)
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%z : !pto.tile_buf<vec, 1x32xf32>)
    } else {
    }
    pto.tabs ins(%x : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    pto.tstore ins(%z : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%y : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%y : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto source = input;
    if (variant == 1) {
      const auto at = source.find("    pto.tabs ins(%x");
      source.insert(at, "    %view = pto.treshape %x : !pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x32xf32>\n");
      const auto use = source.find("pto.tabs ins(%x");
      source.replace(use, std::string("pto.tabs ins(%x").size(), "pto.tabs ins(%view");
    }
    if (variant == 2) {
      const auto at = source.find("    scf.if %choose");
      source.insert(at, "    %unused = arith.addi %a1, %a2 : i64\n    scf.if %choose { }\n");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    const bool parsed = bool(module) && succeeded(verify(*module));
    if (!check(parsed, "native restoration fixture parse")) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("restore_deadline");
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
               "native restoration fixture import")) { return false; }
    // A restricted resource profile exercises scarcity without changing any
    // imported effects, occurrence facts or construction policy.
    const auto P = oahs::Pipe::MTE2, Q = oahs::Pipe::V;
    imported.program.target.keys[unsigned(P)][unsigned(Q)] = {0};
    imported.program.target.keys[unsigned(Q)][unsigned(P)] = {0};
    const auto plan = oahs::constructSelectedPlan(imported.program);
    if (!check(plan.success && oahs::checkCausalFrontier(imported.program, plan.commands).accepted,
               "native scarcity profile failed construction/checking")) { return false; }
    const bool activated = plan.work.deadlineRestorations == 1 && plan.restorations.size() == 1;
    if (!check(activated, "native equivalent spelling lost actual-deadline restoration")) { return false; }
    const auto& restored = plan.restorations.front();
    const auto& wait = plan.ledger[restored.acquisition];
    auto missing = plan.commands;
    auto& word = missing[wait.cut];
    const auto at = std::find_if(word.begin(), word.end(), [&](const auto& command) {
      return oahs::selected::identical(command, wait.command);
    });
    if (!check(at != word.end(), "native restoration lost its emitted wait")) { return false; }
    word.erase(at);
    if (!check(!oahs::checkCausalFrontier(imported.program, missing).accepted,
               "native reused key accepted without restored consumption")) { return false; }
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function)),
               "native restoration fixture reconstruction under target profile")) { return false; }
  }
  return true;
}

bool requiredReturnCoverage(MLIRContext &context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @required_return(%src: !pto.partition_tensor_view<1x32xi32>,
                       %dst: !pto.partition_tensor_view<1x32xi32>, %predicate: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %a0 = arith.constant 0 : i64
    %a1 = arith.constant 256 : i64
    %a2 = arith.constant 512 : i64
    %a3 = arith.constant 768 : i64
    %zero = arith.constant 0 : i32
    %z = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 1x32xi32>
    %x = pto.alloc_tile addr = %a1 : !pto.tile_buf<vec, 1x32xi32>
    %y = pto.alloc_tile addr = %a2 : !pto.tile_buf<vec, 1x32xi32>
    %w = pto.alloc_tile addr = %a3 : !pto.tile_buf<vec, 1x32xi32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xi32>) outs(%x : !pto.tile_buf<vec, 1x32xi32>)
    pto.tadd ins(%x, %x : !pto.tile_buf<vec, 1x32xi32>, !pto.tile_buf<vec, 1x32xi32>)
      outs(%y : !pto.tile_buf<vec, 1x32xi32>)
    pto.tstore ins(%y : !pto.tile_buf<vec, 1x32xi32>) outs(%dst : !pto.partition_tensor_view<1x32xi32>)
    pto.tload ins(%dst : !pto.partition_tensor_view<1x32xi32>) outs(%x : !pto.tile_buf<vec, 1x32xi32>)
    return
  }
})mlir";
  for (unsigned variant = 0; variant < 3; ++variant) {
    auto source = input;
    auto replace = [&](const std::string& from, const std::string& to) {
      source.replace(source.find(from), from.size(), to);
    };
    if (variant == 1) {
      replace("%x = pto.alloc_tile", "%allocation = pto.alloc_tile");
      replace("%y = pto.alloc_tile", "%x = pto.treshape %allocation : "
          "!pto.tile_buf<vec, 1x32xi32> -> !pto.tile_buf<vec, 1x32xi32>\n    %y = pto.alloc_tile");
    }
    if (variant == 2) {
      replace("%z = pto.alloc_tile", "scf.if %predicate { }\n"
          "    %unrelated = arith.addi %a0, %a1 : i64\n    %z = pto.alloc_tile");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    const bool parsed = check(bool(module) && succeeded(verify(*module)),
                              "required-return native fixture failed to parse");
    if (!parsed) {
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("required_return");
    oahs::SelectedPlan plan;
    const bool constructed = check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)) &&
                   std::any_of(plan.decisions.begin(), plan.decisions.end(),
                       [](const auto& decision) { return !decision.supporting.empty(); }) &&
                   !plan.declinedObservation && !plan.declinedFirstUse && !plan.declinedRecurring,
               "native construction missed actual required-return coverage");
    if (!constructed) {
      return false;
    }
  }
  return true;
}

bool uniformEndpointRoles(MLIRContext &context) {
  const char *input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @uniform(%src: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %a = arith.constant 256 : i64
    %b = arith.constant 4096 : i64
    %bank = pto.alloc_tile addr = %a : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %b : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
    pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    scf.for %i = %zero to %two step %one {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%bank : !pto.tile_buf<vec, 1x32xf32>)
      pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    }
    return
  }
})mlir";
  auto module = parseSourceString<ModuleOp>(input, &context);
  if (!check(bool(module), "uniform endpoint fixture failed to parse")) {
    return false;
  }
  auto function = *module->getOps<func::FuncOp>().begin();
  oahs::NativeAnalysis imported;
  if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
             "uniform endpoint import failed")) {
    return false;
  }
  for (const auto &observation : imported.program.observed->observations) {
    for (const auto &atom : observation.atoms) {
      if (!check(atom.kind != oahs::ObservationAtom::LoopHasPrevious &&
                 atom.kind != oahs::ObservationAtom::LoopHasNext,
                 "uniform roles unnecessarily split original control")) {
        return false;
      }
    }
  }
  return check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function)),
               "uniform roles failed native construction/reconstruction");
}

bool slotDependencySlices(MLIRContext &context) {
  const std::string input = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @slice(%src: !pto.partition_tensor_view<1x32xf32>, %n: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %stride = arith.constant 128 : index
    %base = arith.constant 256 : index
    %out_addr = arith.constant 4096 : i64
    %out = pto.alloc_tile addr = %out_addr : !pto.tile_buf<vec, 1x32xf32>
    %result = scf.for %i = %zero to %n step %one iter_args(%slot = %zero) -> index {
      %advance = arith.addi %slot, %one : index
      %next = arith.remui %advance, %two : index
      %selected = arith.addi %slot, %zero : index
      %offset = arith.muli %selected, %stride : index
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
  for (unsigned variant = 0; variant < 20; ++variant) {
    std::string source = input;
    auto replace = [&](const std::string &from, const std::string &to) {
      const auto at = source.find(from);
      if (at == std::string::npos) {
        return false;
      }
      source.replace(at, from.size(), to);
      return true;
    };
    if (variant == 1) {
      replace("%result = scf.for", "%result:2 = scf.for");
      replace("iter_args(%slot = %zero) -> index", "iter_args(%slot = %zero, %unrelated = %n) -> (index, index)");
      replace("scf.yield %next : index", "scf.yield %next, %unrelated : index, index");
    }
    if (variant == 2 || variant == 3 || variant == 10 || variant == 11 || variant == 16) {
      replace("%result = scf.for", "scf.for");
      replace(" iter_args(%slot = %zero) -> index", "");
      replace("%advance = arith.addi %slot, %one : index", "");
      replace("%next = arith.remui %advance, %two : index", "");
      replace("%selected = arith.addi %slot, %zero : index", variant != 3 ?
          "%selected = arith.remui %i, %two : index" : "%selected = arith.andi %i, %one : index");
      replace("scf.yield %next : index", "");
      if (variant == 10 || variant == 11) {
        const std::string identity = variant == 10 ?
            "%identity = arith.addi %i, %zero : index\n" :
            "%cast_iv = arith.index_cast %i : index to i64\n"
            "      %identity = arith.index_cast %cast_iv : i64 to index\n";
        replace("%selected = arith.remui %i, %two : index",
            identity + "      %selected = arith.remui %identity, %two : index");
      }
    }
    if (variant == 4) {
      replace("%next = arith.remui %advance, %two : index", "%next = arith.xori %slot, %one : index");
      replace("%selected = arith.addi %slot, %zero : index", "%selected = arith.subi %slot, %zero : index");
    }
    if (variant == 5) {
      replace("scf.yield %next : index", R"mlir(
      %unrelated = arith.remui %i, %three : index
      %predicate = arith.cmpi eq, %unrelated, %zero : index
      scf.if %predicate { }
      scf.yield %next : index)mlir");
    }
    if (variant == 6 || variant == 17) {
      replace("%bank = pto.alloc_tile", "%allocation = pto.alloc_tile");
      replace("      pto.tload", "      %bank = pto.treshape %allocation : "
          "!pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x32xf32>\n      pto.tload");
    }
    if (variant == 7 || variant >= 18) {
      replace("%two = arith.constant 2", "%two = arith.constant 17");
    }
    if (variant == 8) {
      replace("%next = arith.remui %advance, %two : index", "%next = arith.addi %advance, %n : index");
    }
    if (variant == 9) {
      replace("%stride = arith.constant 128", "%stride = arith.constant 64");
    }
    if (variant == 12) {
      replace("%result = scf.for", "%result:2 = scf.for");
      replace("iter_args(%slot = %zero) -> index",
          "iter_args(%slot = %zero, %other = %zero) -> (index, index)");
      replace("scf.yield %next : index", R"mlir(
      %other_advance = arith.addi %other, %one : index
      %other_next = arith.remui %other_advance, %three : index
      %other_base = arith.constant 2048 : index
      %other_offset = arith.muli %other, %stride : index
      %other_address = arith.addi %other_offset, %other_base : index
      %other_cast = arith.index_cast %other_address : index to i64
      %other_bank = pto.alloc_tile addr = %other_cast : !pto.tile_buf<vec, 1x32xf32>
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%other_bank : !pto.tile_buf<vec, 1x32xf32>)
      pto.tabs ins(%other_bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
      scf.yield %next, %other_next : index, index)mlir");
    }
    if (variant == 14) {
      replace("%result = scf.for", "%result:2 = scf.for");
      replace("iter_args(%slot = %zero) -> index",
          "iter_args(%slot = %zero, %other = %zero) -> (index, index)");
      replace("%selected = arith.addi %slot, %zero : index",
          "%weighted = arith.muli %other, %two : index\n"
          "      %selected = arith.addi %slot, %weighted : index");
      replace("scf.yield %next : index",
          "%other_advance = arith.addi %other, %one : index\n"
          "      %other_next = arith.remui %other_advance, %three : index\n"
          "      scf.yield %next, %other_next : index, index");
    }
    if (variant == 13) {
      replace("to %n step %one iter_args", "to %n step %two iter_args");
      replace("scf.yield %next : index", "scf.for %child = %zero to %one step %one { }\n      scf.yield %next : index");
    }
    if (variant >= 15) {
      replace("pto.tabs ins(%bank : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)",
              "scf.for %child = %zero to %two step %one { }");
    }
    if (variant == 19) {
      std::string padding;
      for (unsigned i = 0; i < 300; ++i) {
        padding += "%padding" + std::to_string(i) + " = arith.constant 0 : index\n";
      }
      replace("%advance = arith.addi", padding + "%advance = arith.addi");
    }
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module) && succeeded(verify(*module)), "parse scalar dependency slice variant")) {
      return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("slice");
    scf::ForOp loop;
    AllocTileOp bank;
    function.walk<WalkOrder::PreOrder>([&](scf::ForOp op) {
      if (!loop) { loop = op; }
    });
    loop.walk([&](AllocTileOp op) {
      if (!bank) { bank = op; }
    });
    if (!check(bool(bank), "dependency fixture lost its bank allocation")) {
      return false;
    }
    auto mapping = SyncSlotMapping::derive(loop, bank.getAddr());
    const unsigned period = (variant == 7 || variant >= 18) ? 17 : (variant == 14 ? 6 : 2);
    if (!check(variant == 8 ? !mapping : mapping && mapping->period == period,
               "selector slice lost a proved relation or admitted unknown evolution, variant " +
                   std::to_string(variant))) {
      return false;
    }
    if (mapping) {
      for (unsigned visit = 0; visit < 40; ++visit) {
        const auto address = SyncSlotMapping::evaluate(bank.getAddr(), mapping->values[visit % period]);
        const auto slot = variant == 14 ? visit % 2 + 2 * (visit % 3) : visit % period;
        if (!check(address == std::optional<uint64_t>(256 + (variant == 9 ? 64 : 128) * slot),
                   "dependency slice disagrees with original scalar semantics")) {
          return false;
        }
      }
    }
    const auto original = text(function);
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::analyzeHandoffSync(function, imported)) && text(function) == original,
               "dependency-sliced native import changed original IR")) {
      return false;
    }
    if (variant >= 15) {
      oahs::NativeAnalysis selected;
      if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, selected)),
                 "enclosing semantic import failed")) {
        return false;
      }
      const bool bankOccurrence = llvm::any_of(selected.program.observed->observations,
          [](const auto& observation) {
            return llvm::any_of(observation.atoms, [](const auto& atom) {
              return atom.kind == oahs::ObservationAtom::LoopResidue;
            });
          });
      if (!check(bankOccurrence == (variant != 19),
                 "enclosing occurrence materialization confused physical facts with protocol or budget")) {
        return false;
      }
      if (variant == 19 && !check(
              llvm::any_of(selected.observationNotes, [](const auto& note) {
                return note.find("bank control: occurrence materialization budget") != std::string::npos;
              }) && llvm::any_of(selected.program.physicalUses, [](const auto& use) {
                return use.period == 17;
              }), "budget refusal erased the independent physical relation")) {
        return false;
      }
    }
    if (variant != 8 && variant != 9) {
      for (unsigned slot = 0; slot < period; ++slot) {
        if (!check(llvm::any_of(imported.program.cells, [&](const auto &cell) {
              return !cell.unknownRange && cell.ranges ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{256 + 128 * slot, 128}};
            }), "native import discarded independent finite bank facts")) {
          return false;
        }
      }
    }
    if (variant == 13) {
      for (const auto &observation : imported.program.observed->observations) {
        if (!check(llvm::none_of(observation.atoms, [](const auto &atom) {
              return atom.kind == oahs::ObservationAtom::LoopResidue;
            }), "non-unit enclosing visits used unnormalized IV guards")) {
          return false;
        }
      }
    }
    if (variant == 12) {
      std::set<unsigned> periods;
      for (const auto& relation : imported.program.physicalUses) {
        periods.insert(relation.period);
      }
      if (!check(periods == std::set<unsigned>{2, 3},
                 "independent occurrence relations were erased or multiplied")) {
        return false;
      }
      for (unsigned slot = 0; slot < 3; ++slot) {
        if (!check(llvm::any_of(imported.program.cells, [&](const auto &cell) {
              return !cell.unknownRange && cell.ranges ==
                  std::vector<std::pair<uint64_t, uint64_t>>{{2048 + 128 * slot, 128}};
            }), "independent period-three relation lost its physical facts")) {
          return false;
        }
      }
    }
    if (variant == 9) {
      bool shared = false;
      for (unsigned cell = 0; cell < imported.program.cells.size(); ++cell) {
        if (imported.program.cells[cell].ranges != std::vector<std::pair<uint64_t, uint64_t>>{{320, 64}}) {
          continue;
        }
        unsigned uses = 0;
        for (const auto &op : imported.program.operations) {
          uses += llvm::any_of(op.accesses, [&](auto access) { return access.cell == cell; });
        }
        shared |= uses >= 2;
      }
      if (!check(shared, "overlapping banks lost their shared physical obligation")) {
        return false;
      }
    }
    // Large relations are physical facts even when binding requires fallback.
    oahs::SelectedPlan plan;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &plan)),
               "dependency-sliced constructor or independent reconstruction failed, variant " +
                   std::to_string(variant) + " reason=" + plan.reason)) {
      return false;
    }
    if (variant <= 6 || variant == 10 || variant == 11) {
      if (!check(plan.work.boundaryAnalysisSites != 0 && plan.work.physicalUseQuerySites != 0 &&
                     !plan.declinedObservation && !plan.declinedFirstUse && !plan.declinedRecurring,
                 "equivalent native representation bypassed shared boundary construction, variant " +
                     std::to_string(variant))) {
        return false;
      }
      for (unsigned slot = 0; slot < 2; ++slot) {
        for (auto sourcePipe : {oahs::Pipe::MTE2, oahs::Pipe::V}) {
          const auto observer = sourcePipe == oahs::Pipe::MTE2 ? oahs::Pipe::V : oahs::Pipe::MTE2;
          const bool retained = llvm::any_of(plan.channels, [&](const auto& channel) {
            return channel.source == sourcePipe && channel.observer == observer &&
                llvm::any_of(channel.cells, [&](unsigned cell) {
                  return imported.program.cells[cell].ranges ==
                      std::vector<std::pair<uint64_t, uint64_t>>{{256 + 128 * slot, 128}};
                });
          });
          if (!check(retained, "equivalent native representation lost a bank readiness/release frontier")) {
            return false;
          }
        }
      }
    }
    const bool unrefinedRetry = bool(plan.declinedObservation) && !plan.declinedFirstUse;
    if (variant == 7 && !check(unrefinedRetry,
                             "unrealizable optional observations were not declined atomically")) {
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
  auto mapping = SyncSlotMapping::derive(loop, bank.getAddr(), 8);
  if (!check(mapping && mapping->period == 3, "derive stride-two modulo-three orbit")) return false;
  uint64_t slot = 2;
  for (unsigned i = 0; i < 30; ++i) {
    slot = (slot + 2) % 3;
    auto address = SyncSlotMapping::evaluate(bank.getAddr(), mapping->values[i % 3]);
    if (!check(address && *address == 256 + 128 * slot, "slot address differs from original recurrence")) return false;
  }
  if (!check(!SyncSlotMapping::derive(loop, bank.getAddr(), 2),
             "finite vocabulary cannot silently truncate the orbit")) return false;
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
  if (!check(plan.work.recurringLocalPackets != 0 && plan.work.recurringAnalysisSites == 0 &&
             plan.work.recurringReplaySites == 0,
             "native recurrence lost immutable interface qualification")) return false;

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
    AllocTileOp changedBank;
    changed.walk([&](AllocTileOp op) { changedBank = op; });
    const auto relation = SyncSlotMapping::derive(changed, changedBank.getAddr(), 8);
    if (!check(bool(relation) == (mutation == 1),
               "dependency recurrence proof or independent induction step changed")) {
      return false;
    }
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
const char *pipeName(oahs::Pipe pipe);
bool runFile(MLIRContext &context, const char *path) {
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) { return false; }
  bool accepted = true;
  module->walk([&](func::FuncOp function) {
    if (function.isDeclaration()) { return; }
    oahs::SelectedPlan report;
    const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report);
    if (report.declinedRecurring) {
      llvm::errs() << "declined recurring: " << report.declinedRecurring->reason
                   << " at " << report.declinedRecurring->cut << "\n";
    }
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
                 << " occurrence_sites=" << work.occurrenceAnalysisSites
                 << " boundary_sites=" << work.boundaryAnalysisSites
                 << " physical_use_sites=" << work.physicalUseQuerySites
                 << " requirement_classifications=" << work.requirementClassifications
                 << " classification_sites=" << work.classificationSites
                 << " classification_origins=" << work.classificationOrigins
                 << " witness_queries=" << work.witnessQueries << " witness_sites=" << work.witnessSites
                 << " producer_support_work=" << work.producerSupportWork
                 << " unsummarized_backedges=" << work.unsummarizedBackedges
                 << " finite_occurrence_transitions=" << work.finiteOccurrenceTransitions
                 << " transition_classification_work=" << work.transitionClassificationWork
                 << " native_endpoint_discovery_work=" << work.nativeEndpointDiscoveryWork
                 << " ownership_queries=" << work.ownershipQueries
                 << " restoration_deadlines=" << work.restorationDeadlineQueries
                 << " restoration_uses=" << work.restorationUseChecks
                 << " restoration_positions=" << work.restorationPositionEntries
                 << " restoration_fallbacks=" << work.restorationDeadlineFallbacks
                 << " deadline_restorations=" << work.deadlineRestorations
                 << " publication_support_sites=" << work.publicationSupportSites
                 << " publication_support_commands=" << work.publicationSupportCommands
                 << " publication_support_checks=" << work.publicationSupportChecks
                 << " publication_support_nodes=" << work.publicationSupportNodes
                 << " ownership_checks=" << work.ownershipChecks
                 << " ownership_sites=" << work.ownershipCheckSites
                 << " ownership_bindings=" << work.ownershipBindings
                 << " acknowledgment_checks=" << work.acknowledgmentChecks
                 << " acknowledgment_check_sites=" << work.acknowledgmentCheckSites
                 << " joined_acknowledgments=" << work.joinedAcknowledgments
                 << " source_gap_queries=" << work.sourceGapQueries
                 << " acknowledgment_prefix_replays=" << work.acknowledgmentPrefixReplays
                 << " acknowledgment_prefix_replay_sites=" << work.acknowledgmentPrefixReplaySites
                 << " source_gap_commands=" << work.sourceGapCommands
                 << " early_publications=" << work.earlyPublications
                 << " key_queries=" << work.keyQueries << " invariant=" << work.invariantSiteEvaluations
                 << " prepare_microseconds=" << work.preparationMicroseconds
                 << " sites=" << work.constructedSites << " words=" << work.commandWords
                 << " cells=" << work.cells << " eligible_keys=" << work.eligibleKeys
                 << " components=" << work.components << " cyclic=" << work.cyclicComponents
                 << " requirement_frontiers=" << work.requirementFrontiers
                 << " source_frontiers=" << work.qualifiedSourceFrontiers
                 << " unqualified_frontiers=" << work.unqualifiedSourceFrontiers
                 << " frontier_classes=" << work.frontierAcyclic << "," << work.frontierSameVisit << ","
                 << work.frontierPreviousUse << "," << work.frontierRegionEntry << ","
                 << work.frontierRegionContinuation << "," << work.frontierGuarded << ","
                 << work.frontierUnknown
                 << " recurring=" << work.recurringChannels
                 << " recurring_proposals=" << work.recurringProposals
                 << " first_use_prefixes=" << report.nativeFirstUsePrefixes
                 << " first_use_declined=" << bool(report.declinedFirstUse)
                 << " first_use_discarded_replay_sites="
                 << (report.declinedFirstUse ? report.declinedFirstUse->work.replaySiteEvaluations : 0)
                 << " first_use_discarded_occurrence_sites="
                 << (report.declinedFirstUse ? report.declinedFirstUse->work.occurrenceAnalysisSites : 0)
                 << " first_use_discarded_native_endpoint_work="
                 << (report.declinedFirstUse ? report.declinedFirstUse->work.nativeEndpointDiscoveryWork : 0)
                 << " first_use_discarded_elapsed_us="
                 << (report.declinedFirstUse ? report.declinedFirstUse->work.elapsedMicroseconds : 0)
                 << " observation_declined=" << bool(report.declinedObservation)
                 << " observation_discarded_replay_sites="
                 << (report.declinedObservation ? report.declinedObservation->work.replaySiteEvaluations : 0)
                 << " observation_discarded_occurrence_sites="
                 << (report.declinedObservation ? report.declinedObservation->work.occurrenceAnalysisSites : 0)
                 << " observation_discarded_native_endpoint_work="
                 << (report.declinedObservation ? report.declinedObservation->work.nativeEndpointDiscoveryWork : 0)
                 << " observation_discarded_producer_support_work="
                 << (report.declinedObservation ? report.declinedObservation->work.producerSupportWork : 0)
                 << " observation_discarded_physical_use_sites="
                 << (report.declinedObservation ? report.declinedObservation->work.physicalUseQuerySites : 0)
                 << " observation_discarded_boundary_sites="
                 << (report.declinedObservation ? report.declinedObservation->work.boundaryAnalysisSites : 0)
                 << " observation_discarded_elapsed_us="
                 << (report.declinedObservation ? report.declinedObservation->work.elapsedMicroseconds : 0)
                 << " recurring_declined=" << bool(report.declinedRecurring)
                 << " discarded_replay_sites="
                 << (report.declinedRecurring ? report.declinedRecurring->work.replaySiteEvaluations : 0)
                 << " discarded_occurrence_sites="
                 << (report.declinedRecurring ? report.declinedRecurring->work.occurrenceAnalysisSites : 0)
                 << " discarded_producer_support_work="
                 << (report.declinedRecurring ? report.declinedRecurring->work.producerSupportWork : 0)
                 << " discarded_physical_use_sites="
                 << (report.declinedRecurring ? report.declinedRecurring->work.physicalUseQuerySites : 0)
                 << " discarded_boundary_sites="
                 << (report.declinedRecurring ? report.declinedRecurring->work.boundaryAnalysisSites : 0)
                 << " discarded_elapsed_us="
                 << (report.declinedRecurring ? report.declinedRecurring->work.elapsedMicroseconds : 0)
                 << " recurring_trials=" << work.recurringTrials
                 << " recurring_removed=" << work.redundantRecurringChannels
                 << " repair_candidates=" << work.repairCandidates
                 << " repair_selected=" << work.repairSelected
                 << " repair_source_commands=" << work.repairSourceCommands
                 << " repair_neighbor_uses=" << work.repairNeighborUses
                 << " corridor_receipt_scans=" << work.corridorReceiptScans
                 << " corridor_word_endpoints=" << work.corridorWordEndpoints
                 << " common_cut_continuation_queries=" << work.commonCutContinuationQueries
                 << " common_cut_continuation_sites=" << work.commonCutContinuationSites
                 << " common_cut_continuation_words=" << work.commonCutContinuationWords
                 << " normal_candidates=" << work.normalCandidates
                 << " normal_selected=" << work.normalSelected
                 << " normal_recurring_selected=" << work.normalRecurringSelected
                 << " normal_publication_sites=" << work.normalPublicationSites
                 << " normal_key_sites=" << work.normalKeySites
                 << " normalized_due=" << work.normalizedDue
                 << " normalization_incidences=" << work.normalizationIncidences
                 << " recurring_interface_queries=" << work.recurringInterfaceQueries
                 << " recurring_interface_sites=" << work.recurringInterfaceSites
                 << " recurring_interface_accesses=" << work.recurringInterfaceAccesses
                 << " recurring_local_packets=" << work.recurringLocalPackets
                 << " recurring_interface_embedding=" << work.recurringInterfaceEmbedding
                 << " recurring_local_declines=" << work.recurringLocalDeclines
                 << " recurring_analysis_sites=" << work.recurringAnalysisSites
                 << " recurring_replay_sites=" << work.recurringReplaySites
                 << " recurring_support_queries=" << work.recurringSupportQueries
                 << " recurring_families=" << work.recurringFamilies
                 << " recurring_index_entries=" << work.recurringIndexEntries
                 << " recurring_candidates=" << work.recurringCandidates
                 << " recurring_attempts=" << work.recurringAttempts
                 << " recurring_activations=" << work.recurringActivations
                 << " recurring_declines=" << work.recurringDeclines
                 << " loop_entry_transfers=" << work.loopEntryTransfers
                 << " loop_entry_analysis_sites=" << work.loopEntryAnalysisSites
                 << " loop_entry_preparation_sites=" << work.loopEntryPreparationSites
                 << " contextual=" << work.contextualReplays
                 << " unreused_updates=" << work.unreusedUpdates
                 << " sources=" << work.sourceHandles << " rearming_discharged=" << work.rearmingDischarged
                 << " rearming_restored=" << work.rearmingRestored
                 << " rearming_deferred=" << work.rearmingDeferred
                 << " deferred_materialized=" << work.deferredMaterialized
                 << " latent_support_checks=" << work.latentSupportChecks
                 << " latent_support_retained=" << work.latentSupportRetained
                 << " rearming_pairs=" << work.rearmingPairVisits
                 << " rearming_query_sites=" << work.rearmingQuerySites
                 << " acknowledgments=" << work.acknowledgments
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
    if (report.declinedObservation) {
      llvm::errs() << "observation-decline cut=" << report.declinedObservation->cut
                   << " reason=" << report.declinedObservation->reason << "\n";
    }
    if (report.declinedFirstUse) {
      llvm::errs() << "first-use-decline cut=" << report.declinedFirstUse->cut
                   << " reason=" << report.declinedFirstUse->reason << "\n";
    }
    for (const auto &fence : report.fences) {
      llvm::errs() << "fence cut=" << fence.cut << " observer=" << pipeName(fence.observer)
                   << " version=" << fence.version << " residuals=";
      for (std::size_t i = 0; i < fence.residuals.size(); ++i) {
        const auto &residual = fence.residuals[i];
        llvm::errs() << (i ? "," : "") << "cell" << residual.cell << ":"
                     << pipeName(residual.source) << ":"
                     << (residual.sourceWrite ? "W" : "R") << "->"
                     << (residual.consumerRead ? "R" : "")
                     << (residual.consumerWrite ? "W" : "");
      }
      llvm::errs() << "\n";
    }
  });
  if (accepted) { module->print(llvm::outs()); }
  return accepted;
}

const char *pipeName(oahs::Pipe pipe) {
  static constexpr const char *names[] = {"S", "V", "M", "MTE1", "MTE2", "MTE3", "FIX"};
  const auto value = unsigned(pipe);
  return value < oahs::PipeCount ? names[value] : "?";
}
const char *relationshipName(oahs::StorageRelationship::Kind kind) {
  switch (kind) {
  case oahs::StorageRelationship::RAW: return "RAW";
  case oahs::StorageRelationship::WAR: return "WAR";
  case oahs::StorageRelationship::WAW: return "WAW";
  }
  return "?";
}
const char *occurrenceName(oahs::selected::RequirementOccurrence occurrence) {
  using O = oahs::selected::RequirementOccurrence;
  switch (occurrence) {
  case O::Acyclic: return "acyclic";
  case O::SameVisit: return "same_visit";
  case O::PreviousUse: return "previous_use";
  case O::RegionEntry: return "region_entry";
  case O::RegionContinuation: return "region_continuation";
  case O::Guarded: return "guarded";
  case O::Unknown: return "unknown";
  case O::Count: break;
  }
  return "?";
}
bool frontierFile(MLIRContext &context, const char *path) {
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) return false;
  bool accepted = true;
  module->walk([&](func::FuncOp function) {
    if (function.isDeclaration()) return;
    oahs::NativeAnalysis imported;
    if (failed(oahs::testing::analyzeSelectedHandoffSync(function, imported))) {
      accepted = false;
      return;
    }
    oahs::selected::Control control(imported.program);
    oahs::StorageFrontierAnalysis storage(imported.program);
    oahs::selected::RequirementFrontiers frontiers(imported.program, control, storage);
    if (!control.complete || !storage.complete() || !frontiers.complete()) {
      llvm::errs() << "frontier analysis failed for " << function.getSymName() << ": "
                   << (!control.complete ? control.reason :
                       !storage.complete() ? storage.reason() : frontiers.reason()) << "\n";
      accepted = false;
      return;
    }
    struct Aggregate {
      std::size_t count = 0;
      oahs::Cut source = oahs::NoAnalysisId, deadline = oahs::NoAnalysisId;
      oahs::Cut publication = oahs::NoAnalysisId;
    };
    auto modeName = [](const oahs::selected::OccurrenceMode &mode) {
      if (!mode.valid) return std::string("-");
      return std::to_string(mode.owner) + ":" + std::to_string(mode.period) + ":" +
             std::to_string(mode.residue) + ":" + (mode.previous ? "P" : "-") +
             (mode.next ? "N" : "-");
    };
    using Key = std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                           std::string, std::string>;
    std::map<Key, Aggregate> groups;
    for (oahs::Cut deadline = 0; deadline < oahs::commandCutCount(imported.program); ++deadline) {
      for (const auto &frontier : frontiers.at(deadline)) {
        const Key key{unsigned(frontier.occurrence), unsigned(frontier.relationship.kind),
                      unsigned(frontier.source), unsigned(frontier.observer),
                      frontier.relationship.cell,
                      modeName(oahs::selected::occurrenceMode(
                          imported.program, frontier.relationship.source.site)),
                      modeName(oahs::selected::occurrenceMode(imported.program, deadline))};
        auto &group = groups[key];
        if (group.count++ == 0) {
          group.source = frontier.relationship.source.site;
          group.deadline = frontier.deadline;
          group.publication = frontier.publication;
        }
      }
    }
    llvm::outs() << "function\t" << function.getSymName() << "\trequirements\t"
                 << frontiers.size() << "\tqualified_sources\t" << frontiers.sourceBoundaries() << "\n";
    for (unsigned cell = 0; cell < imported.program.cells.size(); ++cell) {
      const auto &physical = imported.program.cells[cell];
      llvm::outs() << "cell\t" << cell << "\tspace\t" << physical.addressSpace
                   << "\tstorage\t" << unsigned(physical.storage)
                   << "\tunknown\t" << physical.unknownRange << "\tranges\t";
      for (std::size_t i = 0; i < physical.ranges.size(); ++i)
        llvm::outs() << (i ? "," : "") << physical.ranges[i].first << ":" << physical.ranges[i].second;
      llvm::outs() << "\n";
    }
    llvm::outs() << "occurrence\tkind\tsource\tobserver\tcell\tsource_mode\ttarget_mode\tcount"
                    "\tsample_source_site\tsample_publication\tsample_deadline\n";
    for (const auto &[key, group] : groups) {
      const auto &[occurrence, kind, source, observer, cell, sourceMode, targetMode] = key;
      auto cut = [](oahs::Cut value) -> uint64_t {
        return value == oahs::NoAnalysisId ? std::numeric_limits<uint64_t>::max() : value;
      };
      llvm::outs() << occurrenceName(oahs::selected::RequirementOccurrence(occurrence)) << "\t"
                   << relationshipName(oahs::StorageRelationship::Kind(kind)) << "\t"
                   << pipeName(oahs::Pipe(source)) << "\t" << pipeName(oahs::Pipe(observer)) << "\t"
                   << cell << "\t" << sourceMode << "\t" << targetMode << "\t"
                   << group.count << "\t" << cut(group.source) << "\t"
                   << cut(group.publication) << "\t" << cut(group.deadline) << "\n";
    }
  });
  return accepted;
}
} // namespace
int main(int argc, char **argv) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  context.loadDialect<PTODialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  if (argc == 3 && StringRef(argv[1]) == "--construct") {
    return runFile(context, argv[2]) ? 0 : 1;
  }
  if (argc == 3 && StringRef(argv[1]) == "--frontiers") {
    return frontierFile(context, argv[2]) ? 0 : 1;
  }
  if (argc != 1) {
    llvm::errs() << "usage: pto-oahs-selected-test [--construct INPUT | --frontiers INPUT]\n";
    return 2;
  }
  const bool passed = positive(context, ordinary, "ordinary") && positive(context, loop, "loop") &&
                      positive(context, recurrence, "recurrence") &&
                      positive(context, collective, "collective") &&
                      positive(context, queue, "queue") && mutations(context) && constantAddresses(context) &&
                      slotMappings(context) && slotDependencySlices(context) && originalLoopDomains(context) &&
                      normalizedGuardReadback(context) && exactCommandEmission(context) &&
                      originalResidueDecisions(context) && readerGenerations(context) &&
                      optionalReaderParticipation(context) &&
                      uniformEndpointRoles(context) && sourceGapPlacement(context) && milestoneKeyBinding(context) &&
                      atomicAcknowledgment(context) && restorationDeadlines(context) &&
                      requiredReturnCoverage(context) &&
                      accumulatorOrdering(context) && accumulatorEpisodes(context) && firstUseOrdering(context);
  return passed ? 0 : 1;
}
