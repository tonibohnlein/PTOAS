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
bool exactCommandEmission(MLIRContext &context) {
  auto module = parseSourceString<ModuleOp>(
      "module attributes {pto.target_arch = \"a3\"} {func.func @words() {return}}", &context);
  if (!check(bool(module), "parse exact command-word test")) return false;
  auto function = module->lookupSymbol<func::FuncOp>("words");
  auto anchor = std::make_unique<PlaceHolderInstanceElement>(0, 0);
  anchor->elementOp = function.getBody().front().getTerminator();
  SmallVector<std::unique_ptr<SyncOperation>> storage;
  for (unsigned i=0;i<6;++i) {
    const auto type = i%2 ? SyncOperation::TYPE::WAIT_EVENT : SyncOperation::TYPE::SET_EVENT;
    auto sync = std::make_unique<SyncOperation>(type, PipelineType::PIPE_MTE2,
        PipelineType::PIPE_MTE1, i, 0, std::nullopt);
    sync->eventIds.push_back(0);
    anchor->pipeBefore.push_back(sync.get());storage.push_back(std::move(sync));
  }
  SyncIRs emission;emission.push_back(std::move(anchor));
  SyncCodegen codegen(emission,function,SyncAnalysisMode::NORMALSYNC,true);codegen.Run();
  unsigned count=0;
  for(auto& op:function.getBody().front()) {
    if(isa<func::ReturnOp>(op))continue;
    if(!check(count%2 ? isa<WaitFlagOp>(op) : isa<SetFlagOp>(op),"exact event order changed"))return false;
    ++count;
  }
  return check(count==6,"distinct event generations deduplicated");
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
const char *alternatingQueue = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @alternating_queue(%gm: !pto.ptr<f32, gm>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %base = arith.constant 0 : i32
    %outaddr = arith.constant 1024 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %pipe = pto.initialize_l2g2l_pipe{dir_mask = 3, slot_size = 128,
      slot_num = 2, local_slot_num = 1, flag_base = 0, nosplit = true}
      (%gm : !pto.ptr<f32, gm>, %base : i32, %base : i32) -> !pto.pipe
    scf.for %i = %zero to %n step %one {
      scf.if %active {
        %a = pto.declare_tile -> !pto.tile_buf<vec, 1x32xf32>
        %out = pto.alloc_tile addr = %outaddr : !pto.tile_buf<vec, 1x32xf32>
        pto.tpop(%a, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
        pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
        pto.tpush(%out, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
        pto.tfree(%pipe : !pto.pipe) {split = 0}
      }
    }
    return
  }
})mlir";
bool fifoSlotQualification(MLIRContext &context) {
  for (unsigned variant = 0; variant < 4; ++variant) {
    std::string source = alternatingQueue;
    auto replace = [&](StringRef before, StringRef after) {
      const auto at = source.find(before.str());
      if (at != std::string::npos) source.replace(at, before.size(), after.str());
    };
    if (variant == 1) replace("slot_num = 2", "slot_num = 1");
    if (variant == 2) replace("slot_size = 128", "slot_size = 64");
    if (variant == 3) replace("pto.tpush(%out, %pipe : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}", "");
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse alternating FIFO")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("alternating_queue");
    oahs::NativeAnalysis input;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, input)), "import alternating FIFO")) return false;
    if (!check(bool(input.program.alternatingSlots) == (variant == 0), "FIFO slot qualification boundary")) return false;
    if (variant) continue;
    oahs::SelectedPlan report;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report)), "construct/reconstruct alternating FIFO")) return false;
    if (!check(report.channels.size() == 2, "FIFO uses two shared logical directions")) return false;
    for (auto read : input.program.alternatingSlots->reads)
      for (auto c : report.commands[read])
        if (!check(!(c.kind == oahs::Command::Acquire && c.source == oahs::Pipe::MTE3 &&
                     c.observer == oahs::Pipe::MTE2), "FIFO output receipt must follow ingress")) return false;
  }
  return true;
}
bool staticFifoSlotQualification(MLIRContext& context)
{
    const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @static_fifo(%gm: !pto.ptr<f32, gm>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %base = arith.constant 0 : i32
    %addr = arith.constant 0 : i64
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %pipe = pto.initialize_l2g2l_pipe{dir_mask = 3, slot_size = 1024,
      slot_num = 2, local_slot_num = 1, flag_base = 0, nosplit = true}
      (%gm : !pto.ptr<f32, gm>, %base : i32, %base : i32) -> !pto.pipe
    %out = pto.alloc_tile addr = %addr : !pto.tile_buf<acc, 16x16xf32>
    scf.for %i = %zero to %n step %one {
      scf.if %active {
        pto.tpush(%out, %pipe : !pto.tile_buf<acc, 16x16xf32>, !pto.pipe) {split = 0}
        pto.tpush(%out, %pipe : !pto.tile_buf<acc, 16x16xf32>, !pto.pipe) {split = 0}
        %a = pto.declare_tile -> !pto.tile_buf<mat, 16x16xf32>
        %b = pto.declare_tile -> !pto.tile_buf<mat, 16x16xf32>
        pto.tpop(%a, %pipe : !pto.tile_buf<mat, 16x16xf32>, !pto.pipe) {split = 0}
        pto.tpop(%b, %pipe : !pto.tile_buf<mat, 16x16xf32>, !pto.pipe) {split = 0}
        pto.tfree(%pipe : !pto.pipe) {split = 0}
      }
    }
    return
  }
})mlir";
    for (unsigned variant = 0; variant < 5; ++variant) {
        auto source = fixture;
        auto replace = [&](const std::string& from, const std::string& to) {
            source.replace(source.find(from), from.size(), to);
        };
        if (variant == 1)
            replace("pto.tpush(%out, %pipe : !pto.tile_buf<acc, 16x16xf32>, !pto.pipe) {split = 0}", "");
        if (variant == 2)
            replace("slot_size = 1024", "slot_size = 512");
        if (variant == 3)
            replace("nosplit = true", "nosplit = false");
        if (variant == 4)
            replace("{split = 0}", "{split = 1}");
        auto module = parseSourceString<ModuleOp>(source, &context);
        if (!check(bool(module), "parse static FIFO fixture"))
            return false;
        auto function = module->lookupSymbol<func::FuncOp>("static_fifo");
        oahs::NativeAnalysis input;
        if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, input)), "import static FIFO fixture"))
            return false;
        if (!check(bool(input.program.staticFifoSlots) == (variant == 0), "static FIFO qualification boundary"))
            return false;
        if (variant)
            continue;
        const auto& slots = *input.program.staticFifoSlots;
        if (!check(
                slots.cells.size() == 2 && slots.reads.size() == 2 && slots.writes.size() == 2,
                "shared static slot view"))
            return false;
        for (unsigned slot = 0; slot < 2; ++slot)
            for (auto operation : {slots.reads[slot], slots.writes[slot]}) {
                if (!check(
                        llvm::any_of(
                            input.program.operations[operation].accesses,
                            [&](auto access) { return access.cell == slots.cells[slot] && !access.definiteWrite; }),
                        "send and receive must share the same physical slot"))
                    return false;
            }
    }
    return true;
}
bool firstConsumerPlacement(MLIRContext &context) {
  const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @first_consumer(%src: !pto.partition_tensor_view<1x32xf32>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %lb = arith.constant 3 : index
    %ub = arith.constant 9 : index
    %step = arith.constant 2 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %z = arith.constant 256 : i64
    %w = arith.constant 384 : i64
    %u = arith.constant 512 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %z : !pto.tile_buf<vec, 1x32xf32>
    %other = pto.alloc_tile addr = %w : !pto.tile_buf<vec, 1x32xf32>
    %unused = pto.alloc_tile addr = %u : !pto.tile_buf<vec, 1x32xf32>
    scf.for %entry = %zero to %n step %one {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%unused : !pto.tile_buf<vec, 1x32xf32>)
      scf.for %i = %lb to %ub step %step {
        pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
        // INSERT
        pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    return
  }
})mlir";
  for (unsigned variant = 0; variant != 12; ++variant) {
    const bool classInvariant = variant >= 6;
    const unsigned mutation = variant % 6;
    auto source = fixture;
    auto replace = [&](const std::string &a, const std::string &b) {
      source.replace(source.find(a), a.size(), b);
    };
    if (mutation == 1) replace("%lb to %ub", "%lb to %lb");
    if (mutation == 2) replace("%lb to %ub", "%lb to %n");
    if (mutation == 3) replace("// INSERT", "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)");
    if (mutation == 4) replace("pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>)",
      "scf.if %active { pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>) }");
    if (mutation == 5) replace("// INSERT", "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%unused : !pto.tile_buf<vec, 1x32xf32>)");
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse first-consumer fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("first_consumer");
    oahs::NativeAnalysis input;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, input, classInvariant)), "import first consumer")) return false;
    const bool qualified = input.program.observed->qualification.find("first-consumer-prefix-v1") != std::string::npos;
    if (!check(qualified == (mutation == 0 || (classInvariant && mutation == 5)), "first consumer must be invariant, unconditional and nonempty")) return false;
    if (!qualified) continue; // Unsupported descriptors must retain ordinary control.
    oahs::SelectedPlan report;
    oahs::SelectedOptions options; options.classInvariantInputs = classInvariant;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report, &options)), "first-consumer construction/reconstruction")) {
      llvm::errs() << "variant=" << variant << " cut=" << report.cut << " reason=" << report.reason << "\n";
      return false;
    }
    if (mutation == 0) {
      bool correctThreshold = false;
      function.walk([&](arith::CmpIOp cmp) {
        auto constant = cmp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
        correctThreshold |= constant && constant.value() == 5 &&
            cmp.getPredicate() == arith::CmpIPredicate::slt;
      });
      if (!check(correctThreshold, "nonunit first-visit guard uses original lower bound plus step")) return false;
    }
  }
  return true;
}
bool firstWritePlacement(MLIRContext &context) {
  const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @first_write(%dst: !pto.partition_tensor_view<1x32xf32>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %lo = arith.constant 3 : index
    %hi = arith.constant 9 : index
    %step = arith.constant 2 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %z = arith.constant 256 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    %c = pto.alloc_tile addr = %z : !pto.tile_buf<vec, 1x32xf32>
    scf.for %entry = %zero to %n step %one {
      pto.tstore ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)
      scf.for %i = %lo to %hi step %step {
        pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
        // INSERT
        pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    return
  }
})mlir";
  oahs::SelectedOptions options;
  options.firstWriteConsumers = true;
  for (unsigned variant = 0; variant < 8; ++variant) {
    auto source = fixture;
    auto replace = [&](const std::string &a, const std::string &b) {
      source.replace(source.find(a), a.size(), b);
    };
    const std::string write = "pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)";
    if (variant == 1) replace("%lo to %hi", "%lo to %lo");
    if (variant == 2) replace("%lo to %hi", "%lo to %n");
    if (variant == 3) replace("// INSERT", "pto.tstore ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)");
    if (variant == 4) replace(write, "scf.if %active { " + write + " }");
    if (variant == 5) replace("%hi = arith.constant 9", "%hi = arith.constant 4");
    if (variant == 6) {
      const std::string read = "pto.tstore ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%dst : !pto.partition_tensor_view<1x32xf32>)";
      replace(read, "");
      replace("    return", "    " + read + "\n    return");
    } // A future-only reader is not a first-write reuse opportunity.
    if (variant == 7) replace("// INSERT", "pto.tload ins(%dst : !pto.partition_tensor_view<1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)");
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse first-write fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("first_write");
    oahs::NativeAnalysis input;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, input, false, true)), "import first write")) return false;
    oahs::NativeAnalysis finalOnly;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, finalOnly, false, false, true)),
               "import final-only policy with shared storage view")) return false;
    if (!check(std::all_of(finalOnly.program.observed->loops.begin(), finalOnly.program.observed->loops.end(),
                         [](const auto &loop) { return loop.firstWriteFrontiers.empty(); }),
               "sharing analysis must not enable first-write qualification")) return false;
    const bool qualified = input.program.observed->qualification.find("first-consumer-prefix-v1") != std::string::npos;
    if (!check(qualified == (variant == 0 || variant == 5 || variant == 7), "first write needs nonempty, unconditional, source-inactive participation")) return false;
    if (!qualified) continue;
    oahs::SelectedPlan report;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report, &options)), "first-write construction/reconstruction")) return false;
    bool firstReceipt = false, laterReceipt = false;
    for (const auto &endpoint : report.ledger) {
      if (endpoint.command.kind != oahs::Command::Acquire ||
          endpoint.command.source != oahs::Pipe::MTE3 || endpoint.command.observer != oahs::Pipe::V) continue;
      const auto observation = input.program.observed->sites[endpoint.cut].observation;
      if (observation == oahs::NoControlId) continue;
      for (const auto &atom : input.program.observed->observations[observation].atoms)
        if (atom.kind == oahs::ObservationAtom::LoopHasPrevious) {
          firstReceipt |= atom.value == 0;
          laterReceipt |= atom.value != 0;
        }
    }
    if (!check((firstReceipt || variant == 7) && !laterReceipt, "external-reader receipt belongs only at the first conflicting write, unless already covered")) return false;
    bool threshold = false;
    function.walk([&](arith::CmpIOp cmp) {
      auto constant = cmp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      threshold |= constant && constant.value() == 5 && cmp.getPredicate() == arith::CmpIPredicate::slt;
    });
    if (!check(!firstReceipt || threshold, "first-write guard must retain original nonunit step")) {
      llvm::errs() << "variant=" << variant << "\n" << text(function);
      return false;
    }
    if (variant == 7) {
      bool priorExport = false, hoisted = false;
      function.walk([&](mlir::Operation *op) {
        priorExport |= isa<TLoadOp>(op);
        if (auto wait = dyn_cast<WaitFlagOp>(op))
          if (wait.getSrcPipe().getPipe() == PIPE::PIPE_MTE3 &&
              wait.getDstPipe().getPipe() == PIPE::PIPE_V) {
            auto owner = wait->getParentOfType<scf::ForOp>();
            auto lower = owner ? owner.getLowerBound().getDefiningOp<arith::ConstantIndexOp>() : arith::ConstantIndexOp{};
            if (lower && lower.value() == 3) hoisted |= !priorExport;
          }
      });
      if (!check(priorExport && !hoisted, "native first-write receipt must stay after the earlier outward consumer")) return false;
    }
  }
  for (unsigned mutation = 0; mutation != 2; ++mutation) {
    auto module = parseSourceString<ModuleOp>(fixture, &context);
    auto function = module->lookupSymbol<func::FuncOp>("first_write");
    const auto before = text(function);
    bool changed = false;
    ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) { return success(); });
    const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, [&](func::FuncOp working) {
      if (mutation == 0) {
        WaitFlagOp victim;
        working.walk([&](WaitFlagOp wait) {
          if (!victim && wait.getSrcPipe().getPipe() == PIPE::PIPE_MTE3 &&
              wait.getDstPipe().getPipe() == PIPE::PIPE_V) victim = wait;
        });
        if (victim) { victim.erase(); changed = true; }
      } else {
        working.walk([&](arith::CmpIOp cmp) {
          if (!changed && cmp.getPredicate() == arith::CmpIPredicate::slt) {
            cmp.setPredicate(arith::CmpIPredicate::sle); changed = true;
          }
        });
      }
    }, nullptr, &options);
    if (!check(changed && failed(status) && text(function) == before,
               "missing first-write support or wrong occurrence guard must fail transactionally")) return false;
  }
  return true;
}

bool finalReadSourcePlacement(MLIRContext &context) {
  const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @final_read(%src: !pto.partition_tensor_view<1x32xf32>, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %upper = arith.constant 128 : index
    %step = arith.constant 64 : index
    %addr0 = arith.constant 0 : i64
    %addr1 = arith.constant 128 : i64
    %addr2 = arith.constant 256 : i64
    %a = pto.alloc_tile addr = %addr0 : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %addr1 : !pto.tile_buf<vec, 1x32xf32>
    %c = pto.alloc_tile addr = %addr2 : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    scf.for %i = %zero to %upper step %step {
      pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
      %later = arith.cmpi eq, %i, %zero : index
      scf.if %active {
        pto.tadd ins(%c, %c : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
}
)mlir";
  for (unsigned variant=0;variant<5;++variant) {
    auto source=fixture;
    if(variant==1) source.replace(source.find("%upper = arith.constant 128"),std::string("%upper = arith.constant 128").size(),"%upper = arith.constant 64");
    if(variant==2) source.replace(source.find("ins(%c, %c"),10,"ins(%a, %a");
    if(variant==3) {
      const std::string load="    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)";
      source.erase(source.rfind(load),load.size());
    }
    if(variant==4) source.replace(source.find("%upper = arith.constant 128"),
        std::string("%upper = arith.constant 128").size(),"%upper = arith.constant 0");
    auto module=parseSourceString<ModuleOp>(source,&context);
    if(!check(bool(module),"parse final-read source"))return false;
    auto function=module->lookupSymbol<func::FuncOp>("final_read");
    oahs::NativeAnalysis input;
    if(!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function,input,false,false,true)),
              "import final-read source"))return false;
    const bool qualified=input.program.observed->qualification.find("final-read-source-gaps-v1")!=std::string::npos;
    if(!check(qualified==(variant<2),"final sources require a real return deadline and nonempty read prefix"))return false;
    oahs::SelectedOptions options;options.finalReadSources=true;options.recurring=false;
    options.finalHelperTrials=false;options.recurringOmissionTrials=false;
    oahs::SelectedPlan report;
    if(!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function,{},&report,&options)),
              "final-read native construction/reconstruction"))return false;
    if(!check(report.work.finalReadPublications == (variant<2?1u:0u),
              "final-read source must preserve the conditional suffix and reject later reads"))return false;
  }
  return true;
}
bool lastReaderPlacement(MLIRContext &context) {
  const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @last_reader(%src: !pto.partition_tensor_view<1x32xf32>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %bound = arith.constant 256 : index
    %step = arith.constant 128 : index
    %lo = arith.constant 3 : index
    %hi = arith.constant 6 : index
    %one = arith.constant 1 : index
    %addr0 = arith.constant 0 : i64
    %addr1 = arith.constant 128 : i64
    %addr2 = arith.constant 256 : i64
    %a = pto.alloc_tile addr = %addr0 : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %addr1 : !pto.tile_buf<vec, 1x32xf32>
    %c = pto.alloc_tile addr = %addr2 : !pto.tile_buf<vec, 1x32xf32>
    scf.for %tile = %zero to %bound step %step {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      scf.for %i = %lo to %hi step %one {
        pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
        pto.tadd ins(%c, %c : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
      }
    }
    return
  }
}
)mlir";
  for (unsigned variant = 0; variant < 9; ++variant) {
    auto source = fixture;
    auto replace = [&](const std::string &a, const std::string &b) {
      source.replace(source.find(a), a.size(), b);
    };
    const std::string read = "pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)";
    if (variant == 1) replace("%lo to %hi", "%lo to %lo");
    if (variant == 2) replace("%lo to %hi", "%lo to %n");
    if (variant == 3) replace("%one = arith.constant 1", "%one = arith.constant 2");
    if (variant == 4) replace(read, "scf.if %active { " + read + " }");
    if (variant == 5) replace(read, "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)\n" + read);
    if (variant == 6) replace("ins(%c, %c", "ins(%a, %a");
    if (variant == 7) replace("%lo = arith.constant 3", "%lo = arith.constant -1");
    if (variant == 8) replace("%hi = arith.constant 6", "%hi = arith.constant 4");
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse last-reader fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("last_reader");
    oahs::NativeAnalysis input;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, input)), "import last reader")) return false;
    const bool qualified = input.program.observed->qualification.find("last-visit-words-v1") != std::string::npos;
    if (!check(qualified == (variant == 0 || variant == 8), "last reader must have qualified final participation and trailing work")) return false;
    if (!qualified) continue;
    oahs::SelectedPlan report;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report)), "last-reader construction/reconstruction")) return false;
    bool selected = false;
    for (const auto &channel : report.channels) if (channel.source == oahs::Pipe::V && channel.observer == oahs::Pipe::MTE2)
      for (auto cut : channel.publications) {
        const auto observation = input.program.observed->sites[cut].observation;
        for (const auto &atom : input.program.observed->observations[observation].atoms)
          selected |= atom.kind == oahs::ObservationAtom::LoopHasNext && atom.value == 0;
      }
    if (!check(selected, "last-reader release must use the final visit")) return false;
  }
  return true;
}
bool jointReaderPlacement(MLIRContext &context) {
  const std::string fixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @joint_reader(%src: !pto.partition_tensor_view<1x32xf32>, %n: index, %active: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %bound = arith.constant 256 : index
    %step = arith.constant 128 : index
    %lo = arith.constant 3 : index
    %hi = arith.constant 259 : index
    %addr0 = arith.constant 0 : i64
    %addr1 = arith.constant 128 : i64
    %addr2 = arith.constant 256 : i64
    %addr3 = arith.constant 384 : i64
    %a = pto.alloc_tile addr = %addr0 : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %addr1 : !pto.tile_buf<vec, 1x32xf32>
    %c = pto.alloc_tile addr = %addr2 : !pto.tile_buf<vec, 1x32xf32>
    %d = pto.alloc_tile addr = %addr3 : !pto.tile_buf<vec, 1x32xf32>
    scf.for %tile = %zero to %bound step %step {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
      scf.for %i = %lo to %hi step %step {
        pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
        pto.tadd ins(%b, %b : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%d : !pto.tile_buf<vec, 1x32xf32>)
        pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
        pto.tadd ins(%b, %b : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%d : !pto.tile_buf<vec, 1x32xf32>)
        scf.if %active {
          pto.tadd ins(%d, %d : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%c : !pto.tile_buf<vec, 1x32xf32>)
        }
      }
    }
    return
  }
}
)mlir";
  for (unsigned variant = 0; variant < 10; ++variant) {
    auto source = fixture;
    auto replace = [&](const std::string &a, const std::string &b) { source.replace(source.find(a), a.size(), b); };
    if (variant == 1) replace("%hi = arith.constant 259", "%hi = arith.constant 4"); // first AND final
    if (variant == 2) replace("%lo to %hi", "%lo to %lo");
    if (variant == 3) replace("%lo to %hi", "%lo to %n");
    if (variant == 4) replace("%lo = arith.constant 3", "%lo = arith.constant -1");
    if (variant == 5) replace("scf.if %active {", "scf.if %active { pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)");
    if (variant == 6) replace("ins(%d, %d", "ins(%a, %a"); // suffix path still reads A
    if (variant == 7) replace("%hi = arith.constant 259", "%hi = arith.constant 260"); // visits 3,131,259
    if (variant == 8) {
      replace("%lo = arith.constant 3", "%lo = arith.constant 9223372036854775600");
      replace("%hi = arith.constant 259", "%hi = arith.constant 9223372036854775807"); // final increment overflows
    }
    if (variant == 9) replace("%step = arith.constant 128", "%step = arith.constant 1");
    auto module = parseSourceString<ModuleOp>(source, &context);
    if (!check(bool(module), "parse joint reader fixture")) return false;
    auto function = module->lookupSymbol<func::FuncOp>("joint_reader");
    oahs::NativeAnalysis input;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function,input)), "import joint reader")) return false;
    const bool qualified = input.program.observed->qualification.find("joint-reader-prefix-v1") != std::string::npos;
    if (!check(qualified == (variant == 0 || variant == 1 || variant == 7 || variant == 9),
               "joint reader must preserve bounds, suffix accesses and generation")) {
      llvm::errs() << "joint variant=" << variant << " qualified=" << qualified << "\n";
      return false;
    }
    if (!qualified) continue;
    oahs::SelectedPlan report;
    if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function,{},&report)),
               "joint reader construction/reconstruction")) return false;
    bool first = false, final = false;
    function.walk([&](arith::CmpIOp cmp) {
      auto constant = cmp.getRhs().getDefiningOp<arith::ConstantIndexOp>();
      first |= constant && constant.value() == (variant == 9 ? 4 : 131) && cmp.getPredicate() == arith::CmpIPredicate::slt;
      auto distance = cmp.getLhs().getDefiningOp<arith::SubIOp>();
      final |= distance && constant && constant.value() == (variant == 9 ? 1 : 128) && cmp.getPredicate() == arith::CmpIPredicate::sle;
    });
    if (!check(first && final,"joint native guards use original lower/upper/step")) return false;
  }
  // Emission validation must reject a malformed final predicate, even when
  // selected endpoints themselves were valid and the edited IR still parses.
  auto module = parseSourceString<ModuleOp>(fixture, &context);
  auto function = module->lookupSymbol<func::FuncOp>("joint_reader");
  const auto before = text(function);
  bool changed = false;
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) { return success(); });
  const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function,[&](func::FuncOp working) {
    working.walk([&](arith::CmpIOp cmp) {
      if (!changed && cmp.getPredicate() == arith::CmpIPredicate::sle &&
          cmp.getLhs().getDefiningOp<arith::SubIOp>()) {
        cmp.setPredicate(arith::CmpIPredicate::slt); changed = true;
      }
    });
  });
  return check(changed && failed(status) && text(function) == before,
               "changed final predicate must fail transactionally");
}
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
bool firstUseOrdering(MLIRContext &context) {
  std::string original = matrixInput;
  const auto argument = original.find("%unknown: index");
  original.replace(argument, std::string("%unknown: index").size(),
                   "%unknown: index, %out: !pto.partition_tensor_view<128x256xf32>");
  const auto begin = original.find("    pto.tmatmul ins");
  const auto middle = original.find("    pto.tmatmul.acc ins");
  const auto end = original.find("    return");
  for (unsigned variant = 0; variant < 4; ++variant) {
    std::string source = original.substr(0, begin);
    source += R"mlir(
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %tile = %c0 to %c2 step %c1 {
    scf.for %outer = %c0 to %c2 step %c1 {
      scf.for %inner = %c0 to %c2 step %c1 {
        %firstOuter = arith.cmpi eq, %outer, %c0 : index
        %firstInner = arith.cmpi eq, %inner, %c0 : index
)mlir";
    if (variant == 0 || variant == 3) {
      source += variant == 0 ? "        %first = arith.andi %firstOuter, %firstInner : i1\n"
                             : "        %first = arith.ori %firstOuter, %firstInner : i1\n";
    }
    const char *condition = variant == 1 ? "%firstInner" : variant == 2 ? "%firstOuter" : "%first";
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
    oahs::NativeAnalysis imported;
    if (!check(succeeded(oahs::testing::analyzeSelectedHandoffSync(function, imported)),
               "import nested first-use fixture")) return false;
    SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
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
    if (!check(checked.accepted == (variant == 0),
               "first use or genuinely repeating initialization completion")) return false;
    if (variant != 0) {
      for (oahs::Cut cut = 0; cut < commands.size(); ++cut) {
        if (isa_and_nonnull<TMatmulOp>(imported.cuts[cut]))
          commands[cut].push_back({oahs::Command::Barrier, oahs::Pipe::M});
      }
      if (!check(oahs::checkCausalFrontier(imported.program, commands).accepted,
                 "repeated initialization must retain a real completion repair")) return false;
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
    Value unsupported = builder.create<arith::SubIOp>(function.getLoc(), one, zero);
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
const char *pipeName(oahs::Pipe pipe);
bool runFile(MLIRContext &context, const char *path, oahs::SelectedOptions options = {}) {
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) { return false; }
  bool accepted = true;
  module->walk([&](func::FuncOp function) {
    if (function.isDeclaration()) { return; }
    oahs::SelectedPlan report;
    const auto status = oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report, &options);
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
                 << " split_relays=" << work.splitRelays << " relay_trials=" << work.relayTrials
                 << " relay_trial_sites=" << work.relayTrialSites << " relay_preparation=" << work.relayPreparationSites
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
                 << " policy=" << options.recurringOmissionTrials << options.finalHelperTrials
                 << options.movingFrontiers << options.sourceGaps << options.deferredAcyclicAcknowledgments
                 << options.classInvariantInputs << options.equalCoverageBinding
                 << " final_read_sources=" << options.finalReadSources
                 << " final_read_publications=" << work.finalReadPublications
                 << " closed_reservation_borrows=" << work.closedReservationBorrows
                 << " closed_reservation_checks=" << work.closedReservationChecks
                 << " closed_reservation_check_sites=" << work.closedReservationCheckSites
                 << " final_read_query_sites=" << work.finalReadQuerySites
                 << " first_write_consumers=" << options.firstWriteConsumers
                 << " split_rearming_queries=" << report.work.splitRearmingQueries
                 << " split_rearming_sites=" << report.work.splitRearmingSites
                 << " proposal_sites=" << work.proposalCheckSites
                 << " proposal_microseconds=" << work.proposalCheckMicroseconds
                 << " rejected_protocol=" << work.rejectedProtocolProposals
                 << " rejected_resource=" << work.rejectedResourceProposals
                 << " rejected_support=" << work.rejectedSupportProposals
                 << " gap_publications=" << work.gapPublications
                 << " deferred_acks=" << work.deferredAcknowledgments
                 << " equal_coverage_pairs=" << work.equalCoveragePairs
                 << " binding_probes=" << work.bindingProbes
                 << " binding_choices=" << work.bindingChoices
                 << " recurring=" << work.recurringChannels
                 << " share_reader_returns=" << options.shareReaderReturns
                 << " shared_returns=" << work.sharedReaderReturns
                 << " return_sharing_queries=" << work.returnSharingQueries
                 << " return_sharing_sites=" << work.returnSharingSiteVisits
                 << " recurring_trials=" << work.recurringTrials
                 << " recurring_removed=" << work.redundantRecurringChannels
                 << " recurring_analysis_sites=" << work.recurringAnalysisSites
                 << " qualification_microseconds=" << work.recurringQualificationMicroseconds
                 << " helper_trials=" << work.helperCompositionTrials
                 << " helper_sites=" << work.helperCompositionSiteEvaluations
                 << " helper_microseconds=" << work.helperCompositionMicroseconds
                 << " final_sites=" << work.finalCertificateSiteEvaluations
                 << " final_microseconds=" << work.finalCertificateMicroseconds
                 << " choice_transfers=" << work.choiceTransfers
                 << " choice_trials=" << work.choiceTrials
                 << " choice_analysis_sites=" << work.choiceAnalysisSites
                 << " choice_preparation_sites=" << work.choicePreparationSites
                 << " loop_entry_transfers=" << work.loopEntryTransfers
                 << " loop_entry_analysis_sites=" << work.loopEntryAnalysisSites
                 << " loop_entry_preparation_sites=" << work.loopEntryPreparationSites
                 << " contextual=" << work.contextualReplays
                 << " sibling_reuse=" << options.siblingReplayReuse
                 << " prefix_queries=" << work.replayPrefixQueries
                 << " prefix_span_examinations=" << work.replayPrefixSpanExaminations
                 << " sibling_comparison=" << options.traceReplay
                 << " sibling_components=" << work.siblingReusedComponents
                 << " invalidation_sites=" << work.replayInvalidationSites
                 << " invalidation_edges=" << work.replayInvalidationEdges
                 << " shared_word_occurrences=" << work.replaySharedWordOccurrences
                 << " unreused_updates=" << work.unreusedUpdates
                 << " sources=" << work.sourceHandles << " rearming_discharged=" << work.rearmingDischarged
                 << " rearming_composed=" << work.rearmingComposed
                 << " rearming_restored=" << work.rearmingRestored
                 << " rearming_pairs=" << work.rearmingPairVisits
                 << " rearming_query_sites=" << work.rearmingQuerySites
                 << " acknowledgments=" << work.acknowledgments
                 << " common_cut=" << work.commonCutTransfers
                 << " decisions=" << report.decisions.size() << " endpoints=" << report.ledger.size();
    // Components actually kept across updates, including unchanged siblings.
    // Per-solve traces distinguish prefix reuse from non-prefix reuse and
    // include the dependency walk; this count alone is not a cost measure.
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
    if (!report.replayTraces.empty()) {
      const auto &components = report.replayTraces.front().components;
      for (std::size_t i = 0; i < components.size(); ++i) {
        llvm::errs() << "replay_component function=" << function.getSymName() << " id=" << i
                     << " sites=" << components[i].sites << " cyclic=" << components[i].cyclic
                     << " successors=";
        for (auto next : components[i].successors) llvm::errs() << next << ",";
        llvm::errs() << "\n";
      }
    }
    for (const auto &trace : report.replayTraces) {
      llvm::errs() << "replay_trace function=" << function.getSymName()
                   << " version=" << trace.version << " current=" << trace.current
                   << " active=" << trace.activeComponent << " changed_boundary=" << trace.changedBoundary
                   << " fixed_boundary=" << trace.fixedBoundary << " resume=" << trace.resume
                   << " shared_lowerings=" << trace.sharedWordLowerings
                   << " reused_sites=" << trace.reusedSites << " unique=" << trace.uniqueSites
                   << " sibling_components=" << trace.siblingComponents
                   << " invalidation_sites=" << trace.invalidationSites
                   << " invalidation_edges=" << trace.invalidationEdges
                   << " shared_word_occurrences=" << trace.sharedWordOccurrences
                   << " evaluations=" << trace.evaluations << " joins=" << trace.successorJoins
                   << " changed_joins=" << trace.changedJoins << " finalized=" << trace.finalizedQueries
                   << " microseconds=" << trace.microseconds << " success=" << trace.success << " cuts=";
      for (auto cut : trace.changedCuts) llvm::errs() << cut << ",";
      llvm::errs() << " changed_components=";
      for (auto component : trace.changedComponents) llvm::errs() << component << ",";
      llvm::errs() << " components=";
      for (std::size_t i = 0; i < trace.components.size(); ++i) {
        const auto &component = trace.components[i];
        if (component.evaluations)
          llvm::errs() << i << ":" << component.sites << ":" << component.cyclic << ":"
                       << component.uniqueSites << ":" << component.evaluations << ",";
      }
      llvm::errs() << "\n";
    }
    for (const auto &decision : report.decisions) if (decision.publicationAtWordStart) {
      llvm::errs() << "source_gap publication=" << decision.publication << " gap=word_start consumer="
                   << decision.consumer << " source=" << pipeName(decision.source)
                   << " observer=" << pipeName(decision.observer) << " cells=";
      for (const auto &requirement : decision.required) llvm::errs() << requirement.cell << ",";
      llvm::errs() << "\n";
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
bool frontierFile(MLIRContext &context, const char *path, bool observations = false, bool explain = false, bool firstWrites = false) {
  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) return false;
  bool accepted = true;
  module->walk([&](func::FuncOp function) {
    if (function.isDeclaration()) return;
    oahs::NativeAnalysis imported;
    if (failed(oahs::testing::analyzeSelectedHandoffSync(function, imported, false, firstWrites))) {
      accepted = false;
      return;
    }
    oahs::selected::Control control(imported.program);
    oahs::StorageFrontierAnalysis storage(imported.program);
    oahs::selected::RequirementFrontiers frontiers(imported.program, control, storage);
    if (explain) {
      const auto &p = imported.program;
      llvm::outs() << "explain_function " << function.getSymName() << "\n";
      for (const auto &note : imported.observationNotes)
        llvm::outs() << "observation_note " << note << "\n";
      for (std::size_t id = 0; id < p.operations.size(); ++id) {
        llvm::outs() << "operation " << id << " pipe=" << pipeName(p.operations[id].pipe);
        for (const auto &a : p.operations[id].accesses)
          llvm::outs() << " cell=" << a.cell << ":" << a.read << a.write;
        llvm::outs() << " native=";
        imported.phases[id]->print(llvm::outs());
        llvm::outs() << "\n";
      }
      for (oahs::Cut site = 0; site < oahs::commandCutCount(p); ++site) {
        llvm::outs() << "site " << site << " operation=" << oahs::operationAtCut(p, site)
                     << " word=" << oahs::canonicalCommandCut(p, site)
                     << " cyclic=" << control.components[control.component[site]].cyclic << "\n";
      }
      for (unsigned cell = 0; cell < p.cells.size(); ++cell) {
        llvm::outs() << "cell_topology cell=" << cell << " space=" << p.cells[cell].addressSpace
                     << " acyclic_writers=";
        for (oahs::Cut site = 0; site < control.graph.sites.size(); ++site) {
          const auto operation = control.graph.operations[site];
          if (!control.reachable[site] || operation == oahs::NoAnalysisId ||
              control.components[control.component[site]].cyclic) continue;
          for (const auto &access : p.operations[operation].accesses)
            if (access.cell == cell && access.write)
              llvm::outs() << operation << "@" << site << ",";
        }
        llvm::outs() << "\n";
      }
      oahs::SelectedOptions diagnosticOptions;
      diagnosticOptions.firstWriteConsumers = firstWrites;
      const auto plan = oahs::constructSelectedPlan(p, {}, diagnosticOptions);
      accepted &= plan.success;
      if (!plan.success) for (const auto &endpoint : plan.ledger)
        llvm::outs() << "failed_endpoint id=" << endpoint.id << " cut=" << endpoint.cut
                     << " kind=" << unsigned(endpoint.command.kind)
                     << " source=" << pipeName(endpoint.command.source)
                     << " observer=" << pipeName(endpoint.command.observer)
                     << " key=" << endpoint.command.key << "\n";
      auto requirements = [](const auto &rs) {
        for (const auto &r : rs)
          llvm::outs() << " requirement=" << r.cell << ":" << pipeName(r.source)
                       << ":" << r.sourceWrite << ":" << r.consumer
                       << ":" << r.consumerRead << r.consumerWrite;
      };
      for (const auto &f : plan.fences) {
        llvm::outs() << "fence cut=" << f.cut << " observer=" << pipeName(f.observer);
        requirements(f.residuals);
        llvm::outs() << "\n";
      }
      for (const auto &d : plan.decisions) {
        llvm::outs() << "decision consumer=" << d.consumer << " publication=" << d.publication
                     << " source=" << pipeName(d.source) << " observer=" << pipeName(d.observer)
                     << " common=" << d.commonCut << " enlarged=" << d.enlargedPrefix;
        requirements(d.required);
        llvm::outs() << "\n";
      }
      for (const auto &channel : plan.channels) {
        llvm::outs() << "channel owner=" << channel.owner << " source=" << pipeName(channel.source)
                     << " observer=" << pipeName(channel.observer) << " cell=" << channel.cell
                     << " publications=";
        for (auto cut : channel.publications) llvm::outs() << cut << ",";
        llvm::outs() << " acquisitions=";
        for (auto cut : channel.acquisitions) llvm::outs() << cut << ",";
        llvm::outs() << "\n";
      }
    }
    if (!control.complete || !storage.complete() || !frontiers.complete()) {
      llvm::errs() << "frontier analysis failed for " << function.getSymName() << ": "
                   << (!control.complete ? control.reason :
                       !storage.complete() ? storage.reason() : frontiers.reason()) << "\n";
      accepted = false;
      return;
    }
    if (observations) {
      // Report the existing importer/control views. These are observations and
      // candidate frontiers, not a second lifetime analysis or completion proof.
      for (const auto &note : imported.observationNotes)
        llvm::outs() << "observation_note\t" << note << "\n";
      for (const auto &loop : imported.program.observed->loops) {
        llvm::outs() << "reader_owner\t" << loop.owner
                     << "\tentry\t" << loop.entry << "\texit\t" << loop.exit
                     << "\tfirst_visit_sites\t" << loop.firstVisitPrefix.size();
        if (loop.entry < imported.cuts.size())
          if (auto original = dyn_cast_or_null<scf::ForOp>(imported.cuts[loop.entry])) {
            auto bound = [](Value value) {
              if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
                llvm::outs() << constant.value();
              else llvm::outs() << "unknown";
            };
            llvm::outs() << "\tlower\t"; bound(original.getLowerBound());
            llvm::outs() << "\tupper\t"; bound(original.getUpperBound());
            llvm::outs() << "\tstep\t"; bound(original.getStep());
            unsigned choices = 0;
            original.walk([&](scf::IfOp) { ++choices; });
            llvm::outs() << "\toriginal_choices\t" << choices;
          }
        std::set<oahs::Cut> finalWords;
        for (auto site : loop.sites) {
          const auto observation = imported.program.observed->sites[site].observation;
          if (observation == oahs::NoAnalysisId) continue;
          for (const auto &atom : imported.program.observed->observations[observation].atoms)
            if (atom.kind == oahs::ObservationAtom::LoopHasNext && atom.owner == loop.owner && atom.value == 0)
              finalWords.insert(control.canonicalCut[site]);
        }
        llvm::outs() << "\tfinal_visit_words\t" << finalWords.size() << "\tfirst_input_sites";
        for (const auto &entry : control.loopEntries) if (entry.entry == loop.entry)
          for (auto site : entry.firstInputConsumers) llvm::outs() << "\t" << site;
        llvm::outs() << "\n";
        for (const auto &entry : control.loopEntries) if (entry.entry == loop.entry)
          for (auto site : entry.firstInputConsumers) {
            const auto phase = imported.program.observed->sites[site].operation;
            if (phase == oahs::NoAnalysisId) continue;
            llvm::outs() << "first_input\towner\t" << loop.owner << "\tsite\t" << site
                         << "\tphase\t" << phase << "\tread_cells";
            for (const auto &access : imported.program.operations[phase].accesses)
              if (access.read) llvm::outs() << "\t" << access.cell;
            llvm::outs() << "\n";
          }
      }
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
bool choiceConsumerPlacement(MLIRContext &context) {
  const char *source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @choice_banks(%src: !pto.partition_tensor_view<1x32xf32>, %choose: i1)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %z = arith.constant 256 : i64
    %w = arith.constant 384 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    %out = pto.alloc_tile addr = %z : !pto.tile_buf<vec, 1x32xf32>
    %other = pto.alloc_tile addr = %w : !pto.tile_buf<vec, 1x32xf32>
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
    scf.if %choose {
      pto.tabs ins(%a : !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    } else {
      pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%out : !pto.tile_buf<vec, 1x32xf32>)
    }
    pto.tabs ins(%b : !pto.tile_buf<vec, 1x32xf32>) outs(%other : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  if (!module) return false;
  auto function = *module->getOps<func::FuncOp>().begin();
  oahs::SelectedPlan report;
  if (!check(succeeded(oahs::testing::runSelectedHandoffSyncWithMutation(function, {}, &report)),
             "choice readiness construction/reconstruction")) return false;
  if (!check(report.work.choiceTransfers == 1, "native choice readiness not selected")) return false;
  unsigned loads = 0; bool early = false, outside = false;
  function.walk([&](Operation *op) {
    if (isa<TLoadOp>(op)) ++loads;
    if (auto set = dyn_cast<SetFlagOp>(op))
      early |= loads == 1 && set.getSrcPipe().getPipe() == PIPE::PIPE_MTE2 &&
               set.getDstPipe().getPipe() == PIPE::PIPE_V;
    if (auto wait = dyn_cast<WaitFlagOp>(op))
      outside |= loads == 2 && !op->getParentOfType<scf::IfOp>() &&
                 wait.getSrcPipe().getPipe() == PIPE::PIPE_MTE2 &&
                 wait.getDstPipe().getPipe() == PIPE::PIPE_V;
  });
  return check(early && outside, "native bank readiness lost its early source or common choice receipt");
}

int main(int argc, char **argv) {
  MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<PTODialect, arith::ArithDialect, scf::SCFDialect, func::FuncDialect>();
  if (argc >= 3 && StringRef(argv[1]) == "--construct") {
    oahs::SelectedOptions options;
    for (int i = 3; i < argc; ++i) {
      const StringRef flag(argv[i]);
      if (flag == "--no-recurring-trials") options.recurringOmissionTrials = false;
      else if (flag == "--no-helper-trials") options.finalHelperTrials = false;
      else if (flag == "--no-frontier-motion") options.movingFrontiers = false;
      else if (flag == "--no-reader-return-sharing") options.shareReaderReturns = false;
      else if (flag == "--source-gaps") options.sourceGaps = true;
      else if (flag == "--defer-acyclic-acks") options.deferredAcyclicAcknowledgments = true;
      else if (flag == "--class-invariant-inputs") options.classInvariantInputs = true;
      else if (flag == "--no-choice-consumer-frontiers") options.choiceConsumerFrontiers = false;
      else if (flag == "--final-read-sources") options.finalReadSources = true;
      else if (flag == "--first-write-consumers") options.firstWriteConsumers = true;
      else if (flag == "--equal-coverage-binding") options.equalCoverageBinding = true;
      else if (flag == "--trace-replay") options.traceReplay = true;
      else if (flag == "--prefix-replay") options.siblingReplayReuse = false;
      else { llvm::errs() << "unknown construction option: " << flag << "\n"; return 2; }
    }
    return runFile(context, argv[2], options) ? 0 : 1;
  }
  if (argc == 3 && StringRef(argv[1]) == "--frontiers") {
    return frontierFile(context, argv[2]) ? 0 : 1;
  }
  if (argc == 3 && StringRef(argv[1]) == "--observations") {
    return frontierFile(context, argv[2], true) ? 0 : 1;
  }
  if ((argc == 3 || (argc == 4 && StringRef(argv[3]) == "--first-write-consumers")) &&
      StringRef(argv[1]) == "--explain") {
    return frontierFile(context, argv[2], false, true, argc == 4) ? 0 : 1;
  }
  if (argc != 1) {
    llvm::errs() << "usage: pto-oahs-selected-test [--construct INPUT | --frontiers INPUT | --observations INPUT | --explain INPUT]\n";
    return 2;
  }
  const bool passed = positive(context, ordinary, "ordinary") && positive(context, loop, "loop") &&
                      positive(context, recurrence, "recurrence") &&
                      positive(context, collective, "collective") &&
                      positive(context, queue, "queue") && exactCommandEmission(context) && mutations(context) && constantAddresses(context) &&
                      choiceConsumerPlacement(context) && slotMappings(context) && accumulatorOrdering(context) && firstUseOrdering(context) && fifoSlotQualification(context) && staticFifoSlotQualification(context) && firstConsumerPlacement(context) && firstWritePlacement(context) && finalReadSourcePlacement(context) && lastReaderPlacement(context) && jointReaderPlacement(context);
  return passed ? 0 : 1;
}
