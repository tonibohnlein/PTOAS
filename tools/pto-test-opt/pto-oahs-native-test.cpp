// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace mlir;
using namespace mlir::pto;
static void require(bool condition) {
  if (!condition)
    std::abort();
}
static std::string text(func::FuncOp function) {
  std::string result;
  llvm::raw_string_ostream out(result);
  function.print(out);
  out.flush();
  return result;
}
int main() {
  DialectRegistry registry;
  MLIRContext context(registry);
  context.loadDialect<PTODialect, arith::ArithDialect, scf::SCFDialect,
                      func::FuncDialect>();
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
    for (const auto &phase : report.program.operations)
      for (const auto &access : phase.accesses)
        require(!access.definiteWrite); // native may-overlap witnesses are not
                                        // exact atoms
    require(report.cuts.size() == report.analysis.cuts.size());
    require(report.cuts.size() > report.phases.size());
    for (unsigned i = 0; i < report.phases.size(); ++i)
      require(report.cuts[report.phaseCuts[i]] == report.phases[i]);
    // Import has not invented definite-whole-overwrite evidence from may-alias
    // witness cells. Reference strong updates remain opt-in frontend facts.
    for (const auto &phase : report.program.operations)
      for (const auto &access : phase.accesses)
        require(!access.definiteWrite);
    const auto contextId = report.analysis.cuts[report.phaseCuts[1]].context;
    require(report.analysis.contexts[contextId].kind ==
            oahs::AnalysisContext::ThenArm);
    require(!report.analysis.retirement.empty());
    require(text(function) == before);
    require(succeeded(oahs::analyzeHandoffSync(function)));
    require(text(function) == before);
    // M2 queries own the imported state and never change native IR. The load's
    // enclosing unconditional cut cannot pair with an optional branch consumer.
    oahs::PrefixQuery prefixes(report.program);
    const auto consumer = report.phaseCuts[1];
    const auto cover = prefixes.coverByPrefixes(consumer);
    require(cover.coversAll() && !cover.selected.empty());
    require(cover.candidates[cover.selected.front()].publication == consumer);
    require(
        !prefixes.inspectPrefix(oahs::Pipe::MTE2, report.phaseCuts[0], consumer)
             .matchingEstablished);
    require(text(function) == before);
    // The same imported program/analysis explains and verifies the candidate.
    const auto plan = oahs::construct(report.program);
    require(plan.success &&
            oahs::analyze(report.program, plan.commands).verified());
  }
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("test");
    const std::string before = text(function);
    bool changed = false;
    auto result = oahs::testing::runHandoffSyncWithMutation(
        function, [&](func::FuncOp working) {
          if (!mutation)
            return;
          if (mutation == 3) {
            // An extra global fence can remain memory-safe but is not the
            // selected packet word. Exact emission identity must reject it
            // transactionally.
            auto *ret = working.getBody().front().getTerminator();
            OpBuilder builder(ret);
            builder.create<BarrierOp>(
                ret->getLoc(),
                PipeAttr::get(working.getContext(), PIPE::PIPE_ALL));
            changed = true;
            return;
          }
          scf::IfOp choice;
          working.walk([&](scf::IfOp op) { choice = op; });
          mlir::Operation *victim = nullptr;
          working.walk([&](mlir::Operation *op) {
            if (victim || !isa<BarrierOp, WaitFlagOp>(op))
              return;
            if (mutation == 1 && isa<WaitFlagOp>(op))
              victim = op;
            if (mutation == 2 && isa<BarrierOp>(op) &&
                cast<BarrierOp>(op).getPipe().getPipe() == PIPE::PIPE_ALL &&
                op->getBlock() == &working.getBody().front())
              victim = op;
          });
          require(victim != nullptr);
          victim->moveBefore(choice.getElseRegion().front().getTerminator());
          changed = true;
        });
    if (!mutation)
      require(succeeded(result) && succeeded(verify(function)));
    else
      require(changed && failed(result) && text(function) == before);
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
    const auto cover = query.coverByPrefixes(imported.phaseCuts[2]);
    require(cover.coversAll() && cover.selected.size() == 1);
    const auto &prefix = cover.candidates[cover.selected.front()];
    require(prefix.publication == imported.phaseCuts[1] &&
            prefix.acquisition == imported.phaseCuts[2]);
    require(prefix.sourceContext == prefix.targetContext);
    require(text(function) == before);
    require(succeeded(oahs::runHandoffSync(function)));
    // The analysis report's old phase pointers are invalid after replacement.
    llvm::SmallVector<TLoadOp> loads;
    function.walk([&](TLoadOp load) { loads.push_back(load); });
    require(loads.size() == 2);
    require(loads[1]->getPrevNode() && isa<SetFlagOp>(loads[1]->getPrevNode()));
    require(loads[1]->getNextNode() &&
            isa<WaitFlagOp>(loads[1]->getNextNode()));
    require(succeeded(verify(function)));
  }
  {
    // Native normalized first/tail observations use only the original IV and
    // upper bound. They introduce guarded synchronization, not payload copies.
    const char *loopSource = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @observed_loop(%src: !pto.partition_tensor_view<1x32xf32>, %n: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %x = arith.constant 0 : i64
    %y = arith.constant 128 : i64
    %a = pto.alloc_tile addr = %x : !pto.tile_buf<vec, 1x32xf32>
    %b = pto.alloc_tile addr = %y : !pto.tile_buf<vec, 1x32xf32>
    scf.for %i = %zero to %n step %one {
      pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) outs(%a : !pto.tile_buf<vec, 1x32xf32>)
      pto.tadd ins(%a, %a : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>) outs(%b : !pto.tile_buf<vec, 1x32xf32>)
    }
    return
  }
})mlir";
    for (unsigned mutation = 0; mutation < 2; ++mutation) {
      auto module = parseSourceString<ModuleOp>(loopSource, &context);
      require(bool(module));
      auto function = module->lookupSymbol<func::FuncOp>("observed_loop");
      const auto before = text(function);
      oahs::NativeAnalysis analysis;
      require(succeeded(oahs::analyzeHandoffSync(function, analysis)));
      require(text(function) == before);
      bool observed = false;
      for (const auto &observation : analysis.program.observed->observations)
        observed |= !observation.atoms.empty();
      require(observed);
      bool changed = false;
      auto result = oahs::testing::runHandoffSyncWithMutation(
          function, [&](func::FuncOp working) {
            if (!mutation)
              return;
            scf::IfOp guard;
            working.walk([&](scf::IfOp op) {
              if (!guard)
                guard = op;
            });
            require(bool(guard));
            OpBuilder builder(guard);
            auto always =
                builder.create<arith::ConstantIntOp>(guard.getLoc(), 1, 1);
            guard->setOperand(0, always.getResult());
            changed = true;
          });
      if (mutation)
        require(changed && failed(result) && text(function) == before);
      else {
        require(succeeded(result) && succeeded(verify(function)));
        unsigned loads = 0, adds = 0, loops = 0, guards = 0;
        function.walk([&](TLoadOp) { ++loads; });
        function.walk([&](TAddOp) { ++adds; });
        function.walk([&](scf::ForOp) { ++loops; });
        function.walk([&](scf::IfOp) { ++guards; });
        require(loads == 1 && adds == 1 && loops == 1 && guards > 0);
      }
    }
  }
  {
    // Direct predicate read-back: the normal mutation hook's whole-IR identity
    // check intentionally runs before the decoder, so it cannot test these
    // arithmetic rejection paths by itself. All expressions here are valid IR.
    const char *decoderSource = R"mlir(
module {
  func.func @decoder(%n: index, %other: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %i = %zero to %n step %one {
    }
    scf.for %j = %zero to %n step %one {
    }
    return
  }
})mlir";
    auto decoderModule = parseSourceString<ModuleOp>(decoderSource, &context);
    require(bool(decoderModule));
    auto function = decoderModule->lookupSymbol<func::FuncOp>("decoder");
    llvm::SmallVector<scf::ForOp> loops;
    function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
    require(loops.size() == 2);
    auto loop = loops.front();
    auto *anchor = loop.getRegion().front().getTerminator();
    llvm::SmallVector<std::pair<std::size_t, mlir::Operation *>> owners{
        {42, loop.getOperation()}};
    oahs::OriginalObservation observation{
        0,
        {{oahs::ObservationAtom::LoopResidue, 42, 2, 1},
         {oahs::ObservationAtom::LoopHasPrevious, 42, 2, 1},
         {oahs::ObservationAtom::LoopHasNext, 42, 2, 1}},
        true};
    for (unsigned mutation = 0; mutation < 12; ++mutation) {
      OpBuilder builder(anchor);
      const auto loc = anchor->getLoc();
      auto constant = [&](int64_t value) -> Value {
        return builder.create<arith::ConstantIndexOp>(loc, value);
      };
      Value iv = loop.getInductionVar();
      Value residueInput = mutation == 1 ? function.getArgument(1) : iv;
      Value residue = builder.create<arith::RemUIOp>(
          loc, residueInput, constant(mutation == 2 ? 3 : 2));
      Value residueTest = builder.create<arith::CmpIOp>(
          loc, mutation == 3 ? arith::CmpIPredicate::ne
                             : arith::CmpIPredicate::eq,
          residue, constant(1));
      const bool falseObservations = mutation == 11;
      Value previous = builder.create<arith::CmpIOp>(
          loc, mutation == 4 ? arith::CmpIPredicate::uge
               : falseObservations ? arith::CmpIPredicate::slt
                                   : arith::CmpIPredicate::sge,
          iv, constant(2));
      Value bound = mutation == 5 ? function.getArgument(1)
                                 : loop.getUpperBound();
      Value remaining = mutation == 6
                            ? Value(builder.create<arith::SubIOp>(loc, iv, bound))
                            : Value(builder.create<arith::SubIOp>(loc, bound, iv));
      Value next = builder.create<arith::CmpIOp>(
          loc, mutation == 7 ? arith::CmpIPredicate::sge
               : falseObservations ? arith::CmpIPredicate::sle
                                   : arith::CmpIPredicate::sgt,
          remaining, constant(mutation == 8 ? 3 : 2));
      // Commuting conjuncts is allowed; changing their meaning is not.
      Value pair = builder.create<arith::AndIOp>(
          loc, mutation == 9 ? next : residueTest,
          mutation == 9 ? residueTest : previous);
      Value condition = builder.create<arith::AndIOp>(
          loc, pair, mutation == 9 ? previous : next);
      if (mutation == 10)
        condition = builder.create<arith::ConstantIntOp>(loc, 1, 1);
      auto expected = observation;
      if (falseObservations) {
        expected.atoms[1].value = 0;
        expected.atoms[2].value = 0;
      }
      const auto before = text(function);
      const auto status = oahs::testing::checkHandoffObservationPredicate(
          expected, anchor, owners, condition);
      require(succeeded(status) ==
              (mutation == 0 || mutation == 9 || falseObservations));
      require(text(function) == before && succeeded(verify(function)));
      if (mutation == 0) {
        auto wrongOwner = owners;
        wrongOwner[0].second = loops.back().getOperation();
        require(failed(oahs::testing::checkHandoffObservationPredicate(
            expected, anchor, wrongOwner, condition)));
        wrongOwner[0].first = 43;
        require(failed(oahs::testing::checkHandoffObservationPredicate(
            expected, anchor, wrongOwner, condition)));
        wrongOwner = owners;
        wrongOwner.push_back(owners.front());
        require(failed(oahs::testing::checkHandoffObservationPredicate(
            expected, anchor, wrongOwner, condition)));
      }
    }
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
    many += "pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>) "
            "outs(%a : !pto.tile_buf<vec, 1x32xf32>)\n";
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
  manyFunction.walk([&](BarrierOp barrier) {
    drains += barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
  });
  require(drains == 1); // original invocation retirement, not a budget fallback
  llvm::outs() << "OAHS native placement/atomicity/grouped-footprint/prefix "
                  "and direct predicate-decoder checks passed\n";
}
