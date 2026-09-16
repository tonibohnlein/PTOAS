// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/OAHS/Native.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/SyncOriginClosure.h"
#include "PTO/Transforms/OAHS/Prefixes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace mlir;
using namespace mlir::pto;
namespace {
// Test-only opcodes unknown to the translator and both constructors. Their
// interfaces, including deliberately incomplete declarations, are the test.
class OrdinaryProbeOp
    : public Op<OrdinaryProbeOp, OpTrait::OneOperand, OpTrait::ZeroResults,
                OpPipeInterface::Trait, MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrdinaryProbeOp)
  using Op::Op;
  static StringRef getOperationName() { return "sync_probe.ordinary"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  PIPE getPipe() { return PIPE::PIPE_V; }
  void getEffects(SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
    auto mode = (*this)->getAttrOfType<IntegerAttr>("mode");
    if (mode && mode.getInt() == 1) {
      effects.emplace_back(MemoryEffects::Read::get());
    } else if (mode && mode.getInt() == 2) {
      effects.emplace_back(MemoryEffects::Allocate::get(),
                           &getOperation()->getOpOperand(0));
    } else if (mode && mode.getInt() == 3) {
      effects.emplace_back(MemoryEffects::Read::get(),
                           &getOperation()->getOpOperand(0),
                           SideEffects::AutomaticAllocationScopeResource::get());
    } else {
      effects.emplace_back(MemoryEffects::Read::get(),
                           &getOperation()->getOpOperand(0));
    }
  }
};
class PipeOnlyProbeOp
    : public Op<PipeOnlyProbeOp, OpTrait::OneOperand, OpTrait::ZeroResults,
                OpPipeInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipeOnlyProbeOp)
  using Op::Op;
  static StringRef getOperationName() { return "sync_probe.pipe_only"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  PIPE getPipe() { return PIPE::PIPE_V; }
};
class SyncProbeDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SyncProbeDialect)
  static StringRef getDialectNamespace() { return "sync_probe"; }
  explicit SyncProbeDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<SyncProbeDialect>()) {
    addOperations<OrdinaryProbeOp, PipeOnlyProbeOp>();
  }
};
} // namespace

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
static void testSharedSemantics(MLIRContext &context) {
  // These operations have no OAHS registration or single-phase marker. Their
  // ordinary production interfaces are sufficient for either constructor.
  const char *source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ordinary(%src: !pto.tile_buf<vec, 1x32xf32>,
                      %dst: !pto.tile_buf<vec, 1x32xf32>) {
    pto.tsqrt ins(%src : !pto.tile_buf<vec, 1x32xf32>)
      outs(%dst : !pto.tile_buf<vec, 1x32xf32>)
    pto.tdiv ins(%dst, %src : !pto.tile_buf<vec, 1x32xf32>, !pto.tile_buf<vec, 1x32xf32>)
      outs(%dst : !pto.tile_buf<vec, 1x32xf32>)
    return
  }
})mlir";
  auto module = parseSourceString<ModuleOp>(source, &context);
  require(bool(module));
  auto function = module->lookupSymbol<func::FuncOp>("ordinary");
  const auto before = text(function);
  MemoryDependentAnalyzer aliases;
  SyncIRs phases;
  Buffer2MemInfoMap buffers;
  PTOIRTranslator translator(phases, aliases, buffers, function,
                             SyncAnalysisMode::NORMALSYNC);
  require(!translator.describeSemantics().complete());
  require(succeeded(translator.Build()));
  auto report = translator.describeSemantics();
  require(report.complete());
  unsigned ordinary = 0;
  for (const auto &record : report.operations)
    ordinary += record.kind == SyncSemanticRecord::Ordinary;
  require(ordinary == 2 && text(function) == before);
  auto *first = report.operations.front().phases.front();
  // A successfully built node with an empty effect list is not completeness.
  const auto reads = first->useVec;
  first->useVec.clear();
  require(!translator.describeSemantics().complete());
  first->useVec = reads;
  const auto pipeline = first->kPipeValue;
  first->kPipeValue = PipelineType::PIPE_M;
  require(!translator.describeSemantics().complete());
  first->kPipeValue = pipeline;
  require(translator.describeSemantics().complete());
  oahs::NativeAnalysis analysis;
  require(succeeded(oahs::analyzeHandoffSync(function, analysis)));
  require(analysis.phases.size() == 2 && text(function) == before);
  require(succeeded(oahs::runHandoffSync(function)));

  const char *probe = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @probe(%src: !pto.tile_buf<vec, 1x32xf32>) {
    "sync_probe.ordinary"(%src) : (!pto.tile_buf<vec, 1x32xf32>) -> ()
    return
  }
})mlir";
  auto probeModule = parseSourceString<ModuleOp>(probe, &context);
  require(bool(probeModule));
  auto probeFunction = probeModule->lookupSymbol<func::FuncOp>("probe");
  require(succeeded(oahs::runHandoffSync(probeFunction)));

  const char *gaps = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @gaps(%src: !pto.tile_buf<vec, 1x32xf32>, %unmapped: f32) {
    "sync_probe.ordinary"(%src) {mode = 1 : i32} : (!pto.tile_buf<vec, 1x32xf32>) -> ()
    "sync_probe.ordinary"(%src) {mode = 2 : i32} : (!pto.tile_buf<vec, 1x32xf32>) -> ()
    "sync_probe.ordinary"(%src) {mode = 3 : i32} : (!pto.tile_buf<vec, 1x32xf32>) -> ()
    "sync_probe.ordinary"(%unmapped) : (f32) -> ()
    "sync_probe.pipe_only"(%src) : (!pto.tile_buf<vec, 1x32xf32>) -> ()
    return
  }
})mlir";
  auto gapModule = parseSourceString<ModuleOp>(gaps, &context);
  require(bool(gapModule));
  auto gapFunction = gapModule->lookupSymbol<func::FuncOp>("gaps");
  const auto gapBefore = text(gapFunction);
  SyncIRs gapPhases;
  Buffer2MemInfoMap gapBuffers;
  PTOIRTranslator gapTranslator(gapPhases, aliases, gapBuffers, gapFunction,
                                SyncAnalysisMode::NORMALSYNC);
  require(succeeded(gapTranslator.Build()));
  auto gapReport = gapTranslator.describeSemantics();
  require(!gapReport.complete());
  unsigned gapCount = 0;
  for (const auto &record : gapReport.operations)
    gapCount += !record.gap.empty();
  require(gapCount == 5 && text(gapFunction) == gapBefore);
  {
    ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {
      return success();
    });
    require(failed(oahs::runHandoffSync(gapFunction)));
    require(text(gapFunction) == gapBefore);
  }

  const char *unsupported = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @variant(%src: !pto.partition_tensor_view<1x32xf32>,
                    %dst: !pto.tile_buf<vec, 1x32xf32>) {
    pto.tload ins(%src : !pto.partition_tensor_view<1x32xf32>)
      outs(%dst : !pto.tile_buf<vec, 1x32xf32>) init_out_buffer = true
    return
  }
})mlir";
  auto rejected = parseSourceString<ModuleOp>(unsupported, &context);
  require(bool(rejected));
  auto variant = rejected->lookupSymbol<func::FuncOp>("variant");
  const auto original = text(variant);
  ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic &) {
    return success();
  });
  require(failed(oahs::runHandoffSync(variant)));
  require(text(variant) == original);
}

static void testPreservedProtocols(MLIRContext &context) {
  const char *source = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @queue(%gm: !pto.ptr<f32, gm>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
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
  for (bool mutate : {false, true}) {
    auto module = parseSourceString<ModuleOp>(source, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("queue");
    const auto original = text(function);
    oahs::NativeAnalysis report;
    require(succeeded(oahs::analyzeHandoffSync(function, report)));
    require(text(function) == original && report.phases.size() == 4 &&
            report.protocols.size() == 6);
    bool aliasRelease = false, popReady = false, pushReady = false;
    for (const auto &residual : report.analysis.residuals) {
      const auto &d = residual.demand;
      aliasRelease |= d.producer == 1 && d.consumer == 2 &&
                      residual.kind == oahs::CompletionRequirement::WAR;
      popReady |= d.producer == 0 && d.consumer == 1;
      pushReady |= d.producer == 1 && d.consumer == 3;
    }
    require(aliasRelease && popReady && pushReady);
    for (const auto &protocol : report.protocols)
      require(protocol.complete() && protocol.crossCoreFlagCount == 4 &&
              protocol.crossCoreFlagBase == 0);
    bool changed = false;
    const auto status = oahs::testing::runHandoffSyncWithMutation(
        function, [&](func::FuncOp working) {
          if (!mutate) return;
          working.walk([&](InitializeL2G2LPipeOp init) {
            init->setAttr("flag_base", IntegerAttr::get(
                IntegerType::get(&context, 32), 4));
            changed = true;
          });
        });
    if (mutate)
      require(changed && failed(status) && text(function) == original);
    else {
      require(succeeded(status) && succeeded(verify(function)));
      unsigned pops = 0, frees = 0, pushes = 0;
      function.walk([&](TPopOp) { ++pops; });
      function.walk([&](TFreeOp) { ++frees; });
      function.walk([&](TPushOp) { ++pushes; });
      require(pops == 2 && frees == 2 && pushes == 1);
    }
  }
  // A runtime base and a descriptor rebound from an allocated handle must
  // retain the same hazard. No declaration-only admission shortcut is used.
  for (bool allocated : {false, true}) {
    std::string variant(source);
    if (allocated) {
      const std::string declaration = "pto.declare_tile ->";
      for (size_t pos = 0; (pos = variant.find(declaration, pos)) != std::string::npos; ) {
        variant.replace(pos, declaration.size(), "pto.alloc_tile addr = %outaddr :");
        ++pos;
      }
    } else {
      variant.replace(variant.find("%gm: !pto.ptr<f32, gm>"),
                      std::string("%gm: !pto.ptr<f32, gm>").size(),
                      "%gm: !pto.ptr<f32, gm>, %base: i32");
      variant.erase(variant.find("%base = arith.constant 0 : i32"),
                    std::string("%base = arith.constant 0 : i32").size());
    }
    auto module = parseSourceString<ModuleOp>(variant, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("queue");
    oahs::NativeAnalysis report;
    require(succeeded(oahs::analyzeHandoffSync(function, report)));
    require(llvm::any_of(report.analysis.residuals, [](const auto &r) {
      return r.demand.producer == 1 && r.demand.consumer == 2 &&
             r.kind == oahs::CompletionRequirement::WAR;
    }));
  }
  {
    // Known disjoint local slots must stay disjoint; sharing a GM ring root
    // does not merge the local byte cells transitively.
    std::string disjoint(source);
    const auto position = disjoint.find("%a = pto.declare_tile");
    disjoint.insert(position, R"mlir(
    %far = arith.constant 2048 : i32
    %otherpipe = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 128,
      slot_num = 2, local_slot_num = 1, flag_base = 4, nosplit = true}
      (%gm : !pto.ptr<f32, gm>, %far : i32) -> !pto.pipe
    )mlir");
    disjoint.replace(disjoint.find("pto.tpop(%b, %pipe"),
                     std::string("pto.tpop(%b, %pipe").size(),
                     "pto.tpop(%b, %otherpipe");
    auto module = parseSourceString<ModuleOp>(disjoint, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("queue");
    oahs::NativeAnalysis report;
    require(succeeded(oahs::analyzeHandoffSync(function, report)));
    require(llvm::none_of(report.analysis.residuals, [](const auto &r) {
      return r.demand.producer == 1 && r.demand.consumer == 2;
    }));
  }
  {
    // Bidirectional MAT receives bind the peer operand, not the VEC base.
    const char *peer = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @peer(%gm: !pto.ptr<f32, gm>, %vec: i32, %mat: i32)
     attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %p = pto.initialize_l2g2l_pipe{dir_mask = 3, slot_size = 1024,
    slot_num = 2, local_slot_num = 2, flag_base = 0, nosplit = true}
    (%gm : !pto.ptr<f32, gm>, %vec : i32, %mat : i32) -> !pto.pipe
  %a = pto.declare_tile -> !pto.tile_buf<mat, 16x16xf32>
  pto.tpop(%a, %p : !pto.tile_buf<mat, 16x16xf32>, !pto.pipe) {split = 0}
  pto.tfree(%p : !pto.pipe) {split = 0}
  return
 }
})mlir";
    auto module = parseSourceString<ModuleOp>(peer, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("peer");
    function.walk([&](TPopOp pop) {
      auto model = getSyncProtocolModel(pop);
      require(model && model->complete() &&
              model->localBase == function.getArgument(2));
    });
  }
  {
    // A view of a popped tile must still cover later local ring slots.
    const char *viewSource = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @slots(%gm: !pto.ptr<f32, gm>)
     attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %zero = arith.constant 0 : index
  %base = arith.constant 0 : i32
  %last = arith.constant 128 : i64
  %p = pto.initialize_l2g2l_pipe{dir_mask = 1, slot_size = 128,
    slot_num = 2, local_slot_num = 2, flag_base = 0, nosplit = true}
    (%gm : !pto.ptr<f32, gm>, %base : i32) -> !pto.pipe
  %a = pto.declare_tile -> !pto.tile_buf<vec, 1x32xf32>
  pto.tpop(%a, %p : !pto.tile_buf<vec, 1x32xf32>, !pto.pipe) {split = 0}
  %view = pto.subview %a[%zero, %zero] sizes [1, 16] :
    !pto.tile_buf<vec, 1x32xf32> -> !pto.tile_buf<vec, 1x16xf32>
  %b = pto.alloc_tile addr = %last : !pto.tile_buf<vec, 1x16xf32>
  pto.tabs ins(%view : !pto.tile_buf<vec, 1x16xf32>)
    outs(%b : !pto.tile_buf<vec, 1x16xf32>)
  pto.tfree(%p : !pto.pipe) {split = 0}
  return
 }
})mlir";
    auto module = parseSourceString<ModuleOp>(viewSource, &context);
    require(bool(module));
    auto function = module->lookupSymbol<func::FuncOp>("slots");
    MemoryDependentAnalyzer aliases;
    SyncIRs phases;
    Buffer2MemInfoMap buffers;
    PTOIRTranslator translator(phases, aliases, buffers, function,
                               SyncAnalysisMode::NORMALSYNC);
    require(succeeded(translator.Build()) && translator.describeSemantics().complete());
    Value view, last;
    function.walk([&](SubViewOp op) { view = op.getResult(); });
    function.walk([&](AllocTileOp op) { last = op.getResult(); });
    require(view && last && !buffers[view].empty() && !buffers[last].empty());
    require(aliases.MemAlias(buffers[view].front().get(), buffers[last].front().get()));
  }
  const char *atomic = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @atomic(%tile: !pto.tile_buf<vec, 1x32xf32>,
                    %dst: !pto.partition_tensor_view<1x32xf32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    pto.tstore ins(%tile : !pto.tile_buf<vec, 1x32xf32>)
      outs(%dst : !pto.partition_tensor_view<1x32xf32>)
      {atomicType = #pto<atomic_type atomic_add>}
    return
  }
})mlir";
  auto module = parseSourceString<ModuleOp>(atomic, &context);
  require(bool(module));
  auto function = module->lookupSymbol<func::FuncOp>("atomic");
  function.walk([&](TStoreOp store) {
    SmallVector<MemoryEffects::EffectInstance> effects;
    store.getEffects(effects);
    bool read = false, write = false;
    for (const auto &effect : effects) {
      if (effect.getValue() != store.getDst()) continue;
      read |= isa<MemoryEffects::Read>(effect.getEffect());
      write |= isa<MemoryEffects::Write>(effect.getEffect());
    }
    require(read && write);
  });
  require(succeeded(oahs::runHandoffSync(function)));
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  MLIRContext context(registry);
  context.loadDialect<SyncProbeDialect, PTODialect, arith::ArithDialect, scf::SCFDialect,
                      func::FuncDialect>();
  if (argc == 3 && StringRef(argv[1]) == "--analyze") {
    auto module = parseSourceFile<ModuleOp>(argv[2], &context);
    if (!module)
      return 1;
    bool complete = true;
    module->walk([&](func::FuncOp function) {
      if (function.isDeclaration())
        return;
      const auto before = text(function);
      MemoryDependentAnalyzer aliases;
      SyncIRs phases;
      Buffer2MemInfoMap buffers;
      PTOIRTranslator translator(phases, aliases, buffers, function,
                                 SyncAnalysisMode::NORMALSYNC);
      if (failed(translator.Build()) ||
          failed(closeStructuredSyncOrigins(function, phases, buffers))) {
        complete = false;
        return;
      }
      auto report = translator.describeSemantics();
      unsigned ordinary = 0, protocols = 0, gaps = 0;
      for (const auto &record : report.operations) {
        ordinary += record.kind == SyncSemanticRecord::Ordinary;
        protocols += record.kind == SyncSemanticRecord::Protocol;
        if (!record.gap.empty()) {
          ++gaps;
          llvm::outs() << "gap " << record.operation->getName() << ": "
                       << record.gap << "\n";
        }
      }
      llvm::outs() << "function=" << function.getSymName()
                   << " shared_complete=" << report.complete()
                   << " ordinary_phases=" << ordinary
                   << " preserved_protocol_ops=" << protocols << " gaps=" << gaps
                   << "\n";
      if (report.complete()) {
        oahs::NativeAnalysis analysis;
        const bool analyzed = succeeded(oahs::analyzeHandoffSync(function, analysis));
        llvm::outs() << "native_analysis=" << analyzed
                     << " residuals=" << analysis.analysis.residuals.size()
                     << "\n";
        complete &= analyzed;
      } else {
        complete = false;
      }
      require(text(function) == before);
    });
    return complete ? 0 : 1;
  }
  if (argc != 1) {
    llvm::errs() << "usage: pto-oahs-native-test [--analyze INPUT]\n";
    return 2;
  }
  testSharedSemantics(context);
  testPreservedProtocols(context);
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
                  "and shared-semantics/direct predicate-decoder checks passed\n";
}
