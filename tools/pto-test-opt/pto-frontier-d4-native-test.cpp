// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/D4ProgramQueries.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <functional>

namespace fs = mlir::pto::frontiersynch;
static void require(bool value, const char* reason)
{
    if (!value) {
        llvm::errs() << "D4 native failure: " << reason << "\n";
        std::exit(1);
    }
}
static std::string print(mlir::Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream out(result);
    operation->print(out);
    return result;
}
int main()
{
    mlir::DialectRegistry dialects;
    dialects.insert<mlir::pto::PTODialect, mlir::func::FuncDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect>();
    mlir::MLIRContext context(dialects);
    context.disableMultithreading();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
!vec = !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16,
    blayout=row_major, slayout=none_box, fractal=512, pad=0>
module attributes {pto.target_arch = "a3"} {
  func.func @d4_children(%g: i1, %src_ptr: !pto.ptr<f32>, %dst_ptr: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %src_addr = arith.constant 0 : i64
    %dst_addr = arith.constant 1024 : i64
    %scale = arith.constant 2.0 : f32
    %src = pto.alloc_tile addr = %src_addr : !vec
    %dst = pto.alloc_tile addr = %dst_addr : !vec
    %src_view = pto.make_tensor_view %src_ptr, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %dst_view = pto.make_tensor_view %dst_ptr, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf32>
    %src_part = pto.partition_view %src_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    %dst_part = pto.partition_view %dst_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%dst_part : !pto.partition_tensor_view<16x16xf32>) outs(%dst : !vec)
    scf.if %g {
      pto.tload ins(%src_part : !pto.partition_tensor_view<16x16xf32>) outs(%src : !vec)
      pto.taxpy ins(%src, %scale : !vec, f32) outs(%dst : !vec)
      scf.yield
    }
    scf.if %g {
      %local_reset = arith.constant 0 : index
      scf.yield
    }
    scf.if %g {
      pto.tload ins(%src_part : !pto.partition_tensor_view<16x16xf32>) outs(%src : !vec)
      pto.taxpy ins(%src, %scale : !vec, f32) outs(%dst : !vec)
      scf.yield
    }
    return
  }
}
)mlir", &context);
    require(module && mlir::succeeded(mlir::verify(*module)), "native D4 fixture failed verification");
    const auto before = print(module->getOperation());
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    mlir::pto::SyncInput input;
    fs::OriginalStructure imported;
    require(mlir::succeeded(input.build(function)) &&
                mlir::succeeded(fs::importOriginalStructure(function, input, imported)), "D4 import failed");
    // A whole W;R child inside extra anonymous wrappers must have the same
    // reader slice and physical use as the unwrapped imported program.
    std::function<void(fs::Region&)> addWrappers = [&](fs::Region& region) {
        for (auto& child : region.children) { addWrappers(child); }
        if (region.kind == fs::Region::Sequence && !region.children.empty()) {
            fs::Region wrapper;
            wrapper.children = std::move(region.children);
            region.children = {std::move(wrapper)};
        }
    };
    addWrappers(imported.body);
    fs::ProgramAnalysis analysis(input, std::move(imported));
    require(analysis.complete(), "D4 original analysis incomplete");
    const auto& original = analysis.structure();
    std::vector<std::size_t> children;
    for (std::size_t i = 0; i < original.originalSites.size(); ++i) {
        if (mlir::isa<mlir::scf::IfOp>(original.originalSites[i])) { children.push_back(i); }
    }
    require(children.size() == 3, "missing original child identities");
    auto payloads = [&](std::size_t owner) {
        std::vector<std::size_t> result;
        for (std::size_t i = 0; i < original.operations.size(); ++i) {
            if (original.operations[i].instruction->elementOp->getParentOp() == original.originalSites[owner]) {
                result.push_back(i);
            }
        }
        return result;
    };
    const auto a = payloads(children[0]), b = payloads(children[2]);
    require(a.size() == 2 && b.size() == 2, "native two-access children changed shape");
    std::size_t cell = fs::NoControlId;
    for (const auto& write : original.operations[a[0]].accesses) {
        for (const auto& read : original.operations[a[1]].accesses) {
            if (write.write && write.definiteWrite && read.read && !read.write && write.cell == read.cell) {
                cell = write.cell;
            }
        }
    }
    require(cell != fs::NoControlId, "native qualified full-cell write/read missing");
    fs::OriginalIntervalRequest request;
    request.version = original.version;
    request.selector.cell = cell;
    request.start = fs::OriginalCut::scope(fs::NoControlId, fs::OriginalCut::Before);
    request.stop = fs::OriginalCut::scope(fs::NoControlId, fs::OriginalCut::After);
    request.continuationOwner = fs::NoControlId;
    const auto prepared = analysis.prepareInterval(request);
    require(prepared.valid, "native owning continuation missing");
    const auto& uses = analysis.originalUsesAt(a[0], cell);
    require(uses.complete, "native D4 factored composition is not connected");
    fs::D4ProgramQueries queries(analysis);
    const auto first = queries.fixedChild(prepared.interval, children[0], a[0], a[1], uses.arena);
    const auto empty = queries.noUseChild(prepared.interval, children[1]);
    const auto second = queries.fixedChild(prepared.interval, children[2], b[0], b[1], uses.arena);
    require(first.occurrenceQualified && second.occurrenceQualified && empty.noUseProved,
            "native D1/All child premise not established");
    std::vector<std::vector<fs::OriginalObligationFamilyId>> obligations;
    for (std::size_t i = 0; i < original.originalSites.size(); ++i) { obligations.push_back(analysis.obligationsAt(i)); }
    const auto result = queries.completeUses(prepared.interval, uses.arena, {first, empty, second});
    require(result.relation.complete && result.obligations == &analysis.obligations(),
            "native D4 did not preserve its physical sequence/obligation interface");
    require(!queries.completeUses(prepared.interval, uses.arena, {second}).relation.complete,
            "native D4 omitted a genuine preceding use");
    require(!queries.completeUses(prepared.interval, uses.arena, {second, first}).relation.complete,
            "native D4 changed original child order");
    require(!queries.completeUses(prepared.interval, uses.arena, {first, first, second}).relation.complete,
            "native D4 duplicated a physical use");
    auto bad = second;
    bad.first.cut.side = fs::OriginalCut::After;
    require(!queries.completeUses(prepared.interval, uses.arena, {first, empty, bad}).relation.complete,
            "native D4 accepted a wrong first-use cut");
    bad = second;
    bad.correspondence.reset();
    const auto unresolved = queries.completeUses(prepared.interval, uses.arena, {first, empty, bad});
    require(!unresolved.relation.complete && !unresolved.relation.unresolved.empty() &&
                unresolved.obligations == &analysis.obligations(), "unsupported native transport lost obligations");
    for (std::size_t i = 0; i < obligations.size(); ++i) {
        require(obligations[i] == analysis.obligationsAt(i), "D4 changed an original obligation identity");
    }
    require(before == print(module->getOperation()), "D4 changed original IR");
    llvm::outs() << "D4 native: qualified children, exact order/coverage, immutable obligations and IR PASS\n";
    return 0;
}
