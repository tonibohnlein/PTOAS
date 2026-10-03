// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Test the exact shared preflight service on detached clones. This does not
// manufacture native premises or invoke the production analysis through a bypass.
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"
namespace {
std::string render(mlir::Operation* op)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    op->print(stream);
    return text;
}
bool checkDirectedPools(mlir::MLIRContext& context)
{
    using namespace mlir;
    auto module = parseSourceString<ModuleOp>(R"mlir(
      module attributes {pto.target_arch = "a3"} {
        func.func @candidate() attributes {pto.kernel_kind = #pto.kernel_kind<cube>} { return }
      })mlir", &context);
    pto::SyncInput input;
    if (!module || failed(input.build(*module->getOps<func::FuncOp>().begin()))) {
        return false;
    }
    for (auto source : {pto::PIPE::PIPE_MTE2, pto::PIPE::PIPE_MTE3}) {
        for (bool reverse : {false, true}) {
            auto a = reverse ? pto::PIPE::PIPE_FIX : source;
            auto b = reverse ? source : pto::PIPE::PIPE_FIX;
            auto fact = input.target().eventFact(pto::SyncPhysicalCore::AIC, a, b);
            std::string reason;
            if (fact.availability != pto::SyncMechanismAvailability::Documented ||
                failed(input.target().eventPool(pto::SyncPhysicalCore::AIC, a, b, reason))) {
                return false;
            }
        }
    }
    return true;
}
bool checkClone(mlir::func::FuncOp source)
{
    using namespace mlir;
    pto::SyncInput input;
    if (failed(input.build(source))) {
        return false;
    }
    const auto before = render(source);
    IRMapping mapping;
    OwningOpRef<func::FuncOp> clone(cast<func::FuncOp>(source->clone(mapping)));
    if (failed(input.target().retainSourceIds(clone.get(), mapping))) {
        return false;
    }
    if (source->hasAttr("test.mutate_clone")) {
        clone.get()->setAttr(pto::FunctionKernelKindAttr::name,
                      pto::FunctionKernelKindAttr::get(source.getContext(), pto::FunctionKernelKind::Vector));
    }
    std::string reason;
    bool accepted = succeeded(input.target().preflightMechanisms(clone.get(), mapping, reason));
    auto expected = source->getAttrOfType<BoolAttr>("test.preflight_accept");
    if (!expected || accepted != expected.getValue() || render(source) != before) {
        llvm::errs() << source.getSymName() << ": invalid clone preflight or modified source: " << reason << "\n";
        return false;
    }
    return accepted || !reason.empty();
}
} // namespace
int runSyncTargetPreflightChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) {
        return 1;
    }
    std::size_t checked = 0;
    auto result = module->walk([&](mlir::func::FuncOp function) {
        if (function.isDeclaration()) {
            return mlir::WalkResult::advance();
        }
        ++checked;
        return checkClone(function) ? mlir::WalkResult::advance() : mlir::WalkResult::interrupt();
    });
    if (result.wasInterrupted() || checked != 5 || !checkDirectedPools(context)) {
        return 1;
    }
    llvm::outs() << "verified detached-clone preflight: module-only, conflicts, guards, mutation rollback\n";
    return 0;
}
