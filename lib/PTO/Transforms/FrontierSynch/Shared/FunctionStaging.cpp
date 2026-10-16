// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/FunctionStaging.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
LogicalResult copyDependencies(func::FuncOp original, ModuleOp staging)
{
    auto parent = original->getParentOfType<ModuleOp>();
    llvm::DenseSet<Operation*> seen{original.getOperation()};
    SmallVector<Operation*> pending{original.getOperation()};
    for (std::size_t i = 0; i < pending.size(); ++i) {
        auto uses = SymbolTable::getSymbolUses(pending[i]);
        if (!uses) { return original.emitError("cannot determine staging symbol dependencies"); }
        for (const auto& use : *uses) {
            auto* symbol = SymbolTable::lookupNearestSymbolFrom(use.getUser(), use.getSymbolRef().getRootReference());
            if (!symbol) { return original.emitError("unresolved staging symbol dependency"); }
            // A symbol defined inside the copied operation is already included.
            if (pending[i]->isAncestor(symbol)) { continue; }
            if (!parent || symbol->getParentOp() != parent) {
                return original.emitError("staging dependency is outside the owned module");
            }
            if (seen.insert(symbol).second) {
                pending.push_back(symbol);
                staging.getBody()->push_back(symbol->clone());
            }
        }
    }
    return success();
}
} // namespace
FunctionStaging::FunctionStaging(func::FuncOp original)
    : module(ModuleOp::create(original.getLoc()))
{
    if (auto parent = original->getParentOfType<ModuleOp>()) { (*module)->setAttrs(parent->getAttrs()); }
    if (failed(copyDependencies(original, *module))) { return; }
    function = cast<func::FuncOp>(original->clone(mapping));
    module->push_back(function);
}
void FunctionStaging::commit(func::FuncOp original)
{
    original->setAttrs(function->getAttrs());
    original.getBody().takeBody(function.getBody());
}
} // namespace mlir::pto::frontiersynch
