// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Transactions require exclusive ownership of the enclosing symbol table. The
// public allocator is module-scoped; function-pass workers must not use this API.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FUNCTIONSTAGING_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FUNCTIONSTAGING_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
namespace mlir::pto::frontiersynch {
// Copy only the transitive symbol dependencies needed by the target. Failed
// staging or lowering leaves the original function and all siblings unchanged.
struct FunctionStaging {
    OwningOpRef<ModuleOp> module;
    IRMapping mapping;
    func::FuncOp function;
    explicit FunctionStaging(func::FuncOp original);
    void commit(func::FuncOp original);
};
} // namespace mlir::pto::frontiersynch
#endif
