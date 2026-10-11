// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned scalar normalization for unchanged IR. Each block/width context has
// its own canonical symbols; callers rebind only the expression nodes they use.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARFACTS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCSCALARFACTS_H
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <memory>
namespace mlir::pto {
class SyncScalarFacts {
public:
    SyncScalarFacts();
    ~SyncScalarFacts();
    SyncScalarFacts(const SyncScalarFacts&) = delete;
    SyncScalarFacts& operator=(const SyncScalarFacts&) = delete;
    AffineExpr value(Value value, Operation* access, llvm::function_ref<AffineExpr(Value)> symbol) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto
#endif
