// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTOOperationRegistration.h"
#include "PTO/IR/PTO.h"
#include <initializer_list>

namespace mlir::pto {
namespace {
template <typename... Operations>
void registerOperations(PTODialect& dialect)
{
    // Match Dialect::addOperations. An initializer list also supports operation
    // lists larger than Clang's default fold-expression nesting limit.
    (void)std::initializer_list<int>{0, (RegisteredOperationName::insert<Operations>(dialect), 0)...};
}
} // namespace

void registerPTOOperations(PTODialect& dialect)
{
    registerOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
        >(dialect);
}
} // namespace mlir::pto
