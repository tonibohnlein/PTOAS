// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient invocation-wide source counts; unavailable bounds are export obligations.
#ifndef PTO_FRONTIERSYNCH_INVOCATIONSOURCEBOUNDS_H
#define PTO_FRONTIERSYNCH_INVOCATIONSOURCEBOUNDS_H
#include "InvocationPrefix.h"
namespace mlir::pto::frontiersynch {
// Counts original source actions, including distinct pipes at the same operation.
// Branch alternatives use a per-pipe maximum; siblings sum and loops multiply.
// No runtime visit enumeration, memory-shape assumptions, or selector inference.
// Success guarantees each count and its exclusive +1 sentinel fit signed index64.
FailureOr<std::map<uint32_t, uint64_t>> inferInvocationSourceBounds(
    func::FuncOp original, llvm::ArrayRef<PrefixSource> sources, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
