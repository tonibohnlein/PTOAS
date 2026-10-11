// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned legal-invocation proof for allocation, separate from raw window algebra.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATIONCONTEXT_H
#define PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEALLOCATIONCONTEXT_H
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include <memory>
#include <optional>
namespace mlir::pto::frontiersynch::detail {
// This capability follows the session's unchanged-input contract. It owns the
// const modeled input and original result; source IR must outlive them. Original
// IR/model mutation invalidates the enclosing session. Scalar definition snapshots
// additionally reject same-SSA bound/guard edits before reusing this proof cache.
class BoundedLifetimeAllocationContext {
public:
    static std::shared_ptr<BoundedLifetimeAllocationContext> create(
        std::shared_ptr<const BoundedLifetimeDemandResult> owner, std::shared_ptr<const SyncInput> input);
    std::optional<llvm::ArrayRef<uint8_t>> inactive(RegionExpressions& arena, const LifetimeWindowInput& window,
        llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> demands);
    uint64_t proofBuilds() const;
private:
    struct State;
    explicit BoundedLifetimeAllocationContext(std::shared_ptr<State> state) : state(std::move(state)) {}
    std::shared_ptr<State> state;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
