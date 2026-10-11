// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Immutable reaching descriptor facts for one unchanged function. Identity is
// the descriptor SSA value, never its reused physical allocation address.
#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCDESCRIPTORSTATE_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCDESCRIPTORSTATE_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <array>
#include <memory>
#include <optional>
namespace mlir::pto {
class SyncScalarFacts;
struct SyncDescriptorDimension {
    std::optional<int64_t> constant;
    Value value;
    bool operator==(const SyncDescriptorDimension& other) const
    {
        return constant == other.constant && value == other.value;
    }
};
using SyncDescriptorShape = std::array<SyncDescriptorDimension, 2>;
struct SyncDescriptorWork {
    uint64_t sourceVisits = 0;
    uint64_t indexedOperations = 0;
    uint64_t headerFactLosses = 0;
    uint64_t reusedLoops = 0;
    uint64_t updates = 0;
    uint64_t fixedPointPasses = 0;
    uint64_t lookups = 0;
};
class SyncDescriptorState {
public:
    using Query = std::pair<Operation*, Value>;
    // Additional queries include translator base buffers that are not literal
    // operands. The source IR must outlive this immutable index.
    explicit SyncDescriptorState(
        func::FuncOp function, ArrayRef<Query> queries = {}, std::shared_ptr<const SyncScalarFacts> scalars = {});
    ~SyncDescriptorState();
    SyncDescriptorState(const SyncDescriptorState&) = delete;
    SyncDescriptorState& operator=(const SyncDescriptorState&) = delete;
    std::optional<SyncDescriptorShape> before(Value descriptor, Operation* access) const;
    std::optional<std::array<int64_t, 2>> constantShape(Value descriptor, Operation* access) const;
    SyncDescriptorWork work() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto
#endif
