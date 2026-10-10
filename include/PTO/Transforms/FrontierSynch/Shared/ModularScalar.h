// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact machine scalar normalization into the shared bit-exact circuit arena.
// The immutable binding context supplies only admitted leaf values. No implicit
// input contracts or address-validity assumptions are introduced.
#ifndef PTO_FRONTIERSYNCH_MODULARSCALAR_H
#define PTO_FRONTIERSYNCH_MODULARSCALAR_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "llvm/ADT/STLFunctionalExtras.h"
namespace mlir::pto::frontiersynch {
class ModularScalar {
public:
    using Id = RegionExpressions::Id;
    using Binding = llvm::function_ref<std::optional<Id>(Value)>;
    ModularScalar(RegionExpressions& arena, Operation* anchor);
    // Every call on this object must use the same immutable binding context.
    // Construct another object for a different visit, substitution or parameter binding.
    // Non-Boolean outputs are zero-extended original-width bit patterns.
    // The anchor's data layout defines index width. Unsupported expressions
    // return nullopt; they never acquire independent symbolic parameters.
    std::optional<Id> value(Value input, Binding binding);
    std::optional<Id> signedValue(Value input, Binding binding);
    unsigned width(Type type) const;
    // Conservative range of the arena input: narrow integers use zero-extended
    // bit patterns; original index coordinates retain signed values.
    // The supplied signed source range is certified in the anchor's target layout.
    std::optional<std::pair<int64_t, int64_t>> inputRange(Value input,
        std::pair<int64_t, int64_t> signedRange) const;
private:
    RegionExpressions& e;
    unsigned indexBits = 0;
    DenseMap<Value, std::optional<Id>> memo;
    std::optional<Id> resolve(Value input, Binding binding, unsigned depth);
    std::optional<Id> operation(Value input, Binding binding, unsigned depth);
    Id truncate(Id input, unsigned bits);
    Id signedBits(Id input, unsigned bits);
    Id multiply(Id input, uint64_t factor);
};
} // namespace mlir::pto::frontiersynch
#endif
