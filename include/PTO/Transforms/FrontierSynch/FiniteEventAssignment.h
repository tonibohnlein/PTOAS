// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Causal finite handoff assignment. No issue-order or latency approximation.
#ifndef PTO_FRONTIERSYNCH_FINITE_EVENT_ASSIGNMENT_H
#define PTO_FRONTIERSYNCH_FINITE_EVENT_ASSIGNMENT_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
namespace mlir::pto::frontiersynch {
struct FiniteHandoff { std::size_t publication, acquisition; };
// Ordered covers in one directed pool. sourceTargetPrefix is the saved S
// component at the source for the target pipe, not an issue-order surrogate.
struct OrderedHandoffSummary {
    std::size_t sourceRank, targetRank, sourceTargetPrefix;
};
struct FiniteEventAssignment {
    bool certified = false;
    std::size_t required = 0;
    llvm::SmallVector<unsigned> ids;
    std::string reason;
    // Zero-based first reusable later pair; h denotes no later publication.
    // Thus required = max(thresholds[i] - i), matching the one-based theorem.
    llvm::SmallVector<std::size_t> thresholds;
};
// reach is a strict, acyclic completion-order relation over command nodes.
// Every acquisition must follow its matched publication. Each chosen chain
// certifies acquisition-finish before the next publication-finish on that ID.
FiniteEventAssignment assignFiniteEvents(llvm::ArrayRef<FiniteHandoff> handoffs,
    llvm::ArrayRef<unsigned> eligibleIds, llvm::ArrayRef<llvm::BitVector> reach);
// Requires canonical order-exact direct covers and established native/command
// contracts from the caller. Validates ordered summary inputs; the forward
// scan and cyclic assignment take O(h), excluding supplied eligible-ID input.
FiniteEventAssignment assignOrderedFiniteEvents(llvm::ArrayRef<OrderedHandoffSummary> summaries,
    llvm::ArrayRef<unsigned> eligibleIds);
} // namespace mlir::pto::frontiersynch
#endif
