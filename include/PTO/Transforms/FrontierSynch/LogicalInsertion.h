// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Route-independent preparation and insertion of logical synchronization.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_LOGICALINSERTION_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <cstdint>
#include <memory>
#include <vector>
namespace mlir::pto::frontiersynch {
enum class LogicalCommandKind { Set, Barrier, Wait };
struct PreparedLogicalEndpoint {
    Operation* before = nullptr;
    LogicalCommandKind kind = LogicalCommandKind::Set;
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    int64_t record = 0;
    Value guard; // i1, evaluated at the cut before the command.
    Value identity; // Index source-occurrence identity; absent for a barrier.
};
// A producer builds ordinary arith operations in detached blocks. Inputs may
// refer to original SSA values or results of earlier preparation blocks.
struct LogicalPreparation {
    Operation* before = nullptr;
    std::unique_ptr<Block> code;
};
struct PreparedLogicalPlan {
    explicit PreparedLogicalPlan(int64_t planId) : planId(planId) {}
    ~PreparedLogicalPlan();
    PreparedLogicalPlan(const PreparedLogicalPlan&) = delete;
    PreparedLogicalPlan& operator=(const PreparedLogicalPlan&) = delete;
    Block& addPreparation(Operation* before);
    int64_t planId;
    bool completeInvocation = false; // Drain whole-function work before return.
    // Optional producer certificate for the immediately following allocation
    // pass. It owns no borrowed analysis state; changing the plan invalidates it.
    DictionaryAttr allocationCertificate;
    std::vector<LogicalPreparation> preparation;
    std::vector<PreparedLogicalEndpoint> endpoints;
};
// Producer obligations: certified demands, legal cuts, safely evaluable arithmetic,
// paired guards and matching identities, and all endpoints sharing a cut in one
// batch. No original IR may change between preparation and insertion. This API
// checks structural validity, availability and namespace freshness, not the proof
// of the supplied demand relation or dynamic matching.
// Failure leaves the function intact. Success consumes detached preparation and
// emits SET, coalesced local barriers, then WAIT at each cut. No physical IDs.
LogicalResult insertLogicalSynchronization(func::FuncOp function, PreparedLogicalPlan& plan);
} // namespace mlir::pto::frontiersynch
#endif
