// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Budgets do not assign overlapping physical resources to independent directions.
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
#include "llvm/Support/JSON.h"
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAllocation(const fs::PeriodicAllocation& result)
{
    llvm::json::Array directions;
    for (const auto& direction : result.directions) {
        llvm::json::Array handoffs;
        for (const auto& handoff : direction.handoffs) {
            handoffs.push_back(llvm::json::Object{{"record", handoff.record}, {"source", handoff.source},
                {"target", handoff.target}, {"displacement", handoff.displacement},
                {"first_reuse", handoff.firstReuse ? llvm::json::Value(*handoff.firstReuse) :
                    llvm::json::Value(nullptr)}});
        }
        directions.push_back(llvm::json::Object{{"source_pipe", direction.sourcePipe},
            {"target_pipe", direction.targetPipe}, {"payload_types", direction.payloadTypes},
            {"handoffs", std::move(handoffs)}, {"uniform_budget", direction.uniformBudget ?
                llvm::json::Value(*direction.uniformBudget) : llvm::json::Value(nullptr)}});
    }
    return llvm::json::Object{{"error", result.error}, {"directions", std::move(directions)},
        {"budgets_ready", result.error.empty()}, {"physical_ids_ready", false}, {"ir_emitted", false}};
}
