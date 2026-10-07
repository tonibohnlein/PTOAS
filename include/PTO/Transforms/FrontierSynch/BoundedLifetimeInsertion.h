// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEINSERTION_H
#define PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEINSERTION_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
namespace mlir::pto::frontiersynch {
struct ProgramRecognition;
// Builds window predicates indexed by actual source ordinals. Guard replay is
// checked at both original cuts; failure leaves the source function unchanged.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeEndpoints(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
