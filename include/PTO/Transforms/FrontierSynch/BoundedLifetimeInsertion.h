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
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
namespace mlir::pto::frontiersynch {
struct ProgramRecognition;
class IterationPredicates;
// Owns the mathematical result and every placeholder referenced by its DAG.
// Original IR and shared input phases must outlive this object, as for regional
// analyses. An endpoint failure does not invalidate the source demand circuits.
struct BoundedLifetimeDemandResult {
    BoundedLifetimeDemandResult();
    ~BoundedLifetimeDemandResult();
    BoundedLifetimeDemandResult(const BoundedLifetimeDemandResult&) = delete;
    BoundedLifetimeDemandResult& operator=(const BoundedLifetimeDemandResult&) = delete;
    Block symbols;
    std::unique_ptr<IterationPredicates> predicates;
    RegionExpressions expressions;
    std::vector<TemplateEndpointAnchor> anchors;
    LifetimeWindowInput window;
    LifetimeWindowAnalysis analysis;
};
// Builds window predicates indexed by actual source ordinals. Guard replay is
// checked at both original cuts; failure leaves the source function unchanged.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeEndpoints(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error);
// The optional output is assigned only after whole-invocation exact demand
// analysis succeeds; endpoint/physical export failures leave it available.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error,
    std::shared_ptr<BoundedLifetimeDemandResult>* demands = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
