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
struct StructureNode;
class IterationPredicates;
struct StorageLaneCell;
// Owns the mathematical result and every placeholder referenced by its DAG.
// Original IR and shared input phases must outlive this object, as for regional
// analyses. An endpoint failure does not invalidate the source demand circuits.
struct BoundedLifetimeDemandResult {
    explicit BoundedLifetimeDemandResult(std::shared_ptr<RegionExpressions> arena);
    ~BoundedLifetimeDemandResult();
    BoundedLifetimeDemandResult(const BoundedLifetimeDemandResult&) = delete;
    BoundedLifetimeDemandResult& operator=(const BoundedLifetimeDemandResult&) = delete;
    std::shared_ptr<RegionExpressions> arena;
    RegionExpressions& expressions;
    Block symbols;
    std::unique_ptr<IterationPredicates> predicates;
    func::FuncOp function;
    scf::ForOp loop;
    const SyncInput* input = nullptr;
    RegionExpressions::Id base = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
    std::vector<std::pair<RegionExpressions::Id, std::pair<Value, uint64_t>>> predicateBindings;
    std::vector<StorageLaneCell> physicalCells;
    std::vector<uint8_t> unconditional;
    std::vector<TemplateEndpointAnchor> anchors;
    LifetimeWindowInput window;
    LifetimeWindowAnalysis analysis;
};
// Construct once for any qualifying original loop, independent of siblings
// and command availability. Local window rows are not global regional queries.
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> analyzeBoundedLifetimeRegion(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error,
    std::shared_ptr<RegionExpressions> arena = {});
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> cachedBoundedLifetimeRegion(
    func::FuncOp function, const StructureNode& node, const PhaseIndex& index,
    const SyncInput& input, std::string& error);
// Reuse an owned successful analysis. The loop remains in its original IR;
// command preparation does not repeat recognition or the window reduction.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeResult(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, std::string& error,
    bool completeInvocation = false);
// The optional output is assigned only after whole-invocation exact demand
// analysis succeeds; endpoint/physical export failures leave it available.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error,
    std::shared_ptr<BoundedLifetimeDemandResult>* demands = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
