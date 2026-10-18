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
#include "PTO/Transforms/FrontierSynch/Shared/Recognition.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetime.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateEndpoints.h"
namespace mlir::pto::frontiersynch {
struct ProgramRecognition;
struct StructureNode;
class IterationPredicates;
struct StorageLaneCell;
// Owns the mathematical result and every placeholder referenced by its DAG.
// Original IR and shared input phases must outlive this object, as for regional
// analyses. An endpoint failure does not invalidate the source demand circuits.
struct BoundedLifetimeAllocationRecipe;
struct BoundedLifetimeDemandResult {
    explicit BoundedLifetimeDemandResult(std::shared_ptr<RegionExpressions> arena);
    ~BoundedLifetimeDemandResult();
    BoundedLifetimeDemandResult(const BoundedLifetimeDemandResult&) = delete;
    BoundedLifetimeDemandResult& operator=(const BoundedLifetimeDemandResult&) = delete;
    std::shared_ptr<RegionExpressions> arena;
    RegionExpressions& expressions;
    std::shared_ptr<Block> symbolOwner = std::make_shared<Block>();
    Block& symbols;
    std::unique_ptr<IterationPredicates> predicates;
    func::FuncOp function;
    scf::ForOp loop;
    const SyncInput* input = nullptr;
    RegionExpressions::Id base = RegionExpressions::invalid, ordinal = RegionExpressions::invalid;
    std::vector<std::pair<RegionExpressions::Id, std::pair<Value, uint64_t>>> predicateBindings;
    std::vector<StorageLaneCell> physicalCells;
    std::vector<uint8_t> unconditional;
    std::vector<TemplateEndpointAnchor> anchors;
    // Shares the recognition's original prerequisite mapping; a local window
    // does not consume its external entry/exit obligations.
    std::shared_ptr<const BoundedPrerequisiteCertificate> prerequisites;
    LifetimeWindowInput window;
    LifetimeWindowAnalysis analysis;
    std::shared_ptr<BoundedLifetimeAllocationRecipe> allocationRecipe;
};
// Construct once for any qualifying original loop, independent of siblings
// and command availability. Local window rows are not global regional queries.
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> analyzeBoundedLifetimeRegion(
    func::FuncOp function, scf::ForOp loop, const PhaseIndex& index, const SyncInput& input,
    const BoundedLifetimeRecognition& recognized, std::string& error,
    std::shared_ptr<RegionExpressions> arena = {}, std::optional<uint64_t> exportSpan = {});
FailureOr<std::shared_ptr<BoundedLifetimeDemandResult>> cachedBoundedLifetimeRegion(
    func::FuncOp function, const StructureNode& node, const PhaseIndex& index,
    const SyncInput& input, std::string& error);
// Explicit diagnostic workload: each retained pair is prepared in an isolated
// detached fragment. It never changes original IR or reruns demand analysis.
std::size_t auditBoundedLifetimeEndpoints(std::shared_ptr<BoundedLifetimeDemandResult> demands,
                                        llvm::raw_ostream& out);
// Reuse an owned successful analysis. The loop remains in its original IR;
// command preparation does not repeat recognition or the window reduction.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeLogicalResult(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, std::string& error,
    bool completeInvocation = false);
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeResult(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, std::string& error,
    bool completeInvocation = false);
// Use the same prerequisite index that qualified program recognition.
// The optional output is assigned only after whole-invocation exact demand
// analysis succeeds; endpoint/physical export failures leave it available.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBoundedLifetimeInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program,
    const PhaseIndex& index, std::string& error,
    std::shared_ptr<BoundedLifetimeDemandResult>* demands = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
