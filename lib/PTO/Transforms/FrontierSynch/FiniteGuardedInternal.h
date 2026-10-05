// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Internal shared state; export closures own it without owning each other.
#ifndef PTO_FRONTIERSYNCH_FINITEGUARDEDINTERNAL_H
#define PTO_FRONTIERSYNCH_FINITEGUARDEDINTERNAL_H
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
namespace mlir::pto::frontiersynch {
struct GuardedDemand { uint32_t source = 0, target = 0; RegionExpressions::Id guard = 0; };
struct FiniteGuardedState {
    using Expr = RegionExpressions::Id;
    func::FuncOp function;
    std::shared_ptr<RegionExpressions> arena;
    std::vector<TemplateEndpointAnchor> anchors;
    std::vector<Expr> presence;
    std::vector<ExplicitEffects> effects;
    std::vector<std::vector<Expr>> graph;
    std::vector<GuardedDemand> retained;
    std::vector<RegionalStorageBoundary> storageBoundary;
    std::map<uint32_t, std::vector<RegionalSelector>> firstPayloads, lastPayloads;
    RegionalCost cost;
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayAlias;
    std::string insertionError;
    uint32_t pipe(uint32_t type) const { return effects[type].pipe; }
    Expr yes() { return arena->boolean(true); }
    Expr no() { return arena->boolean(false); }
    Expr both(Expr a, Expr b) { return arena->land(a,b); }
    Expr either(Expr a, Expr b) { return arena->lor(a,b); }
    Expr negate(Expr a) { return arena->lnot(a); }
    void closeAndReduce();
    void summarize(const SyncInput& input);
    FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepare();
};
} // namespace mlir::pto::frontiersynch
#endif
