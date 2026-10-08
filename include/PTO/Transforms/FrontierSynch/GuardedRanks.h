// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDRANKS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDRANKS_H
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
namespace mlir::pto::frontiersynch {
struct GuardedRankPayload {
    uint32_t pipe;
    RegionExpressions::Id present;
};
struct GuardedRankEdge {
    uint32_t source, target;
    RegionExpressions::Id guard;
};
struct GuardedRanks {
    std::string error;
    std::vector<GuardedRankPayload> payloads;
    std::vector<uint32_t> columns;
    std::vector<RegionExpressions::Id> ranks;
    std::vector<std::vector<RegionExpressions::Id>> starts, completions;
    std::vector<GuardedRankEdge> retained;
    std::optional<RegionExpressions::Id> query(
        RegionExpressions& expressions, PeriodicEvent source, PeriodicEvent target) const;
};
// Register form of the forward reducer. Sources are supplied in decreasing
// reference order; completion rows and ranks refer to that same execution.
// The supplier bounds total executed ranks by UINT64_MAX and supplies a
// reachable register state. The transition does not retain previous owners.
struct GuardedRankFrontier {
    std::vector<RegionExpressions::Id> counters;
    std::vector<std::vector<RegionExpressions::Id>> starts, completions;
};
// Incoming guards imply source presence; the transition adds target presence.
struct GuardedRankIncoming {
    uint32_t column;
    RegionExpressions::Id rank, guard;
    llvm::ArrayRef<RegionExpressions::Id> completion;
};
struct GuardedRankStep {
    std::string error;
    RegionExpressions::Id rank = RegionExpressions::invalid;
    std::vector<RegionExpressions::Id> start, completion, retained;
};
GuardedRankFrontier initGuardedRankFrontier(RegionExpressions& expressions, uint32_t columns);
GuardedRankStep advanceGuardedRank(RegionExpressions& expressions, GuardedRankFrontier& frontier,
    uint32_t column, RegionExpressions::Id present, llvm::ArrayRef<GuardedRankIncoming> candidates,
    llvm::ArrayRef<GuardedRankIncoming> native = {});
// All edges are forward in the potential occurrence order. Guards imply
// endpoint presence; the implementation conjoins presence as well. Duplicates
// are ORed. Native prerequisites seed each start row before cover tests.
GuardedRanks reduceGuardedRanks(
    RegionExpressions& expressions, llvm::ArrayRef<GuardedRankPayload> payloads,
    llvm::ArrayRef<GuardedRankEdge> generators, llvm::ArrayRef<GuardedRankEdge> nativePrerequisites = {});
} // namespace mlir::pto::frontiersynch
#endif
