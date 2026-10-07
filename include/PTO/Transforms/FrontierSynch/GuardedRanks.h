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
// All edges are forward in the potential occurrence order. Guards imply
// endpoint presence; the implementation conjoins presence as well. Duplicates
// are ORed. Native prerequisites seed each start row before cover tests.
GuardedRanks reduceGuardedRanks(
    RegionExpressions& expressions, llvm::ArrayRef<GuardedRankPayload> payloads,
    llvm::ArrayRef<GuardedRankEdge> generators, llvm::ArrayRef<GuardedRankEdge> nativePrerequisites = {});
} // namespace mlir::pto::frontiersynch
#endif
