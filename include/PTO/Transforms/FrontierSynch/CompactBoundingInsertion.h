// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Production fallback on unchanged shared modeled effects. Exact dispatcher
// successes remain preferred even when their allocation export is unavailable.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDINGINSERTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDINGINSERTION_H
#include "PTO/Transforms/FrontierSynch/CompactClassRepetition.h"
namespace mlir::pto::frontiersynch {
struct CompactBoundingOwner {
    // Keeps every borrowed shared effect/phase pointer alive after preparation.
    std::shared_ptr<const SyncInput> input;
    CompactClasses boundary;
    std::vector<CompactClasses> captured;
    bool placementMayStrengthen = true;
    // Describes missing exports only; never a physical scarcity/minimality claim.
    std::string allocationError;
};
struct CompactBoundingPreparation {
    std::string error, exportError;
    std::shared_ptr<const CompactBoundingOwner> owner;
    std::unique_ptr<PreparedLogicalPlan> prepared;
};
// Uses the original fixed/balanced compact input and qualified class sequence/
// repetition producers. No symbolic arithmetic fallback, address recovery,
// instruction whitelist or runtime occurrence expansion. Upper queries describe
// the selected graph; consumer-adjacent local barriers may strengthen it.
// Failed endpoint/allocation exports preserve the mathematical owner. A root
// balanced loop is distributed across its original cuts; mixed/nested balanced
// slots still need a regional multiple-cut interface and remain unavailable.
CompactBoundingPreparation prepareCompactBoundingInsertion(
    func::FuncOp function, std::shared_ptr<const SyncInput> input);
} // namespace mlir::pto::frontiersynch
#endif
