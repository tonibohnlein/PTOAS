// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// An owned proof that omitted GM generators are redundant, never disjoint.
#ifndef PTO_FRONTIERSYNCH_ORDERCOVEREDPERIODIC_H
#define PTO_FRONTIERSYNCH_ORDERCOVEREDPERIODIC_H
#include "PTO/Transforms/FrontierSynch/Structured/RotatingAnalysis.h"
#include <memory>
namespace mlir::pto::frontiersynch {
struct CoveredEffectPair {
    std::size_t sourceEffect, targetEffect;
    uint32_t sourceSite, targetSite;
    uint64_t firstForward, threshold;
};
// Original IR must outlive this immutable owner. No regional byte selectors or
// re-entry/disjointness certificate is implied by the omitted effect identities.
struct OrderCoveredPeriodicResult {
    std::shared_ptr<const SyncInput> input;
    std::shared_ptr<const PhaseIndex> index;
    std::size_t originalRegion;
    std::vector<std::size_t> omittedEffects;
    std::vector<CoveredEffectPair> coveredPairs;
    RotatingAnalysis retained;
    bool applies(const SyncInput* expectedInput, const PhaseIndex* expectedIndex, scf::ForOp loop) const;
};
std::shared_ptr<const OrderCoveredPeriodicResult> analyzeOrderCoveredPeriodic(
    scf::ForOp loop, std::size_t originalRegion, std::shared_ptr<const SyncInput> input,
    std::shared_ptr<const PhaseIndex> index, uint64_t& reductions, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
