// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structural work for the two guarded quotient algorithms. Exact grouped counts
// describe producer loops; pre-construction upper estimates use the same shape.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICWORK_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDPERIODICWORK_H
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
namespace mlir::pto::frontiersynch {
inline EstimatedCount guardedPeriodicRounds(EstimatedCount vertices, EstimatedCount pipes, bool coreChains)
{
    if (coreChains) { return pipes; }
    return vertices ? EstimatedCount(*vertices ? *vertices - 1 : 0) : EstimatedCount{};
}
struct GuardedPeriodicWork {
    EstimatedCount edges, relaxations, nativeScanVisits, exclusions;
};
inline GuardedPeriodicWork guardedPeriodicWork(EstimatedCount payloads, EstimatedCount pipes,
    EstimatedCount groups, EstimatedCount native)
{
    if (!native) { return {}; }
    const bool coreChains = *native == 0;
    const auto rounds = guardedPeriodicRounds(estimatedMultiply(2, payloads), pipes, coreChains);
    GuardedPeriodicWork work;
    work.edges = coreChains ? groups : estimatedAdd(estimatedAdd(groups, native), estimatedMultiply(3, payloads));
    work.relaxations = estimatedMultiply(estimatedMultiply(estimatedMultiply(2, pipes), rounds), work.edges);
    // Each of two frontiers closes native order before and after every round:
    // two directions for each of start/completion. Retention adds one scan pair.
    work.nativeScanVisits = coreChains ? estimatedMultiply(estimatedMultiply(pipes, payloads),
        estimatedAdd(estimatedMultiply(8, estimatedAdd(rounds, 1)), 2)) : EstimatedCount(0);
    // Only edges targeting starts enter the exclusion pass. Generic native
    // chains add one start/start edge per site; I->C and C->C are omitted.
    work.exclusions = estimatedMultiply(pipes,
        coreChains ? groups : estimatedAdd(estimatedAdd(groups, native), payloads));
    return work;
}
} // namespace mlir::pto::frontiersynch
#endif
