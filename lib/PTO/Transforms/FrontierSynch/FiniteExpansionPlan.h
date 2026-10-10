// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_FINITEEXPANSIONPLAN_H
#define PTO_FRONTIERSYNCH_FINITEEXPANSIONPLAN_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "NormalizedControl.h"
namespace mlir::pto::frontiersynch {
struct FiniteGuardedAnalysis;
// Preflight owns original occurrence identities and bounded fixed-coordinate
// control decisions only. It constructs no accesses, relations or endpoints.
// IR/input/index pointers are borrowed from one unchanged source context.
// Applicable certifies bounded control enumeration, not exact accesses/demands.
struct FiniteExpansionVisit {
    Operation* operation = nullptr;
    ArithmeticSite context;
    bool fixedBranch = false;
};
struct FiniteExpansionPlan {
    ArithmeticRegionContext context;
    const PhaseIndex* index = nullptr;
    const SyncInput* input = nullptr;
    FiniteExpansionLimits limits;
    RecognitionResult result;
    SmallVector<FiniteExpansionVisit> visits;
    SmallVector<ArithmeticSite> sites;
    uint64_t foldOperations = 0, prunedArms = 0;
    bool expandsLoops = false;
    std::shared_ptr<const NormalizedControlDescription> normalized;
};
FiniteExpansionPlan preflightFiniteExpansion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, const FiniteExpansionLimits& limits = {},
    std::shared_ptr<const NormalizedControlDescription> normalized = {});
ArithmeticProgram materializeFiniteExpansion(const FiniteExpansionPlan& plan,
    const PhaseIndex& index, const SyncInput& input);
// Session results may share an arena; unsuccessful predicate construction is
// transactional and never poisons previously retained query circuits.
FiniteGuardedAnalysis analyzeExpandedFinite(const FiniteExpansionPlan& plan,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions = {});
} // namespace mlir::pto::frontiersynch
#endif
