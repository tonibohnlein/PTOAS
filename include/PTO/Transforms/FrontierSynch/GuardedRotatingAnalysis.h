// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_GUARDEDROTATINGANALYSIS_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
namespace mlir::pto::frontiersynch {
struct GuardedRotatingAnalysis {
    std::string error;
    scf::ForOp loop;
    std::shared_ptr<RegionExpressions> expressions;
    SmallVector<const CompoundInstanceElement*> phases;
    std::vector<GuardedPeriodicPayload> payloads;
    std::vector<GuardedPeriodicRecord> generators;
    GuardedPeriodicQuotient periodic;
    uint64_t refreshBound = 0;
};
// Immutable presence and offset parameters, exact fixed within-slot atoms, and
// one potential payload skeleton. No iterations, slot values or guard valuations
// are enumerated. Conditional alias normalization and strict writer selection
// cost O(A^2) circuit operations, excluding encoded arithmetic bit costs.
// Queries/retention are supplied by the shared parameterized quotient. Insertion
// must separately establish legal matching cuts and adjacent local demands.
// Potential target-protected accumulator writer pairs are unsupported here;
// use the numerical protected route until conditional protection is represented
// in the required graph. Ordinary unprotected ACC effects are supported.
GuardedRotatingAnalysis analyzeGuardedRotating(scf::ForOp loop, const SyncInput& input,
                                               const GuardedRecognition& recognition);
} // namespace mlir::pto::frontiersynch
#endif
