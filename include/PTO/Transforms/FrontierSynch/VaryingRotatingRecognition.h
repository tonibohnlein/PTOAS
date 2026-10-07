// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_VARYINGROTATINGRECOGNITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_VARYINGROTATINGRECOGNITION_H
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "PTO/Transforms/FrontierSynch/RotatingBoundary.h"
namespace mlir::pto::frontiersynch {
struct VaryingRotatingRecognition {
    RecognitionResult result;
    RecognitionResult child;
    scf::ForOp outer, inner;
    uint64_t slope = 0, intercept = 0;
};
// A single rotating child with K(t)=a*t+b, unchanged effects across visits.
// Uses shared machine-integer normalization; no IR mutation or demand backend.
VaryingRotatingRecognition recognizeVaryingRotating(scf::ForOp outer, const PhaseIndex& index, const SyncInput& input);
// Produces child F* records and startup/seam/suffix crossing F* recipes.
// Endpoint preparation and arbitrary-event regional queries are separate exports.
AffineRotatingVisits analyzeVaryingRotating(
    const VaryingRotatingRecognition& recognized, const PhaseIndex& index, const SyncInput& input);
} // namespace mlir::pto::frontiersynch
#endif
