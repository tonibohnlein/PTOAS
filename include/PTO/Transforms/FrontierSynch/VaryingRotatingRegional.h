// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_FRONTIERSYNCH_VARYINGROTATINGREGIONAL_H
#define PTO_FRONTIERSYNCH_VARYINGROTATINGREGIONAL_H
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRecognition.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
// Exact visit-indexed exports. Numerical port transfers are composed by shared
// Boolean matrix powers; no runtime visit or inner iteration is expanded.
// A supplied certificate must come from this recognizer and unchanged SyncInput.
FailureOr<RegionalAnalysis> varyingRotatingRegionalResult(
    func::FuncOp function, const VaryingRotatingRecognition& recognized, const PhaseIndex& index,
    const SyncInput& input, std::shared_ptr<RegionExpressions> expressions, std::string& error,
    const AffineRotatingVisits* certificate = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
