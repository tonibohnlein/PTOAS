// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICREGIONAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICREGIONAL_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
// Analyze one original region with shared entry bindings. The finite-boundary
// adapter enumerates bounded physical bytes, never dynamic payload occurrences;
// adjacent bytes with identical selector tuples share one exported cell.
FailureOr<RegionalAnalysis> analyzeArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error);
}
#endif
