// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Certified uniform global queries for a bounded lifetime result.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEREGIONAL_H
#define PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEREGIONAL_H
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
FailureOr<RegionalAnalysis> boundedLifetimeRegionalQueries(
    std::shared_ptr<BoundedLifetimeDemandResult> demands, const BoundedLifetimeRecognition& recognized,
    const PhaseIndex& index, std::shared_ptr<RegionExpressions> arena, std::string& error,
    std::shared_ptr<const SyncInput> inputOwner = {});
FailureOr<RegionalAnalysis> boundedLifetimeRegionalSelectors(const BoundedLifetimeDemandResult& demands,
    const BoundedLifetimeRecognition& recognized, const RegionalAnalysis& queries, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
