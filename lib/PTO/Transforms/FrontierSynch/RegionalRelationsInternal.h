// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_REGIONALRELATIONSINTERNAL_H
#define PTO_FRONTIERSYNCH_REGIONALRELATIONSINTERNAL_H
#include "PTO/Transforms/FrontierSynch/RegionalRelations.h"
namespace mlir::pto::frontiersynch {
// Lossless quotient-to-raw conversion and shared parameter-column alignment.
LogicalResult normalizeRegionalRelationData(RegionalRelationData& data,
    ArrayRef<Value> parameters, ArrayRef<RegionExpressions::Id> bindings, std::string& error);
struct RegionalRelationRequest {
    virtual ~RegionalRelationRequest() = default;
    virtual ArrayRef<Value> parameters() const = 0;
    virtual FailureOr<RegionalRelationData> lower(ArrayRef<Value> parameters, std::string& error) = 0;
    std::shared_ptr<Block> variables;
};
FailureOr<std::unique_ptr<RegionalRelationRequest>> requestCallbackRegionalRelations(
    const RegionalAnalysis& region, func::FuncOp function, const SyncInput& input,
    const PhaseIndex& index, std::string& error);
FailureOr<RegionalAnalysis> exportRegionalRelationData(std::shared_ptr<RegionalRelations> relation,
    std::shared_ptr<RegionExpressions> expressions, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
