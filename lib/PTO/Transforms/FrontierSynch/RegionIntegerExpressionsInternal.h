// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_REGIONINTEGEREXPRESSIONSINTERNAL_H
#define PTO_FRONTIER_REGIONINTEGEREXPRESSIONSINTERNAL_H
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
namespace mlir::pto::frontiersynch {
struct RegionExpressions::IntegerRecipe {
    bool predicate = true;
    IntegerSystem system;
    IntegerAffine numerator;
    BoundInteger denominator{1};
    std::vector<Id> inputs;
    std::vector<AffineExpr> coordinates, coordinateOrder;
    std::vector<uint64_t> residues;
    uint64_t period = 1, outputResidue = 0;
    std::string signature;
};
}
#endif
