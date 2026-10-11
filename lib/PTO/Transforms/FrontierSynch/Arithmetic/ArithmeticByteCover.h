// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_ARITHMETICBYTECOVER_H
#define PTO_FRONTIERSYNCH_ARITHMETICBYTECOVER_H
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <optional>
#include <utility>
namespace mlir::pto::frontiersynch {
// A sufficient enclosing interval, never an exact footprint. The rational
// relaxation contains every integer point, including every congruence class.
// Failure to obtain bounds leaves the exact symbolic storage adapter available.
inline std::optional<std::pair<BoundInteger, BoundInteger>> rationalCoordinateCover(
    const IntegerSystem& system, unsigned coordinate, uint64_t& proofRequests)
{
    if (coordinate >= system.dimensions()) { return std::nullopt; }
    presburger::Simplex relaxation(system.dimensions());
    for (const auto& row : system.constraints()) {
        std::vector<BoundInteger> coefficients;
        for (const auto& coefficient : row.coefficients) { coefficients.push_back(-coefficient); }
        coefficients.push_back(row.bound);
        relaxation.addInequality(coefficients);
    }
    if (relaxation.isEmpty()) { return std::nullopt; }
    std::vector<BoundInteger> objective(static_cast<std::size_t>(system.dimensions()) + 1, BoundInteger(0));
    objective[coordinate] = 1;
    accumulateCost(proofRequests, 1);
    auto upper = relaxation.computeOptimum(presburger::Simplex::Direction::Up, objective);
    if (!upper.isBounded()) { return std::nullopt; }
    objective[coordinate] = -1;
    accumulateCost(proofRequests, 1);
    auto negativeLower = relaxation.computeOptimum(presburger::Simplex::Direction::Up, objective);
    if (!negativeLower.isBounded()) { return std::nullopt; }
    return std::make_pair(-floorDiv(negativeLower->num, negativeLower->den), floorDiv(upper->num, upper->den));
}
} // namespace mlir::pto::frontiersynch
#endif
