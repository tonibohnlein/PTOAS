// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_DIFFERENCE_INTEGER_CONVERSION_H
#define PTO_FRONTIER_DIFFERENCE_INTEGER_CONVERSION_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
namespace mlir::pto::frontiersynch::detail {
// Exact representation adapter. No feasibility, union minimization or demand
// reduction is requested. Arbitrary-precision bounds and empty systems survive.
inline FailureOr<IntegerSystem> integerDifferenceRows(unsigned dimensions,
    ArrayRef<DifferenceBoundConstraint> atoms, ArithmeticAnalysisCost* cost = nullptr)
{
    std::vector<IntegerConstraint> rows;
    for (const auto& atom : atoms) {
        if (atom.lhs > dimensions || atom.rhs > dimensions) { return failure(); }
        IntegerConstraint row{std::vector<BoundInteger>(dimensions), atom.bound};
        if (atom.lhs) { row.coefficients[atom.lhs - 1] += BoundInteger(1); }
        if (atom.rhs) { row.coefficients[atom.rhs - 1] -= BoundInteger(1); }
        rows.push_back(std::move(row));
        if (cost) {
            accumulateCost(cost->differenceExportAtoms, 1);
            accumulateCost(cost->differenceExportCoefficients, dimensions);
        }
    }
    if (cost) { accumulateCost(cost->differenceExportSystems, 1); }
    return IntegerSystem::create(dimensions, rows);
}
inline FailureOr<GeneralArithmeticRelation> exportDifferenceRelation(
    const ArithmeticRelation& relation, ArithmeticAnalysisCost* cost = nullptr)
{
    GeneralArithmeticRelation result;
    for (const auto& [key, pieces] : relation) {
        for (const auto& piece : pieces) {
            // Empty union members contribute no endpoints (including allocation families).
            if (piece.isEmpty()) { continue; }
            auto converted = integerDifferenceRows(piece.dimensions(), piece.constraints(), cost);
            if (failed(converted)) { return failure(); }
            result[key].push_back(std::move(*converted));
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
