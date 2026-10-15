// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact consistency certificate for the congruence part of a normalized
// IntegerSystem. A consistent result says nothing about its inequalities.
#ifndef PTO_FRONTIER_INTEGERCONGRUENCECONSISTENCY_H
#define PTO_FRONTIER_INTEGERCONGRUENCECONSISTENCY_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include <limits>
#include <optional>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
enum class CongruenceConsistency { Consistent, Inconsistent, Unknown };
// Inputs have already been validated and normalized by IntegerSystem::create.
inline std::optional<mlir::presburger::IntMatrix> congruenceMatrix(
    unsigned dimensions, llvm::ArrayRef<IntegerCongruence> congruences)
{
    const auto maximum = std::numeric_limits<unsigned>::max();
    if (congruences.empty() || congruences.size() > maximum) { return std::nullopt; }
    for (const auto& atom : congruences) {
        if (atom.coefficients.size() != dimensions || atom.modulus <= 0) { return std::nullopt; }
    }
    // Identically zero A columns contribute nothing to the column lattice of
    // [A,diag(m)]. Keep all modulus columns and every used original coordinate.
    // Inequality-only coordinates remain in the enclosing IntegerSystem.
    std::vector<unsigned> active;
    for (unsigned column = 0; column < dimensions; ++column) {
        if (llvm::any_of(congruences, [&](const auto& atom) { return atom.coefficients[column] != 0; })) {
            active.push_back(column);
        }
    }
    if (congruences.size() > maximum - active.size()) { return std::nullopt; }
    const auto rows = static_cast<unsigned>(congruences.size());
    const auto variables = static_cast<unsigned>(active.size());
    const auto columns = variables + rows;
    // Matrix uses unsigned products, and HNF also constructs an N-by-N U.
    if (columns > maximum / columns ||
        static_cast<uint64_t>(columns) * columns > llvm::SmallVector<BoundInteger, 16>().max_size()) {
        return std::nullopt;
    }
    mlir::presburger::IntMatrix matrix(rows, columns);
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned column = 0; column < variables; ++column) {
            matrix(row, column) = congruences[row].coefficients[active[column]];
        }
        matrix(row, variables + row) = congruences[row].modulus;
    }
    return matrix;
}
inline bool fullCongruenceBasis(const mlir::presburger::IntMatrix& h, unsigned rows)
{
    if (h.getNumRows() != rows || h.getNumColumns() < rows) { return false; }
    for (unsigned row = 0; row < rows; ++row) {
        if (h(row, row) <= 0) { return false; }
        for (unsigned column = row + 1; column < h.getNumColumns(); ++column) {
            if (h(row, column) != 0) { return false; }
        }
    }
    return true;
}
inline CongruenceConsistency solveCongruenceBasis(
    const mlir::presburger::IntMatrix& h, llvm::ArrayRef<IntegerCongruence> congruences)
{
    std::vector<BoundInteger> solution(congruences.size(), BoundInteger(0));
    for (unsigned row = 0; row < congruences.size(); ++row) {
        auto residual = congruences[row].residue;
        for (unsigned column = 0; column < row; ++column) { residual -= h(row, column) * solution[column]; }
        if (mod(residual, h(row, row)) != 0) { return CongruenceConsistency::Inconsistent; }
        solution[row] = residual / h(row, row);
    }
    return CongruenceConsistency::Consistent;
}
inline CongruenceConsistency congruenceConsistency(
    unsigned dimensions, llvm::ArrayRef<IntegerCongruence> congruences, IntegerFeasibilityCost* cost = nullptr)
{
    if (congruences.size() < 2) { return CongruenceConsistency::Consistent; }
    auto matrix = congruenceMatrix(dimensions, congruences);
    if (!matrix) { return CongruenceConsistency::Unknown; }
    if (cost) {
        accumulateCost(cost->congruenceChecks, 1);
        accumulateCost(cost->congruenceMatrixColumns, matrix->getNumColumns());
    }
    // M=[A,diag(m)] has full row rank. H=M*U has the same column lattice,
    // with a positive lower-triangular basis in its first `rows` columns.
    auto normal = matrix->computeHermiteNormalForm();
    if (!fullCongruenceBasis(normal.first, static_cast<unsigned>(congruences.size()))) {
        return CongruenceConsistency::Unknown;
    }
    return solveCongruenceBasis(normal.first, congruences);
}
} // namespace mlir::pto::frontiersynch::detail
#endif
